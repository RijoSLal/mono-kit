import os
import gc
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as Tv
import torchaudio
import torchaudio.transforms as Tat
import cv2
import tiktoken # type:ignore
from typing import Union, Optional, Any
import logging

# silence noisy decoder warnings from ffmpeg/opencv
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["FFMPEG_LOG_LEVEL"] = "QUIET"

# configure logging
logger = logging.getLogger(__name__)


def sample_video_frames(path: str, n_frames: int = 10, max_candidates: int = 30) -> list[np.ndarray]:
    """
    sample video frames using uniform candidates + pixel drift selection.
    pass 1 keeps only small grayscale thumbnails; pass 2 decodes selected RGB frames.
    """
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            while cap.grab():
                total += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if total == 0:
            raise ValueError("empty video")

        target_indices = np.linspace(0, total - 1, min(max_candidates, total)).astype(int)

        grays = []
        curr_idx = 0
        for target_idx in target_indices:
            while curr_idx < target_idx:
                if not cap.grab():
                    break
                curr_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            curr_idx += 1
            grays.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)))
            del frame

        if not grays:
            raise ValueError("read failure")

        if len(grays) <= n_frames:
            pick_indices = np.arange(len(grays))
        else:
            diffs = [float("inf")] + [
                cv2.absdiff(grays[i], grays[i - 1]).mean() for i in range(1, len(grays))
            ]
            pick_indices = np.sort(np.argsort(diffs)[-n_frames:])

        del grays
        gc.collect()

        selected_targets = target_indices[pick_indices]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames_by_num = {}
        curr_idx = 0
        for frame_num in sorted(set(selected_targets.tolist())):
            while curr_idx < frame_num:
                if not cap.grab():
                    break
                curr_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            curr_idx += 1
            frames_by_num[frame_num] = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            del frame

        frames_list = [frames_by_num[int(target_indices[i])] for i in pick_indices]
        if not frames_list:
            raise ValueError("read failure")

        pad = np.zeros((224, 224, 3), dtype=np.uint8)
        while len(frames_list) < n_frames:
            frames_list.append(pad)

        return frames_list[:n_frames]
    finally:
        cap.release()


def _video_frames_to_tensor(frames_list: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """convert sampled RGB frames to a normalized half-precision batch tensor on device."""
    n = len(frames_list)
    frames_tensor = torch.empty((n, 3, 224, 224), dtype=torch.float16)
    for i, f in enumerate(frames_list):
        if f.shape[:2] != (224, 224):
            f = cv2.resize(f, (224, 224))
        t = torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).to(torch.float16).div_(255.0)
        t[0].sub_(0.485).div_(0.229)
        t[1].sub_(0.456).div_(0.224)
        t[2].sub_(0.406).div_(0.225)
        frames_tensor[i] = t

    return frames_tensor.unsqueeze(0).to(device, non_blocking=True)


class EmbeddingPreProcessor:
    """unified preprocessor for image, audio, video, and text modalities."""

    def __init__(self, device: Optional[str] = None):
        """
        initializes preprocessor with auto device detection and pre-instantiated transforms.

        args:
            device (str, optional): target device (cuda/cpu).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.allowed_types = {"image", "audio", "video", "text"}

        # vision normalization
        self.vision_mean = [0.485, 0.456, 0.406]
        self.vision_std = [0.229, 0.224, 0.225]

        self.vision_transform = Tv.Compose([
            Tv.Resize((224, 224), interpolation=Image.Resampling.LANCZOS),
            Tv.ToTensor(),
            Tv.Normalize(mean=self.vision_mean, std=self.vision_std)
        ])

        # video frame normalize
        self.video_normalize = Tv.Normalize(mean=self.vision_mean, std=self.vision_std)

        # audio transforms - pre-instantiate to save ram and time
        self.resampler_16k = {} # cache resamplers by source sr
        self.mel_op = Tat.MelSpectrogram(
            sample_rate=16000, 
            n_mels=128, 
            n_fft=1024,
            hop_length=160
        ).to(self.device)
        self.db_op = Tat.AmplitudeToDB().to(self.device)

        # tiktoken encoding
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.pad_token = self.encoding.n_vocab + 1
        logger.debug(f"preprocessor initialized on {self.device}")

    def __call__(self, data: Any, data_type: str, **kwargs) -> torch.Tensor:
        """
        unified entry point for all preprocessing.

        args:
            data (any): input data (path, array, or string).
            data_type (str): modality type.
            **kwargs: modality-specific options.

        returns:
            torch.tensor: processed tensor.
        """
        if data_type not in self.allowed_types:
            raise ValueError(f"invalid data_type '{data_type}'. must be one of {self.allowed_types}")

        match data_type:
            case "image":
                return self.preprocess_image(data)
            case "audio":
                return self.preprocess_audio(data, **kwargs)
            case "video":
                return self.preprocess_video(data, **kwargs)
            case "text":
                return self.preprocess_text(data, **kwargs)
            case _:
                raise ValueError(f"unsupported: {data_type}")

    def preprocess_image(self, input_data: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        processes image to (1, 3, 224, 224) tensor.

        args:
            input_data (str | ndarray | image): raw image input.

        returns:
            torch.tensor: processed image tensor.
        """
        if isinstance(input_data, str):
            # cv2 is generally faster than pil for loading
            img = cv2.imread(input_data)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # manual resize/normalize is often faster than tv for single images
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                del img
                img_tensor.div_(255.0)
                # fused normalization
                img_tensor[0].sub_(0.485).div_(0.229)
                img_tensor[1].sub_(0.456).div_(0.224)
                img_tensor[2].sub_(0.406).div_(0.225)
                return img_tensor.unsqueeze(0).to(self.device, non_blocking=True)
            image = Image.open(input_data).convert("RGB")
        elif isinstance(input_data, np.ndarray):
            image = Image.fromarray(input_data).convert("RGB")
        elif isinstance(input_data, Image.Image):
            image = input_data.convert("RGB")
        else:
            raise TypeError(f"unsupported image type: {type(input_data)}")

        res = self.vision_transform(image).unsqueeze(0).to(self.device, non_blocking=True)
        del image
        return res

    def preprocess_audio(self, input_data: Union[str, np.ndarray], sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        processes audio to (1, 1, 128, 204) log-mel spectrogram tensor.

        args:
            input_data (str | ndarray): raw audio input.
            sample_rate (int, optional): required if input is numpy array.

        returns:
            torch.tensor: processed log-mel tensor.
        """
        if isinstance(input_data, str):
            waveform, sr = torchaudio.load(input_data)
        elif isinstance(input_data, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate required for numpy input.")
            waveform = torch.from_numpy(input_data).float()
            sr = sample_rate
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            raise TypeError(f"unsupported audio type: {type(input_data)}")

        # non_blocking transfer if possible
        waveform = waveform.to(self.device, non_blocking=True)

        # mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample to 16khz - use cached resampler
        if sr != 16000:
            if sr not in self.resampler_16k:
                self.resampler_16k[sr] = Tat.Resample(sr, 16000).to(self.device)
            waveform = self.resampler_16k[sr](waveform)

        # pad/crop to 2s
        target_len = 2 * 16000 
        current_len = waveform.shape[1]
        if current_len > target_len:
            waveform = waveform[:, :target_len]
        elif current_len < target_len:
            waveform = nn.functional.pad(waveform, (0, target_len - current_len))

        # melspec - executed on device using pre-instantiated op
        mel_spec = self.mel_op(waveform)
        del waveform

        # ensure exact (128, 204)
        if mel_spec.shape[2] > 204:
            mel_spec = mel_spec[:, :, :204]
        elif mel_spec.shape[2] < 204:
            mel_spec = nn.functional.pad(mel_spec, (0, 204 - mel_spec.shape[2]))

        # db scale & normalize using pre-instantiated op
        log_mel = self.db_op(mel_spec)
        del mel_spec
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        return log_mel.unsqueeze(0)

    def preprocess_video(self, input_data: Union[str, np.ndarray], n_frames: int = 10) -> torch.Tensor:
        """
        processes video to (1, n_frames, 3, 224, 224) sequence tensor using hybrid sampling.
        """
        if isinstance(input_data, str):
            frames_list = sample_video_frames(input_data, n_frames=n_frames)
        elif isinstance(input_data, np.ndarray):
            total_frames = input_data.shape[0]
            indices = np.linspace(0, total_frames - 1, n_frames).astype(int)
            frames_list = [input_data[i] for i in indices]
        else:
            raise TypeError(f"unsupported video type: {type(input_data)}")

        out = _video_frames_to_tensor(frames_list, self.device)
        del frames_list
        gc.collect()
        return out

    def preprocess_text(self, text_input: str, max_tokens: int = 77) -> torch.Tensor:
        """
        tokenizes text to (1, max_tokens) id tensor.

        args:
            text_input (str): raw text.
            max_tokens (int): sequence length limit.

        returns:
            torch.tensor: tokenized tensor.
        """
        if not isinstance(text_input, str):
            text_input = str(text_input)

        tokens = self.encoding.encode(text_input)[:max_tokens]
        pad_len = max_tokens - len(tokens)
        if pad_len > 0:
            tokens = tokens + [self.pad_token] * pad_len

        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

