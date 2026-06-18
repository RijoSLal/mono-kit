import numpy as np
import os

# silence noisy decoder warnings from ffmpeg/opencv
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["FFMPEG_LOG_LEVEL"] = "QUIET"

try:
    from .rust_engine import MemmapIndex
except ImportError:
    raise ImportError("rust_engine not found. Run 'pip install .' to compile.")

import torch
import torch.nn as nn
import cv2
import os
import logging
import gc
from PIL import Image
from torchvision import transforms
from typeguard import typechecked
from imagebind import data # type: ignore
from imagebind.models import imagebind_model # type: ignore
from imagebind.models.imagebind_model import ModalityType # type: ignore

logger = logging.getLogger(__name__)

class MonoDB:
    """main database interface for multimodal embeddings."""
    def __init__(self, dir: str = "./store", embedding_func = None, device: str = None):
        self.db = MemmapIndex(dir) 
        self.__alloc_types = {"image", "text", "audio", "video"}
        self.embedding_func = embedding_func
        self._inbuilt_imagebind_model = None
        self._imagebind_load_failed = False
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self._imagebind_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4814, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2757)),
        ])

    @property
    def inbuilt_imagebind_model(self):
        if self._imagebind_load_failed: raise RuntimeError("ImageBind failed to load.")
        if self._inbuilt_imagebind_model: return self._inbuilt_imagebind_model

        from bitsandbytes.nn import Linear4bit
        from bitsandbytes.nn.modules import Params4bit
        from bitsandbytes.functional import QuantState
        from accelerate import init_empty_weights
        from huggingface_hub import hf_hub_download #type: ignore

        try:
            # patch multiheadattention for 4-bit uint8 weight compatibility
            class LinearMHA(nn.Module):
                def __init__(self, embed_dim, num_heads, bias=True, add_bias_kv=False, **kwargs):
                    super().__init__()
                    self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
                    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                    self.h, self.d = num_heads, embed_dim // num_heads
                def forward(self, x, attn_mask=None):
                    x = x.half()
                    L, B, C = x.shape
                    qkv = self.in_proj(x).reshape(L, B, 3, self.h, self.d).permute(2, 1, 3, 0, 4)
                    if attn_mask is not None and attn_mask.dim() == 2:
                        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).half()
                    res = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=attn_mask)
                    return self.out_proj(res.permute(2, 0, 1, 3).reshape(L, B, C))
            
            imagebind_model.MultiheadAttention = LinearMHA

            torch.set_default_dtype(torch.float16)
            with init_empty_weights():
                model = imagebind_model.imagebind_huge(pretrained=False)
            torch.set_default_dtype(torch.float32)

            # prune unused modalities
            for attr in ["modality_trunks", "modality_heads", "modality_postprocessors"]:
                if hasattr(model, attr):
                    container = getattr(model, attr)
                    for m in [ModalityType.IMU, ModalityType.THERMAL, ModalityType.DEPTH]:
                        if m in container: del container[m]

            def to_4bit(m):
                for name, child in m.named_children():
                    if isinstance(child, nn.Linear) and name != "in_proj":
                        setattr(m, name, Linear4bit(child.in_features, child.out_features, bias=child.bias is not None, compute_dtype=torch.float16, quant_type="nf4"))
                    else: to_4bit(child)
            to_4bit(model)

            path = hf_hub_download("RijoSLal/ImageBind-4-bit-quantized", "imagebind_4bit.pth", local_dir=".checkpoints")
            sd = torch.load(path, map_location='cpu', weights_only=True)
            
            sd = {k.replace("in_proj_weight", "in_proj.weight").replace("in_proj_bias", "in_proj.bias"): v for k, v in sd.items()}

            # manual params4bit reconstruction
            for name, module in model.named_modules():
                if isinstance(module, Linear4bit):
                    w_key = f"{name}.weight"
                    if w_key in sd:
                        pw = sd.pop(w_key)
                        if f"{name}.weight.absmax" in sd:
                            qs = QuantState.__new__(QuantState)
                            qs.quant_type, qs.blocksize, qs.dtype, qs.nested = 'nf4', 64, torch.float32, False
                            qs.absmax = sd.pop(f"{name}.weight.absmax")
                            qs.quant_map = sd.pop(f"{name}.weight.quant_map")
                            qs.code, qs.shape = qs.quant_map, (module.out_features, module.in_features)
                            
                            if f"{name}.weight.nested_absmax" in sd:
                                qs.nested, qs.nested_blocksize, qs.nested_dtype = True, 256, torch.float32
                                qs.nested_absmax = sd.pop(f"{name}.weight.nested_absmax")
                                qs.nested_quant_map = sd.pop(f"{name}.weight.nested_quant_map")

                            new_param = Params4bit(pw, requires_grad=False, quant_type="nf4")
                            new_param.bnb_quantized, new_param.quant_state = True, qs
                            del module.weight
                            module.register_parameter("weight", new_param)
                            module.quant_state, module.compute_dtype = qs, torch.float16
                        else:
                            module.weight = nn.Parameter(pw)
                        
                    b_key = f"{name}.bias"
                    if b_key in sd and module.bias is not None: module.bias.data = sd.pop(b_key)
                    elif b_key in sd: sd.pop(b_key)

                    prefix = f"{name}.weight."
                    for k in list(sd.keys()):
                        if k.startswith(prefix): sd.pop(k)

                elif not isinstance(module, Linear4bit) and (any(p.is_meta for p in module.parameters()) or any(b.is_meta for b in module.buffers())):
                    module.to_empty(device="cpu")

            model.load_state_dict(sd, strict=False)
            del sd
            gc.collect()
            
            # cast remaining standard weights to half precision
            model.half()
            model.to(self.device)
            model.eval()
            self._inbuilt_imagebind_model = model
            return model

        except Exception as e:
            self._imagebind_load_failed = True
            logger.error(f"ImageBind load failed: {e}")
            raise e

    def _imagebind(self, input_data, data_type):
        if data_type == "image":
            inputs = {ModalityType.VISION: data.load_and_transform_vision_data([input_data] if isinstance(input_data, str) else input_data, self.device)}
        elif data_type == "audio":
            inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data([input_data] if isinstance(input_data, str) else input_data, self.device)}
        elif data_type == "text":
            inputs = {ModalityType.TEXT: data.load_and_transform_text([input_data] if isinstance(input_data, str) else input_data, self.device)}
        elif data_type == "video":
            if not os.path.exists(input_data): raise FileNotFoundError(f"video not found: {input_data}")
            from mono_v2.own_embedding_model.preprocess_pipeline import sample_video_frames

            frames = sample_video_frames(input_data, n_frames=10)
            collage = np.zeros((224, 224, 3), dtype=np.uint8)
            for i, f in enumerate(frames):
                thumb = cv2.resize(f, (44, 112), interpolation=cv2.INTER_AREA)
                r, c = i // 5, i % 5
                collage[r * 112:(r + 1) * 112, c * 44:(c + 1) * 44] = thumb

            del frames
            gc.collect()
            inputs = {ModalityType.VISION: self._imagebind_transform(Image.fromarray(collage)).unsqueeze(0).to(self.device)}
            del collage
        else: raise ValueError(f"Unsupported type: {data_type}")

        with torch.no_grad():
            for k in inputs:
                if k != ModalityType.TEXT: inputs[k] = inputs[k].half()
            out = self.inbuilt_imagebind_model(inputs)
        
        res = out[next(iter(inputs.keys()))].squeeze().float().cpu().numpy()
        del inputs, out
        return res

    def clear_cache(self):
        """internally handle memory management to keep the system responsive."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """ensure resources are cleaned up on deletion."""
        self.clear_cache()

    @typechecked
    def embed(self, input_data: str | list, data_type: str, inbuilt_model: bool = True, model_path: str = "mono_v2/model/mono_emb_model.pth"):
        if inbuilt_model: 
            # unload custom model if it exists to save ram
            if hasattr(self, "mono_model"):
                del self.mono_model
                del self.preprocessor
                self.clear_cache()
            res = self._imagebind(input_data, data_type)
        else:
            # unload imagebind model if it exists to save ram
            if self._inbuilt_imagebind_model is not None:
                del self._inbuilt_imagebind_model
                self._inbuilt_imagebind_model = None
                self.clear_cache()
                
            from mono_v2.own_embedding_model.inference_pipeline import mono_model
            from mono_v2.own_embedding_model.preprocess_pipeline import EmbeddingPreProcessor
            if not hasattr(self, "mono_model"): self.mono_model = mono_model(model_path, self.device)
            if not hasattr(self, "preprocessor"): self.preprocessor = EmbeddingPreProcessor(self.device)
            proc = self.preprocessor(input_data, data_type)
            with torch.no_grad():
                if isinstance(proc, torch.Tensor) and proc.dtype.is_floating_point: proc = proc.half()
                emb = self.mono_model(proc, data_type)
            res = emb.detach().cpu().numpy().squeeze()
            del proc, emb
        
        # auto-manage memory after embedding
        self.clear_cache()
        return res

    @typechecked
    def insert(self, idx: str, embedding: np.ndarray, type: str, meta: dict = {}):
        if embedding.dtype != np.float32: embedding = embedding.astype(np.float32)
        meta["dtype"] = type
        self.db.insert(idx, embedding, meta)

    @typechecked
    def update(self, idx: str, embedding: np.ndarray, type: str, meta: dict = {}):
        if embedding.dtype != np.float32: embedding = embedding.astype(np.float32)
        meta["dtype"] = type
        self.db.update(idx, embedding, meta)

    @typechecked
    def delete(self, idx: str | list):
        if isinstance(idx, str): self.db.delete(idx)
        else: [self.db.delete(i) for i in idx]

    def list_all(self): return self.db.list_all()

    @typechecked
    def topk(self, embedding: np.ndarray, k: int, batch_size: int, type: str):
        if embedding.dtype != np.float32: embedding = embedding.astype(np.float32)
        return self.db.topk(embedding, k, batch_size, types=[type])
