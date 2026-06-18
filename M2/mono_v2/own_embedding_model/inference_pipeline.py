import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import os
import gc
import logging

# configure logging
logger = logging.getLogger(__name__)

PAD_TOKEN = 100277 + 1 # based on tiktoken cl100k_base n_vocab + 1

def rotate_half(x):
    """
    rotates half the hidden dim.
    
    args:
        x (torch.tensor): input tensor.
        
    returns:
        torch.tensor: rotated tensor.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """
    applies rotary positional embeddings.
    
    args:
        x (torch.tensor): input tensor.
        cos (torch.tensor): cosine embeddings.
        sin (torch.tensor): sine embeddings.
        
    returns:
        torch.tensor: embedded tensor.
    """
    return (x * cos) + (rotate_half(x) * sin)

class RotaryPositionalEncoding(nn.Module):
    """rope positional encoding module."""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", position, inv_freq)
        cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        """applies positional encoding forward pass."""
        B, T, H, D = x.shape
        cos = self.cos[:T].view(1, T, 1, D).to(x.device)
        sin = self.sin[:T].view(1, T, 1, D).to(x.device)
        return apply_rotary_pos_emb(x, cos, sin)

class RoPEMultiheadAttention(nn.Module):
    """multihead attention with rotary positional embeddings."""
    def __init__(self, embed_dim=1024, num_heads=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryPositionalEncoding(dim=self.head_dim)

    def forward(self, q_in, k_in, v_in):
        """executes multihead attention with rope."""
        B, T, C = q_in.shape
        q = self.q_proj(q_in).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(k_in).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(v_in).view(B, T, self.num_heads, self.head_dim)
        q = self.rope(q)
        k = self.rope(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

class ImageEncoder(nn.Module):
    """efficientnet based image encoder."""
    def __init__(self, eff_net_sl, output_dim=1024, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eff_net_sl = eff_net_sl
        self.seq = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=output_dim)
        )

    def forward(self, input):
        """processes image input to embedding."""
        if input.dim() == 3:
            input = input.unsqueeze(0)
        efficientnet = self.eff_net_sl(input)
        efficientnet = efficientnet.flatten(1)
        sequential = self.seq(efficientnet)
        return sequential.unsqueeze(1)

class AudioEncoder(nn.Module):
    """cnn based audio encoder."""
    def __init__(self, proj_out_dim=1024, device=None):
       super().__init__()
       self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.cnn = nn.Sequential(
          nn.Conv2d(1, 32, 3, padding=1),
          nn.BatchNorm2d(32),
          nn.Tanh(),
          nn.Conv2d(32, 64, 3, stride=2, padding=1),
          nn.BatchNorm2d(64),
          nn.GELU(),
          nn.Conv2d(64, 128, 3, stride=2, padding=1),
          nn.BatchNorm2d(128),
          nn.GELU(),
       )
       self.patch_proj = nn.Conv2d(128, 1024, kernel_size=(4, 32), stride=(4, 32))

    def forward(self, x):
        """processes audio spectrogram to embedding."""
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.patch_proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class VideoEncoder(nn.Module):
    """attention based video encoder."""
    def __init__(self, img_enc, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_enc = img_enc
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)

    def forward(self, x):
        """processes video frames to sequence-aware embedding."""
        if x.dim() == 4:
            x = x.unsqueeze(0)
        B, F, C, H, W = x.shape
        encodings = []
        for f_idx in range(F):
            encodings.append(self.img_enc(x[:, f_idx]))
        x = torch.cat(encodings, dim=1)
        attn_out, _ = self.attn(x, x, x)
        return attn_out

class TextEncoder(nn.Module):
    """embedding based text encoder."""
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(num_embeddings=100277 + 2, embedding_dim=1024, padding_idx=PAD_TOKEN)

    def forward(self, input_ids):
        """processes token ids to embedding."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.embedding(input_ids)

class MonoEmbeddingModel(nn.Module):
    """final multimodal embedding model combining all encoders."""
    def __init__(self, audio_enc=None, video_enc=None, text_enc=None, image_enc=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_enc = audio_enc
        self.video_enc = video_enc
        self.text_enc = text_enc
        self.image_enc = image_enc
        self.attn = RoPEMultiheadAttention(embed_dim=1024, num_heads=16)
        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(1024)
        self.ffn = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 1024)
        )

    def forward(self, data, data_type=None):
        """
        multimodal forward pass.
        
        args:
            data (dict | torch.tensor): input data.
            data_type (str, optional): modality type if data is not a dict.
            
        returns:
            torch.tensor: mean-pooled embedding.
        """
        # handle legacy dict input
        if isinstance(data, dict):
            input_dict = data
        elif data_type is not None:
            input_dict = {data_type: data}
        else:
            raise ValueError("must provide data_type or a dict.")

        embeddings = []
        if self.audio_enc and "audio" in input_dict:
            embeddings.append(self.audio_enc(input_dict["audio"]))
        if self.video_enc and "video" in input_dict:
            embeddings.append(self.video_enc(input_dict["video"]))
        if self.text_enc and "text" in input_dict:
            embeddings.append(self.text_enc(input_dict["text"]))
        if self.image_enc and "image" in input_dict:
            embeddings.append(self.image_enc(input_dict["image"]))
        
        if not embeddings:
            raise ValueError("no valid modality found in input.")

        cross = torch.cat(embeddings, dim=1)
        attn = self.attn(cross, cross, cross)
        norm1 = self.norm1(attn + cross)
        ffn = self.ffn(norm1)
        final = self.norm2(norm1 + ffn)
        return final.mean(dim=1)

def mono_model(weights_path="mono_v2/model/mono_emb_model.pth", device=None, compile=False):
    """
    instantiate and load weights with auto device detection and memory optimizations.
    
    args:
        weights_path (str): path to model weights.
        device (str | torch.device): target device.
        compile (bool): whether to use torch.compile.
        
    returns:
        MonoEmbeddingModel: loaded and ready model.
    """
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    
    # speed optimization: enable cudnn benchmark
    if target_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    logger.info("instantiating efficientnet_b0 for image encoder...")
    efficientnet_b0 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    eff_net_sl = nn.Sequential(*list(efficientnet_b0.children())[:-1])
    
    # explicitly delete the full model to save ram
    del efficientnet_b0
    gc.collect()
    
    img_enc = ImageEncoder(eff_net_sl, device=target_device)
    video_enc = VideoEncoder(img_enc, device=target_device)
    audio_enc = AudioEncoder(device=target_device)
    text_enc = TextEncoder(device=target_device)
    
    model = MonoEmbeddingModel(audio_enc, video_enc, text_enc, img_enc, device=target_device)
    
    # check fallback weights path
    actual_weights_path = weights_path
    if not os.path.exists(actual_weights_path):
        # try a few common locations
        fallbacks = [
            "mono_v2/model/mono_emb_model.pth",
            "model/mono_emb_model.pth",
            "own_embedding_model/model/mono_emb_model.pth"
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                actual_weights_path = fb
                break
            
    if os.path.exists(actual_weights_path):
        logger.info(f"loading weights from {actual_weights_path}...")
        state_dict = torch.load(actual_weights_path, map_location=target_device, weights_only=True)
        model.load_state_dict(state_dict)
        del state_dict
        gc.collect()
    else:
        logger.warning(f"weights not found at {actual_weights_path}, using random initialization.")
    
    model = model.to(target_device).half().eval()
    
    # speed optimization: torch.compile (optional as it uses massive ram)
    if compile:
        logger.info("compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.error(f"torch.compile failed: {e}")
        
    # warmup
    if target_device.type == "cuda":
        logger.info("running model warmup...")
        with torch.inference_mode():
            dummy_text = torch.zeros((1, 77), dtype=torch.long, device=target_device)
            _ = model({"text": dummy_text})
            
    return model
