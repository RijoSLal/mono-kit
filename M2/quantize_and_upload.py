"""
This script automates the 4-bit quantization of the ImageBind Huge model and uploads the result to Hugging Face.
It prunes unnecessary modalities (IMU, Thermal, Depth) to save memory, applies NF4 quantization using bitsandbytes,
and saves the state dictionary to 'imagebind_4bit.pth' before uploading to RijoSLal/ImageBind-4-bit-quantized.
"""

import torch
import os
import gc
import logging
from imagebind.models import imagebind_model # type:ignore
from imagebind.models.imagebind_model import ModalityType # type:ignore
from huggingface_hub import HfApi, create_repo, login # type:ignore
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_and_save():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. load full model temporarily
    logger.info("Loading full ImageBind model...")
    # downloading the huge model takes ~4.8gb ram. this might be risky given the user's constraints.
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()

    # 2. prune
    unused_modalities = [ModalityType.IMU, ModalityType.THERMAL, ModalityType.DEPTH]
    for attr in ["modality_trunks", "modality_heads", "modality_postprocessors"]:
        if hasattr(model, attr):
            container = getattr(model, attr)
            for m in unused_modalities:
                if m in container:
                    del container[m]
    logger.info("Pruned unused modalities")

    # 3. save pruned weights temporarily (accelerate needs to load from a file)
    temp_path = ".checkpoints/temp_pruned.pth"
    os.makedirs(".checkpoints", exist_ok=True)
    torch.save(model.state_dict(), temp_path)
    del model
    gc.collect()

    # 4. use accelerate for clean native quantization
    logger.info("Applying native 4-bit quantization via Accelerate...")
    
    # we must instantiate the empty architecture again for accelerate to fill
    torch.set_default_dtype(torch.float16)
    with init_empty_weights():
        empty_model = imagebind_model.imagebind_huge(pretrained=False)
        # apply the same pruning to the empty skeleton
        for attr in ["modality_trunks", "modality_heads", "modality_postprocessors"]:
            if hasattr(empty_model, attr):
                container = getattr(empty_model, attr)
                for m in unused_modalities:
                    if m in container: del container[m]
    torch.set_default_dtype(torch.float32)

    bnb_config = BnbQuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    quantized_model = load_and_quantize_model(
        empty_model,
        weights_location=temp_path,
        bnb_quantization_config=bnb_config,
        device_map="auto"
    )

    save_path = ".checkpoints/imagebind_4bit.pth"
    logger.info(f"Saving quantized model to {save_path}...")
    torch.save(quantized_model.state_dict(), save_path)
    
    os.remove(temp_path)
    return save_path

def upload_to_hf(file_path):
    repo_id = "RijoSLal/ImageBind-4-bit-quantized"
    api = HfApi()
    try:
        api.whoami()
    except Exception:
        login()
        api = HfApi()

    logger.info(f"Uploading {file_path} to {repo_id}...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="imagebind_4bit.pth",
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info("Upload successful")
    except Exception as e:
        logger.error(f"Upload failed: {e}")

if __name__ == "__main__":
    path = quantize_and_save()
    upload_to_hf(path)
