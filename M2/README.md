# Mono-Kit: Multimodal Retrieval Tool Kit

**Mono-Kit** is an ML toolkit for developers designed for seamless multimodal retrieval. It features a custom **Rust-powered vector database engine**  and **embedding models** that maps text, image, audio, and video into the **same embedding space**, allowing for cross-modal similarity search with ease.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from mono_v2 import MonoDB

# Initialize the database
db = MonoDB(dir="./store")
```

## Core Operations

### 1. Generating Embeddings (`embed`)

Convert raw data into vector embeddings. By default, Mono-Kit can use an inbuilt ImageBind model or a custom `mono_model`.

```python
# Text Embedding
text_emb = db.embed("Hello world", data_type="text")

# Image Embedding
image_emb = db.embed("path/to/image.png", data_type="image")

# Audio Embedding
audio_emb = db.embed("path/to/audio.mp3", data_type="audio")

# Video Embedding
video_emb = db.embed("path/to/video.mp4", data_type="video")
```

### 2. Inserting Data (`insert`)

Store vectors with associated IDs and metadata.

```python
db.insert(
    idx="unique_id_1",
    embedding=image_emb,
    type="image",
    meta={"category": "nature"}
)
```

### 3. Updating Data (`update`)

Update an existing record's embedding or metadata.

```python
db.update(
    idx="unique_id_1",
    embedding=image_emb,
    type="image",
    meta={"category": "forest"}
)
```

### 4. Top-K Similarity Search (`topk`)

Search for the most similar items across a specific modality.

```python
results = db.topk(
    embedding=image_emb,
    k=5,
    batch_size=32,
    type="image"
)

for idx, score in results:
    print(f"ID: {idx}, Similarity Score: {score}")
```

### 5. Deleting Data (`delete`)

```python
db.delete("unique_id_1")
```

### 6. Listing All IDs (`list_all`)

```python
all_ids = db.list_all()
```

## Technical Architecture: `mono_model`

When `inbuilt_model=False` is passed to the `embed` method, Mono-Kit uses its internal `mono_model` architecture. Otherwise, it uses a `4-bit quantized pretrained ImageBind` model. These models are designed to project different data types into a single shared vector space.

| Modality | Encoder |
|----------|----------|
| Image | EfficientNet-B0 |
| Audio | Log-Mel Spectrogram + RoPE-based encoder |
| Video | Image Encoder + LRCN-inspired architecture with Attention layers |
| Text | LLM-style RoPE-based encoder |

### Device Selection
Force usage of CPU or a specific GPU during initialization:

```python
db = MonoDB(device="cpu")
# or
db = MonoDB(device="cuda:0")
```
