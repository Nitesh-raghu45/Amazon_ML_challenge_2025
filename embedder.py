"""
embedder.py — CLIP-based image+text embedding generation.
Cleaned and modularized from EmbeddingImageText.py.
Supports batched processing with GPU, parallel image loading, and caching.
"""

import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

from config import (
    CLIP_MODEL_NAME, EMBEDDING_BATCH_SIZE, NUM_WORKERS, EMBEDDING_DIR
)


# ── Model Load (singleton) ────────────────────────────────────────────────────

_clip_model = None
_clip_processor = None


def get_clip_model():
    """Lazy-load CLIP model (singleton)."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model on {device}...")
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.eval()
        print("CLIP model loaded.")
    return _clip_model, _clip_processor


# ── Image Loading ─────────────────────────────────────────────────────────────

def _load_single(sample_id: str, text: str, name: str):
    """Load a single image safely. Returns None if image not found."""
    img_path = f"Images/{name}/{sample_id}.jpg"
    try:
        image = Image.open(img_path).convert("RGB")
        return sample_id, text, image
    except Exception:
        return None


# ── Embedding Generator ───────────────────────────────────────────────────────

def generate_embeddings(
    name: str,
    df: pd.DataFrame,
    batch_size: int = None,
    num_workers: int = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Generate CLIP embeddings for all rows and save to CSV.

    Args:
        name: Dataset split ('train', 'test', 'sample_test').
        df: DataFrame with 'sample_id' and 'catalog_content'.
        batch_size: Number of rows per GPU batch.
        num_workers: Number of parallel image-loading threads.
        force_recompute: If False, skip if embedded CSV already exists.

    Returns:
        DataFrame with original columns + embedding columns (emb_0..emb_N).
    """
    batch_size = batch_size or EMBEDDING_BATCH_SIZE
    num_workers = num_workers or NUM_WORKERS

    out_path = EMBEDDING_DIR / name / "embedded.csv"

    if not force_recompute and out_path.exists():
        print(f"Embeddings already exist at {out_path}, loading...")
        return pd.read_csv(out_path)

    model, processor = get_clip_model()
    device = next(model.parameters()).device

    df = df.copy()
    sample_ids = df["sample_id"].tolist()
    texts = df["catalog_content"].astype(str).tolist()

    combined_rows = []

    for i in tqdm(range(0, len(df), batch_size), desc=f"Embedding [{name}]"):
        batch_ids = sample_ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        # Parallel image loading for this batch
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                executor.map(
                    lambda x: _load_single(x[0], x[1], name),
                    zip(batch_ids, batch_texts),
                )
            )

        valid = [r for r in results if r is not None]
        if not valid:
            # All images missing — add text-only placeholder rows
            for sid in batch_ids:
                combined_rows.append({"sample_id": sid})
            continue

        ids = [x[0] for x in valid]
        texts_batch = [x[1] for x in valid]
        images = [x[2] for x in valid]

        inputs = processor(
            text=texts_batch,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            text_emb = outputs.text_embeds          # [B, 512]
            image_emb = outputs.image_embeds        # [B, 512]
            combined = torch.cat([image_emb, text_emb], dim=1)  # [B, 1024]

        combined_np = combined.cpu().numpy()

        for j, sid in enumerate(ids):
            row = {"sample_id": sid}
            for k, val in enumerate(combined_np[j]):
                row[f"emb_{k}"] = float(val)
            combined_rows.append(row)

        # Explicit GPU memory cleanup
        del inputs, outputs, combined
        torch.cuda.empty_cache()

    emb_df = pd.DataFrame(combined_rows)
    out = df.merge(emb_df, on="sample_id", how="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved embeddings: {out_path}")

    return out


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import TRAIN_CSV
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(train_df)} training samples.")
    embedded = generate_embeddings("train", train_df)
    print(f"Embedded shape: {embedded.shape}")
