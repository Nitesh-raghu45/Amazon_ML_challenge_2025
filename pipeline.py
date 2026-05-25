"""
pipeline.py — End-to-End Amazon ML 2025 Pipeline Orchestrator.

Stages:
  1. DOWNLOAD  — Download images from URLs in train/test CSV
  2. EXTRACT   — Use VLM (Ollama) to extract structured features
  3. EMBED     — Generate CLIP embeddings (image + text fusion)
  4. TRAIN     — Train price prediction model (XGBoost + LightGBM)
  5. SUBMIT    — Generate submission CSV

Run all:    python pipeline.py
Run stage:  python pipeline.py --stage embed
"""

import argparse
import time
import sys
from pathlib import Path

from config import (
    ensure_dirs, TRAIN_CSV, TEST_CSV, SAMPLE_TEST_CSV,
    EXTRACTION_DIR, EMBEDDING_DIR, OUTPUT_DIR
)


# ── Stage: Download ───────────────────────────────────────────────────────────

def stage_download(splits: list = None):
    """Download product images for specified splits."""
    from downloader import download_images

    splits = splits or ["train", "test", "sample_test"]
    csv_map = {
        "train": str(TRAIN_CSV),
        "test": str(TEST_CSV),
        "sample_test": str(SAMPLE_TEST_CSV),
    }

    total_stats = {}
    for split in splits:
        if split not in csv_map:
            print(f"Unknown split: {split}, skipping.")
            continue
        print(f"\n{'='*50}")
        print(f"  Downloading [{split}] images")
        print(f"{'='*50}")
        stats = download_images(split, csv_map[split])
        total_stats[split] = stats

    print("\n=== Download Summary ===")
    for split, stats in total_stats.items():
        print(f"  {split}: {stats}")


# ── Stage: Extract ────────────────────────────────────────────────────────────

def stage_extract(
    splits: list = None,
    batch_size: int = 10,
    start_from: int = 0,
):
    """Extract structured features using VLM for specified splits."""
    import pandas as pd
    from extractor import extract_batch

    splits = splits or ["train", "test"]
    csv_map = {
        "train": str(TRAIN_CSV),
        "test": str(TEST_CSV),
        "sample_test": str(SAMPLE_TEST_CSV),
    }

    for split in splits:
        if split not in csv_map:
            continue

        print(f"\n{'='*50}")
        print(f"  Extracting [{split}] features")
        print(f"{'='*50}")

        df = pd.read_csv(csv_map[split])
        total = len(df)
        print(f"Total rows: {total}")

        for start in range(start_from, total, batch_size):
            end = min(start + batch_size, total)
            print(f"\nProcessing rows {start}–{end}...")
            extract_batch(
                df=df,
                name=split,
                start=start,
                end=end,
                output_dir=str(EXTRACTION_DIR),
            )

        # Merge all batch CSVs into one
        _merge_extraction_batches(split, df)


def _merge_extraction_batches(split: str, base_df):
    """Merge all batch extraction CSVs into a single file."""
    import pandas as pd

    batch_dir = EXTRACTION_DIR / split
    if not batch_dir.exists():
        return

    batch_files = sorted(batch_dir.glob("*.csv"))
    if not batch_files:
        return

    dfs = []
    for f in batch_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=["sample_id"], keep="last")
        out = EXTRACTION_DIR / f"{split}_extracted.csv"
        merged.to_csv(out, index=False)
        print(f"Merged {len(batch_files)} batches → {out} ({len(merged)} rows)")


# ── Stage: Embed ──────────────────────────────────────────────────────────────

def stage_embed(splits: list = None):
    """Generate CLIP embeddings for specified splits."""
    import pandas as pd
    from embedder import generate_embeddings

    splits = splits or ["train", "test"]
    csv_map = {
        "train": str(TRAIN_CSV),
        "test": str(TEST_CSV),
        "sample_test": str(SAMPLE_TEST_CSV),
    }

    for split in splits:
        if split not in csv_map:
            continue

        print(f"\n{'='*50}")
        print(f"  Generating embeddings [{split}]")
        print(f"{'='*50}")

        df = pd.read_csv(csv_map[split])
        embedded = generate_embeddings(split, df)
        print(f"Embedding complete: {embedded.shape}")


# ── Stage: Train ──────────────────────────────────────────────────────────────

def stage_train():
    """Train the price prediction model."""
    print(f"\n{'='*50}")
    print("  Training Price Prediction Model")
    print(f"{'='*50}")

    from train_price_model import main as train_main
    train_main()


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(args):
    """Run all pipeline stages end-to-end."""
    start = time.time()

    print("\n" + "=" * 60)
    print("  AMAZON ML 2025 — END-TO-END PIPELINE")
    print("=" * 60)

    if not args.skip_download:
        stage_download(splits=args.splits)

    if not args.skip_extract:
        stage_extract(
            splits=args.splits,
            batch_size=args.extract_batch_size,
            start_from=args.extract_start,
        )

    if not args.skip_embed:
        stage_embed(splits=args.splits)

    if not args.skip_train:
        stage_train()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Submission: {OUTPUT_DIR / 'submission.csv'}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML 2025 — End-to-End Price Prediction Pipeline"
    )

    parser.add_argument(
        "--stage",
        choices=["download", "extract", "embed", "train", "all"],
        default="all",
        help="Which stage to run (default: all)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to process (default: train test)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip image download stage"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip VLM extraction stage"
    )
    parser.add_argument(
        "--skip-embed", action="store_true",
        help="Skip CLIP embedding stage"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model training stage"
    )
    parser.add_argument(
        "--extract-batch-size", type=int, default=10,
        help="Rows per batch for extraction (default: 10)"
    )
    parser.add_argument(
        "--extract-start", type=int, default=0,
        help="Start row index for extraction (default: 0)"
    )

    args = parser.parse_args()

    # Ensure all required directories exist
    ensure_dirs()

    if args.stage == "download":
        stage_download(splits=args.splits)
    elif args.stage == "extract":
        stage_extract(splits=args.splits, batch_size=args.extract_batch_size, start_from=args.extract_start)
    elif args.stage == "embed":
        stage_embed(splits=args.splits)
    elif args.stage == "train":
        stage_train()
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
