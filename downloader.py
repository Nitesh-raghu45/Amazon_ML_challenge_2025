"""
downloader.py — Parallel image downloader for Amazon ML 2025 dataset.
Cleaned from download.py with better error handling and progress reporting.
"""

import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import DOWNLOAD_MAX_THREADS, DOWNLOAD_TIMEOUT, IMAGES_DIR, TRAIN_CSV, TEST_CSV, SAMPLE_TEST_CSV


# ── Core Downloader ───────────────────────────────────────────────────────────

def download_image(link: str, img_path: str) -> tuple[bool, str]:
    """
    Download a single image from a URL.

    Returns:
        (success: bool, img_path: str)
    """
    try:
        res = requests.get(link, timeout=DOWNLOAD_TIMEOUT)
        res.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(res.content)
        return True, img_path
    except Exception as e:
        return False, str(e)


def download_images(
    name: str,
    csv_path: str,
    max_threads: int = None,
    overwrite: bool = False,
) -> dict:
    """
    Download all images for a dataset split in parallel.

    Args:
        name: Dataset split ('train', 'test', 'sample_test').
        csv_path: Path to the CSV file with 'sample_id' and 'image_link' columns.
        max_threads: Number of download threads (defaults to config value).
        overwrite: If False, skip already-downloaded images.

    Returns:
        dict with 'total', 'skipped', 'success', 'failed'.
    """
    max_threads = max_threads or DOWNLOAD_MAX_THREADS

    folder = IMAGES_DIR / name
    folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    tasks = []
    skipped = 0
    for sample_id, link in zip(df["sample_id"], df["image_link"]):
        img_path = str(folder / f"{sample_id}.jpg")
        if not overwrite and os.path.exists(img_path):
            skipped += 1
            continue
        tasks.append((str(link), img_path))

    print(f"[{name}] Total: {len(df)} | Already downloaded: {skipped} | To download: {len(tasks)}")

    success = 0
    failed = 0

    if tasks:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {
                executor.submit(download_image, link, path): (link, path)
                for link, path in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading [{name}]"):
                ok, _ = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1

    print(f"[{name}] Done. Success: {success}, Failed: {failed}, Skipped: {skipped}")
    return {"total": len(df), "skipped": skipped, "success": success, "failed": failed}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Downloading Train Images ===")
    download_images("train", str(TRAIN_CSV))

    print("\n=== Downloading Test Images ===")
    download_images("test", str(TEST_CSV))

    print("\n=== Downloading Sample Test Images ===")
    download_images("sample_test", str(SAMPLE_TEST_CSV))
