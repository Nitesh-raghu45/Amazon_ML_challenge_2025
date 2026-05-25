"""
config.py — Centralized configuration for Amazon ML 2025 Pipeline
All paths, hyperparameters, and model settings live here.
"""

import os
from pathlib import Path

# ── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ARCHIVE_DIR = BASE_DIR / "archive"
IMAGES_DIR = BASE_DIR / "Images"
EMBEDDING_DIR = ARCHIVE_DIR / "embedding"
OUTPUT_DIR = BASE_DIR / "outputs"

# ── Dataset Paths ────────────────────────────────────────────────────────────
TRAIN_CSV = ARCHIVE_DIR / "train.csv"
TEST_CSV = ARCHIVE_DIR / "test.csv"
SAMPLE_TEST_CSV = ARCHIVE_DIR / "sample_test.csv"

# ── Extraction Output Paths ──────────────────────────────────────────────────
EXTRACTION_DIR = ARCHIVE_DIR / "extraction"

# ── Model Settings ───────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Ollama / Cloud LLM (for extraction)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "your-ollama-api-key-here")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:14b-cloud")

# HuggingFace VLM (alternative)
HF_VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# ── Extraction Settings ──────────────────────────────────────────────────────
MAX_RETRIES = 5
MAX_NEW_TOKENS = 1024
EXTRACTION_BATCH_SIZE = 10  # rows per CSV save checkpoint

# ── Embedding Settings ───────────────────────────────────────────────────────
EMBEDDING_BATCH_SIZE = 128
NUM_WORKERS = 32

# ── Training Settings ────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# XGBoost
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# LightGBM
LGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# ── Download Settings ────────────────────────────────────────────────────────
DOWNLOAD_MAX_THREADS = 256
DOWNLOAD_TIMEOUT = 10

# ── Ensure output directories exist ─────────────────────────────────────────
def ensure_dirs():
    for d in [ARCHIVE_DIR, IMAGES_DIR, EMBEDDING_DIR, OUTPUT_DIR,
              EXTRACTION_DIR, IMAGES_DIR / "train", IMAGES_DIR / "test",
              IMAGES_DIR / "sample_test"]:
        d.mkdir(parents=True, exist_ok=True)
