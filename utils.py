"""
utils.py — Shared utility functions for the Amazon ML 2025 pipeline.
"""

import json
import re
import os
import time
import logging
from pathlib import Path
from typing import Optional


# ── Logging Setup ─────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Get a configured logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(name)s: %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ── JSON Utilities ────────────────────────────────────────────────────────────

def extract_json_from_text(text: str) -> dict:
    """
    Robustly extract the first JSON object from raw LLM output.
    Handles markdown code fences, leading text, etc.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output.")
    obj, _ = decoder.raw_decode(text, start)
    return obj


# ── Retry Decorator ───────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """Decorator that retries a function on failure."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_err = e
                    if attempt < max_attempts:
                        time.sleep(delay * attempt)
            raise last_err
        return wrapper
    return decorator


# ── Price Parsing ─────────────────────────────────────────────────────────────

def parse_price(price_str) -> Optional[float]:
    """
    Parse price strings like '₹199', '$14.50', '25.0' to float.
    Returns None if unparseable.
    """
    if price_str is None:
        return None
    s = str(price_str).strip()
    # Remove currency symbols and commas
    s = re.sub(r"[₹$€£,]", "", s).strip()
    try:
        return float(s)
    except ValueError:
        return None


# ── Path Helpers ──────────────────────────────────────────────────────────────

def safe_mkdir(path: str | Path) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def image_exists(name: str, sample_id: str, base_dir: str = "Images") -> bool:
    """Check if a product image exists on disk."""
    return Path(f"{base_dir}/{name}/{sample_id}.jpg").exists()


# ── Progress Timer ────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager timer."""

    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"[{self.label}] Elapsed: {elapsed:.2f}s ({elapsed/60:.2f}m)")

    @property
    def elapsed(self):
        return time.time() - self.start
