"""
extractor.py — VLM-based product data extractor using Ollama (OpenAI-compatible API).
Fixed:
  - ProductExtraction now imported from schema.py
  - API key loaded from env (not hardcoded)
  - Proper normalization for enums
  - Clean retry logic with structured feedback
"""

import json
import re
import base64
import os
from pathlib import Path
from openai import OpenAI
from pydantic import ValidationError

from schema import ProductExtraction
from prompts import SYSTEM_PROMPT
from config import (
    OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL,
    MAX_RETRIES, MAX_NEW_TOKENS
)


# ── Client Setup ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> tuple[str, str]:
    """Base64-encode a local image for multimodal API calls."""
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return mime, b64


def _extract_json(text: str) -> dict:
    """Extract the first complete JSON object from text."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output.")
    obj, _ = decoder.raw_decode(text, start)
    return obj


def _normalize_fields(data: dict) -> dict:
    """Normalize common LLM output quirks before Pydantic validation."""
    if "declared_unit" in data and data["declared_unit"] is not None:
        data["declared_unit"] = str(data["declared_unit"]).lower().strip()

    if "product_form" in data and data["product_form"] is not None:
        val = str(data["product_form"]).lower().strip()
        # Common LLM aliases
        aliases = {
            "tea": "Tea Bags",
            "beans": "Whole Bean",
            "coffee beans": "Whole Bean",
            "loose": "Loose Leaf",
            "capsules": "Capsule",
            "wipes": "Wipe",
            "tablets": "tablet",
        }
        data["product_form"] = aliases.get(val, val)

    return data


ENUM_HINT = """
Allowed values:

product_form:
['liquid','solid','powder','gel','spray','cream','bar','tablet','strip','granule',
'Whole Bean','Ground','Loose Leaf','Tea Bags','Capsule','Wipe']

declared_unit:
['oz','fl_oz','g','kg','lb','ml','l','count','piece','dram','other']

category: Use EXACT category names like 'Grocery & Gourmet Food', 'Electronics', 'Beauty & Personal Care', etc.
"""


# ── Main Extractor ────────────────────────────────────────────────────────────

def extract_and_validate(
    image_path: str = "",
    catalog_text: str = "",
    model: str = None,
) -> ProductExtraction:
    """
    Extract structured product data from an image and/or catalog text.

    Args:
        image_path: Local path to product image (optional).
        catalog_text: Catalog description text (optional).
        model: Override the Ollama model (defaults to config.OLLAMA_MODEL).

    Returns:
        ProductExtraction: Validated pydantic model.

    Raises:
        ValueError: If extraction fails after MAX_RETRIES.
        RuntimeError: If neither image_path nor catalog_text is provided.
    """
    if not image_path and not catalog_text:
        raise RuntimeError("At least one of image_path or catalog_text must be provided.")

    model = model or OLLAMA_MODEL

    # Build initial user message
    user_content = []
    if image_path and Path(image_path).exists():
        mime, b64 = _encode_image(image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })
    if catalog_text:
        user_content.append({"type": "text", "text": catalog_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
        )

        raw = response.choices[0].message.content
        json_str = re.sub(r"```(?:json)?|```", "", raw).strip()

        # ── JSON Parse ────────────────────────────────────────────────────────
        try:
            data = _extract_json(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = str(e)
            print(f"[Attempt {attempt}/{MAX_RETRIES}] JSON parse error → retrying...")
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"JSON ERROR OCCURRED.\n\nError:\n{last_error}\n\n"
                    "Fix ONLY the JSON formatting.\n"
                    "Do NOT change field values.\n"
                    "Do NOT reprocess image or text.\n"
                    "Return ONLY valid JSON.\nNO markdown.\nNO explanation."
                )},
            ]
            continue

        # ── Normalize ─────────────────────────────────────────────────────────
        data = _normalize_fields(data)

        # ── Pydantic Validation ───────────────────────────────────────────────
        try:
            return ProductExtraction(**data)
        except ValidationError as e:
            last_error = str(e)
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Validation error → retrying...")
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"PYDANTIC VALIDATION ERROR.\n\nError:\n{last_error}\n\n"
                    f"{ENUM_HINT}\n\n"
                    "STRICT RULES:\n"
                    "- Fix ONLY invalid fields\n"
                    "- Keep all correct fields unchanged\n"
                    "- DO NOT reprocess image/text\n"
                    "- Choose ONLY from allowed enums\n\n"
                    "Return ONLY corrected JSON.\nNO explanation."
                )},
            ]
            continue

    raise ValueError(f"Failed after {MAX_RETRIES} attempts.\nLast error:\n{last_error}")


# ── Batch Extraction ──────────────────────────────────────────────────────────

def extract_batch(
    df,
    name: str,
    start: int,
    end: int,
    output_dir: str = "./archive",
) -> None:
    """
    Extract structured data for a batch of rows and save to CSV.

    Args:
        df: DataFrame with 'sample_id', 'catalog_content', optionally 'image_link'.
        name: Dataset split name ('train', 'test', 'sample_test').
        start: Start row index.
        end: End row index (exclusive).
        output_dir: Directory to save output CSVs.
    """
    import pandas as pd
    import os

    batch_df = df.iloc[start:end].copy()
    rows = []

    for _, row in batch_df.iterrows():
        sample_id = row["sample_id"]
        catalog_text = str(row.get("catalog_content", ""))
        image_path = f"./Images/{name}/{sample_id}.jpg"

        try:
            result = extract_and_validate(
                image_path=image_path if os.path.exists(image_path) else "",
                catalog_text=catalog_text,
            )
            flat = result.to_flat_dict()
            flat["sample_id"] = sample_id
            rows.append(flat)
            print(f"  ✓ {sample_id}")
        except Exception as e:
            print(f"  ✗ {sample_id}: {e}")
            rows.append({"sample_id": sample_id})

    result_df = pd.DataFrame(rows)
    merged = batch_df.merge(result_df, on="sample_id", how="left")

    os.makedirs(f"{output_dir}/{name}", exist_ok=True)
    out_path = f"{output_dir}/{name}/{start}-{end}.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Quick test
    result = extract_and_validate(
        catalog_text="Item Name: Tata Salt | Category: Grocery | Price: ₹25 | Weight: 1 kg"
    )
    print(result.model_dump_json(indent=2))
