import json
import re
import base64
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum
from pydantic import ValidationError

OLLAMA_API_KEY= "83d50710f12849509201568faf3251a5.IhV2rxMHRwfXWLBAeSK-65_a"# "98133eec1cad4201ac776ef5c4e5ffc0.KM4NGax25XdJa3muLvFeBSdH"

# ── Client Setup ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=OLLAMA_API_KEY,                    # required by client, ignored by Ollama
)

OLLAMA_MODEL =  "ministral-3:14b-cloud"# "gemma4:31b-cloud"# "devstral-small-2:24b-cloud"# "qwen3.5:397b-cloud"   # devstral-small-2:24b-cloud         # closest Ollama equivalent to Qwen/Qwen2.5-VL-3B


def _encode_image(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime   = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return mime, b64


MAX_RETRIES = 5

def _extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output.")
    obj, _ = decoder.raw_decode(text, start)
    return obj

def extract_and_validate(image_path: str = "", catalog_text: str = "") -> ProductExtraction:

    user_content = []
    if image_path:
        mime, b64 = _encode_image(image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })
    if catalog_text:
        user_content.append({"type": "text", "text": catalog_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content}
    ]

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content
        json_str = re.sub(r"```(?:json)?|```", "", raw).strip()

        # ── JSON PARSE ────────────────────────────────────────────────────────
        try:
            data = _extract_json(json_str)

        except (ValueError, json.JSONDecodeError) as e:
            last_error = str(e)

            print(f"[Attempt {attempt}] JSON error → retrying...\n")

            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"""
JSON ERROR OCCURRED.

Error:
{last_error}

Fix ONLY the JSON formatting.
Do NOT change field values.
Do NOT reprocess image or text.

Return ONLY valid JSON.
NO markdown.
NO explanation.
"""}
            ]
            continue

        # ── OPTIONAL NORMALIZATION (VERY IMPORTANT) ───────────────────────────
        if "declared_unit" in data:
            data["declared_unit"] = str(data["declared_unit"]).lower()

        if "product_form" in data:
            val = str(data["product_form"]).lower()
            if val == "tea":
                data["product_form"] = "Tea Bags"

        # ── PYDANTIC VALIDATION ───────────────────────────────────────────────
        try:
            return ProductExtraction(**data)

        except ValidationError as e:
            last_error = str(e)

            print(f"[Attempt {attempt}] Validation error → retrying...\n")

            enum_hint = """
Allowed values:

product_form:
['liquid','solid','powder','gel','spray','cream','bar','tablet','strip','granule',
'Whole Bean','Ground','Loose Leaf','Tea Bags','Capsule','Wipe']

declared_unit:
['oz','fl_oz','g','kg','lb','ml','l','count','piece','dram','other']
"""

            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"""
PYDANTIC VALIDATION ERROR.

Error:
{last_error}

{enum_hint}

STRICT RULES:
- Fix ONLY invalid fields
- Keep all correct fields unchanged
- DO NOT reprocess image/text
- DO NOT invent new values
- Choose ONLY from allowed enums

Return ONLY corrected JSON.
NO explanation.
"""}
            ]
            continue

    raise ValueError(f"Failed after {MAX_RETRIES} attempts.\nLast error:\n{last_error}")
    
# result = extract_and_validate(IMAGE_PATH, CATALOG_TEXT)