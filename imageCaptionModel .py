import json
import re
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum

# ── Model Load ───────────────────────────────────────────────────────────────

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B",device='cuda')
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B",device_map="cuda")

# ── Inference + Validation ───────────────────────────────────────────────────

from pydantic import ValidationError

MAX_RETRIES = 5

def _extract_json(text: str) -> dict:
    """Extract the first complete JSON object from text, ignoring trailing garbage."""
    decoder = json.JSONDecoder()
    # find the first '{' and decode only up to the matching '}'
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output.")
    obj, _ = decoder.raw_decode(text, start)
    return obj

def extract_and_validate(image_path: str = "", catalog_text: str = "") -> ProductExtraction:

    base_user_content = []
    if image_path:
        base_user_content.append({"type": "image", "url": image_path})
    if catalog_text:
        base_user_content.append({"type": "text", "text": catalog_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": base_user_content}
    ]

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=1024)

        raw = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        json_str = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            data = _extract_json(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = f"JSON parse error: {e}"

            # 🔥 FEEDBACK MESSAGE
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": raw}]
            })

            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"""
JSON ERROR OCCURRED.

Error:
{last_error}

Fix the JSON formatting ONLY.
Return valid JSON.
Do NOT reprocess the image or text.
Output ONLY corrected JSON.
"""
                }]
            })

            print(f"[Attempt {attempt}] JSON error → retrying...\n")
            continue

        try:
            return ProductExtraction(**data)

        except ValidationError as e:
            last_error = str(e)

            # 🔥 ENUM GUIDANCE
            enum_hint = """
Allowed values:

product_form:
['liquid','solid','powder','gel','spray','cream','bar','tablet','strip','granule',
'Whole Bean','Ground','Loose Leaf','Tea Bags','Capsule','Wipe']

declared_unit:
['oz','fl_oz','g','kg','lb','ml','l','count','piece','dram','other']
"""

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": raw}]
            })

            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"""
PYDANTIC VALIDATION ERROR.

Error:
{last_error}

{enum_hint}

Fix ONLY the invalid fields.
Do NOT change correct fields.
Do NOT reprocess image/text.

Return ONLY corrected JSON.
"""
                }]
            })

            print(f"[Attempt {attempt}] Validation error → retrying...\n")
            continue

    raise ValueError(f"Failed after {MAX_RETRIES} attempts:\n{last_error}")
# # ── Run ───────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     # all 4 valid combinations handled cleanly
#     result = extract_and_validate(
#         image_path   = "./product_images/candy.jpg",    # optional — remove or "" if missing
#         catalog_text = "Item Name: Skittles Original..."# optional — remove or "" if missing
#     )
#     print(result.model_dump_json(indent=2))