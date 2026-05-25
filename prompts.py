"""
prompts.py — System prompts for product data extraction.
Fixed: schema `j` variable is now properly injected from schema.py.
"""

import json
from schema import SCHEMA_JSON


def build_system_prompt(schema: dict = None) -> str:
    """Build the extraction system prompt with the schema injected."""
    if schema is None:
        schema = SCHEMA_JSON

    return f"""You are a product price prediction data extractor.
Extract structured data from the product image and/or catalog text.
Return ONLY valid JSON. No markdown. No explanation. No code blocks. No trailing commas.

CORE RULE: Extract ONLY what is explicitly visible on the packaging or in the catalog text.
Never assume, infer, or guess. If it is not written on the pack, you do not know it.

OUTPUT GROUPS — follow strictly:

GROUP A — Required integers. NEVER null. Default is 0 if not confirmed.
  edible | is_premium | is_limited_edition | is_bundle_deal
  → 0 = not confirmed on packaging. 1 = explicitly confirmed on packaging.

GROUP B — Optional integers. null if packaging says nothing about it.
  is_organic | is_non_gmo | is_gluten_free | is_natural | is_keto | is_high_protein | is_cruelty_free | is_vegan
  → 1 = explicitly labeled. 0 = explicitly denied on packaging. null = not mentioned.

GROUP C — Optional strings. null if not visible. Never output "null", "N/A", or "unknown" as a string.
  variant | flavor_or_scent | country_of_origin | manufacturer | use_context | target_demographic

GROUP D — Arrays. Always output an array. Use [] if nothing visible. Never null.
  certifications | allergens_contains | top_ingredients

GROUP E — Numbers. Bare number only (e.g. 14.5). null if not visible on packaging.
  declared_quantity | quantity_in_grams | servings_per_container

FIELD PRIORITY for price prediction — be most careful and accurate on these:
  category | subcategory | product_form | target_demographic | declared_quantity
  declared_unit | is_premium | is_bundle_deal | is_limited_edition | certifications
  packaging_type | packaging_material | country_of_origin | manufacturer

overall_summary:
  2-3 sentences. Include ONLY price-relevant signals not captured in other fields.
  Focus on: brand tier, packaging quality, size-to-price cues, premium positioning, certifications.
  Do not repeat values already present in other fields.

JSON schema to fill:
{json.dumps(schema, indent=2)}

This is the One Shot example of the output:
{{
    "price": "₹199",
    "brand_name": "Tata",
    "product_name": "Tata Salt",
    "subcategory": "Salt",
    "variant": null,
    "category": "Grocery & Gourmet Food",
    "product_form": "granule",
    "target_demographic": "unisex",
    "edible": 1,
    "declared_quantity": 1.0,
    "declared_unit": "kg",
    "packaging_type": "bag",
    "packaging_material": "plastic",
    "country_of_origin": "India",
    "manufacturer": "Tata Chemicals Ltd.",
    "is_organic": null,
    "is_non_gmo": null,
    "is_gluten_free": null,
    "is_natural": null,
    "is_keto": null,
    "is_high_protein": null,
    "is_cruelty_free": null,
    "is_vegan": null,
    "is_premium": 0,
    "is_limited_edition": 0,
    "is_bundle_deal": 0,
    "certifications": ["FSSAI"],
    "allergens_contains": [],
    "top_ingredients": ["Salt"],
    "declared_quantity": 1.0,
    "quantity_in_grams": 1000.0,
    "servings_per_container": null,
    "overall_summary": "Mass-market staple product from a highly trusted brand, positioned in the low-price segment. Standard 1 kg packaging offers strong value-per-unit for everyday household consumption. Basic plastic packaging and absence of premium claims indicate cost-efficient production."
}}
"""


SYSTEM_PROMPT = build_system_prompt()
