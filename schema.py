"""
schema.py — Pydantic schema for structured product data extraction.
Consolidated and cleaned from pydanticCheckJson.py.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class Category(str, Enum):
    appliances = "Appliances"
    apps_games = "Apps & Games"
    arts_crafts = "Arts, Crafts & Sewing"
    automotive = "Automotive Parts & Accessories"
    baby = "Baby"
    beauty = "Beauty & Personal Care"
    books = "Books"
    cds_vinyl = "CDs & Vinyl"
    cell_phones = "Cell Phones & Accessories"
    clothing = "Clothing, Shoes & Jewelry"
    collectibles = "Collectibles & Fine Art"
    computers = "Computers"
    digital_music = "Digital Music"
    electronics = "Electronics"
    entertainment = "Entertainment Collectibles"
    gift_cards = "Gift Cards"
    grocery = "Grocery & Gourmet Food"
    handmade = "Handmade"
    health = "Health & Household"
    home_kitchen = "Home & Kitchen"
    industrial = "Industrial & Scientific"
    kindle = "Kindle Store"
    luggage = "Luggage & Travel Gear"
    luxury = "Luxury Stores"
    movies_tv = "Movies & TV"
    musical = "Musical Instruments"
    office = "Office Products"
    patio = "Patio, Lawn & Garden"
    pet = "Pet Supplies"
    smart_home = "Smart Home"
    software = "Software"
    sports = "Sports & Outdoors"
    subscriptions = "Subscription Boxes"
    tools = "Tools & Home Improvement"
    toys = "Toys & Games"
    video_games = "Video Games"


class ProductForm(str, Enum):
    liquid = "liquid"
    solid = "solid"
    powder = "powder"
    gel = "gel"
    spray = "spray"
    cream = "cream"
    bar = "bar"
    tablet = "tablet"
    strip = "strip"
    granule = "granule"
    whole_bean = "Whole Bean"
    ground = "Ground"
    loose_leaf = "Loose Leaf"
    tea_bags = "Tea Bags"
    capsule = "Capsule"
    wipe = "Wipe"


class DeclaredUnit(str, Enum):
    oz = "oz"
    fl_oz = "fl_oz"
    g = "g"
    kg = "kg"
    lb = "lb"
    ml = "ml"
    l = "l"
    count = "count"
    piece = "piece"
    dram = "dram"
    other = "other"


class PackagingType(str, Enum):
    box = "box"
    bag = "bag"
    bottle = "bottle"
    can = "can"
    jar = "jar"
    pouch = "pouch"
    container = "container"
    tube = "tube"
    carton = "carton"
    wrapper = "wrapper"
    basket = "basket"
    block = "block"
    other = "other"


class PackagingMaterial(str, Enum):
    cardboard = "cardboard"
    plastic = "plastic"
    metal = "metal"
    glass = "glass"
    wood = "wood"
    mixed = "mixed"
    other = "other"


class TargetDemographic(str, Enum):
    unisex = "unisex"
    female = "female"
    male = "male"
    kids = "kids"
    baby = "baby"
    adult = "adult"


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

NULL_STRINGS = {"null", "n/a", "unknown", "none", ""}


def is_null_like(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in NULL_STRINGS)


class GroupA_RequiredFlags(BaseModel):
    """Group A — Required binary integers, default 0 if not confirmed."""
    edible: int = Field(default=0)
    is_premium: int = Field(default=0)
    is_bundle_deal: int = Field(default=0)
    is_limited_edition: int = Field(default=0)

    @field_validator("edible", "is_premium", "is_bundle_deal", "is_limited_edition", mode="before")
    @classmethod
    def must_be_binary(cls, v):
        if is_null_like(v):
            return 0
        if isinstance(v, str):
            v = v.strip()
            if v in ("0", "1"):
                v = int(v)
            else:
                raise ValueError("Must be 0 or 1")
        if v not in (0, 1):
            raise ValueError("Group A fields must be 0 or 1, never null.")
        return v


class GroupB_OptionalFlags(BaseModel):
    """Group B — Optional binary integers, null if not mentioned."""
    is_organic: Optional[int] = Field(default=None)
    is_non_gmo: Optional[int] = Field(default=None)
    is_gluten_free: Optional[int] = Field(default=None)
    is_natural: Optional[int] = Field(default=None)
    is_keto: Optional[int] = Field(default=None)
    is_high_protein: Optional[int] = Field(default=None)
    is_cruelty_free: Optional[int] = Field(default=None)
    is_vegan: Optional[int] = Field(default=None)

    @field_validator(
        "is_organic", "is_non_gmo", "is_gluten_free", "is_natural",
        "is_keto", "is_high_protein", "is_cruelty_free", "is_vegan",
        mode="before"
    )
    @classmethod
    def must_be_binary_or_null(cls, v):
        if is_null_like(v):
            return None
        if isinstance(v, str):
            v = v.strip()
            if v in ("0", "1"):
                v = int(v)
            else:
                raise ValueError("Must be 0, 1, or null")
        if v not in (0, 1):
            raise ValueError("Group B fields must be 0, 1, or null.")
        return v


class GroupC_OptionalStrings(BaseModel):
    """Group C — Optional strings, null if not visible on packaging."""
    variant: Optional[str] = Field(default=None)
    flavor_or_scent: Optional[str] = Field(default=None)
    country_of_origin: Optional[str] = Field(default=None)
    manufacturer: Optional[str] = Field(default=None)
    use_context: Optional[str] = Field(default=None)
    target_demographic: Optional[TargetDemographic] = Field(default=None)

    @field_validator("variant", "flavor_or_scent", "country_of_origin",
                     "manufacturer", "use_context", mode="before")
    @classmethod
    def reject_placeholder_strings(cls, v):
        if is_null_like(v):
            return None
        return v


class GroupD_Arrays(BaseModel):
    """Group D — Arrays, always an array. Never null."""
    certifications: List[str] = Field(default_factory=list)
    allergens_contains: List[str] = Field(default_factory=list)
    top_ingredients: List[str] = Field(default_factory=list)

    @field_validator("certifications", "allergens_contains", "top_ingredients", mode="before")
    @classmethod
    def must_be_list(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("Must be an array.")
        return v


class GroupE_Numbers(BaseModel):
    """Group E — Numbers, null if not visible on packaging."""
    declared_quantity: Optional[float] = Field(default=None)
    quantity_in_grams: Optional[float] = Field(default=None)
    servings_per_container: Optional[float] = Field(default=None)

    @field_validator("declared_quantity", "quantity_in_grams", "servings_per_container", mode="before")
    @classmethod
    def must_be_number_or_null(cls, v):
        if is_null_like(v):
            return None
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                raise ValueError(f"Must be a bare number, not a string: '{v}'")
        return float(v)


# ─────────────────────────────────────────────────────────────────────────────
# FULL PRODUCT EXTRACTION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ProductExtraction(
    GroupA_RequiredFlags,
    GroupB_OptionalFlags,
    GroupC_OptionalStrings,
    GroupD_Arrays,
    GroupE_Numbers,
):
    """Full product extraction schema for price prediction."""
    price: Optional[str] = Field(default=None)
    brand_name: Optional[str] = None
    product_name: Optional[str] = None
    subcategory: Optional[str] = None

    category: Optional[Category] = None
    product_form: Optional[ProductForm] = None
    declared_unit: Optional[DeclaredUnit] = None
    packaging_type: Optional[PackagingType] = None
    packaging_material: Optional[PackagingMaterial] = None

    overall_summary: Optional[str] = None

    model_config = {"use_enum_values": True}

    def to_flat_dict(self) -> dict:
        """Flatten arrays to indexed columns for DataFrame storage."""
        d = self.model_dump()
        flat = {}
        for key, value in d.items():
            if isinstance(value, list):
                for i, item in enumerate(value[:3]):  # max 3 per array
                    flat[f"{key}_{i}"] = item
                for i in range(len(value), 3):
                    flat[f"{key}_{i}"] = None
            else:
                flat[key] = value
        return flat


# ─────────────────────────────────────────────────────────────────────────────
# Schema JSON (for system prompt injection)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_JSON = {
    "price": "string or null",
    "brand_name": "string or null",
    "product_name": "string or null",
    "subcategory": "string or null",
    "category": "one of the Category enum values or null",
    "product_form": "one of the ProductForm enum values or null",
    "declared_unit": "one of the DeclaredUnit enum values or null",
    "packaging_type": "one of the PackagingType enum values or null",
    "packaging_material": "one of the PackagingMaterial enum values or null",
    "target_demographic": "unisex|female|male|kids|baby|adult or null",
    "edible": "0 or 1 (required)",
    "is_premium": "0 or 1 (required)",
    "is_bundle_deal": "0 or 1 (required)",
    "is_limited_edition": "0 or 1 (required)",
    "is_organic": "0|1|null",
    "is_non_gmo": "0|1|null",
    "is_gluten_free": "0|1|null",
    "is_natural": "0|1|null",
    "is_keto": "0|1|null",
    "is_high_protein": "0|1|null",
    "is_cruelty_free": "0|1|null",
    "is_vegan": "0|1|null",
    "variant": "string or null",
    "flavor_or_scent": "string or null",
    "country_of_origin": "string or null",
    "manufacturer": "string or null",
    "use_context": "string or null",
    "declared_quantity": "number or null",
    "quantity_in_grams": "number or null",
    "servings_per_container": "number or null",
    "certifications": "array of strings (can be [])",
    "allergens_contains": "array of strings (can be [])",
    "top_ingredients": "array of strings (can be [])",
    "overall_summary": "string or null",
}


if __name__ == "__main__":
    import json
    # Test with Tata Salt example
    raw = {
        "price": "₹199",
        "brand_name": "Tata",
        "product_name": "Tata Salt",
        "variant": None,
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
        "is_organic": None,
        "is_natural": None,
        "is_premium": 0,
        "is_limited_edition": 0,
        "is_bundle_deal": 0,
        "certifications": ["FSSAI"],
        "allergens_contains": [],
        "top_ingredients": ["Salt"],
    }
    product = ProductExtraction(**raw)
    print(product.model_dump_json(indent=2))
    print("\nFlat dict:")
    print(json.dumps(product.to_flat_dict(), indent=2, default=str))
