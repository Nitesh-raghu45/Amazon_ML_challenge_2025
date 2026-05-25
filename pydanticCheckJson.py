from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

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


NULL_STRINGS = {"null", "n/a", "unknown", "none", ""}


def is_null_like(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in NULL_STRINGS)


# ─────────────────────────────────────────────
# GROUP A — Required integers, NEVER null
# ─────────────────────────────────────────────

class GroupA_RequiredFlags(BaseModel):
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


# ─────────────────────────────────────────────
# GROUP B — Optional integers, null if not mentioned
# ─────────────────────────────────────────────

class GroupB_OptionalFlags(BaseModel):
    is_organic: Optional[int] = Field(default=None)
    is_natural: Optional[int] = Field(default=None)

    @field_validator("is_organic", "is_natural", mode="before")
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


# ─────────────────────────────────────────────
# GROUP C — Optional strings, null if not visible
# ─────────────────────────────────────────────

class GroupC_OptionalStrings(BaseModel):
    variant: Optional[str] = Field(default=None)
    target_demographic: Optional[TargetDemographic] = Field(default=None)
    country_of_origin: Optional[str] = Field(default=None)
    manufacturer: Optional[str] = Field(default=None)

    @field_validator("variant", "country_of_origin", "manufacturer", mode="before")
    @classmethod
    def reject_placeholder_strings(cls, v):
        if is_null_like(v):
            return None
        return v


# ─────────────────────────────────────────────
# GROUP D — Arrays, always output an array
# ─────────────────────────────────────────────

class GroupD_Arrays(BaseModel):
    certifications: list[str] = Field(default_factory=list)

    @field_validator("certifications", mode="before")
    @classmethod
    def must_be_list(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("certifications must be an array.")
        return v


# ─────────────────────────────────────────────
# GROUP E — Numbers, null if not visible
# ─────────────────────────────────────────────

class GroupE_Numbers(BaseModel):
    declared_quantity: Optional[float] = Field(default=None)

    @field_validator("declared_quantity", mode="before")
    @classmethod
    def must_be_number_or_null(cls, v):
        if is_null_like(v):
            return None
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except ValueError:
                raise ValueError(f"declared_quantity must be a bare number, not a string: '{v}'")
        return float(v)


# ─────────────────────────────────────────────
# FULL PRODUCT EXTRACTION MODEL
# ─────────────────────────────────────────────

class ProductExtraction(
    GroupA_RequiredFlags,
    GroupB_OptionalFlags,
    GroupC_OptionalStrings,
    GroupD_Arrays,
    GroupE_Numbers,
):
    price: Optional[str] = Field(default=None)
    brand_name: Optional[str] = None
    product_name: Optional[str] = None

    category: Optional[Category] = None
    product_form: Optional[ProductForm] = None
    declared_unit: Optional[DeclaredUnit] = None
    packaging_type: Optional[PackagingType] = None
    packaging_material: Optional[PackagingMaterial] = None

    overall_summary: Optional[str] = None

    model_config = {"use_enum_values": True}


if __name__ == "__main__":
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
    }

    product = ProductExtraction(**raw)
    print(product.model_dump_json(indent=5))