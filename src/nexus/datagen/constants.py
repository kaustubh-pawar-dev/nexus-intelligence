"""Shared constants for synthetic data generation (all data is SYNTHETIC)."""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

# Fixed observation window so that results are reproducible.
DATA_START = date(2023, 1, 1)
DATA_END = date(2025, 12, 31)

N_SUPPLIERS = 24
N_PRODUCTS = 120

# Each table gets its own random stream, derived from [seed, stream_id].
# Adding a new table later never changes the output of existing tables.
STREAM_SUPPLIERS = 1
STREAM_PRODUCTS = 2


class CountryProfile(NamedTuple):
    region: str
    base_lead_days: int


COUNTRY_PROFILES: dict[str, CountryProfile] = {
    "India": CountryProfile("Asia", 12),
    "China": CountryProfile("Asia", 25),
    "Vietnam": CountryProfile("Asia", 22),
    "Germany": CountryProfile("Europe", 10),
    "Poland": CountryProfile("Europe", 9),
    "Turkey": CountryProfile("Europe", 12),
    "Mexico": CountryProfile("North America", 8),
    "United States": CountryProfile("North America", 6),
    "Brazil": CountryProfile("South America", 20),
    "United Arab Emirates": CountryProfile("Middle East", 10),
}

SUPPLIER_NAME_PREFIXES = (
    "Apex", "Brightline", "Cobalt", "Dynamo", "Evergreen", "Falcon",
    "Granite", "Horizon", "Ironwood", "Juniper", "Keystone", "Lumen",
)
SUPPLIER_NAME_SUFFIXES = (
    "Supply Co", "Industries", "Trading", "Manufacturing", "Logistics", "Components",
)

PAYMENT_TERMS_DAYS = (15, 30, 45, 60, 90)
PAYMENT_TERMS_PROBS = (0.10, 0.35, 0.25, 0.20, 0.10)


class CategoryProfile(NamedTuple):
    code: str
    cost_low: float
    cost_high: float
    markup_low: float
    markup_high: float
    nouns: tuple[str, ...]


CATEGORY_PROFILES: dict[str, CategoryProfile] = {
    "Electronics": CategoryProfile(
        "ELE", 15.0, 400.0, 0.25, 0.55,
        ("Headphones", "Speaker", "Charger", "Webcam", "Power Bank", "Keyboard"),
    ),
    "Home & Kitchen": CategoryProfile(
        "HOM", 5.0, 120.0, 0.35, 0.80,
        ("Blender", "Kettle", "Cookware Set", "Storage Jar", "Knife Set", "Air Fryer"),
    ),
    "Office Supplies": CategoryProfile(
        "OFF", 2.0, 60.0, 0.40, 0.90,
        ("Notebook", "Desk Organizer", "Pen Set", "Stapler", "Planner", "File Folder"),
    ),
    "Sports & Outdoors": CategoryProfile(
        "SPO", 8.0, 200.0, 0.30, 0.70,
        ("Yoga Mat", "Water Bottle", "Backpack", "Dumbbell Set", "Camping Lantern", "Jump Rope"),
    ),
    "Health & Personal Care": CategoryProfile(
        "HEA", 3.0, 80.0, 0.45, 1.00,
        ("Trimmer", "Massager", "Vitamin Pack", "Sunscreen", "Hair Dryer", "Toothbrush"),
    ),
    "Toys & Games": CategoryProfile(
        "TOY", 4.0, 90.0, 0.40, 0.85,
        ("Puzzle", "Board Game", "Building Blocks", "Plush Toy", "RC Car", "Card Game"),
    ),
}

PRODUCT_ADJECTIVES = (
    "Premium", "Classic", "Compact", "Smart", "Eco",
    "Pro", "Essential", "Deluxe", "Everyday", "Ultra",
)
# ---------------------------------------------------------------------------
# Customers (SYNTHETIC)
# ---------------------------------------------------------------------------
N_CUSTOMERS = 5000
STREAM_CUSTOMERS = 3

# Customers can sign up before the analysis window, so history exists on day one.
CUSTOMER_SIGNUP_START = date(2021, 1, 1)


class CustomerCountry(NamedTuple):
    region: str
    weight: float


CUSTOMER_COUNTRY_PROFILES: dict[str, CustomerCountry] = {
    "India": CustomerCountry("Asia", 0.28),
    "United States": CustomerCountry("North America", 0.24),
    "United Kingdom": CustomerCountry("Europe", 0.10),
    "Germany": CustomerCountry("Europe", 0.08),
    "Australia": CustomerCountry("Oceania", 0.07),
    "United Arab Emirates": CustomerCountry("Middle East", 0.07),
    "Singapore": CustomerCountry("Asia", 0.06),
    "Canada": CustomerCountry("North America", 0.10),
}

CUSTOMER_SEGMENTS = ("Consumer", "Small Business", "Enterprise")
SEGMENT_PROBS = (0.70, 0.22, 0.08)

ACQUISITION_CHANNELS = (
    "Organic Search", "Paid Search", "Social Media", "Email", "Referral", "Direct",
)
CHANNEL_PROBS = (0.28, 0.22, 0.15, 0.10, 0.12, 0.13)

MARKETING_OPT_IN_RATE = 0.62