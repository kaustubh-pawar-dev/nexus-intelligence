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

# ---------------------------------------------------------------------------
# Orders and order lines (SYNTHETIC)
# ---------------------------------------------------------------------------
STREAM_ORDERS = 4
STREAM_ORDER_ITEMS = 5

# Average orders per active customer per month, before channel, opt-in and noise.
SEGMENT_MONTHLY_ORDER_RATE = {"Consumer": 0.18, "Small Business": 0.45, "Enterprise": 1.20}
CHANNEL_RATE_MULTIPLIER = {
    "Organic Search": 1.00,
    "Paid Search": 0.90,
    "Social Media": 0.85,
    "Email": 1.10,
    "Referral": 1.15,
    "Direct": 1.05,
}
OPT_IN_RATE_MULTIPLIER = 1.10
# Per-customer rate noise: gamma with mean 1 (some customers are much more active).
CUSTOMER_RATE_GAMMA_SHAPE = 2.0

# A customer stops ordering after an exponential lifetime counted from signup.
SEGMENT_MEAN_LIFETIME_DAYS = {"Consumer": 540, "Small Business": 800, "Enterprise": 1200}
CHANNEL_LIFETIME_MULTIPLIER = {
    "Organic Search": 1.00,
    "Paid Search": 0.80,
    "Social Media": 0.75,
    "Email": 1.10,
    "Referral": 1.25,
    "Direct": 1.00,
}

# Seasonality: index 0 = January. Weekday factors: index 0 = Monday.
MONTH_SEASONALITY = (0.90, 0.85, 0.95, 0.95, 1.00, 0.95, 0.95, 1.00, 1.00, 1.10, 1.35, 1.30)
WEEKDAY_FACTOR = (1.00, 1.00, 1.00, 1.00, 1.05, 1.15, 1.10)

SALES_CHANNELS = ("Web", "Mobile App", "Marketplace")
SALES_CHANNEL_PROBS = (0.55, 0.35, 0.10)

ORDER_STATUSES = ("Completed", "Cancelled")
ORDER_CANCEL_RATE = 0.04

DISCOUNT_LEVELS = (0.05, 0.10, 0.15, 0.20)
DISCOUNT_PROBS = (0.40, 0.30, 0.20, 0.10)
DISCOUNT_PROB_BASE = 0.20
DISCOUNT_PROB_NOVEMBER = 0.45

# Lines per order = base + Poisson(extra mean), capped.
SEGMENT_ITEMS_BASE = {"Consumer": 1, "Small Business": 1, "Enterprise": 2}
SEGMENT_ITEMS_EXTRA_MEAN = {"Consumer": 0.6, "Small Business": 1.2, "Enterprise": 2.0}
MAX_ITEMS_PER_ORDER = 8

# Quantity per line = base + Poisson(extra mean), capped.
SEGMENT_QTY_BASE = {"Consumer": 1, "Small Business": 1, "Enterprise": 2}
SEGMENT_QTY_EXTRA_MEAN = {"Consumer": 0.4, "Small Business": 1.5, "Enterprise": 4.0}
MAX_QUANTITY = 20

PRODUCT_POPULARITY_SIGMA = 0.8
# Inactive products are not sold in the last N days of the window.
INACTIVE_PRODUCT_CUTOFF_DAYS = 180

# ---------------------------------------------------------------------------
# Payments and refunds (SYNTHETIC)
# ---------------------------------------------------------------------------
STREAM_PAYMENTS = 6
STREAM_REFUNDS = 7

PAYMENT_METHODS = (
    "Credit Card", "Debit Card", "UPI", "Net Banking", "Digital Wallet", "Bank Transfer",
)
# Probabilities follow the order of PAYMENT_METHODS and each row sums to 1.
SEGMENT_PAYMENT_METHOD_PROBS = {
    "Consumer": (0.30, 0.20, 0.25, 0.08, 0.12, 0.05),
    "Small Business": (0.35, 0.15, 0.15, 0.10, 0.05, 0.20),
    "Enterprise": (0.15, 0.02, 0.03, 0.10, 0.00, 0.70),
}

PAYMENT_STATUSES = ("Captured", "Failed")
# A failed payment on a completed order is simulated uncollected revenue.
PAYMENT_FAILURE_RATE = {
    "Credit Card": 0.025,
    "Debit Card": 0.035,
    "UPI": 0.030,
    "Net Banking": 0.060,
    "Digital Wallet": 0.020,
    "Bank Transfer": 0.010,
}

# Days between order and payment. Bank transfers add 1 to 7 extra days.
PAYMENT_DELAY_DAYS = (0, 1, 2, 3)
PAYMENT_DELAY_PROBS = (0.80, 0.12, 0.05, 0.03)
BANK_TRANSFER_EXTRA_DELAY_MAX = 7

# Refund probability per captured payment = base + sensitivity * order defect score,
# where the defect score is the line-value-weighted supplier defect rate of the order.
REFUND_BASE_PROB = 0.02
REFUND_DEFECT_SENSITIVITY = 2.0
REFUND_FULL_SHARE = 0.60
REFUND_PARTIAL_RANGE = (0.20, 0.80)
REFUND_MIN_DAYS = 3
REFUND_MAX_DAYS = 30
REFUND_TYPES = ("Full", "Partial")

DEFECTIVE_REASON = "Defective"
DEFECTIVE_REASON_BASE = 0.15
DEFECTIVE_REASON_SLOPE = 8.0
DEFECTIVE_REASON_MAX = 0.70
OTHER_REFUND_REASONS = ("Not As Described", "Changed Mind", "Late Delivery", "Wrong Item")
OTHER_REFUND_REASON_PROBS = (0.30, 0.35, 0.20, 0.15)
REFUND_REASONS = (DEFECTIVE_REASON, *OTHER_REFUND_REASONS)

# ---------------------------------------------------------------------------
# Purchase orders and supplier deliveries (SYNTHETIC)
# ---------------------------------------------------------------------------
STREAM_PURCHASE_ORDERS = 8

MIN_PO_QTY = 5

# Orders placed in these months face congestion: on-time probability drops for all suppliers.
PEAK_SEASON_MONTHS = (10, 11, 12)
PEAK_SEASON_ON_TIME_PENALTY = 0.06
ON_TIME_PROB_MIN = 0.05
ON_TIME_PROB_MAX = 0.995

# On time: delivery 0 to N days early. Late: 1 + Poisson(extra mean) days after the plan.
EARLY_DELIVERY_MAX_DAYS = 2
LATE_DELAY_EXTRA_MEAN = 4.0

# Short shipments: a share of deliveries arrive with fewer units than ordered.
SHORT_SHIPMENT_PROB = 0.05
SHORT_SHIPMENT_RANGE = (0.05, 0.30)

DELIVERY_STATUSES = ("Delivered", "In Transit")


# ---------------------------------------------------------------------------
# Stock-aware replenishment (SYNTHETIC)
# ---------------------------------------------------------------------------
# Each purchase order tops the inventory position up to
#   demand over (planned lead time + PO_REVIEW_DAYS) * PO_SAFETY_BUFFER * forecast noise.
PO_REVIEW_DAYS = 45
PO_SAFETY_BUFFER = 1.15
PO_FORECAST_NOISE_SIGMA = 0.10

# Each product starts with stock on its first listed day (window start for older products).
INITIAL_STOCK_COVER_DAYS = 45
INITIAL_STOCK_BUFFER = 1.25