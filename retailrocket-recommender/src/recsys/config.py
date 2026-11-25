"""Central configuration: file locations, event weights and experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EVENTS_FILE = RAW_DIR / "events.csv"
CATEGORY_FILE = RAW_DIR / "category_tree.csv"
ITEM_PROPERTY_FILES = [
    RAW_DIR / "item_properties_part1.csv",
    RAW_DIR / "item_properties_part2.csv",
]

# Implicit-feedback strength per event type
EVENT_WEIGHTS = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
# Events treated as "relevant" when evaluating recommendations
POSITIVE_EVENTS = {"addtocart", "transaction"}

# Data preparation
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
TEST_FRACTION = 0.2          # last 20% of the timeline is the test period

# Models
KNN_NEIGHBORS = 50
SVD_COMPONENTS = 64

# Evaluation
TOP_K = (5, 10, 20)
MAX_EVAL_USERS = 5000
EVAL_BATCH_SIZE = 500
RANDOM_SEED = 42
