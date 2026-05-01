"""
utils.py — Shared constants, env loading, and CSV I/O helpers.
"""

from __future__ import annotations

import os
import pathlib
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data"
SUPPORT_DIR = REPO_ROOT / "support_tickets"

INPUT_CSV = SUPPORT_DIR / "support_tickets.csv"
OUTPUT_CSV = SUPPORT_DIR / "output.csv"
SAMPLE_CSV = SUPPORT_DIR / "sample_support_tickets.csv"

DOMAIN_DIRS: Dict[str, pathlib.Path] = {
    "hackerrank": DATA_DIR / "hackerrank",
    "claude": DATA_DIR / "claude",
    "visa": DATA_DIR / "visa",
}

# ---------------------------------------------------------------------------
# Claude API constants
# ---------------------------------------------------------------------------

MODEL_NAME = "claude-haiku-4-5"
MAX_TOKENS = 1500
TEMPERATURE = 0  # determinism is evaluated

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]

VALID_STATUSES = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

# ---------------------------------------------------------------------------
# Retrieval constants
# ---------------------------------------------------------------------------

TOP_K = 7
DOMAIN_BOOST = 1.3  # multiplied onto RRF score for matching-domain chunks
RRF_K = 60          # standard RRF constant

# ---------------------------------------------------------------------------
# Env / API key
# ---------------------------------------------------------------------------


def load_env() -> str:
    """Load environment variables from .env if present, return the API key."""
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it as an environment variable or add it to a .env file in the project root.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return api_key


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def read_tickets(path: pathlib.Path | str = INPUT_CSV) -> pd.DataFrame:
    """Load the support tickets CSV, normalising column names."""
    df = pd.read_csv(str(path), encoding="utf-8", encoding_errors="replace")
    # Normalise column names: strip whitespace, lowercase
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Fill NaN company with the string "None"
    if "company" in df.columns:
        df["company"] = df["company"].fillna("None").astype(str).str.strip()
    # Ensure issue / subject are strings
    for col in ("issue", "subject"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def write_output(results: List[Dict[str, str]], path: pathlib.Path | str = OUTPUT_CSV) -> None:
    """Write the triage results to output.csv with exactly the expected column order."""
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    # Coerce to expected values just in case
    df["status"] = df["status"].str.lower().str.strip()
    df["request_type"] = df["request_type"].str.lower().str.strip()
    df.to_csv(str(out_path), index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, appending '...' if shortened."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def normalise_company(raw: str) -> str | None:
    """
    Normalise the Company column value to a canonical domain string
    used in DOMAIN_DIRS.  Returns None if company is unknown.
    """
    mapping = {
        "hackerrank": "hackerrank",
        "claude": "claude",
        "anthropic": "claude",
        "visa": "visa",
        "none": None,
        "": None,
    }
    return mapping.get(raw.lower().strip())
