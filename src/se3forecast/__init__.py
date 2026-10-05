"""Day-ahead electricity price forecasting for Stockholm (bidding zone SE3)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "reports" / "figures"

TZ = "Europe/Stockholm"
AREAS = ["SE1", "SE2", "SE3", "SE4"]
