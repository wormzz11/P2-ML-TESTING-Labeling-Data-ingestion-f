from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "ground_truth_fallback" / "combined_truth.csv"
from dataclasses import dataclass

@dataclass
class ThresholdConfig:
    high_neg: float = 0.2
    high_pos: float = 0.26
    