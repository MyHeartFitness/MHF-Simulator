import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ----------------- Helpers -----------------
def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _to_bool(x) -> bool:
    if x is None:
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x) != 0.0
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _is_male(x) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"m", "male", "man", "masculine"}


def _smoking_status(row: pd.Series) -> str:
    """Return normalized smoking status token for mortality models."""
    raw = str(row.get("smoking_status", "")).strip().lower()
    mapping = {
        "never": "never",
        "former_gt1": "former_gt1",
        "former_lt1": "former_lt1",
        "current_le10": "current_le10",
        "current_gt10": "current_gt10",
    }
    if raw in mapping:
        return mapping[raw]

    # Fallbacks using more human labels
    if "never" in raw:
        return "never"
    if "former" in raw and (">" in raw or "gt" in raw or "1+" in raw or "over" in raw):
        return "former_gt1"
    if "former" in raw and ("<" in raw or "lt" in raw or "under" in raw):
        return "former_lt1"
    if "current" in raw:
        if "10" in raw and ("<=" in raw or "<=10" in raw or "le" in raw or "less" in raw):
            return "current_le10"
        if "10" in raw and (">" in raw or "gt" in raw or "more" in raw):
            return "current_gt10"
        # If we only know they are current, treat as heavier group conservatively
        return "current_gt10"

    # If we only have a boolean smoker column
    smoker_bool = row.get("smoker")
    if _to_bool(smoker_bool):
        return "current_gt10"
    return "never"


# ----------------- Lee score -----------------
_LEE_AGE_BANDS = [
    (60, 64, 1),
    (65, 69, 2),
    (70, 74, 3),
    (75, 79, 4),
    (80, 84, 5),
    (85, 200, 7),
]

_LEE_CONDITION_POINTS = {
    "diabetes": 1,
    "non_skin_cancer": 2,
    "copd": 2,
    "heart_failure": 2,
}

_LEE_FUNCTION_POINTS = {
    "difficulty_bathing": 2,
    "difficulty_managing_money": 2,
    "difficulty_walking": 2,
}

_LEE_PTS_TO_MORTALITY = [
    ((0, 4), 4.0),
    ((5, 6), 6.0),
    ((7, 8), 10.0),
    ((9, 10), 15.0),
    ((11, 12), 22.0),
    ((13, 14), 32.0),
    ((15, 16), 45.0),
    ((17, 1000), 64.0),
]


def lee_points_from_row(row: pd.Series) -> Optional[int]:
    """Return Lee total points for a single row or None if age missing."""
    age = _to_float(row.get("age"))
    if np.isnan(age):
        return None
    if age < 50:
        return None  # Lee score not computed below age 50

    pts = 0

    # Age bands (<60 -> 0)
    age_int = int(round(age))
    for lo, hi, band_pts in _LEE_AGE_BANDS:
        if lo <= age_int <= hi:
            pts += band_pts
            break

    # Sex
    if _is_male(row.get("sex")):
        pts += 2

    # Current smoker (any current category)
    status = _smoking_status(row)
    if status.startswith("current"):
        pts += 2

    # Conditions
    for field, p in _LEE_CONDITION_POINTS.items():
        if _to_bool(row.get(field)):
            pts += p

    # Functional difficulties
    for field, p in _LEE_FUNCTION_POINTS.items():
        if _to_bool(row.get(field)):
            pts += p

    # BMI <25
    bmi = _to_float(row.get("bmi"))
    if not np.isnan(bmi) and bmi < 25.0:
        pts += 1

    return int(pts)


def lee_mortality_pct(points: Optional[int]) -> float:
    """Return Lee 4-year mortality % given total points (np.nan if None)."""
    if points is None:
        return float("nan")
    for (lo, hi), pct in _LEE_PTS_TO_MORTALITY:
        if lo <= points <= hi:
            return pct
    return float("nan")


def compute_lee(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    pts = df.apply(lee_points_from_row, axis=1)
    pct = pts.apply(lee_mortality_pct)
    return pts, pct


# ----------------- C-Score -----------------
_SRH_POINTS = {
    "excellent": 25,
    "very good": 22,
    "very_good": 22,
    "good": 17,
    "fair": 8,
    "poor": 0,
}

_SMOKING_POINTS = {
    "never": 15,
    "former_gt1": 10,
    "former_lt1": 5,
    "current_le10": 3,
    "current_gt10": 0,
}


def _srh_points(val) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip().lower()
    return _SRH_POINTS.get(s, None)


def _whtr_points(whtr: float) -> Optional[int]:
    v = _to_float(whtr)
    if np.isnan(v):
        return None
    if v < 0.50:
        return 20
    if 0.50 <= v <= 0.54:
        return 14
    if 0.55 <= v <= 0.59:
        return 6
    return 0  # >=0.60


def _rhr_points(rhr: float) -> Optional[int]:
    v = _to_float(rhr)
    if np.isnan(v):
        return None
    if 50 <= v <= 65:
        return 20
    if 66 <= v <= 75:
        return 14
    if 76 <= v <= 85:
        return 7
    return 0


def _smoking_points(status: str) -> Optional[int]:
    return _SMOKING_POINTS.get(status, None)


def _alcohol_points(drinks_per_week: float) -> Optional[int]:
    v = _to_float(drinks_per_week)
    if np.isnan(v):
        return None
    if 0 <= v <= 7:
        return 10
    if 8 <= v <= 14:
        return 7
    if 15 <= v <= 21:
        return 3
    return 0


def _sleep_points(hours: float) -> Optional[int]:
    v = _to_float(hours)
    if np.isnan(v):
        return None
    if 7 <= v <= 8:
        return 10
    if v in {6, 9}:  # exact 6 or 9
        return 7
    if v in {5, 10}:  # exact 5 or 10
        return 3
    if v < 5 or v > 10:
        return 0
    # Non-integer between bands: interpolate by nearest band rules
    if 6 < v < 7:
        return 7
    if 8 < v < 9:
        return 7
    if 5 < v < 6:
        return 3
    if 9 < v < 10:
        return 3
    return 0


def c_score_from_row(row: pd.Series) -> Optional[int]:
    srh_pts = _srh_points(row.get("self_rated_health"))
    whtr_pts = _whtr_points(row.get("whtr"))
    rhr_pts = _rhr_points(row.get("resting_hr"))
    smoking_pts = _smoking_points(_smoking_status(row))
    alcohol_pts = _alcohol_points(row.get("drinks_per_week"))
    sleep_pts = _sleep_points(row.get("sleep_hours"))

    comps = [srh_pts, whtr_pts, rhr_pts, smoking_pts, alcohol_pts, sleep_pts]
    if any(c is None for c in comps):
        return None
    return int(sum(comps))


def c_score_category(score: Optional[int]) -> str:
    if score is None:
        return "unknown"
    if 90 <= score <= 100:
        return "Excellent, low risk"
    if 75 <= score <= 89:
        return "Good"
    if 60 <= score <= 74:
        return "Needs improvement"
    if score < 60:
        return "Elevated risk"
    return "unknown"


def compute_c_score(df: pd.DataFrame) -> pd.Series:
    return df.apply(c_score_from_row, axis=1)
