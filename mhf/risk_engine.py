import numpy as np
import pandas as pd

def compute_risk_row(row: pd.Series) -> float:
    """
    Replace the toy equation below with your validated MHF formula.
    Required fields: age, sex, bmi, cvd, sbp, on_bp_meds, tc, hdl, diabetes, smoker, met_min
    Return: risk percentage 0-100 (float)
    """
    # --- PLACEHOLDER (illustrative only) ---
    age_pts = (row.age - 50) * 0.2
    sbp_pts = (row.sbp - 120) * (0.06 if row.on_bp_meds else 0.08)
    lip_pts = (row.tc - 5.2) * 0.5 - (row.hdl - 1.3) * 1.2
    dm_pts  = 5.0 if row.diabetes else 0.0
    smk_pts = 6.0 if row.smoker else 0.0
    cvd_pts = 8.0 if row.cvd else 0.0
    bmi_pts = max(0.0, (row.bmi - 25) * 0.3)
    fit_off = - min(10.0, (row.met_min / 150.0))  # −1 point per 150 MET-min, cap at −10

    score = 5.0 + age_pts + sbp_pts + lip_pts + dm_pts + smk_pts + cvd_pts + bmi_pts + fit_off
    risk = 1 / (1 + np.exp(-(score - 10)/3))  # map to 0..1
    return float(100 * np.clip(risk, 0, 1))

def compute_risk(df: pd.DataFrame) -> pd.Series:
    return df.apply(compute_risk_row, axis=1)

import numpy as np
import pandas as pd

MGDL_PER_MMOLL = 38.67  # mmol/L -> mg/dL

def _as_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    # numbers: 0=false, nonzero=true
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x) != 0.0
    s = str(x).strip().lower()
    return s in {"1","true","t","yes","y","on"}

def _is_male(x) -> bool:
    return str(x).strip().lower() in {"male","m","man"}

def _f(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

# ---- Framingham (CCS) refactor (fixed) ----
import numpy as np
import pandas as pd
from typing import Optional

MGDL_PER_MMOLL = 38.67  # mg/dL per mmol/L (cholesterol conversion)

# ----------------- Helpers -----------------
def _to_float(x) -> float:
    """Return float or np.nan (do NOT silently treat invalid as 0)."""
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

def _is_valid_number(x) -> bool:
    v = _to_float(x)
    return not np.isnan(v)

def _ensure_mmol_tc(value: float) -> float:
    """
    Accept either mmol/L *or* mg/dL. Heuristic: if value > 20 -> treat as mg/dL and convert.
    """
    if np.isnan(value):
        return float("nan")
    if value > 20.0:
        # likely mg/dL
        return value / MGDL_PER_MMOLL
    return value

def _round2(x: float) -> float:
    if np.isnan(x):
        return x
    return round(float(x) + 1e-12, 2)

# ----------------- Tables & rules -----------------
# Age points (men/women) using lower bounds — adjusted to start at age 0
# Ages <30 are mapped to the same points as the original 30-34 bucket.
AGE_THRESH_M = [
    (0,  0),  # <30 mapped to same as 30-34 (0 pts)
    (30, 0), (35, 2), (40, 5), (45, 6), (50, 8),
    (55,10), (60,11), (65,12), (70,14), (75,15),
]
AGE_THRESH_W = [
    (0,  0),  # <30 mapped to same as 30-34 (0 pts)
    (30, 0), (35, 2), (40, 4), (45, 5), (50, 7),
    (55, 8), (60, 9), (65,10), (70,11), (75,12),
]

def _age_points_ccs(age: float, male: bool) -> Optional[int]:
    """
    Return integer age points. Ages below 30 are accepted and mapped to the same
    points as the 30-34 bucket (i.e., 0 pts). Returns None for invalid input.
    """
    a = _to_float(age)
    if np.isnan(a) or a < 0:
        return None
    a = int(round(a))
    table = AGE_THRESH_M if male else AGE_THRESH_W
    pts = table[-1][1]
    for lo, p in reversed(table):  # find highest lo <= a
        if a >= lo:
            pts = p
            break
    return int(pts)

# HDL (mmol/L)
def _hdl_points_ccs(hdl_mmol: float) -> Optional[int]:
    v = _round2(_to_float(hdl_mmol))
    if np.isnan(v):
        return None
    if v >= 1.60:
        return -2
    if 1.30 <= v < 1.60:
        return -1
    if 1.20 <= v < 1.30:
        return 0
    if 0.90 <= v < 1.20:
        return 1
    return 2  # <0.90

# Total cholesterol (mmol/L)
# ---- Replace the old _tc_points_ccs with this sex-specific version ----
def _tc_points_ccs(tc_mmol: float, male: bool) -> Optional[int]:
    """
    Return TC points by sex per CCS table.
    Accepts tc_mmol (mmol/L). Returns None for invalid input.
    """
    v = _round2(_to_float(tc_mmol))
    if np.isnan(v):
        return None

    if male:
        # Men: <4.10:0 ; 4.10–5.19:+1 ; 5.20–6.19:+2 ; 6.20–7.19:+3 ; >=7.20:+4
        if v < 4.10:
            return 0
        if 4.10 <= v < 5.20:
            return 1
        if 5.20 <= v < 6.20:
            return 2
        if 6.20 <= v < 7.20:
            return 3
        return 4
    else:
        # Women: <4.10:0 ; 4.10–5.19:+1 ; 5.20–6.19:+3 ; 6.20–7.19:+4 ; >=7.20:+5
        if v < 4.10:
            return 0
        if 4.10 <= v < 5.20:
            return 1
        if 5.20 <= v < 6.20:
            return 3
        if 6.20 <= v < 7.20:
            return 4
        return 5

# ---- Also update the call site inside framingham_points_from_row ----
# Find where you compute tc_pts and replace that single line:
#   tc_pts   = _tc_points_ccs(tc_mmol)
# with:
    tc_pts   = _tc_points_ccs(tc_mmol, male)


# SBP points: use explicit ranges (men/women, treated/untreated)
def _sbp_points_ccs(sbp: float, treated: bool, male: bool) -> Optional[int]:
    s = _to_float(sbp)
    if np.isnan(s):
        return None
    if male:
        if treated:
            bands = [(0,119,0),(120,129,2),(130,139,3),(140,149,4),(150,159,4),(160,9999,5)]
        else:
            bands = [(0,119,-2),(120,129,0),(130,139,1),(140,149,2),(150,159,2),(160,9999,3)]
    else:
        if treated:
            bands = [(0,119,-1),(120,129,2),(130,139,3),(140,149,5),(150,159,6),(160,9999,7)]
        else:
            bands = [(0,119,-3),(120,129,0),(130,139,1),(140,149,2),(150,159,4),(160,9999,5)]
    s_ = int(round(s))
    for lo, hi, pts in bands:
        if lo <= s_ <= hi:
            return pts
    return None

# Smoking / diabetes points (helpers were missing previously)
SMOKER_PTS_M, SMOKER_PTS_W = 4, 3
DIAB_PTS_M,   DIAB_PTS_W   = 3, 4

def _smoker_points_ccs(male: bool, smoker) -> int:
    """Return smoker points per CCS: men=4, women=3 (0 if not smoker)."""
    return (SMOKER_PTS_M if male else SMOKER_PTS_W) if _to_bool(smoker) else 0

def _diabetes_points_ccs(male: bool, diabetes) -> int:
    """Return diabetes points per CCS: men=3, women=4 (0 if no diabetes)."""
    return (DIAB_PTS_M if male else DIAB_PTS_W) if _to_bool(diabetes) else 0

# Points -> 10-year CVD risk (%) per CCS chart
PTS_TO_RISK_M = {
    -3: 0.9, -2: 1.1, -1: 1.4, 0: 1.6, 1: 1.9, 2: 2.3, 3: 2.8, 4: 3.3,
     5: 3.9, 6: 4.7, 7: 5.6, 8: 6.7, 9: 7.9, 10: 9.4, 11: 11.2, 12: 13.2,
    13: 15.6, 14: 18.4, 15: 21.6, 16: 25.3, 17: 29.4
}
PTS_TO_RISK_W = {
    -3: 0.9, -2: 0.9, -1: 1.0, 0: 1.2, 1: 1.5, 2: 1.7, 3: 2.0, 4: 2.4,
     5: 2.8, 6: 3.3, 7: 3.9, 8: 4.5, 9: 5.3, 10: 6.3, 11: 7.3, 12: 8.6,
    13: 10.0, 14: 11.7, 15: 13.7, 16: 15.9, 17: 18.5, 18: 21.5, 19: 24.8, 20: 28.5
}

def _risk_from_points_ccs(total_pts: int, male: bool) -> float:
    """
    Return 10-year CVD risk (%) per CCS chart.
    Values above the highest listed point bucket are returned as 30.0 (">30%").
    Values "<1%" are represented numerically as 0.9.
    """
    if male:
        if total_pts <= -3:
            return PTS_TO_RISK_M[-3]
        if total_pts >= 18:
            return 30.0
        return float(PTS_TO_RISK_M.get(total_pts, 30.0))
    else:
        if total_pts <= -3:
            return PTS_TO_RISK_W[-3]
        if total_pts >= 21:
            return 30.0
        return float(PTS_TO_RISK_W.get(total_pts, 30.0))

# ----------------- Primary API -----------------
def framingham_points_from_row(row: pd.Series) -> Optional[int]:
    """Return integer total points or None if missing/invalid input."""
    male     = _is_male(row.get("sex"))
    age_pts  = _age_points_ccs(row.get("age"), male)
    hdl_raw  = _to_float(row.get("hdl"))
    tc_raw   = _to_float(row.get("tc"))

    # Accept mg/dL or mmol/L for TC and HDL: if >20 treat as mg/dL
    if not np.isnan(tc_raw):
        tc_mmol = _ensure_mmol_tc(tc_raw)
    else:
        tc_mmol = float("nan")
    if not np.isnan(hdl_raw):
        hdl_mmol = _ensure_mmol_tc(hdl_raw)
    else:
        hdl_mmol = float("nan")

    hdl_pts  = _hdl_points_ccs(hdl_mmol)
    tc_pts   = _tc_points_ccs(tc_mmol, male)
    sbp_pts  = _sbp_points_ccs(row.get("sbp"), _to_bool(row.get("on_bp_meds")), male)
    smoker_pts = _smoker_points_ccs(male, row.get("smoker"))
    diab_pts   = _diabetes_points_ccs(male, row.get("diabetes"))

    # If any required component is None -> return None to signal invalid
    for comp in (age_pts, hdl_pts, tc_pts, sbp_pts, smoker_pts, diab_pts):
        if comp is None:
            return None

    total = int(age_pts + hdl_pts + tc_pts + sbp_pts + smoker_pts + diab_pts)
    return total

def framingham_risk_from_row(row: pd.Series) -> float:
    """
    Return 10-year risk percentage (float). If inputs invalid -> return np.nan.
    """
    pts = framingham_points_from_row(row)
    if pts is None:
        return float("nan")
    male = _is_male(row.get("sex"))
    return _risk_from_points_ccs(pts, male)

def compute_framingham(df: pd.DataFrame) -> pd.Series:
    return df.apply(framingham_risk_from_row, axis=1)

# ----------------- Small test harness -----------------
if __name__ == "__main__":  # quick check when module run directly
    test_rows = [
        {"sex":"M","age":55,"tc":5.2,"hdl":1.3,"sbp":140,"on_bp_meds":False,"smoker":True,"diabetes":False},
        {"sex":"F","age":60,"tc":6.3,"hdl":1.0,"sbp":150,"on_bp_meds":True,"smoker":False,"diabetes":True},
        {"sex":"M","age":50,"tc":200,"hdl":50,"sbp":130,"on_bp_meds":False,"smoker":False,"diabetes":False},
    ]
    df_test = pd.DataFrame(test_rows)
    df_test["framingham_%"] = compute_framingham(df_test)
    print(df_test[["sex","age","tc","hdl","sbp","on_bp_meds","smoker","diabetes","framingham_%"]])



# ----------------- Backwards-compatible wrappers (fix NameError) -----------------
def framingham_risk_row(row: pd.Series) -> float:
    """
    Backwards-compatible wrapper: older code expects framingham_risk_row.
    Delegates to framingham_risk_from_row (refactored canonical function).
    """
    return framingham_risk_from_row(row)

def framingham_points_row_ccs(row: pd.Series) -> Optional[int]:
    """
    Backwards-compatible wrapper: older code (and tests) might expect
    framingham_points_row_ccs to return integer points or None.
    Delegates to framingham_points_from_row.
    """
    return framingham_points_from_row(row)

def framingham_debug_row(row: pd.Series) -> dict:
    """Return a breakdown of points and interpreted booleans for a single row."""
    male = _is_male(row.get("sex"))
    age_pts = _age_points_ccs(row.get("age"), male)
    tc_raw = _to_float(row.get("tc"))
    hdl_raw = _to_float(row.get("hdl"))
    tc_mmol = _ensure_mmol_tc(tc_raw) if not np.isnan(tc_raw) else float("nan")
    hdl_mmol = _ensure_mmol_tc(hdl_raw) if not np.isnan(hdl_raw) else float("nan")

    hdl_pts = _hdl_points_ccs(hdl_mmol)
    tc_pts = _tc_points_ccs(tc_mmol)
    sbp_raw = row.get("sbp")
    on_bp_raw = row.get("on_bp_meds")
    on_bp_bool = _to_bool(on_bp_raw)
    sbp_pts = _sbp_points_ccs(sbp_raw, on_bp_bool, male)
    smoker_raw = row.get("smoker")
    smoker_bool = _to_bool(smoker_raw)
    smoker_pts = _smoker_points_ccs(male, smoker_raw)
    diab_raw = row.get("diabetes")
    diab_bool = _to_bool(diab_raw)
    diab_pts = _diabetes_points_ccs(male, diab_raw)

    total = None
    comps = (age_pts, hdl_pts, tc_pts, sbp_pts, smoker_pts, diab_pts)
    if None not in comps:
        total = int(sum(comps))

    return {
        "age": row.get("age"),
        "sex": row.get("sex"),
        "age_pts": age_pts,
        "tc_raw": tc_raw,
        "tc_mmol": tc_mmol,
        "tc_pts": tc_pts,
        "hdl_raw": hdl_raw,
        "hdl_mmol": hdl_mmol,
        "hdl_pts": hdl_pts,
        "sbp_raw": sbp_raw,
        "on_bp_meds_raw": on_bp_raw,
        "on_bp_meds_bool": on_bp_bool,
        "sbp_pts": sbp_pts,
        "smoker_raw": smoker_raw,
        "smoker_bool": smoker_bool,
        "smoker_pts": smoker_pts,
        "diabetes_raw": diab_raw,
        "diabetes_bool": diab_bool,
        "diabetes_pts": diab_pts,
        "total_points": total,
        "framingham_pct": (_risk_from_points_ccs(total, male) if total is not None else float("nan"))
    }


def mhf_risk_row(row: pd.Series) -> float:
    """MHF risk = Framingham 10-yr % adjusted for activity (MET-min).
    Removed previous CVD override so we always start from Framingham %.
    """
    # Base is always the Framingham 10-year % (no CVD override)
    base = framingham_risk_row(row)

    # MET reduction (cap 30% at 1000 MET-min/wk)
    met_minutes = _f(row.get("met_min"))
    reduction = min(0.3, 0.3 * (met_minutes / 1000.0))
    adjusted = base * (1 - reduction)
    return float(np.clip(adjusted, 0.0, 100.0))

def compute_mhf(df: pd.DataFrame) -> pd.Series:
    return df.apply(mhf_risk_row, axis=1)

