import numpy as np
import pandas as pd
from .sampling import (
    sample_uniform, sample_normal, sample_lognormal, sample_beta_scaled,
    sample_binary, sample_categorical
)

def generate_profiles(params: dict) -> pd.DataFrame:
    """
    params contains:
      n, ranges (age,bmi,sbp,tc,hdl,met_min), toggles (skew_enabled),
      distributions config per field, categorical percentages, etc.
    """
    n = int(params["n"])
    rng = params["ranges"]
    cfg = params["distributions"]
    cats = params["categoricals"]

    # --- Numeric fields
    def draw_numeric(field, low, high):
        dcfg = cfg[field]
        kind = dcfg["kind"]
        if kind == "uniform":
            return sample_uniform(low, high, n)
        elif kind == "normal":
            return sample_normal(low, high, dcfg["mean"], dcfg["sd"], n)
        elif kind == "lognormal":
            return sample_lognormal(low, high, dcfg["mu"], dcfg["sigma"], n)
        elif kind == "beta":
            return sample_beta_scaled(low, high, dcfg["alpha"], dcfg["beta"], n)
        else:
            return sample_uniform(low, high, n)  # safe default

    age     = draw_numeric("age",     rng["age_min"],     rng["age_max"])
    bmi     = draw_numeric("bmi",     rng["bmi_min"],     rng["bmi_max"])
    sbp     = draw_numeric("sbp",     rng["sbp_min"],     rng["sbp_max"])
    tc      = draw_numeric("tc",      rng["tc_min"],      rng["tc_max"])
    hdl     = draw_numeric("hdl",     rng["hdl_min"],     rng["hdl_max"])
    met_min = draw_numeric("met_min", rng["met_min_min"], rng["met_min_max"])

    # --- Categoricals
    # Sex: proportions [male, female, other]
    sex_idx = sample_categorical([
        cats["sex_male"]/100.0,
        cats["sex_female"]/100.0,
        max(0.0, 1.0 - (cats["sex_male"]+cats["sex_female"])/100.0)
    ], n)
    sex = np.array(["male","female","other"])[sex_idx]

    def bern(pct): return sample_binary(pct/100.0, n)

    cvd       = bern(cats["cvd_yes"])
    on_bp_meds= bern(cats["on_bp_meds_yes"])
    diabetes  = bern(cats["diabetes_yes"])
    smoker    = bern(cats["smoker_yes"])

    # Optional SBP shift for those on meds (if enabled)
    sbp_shift = params.get("sbp_meds_shift", 0.0)
    sbp = sbp - (on_bp_meds.astype(float) * sbp_shift)

    df = pd.DataFrame({
        "age": age.round(0).astype(int),
        "sex": sex,
        "bmi": bmi,
        "cvd": cvd.astype(bool),
        "sbp": sbp,
        "on_bp_meds": on_bp_meds.astype(bool),
        "tc": tc,
        "hdl": hdl,
        "diabetes": diabetes.astype(bool),
        "smoker": smoker.astype(bool),
        "met_min": met_min
    })
    return df


def generate_mortality_profiles(params: dict) -> pd.DataFrame:
    """Generate synthetic profiles for mortality (Lee + C-score)."""
    n = int(params["n"])
    rng = params["ranges"]
    cfg = params["distributions"]
    cats = params["categoricals"]
    srh_cfg = params["self_rated_health"]
    whtr_coupling = params.get("whtr_bmi_coupling", 0.0)

    def draw_numeric(field, low, high):
        dcfg = cfg[field]
        kind = dcfg["kind"]
        if kind == "uniform":
            return sample_uniform(low, high, n)
        elif kind == "normal":
            return sample_normal(low, high, dcfg["mean"], dcfg["sd"], n)
        elif kind == "lognormal":
            return sample_lognormal(low, high, dcfg["mu"], dcfg["sigma"], n)
        elif kind == "beta":
            return sample_beta_scaled(low, high, dcfg["alpha"], dcfg["beta"], n)
        else:
            return sample_uniform(low, high, n)

    age         = draw_numeric("age",         rng["age_min"],         rng["age_max"])
    bmi         = draw_numeric("bmi",         rng["bmi_min"],         rng["bmi_max"])
    whtr_base   = draw_numeric("whtr",        rng["whtr_min"],       rng["whtr_max"])
    resting_hr  = draw_numeric("resting_hr",  rng["resting_hr_min"],  rng["resting_hr_max"])
    drinks_week = draw_numeric("drinks_per_week", rng["drinks_min"], rng["drinks_max"])
    sleep_hours = draw_numeric("sleep_hours", rng["sleep_min"],       rng["sleep_max"])

    # Correlate WHtR with BMI with a gentle coupling factor
    whtr = whtr_base + whtr_coupling * (bmi - bmi.mean())
    whtr = np.clip(whtr, rng["whtr_min"], rng["whtr_max"])

    # Sex distribution
    sex_idx = sample_categorical([
        cats["sex_male"] / 100.0,
        cats["sex_female"] / 100.0,
        max(0.0, 1.0 - (cats["sex_male"] + cats["sex_female"]) / 100.0),
    ], n)
    sex = np.array(["male", "female", "other"])[sex_idx]

    # Smoking categories (normalized automatically)
    smk_cfg = cats["smoking"]
    smk_probs = np.array([
        smk_cfg["never"],
        smk_cfg["former_gt1"],
        smk_cfg["former_lt1"],
        smk_cfg["current_le10"],
        smk_cfg["current_gt10"],
    ], dtype=float)
    smk_probs = smk_probs / smk_probs.sum() if smk_probs.sum() > 0 else np.ones(5) / 5
    smk_idx = sample_categorical(smk_probs, n)
    smk_labels = np.array(["never", "former_gt1", "former_lt1", "current_le10", "current_gt10"])
    smoking_status = smk_labels[smk_idx]
    current_smoker = np.isin(smoking_status, ["current_le10", "current_gt10"])

    def bern(pct):
        return sample_binary(pct / 100.0, n)

    diabetes     = bern(cats["diabetes_yes"]).astype(bool)
    non_skin_ca  = bern(cats["non_skin_cancer_yes"]).astype(bool)
    copd         = bern(cats["copd_yes"]).astype(bool)
    heart_fail   = bern(cats["heart_failure_yes"]).astype(bool)
    diff_bathing = bern(cats["difficulty_bathing_yes"]).astype(bool)
    diff_money   = bern(cats["difficulty_managing_money_yes"]).astype(bool)
    diff_walk    = bern(cats["difficulty_walking_yes"]).astype(bool)

    srh_probs = np.array([
        srh_cfg["excellent"],
        srh_cfg["very_good"],
        srh_cfg["good"],
        srh_cfg["fair"],
        srh_cfg["poor"],
    ], dtype=float)
    srh_probs = srh_probs / srh_probs.sum() if srh_probs.sum() > 0 else np.ones(5) / 5
    srh_idx = sample_categorical(srh_probs, n)
    srh_labels = np.array(["Excellent", "Very good", "Good", "Fair", "Poor"])
    srh = srh_labels[srh_idx]

    df = pd.DataFrame({
        "age": age.round(0).astype(int),
        "sex": sex,
        "bmi": bmi,
        "smoking_status": smoking_status,
        "current_smoker": current_smoker.astype(bool),
        "diabetes": diabetes,
        "non_skin_cancer": non_skin_ca,
        "copd": copd,
        "heart_failure": heart_fail,
        "difficulty_bathing": diff_bathing,
        "difficulty_managing_money": diff_money,
        "difficulty_walking": diff_walk,
        "self_rated_health": srh,
        "whtr": whtr,
        "resting_hr": resting_hr.round(0).astype(int),
        "drinks_per_week": np.round(drinks_week, 1),
        "sleep_hours": np.round(sleep_hours, 1),
    })
    return df
