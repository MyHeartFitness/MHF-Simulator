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
