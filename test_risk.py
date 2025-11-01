import pandas as pd
from mhf.risk_engine import compute_framingham, compute_mhf

# One sample profile (uses mmol/L for TC/HDL, as per your app)
row = pd.DataFrame([{
    "age": 55, "sex": "male", "bmi": 28, "cvd": False, "sbp": 140, "on_bp_meds": True,
    "tc": 5.5, "hdl": 1.1, "diabetes": False, "smoker": True, "met_min": 1200
}])

print("Framingham:", round(compute_framingham(row).iloc[0], 3))
print("MHF (with MET adj):", round(compute_mhf(row).iloc[0], 3))

import pandas as pd
from mhf.risk_engine import compute_framingham, compute_mhf

rows = pd.DataFrame([
    # baseline, numeric booleans
    {"age":55,"sex":"male","bmi":28,"cvd":0,"sbp":140,"on_bp_meds":1,
     "tc":5.5,"hdl":1.1,"diabetes":0,"smoker":1,"met_min":1200},
    # same but strings (should produce identical results to row 0)
    {"age":55,"sex":"Male","bmi":28,"cvd":"no","sbp":140,"on_bp_meds":"yes",
     "tc":5.5,"hdl":1.1,"diabetes":"no","smoker":"yes","met_min":"1200"},
    # non-male sex path & MET=0 (so MHF==Framingham)
    {"age":55,"sex":"male","bmi":28,"cvd":False,"sbp":140,"on_bp_meds":True,
     "tc":5.5,"hdl":1.1,"diabetes":False,"smoker":True,"met_min":0},
    # CVD override example (30% base then MET cut)
    {"age":55,"sex":"male","bmi":28,"cvd":True,"sbp":140,"on_bp_meds":True,
     "tc":5.5,"hdl":1.1,"diabetes":False,"smoker":True,"met_min":1000},
])

print(pd.DataFrame({
    "Framingham": compute_framingham(rows).round(3),
    "MHF": compute_mhf(rows).round(3)
}))
