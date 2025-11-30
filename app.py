import streamlit as st
import yaml
from yaml.loader import SafeLoader
import plotly.express as px
import pandas as pd
from pathlib import Path
from collections.abc import Mapping
import streamlit_authenticator as stauth
import re
from io import BytesIO

from mhf.generator import generate_profiles, generate_mortality_profiles
from mhf.risk_engine import compute_framingham, compute_mhf
from mhf.mortality import compute_lee, compute_c_score, c_score_category
from mhf.export import to_excel




st.set_page_config(page_title="MHF Innovation Lab Simulator", page_icon="🧪", layout="wide")

MODE_CVD = "Framingham / MHF"
MODE_MORT = "Mortality (Lee + C-score)"

# Columns expected in the uploaded file (order doesn’t matter)
REQUIRED_COLS = [
    "age","sex","bmi","cvd","sbp","on_bp_meds","tc","hdl","diabetes","smoker","met_min"
]

# Friendly renaming: handle common header variations (case/spacing doesn’t matter)
_SYNS = {
    "gender": "sex",
    "systolic_bp": "sbp",
    "systolic_blood_pressure": "sbp",
    "bp_meds": "on_bp_meds",
    "bp_medication": "on_bp_meds",
    "on_bp_medication": "on_bp_meds",
    "total_chol": "tc",
    "total_cholesterol": "tc",
    "hdl_chol": "hdl",
    "hdl_cholesterol": "hdl",
    "smoking": "smoker",
    "smoking_status": "smoker",
    "met_minutes": "met_min",
    "met_mins": "met_min",
    "body_mass_index": "bmi",
    "cvd_status": "cvd",
    "cardiovascular_disease": "cvd",
}

def _canon(s: str) -> str:
    # lower → strip → collapse non-alnum to underscores
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def normalize_and_validate(df: pd.DataFrame):
    # normalize column names
    mapping = {}
    for c in df.columns:
        k = _canon(c)
        mapping[c] = _SYNS.get(k, k)
    df = df.rename(columns=mapping)

    # check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return df, missing


# ---------- Styles / Branding ----------
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Header row: logo + title
col_logo, col_title = st.columns([0.12, 0.88])

with col_logo:
    # Use SVG by default; change to "assets/mhf_logo.png" if you have a PNG
    st.image("assets/mhf-logo.png", width=180)

with col_title:
    st.markdown("""
        <div class="mhf-header">
            <h1 class="mhf-h1">MHF Innovation Lab Simulator</h1>
            <p class="mhf-sub">Generate synthetic cohorts with customized ditrubutions to visulize and download MHF risk and Framingham Risk for 1000 - 1,000,000 synthetic profiles.</p>
        </div>
    """, unsafe_allow_html=True)




# ---------- Auth ----------

def _to_dict(obj):
    if isinstance(obj, Mapping):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def load_auth_config():
    if "auth" in st.secrets:
        return _to_dict(st.secrets["auth"])

    cfg_path = Path("credentials.yaml")
    if cfg_path.exists():
        with cfg_path.open("r") as file:
            return yaml.load(file, Loader=SafeLoader)

    st.error(
        "No authentication configuration found. "
        "Add credentials to `.streamlit/secrets.toml` or create `credentials.yaml`."
    )
    st.stop()


config = load_auth_config()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config.get('preauthorized', {}).get('emails', [])
)

# Render login in the sidebar (no tuple unpacking)
authenticator.login(
    location="sidebar",
    fields={"Form name": "Login", "Username": "Username", "Password": "Password", "Login": "Login"},
    key="auth",
)

# Read results from session_state
auth_status = st.session_state.get("authentication_status", None)

if auth_status:
    # Logged in: show the rest of the app
    name = st.session_state.get("name")
    username = st.session_state.get("username")
    st.success(f"Welcome {name}!")
elif auth_status is False:
    st.error("Invalid username or password.")
    st.stop()
else:
    st.info("Please log in using the sidebar.")
    st.stop()

# After successful login, inside the auth_status block:
authenticator.logout(
    button_name="Logout",
    location="sidebar",
    key="logout_btn",      # <-- make sure this key isn’t used anywhere else
)




# ---------- Sidebar: Parameters ----------
st.sidebar.subheader("Simulation Parameters")

mode = st.sidebar.radio(
    "Simulator mode",
    [MODE_CVD, MODE_MORT],
    index=0,
)
mode_key = "cvd" if mode == MODE_CVD else "mortality"

n = st.sidebar.number_input("Number of profiles", min_value=100, max_value=100000, value=10000, step=100)


def dist_block(label, defaults, prefix=""):
    st.write(f"**{label}**")
    colA, colB, colC, colD = st.columns(4)

    kind = colA.selectbox(
        "Type",
        ["uniform", "normal", "lognormal", "beta"],
        index=["uniform","normal","lognormal","beta"].index(defaults["kind"]),
        key=f"{prefix}{label}_kind"
    )
    cfg = {"kind": kind}

    if kind == "normal":
        cfg["mean"] = colB.number_input("Mean", value=defaults.get("mean", defaults.get("mu", 40.0)), key=f"{prefix}{label}_mean")
        cfg["sd"]   = colC.number_input("Standard deviation", value=defaults.get("sd", 10.0), key=f"{prefix}{label}_sd")
    elif kind == "lognormal":
        cfg["mu"]    = colB.number_input("μ (log mean)", value=defaults.get("mu", 2.5), key=f"{prefix}{label}_mu")
        cfg["sigma"] = colC.number_input("σ (log standard deviation)", value=defaults.get("sigma", 0.6), key=f"{prefix}{label}_sigma")
    elif kind == "beta":
        cfg["alpha"] = colB.number_input("Alpha (α)", value=defaults.get("alpha", 2.0), key=f"{prefix}{label}_alpha")
        cfg["beta"]  = colC.number_input("Beta (β)",  value=defaults.get("beta", 5.0), key=f"{prefix}{label}_beta")

    st.markdown("")  # spacing
    return cfg


distributions = {}
params = {}
session_key = "profiles"
params_key = "params"

if mode == MODE_CVD:
    session_key = "profiles_cvd"
    params_key = "params_cvd"

    st.sidebar.markdown("---")
    st.sidebar.caption("Ranges")

    age_min, age_max = st.sidebar.slider("Age range", 18, 100, (18, 60))
    bmi_min, bmi_max = st.sidebar.slider("BMI range", 10.0, 60.0, (18.0, 38.0))
    sbp_min, sbp_max = st.sidebar.slider("Systolic BP (mmHg)", 80, 220, (120, 150))
    tc_min, tc_max   = st.sidebar.slider("Total Cholestrol range (mmol/L)", 2.0, 10.0, (3.5, 7.5))
    hdl_min, hdl_max = st.sidebar.slider("HDL range (mmol/L)", 0.5, 3.0, (0.9, 2.0))
    met_min_min, met_min_max = st.sidebar.slider("MET-minutes / week Range", 0, 8000, (0, 3000))

    st.sidebar.markdown("---")
    st.sidebar.caption("Categoricals (% yes / split)")

    sex_m = st.sidebar.slider(" % Of males in synthetic cohort)", 0, 100, 75)
    sex_f = st.sidebar.slider(" % of females in synthetic cohort", 0, 100 - sex_m, 25)
    cvd_yes = st.sidebar.slider(" % of people that have CVD in synthetic cohort ", 0, 100, 10)
    on_bp_meds_yes = st.sidebar.slider(" % of people that are on On BP meds in synthetic cohort", 0, 100, 30)
    diabetes_yes = st.sidebar.slider(" % of people that are diabetic in synthetic cohort", 0, 100, 25)
    smoker_yes   = st.sidebar.slider(" % of people that are smokers in synthetic cohort", 0, 100, 15)

    st.sidebar.markdown("---")
    skew_enabled = st.sidebar.checkbox("Advanced distributions (skew)", value=True, key="cvd_skew")

    # ---------- Sidebar: Upload cohort (optional) ----------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Would you like to upload your own synthetic cohort? Use the upload function below or download our template and populate it with your dataset to get started")

    # Downloadable template
    _template_df = pd.DataFrame(columns=REQUIRED_COLS)
    _buf = BytesIO()
    with pd.ExcelWriter(_buf, engine="openpyxl") as w:
        _template_df.to_excel(w, index=False, sheet_name="Template")

    st.sidebar.download_button(
        "Download upload template (.xlsx)",
        data=_buf.getvalue(),
        file_name="mhf_upload_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    uploaded = st.sidebar.file_uploader(
        "Upload Excel (.xlsx) or CSV with required columns",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
    )

    # Handle uploaded file
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            up = pd.read_csv(uploaded)
        else:
            up = pd.read_excel(uploaded)

        up, missing = normalize_and_validate(up)
        if missing:
            st.sidebar.error(f"Missing required columns: {missing}. Download the template and match headers.")
        else:
            for c in ["age", "bmi", "sbp", "tc", "hdl", "met_min"]:
                if c in up.columns:
                    up[c] = pd.to_numeric(up[c], errors="coerce")

            up["framingham_risk"] = compute_framingham(up).round(2)
            up["mhf_risk"]        = compute_mhf(up).round(2)
            up["risk_diff"]       = (up["mhf_risk"] - up["framingham_risk"]).round(2)

            st.session_state[session_key] = up
            st.session_state[params_key] = {"source": "upload", "n": len(up)}
            st.sidebar.success(f"Uploaded {len(up):,} rows. Using uploaded data.")

    if skew_enabled:
        st.markdown('<div class="mhf-dist">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="mhf-dist-help">
            <strong>Distribution types:</strong>
            <em>uniform</em> = equal chance across min–max;
            <em>normal</em> = bell curve (Mean &amp; Standard deviation);
            <em>lognormal</em> = right-skewed, defined by μ (log mean) &amp; σ (log standard deviation);
            <em>beta</em> = bounded 0–1, re-scaled to your chosen range (α, β control skew).
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Distribution settings", expanded=False):
            st.markdown('<div class="mhf-dist-panel">', unsafe_allow_html=True)

            distributions["age"]     = dist_block("age",     {"kind":"normal","mean":42.0,"sd":10.0}, prefix="cvd_")
            distributions["bmi"]     = dist_block("bmi",     {"kind":"lognormal","mu":3.2,"sigma":0.25}, prefix="cvd_")
            distributions["sbp"]     = dist_block("sbp",     {"kind":"normal","mean":132.0,"sd":12.0}, prefix="cvd_")
            distributions["tc"]      = dist_block("tc",      {"kind":"normal","mean":5.2,"sd":0.8}, prefix="cvd_")
            distributions["hdl"]     = dist_block("hdl",     {"kind":"normal","mean":1.3,"sd":0.3}, prefix="cvd_")
            distributions["met_min"] = dist_block("met_min", {"kind":"beta","alpha":1.3,"beta":4.0}, prefix="cvd_")

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <style id="mhf-dist-dark">
            /* Expander shell */
            .mhf-dist [data-testid="stExpander"] > details{
              background:#000 !important;
              border:1px solid rgba(255,255,255,.14) !important;
              border-radius:12px !important;
              overflow:hidden;
            }
            .mhf-dist [data-testid="stExpander"] > details > summary{
              background:#000 !important;
              color:#fff !important;
              padding:12px 14px !important;
              border-bottom:1px solid rgba(255,255,255,.06) !important;
            }
            .mhf-dist [data-testid="stExpander"] > details > div{
              background:#000 !important;
              padding:12px 14px 10px !important;
            }

            /* All text in this section = white */
            .mhf-dist [data-testid="stExpander"] *{
              color:#fff !important;
            }

            /* Select (BaseWeb) */
            .mhf-dist div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
              background:#111 !important;
              border:1px solid rgba(255,255,255,.25) !important;
              border-radius:10px !important;
              box-shadow:none !important;
            }
            .mhf-dist div[data-testid="stSelectbox"] [data-baseweb="select"] *{
              color:#fff !important;
              fill:#fff !important;
            }

            /* Number input */
            .mhf-dist div[data-testid="stNumberInput"] input{
              background:#111 !important;
              color:#fff !important;
              border:1px solid rgba(255,255,255,.25) !important;
              border-radius:10px !important;
              box-shadow:none !important;
            }
            .mhf-dist div[data-testid="stNumberInput"] button{
              background:#111 !important;
              color:#fff !important;
              border:1px solid rgba(255,255,255,.25) !important;
              border-radius:8px !important;
            }
            .mhf-dist div[data-testid="stNumberInput"] svg{
              color:#fff !important; fill:#fff !important;
            }

            /* Focus ring */
            .mhf-dist div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
            .mhf-dist div[data-testid="stNumberInput"] input:focus{
              outline:2px solid #9acefe !important;
              outline-offset:0 !important;
              box-shadow:none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        distributions = {k: {"kind": "uniform"} for k in ["age","bmi","sbp","tc","hdl","met_min"]}

    params = {
        "n": n,
        "ranges": {
            "age_min": age_min, "age_max": age_max,
            "bmi_min": bmi_min, "bmi_max": bmi_max,
            "sbp_min": sbp_min, "sbp_max": sbp_max,
            "tc_min": tc_min,   "tc_max": tc_max,
            "hdl_min": hdl_min, "hdl_max": hdl_max,
            "met_min_min": met_min_min, "met_min_max": met_min_max
        },
        "distributions": distributions,
        "categoricals": {
            "sex_male": sex_m, "sex_female": sex_f,
            "cvd_yes": cvd_yes,
            "on_bp_meds_yes": on_bp_meds_yes,
            "diabetes_yes": diabetes_yes,
            "smoker_yes": smoker_yes
        }
    }

else:
    session_key = "profiles_mortality"
    params_key = "params_mortality"

    st.sidebar.markdown("---")
    st.sidebar.caption("Ranges")
    age_min, age_max = st.sidebar.slider("Age range", 18, 100, (60, 90))
    bmi_min, bmi_max = st.sidebar.slider("BMI range", 14.0, 60.0, (18.0, 35.0))
    whtr_min, whtr_max = st.sidebar.slider("Waist-to-height ratio (WHtR)", 0.35, 0.80, (0.45, 0.65))
    rhr_min, rhr_max = st.sidebar.slider("Resting heart rate (bpm)", 40, 130, (55, 95))
    drinks_min, drinks_max = st.sidebar.slider("Alcohol (drinks/week)", 0, 40, (0, 14))
    sleep_min, sleep_max = st.sidebar.slider("Sleep (hours/night)", 3.0, 12.0, (6.0, 9.0))
    whtr_coupling = st.sidebar.slider("WHtR coupling to BMI", 0.0, 0.01, 0.002, step=0.0005, help="Higher = WHtR moves more with BMI.")

    st.sidebar.caption("Sex split (%)")
    sex_m = st.sidebar.slider("Male", 0, 100, 50, key="mort_sex_m")
    sex_f = st.sidebar.slider("Female", 0, 100 - sex_m, 50, key="mort_sex_f")

    st.sidebar.markdown("---")
    st.sidebar.caption("Smoking status mix (%)")
    smk_never = st.sidebar.number_input("Never smoker", 0, 100, 50)
    smk_former_gt1 = st.sidebar.number_input("Former (>1 yr)", 0, 100, 20)
    smk_former_lt1 = st.sidebar.number_input("Former (<1 yr)", 0, 100, 5)
    smk_current_le10 = st.sidebar.number_input("Current ≤10 cig/day", 0, 100, 15)
    smk_current_gt10 = st.sidebar.number_input("Current >10 cig/day", 0, 100, 10)

    st.sidebar.caption("Conditions (% yes)")
    diabetes_yes = st.sidebar.slider("Diabetes", 0, 100, 20)
    non_skin_cancer_yes = st.sidebar.slider("Non-skin cancer", 0, 100, 5)
    copd_yes = st.sidebar.slider("Chronic lung disease (COPD/bronchitis)", 0, 100, 8)
    heart_failure_yes = st.sidebar.slider("Heart failure", 0, 100, 5)

    st.sidebar.caption("Functional difficulties (% yes)")
    difficulty_bathing_yes = st.sidebar.slider("Difficulty bathing/showering", 0, 100, 8)
    difficulty_money_yes = st.sidebar.slider("Difficulty managing money", 0, 100, 10)
    difficulty_walking_yes = st.sidebar.slider("Difficulty walking several blocks", 0, 100, 12)

    st.sidebar.caption("Self-rated health mix (%)")
    srh_excellent = st.sidebar.number_input("Excellent", 0, 100, 20)
    srh_very_good = st.sidebar.number_input("Very good", 0, 100, 30)
    srh_good = st.sidebar.number_input("Good", 0, 100, 30)
    srh_fair = st.sidebar.number_input("Fair", 0, 100, 15)
    srh_poor = st.sidebar.number_input("Poor", 0, 100, 5)

    st.sidebar.markdown("---")
    skew_enabled = st.sidebar.checkbox("Advanced distributions (skew)", value=True, key="mort_skew")
    st.sidebar.info("Upload currently supports Framingham/MHF mode. Mortality mode uses generated cohorts.")

    if skew_enabled:
        st.markdown('<div class="mhf-dist">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="mhf-dist-help">
            <strong>Distribution types:</strong>
            <em>uniform</em> = equal chance across min–max;
            <em>normal</em> = bell curve (Mean &amp; Standard deviation);
            <em>lognormal</em> = right-skewed, defined by μ (log mean) &amp; σ (log standard deviation);
            <em>beta</em> = bounded 0–1, re-scaled to your chosen range (α, β control skew).
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Distribution settings (mortality)", expanded=False):
            st.markdown('<div class="mhf-dist-panel">', unsafe_allow_html=True)
            distributions["age"] = dist_block("age", {"kind": "normal", "mean": 70.0, "sd": 8.0}, prefix="mort_")
            distributions["bmi"] = dist_block("bmi", {"kind": "lognormal", "mu": 3.1, "sigma": 0.22}, prefix="mort_")
            distributions["whtr"] = dist_block("whtr", {"kind": "normal", "mean": 0.55, "sd": 0.06}, prefix="mort_")
            distributions["resting_hr"] = dist_block("resting_hr", {"kind": "normal", "mean": 70.0, "sd": 10.0}, prefix="mort_")
            distributions["drinks_per_week"] = dist_block("drinks_per_week", {"kind": "beta", "alpha": 2.0, "beta": 4.0}, prefix="mort_")
            distributions["sleep_hours"] = dist_block("sleep_hours", {"kind": "normal", "mean": 7.5, "sd": 1.0}, prefix="mort_")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <style id="mhf-dist-dark">
            .mhf-dist [data-testid="stExpander"] > details{
              background:#000 !important;
              border:1px solid rgba(255,255,255,.14) !important;
              border-radius:12px !important;
              overflow:hidden;
            }
            .mhf-dist [data-testid="stExpander"] > details > summary{
              background:#000 !important;
              color:#fff !important;
              padding:12px 14px !important;
              border-bottom:1px solid rgba(255,255,255,.06) !important;
            }
            .mhf-dist [data-testid="stExpander"] > details > div{
              background:#000 !important;
              padding:12px 14px 10px !important;
            }
            .mhf-dist [data-testid="stExpander"] *{
              color:#fff !important;
            }
            .mhf-dist div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
              background:#111 !important;
              border:1px solid rgba(255,255,255,.25) !important;
              border-radius:10px !important;
              box-shadow:none !important;
            }
            .mhf-dist div[data-testid="stSelectbox"] [data-baseweb="select"] *{
              color:#fff !important;
              fill:#fff !important;
            }
            .mhf-dist div[data-testid="stNumberInput"] input{
              background:#111 !important;
              color:#fff !important;
              border:1px solid rgba(255,255,255,.25) !important;
              border-radius:10px !important;
              box-shadow:none !important;
            }
            .mhf-dist div[data-testid="stNumberInput"] button{
              background:#111 !important;
              color:#fff !important;
              border:1px solid rgba(255,255,255,.25) !important;
              border-radius:8px !important;
            }
            .mhf-dist div[data-testid="stNumberInput"] svg{
              color:#fff !important; fill:#fff !important;
            }
            .mhf-dist div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
            .mhf-dist div[data-testid="stNumberInput"] input:focus{
              outline:2px solid #9acefe !important;
              outline-offset:0 !important;
              box-shadow:none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        distributions = {k: {"kind": "uniform"} for k in ["age", "bmi", "whtr", "resting_hr", "drinks_per_week", "sleep_hours"]}

    params = {
        "n": n,
        "ranges": {
            "age_min": age_min, "age_max": age_max,
            "bmi_min": bmi_min, "bmi_max": bmi_max,
            "whtr_min": whtr_min, "whtr_max": whtr_max,
            "resting_hr_min": rhr_min, "resting_hr_max": rhr_max,
            "drinks_min": drinks_min, "drinks_max": drinks_max,
            "sleep_min": sleep_min, "sleep_max": sleep_max,
        },
        "distributions": distributions,
        "categoricals": {
            "sex_male": sex_m,
            "sex_female": sex_f,
            "smoking": {
                "never": smk_never,
                "former_gt1": smk_former_gt1,
                "former_lt1": smk_former_lt1,
                "current_le10": smk_current_le10,
                "current_gt10": smk_current_gt10,
            },
            "diabetes_yes": diabetes_yes,
            "non_skin_cancer_yes": non_skin_cancer_yes,
            "copd_yes": copd_yes,
            "heart_failure_yes": heart_failure_yes,
            "difficulty_bathing_yes": difficulty_bathing_yes,
            "difficulty_managing_money_yes": difficulty_money_yes,
            "difficulty_walking_yes": difficulty_walking_yes,
        },
        "self_rated_health": {
            "excellent": srh_excellent,
            "very_good": srh_very_good,
            "good": srh_good,
            "fair": srh_fair,
            "poor": srh_poor,
        },
        "whtr_bmi_coupling": whtr_coupling,
    }

st.markdown(
    """
    <style>
    /* Expander shell */
    [data-testid="stExpander"] > details{
      background:#000 !important;
      border:1px solid rgba(255,255,255,.14) !important;
      border-radius:12px !important;
      overflow:hidden;
    }
    [data-testid="stExpander"] > details > summary{
      background:#000 !important;
      color:#fff !important;
      padding:12px 14px !important;
      border-bottom:1px solid rgba(255,255,255,.06) !important;
    }
    [data-testid="stExpander"] > details > div{
      background:#000 !important;
      padding:12px 14px 10px !important;
    }

    /* All text inside expander = white */
    [data-testid="stExpander"] *{
      color:#fff !important;
    }

    /* Select (BaseWeb) */
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
      background:#111 !important;
      border:1px solid rgba(255,255,255,.25) !important;
      border-radius:10px !important;
      box-shadow:none !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] *{
      color:#fff !important;
      fill:#fff !important;
    }

    /* Number input */
    div[data-testid="stNumberInput"] input{
      background:#111 !important;
      color:#fff !important;
      border:1px solid rgba(255,255,255,.25) !important;
      border-radius:10px !important;
      box-shadow:none !important;
    }
    div[data-testid="stNumberInput"] button{
      background:#111 !important;
      color:#fff !important;
      border:1px solid rgba(255,255,255,.25) !important;
      border-radius:8px !important;
    }
    div[data-testid="stNumberInput"] svg{
      color:#fff !important; fill:#fff !important;
    }

    /* Focus ring */
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
    div[data-testid="stNumberInput"] input:focus{
      outline:2px solid #9acefe !important;
      outline-offset:0 !important;
      box-shadow:none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Generate ----------
st.markdown("### 1) Generate cohort")
generate = st.button(
    "Generate Profiles" if mode == MODE_CVD else "Generate Mortality Profiles",
    type="primary",
    key=f"gen_btn_{mode_key}"
)

if session_key not in st.session_state:
    st.session_state[session_key] = None
if params_key not in st.session_state:
    st.session_state[params_key] = None

if generate:
    if mode == MODE_CVD:
        df = generate_profiles(params)
        df["framingham_risk"] = compute_framingham(df).round(2)
        df["mhf_risk"] = compute_mhf(df).round(2)
        df["risk_diff"] = (df["mhf_risk"] - df["framingham_risk"]).round(2)
    else:
        df = generate_mortality_profiles(params)
        lee_pts, lee_pct = compute_lee(df)
        df["lee_points"] = lee_pts
        df["lee_4yr_mortality_pct"] = lee_pct
        df["c_score"] = compute_c_score(df)
        df["c_score_category"] = df["c_score"].apply(c_score_category)

    st.session_state[session_key] = df.copy()
    st.session_state[params_key] = params

df = st.session_state.get(session_key, None)
params_used = st.session_state.get(params_key, None)

# ---------- Preview & Summary ----------
st.markdown("### 2) Preview & Summary")
with st.container():
    st.markdown('<div class="mhf-card">', unsafe_allow_html=True)
    if df is None:
        st.info("Click **Generate Profiles** to create a synthetic cohort.")
    else:
        st.write(f"**Rows:** {len(df):,}")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("**Summary stats**")
        st.dataframe(df.describe(include='all').T, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Distributions ----------
st.markdown("### 3) Distributions")
if df is not None and not df.empty:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if mode == MODE_CVD:
        preferred_cols = [
            "age","bmi","sbp","tc","hdl","met_min","framingham_risk","mhf_risk","risk_diff"
        ]
    else:
        preferred_cols = [
            "age","bmi","whtr","resting_hr","drinks_per_week","sleep_hours","lee_points","lee_4yr_mortality_pct","c_score"
        ]

    cols_to_plot = [c for c in preferred_cols if c in numeric_cols]
    if not cols_to_plot:
        cols_to_plot = numeric_cols

    grid = st.columns(3)
    for i, col in enumerate(cols_to_plot):
        with grid[i % 3]:
            fig = px.histogram(df, x=col, nbins=40, title=col, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Custom plot ----------
st.markdown("### 4) Custom Plot")
if df is not None:
    st.markdown('<div class="mhf-plot">', unsafe_allow_html=True)

    all_cols = list(df.columns)

    hdr_x, hdr_y, hdr_c = st.columns(3)
    hdr_x.markdown('<span class="mhf-label">X</span>', unsafe_allow_html=True)
    hdr_y.markdown('<span class="mhf-label">Y</span>', unsafe_allow_html=True)
    hdr_c.markdown('<span class="mhf-label">Color by (optional)</span>', unsafe_allow_html=True)

    colx, coly, colh = st.columns(3)
    default_x_idx = all_cols.index("sbp") if mode == MODE_CVD and "sbp" in all_cols else 0
    default_y_idx = all_cols.index("age") if "age" in all_cols else 0
    x = colx.selectbox(
        "", all_cols,
        index=default_x_idx,
        label_visibility="collapsed",
        key=f"plot_x_{mode_key}",
    )
    y = coly.selectbox(
        "", all_cols,
        index=default_y_idx,
        label_visibility="collapsed",
        key=f"plot_y_{mode_key}",
    )
    color_by = colh.selectbox(
        "", ["(none)"] + all_cols,
        index=0,
        label_visibility="collapsed",
        key=f"plot_color_{mode_key}",
    )

    st.markdown('<span class="mhf-label">Chart type</span>', unsafe_allow_html=True)
    chart_type = st.selectbox(
        "", ["Scatter", "Histogram", "Box"],
        label_visibility="collapsed",
        key=f"plot_type_{mode_key}",
    )

    if chart_type == "Scatter":
        fig = px.scatter(df, x=x, y=y, color=None if color_by == "(none)" else color_by, opacity=0.6)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x, color=None if color_by == "(none)" else color_by, nbins=40)
    else:
        fig = px.box(df, x=None if color_by == "(none)" else color_by, y=y, points="outliers")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <style>
      .mhf-plot .mhf-label{
        display:inline-block;
        margin: 4px 0 6px;
        font-weight:600;
        color:#000 !important;
      }
    </style>
    """, unsafe_allow_html=True)


# ---------- Download Excel ----------
st.markdown("### 5) Download")
if df is not None:
    excel_bytes = to_excel(df, params_used or {})
    out_name = "mhf_simulation.xlsx" if mode == MODE_CVD else "mortality_simulation.xlsx"
    st.download_button(
        label="Download Excel (.xlsx)",
        data=excel_bytes,
        file_name=out_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
