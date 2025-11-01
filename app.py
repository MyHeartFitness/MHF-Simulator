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
import pandas as pd

from mhf.generator import generate_profiles
from mhf.risk_engine import compute_framingham, compute_mhf
from mhf.export import to_excel




st.set_page_config(page_title="MHF Innovation Lab Simulator", page_icon="🧪", layout="wide")

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

n = st.sidebar.number_input("Number of profiles", min_value=100, max_value=100000, value=10000, step=100)



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
skew_enabled = st.sidebar.checkbox("Advanced distributions (skew)", value=True)

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
    # Read file
    if uploaded.name.lower().endswith(".csv"):
        up = pd.read_csv(uploaded)
    else:
        up = pd.read_excel(uploaded)

    # Normalize headers and validate
    up, missing = normalize_and_validate(up)
    if missing:
        st.sidebar.error(f"Missing required columns: {missing}. Download the template and match headers.")
    else:
        # Coerce numeric fields (others handled by risk engine)
        for c in ["age", "bmi", "sbp", "tc", "hdl", "met_min"]:
            if c in up.columns:
                up[c] = pd.to_numeric(up[c], errors="coerce")

        # Compute risks on uploaded data
        up["framingham_risk"] = compute_framingham(up).round(2)
        up["mhf_risk"]        = compute_mhf(up).round(2)
        up["risk_diff"]       = (up["mhf_risk"] - up["framingham_risk"]).round(2)

        # Use uploaded cohort as the active dataset
        st.session_state["profiles"] = up
        st.sidebar.success(f"Uploaded {len(up):,} rows. Using uploaded data.")


def dist_block(label, defaults):
    st.write(f"**{label}**")
    colA, colB, colC, colD = st.columns(4)

    kind = colA.selectbox(
        "Type",
        ["uniform", "normal", "lognormal", "beta"],
        index=["uniform","normal","lognormal","beta"].index(defaults["kind"]),
        key=f"{label}_kind"
    )
    cfg = {"kind": kind}

    if kind == "normal":
        cfg["mean"] = colB.number_input("Mean", value=defaults.get("mean", defaults.get("mu", 40.0)), key=f"{label}_mean")
        cfg["sd"]   = colC.number_input("Standard deviation", value=defaults.get("sd", 10.0), key=f"{label}_sd")
    elif kind == "lognormal":
        cfg["mu"]    = colB.number_input("μ (log mean)", value=defaults.get("mu", 2.5), key=f"{label}_mu")
        cfg["sigma"] = colC.number_input("σ (log standard deviation)", value=defaults.get("sigma", 0.6), key=f"{label}_sigma")
    elif kind == "beta":
        cfg["alpha"] = colB.number_input("Alpha (α)", value=defaults.get("alpha", 2.0), key=f"{label}_alpha")
        cfg["beta"]  = colC.number_input("Beta (β)",  value=defaults.get("beta", 5.0), key=f"{label}_beta")

    st.markdown("")  # spacing
    return cfg


distributions = {}
if skew_enabled:
    # Wrapper we will target with CSS
    st.markdown('<div class="mhf-dist">', unsafe_allow_html=True)

    # One-liner help
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
        # A panel div around the body
        st.markdown('<div class="mhf-dist-panel">', unsafe_allow_html=True)

        distributions["age"]     = dist_block("age",     {"kind":"normal","mean":42.0,"sd":10.0})
        distributions["bmi"]     = dist_block("bmi",     {"kind":"lognormal","mu":3.2,"sigma":0.25})
        distributions["sbp"]     = dist_block("sbp",     {"kind":"normal","mean":132.0,"sd":12.0})
        distributions["tc"]      = dist_block("tc",      {"kind":"normal","mean":5.2,"sd":0.8})
        distributions["hdl"]     = dist_block("hdl",     {"kind":"normal","mean":1.3,"sd":0.3})
        distributions["met_min"] = dist_block("met_min", {"kind":"beta","alpha":1.3,"beta":4.0})

        st.markdown('</div>', unsafe_allow_html=True)  # end .mhf-dist-panel

    # Inject CSS AFTER the widgets render so it wins
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

    # Close the wrapper INSIDE the if-block
    st.markdown('</div>', unsafe_allow_html=True)

else:
    distributions = {k: {"kind": "uniform"} for k in ["age","bmi","sbp","tc","hdl","met_min"]}

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

st.markdown("### 1) Generate cohort")
generate = st.button("Generate Profiles", type="primary")

if "profiles" not in st.session_state:
    st.session_state["profiles"] = None

if generate:
    df = generate_profiles(params)

    # NEW: add both risks and their difference (percentage points)
    df["framingham_risk"] = compute_framingham(df).round(2)
    df["mhf_risk"] = compute_mhf(df).round(2)
    df["risk_diff"] = (df["mhf_risk"] - df["framingham_risk"]).round(2)

    st.session_state["profiles"] = df.copy()


df = st.session_state.get("profiles", None)

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
    # Build the list of numeric columns dynamically
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # (Optional) keep a friendly order & include only the ones you care about
    cols_to_plot = [c for c in [
        "age","bmi","sbp","tc","hdl","met_min","framingham_risk","mhf_risk","risk_diff"
    ] if c in numeric_cols]

    grid = st.columns(3)
    for i, col in enumerate(cols_to_plot):
        with grid[i % 3]:
            fig = px.histogram(df, x=col, nbins=40, title=col, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Custom plot ----------
# ---------- Custom plot ----------
st.markdown("### 4) Custom Plot")
if df is not None:
    # wrapper for scoped styles
    st.markdown('<div class="mhf-plot">', unsafe_allow_html=True)

    all_cols = list(df.columns)

    # Row of custom labels (black, forced)
    hdr_x, hdr_y, hdr_c = st.columns(3)
    hdr_x.markdown('<span class="mhf-label">X</span>', unsafe_allow_html=True)
    hdr_y.markdown('<span class="mhf-label">Y</span>', unsafe_allow_html=True)
    hdr_c.markdown('<span class="mhf-label">Color by (optional)</span>', unsafe_allow_html=True)

    # The widgets with their labels hidden
    colx, coly, colh = st.columns(3)
    x = colx.selectbox(
        "", all_cols,
        index=all_cols.index("sbp") if "sbp" in all_cols else 0,
        label_visibility="collapsed",
        key="plot_x",
    )
    y = coly.selectbox(
        "", all_cols,
        index=all_cols.index("age") if "age" in all_cols else 0,
        label_visibility="collapsed",
        key="plot_y",
    )
    color_by = colh.selectbox(
        "", ["(none)"] + all_cols,
        index=0,
        label_visibility="collapsed",
        key="plot_color",
    )

    # Chart type label + widget
    st.markdown('<span class="mhf-label">Chart type</span>', unsafe_allow_html=True)
    chart_type = st.selectbox(
        "", ["Scatter", "Histogram", "Box"],
        label_visibility="collapsed",
        key="plot_type",
    )

    # Render chart
    if chart_type == "Scatter":
        fig = px.scatter(df, x=x, y=y, color=None if color_by == "(none)" else color_by, opacity=0.6)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x, color=None if color_by == "(none)" else color_by, nbins=40)
    else:
        fig = px.box(df, x=None if color_by == "(none)" else color_by, y=y, points="outliers")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end .mhf-plot

    # Scoped CSS: make our custom labels black and slightly bold
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
    excel_bytes = to_excel(df, params)
    st.download_button(
        label="Download Excel (.xlsx)",
        data=excel_bytes,
        file_name="mhf_simulation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
