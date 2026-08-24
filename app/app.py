"""
Student Mental Health Check-In — Streamlit app
================================================
Run locally with:
    streamlit run app.py

Requires model_artifact.joblib in the same folder (produced by
mental_health_risk_ml.py).
"""

import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# ---------------------------------------------------------------------------
# Page config + design tokens
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Check-In",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-contrast color palette
INK = "#1A1C2E"          # Darker ink for maximum contrast
BG = "#F8F9F6"           # Clean, soft background
CARD = "#FFFFFF"         # Solid white card background
SAGE = "#4E7362"         # Darker sage green for accessibility
SAGE_LIGHT = "#88B09A"
CLAY = "#C47B5A"         
CLAY_DARK = "#9E4B27"
BORDER = "#D1D5DB"       # Clean, visible border lines
MUTED = "#4A4D58"        # Dark gray for accessible body/secondary text

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@600&display=swap');

/* Base font rules */
html, body, .stApp {{
    background-color: {BG} !important;
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    color: {INK};
}}

/* General typography */
p, span, label, li, div {{
    color: {INK};
    font-size: 1.02rem;
    line-height: 1.6;
}}

h1, h2, h3, h4 {{
    font-family: 'Fraunces', serif !important;
    color: {INK} !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}}

h4 {{
    font-size: 1.25rem !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}}

/* Widget labels */
[data-testid="stWidgetLabel"] p {{
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: {INK} !important;
}}

/* -------------------------------------------------------------------------
   Explicit Fixes for Streamlit Selectboxes & BaseWeb Dropdown Menus
   ------------------------------------------------------------------------- */
div[data-baseweb="select"] > div {{
    background-color: #FFFFFF !important;
    color: {INK} !important;
    border-color: {BORDER} !important;
}}

div[data-baseweb="select"] * {{
    color: {INK} !important;
}}

/* Fix Streamlit popup container portals */
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
div[role="listbox"] {{
    background-color: #FFFFFF !important;
}}

/* Fix dropdown menu items and text */
[data-baseweb="popover"] li,
div[role="option"] {{
    background-color: #FFFFFF !important;
    color: {INK} !important;
}}

[data-baseweb="popover"] li * {{
    color: {INK} !important;
}}

/* Hover and selection states */
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] li[aria-selected="true"],
div[role="option"]:hover {{
    background-color: #EFEFEA !important;
    color: {INK} !important;
}}

/* Slider ticks & values */
[data-testid="stTickBar"] * ,
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] {{
    color: {MUTED} !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}}

/* Hero section */
.hero-title {{
    font-family: 'Fraunces', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: {INK};
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}}
.hero-sub {{
    font-family: 'Inter', sans-serif;
    color: {MUTED};
    font-size: 1.15rem;
    max-width: 720px;
    line-height: 1.6;
}}

/* Card containers */
.card {{
    background: {CARD};
    border: 1.5px solid {BORDER};
    border-radius: 16px;
    padding: 1.8rem 2rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}}

.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SAGE};
    font-weight: 700;
    margin-bottom: 0.4rem;
}}

.badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.badge-calm {{ background: rgba(78,115,98,0.15); color: {SAGE} !important; }}
.badge-attn {{ background: rgba(158,75,39,0.15); color: {CLAY_DARK} !important; }}

.metric-number {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 2.8rem;
    color: {INK};
}}
.metric-label {{
    font-family: 'Inter', sans-serif;
    color: {MUTED};
    font-size: 0.98rem;
    font-weight: 500;
}}

.disclaimer {{
    font-size: 0.95rem;
    color: {INK};
    background: rgba(78,115,98,0.1);
    border-left: 4px solid {SAGE};
    padding: 0.9rem 1.1rem;
    border-radius: 8px;
    line-height: 1.6;
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background-color: #EFEFEA;
    border-right: 1.5px solid {BORDER};
}}

/* Primary Submit Button */
.stButton>button {{
    background-color: {INK} !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}}
.stButton>button * {{
    color: #FFFFFF !important;
}}
.stButton>button:hover {{
    background-color: {SAGE} !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model artifact
# ---------------------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model_artifact.joblib"


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(
        f"Model file not found at: {MODEL_PATH}"
    )
    st.stop()

pipeline = artifact["pipeline"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]
FREQ_MAP = artifact["freq_map"]
options = artifact["options"]
fi = pd.DataFrame(artifact["feature_importance"])
train_df = artifact["training_dataframe"]
ds_summary = artifact["dataset_summary"]


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------
def draw_gauge(probability: float):
    fig, ax = plt.subplots(figsize=(4.5, 2.7), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    n = 100
    for i in range(n):
        t0 = 180 - i * 180 / n
        t1 = 180 - (i + 1) * 180 / n
        frac = i / n
        c1 = np.array([0x4E, 0x73, 0x62]) / 255
        c2 = np.array([0xC4, 0x7B, 0x5A]) / 255
        color = c1 + (c2 - c1) * frac
        ax.add_patch(Wedge((0, 0), 1.0, t1, t0, width=0.28, facecolor=color, edgecolor="none"))

    ax.add_patch(Circle((0, 0), 0.68, facecolor=BG, edgecolor="none", zorder=3))

    angle = np.radians(180 - probability * 180)
    x, y = 0.6 * np.cos(angle), 0.6 * np.sin(angle)
    ax.plot([0, x], [0, y], color=INK, linewidth=3.0, solid_capstyle="round", zorder=4)
    ax.add_patch(Circle((0, 0), 0.05, facecolor=INK, zorder=5))

    ax.text(0, -0.18, f"{probability*100:.0f}%", ha="center", va="center",
             fontsize=26, fontweight="bold", family="monospace", color=INK, zorder=6)
    ax.text(0, -0.42, "elevated-risk likelihood", ha="center", va="center",
             fontsize=10.5, fontweight="semibold", color=MUTED, zorder=6)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.55, 1.1)
    ax.axis("off")
    return fig


# ---------------------------------------------------------------------------
# Sidebar — about
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='eyebrow'>ABOUT THIS TOOL</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size:0.98rem; line-height:1.6; color:#1A1C2E; margin-top:0.4rem;'>
        This is a self check-in built on survey data from higher/senior-secondary
        students. It looks for patterns linked to academic stress, sleep, and
        social wellbeing — the same signals a brief screening questionnaire
        would use.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='disclaimer'>
        <b>Not a diagnosis.</b> This is a prototype trained on a small,
        dataset ({ds_summary['n_rows']} responses). It cannot
        replace a conversation with a counsellor, doctor, or someone you trust.
        If you're struggling, please reach out to a mental health professional
        or a crisis line in your area.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div class='hero-title'>How are you carrying things this term? 🌿</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>Answer a few questions about your workload, sleep, and "
    "support system. This gives you a quick read on patterns linked to student "
    "burnout and low mood — not a verdict, just a signal worth paying attention to.</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

tab_checkin, tab_insights = st.tabs(["🧭  Take the Check-In", "📊  Cohort Insights"])

# ---------------------------------------------------------------------------
# TAB 1 — Check-in form
# ---------------------------------------------------------------------------
with tab_checkin:
    col_form, col_result = st.columns([1.2, 1], gap="large")

    with col_form:
        st.markdown("#### Tell us about your week")

        c1, c2 = st.columns(2)
        with c1:
            age_group = st.selectbox("Age group", options["Q2. What is your age group?"])
            gender = st.selectbox("Gender", options["Q3. What is your gender?"])
            education = st.selectbox("Education level", options["Q4. What is your current level of education?"])
            stream = st.selectbox("Stream / programme area", options["Q6. What is your academic stream/programme area?"])
        with c2:
            study_hours = st.selectbox("Study hours / day", options["Q7. On average, how many hours do you spend studying or working on academic activities per day?"])
            sleep_hours = st.selectbox("Sleep hours / night", options["Q11. How many hours do you usually sleep per night?"])
            exercise = st.selectbox("Exercise frequency", options["How often do you participate in physical exercise, sports, or other physical activities?"])
            screen_time = st.selectbox("Non-study screen time / day", options["Approximately how much screen time do you have per day outside academic/study requirements?"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Rate the following (1 = low, 5 = high)")

        r1, r2, r3 = st.columns(3)
        with r1:
            academic_pressure = st.slider("Academic pressure", 1, 5, 3)
            academic_satisfaction = st.slider("Satisfaction with performance", 1, 5, 3)
        with r2:
            sleep_quality = st.slider("Sleep quality", 1, 5, 3)
            social_support = st.slider("Satisfaction with social support", 1, 5, 3)
        with r3:
            financial_stress = st.slider("Financial stress", 1, 5, 3)
            balance = st.slider("Work-life balance ability", 1, 5, 3)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### How often, lately...")
        freq_options = list(FREQ_MAP.keys())
        f1, f2 = st.columns(2)
        with f1:
            exam_stress_freq = st.select_slider("...do deadlines/exams stress you out?", freq_options, value="Sometimes")
        with f2:
            isolation_freq = st.select_slider("...do you feel isolated or disconnected?", freq_options, value="Rarely")

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Check in →", use_container_width=False)

    with col_result:
        if run:
            row = {
                "Q8. How would you rate the academic pressure you currently experience?": academic_pressure,
                "Q9. How satisfied are you with your academic performance? ": academic_satisfaction,
                "How would you rate your sleep quality?": sleep_quality,
                "Q15. How satisfied are you with the social support you receive from friends, family, or people you trust?": social_support,
                "Q17. How much financial stress do you currently experience?": financial_stress,
                "Q18. How would you rate your ability to balance academic responsibilities with your personal life?": balance,
                "Q10. How frequently do you feel stressed because of examinations, assignments, deadlines, or academic expectations?_num": FREQ_MAP[exam_stress_freq],
                "Q16. How often do you feel socially isolated or disconnected from others? _num": FREQ_MAP[isolation_freq],
                "Q2. What is your age group?": age_group,
                "Q3. What is your gender?": gender,
                "Q4. What is your current level of education?": education,
                "Q6. What is your academic stream/programme area?": stream,
                "Q7. On average, how many hours do you spend studying or working on academic activities per day?": study_hours,
                "Q11. How many hours do you usually sleep per night?": sleep_hours,
                "How often do you participate in physical exercise, sports, or other physical activities?": exercise,
                "Approximately how much screen time do you have per day outside academic/study requirements?": screen_time,
            }
            X_input = pd.DataFrame([row])[numeric_features + categorical_features]
            proba = pipeline.predict_proba(X_input)[0, 1]
            pred = int(proba >= 0.5)

            fig = draw_gauge(proba)
            st.pyplot(fig, use_container_width=True)

            if pred == 1:
                st.markdown("<span class='badge badge-attn'>WORTH PAYING ATTENTION TO</span>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='margin-top:1rem; line-height:1.65; font-size:1.05rem; color:{INK};'>"
                    "Your answers show a pattern — high stress load, low sleep quality, "
                    "or reduced social support — that's often linked to burnout risk in "
                    "this dataset. Consider talking to someone you trust, a campus "
                    "counsellor, or a doctor about how you've been feeling."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<span class='badge badge-calm'>STEADY FOR NOW</span>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='margin-top:1rem; line-height:1.65; font-size:1.05rem; color:{INK};'>"
                    "Your answers don't show the stress/sleep/isolation pattern most "
                    "linked to burnout risk in this dataset. Worth checking in with "
                    "yourself again if things change."
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div style='text-align:center; padding: 3rem 1rem; color:{MUTED}; font-size:1.05rem;'>"
                "Fill in the form and press <b>Check in →</b> to see your result here."
                "</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# TAB 2 — Cohort insights
# ---------------------------------------------------------------------------
with tab_insights:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>SAMPLE SIZE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-number'>{ds_summary['n_rows']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>respondents in training data</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>ELEVATED-RISK RATE</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-number'>{ds_summary['high_risk_rate']*100:.0f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>of respondents flagged high risk</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        best_auc = max(m["test_auc"] for m in artifact["test_metrics"].values())
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>MODEL TEST ROC-AUC</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-number'>{best_auc:.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Random Forest, held-out test set</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### What drives the model's predictions")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        top = fi.head(8).iloc[::-1]
        labels = [f.split("?")[0].split(".", 1)[-1].strip()[:38] for f in top["feature"]]
        ax.barh(labels, top["importance"], color=SAGE, height=0.55)
        ax.set_xlabel("Relative importance", fontsize=11, color=INK, fontweight="bold")
        ax.tick_params(labelsize=10, colors=INK)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(BORDER)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Well-being across the cohort")
        wb_col = "Q23. How would you describe your current overall mental well-being?"
        order = ["Very Poor", "Poor", "Fair", "Good", "Very Good"]
        counts = train_df[wb_col].value_counts().reindex(order).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")
        colors = [CLAY_DARK, CLAY, "#D9CBA0", SAGE_LIGHT, SAGE]
        ax2.bar(order, counts.values, color=colors, width=0.6)
        ax2.set_ylabel("Respondents", fontsize=11, color=INK, fontweight="bold")
        ax2.tick_params(labelsize=10, colors=INK, rotation=15)
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax2.spines[spine].set_color(BORDER)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='disclaimer'>
        This dataset is statistically modeled on
        200 real survey responses, used to demonstrate the pipeline end-to-end.
        Treat every number on this page as a prototype signal, not a validated
        clinical or population-level finding. For adequate training for a regular ML Model,
        more than 1000+ samples are needed. Treat it like a prototype.
        </div>
        """,
        unsafe_allow_html=True,
    )
