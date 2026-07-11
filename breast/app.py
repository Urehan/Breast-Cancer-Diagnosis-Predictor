import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Custom CSS — modern, flat, responsive
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.hero {
    background: linear-gradient(135deg, #993556 0%, #72243E 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: white;
}
.hero p {
    font-size: 14px;
    opacity: 0.9;
    margin: 0;
    color: #FBEAF0;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E5E7E5;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    height: 100%;
}
.metric-card .label {
    font-size: 12px;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
}

.result-benign {
    background: #EAF3DE;
    border: 1px solid #97C459;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.result-malignant {
    background: #FCEBEB;
    border: 1px solid #F09595;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.result-benign h2 { color: #27500A; margin: 0 0 4px 0; font-size: 22px; }
.result-malignant h2 { color: #791F1F; margin: 0 0 4px 0; font-size: 22px; }
.result-benign p, .result-malignant p { margin: 0; font-size: 14px; color: #444; }

section[data-testid="stSidebar"] {
    background: #F6F1F3;
}

div.stButton > button {
    width: 100%;
    background: #993556;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    font-size: 15px;
    transition: background 0.2s ease;
}
div.stButton > button:hover {
    background: #72243E;
    color: white;
}

.disclaimer {
    background: #FAEEDA;
    border: 1px solid #EF9F27;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 13px;
    color: #633806;
    margin: 1rem 0;
}

@media (max-width: 768px) {
    .hero { padding: 1.5rem; }
    .hero h1 { font-size: 22px; }
    .main .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "breast_cancer_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    model_loaded = True
    load_error = None
except Exception as e:
    model = None
    model_loaded = False
    load_error = str(e)

FEATURE_COLUMNS = [
    'id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
    'shape_irregularity', 'border_complexity', 'tumor_aggressiveness',
    'radius_texture_interaction', 'radius_concavity_interaction',
    'concavity_density', 'malignancy_risk_score'
]

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🩺 Breast Cancer Diagnosis Predictor</h1>
    <p>DecisionTreeClassifier trained on tumor measurements — predicts Benign vs Malignant.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ For educational/demo purposes only. Not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        "⚠️ Model file could not be loaded. Make sure a valid `breast_cancer_model.pkl` "
        f"is placed next to `app.py`.\n\nDetails: {load_error}"
    )

# ------------------------------------------------------------------
# Sidebar — inputs
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Sample identifier")
    id_val = st.number_input("Sample ID", value=842302, step=1)

    st.markdown("### Raw tumor measurements")
    radius_mean = st.number_input("Radius (mean)", value=14.0, step=0.1, format="%.3f")
    texture_mean = st.number_input("Texture (mean)", value=19.0, step=0.1, format="%.3f")
    perimeter_mean = st.number_input("Perimeter (mean)", value=92.0, step=0.5, format="%.3f")
    area_mean = st.number_input("Area (mean)", value=655.0, step=5.0, format="%.3f")
    smoothness_mean = st.number_input("Smoothness (mean)", value=0.096, step=0.001, format="%.4f")
    compactness_mean = st.number_input("Compactness (mean)", value=0.104, step=0.001, format="%.4f")
    concavity_mean = st.number_input("Concavity (mean)", value=0.089, step=0.001, format="%.4f")
    concave_points_mean = st.number_input("Concave points (mean)", value=0.048, step=0.001, format="%.4f")

    st.markdown("### Engineered features")
    st.caption("Formulas unknown — enter values directly.")
    shape_irregularity = st.number_input("Shape irregularity", value=0.0, step=0.01, format="%.4f")
    border_complexity = st.number_input("Border complexity", value=0.0, step=0.01, format="%.4f")
    tumor_aggressiveness = st.number_input("Tumor aggressiveness", value=0.0, step=0.01, format="%.4f")
    radius_texture_interaction = st.number_input("Radius x texture interaction", value=0.0, step=0.1, format="%.4f")
    radius_concavity_interaction = st.number_input("Radius x concavity interaction", value=0.0, step=0.01, format="%.4f")
    concavity_density = st.number_input("Concavity density", value=0.0, step=0.01, format="%.4f")
    malignancy_risk_score = st.number_input("Malignancy risk score", value=0.0, step=0.01, format="%.4f")

    predict_clicked = st.button("🔮 Predict")

# ------------------------------------------------------------------
# Main area — quick summary + result
# ------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
for col, label, value in zip(
    [c1, c2, c3, c4],
    ["Radius mean", "Texture mean", "Perimeter mean", "Area mean"],
    [radius_mean, texture_mean, perimeter_mean, area_mean]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


def make_gauge(prob_malignant: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_malignant * 100, 1),
        number={'suffix': "%", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#993556"},
            'steps': [
                {'range': [0, 40], 'color': "#EAF3DE"},
                {'range': [40, 70], 'color': "#FAEEDA"},
                {'range': [70, 100], 'color': "#FCEBEB"},
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=10))
    return fig


if predict_clicked:
    row = {
        'id': id_val,
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'shape_irregularity': shape_irregularity,
        'border_complexity': border_complexity,
        'tumor_aggressiveness': tumor_aggressiveness,
        'radius_texture_interaction': radius_texture_interaction,
        'radius_concavity_interaction': radius_concavity_interaction,
        'concavity_density': concavity_density,
        'malignancy_risk_score': malignancy_risk_score,
    }
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    if model_loaded:
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        prob_m = proba[classes.index('M')] if 'M' in classes else max(proba)

        result_col, gauge_col = st.columns([1, 1])
        with result_col:
            if pred == 'M':
                st.markdown(f"""
                <div class="result-malignant">
                    <h2>⚠️ Malignant</h2>
                    <p>Confidence: {prob_m:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-benign">
                    <h2>✅ Benign</h2>
                    <p>Confidence: {(1 - prob_m):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        with gauge_col:
            st.plotly_chart(make_gauge(prob_m), use_container_width=True)

        with st.expander("See feature vector sent to the model"):
            st.dataframe(X, use_container_width=True)
    else:
        st.info("Model not loaded — showing the feature vector that would have been sent:")
        st.dataframe(X, use_container_width=True)
else:
    st.info("Fill in the tumor measurements in the sidebar and click **Predict** to see the result.")

st.markdown("---")
st.caption("Built with Streamlit • DecisionTreeClassifier • Breast cancer diagnosis dataset")
