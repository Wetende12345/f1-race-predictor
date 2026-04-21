# streamlit_app/app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
st.set_page_config(page_title="F1 Predictor", layout="wide", page_icon="🏎️")
st.title("🏎️ F1 Race Outcome Predictor 2026")
st.markdown("**Mastery Project** — Data Engineering + ML Engineering")

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# ===================== LOAD DATA & MODELS =====================
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / "f1_ml_ready_2024_2025.parquet")

@st.cache_resource
def load_models():
    models = {}
    for name in ["position_model", "points_model", "top3_model", "winner_model"]:
        models[name] = joblib.load(MODELS_DIR / f"{name}.pkl")
    return models

ml_df = load_data()
models = load_models()

# ===================== MAIN APP =====================
tab1, tab2, tab3 = st.tabs(["🔮 What-If Simulator", "📊 Model Explainability", "📋 Data Explorer"])

with tab1:
    st.header("What-If Scenario Simulator")
    st.markdown("Change inputs and see how prediction changes in real-time")

    col_a, col_b = st.columns(2)
    with col_a:
        year = st.selectbox("Year", sorted(ml_df['year'].unique()), index=1)
        race = st.selectbox("Race", sorted(ml_df[ml_df['year']==year]['event_name'].unique()))
        driver = st.selectbox("Driver", sorted(ml_df[(ml_df['year']==year) & (ml_df['event_name']==race)]['FullName'].unique()))
    
    base = ml_df[(ml_df['year']==year) & (ml_df['event_name']==race) & (ml_df['FullName']==driver)].iloc[0]

    with col_b:
        grid = st.slider("Grid Position", 1, 20, int(base['grid']))
        recent_form = st.slider("Recent Form (avg points)", 0.0, 26.0, float(base['recent_form']), 0.1)
        vs_teammate = st.slider("vs Teammate", -1.0, 1.0, float(base['vs_teammate_pos']), 0.1)
        track_exp = st.slider("Track Experience", 0, 10, int(base['track_experience']))

    if st.button("Run Prediction", type="primary"):
        input_data = pd.DataFrame([{
            'grid': grid,
            'quali_3_race_avg': base.get('quali_3_race_avg', grid),
            'recent_form': recent_form,
            'vs_teammate_pos': vs_teammate,
            'track_experience': track_exp,
            'year': year,
            'round_number': base['round_number']
        }])

        pred_pos = models['position_model'].predict(input_data)[0]
        pred_points = models['points_model'].predict(input_data)[0]
        top3_prob = models['top3_model'].predict_proba(input_data)[0][1]
        win_prob = models['winner_model'].predict_proba(input_data)[0][1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Finish", f"P{int(round(pred_pos))}")
        c2.metric("Expected Points", f"{max(0, pred_points):.1f}")
        c3.metric("Top 3 Chance", f"{top3_prob:.1%}")
        c4.metric("Win Chance", f"{win_prob:.1%}")

with tab2:
    st.header("Model Explainability (SHAP)")
    st.info("This section will show which features influence predictions the most — very important for production ML")

with tab3:
    st.header("Raw Data Explorer")
    st.dataframe(ml_df.head(20), use_container_width=True)

st.caption("Built for mastery • Automated with GitHub Actions • Tracked with MLflow")
