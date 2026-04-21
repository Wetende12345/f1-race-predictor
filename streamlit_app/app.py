# streamlit_app/app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="F1 Predictor", layout="wide", page_icon="🏎️")
st.title("🏎️ F1 Race Outcome Predictor 2026")
st.markdown("**Mastery Project** — Explainable AI + What-If Analysis")

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

ml_df = pd.read_parquet(DATA_DIR / "f1_ml_ready_2024_2025.parquet")

@st.cache_resource
def load_models():
    return {
        "position": joblib.load(MODELS_DIR / "position_model.pkl"),
        "points": joblib.load(MODELS_DIR / "points_model.pkl"),
        "top3": joblib.load(MODELS_DIR / "top3_model.pkl"),
        "winner": joblib.load(MODELS_DIR / "winner_model.pkl")
    }

models = load_models()

tab1, tab2 = st.tabs(["🔮 What-If Simulator", "🔍 Model Explainability (SHAP)"])

with tab1:
    st.header("What-If Simulator")
    col_a, col_b = st.columns(2)
    with col_a:
        year = st.selectbox("Year", sorted(ml_df['year'].unique()), index=1)
        race = st.selectbox("Race", sorted(ml_df[ml_df['year']==year]['event_name'].unique()))
        driver = st.selectbox("Driver", sorted(ml_df[(ml_df['year']==year) & (ml_df['event_name']==race)]['FullName'].unique()))
    
    base = ml_df[(ml_df['year']==year) & (ml_df['event_name']==race) & (ml_df['FullName']==driver)].iloc[0]

    with col_b:
        grid = st.slider("Grid Position", 1, 20, int(base['grid']))
        recent_form = st.slider("Recent Form", 0.0, 26.0, float(base['recent_form']), 0.1)
        vs_teammate = st.slider("vs Teammate", -1.0, 1.0, float(base['vs_teammate_pos']), 0.1)
        track_exp = st.slider("Track Experience", 0, 10, int(base['track_experience']))

    if st.button("🚀 Run Prediction", type="primary"):
        input_df = pd.DataFrame([{
            'grid': grid,
            'quali_3_race_avg': base.get('quali_3_race_avg', grid),
            'recent_form': recent_form,
            'vs_teammate_pos': vs_teammate,
            'track_experience': track_exp,
            'year': year,
            'round_number': base['round_number']
        }])

        pred_pos = models['position'].predict(input_df)[0]
        pred_points = models['points'].predict(input_df)[0]
        top3_p = models['top3'].predict_proba(input_df)[0][1]
        win_p = models['winner'].predict_proba(input_df)[0][1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Position", f"P{int(round(pred_pos))}")
        c2.metric("Predicted Points", f"{max(0, pred_points):.1f}")
        c3.metric("Top 3 Probability", f"{top3_p:.1%}")
        c4.metric("Win Probability", f"{win_p:.1%}")

with tab2:
    st.header("Model Explainability (SHAP)")
    st.markdown("Which features influence the position prediction the most?")
    
    if st.button("Show Global Feature Importance"):
        explainer = shap.TreeExplainer(models['position'])
        sample = ml_df.sample(100, random_state=42)
        shap_values = explainer.shap_values(sample[ ['grid', 'quali_3_race_avg', 'recent_form', 'vs_teammate_pos', 'track_experience', 'year', 'round_number'] ])
        
        fig = plt.figure()
        shap.summary_plot(shap_values, sample[['grid', 'quali_3_race_avg', 'recent_form', 'vs_teammate_pos', 'track_experience', 'year', 'round_number']], plot_type="bar", show=False)
        st.pyplot(fig)

st.caption("Built for mastery • GitHub Actions automated • MLflow tracked")
