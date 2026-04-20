# streamlit_app/app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="F1 Race Predictor", layout="wide")
st.title("🏎️ F1 Race Outcome Predictor 2026")
st.markdown("**Trained on 2024-2025 data • Built step-by-step for mastery**")

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# Load data and models
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / "f1_ml_ready_2024_2025.parquet")

@st.cache_resource
def load_models():
    pos_model = joblib.load(MODELS_DIR / "position_model.pkl")
    points_model = joblib.load(MODELS_DIR / "points_model.pkl")
    top3_model = joblib.load(MODELS_DIR / "top3_model.pkl")
    win_model = joblib.load(MODELS_DIR / "winner_model.pkl")
    return pos_model, points_model, top3_model, win_model

ml_df = load_data()
pos_model, points_model, top3_model, win_model = load_models()

# Sidebar
st.sidebar.header("🔧 Prediction Settings")

selected_year = st.sidebar.selectbox("Year", sorted(ml_df['year'].unique()), index=1)
races = sorted(ml_df[ml_df['year'] == selected_year]['event_name'].unique())
selected_race = st.sidebar.selectbox("Race", races)

drivers = sorted(ml_df[(ml_df['year'] == selected_year) & 
                      (ml_df['event_name'] == selected_race)]['FullName'].unique())
selected_driver = st.sidebar.selectbox("Driver", drivers)

if st.sidebar.button("🚀 Make Prediction", type="primary"):
    
    driver_row = ml_df[(ml_df['year'] == selected_year) & 
                       (ml_df['event_name'] == selected_race) & 
                       (ml_df['FullName'] == selected_driver)].iloc[0]
    
    features = pd.DataFrame([{
        'grid': driver_row['grid'],
        'quali_3_race_avg': driver_row.get('quali_3_race_avg', driver_row['grid']),
        'recent_form': driver_row['recent_form'],
        'vs_teammate_pos': driver_row['vs_teammate_pos'],
        'track_experience': driver_row['track_experience'],
        'year': driver_row['year'],
        'round_number': driver_row['round_number']
    }])
    
    pred_pos = pos_model.predict(features)[0]
    pred_points = points_model.predict(features)[0]
    pred_top3_prob = top3_model.predict_proba(features)[0][1]
    pred_win_prob = win_model.predict_proba(features)[0][1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Position", f"P{int(round(pred_pos))}")
    with col2:
        st.metric("Predicted Points", f"{max(0, pred_points):.1f}")
    with col3:
        st.metric("Top 3 Probability", f"{pred_top3_prob:.1%}")
    with col4:
        st.metric("Win Probability", f"{pred_win_prob:.1%}")
    
    st.success(f"**Prediction for {selected_driver} at {selected_race} ({selected_year})**")

# Preview
st.subheader("📊 Recent Data Preview")
st.dataframe(ml_df[['FullName', 'event_name', 'grid', 'recent_form', 'race_pos']].head(10), 
             use_container_width=True)

st.caption("F1 Mastery Project - Built step by step")
