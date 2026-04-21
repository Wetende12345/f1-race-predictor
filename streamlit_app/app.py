# streamlit_app/app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

st.set_page_config(page_title="F1 Race Predictor", layout="wide")
st.title("🏎️ F1 Race Outcome Predictor 2026")
st.markdown("**Trained on 2024-2025 data • Built step-by-step for mastery**")

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR.mkdir(exist_ok=True)

# Load or train models
@st.cache_resource
def load_or_train_models():
    model_files = ["position_model.pkl", "points_model.pkl", "top3_model.pkl", "winner_model.pkl"]
    
    # Check if models exist
    if all((MODELS_DIR / f).exists() for f in model_files):
        st.success("✅ Loaded pre-trained models")
        return [joblib.load(MODELS_DIR / f) for f in model_files]
    
    st.warning("⚠️ Training models on first run (this may take 10-30 seconds)...")
    
    # Load data
    df = pd.read_parquet(DATA_DIR / "f1_ml_ready_2024_2025.parquet")
    
    feature_cols = ['grid', 'quali_3_race_avg', 'recent_form', 'vs_teammate_pos', 
                   'track_experience', 'year', 'round_number']
    
    X = df[feature_cols].copy()
    y_pos = df['race_pos']
    y_points = df['points']
    y_top3 = (df['race_pos'] <= 3).astype(int)
    y_win = (df['race_pos'] == 1).astype(int)
    
    X_train, _, y_pos_train, _, y_points_train, _, y_top3_train, _, y_win_train, _ = train_test_split(
        X, y_pos, y_points, y_top3, y_win, test_size=0.2, random_state=42
    )
    
    # Train models
    pos_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    points_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    top3_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    win_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    
    pos_model.fit(X_train, y_pos_train)
    points_model.fit(X_train, y_points_train)
    top3_model.fit(X_train, y_top3_train)
    win_model.fit(X_train, y_win_train)
    
    # Save models
    joblib.dump(pos_model, MODELS_DIR / "position_model.pkl")
    joblib.dump(points_model, MODELS_DIR / "points_model.pkl")
    joblib.dump(top3_model, MODELS_DIR / "top3_model.pkl")
    joblib.dump(win_model, MODELS_DIR / "winner_model.pkl")
    
    st.success("✅ Models trained and saved!")
    return pos_model, points_model, top3_model, win_model

# Load data and models
ml_df = pd.read_parquet(DATA_DIR / "f1_ml_ready_2024_2025.parquet")
pos_model, points_model, top3_model, win_model = load_or_train_models()

# Rest of the app (same as before)
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

st.subheader("📊 Recent Data Preview")
st.dataframe(ml_df[['FullName', 'event_name', 'grid', 'recent_form', 'race_pos']].head(10), 
             use_container_width=True)

st.caption("F1 Mastery Project - Built step by step")
