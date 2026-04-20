# src/train_models.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("📁 Loading ML-ready data...\n")

ml_df = pd.read_parquet(DATA_PROCESSED_DIR / "f1_ml_ready_2024_2025.parquet")

print(f"Loaded {len(ml_df)} race records\n")


# ===================== PREPARE FEATURES & TARGETS =====================

feature_cols = [
    'grid', 'quali_3_race_avg', 'recent_form', 
    'vs_teammate_pos', 'track_experience', 
    'year', 'round_number'
]

X = ml_df[feature_cols].copy()
y_position = ml_df['race_pos']
y_points = ml_df['points']

# Create classification targets
y_top3 = (ml_df['race_pos'] <= 3).astype(int)
y_winner = (ml_df['race_pos'] == 1).astype(int)

print("Features used:", feature_cols)
print(f"Target - Race Position: {y_position.mean():.2f} average")
print(f"Target - Top 3 rate: {y_top3.mean():.1%}")
print(f"Target - Winner rate: {y_winner.mean():.1%}\n")


# ===================== TRAIN / TEST SPLIT =====================

X_train, X_test, y_pos_train, y_pos_test, y_points_train, y_points_test, y_top3_train, y_top3_test, y_win_train, y_win_test = train_test_split(
    X, y_position, y_points, y_top3, y_winner, test_size=0.25, random_state=42
)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}\n")


# ===================== TRAIN MODELS =====================

print("🚀 Training models...\n")

# 1. Race Position Predictor (Regression)
pos_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
pos_model.fit(X_train, y_pos_train)

# 2. Points Predictor (Regression)
points_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
points_model.fit(X_train, y_points_train)

# 3. Top 3 Probability (Classification)
top3_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
top3_model.fit(X_train, y_top3_train)

# 4. Winner Probability (Classification)
win_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
win_model.fit(X_train, y_win_train)


# ===================== EVALUATE MODELS =====================

print("📊 Model Performance on Test Set:\n")

# Position Prediction
pred_pos = pos_model.predict(X_test)
print(f"Race Position MAE: {mean_absolute_error(y_pos_test, pred_pos):.3f} places")

# Points Prediction
pred_points = points_model.predict(X_test)
print(f"Points MAE: {mean_absolute_error(y_points_test, pred_points):.3f} points")

# Top 3 Accuracy
pred_top3 = top3_model.predict(X_test)
pred_top3_prob = top3_model.predict_proba(X_test)[:, 1]
print(f"Top 3 Accuracy: {accuracy_score(y_top3_test, pred_top3):.1%}")

# Winner Accuracy
pred_win = win_model.predict(X_test)
pred_win_prob = win_model.predict_proba(X_test)[:, 1]
print(f"Winner Accuracy: {accuracy_score(y_win_test, pred_win):.1%}")


# ===================== SAVE MODELS =====================
import joblib

models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(pos_model, models_dir / "position_model.pkl")
joblib.dump(points_model, models_dir / "points_model.pkl")
joblib.dump(top3_model, models_dir / "top3_model.pkl")
joblib.dump(win_model, models_dir / "winner_model.pkl")

print(f"\n✅ All models saved to /models/ folder")
