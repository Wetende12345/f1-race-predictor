# src/prepare_ml_data.py

import pandas as pd
from pathlib import Path

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("📁 Loading advanced features data...\n")

df = pd.read_parquet(DATA_PROCESSED_DIR / "f1_features_advanced_2024_2025.parquet")

print(f"Loaded shape: {df.shape}\n")


# ===================== CREATE ML DATASET =====================

print("🔍 Creating ML-ready dataset...\n")

# Work only with Race sessions
ml_df = df[df['session_type'] == 'R'].copy()

print(f"Race sessions found: {len(ml_df)} rows")

# === CRITICAL FIX: Re-create position columns from original data ===
ml_df['race_pos'] = pd.to_numeric(ml_df['Position'], errors='coerce')
ml_df['grid'] = pd.to_numeric(ml_df['GridPosition'], errors='coerce')

# Use existing features where available
ml_df['qualifying_pos'] = pd.to_numeric(ml_df['qualifying_pos'], errors='coerce')

print(f"After converting positions - race_pos NaNs: {ml_df['race_pos'].isna().sum()}")
print(f"grid NaNs: {ml_df['grid'].isna().sum()}")
print(f"qualifying_pos NaNs: {ml_df['qualifying_pos'].isna().sum()}")


# ===================== FEATURES & TARGETS =====================

feature_cols = [
    'grid', 'qualifying_pos', 'quali_3_race_avg',
    'recent_form', 'vs_teammate_pos', 'track_experience', 
    'finished_race', 'year', 'round_number'
]

target_cols = ['race_pos', 'points', 'position_gain']

info_cols = ['DriverId', 'FullName', 'event_name', 'TeamName']

ml_df = ml_df[info_cols + feature_cols + target_cols].copy()


# ===================== HANDLE MISSING VALUES =====================

print("\n🧹 Handling missing values...")

ml_df['vs_teammate_pos'] = ml_df['vs_teammate_pos'].fillna(0)
ml_df['quali_3_race_avg'] = ml_df['quali_3_race_avg'].fillna(ml_df['qualifying_pos'])
ml_df['recent_form'] = ml_df['recent_form'].fillna(ml_df['points'].mean())
ml_df['track_experience'] = ml_df['track_experience'].fillna(0)
ml_df['position_gain'] = ml_df['position_gain'].fillna(0)

# Drop rows only if we can't use them for training
ml_df = ml_df.dropna(subset=['race_pos', 'grid'])

print(f"\n✅ Final shape after cleaning: {ml_df.shape}")


# ===================== SAVE =====================
output_file = DATA_PROCESSED_DIR / "f1_ml_ready_2024_2025.parquet"
ml_df.to_parquet(output_file, index=False)

print(f"\n💾 Final ML dataset saved to: {output_file}")
print(f"Final shape: {ml_df.shape}")

print("\nPreview of final data:")
print(ml_df[['FullName', 'event_name', 'grid', 'qualifying_pos', 
             'recent_form', 'vs_teammate_pos', 'race_pos', 'points']].head(10))

print("\nMissing values summary:")
print(ml_df.isnull().sum())
