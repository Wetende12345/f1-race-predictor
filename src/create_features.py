# src/create_features.py

import pandas as pd
from pathlib import Path

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("📁 Loading cleaned data...\n")

df = pd.read_parquet(DATA_PROCESSED_DIR / "f1_cleaned_2024_2025.parquet")

print(f"Loaded shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ===================== FEATURE ENGINEERING =====================

print("🔧 Starting Feature Engineering...\n")

# 1. Convert key columns to numeric (safety step)
df['position'] = pd.to_numeric(df['Position'], errors='coerce')
df['grid'] = pd.to_numeric(df['GridPosition'], errors='coerce')
df['points'] = pd.to_numeric(df['Points'], errors='coerce')

print("✅ Step 1: Basic numeric conversion done")

# 2. Split position into Qualifying and Race
df['qualifying_pos'] = df['position'].where(df['session_type'] == 'Q')
df['race_pos'] = df['position'].where(df['session_type'] == 'R')

# 3. Did the driver finish the race?
df['finished_race'] = (df['session_type'] == 'R') & (df['position'].notna())

# 4. Position gain/loss (overtaking performance)
df['position_gain'] = df['grid'] - df['race_pos']

print("✅ Step 2-4: Qualifying/Race split, finished flag, and position gain created")
print("\nSample of new features:")
print(df[['session_type', 'position', 'grid', 'position_gain', 'finished_race']].head(8))

# 5. Driver Recent Form (Rolling Average Points)
# First sort the data properly by time
df = df.sort_values(['DriverId', 'year', 'round_number', 'session_type'])

def calculate_form(group):
    """Calculate rolling average points for races only"""
    # Take only race sessions for this driver
    races = group[group['session_type'] == 'R'].copy()
    # Rolling mean of last 5 races
    races['recent_form'] = races['points'].rolling(window=5, min_periods=1).mean()
    return races

# Apply the function to each driver
form_df = df.groupby('DriverId', group_keys=False).apply(calculate_form)

# Merge the recent_form back to main dataframe
df = df.merge(
    form_df[['year', 'round_number', 'DriverId', 'recent_form']], 
    on=['year', 'round_number', 'DriverId'], 
    how='left'
)

print("\n✅ Step 5: Driver recent form (last 5 races) added")
print("\nRecent Form Sample (Top drivers in latest races):")
latest_form = df[df['session_type'] == 'R'].sort_values(['year', 'round_number']).groupby('DriverId').tail(1)
print(latest_form[['FullName', 'recent_form', 'points']].sort_values('recent_form', ascending=False).head(10))

# ===================== SAVE =====================
output_file = DATA_PROCESSED_DIR / "f1_features_2024_2025.parquet"
df.to_parquet(output_file, index=False)

print(f"\n💾 Features saved successfully to: {output_file}")
print(f"Final shape: {df.shape}")
print(f"Total columns now: {len(df.columns)}")


