# src/create_features_advanced.py

import pandas as pd
from pathlib import Path

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("📁 Loading featured data...\n")

df = pd.read_parquet(DATA_PROCESSED_DIR / "f1_features_2024_2025.parquet")

print(f"Loaded shape: {df.shape}\n")

# ===================== ADVANCED FEATURE ENGINEERING =====================

print("🔧 Creating Advanced Features...\n")

# 1. Teammate Comparison (very strong feature)
df = df.sort_values(['year', 'round_number', 'session_type', 'TeamName'])

def teammate_comparison(group):
    if len(group) < 2:
        group['vs_teammate_pos'] = None
        return group
    
    # For race sessions only
    races = group[group['session_type'] == 'R'].copy()
    if len(races) == 2:
        races = races.sort_values('race_pos')
        races['vs_teammate_pos'] = [1, -1]   # Better than teammate = +1, worse = -1
    else:
        races['vs_teammate_pos'] = None
    return races

# Apply teammate comparison
team_df = df.groupby(['year', 'round_number', 'TeamName'], group_keys=False).apply(teammate_comparison)
df = df.merge(team_df[['year', 'round_number', 'DriverId', 'vs_teammate_pos']], 
              on=['year', 'round_number', 'DriverId'], how='left')

print("✅ Teammate comparison feature added")

# 2. Qualifying Strength (average quali position last 3 races)
df = df.sort_values(['DriverId', 'year', 'round_number'])

df['quali_3_race_avg'] = df.groupby('DriverId')['qualifying_pos'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

print("✅ Qualifying strength (3-race avg) added")

# 3. Track Experience (number of previous races at this track)
df['track_experience'] = df.groupby(['DriverId', 'event_name']).cumcount()

print("✅ Track experience feature added")

# ===================== SAVE =====================
output_file = DATA_PROCESSED_DIR / "f1_features_advanced_2024_2025.parquet"
df.to_parquet(output_file, index=False)

print(f"\n💾 Advanced features saved to: {output_file}")
print(f"Final shape: {df.shape}")
print("\nNew columns added:", ['vs_teammate_pos', 'quali_3_race_avg', 'track_experience'])
print("\nSample of new features:")
print(df[df['session_type']=='R'][['FullName', 'event_name', 'vs_teammate_pos', 
                                   'quali_3_race_avg', 'track_experience']].tail(10))


