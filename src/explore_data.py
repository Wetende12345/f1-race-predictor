# src/explore_data.py

import pandas as pd
from pathlib import Path

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

print("📁 Loading data...\n")

# Load both years
df_2024 = pd.read_parquet(DATA_RAW_DIR / "f1_data_2024.parquet")
df_2025 = pd.read_parquet(DATA_RAW_DIR / "f1_data_2025.parquet")

# Combine them into one DataFrame
df = pd.concat([df_2024, df_2025], ignore_index=True)

print(f"✅ Total rows loaded: {len(df):,}")
print(f"✅ Columns ({len(df.columns)}):")
print(df.columns.tolist())
print("\n" + "="*80)

# ===================== BASIC EXPLORATION =====================

print("📊 Basic Information:")
print(f"Years in data     : {sorted(df['year'].unique())}")
print(f"Number of races   : {df['event_name'].nunique()}")
print(f"Session types     : {df['session_type'].unique()}")
print(f"Number of drivers : {df['DriverId'].nunique()}\n")

print("🔍 Sample of data (first 5 rows):")
print(df.head(5)[['year', 'event_name', 'session_type', 'DriverId', 
                  'Position', 'GridPosition', 'Points', 'TeamName']])

print("\n" + "="*80)
print("🏆 Top 10 drivers by total points (2024-2025):")
points_summary = df.groupby(['DriverId', 'FullName'])['Points'].sum().reset_index()
points_summary = points_summary.sort_values('Points', ascending=False).head(10)
print(points_summary)

print("\n" + "="*80)
print("🏁 Race wins count:")
race_wins = df[(df['session_type'] == 'R') & (df['Position'] == 1)]
wins_summary = race_wins.groupby(['DriverId', 'FullName']).size().reset_index(name='Wins')
print(wins_summary.sort_values('Wins', ascending=False))
