# src/clean_data.py

import pandas as pd
from pathlib import Path

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Create processed folder
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("📁 Loading raw data...\n")

# Load the raw files
df_2024 = pd.read_parquet(DATA_RAW_DIR / "f1_data_2024.parquet")
df_2025 = pd.read_parquet(DATA_RAW_DIR / "f1_data_2025.parquet")

df = pd.concat([df_2024, df_2025], ignore_index=True)

print(f"Original shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}\n")

# ===================== CLEANING =====================

# 1. Keep only the columns we need
columns_to_keep = [
    'year', 'event_name', 'round_number', 'session_type',
    'DriverId', 'FullName', 'TeamName',
    'Position', 'GridPosition', 'Points', 'Laps',
    'Q1', 'Q2', 'Q3', 'Time', 'Status'
]

df_clean = df[columns_to_keep].copy()

print(f"After selecting columns: {df_clean.shape}")
# 2. Fix data types
df_clean['year'] = df_clean['year'].astype(int)
df_clean['round_number'] = df_clean['round_number'].astype(int)
df_clean['Position'] = pd.to_numeric(df_clean['Position'], errors='coerce')
df_clean['GridPosition'] = pd.to_numeric(df_clean['GridPosition'], errors='coerce')
df_clean['Points'] = pd.to_numeric(df_clean['Points'], errors='coerce')

# 3. Create a unique race identifier
df_clean['race_id'] = df_clean['year'].astype(str) + "_" + df_clean['round_number'].astype(str).str.zfill(2)

print("\n✅ Basic cleaning done")
print(f"Final shape: {df_clean.shape}")
print("\nSample after cleaning:")
print(df_clean.head(3)[['year', 'event_name', 'session_type', 'DriverId', 'Position', 'Points']])

# ===================== SAVE =====================
output_file = DATA_PROCESSED_DIR / "f1_cleaned_2024_2025.parquet"
df_clean.to_parquet(output_file, index=False)
print(f"\n💾 Cleaned data saved to: {output_file}")
