# src/ingest_data.py

import fastf1
import pandas as pd
from pathlib import Path
import warnings
import time   # NEW: for adding delays

warnings.filterwarnings("ignore")

# ===================== CONFIGURATION =====================
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
CACHE_DIR = BASE_DIR / "data" / "cache"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

print("📁 Folders ready\n")


def download_season(year: int, sleep_seconds=1.5):
    """Download data for one year with rate limit protection"""
    print(f"\n🚀 Starting download for year {year}...")

    try:
        schedule = fastf1.get_event_schedule(year)
        print(f"Found {len(schedule)} events in {year}")
        
        all_sessions = []
        
        for idx, event in schedule.iterrows():
            event_name = event['EventName']
            round_number = event['RoundNumber']
            
            print(f"   📍 {event_name} (Round {round_number})")
            
            for session_type in ['Q', 'R']:
                try:
                    session = fastf1.get_session(year, round_number, session_type)
                    session.load()
                    
                    results = session.results.copy()
                    
                    results['year'] = year
                    results['event_name'] = event_name
                    results['round_number'] = round_number
                    results['session_type'] = session_type
                    
                    all_sessions.append(results)
                    print(f"     ✅ {session_type} loaded - {len(results)} drivers")
                    
                    time.sleep(sleep_seconds)   # NEW: Pause to avoid rate limit
                    
                except Exception as e:
                    print(f"     ⚠️  {session_type} skipped: {str(e)[:100]}")
            
            time.sleep(2)  # Extra pause between race weekends
        
        if all_sessions:
            final_df = pd.concat(all_sessions, ignore_index=True)
            filename = DATA_RAW_DIR / f"f1_data_{year}.parquet"
            final_df.to_parquet(filename, index=False)
            
            print(f"\n✅ SUCCESS: Saved {len(final_df):,} rows for {year}")
            return final_df
        else:
            print("❌ No data collected")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading {year}: {e}")
        return None


# ===================== RUN =====================
if __name__ == "__main__":
    # Run one year at a time to avoid rate limits
    years = [2025]          # Change this to [2023] or [2026] later
    
    for year in years:
        df = download_season(year, sleep_seconds=2.0)   # 2 seconds delay
        print("-" * 60)
    
    print("\n🎉 Finished this run!")
