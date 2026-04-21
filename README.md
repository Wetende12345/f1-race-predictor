# 🏎️ F1 Race Outcome Predictor 2026

**End-to-End Machine Learning Project** for predicting Formula 1 race results.

Built step-by-step from data ingestion to deployment for **mastery** of data engineering and ML workflows.

---

## ✨ Project Features

- **Real F1 Data Ingestion** using FastF1 (2024–2025 seasons)
- **Advanced Feature Engineering**:
  - Driver recent form (last 5 races)
  - Teammate performance comparison
  - Qualifying strength
  - Track experience
  - Position gain/loss
- **Machine Learning Models** (XGBoost):
  - Race Position Prediction (MAE: **2.98** places)
  - Points Prediction (MAE: **2.27** points)
  - Top 3 Finish Classification (Accuracy: **93.5%**)
  - Race Winner Classification (Accuracy: **97.0%**)
- **Interactive Streamlit Dashboard** for live predictions

---

## 🛠️ Tech Stack

- **Data**: FastF1, Pandas, Parquet
- **Modeling**: XGBoost, Scikit-learn
- **Frontend**: Streamlit
- **Environment**: uv (Python package manager)

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Wetende12345/f1-race-predictor.git
cd f1-race-predictor

# 2. Install dependencies
uv sync

# 3. Run the Streamlit app
uv run streamlit run streamlit_app/app.py
