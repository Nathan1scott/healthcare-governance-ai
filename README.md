![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)
![API](https://img.shields.io/badge/API-FastAPI-red)
# 🏥 Healthcare Governance AI

An AI-powered prior authorization system that predicts medication approval probability in real-time using XGBoost.

## 🚀 Live Demo

1. Clone the repo
2. Run `python main.py`
3. Open `dashboard.html`
4. Test with patient ID: `9a76060b-3539-2fa6-35e0-73e942e4ee2b`

## 📊 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Check API status |
| `/prior_auth/{patient_id}/{medication}` | Check one medication |
| `/prior_auth_batch/{patient_id}` | Check all medications |
| `/generate_report/{patient_id}` | Download PDF report |

## 📊 Model Performance Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 87% | Correct predictions 87% of the time |
| **Precision** | 85% | When predicting "approved", 85% are correct |
| **Recall** | 88% | Captured 88% of actual approvals |
| **AUC-ROC** | 0.91 | Excellent discrimination ability |
| **F1-Score** | 0.86 | Balanced precision and recall |

## 🔬 Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Number of Conditions | 32% |
| 2 | Medication Cost Tier | 28% |
| 3 | Age | 18% |
| 4 | Chronic Disease Status | 14% |
| 5 | Gender | 5% |
| 6 | Race | 3% |

## 📈 Model Comparison

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 78% | 0.82 |
| Random Forest | 84% | 0.88 |
| **XGBoost (Final)** | **87%** | **0.91** |

## 🛠️ Tech Stack

- Python + FastAPI
- XGBoost Machine Learning
- Pandas for data processing
- ReportLab for PDF generation

---

**Created by Nathan Aniakwa**
