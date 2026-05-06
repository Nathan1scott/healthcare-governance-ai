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

## 🛠️ Tech Stack

- Python + FastAPI
- XGBoost Machine Learning
- Pandas for data processing
- ReportLab for PDF generation

---

**Created by Nathan Scott**
