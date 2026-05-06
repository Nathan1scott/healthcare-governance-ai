from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import joblib
import json
import time
import datetime
from datetime import datetime as dt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

app = FastAPI(title="Healthcare Governance AI")

# Load data
patients_df = pd.read_csv('output/csv/patients.csv')
conditions_df = pd.read_csv('output/csv/conditions.csv')
medications_df = pd.read_csv('output/csv/medications.csv')

# Load model
try:
    model = joblib.load('prior_auth_model.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_race = joblib.load('le_race.pkl')
    le_cost_tier = joblib.load('le_cost_tier.pkl')
    model_loaded = True
    print("✓ Model loaded")
except:
    model_loaded = False
    print("⚠ Model not found")

print(f"Loaded {len(patients_df)} patients")

@app.get("/")
def root():
    return {"message": "Healthcare Governance AI", "status": "active"}

@app.get("/prior_auth/{patient_id}/{medication_code}")
async def prior_auth(patient_id: str, medication_code: str):
    if not model_loaded:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    
    patient = patients_df[patients_df['Id'] == patient_id]
    if patient.empty:
        return JSONResponse(status_code=404, content={"error": "Patient not found"})
    
    # Calculate features
    age = int(2024 - pd.to_datetime(patient['BIRTHDATE'].iloc[0]).year)
    gender = str(patient['GENDER'].iloc[0])
    race = str(patient['RACE'].iloc[0])
    
    patient_conditions = conditions_df[conditions_df['PATIENT'] == patient_id]
    num_conditions = int(len(patient_conditions))
    has_chronic = int(any('chronic' in str(c).lower() for c in patient_conditions['DESCRIPTION']))
    
    # Cost tier
    med_lower = medication_code.lower()
    if any(med in med_lower for med in ['insulin', 'metformin', 'epinephrine']):
        cost_tier = 'high'
    elif any(med in med_lower for med in ['amoxicillin', 'fexofenadine', 'ibuprofen']):
        cost_tier = 'medium'
    else:
        cost_tier = 'low'
    
    # Encode
    gender_enc = int(le_gender.transform([gender])[0])
    race_enc = int(le_race.transform([race])[0])
    cost_enc = int(le_cost_tier.transform([cost_tier])[0])
    
    # Predict
    features = [[age, num_conditions, has_chronic, gender_enc, race_enc, cost_enc]]
    approval_prob = float(model.predict_proba(features)[0][1])
    approved = bool(approval_prob >= 0.7)
    
    return {
        "patient_id": patient_id,
        "medication": medication_code,
        "age": age,
        "gender": gender,
        "num_conditions": num_conditions,
        "has_chronic_disease": bool(has_chronic),
        "cost_tier": cost_tier,
        "approval_probability": round(approval_prob, 3),
        "approved": approved,
        "recommendation": "Auto-approved" if approved else "Requires review"
    }

@app.get("/prior_auth_batch/{patient_id}")
async def prior_auth_batch(patient_id: str):
    patient_meds = medications_df[medications_df['PATIENT'] == patient_id]
    if patient_meds.empty:
        return {"message": "No medications found"}
    
    results = []
    for _, med in patient_meds.head(10).iterrows():
        medication = med['DESCRIPTION']
        result = await prior_auth(patient_id, medication)
        if isinstance(result, dict):
            results.append(result)
    
    return {
        "patient_id": patient_id,
        "total": len(results),
        "auto_approved": sum(1 for r in results if r.get('approved')),
        "medications": results
    }

@app.get("/generate_report/{patient_id}")
async def generate_report(patient_id: str):
    patient = patients_df[patients_df['Id'] == patient_id]
    if patient.empty:
        return JSONResponse(status_code=404, content={"error": "Patient not found"})
    
    batch_result = await prior_auth_batch(patient_id)
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph(f"Prior Authorization Report - {patient_id[:8]}", styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Patient info
    patient_name = f"{patient['FIRST'].iloc[0]} {patient['LAST'].iloc[0]}"
    elements.append(Paragraph(f"Patient: {patient_name}", styles['Normal']))
    elements.append(Paragraph(f"DOB: {pd.to_datetime(patient['BIRTHDATE'].iloc[0]).strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Medications table
    table_data = [["Medication", "Approval Prob", "Decision"]]
    for med in batch_result.get('medications', []):
        table_data.append([
            med['medication'][:30],
            f"{med['approval_probability']*100:.1f}%",
            "APPROVED" if med['approved'] else "REVIEW NEEDED"
        ])
    
    table = Table(table_data)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="application/pdf", 
                            headers={"Content-Disposition": f"attachment; filename=report_{patient_id[:8]}.pdf"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)