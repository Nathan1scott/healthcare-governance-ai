import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# Load the CSV files
patients = pd.read_csv('output/csv/patients.csv')
encounters = pd.read_csv('output/csv/encounters.csv')
conditions = pd.read_csv('output/csv/conditions.csv')
medications = pd.read_csv('output/csv/medications.csv')

print(f"Loaded {len(patients)} patients")
print(f"Loaded {len(medications)} medication records")

# Create a prior authorization dataset
# In real life, this would have historical approval/denial data
# We'll simulate realistic approval patterns based on clinical rules

# First, create a list of medications with different approval rates
high_cost_meds = ['insulin', 'metformin', 'levothyroxine', 'lisinopril', 'atorvastatin']
medium_cost_meds = ['amlodipine', 'metoprolol', 'omeprazole', 'sertraline', 'albuterol']
low_cost_meds = ['ibuprofen', 'acetaminophen', 'aspirin', 'loratadine', 'cetirizine']

# Build training dataset
training_data = []

for idx, med in medications.iterrows():
    patient_id = med['PATIENT']
    medication = med['DESCRIPTION'].lower()
    
    # Get patient info
    patient = patients[patients['Id'] == patient_id]
    if patient.empty:
        continue
    
    age = 2024 - pd.to_datetime(patient['BIRTHDATE'].iloc[0]).year
    gender = patient['GENDER'].iloc[0]
    race = patient['RACE'].iloc[0]
    
    # Get patient's conditions ( comorbidities)
    patient_conditions = conditions[conditions['PATIENT'] == patient_id]
    num_conditions = len(patient_conditions)
    has_chronic_disease = any('chronic' in str(c).lower() for c in patient_conditions['DESCRIPTION'])
    
    # Determine medication cost tier
    if any(med in medication for med in high_cost_meds):
        cost_tier = 'high'
        base_approval_prob = 0.65  # 65% approval rate for high-cost meds
    elif any(med in medication for med in medium_cost_meds):
        cost_tier = 'medium'
        base_approval_prob = 0.85  # 85% approval rate for medium-cost meds
    else:
        cost_tier = 'low'
        base_approval_prob = 0.95  # 95% approval rate for low-cost meds
    
    # Adjust probability based on patient factors
    approval_prob = base_approval_prob
    if age > 65:
        approval_prob += 0.05  # Seniors get more approvals
    if num_conditions > 3:
        approval_prob += 0.03  # More comorbidities = higher need
    if has_chronic_disease:
        approval_prob += 0.07  # Chronic conditions justify more meds
    if cost_tier == 'high' and num_conditions < 2:
        approval_prob -= 0.10  # High-cost meds with few conditions more likely denied
    
    # Cap at 0.05 to 0.99
    approval_prob = max(0.05, min(0.99, approval_prob))
    
    # Determine approval outcome based on probability
    approved = np.random.random() < approval_prob
    
    training_data.append({
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'race': race,
        'num_conditions': num_conditions,
        'has_chronic_disease': int(has_chronic_disease),
        'cost_tier': cost_tier,
        'medication': medication[:50],  # First 50 chars
        'approval_probability': approval_prob,
        'approved': int(approved)
    })

# Convert to DataFrame
df = pd.DataFrame(training_data)
print(f"\nCreated {len(df)} prior authorization cases")
print(f"Approval rate: {df['approved'].mean()*100:.1f}%")

# Prepare features for XGBoost
print("\nPreparing features...")

# Encode categorical variables
le_gender = LabelEncoder()
le_race = LabelEncoder()
le_cost_tier = LabelEncoder()

df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['race_encoded'] = le_race.fit_transform(df['race'])
df['cost_tier_encoded'] = le_cost_tier.fit_transform(df['cost_tier'])

# Select features for model
feature_cols = ['age', 'num_conditions', 'has_chronic_disease', 
                'gender_encoded', 'race_encoded', 'cost_tier_encoded']
X = df[feature_cols]
y = df['approved']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train XGBoost model
print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Save model and encoders
print("\nSaving model and encoders...")
joblib.dump(model, 'prior_auth_model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_race, 'le_race.pkl')
joblib.dump(le_cost_tier, 'le_cost_tier.pkl')

# Save feature columns list
import json
with open('feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

print("✓ Model saved as 'prior_auth_model.pkl'")
print("✓ Encoders saved")
print("\nReady to integrate with the API!")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)