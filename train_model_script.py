# train_model_script.py
"""
Water Potability Model Training Script
======================================
Script untuk melatih model Random Forest dengan Hyperparameter Tuning.
Menghasilkan model, scaler, dan imputer yang siap digunakan oleh aplikasi Streamlit.

Author: Atep Solihin - 301230038 - IF 5A
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def run_training_pipeline():
    """Execute the complete training pipeline."""
    print("="*60)
    print("WATER POTABILITY MODEL TRAINING PIPELINE")
    print("="*60)
    print()
    
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print("[1/7] Loading dataset...")
    try:
        df = pd.read_csv('water_potability.csv')
        print(f"      ✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    except FileNotFoundError:
        print("      ✗ ERROR: 'water_potability.csv' not found!")
        return
    
    # ==========================================
    # 2. DATA SPLITTING (Leakage-Free)
    # ==========================================
    print("[2/7] Splitting data (80% train, 20% test)...")
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      ✓ Training set: {X_train.shape[0]} samples")
    print(f"      ✓ Test set: {X_test.shape[0]} samples")
    
    # ==========================================
    # 3. IMPUTATION (Handle Missing Values)
    # ==========================================
    print("[3/7] Imputing missing values (mean strategy)...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    missing_before = df.isnull().sum().sum()
    print(f"      ✓ Filled {missing_before} missing values")
    
    # ==========================================
    # 4. SMOTE (Handle Class Imbalance)
    # ==========================================
    print("[4/7] Applying SMOTE for class balancing...")
    print(f"      Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
    
    print(f"      After SMOTE:  {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
    print(f"      ✓ Training samples increased: {len(y_train)} → {len(y_train_resampled)}")
    
    # ==========================================
    # 5. SCALING (Standardization)
    # ==========================================
    print("[5/7] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_resampled)
    X_test_final = scaler.transform(X_test_imputed)
    print("      ✓ Features normalized (mean=0, std=1)")
    
    # ==========================================
    # 6. MODEL TRAINING WITH HYPERPARAMETER TUNING
    # ==========================================
    print("[6/7] Training Random Forest with RandomizedSearchCV...")
    print("      (This may take a few minutes...)")
    
    # Parameter grid for tuning
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    
    rf_random.fit(X_train_final, y_train_resampled)
    best_rf = rf_random.best_estimator_
    
    print(f"      ✓ Best Parameters: {rf_random.best_params_}")
    
    # ==========================================
    # 7. EVALUATION
    # ==========================================
    print("[7/7] Evaluating model performance...")
    
    y_pred = best_rf.predict(X_test_final)
    y_prob = best_rf.predict_proba(X_test_final)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    print(f"\n      {'='*40}")
    print(f"      MODEL PERFORMANCE METRICS")
    print(f"      {'='*40}")
    print(f"      Accuracy:  {accuracy*100:.2f}%")
    print(f"      AUC-ROC:   {auc_score:.4f}")
    print(f"      {'='*40}")
    print("\n      Classification Report:")
    print("      " + "-"*40)
    report = classification_report(y_test, y_pred, target_names=['Not Potable', 'Potable'])
    for line in report.split('\n'):
        print(f"      {line}")
    
    # ==========================================
    # SAVE ARTIFACTS
    # ==========================================
    print("\n" + "="*60)
    print("SAVING MODEL ARTIFACTS")
    print("="*60)
    
    # Save to root folder (compatible with train_model.py output)
    joblib.dump(best_rf, 'model_water_rf.pkl')
    joblib.dump(scaler, 'scaler_water.pkl')
    joblib.dump(imputer, 'imputer_water.pkl')
    
    print("✓ model_water_rf.pkl  - Random Forest Model (Tuned)")
    print("✓ scaler_water.pkl    - StandardScaler")
    print("✓ imputer_water.pkl   - SimpleImputer")
    
    print("\n" + "="*60)
    print("SUCCESS! Model artifacts saved.")
    print("You can now run: streamlit run app.py")
    print("="*60)
    
    # Feature Importance
    print("\n" + "-"*40)
    print("FEATURE IMPORTANCE RANKING:")
    print("-"*40)
    
    feature_names = X.columns.tolist()
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


if __name__ == "__main__":
    run_training_pipeline()
