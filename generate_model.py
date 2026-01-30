# generate_model.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

def run_training_pipeline():
    print("MEMULAI PIPELINE DATA SCIENCE...")

    # 1. Setup Folder Models
    # Kita paksa buat folder 'models' biar rapi
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Folder 'models/' berhasil dibuat.")

    # 2. Load Data
    try:
        df = pd.read_csv('water_potability.csv')
        print(f"Dataset ditemukan: {df.shape}")
    except FileNotFoundError:
        print("ERROR FATAL: File 'water_potability.csv' tidak ada di folder ini!")
        return

    # 3. Data Splitting
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Stratify agar proporsi kelas seimbang di test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data berhasil dipisah (Train/Test).")

    # 4. Imputasi (Mengisi Data Kosong)
    # Kita pakai SimpleImputer (Mean) agar robust dan file size kecil
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    # Ingat: Test set hanya di-transform!
    
    # 5. SMOTE (Menyeimbangkan Kelas)
    print("Menyeimbangkan data dengan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

    # 6. Scaling (Standarisasi)
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_resampled)
    
    # 7. Training Model
    print("🌲 Sedang melatih Random Forest (Tunggu sebentar)...")
    # Menggunakan parameter yang cukup optimal
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train_final, y_train_resampled)

    # 8. Evaluasi Singkat (Untuk memastikan model waras)
    # Kita perlu pre-process data test dulu
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"AkurasI Model di Data Test: {acc*100:.2f}%")

    # 9. Saving Artifacts (INTI DARI SEMUANYA)
    print("Menyimpan file ke folder 'models/'...")
    joblib.dump(model, 'models/model_water_rf.pkl')
    joblib.dump(scaler, 'models/scaler_water.pkl')
    joblib.dump(imputer, 'models/imputer_water.pkl')

    print("\n" + "="*50)
    print("SUKSES! LINGKUNGAN SIAP UNTUK CLAUDE.")
    print("File berikut sudah ada di folder 'models/':")
    print("1. model_water_rf.pkl")
    print("2. scaler_water.pkl")
    print("3. imputer_water.pkl")
    print("="*50)

if __name__ == "__main__":
    run_training_pipeline()