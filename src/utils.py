# src/utils.py
"""
Utility Functions for Water Potability Classification App
=========================================================
Contains functions for loading ML artifacts, preprocessing data,
and making predictions.

Supports models from train_model.py (root folder) or generate_model.py (models folder)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple, Dict, Any, Optional, List

# Feature configuration - MUST maintain this exact order
FEATURE_NAMES = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

# Feature descriptions for UI
FEATURE_INFO = {
    'ph': {
        'label': 'pH Level',
        'unit': '',
        'description': 'Acidity/alkalinity of water (0-14 scale)',
        'min': 0.0,
        'max': 14.0,
        'default': 7.0,
        'icon': '🧪'
    },
    'Hardness': {
        'label': 'Hardness',
        'unit': 'mg/L',
        'description': 'Capacity of water to precipitate soap (calcium/magnesium)',
        'min': 47.0,
        'max': 324.0,
        'default': 196.0,
        'icon': '💎'
    },
    'Solids': {
        'label': 'Total Dissolved Solids',
        'unit': 'ppm',
        'description': 'Total dissolved solids in water',
        'min': 320.0,
        'max': 61227.0,
        'default': 20927.0,
        'icon': '�ite'
    },
    'Chloramines': {
        'label': 'Chloramines',
        'unit': 'ppm',
        'description': 'Amount of chloramines in water',
        'min': 0.35,
        'max': 13.13,
        'default': 7.12,
        'icon': '🧫'
    },
    'Sulfate': {
        'label': 'Sulfate',
        'unit': 'mg/L',
        'description': 'Amount of sulfate dissolved in water',
        'min': 129.0,
        'max': 481.0,
        'default': 333.0,
        'icon': '⚗️'
    },
    'Conductivity': {
        'label': 'Conductivity',
        'unit': 'μS/cm',
        'description': 'Electrical conductivity of water',
        'min': 181.0,
        'max': 753.0,
        'default': 426.0,
        'icon': '⚡'
    },
    'Organic_carbon': {
        'label': 'Organic Carbon',
        'unit': 'ppm',
        'description': 'Amount of organic carbon in water',
        'min': 2.2,
        'max': 28.3,
        'default': 14.28,
        'icon': '🌿'
    },
    'Trihalomethanes': {
        'label': 'Trihalomethanes',
        'unit': 'μg/L',
        'description': 'Amount of trihalomethanes (THMs) in water',
        'min': 0.74,
        'max': 124.0,
        'default': 66.4,
        'icon': '🔬'
    },
    'Turbidity': {
        'label': 'Turbidity',
        'unit': 'NTU',
        'description': 'Measure of water clarity',
        'min': 1.45,
        'max': 6.74,
        'default': 3.97,
        'icon': '💧'
    }
}


def get_feature_names() -> list:
    """Return the list of feature names in correct order."""
    return FEATURE_NAMES.copy()


def get_feature_ranges() -> Dict[str, Dict[str, Any]]:
    """Return feature information including ranges and descriptions."""
    return FEATURE_INFO.copy()


def find_model_files() -> Tuple[str, str, str]:
    """
    Find model files - supports both train_model.py output (root) and 
    generate_model.py output (models folder).
    
    Returns:
        Tuple of (model_path, scaler_path, imputer_path)
    """
    # Priority 1: Root folder (from train_model.py)
    root_paths = (
        'model_water_rf.pkl',
        'scaler_water.pkl', 
        'imputer_water.pkl'
    )
    
    # Priority 2: models folder (from generate_model.py)
    models_paths = (
        'models/model_water_rf.pkl',
        'models/scaler_water.pkl',
        'models/imputer_water.pkl'
    )
    
    # Check root first
    if all(os.path.exists(p) for p in root_paths):
        return root_paths
    
    # Check models folder
    if all(os.path.exists(p) for p in models_paths):
        return models_paths
    
    # Return root paths (will raise error in load_artifacts)
    return root_paths


@st.cache_resource
def load_artifacts() -> Tuple[Any, Any, Any]:
    """
    Load all ML artifacts (model, scaler, imputer).
    Supports both train_model.py and generate_model.py outputs.
    Uses Streamlit caching for performance optimization.
    
    Returns:
        Tuple of (model, scaler, imputer)
        
    Raises:
        FileNotFoundError: If any artifact file is missing
    """
    model_path, scaler_path, imputer_path = find_model_files()
    
    # Check if all files exist
    missing_files = []
    for path, name in [(model_path, 'Model'), (scaler_path, 'Scaler'), (imputer_path, 'Imputer')]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(
            f"Artifact files not found:\n" + "\n".join(missing_files) +
            "\n\nPlease run 'python train_model.py' or 'python generate_model.py' first."
        )
    
    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    
    return model, scaler, imputer


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    try:
        model, _, _ = load_artifacts()
        model_path, _, _ = find_model_files()
        
        info = {
            'type': type(model).__name__,
            'source': 'train_model.py' if 'models' not in model_path else 'generate_model.py',
            'path': model_path,
        }
        
        # Get Random Forest specific info
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth if model.max_depth else 'None'
        if hasattr(model, 'feature_importances_'):
            info['has_feature_importance'] = True
            
        return info
    except:
        return {'type': 'Unknown', 'source': 'Not loaded'}


def preprocess_input(
    data: pd.DataFrame,
    imputer: Any,
    scaler: Any
) -> np.ndarray:
    """
    Preprocess input data through the pipeline: Imputer -> Scaler
    
    Args:
        data: DataFrame with feature columns in correct order
        imputer: Fitted SimpleImputer instance
        scaler: Fitted StandardScaler instance
        
    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Ensure correct column order
    data_ordered = data[FEATURE_NAMES].copy()
    
    # Apply imputation (handles missing values)
    data_imputed = imputer.transform(data_ordered)
    
    # Apply scaling
    data_scaled = scaler.transform(data_imputed)
    
    return data_scaled


def predict_single(
    input_values: Dict[str, float],
    model: Any,
    scaler: Any,
    imputer: Any
) -> Tuple[int, float]:
    """
    Make prediction for a single sample.
    
    Args:
        input_values: Dictionary mapping feature names to values
        model: Trained classifier
        scaler: Fitted scaler
        imputer: Fitted imputer
        
    Returns:
        Tuple of (prediction_class, confidence_score)
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_values])
    
    # Preprocess
    X_processed = preprocess_input(df, imputer, scaler)
    
    # Predict
    prediction = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed)[0]
    confidence = probabilities[prediction]
    
    return int(prediction), float(confidence)


def predict_batch(
    data: pd.DataFrame,
    model: Any,
    scaler: Any,
    imputer: Any
) -> pd.DataFrame:
    """
    Make predictions for multiple samples (batch prediction).
    
    Args:
        data: DataFrame with feature columns
        model: Trained classifier
        scaler: Fitted scaler
        imputer: Fitted imputer
        
    Returns:
        DataFrame with original data plus prediction columns
    """
    # Ensure we have required columns
    missing_cols = set(FEATURE_NAMES) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Preprocess
    X_processed = preprocess_input(data, imputer, scaler)
    
    # Predict
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)
    
    # Create result DataFrame
    result = data.copy()
    result['Prediction'] = predictions
    result['Potability_Label'] = result['Prediction'].map({
        0: 'Not Potable ❌',
        1: 'Potable ✅'
    })
    result['Confidence'] = [
        probabilities[i][pred] for i, pred in enumerate(predictions)
    ]
    result['Confidence_Pct'] = (result['Confidence'] * 100).round(2).astype(str) + '%'
    
    return result


def load_dataset(filepath: str = 'water_potability.csv') -> Optional[pd.DataFrame]:
    """
    Load the original dataset for dashboard statistics.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame or None if file not found
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        return None


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics from the dataset for dashboard display.
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        Dictionary with statistics
    """
    total_samples = len(df)
    potable_count = df['Potability'].sum()
    not_potable_count = total_samples - potable_count
    potable_pct = (potable_count / total_samples) * 100
    not_potable_pct = (not_potable_count / total_samples) * 100
    missing_values = df.isnull().sum().sum()
    
    return {
        'total_samples': total_samples,
        'potable_count': int(potable_count),
        'not_potable_count': int(not_potable_count),
        'potable_pct': round(potable_pct, 1),
        'not_potable_pct': round(not_potable_pct, 1),
        'missing_values': int(missing_values),
        'feature_count': len(FEATURE_NAMES)
    }


def get_feature_importance(model: Any) -> Optional[pd.DataFrame]:
    """
    Get feature importance from the model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        
    Returns:
        DataFrame with feature importance or None
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df
    return None


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate detailed statistics for each feature.
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        DataFrame with statistics per feature
    """
    stats_list = []
    for feature in FEATURE_NAMES:
        stats = {
            'Feature': feature,
            'Label': FEATURE_INFO[feature]['label'],
            'Mean': df[feature].mean(),
            'Std': df[feature].std(),
            'Min': df[feature].min(),
            'Max': df[feature].max(),
            'Missing': df[feature].isnull().sum(),
            'Missing_Pct': (df[feature].isnull().sum() / len(df)) * 100
        }
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)
