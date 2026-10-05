"""
Automated Pipeline Sanity Test
==============================
Verifies data loading, preprocessing, model predictions, and PCA projection.
Run with: pytest test_app_pipeline.py  OR  python test_app_pipeline.py
"""

import os
import pandas as pd
import numpy as np
from app import load_full_pipeline

def test_pipeline_initialization():
    """Verify that models and dataset load and train without exceptions."""
    pipeline = load_full_pipeline("transactions.parquet")
    assert pipeline is not None
    assert len(pipeline["df_raw"]) > 0
    assert pipeline["kmeans"] is not None
    assert pipeline["iso_forest"] is not None
    assert len(pipeline["expected_features"]) > 0

def test_single_inference_flow():
    """Verify that a synthetic sample can be encoded, scaled, and predicted."""
    pipeline = load_full_pipeline("transactions.parquet")
    
    sample_df = pd.DataFrame([{
        "amount": 250.0,
        "frequency": 12,
        "balance": 15000.0,
        "duration": 18.0,
        "n_items": 3,
        "discount": 5.0,
        "age": 35,
        "days_since": 4.0,
        "tenure": 24,
        "credit_score": 710,
        "merchant_risk": 0.15,
        "distance_km": 8.5,
        "time_of_day": "afternoon",
        "channel": "pos",
        "device": "mobile",
        "card_type": "debit"
    }])
    
    # 1. Align features
    encoded = pd.get_dummies(sample_df, drop_first=True)
    aligned = encoded.reindex(columns=pipeline["expected_features"], fill_value=0)
    assert aligned.shape[1] == len(pipeline["expected_features"])
    
    # 2. Scale
    scaler = pipeline["preprocessor"].get_scaler()
    scaled = scaler.transform(aligned)
    assert scaled.shape == (1, len(pipeline["expected_features"]))
    
    # 3. Predict Cluster & Anomaly
    cluster = pipeline["kmeans"].predict(scaled)[0]
    assert cluster in [0, 1, 2, 3]
    
    anomaly_flag = pipeline["iso_forest"].predict(scaled)[0]
    assert anomaly_flag in [-1, 1]
    
    # 4. Project into 2D PCA
    pca_coords = pipeline["pca_2d"].transform(scaled)[0]
    assert len(pca_coords) == 2

if __name__ == "__main__":
    test_pipeline_initialization()
    test_single_inference_flow()
    print("✅ All pipeline automated tests passed successfully!")
