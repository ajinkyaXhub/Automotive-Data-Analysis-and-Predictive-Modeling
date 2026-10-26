"""
train_model.py
=============================================================================
Used Car Price Prediction - Model Training and Evaluation Pipeline
College Mini-Project in Machine Learning

This script performs:
1. Dataset inspection and identification of actual CSV columns.
2. Data cleaning: duplicate removal, missing value handling, and brand extraction.
3. Feature preprocessing: ColumnTransformer with OneHotEncoder and StandardScaler.
4. Train/Test split (80% train, 20% test) with fixed random_state for reproducibility.
5. Model training: Linear Regression vs Random Forest Regressor.
6. Honest evaluation on unseen test data using MAE, RMSE, and R² score.
7. Model selection and serializing the winning pipeline to 'car_price_pipeline.joblib'.
8. Saving model metadata and evaluation results to 'model_metrics.json'.
=============================================================================
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def locate_dataset():
    """Locate the dataset file in the workspace."""
    candidate_paths = [
        "car data.csv",
        "CAR DETAILS FROM CAR DEKHO copy.csv",
        "CAR DETAILS FROM CAR DEKHO.csv",
        "car_data.csv"
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    
    # Fallback search for any CSV
    for file in os.listdir("."):
        if file.lower().endswith(".csv"):
            return file
            
    raise FileNotFoundError("Could not find any CSV dataset in the workspace.")


def load_and_clean_data(dataset_path):
    """
    Load dataset, inspect columns, remove duplicates, handle missing values,
    and perform feature extraction (extracting brand from car name).
    """
    print(f"[*] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    raw_shape = df.shape
    print(f"[i] Raw dataset shape: {raw_shape[0]} rows, {raw_shape[1]} columns")
    print(f"[i] Actual columns identified: {list(df.columns)}")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    print(f"[i] Missing values count per column:\n{missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else '    None (0 missing values)'}")
    
    # Remove duplicate records
    duplicates_count = df.duplicated().sum()
    print(f"[i] Duplicate rows found: {duplicates_count}")
    if duplicates_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[+] Dropped duplicates. New shape: {df.shape}")
        
    # Drop any remaining nulls just in case
    df = df.dropna().reset_index(drop=True)
    
    # Extract brand from car name (first word of 'name')
    if 'brand' not in df.columns and 'name' in df.columns:
        print("[+] Extracting 'brand' feature from 'name' column...")
        df['brand'] = df['name'].apply(lambda x: str(x).split()[0].strip())
    
    return df, raw_shape


def build_preprocessor(cat_features, num_features):
    """
    Build scikit-learn ColumnTransformer for categorical and numerical features.
    - Categorical: OneHotEncoder (handles unseen categories safely)
    - Numerical: StandardScaler
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ],
        remainder='drop'
    )
    return preprocessor


def train_and_evaluate():
    # 1. Load data
    dataset_path = locate_dataset()
    df, raw_shape = load_and_clean_data(dataset_path)
    
    # 2. Define features and target
    target_col = 'selling_price'
    num_features = ['year', 'km_driven']
    cat_features = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']
    feature_cols = ['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
    
    # Verify all required columns exist
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
    X = df[feature_cols]
    y = df[target_col]
    
    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[*] Train/Test Split completed:")
    print(f"    - Training samples: {len(X_train)}")
    print(f"    - Testing samples:  {len(X_test)}")
    
    # 4. Create Preprocessor and Model Pipelines
    preprocessor = build_preprocessor(cat_features, num_features)
    
    # Linear Regression Pipeline
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    
    # Random Forest Regressor Pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # 5. Train Models
    print("\n[*] Training Linear Regression model...")
    lr_pipeline.fit(X_train, y_train)
    
    print("[*] Training Random Forest Regressor model...")
    rf_pipeline.fit(X_train, y_train)
    
    # 6. Evaluate Models on Test Data (Strictly unseen)
    print("\n" + "=" * 65)
    print("                MODEL EVALUATION ON TEST SET")
    print("=" * 65)
    
    # Predictions
    y_pred_lr = lr_pipeline.predict(X_test)
    y_pred_rf = rf_pipeline.predict(X_test)
    
    # Linear Regression metrics
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = root_mean_squared_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)
    
    # Random Forest metrics
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = root_mean_squared_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    print(f"{'Metric':<25} | {'Linear Regression':<17} | {'Random Forest':<17}")
    print("-" * 65)
    print(f"{'MAE (Mean Abs Error)':<25} | Rs. {lr_mae:>12,.2f} | Rs. {rf_mae:>12,.2f}")
    print(f"{'RMSE (Root MSE)':<25} | Rs. {lr_rmse:>12,.2f} | Rs. {rf_rmse:>12,.2f}")
    print(f"{'R2 Score (Accuracy/Fit)':<25} | {lr_r2:>16.4f}  | {rf_r2:>16.4f}")
    print("-" * 65)
    
    # 7. Select Better Model
    # Higher R² and lower MAE/RMSE is better
    if rf_r2 > lr_r2:
        best_model_name = "Random Forest Regressor"
        best_pipeline = rf_pipeline
        reason = f"Random Forest achieved higher R2 ({rf_r2:.4f} vs {lr_r2:.4f}) and lower MAE (Rs. {rf_mae:,.0f} vs Rs. {lr_mae:,.0f})."
    else:
        best_model_name = "Linear Regression"
        best_pipeline = lr_pipeline
        reason = f"Linear Regression achieved higher R2 ({lr_r2:.4f} vs {rf_r2:.4f})."
        
    print(f"\n[+] Selected Champion Model: {best_model_name}")
    print(f"    Reason: {reason}")
    
    # 8. Save Pipeline
    model_filename = "car_price_pipeline.joblib"
    joblib.dump(best_pipeline, model_filename)
    print(f"[+] Saved trained pipeline to '{model_filename}'")
    
    # 9. Extract metadata & categories for Streamlit UI
    brands = sorted(df['brand'].unique().tolist())
    fuels = sorted(df['fuel'].unique().tolist())
    seller_types = sorted(df['seller_type'].unique().tolist())
    transmissions = sorted(df['transmission'].unique().tolist())
    owners = sorted(df['owner'].unique().tolist())
    
    metadata = {
        "best_model_name": best_model_name,
        "selection_reason": reason,
        "metrics": {
            "Linear Regression": {
                "mae": round(float(lr_mae), 2),
                "rmse": round(float(lr_rmse), 2),
                "r2": round(float(lr_r2), 4)
            },
            "Random Forest Regressor": {
                "mae": round(float(rf_mae), 2),
                "rmse": round(float(rf_rmse), 2),
                "r2": round(float(rf_r2), 4)
            }
        },
        "dataset_summary": {
            "source_file": dataset_path,
            "raw_records": int(raw_shape[0]),
            "cleaned_records": int(df.shape[0]),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "target_mean": round(float(df[target_col].mean()), 2),
            "target_median": round(float(df[target_col].median()), 2),
            "target_min": round(float(df[target_col].min()), 2),
            "target_max": round(float(df[target_col].max()), 2)
        },
        "features": feature_cols,
        "categories": {
            "brand": brands,
            "fuel": fuels,
            "seller_type": seller_types,
            "transmission": transmissions,
            "owner": owners
        },
        "ranges": {
            "year_min": int(df['year'].min()),
            "year_max": int(df['year'].max()),
            "km_min": int(df['km_driven'].min()),
            "km_max": int(df['km_driven'].max())
        }
    }
    
    metadata_filename = "model_metrics.json"
    with open(metadata_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved model metadata & evaluation metrics to '{metadata_filename}'")
    
    # 10. Sanity check test prediction
    test_sample = pd.DataFrame([{
        'brand': 'Maruti',
        'year': 2017,
        'km_driven': 45000,
        'fuel': 'Petrol',
        'seller_type': 'Individual',
        'transmission': 'Manual',
        'owner': 'First Owner'
    }])
    pred_val = best_pipeline.predict(test_sample)[0]
    print(f"\n[*] Sample Prediction Verification:")
    print(f"    Input: Maruti (2017, 45,000 km, Petrol, Manual, First Owner, Individual)")
    print(f"    Estimated Resale Price: Rs. {pred_val:,.2f} (~ Rs. {pred_val / 100000:.2f} Lakhs)")
    print("\n[SUCCESS] Training pipeline completed successfully!\n")


if __name__ == "__main__":
    train_and_evaluate()
