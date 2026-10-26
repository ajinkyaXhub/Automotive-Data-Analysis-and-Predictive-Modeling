"""
flask_app.py
=============================================================================
Carlytics — Machine Learning & Automotive Data Analytics
College Mini-Project in Machine Learning (Flask Web Application Entry Point)

Features:
- Enterprise Flask backend serving semantic HTML5, CSS3, and modern JavaScript.
- Direct integration with saved Random Forest pipeline (car_price_pipeline.joblib).
- What-If Price Simulator with real-time model predictions and sensitivity curves.
- Model Explainability: Global Feature Importance (MDI) & local feature sensitivity.
- Authentic model metrics from model_metrics.json and 3,577 cleaned records.
- Responsive, accessible dashboard with Indian Rupee formatting.
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sklearn.model_selection import train_test_split


app = Flask(__name__)

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(BASE_DIR, "car_price_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "car data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "CAR DETAILS FROM CAR DEKHO copy.csv")


# ---------------------------------------------------------------------------
# LOAD ML PIPELINE, METADATA & DATASET
# ---------------------------------------------------------------------------
def load_resources():
    """Load model pipeline, evaluation metadata, and cleaned dataset."""
    pipeline = None
    metadata = {}
    df_data = None
    
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    if os.path.exists(DATA_PATH):
        df_data = pd.read_csv(DATA_PATH)
        df_data = df_data.drop_duplicates().reset_index(drop=True)
        if 'brand' not in df_data.columns and 'name' in df_data.columns:
            df_data['brand'] = df_data['name'].apply(lambda x: str(x).split()[0].strip())
            
    return pipeline, metadata, df_data


PIPELINE, METADATA, DF_DATA = load_resources()


# ---------------------------------------------------------------------------
# GLOBAL FEATURE IMPORTANCE EXTRACTION
# ---------------------------------------------------------------------------
def extract_feature_importances():
    """
    Extract genuine Mean Decrease in Impurity (MDI) feature importances
    from the fitted Random Forest estimator and ColumnTransformer.
    """
    if PIPELINE is None:
        return [], []
        
    try:
        preprocessor = PIPELINE.named_steps['preprocessor']
        rf = PIPELINE.named_steps['model']
        feature_names = preprocessor.get_feature_names_out()
        importances = rf.feature_importances_
        
        # 1. Individual Transformed Features
        transformed_list = []
        for name, imp in zip(feature_names, importances):
            clean_name = name.replace('cat__', '').replace('num__', '')
            transformed_list.append({
                'name': clean_name,
                'importance': float(imp),
                'percentage': round(float(imp) * 100, 2)
            })
        transformed_list.sort(key=lambda x: x['importance'], reverse=True)
        
        # 2. Aggregated Base Features
        base_groups = {
            'brand': {'label': 'Manufacturer / Brand', 'importance': 0.0},
            'transmission': {'label': 'Transmission Type', 'importance': 0.0},
            'year': {'label': 'Manufacturing Year', 'importance': 0.0},
            'km_driven': {'label': 'Kilometers Driven', 'importance': 0.0},
            'fuel': {'label': 'Fuel Mechanism', 'importance': 0.0},
            'seller_type': {'label': 'Seller Channel', 'importance': 0.0},
            'owner': {'label': 'Previous Ownership', 'importance': 0.0}
        }
        
        for name, imp in zip(feature_names, importances):
            if 'brand' in name:
                base_groups['brand']['importance'] += imp
            elif 'transmission' in name:
                base_groups['transmission']['importance'] += imp
            elif 'year' in name:
                base_groups['year']['importance'] += imp
            elif 'km_driven' in name:
                base_groups['km_driven']['importance'] += imp
            elif 'fuel' in name:
                base_groups['fuel']['importance'] += imp
            elif 'seller_type' in name:
                base_groups['seller_type']['importance'] += imp
            elif 'owner' in name:
                base_groups['owner']['importance'] += imp
                
        base_list = []
        for k, v in base_groups.items():
            base_list.append({
                'key': k,
                'name': v['label'],
                'importance': float(v['importance']),
                'percentage': round(float(v['importance']) * 100, 2)
            })
        base_list.sort(key=lambda x: x['importance'], reverse=True)
        
        return base_list, transformed_list
    except Exception as e:
        print("[!] Error extracting feature importances:", e)
        return [], []


BASE_IMPORTANCES, TRANSFORMED_IMPORTANCES = extract_feature_importances()

# Population reference vehicle (median/mode of training set)
REFERENCE_VEHICLE = {
    'brand': 'Maruti',
    'year': 2014,
    'km_driven': 60000,
    'fuel': 'Diesel',
    'seller_type': 'Individual',
    'transmission': 'Manual',
    'owner': 'First Owner'
}


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------
def format_inr(value):
    """
    Format numeric values into standard Indian Rupee notation.
    Returns: (formatted_full_string, denomination_string)
    Example: 485000 -> ("₹ 4,85,000", "₹ 4.85 Lakhs")
    """
    if value is None or np.isnan(value):
        return "₹ 0", "₹ 0"
        
    num_int = int(round(value))
    abs_num = abs(num_int)
    s = str(abs_num)
    
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted = ",".join(groups) + "," + last3
        
    prefix = "₹ " if num_int >= 0 else "-₹ "
    full_str = f"{prefix}{formatted}"
    
    if abs_num >= 10000000:
        denom_str = f"₹ {num_int / 10000000:.2f} Cr"
    elif abs_num >= 100000:
        denom_str = f"₹ {num_int / 100000:.2f} Lakhs"
    elif abs_num >= 1000:
        denom_str = f"₹ {num_int / 1000:.1f}k"
    else:
        denom_str = full_str
        
    return full_str, denom_str


def get_test_predictions():
    """
    Deterministically reproduce the test set predictions from the 20% held-out test split.
    Matches the exact training random_state=42.
    """
    if DF_DATA is None or PIPELINE is None:
        return None, None, None
        
    features = ['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
    target = 'selling_price'
    
    X = DF_DATA[features]
    y = DF_DATA[target]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    y_pred = PIPELINE.predict(X_test)
    
    return X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# FLASK APPLICATION ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Redirect root to overview dashboard."""
    return redirect(url_for("overview"))


@app.route("/overview")
def overview():
    """Overview dashboard page with KPI metrics, market analytics, and recent valuations."""
    summary = METADATA.get("dataset_summary", {})
    metrics = METADATA.get("metrics", {})
    rf_m = metrics.get("Random Forest Regressor", {})
    lr_m = metrics.get("Linear Regression", {})
    
    mae_full, mae_denom = format_inr(rf_m.get("mae", 159985.64))
    r2_score = rf_m.get("r2", 0.5780)
    cleaned_count = summary.get("cleaned_records", len(DF_DATA) if DF_DATA is not None else 3577)
    raw_count = summary.get("raw_records", 4340)
    
    # 1. Price Distribution Chart Data (under 25 Lakhs)
    p_sub = DF_DATA[DF_DATA['selling_price'] <= 2500000]['selling_price'] / 100000 if DF_DATA is not None else pd.Series([])
    price_hist_data = {
        'prices': p_sub.round(2).tolist()
    }
    
    # 2. Depreciation Trend Data (median price by manufacturing year)
    if DF_DATA is not None:
        trend = DF_DATA.groupby('year')['selling_price'].median().reset_index()
        trend['price_lakhs'] = (trend['selling_price'] / 100000).round(2)
        trend_data = {
            'years': trend['year'].tolist(),
            'prices': trend['price_lakhs'].tolist()
        }
    else:
        trend_data = {'years': [], 'prices': []}
        
    # 3. Fuel & Transmission Breakdown
    if DF_DATA is not None:
        ft_counts = DF_DATA.groupby(['fuel', 'transmission']).size().unstack(fill_value=0).reset_index()
        fuel_trans_data = {
            'fuels': ft_counts['fuel'].tolist(),
            'manual': ft_counts.get('Manual', pd.Series(0)).tolist(),
            'automatic': ft_counts.get('Automatic', pd.Series(0)).tolist()
        }
    else:
        fuel_trans_data = {'fuels': [], 'manual': [], 'automatic': []}
        
    # 4. Top 10 Manufacturers by Average Resale Price
    if DF_DATA is not None:
        top_brands = DF_DATA['brand'].value_counts().head(10).index
        brand_avg = DF_DATA[DF_DATA['brand'].isin(top_brands)].groupby('brand')['selling_price'].mean().sort_values(ascending=True) / 100000
        brand_data = {
            'brands': brand_avg.index.tolist(),
            'avg_prices': brand_avg.round(2).tolist()
        }
    else:
        brand_data = {'brands': [], 'avg_prices': []}
        
    return render_template(
        "overview.html",
        active_page="overview",
        summary=summary,
        cleaned_count=cleaned_count,
        raw_count=raw_count,
        mae_denom=mae_denom,
        mae_full=mae_full,
        r2_score=r2_score,
        r2_pct=f"{r2_score * 100:.1f}%",
        price_hist_data=json.dumps(price_hist_data),
        trend_data=json.dumps(trend_data),
        fuel_trans_data=json.dumps(fuel_trans_data),
        brand_data=json.dumps(brand_data)
    )


@app.route("/estimator")
def estimator():
    """Price Estimator page with structured inputs and real-time prediction."""
    categories = METADATA.get("categories", {})
    brands = categories.get("brand", [])
    fuels = categories.get("fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmissions = categories.get("transmission", ["Manual", "Automatic"])
    sellers = categories.get("seller_type", ["Individual", "Dealer", "Trustmark Dealer"])
    owners = categories.get("owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    
    return render_template(
        "estimator.html",
        active_page="estimator",
        brands=brands,
        fuels=fuels,
        transmissions=transmissions,
        sellers=sellers,
        owners=owners
    )


@app.route("/simulator")
def simulator():
    """
    What-If Price Simulator page:
    Interactive tool comparing baseline vs modified configurations with sensitivity curves.
    """
    categories = METADATA.get("categories", {})
    brands = categories.get("brand", [])
    fuels = categories.get("fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmissions = categories.get("transmission", ["Manual", "Automatic"])
    sellers = categories.get("seller_type", ["Individual", "Dealer", "Trustmark Dealer"])
    owners = categories.get("owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    
    return render_template(
        "simulator.html",
        active_page="simulator",
        brands=brands,
        fuels=fuels,
        transmissions=transmissions,
        sellers=sellers,
        owners=owners,
        default_baseline={
            'brand': 'Maruti',
            'year': 2017,
            'km_driven': 45000,
            'fuel': 'Petrol',
            'seller_type': 'Individual',
            'transmission': 'Manual',
            'owner': 'First Owner'
        }
    )


@app.route("/explainability")
def explainability():
    """
    Model Explainability page:
    Global Feature Importance (MDI) and local feature sensitivity diagnostics.
    """
    categories = METADATA.get("categories", {})
    brands = categories.get("brand", [])
    
    return render_template(
        "explainability.html",
        active_page="explainability",
        brands=brands,
        base_importances=BASE_IMPORTANCES,
        transformed_importances=TRANSFORMED_IMPORTANCES,
        base_importances_json=json.dumps(BASE_IMPORTANCES),
        transformed_importances_json=json.dumps(TRANSFORMED_IMPORTANCES),
        reference_vehicle=REFERENCE_VEHICLE
    )


@app.route("/explorer")
def explorer():
    """Data Explorer page with search, filters, data preview, and market patterns."""
    categories = METADATA.get("categories", {})
    brands = categories.get("brand", [])
    fuels = categories.get("fuel", [])
    transmissions = categories.get("transmission", [])
    
    sample_records = []
    if DF_DATA is not None:
        sample_df = DF_DATA.head(20).copy()
        for _, row in sample_df.iterrows():
            sample_records.append({
                'name': row.get('name', ''),
                'brand': row.get('brand', ''),
                'year': int(row.get('year', 0)),
                'selling_price_raw': int(row.get('selling_price', 0)),
                'selling_price_fmt': f"₹ {row.get('selling_price', 0) / 100000:.2f} L",
                'km_driven_fmt': f"{int(row.get('km_driven', 0)):,} km",
                'fuel': row.get('fuel', ''),
                'transmission': row.get('transmission', ''),
                'owner': row.get('owner', '')
            })
            
    # Boxplot data: Fuel & Transmission vs Selling Price (in Lakhs)
    boxplot_data = []
    if DF_DATA is not None:
        for fuel_type in ['Petrol', 'Diesel', 'CNG']:
            for trans_type in ['Manual', 'Automatic']:
                sub_df = DF_DATA[(DF_DATA['fuel'] == fuel_type) & (DF_DATA['transmission'] == trans_type)]
                if len(sub_df) > 0:
                    boxplot_data.append({
                        'fuel': fuel_type,
                        'transmission': trans_type,
                        'prices': (sub_df['selling_price'] / 100000).round(2).tolist()
                    })
                    
    # Scatter sample data: Km vs Price
    scatter_data = []
    if DF_DATA is not None:
        scat_df = DF_DATA[DF_DATA['km_driven'] <= 200000].sample(n=min(300, len(DF_DATA)), random_state=42)
        for _, r in scat_df.iterrows():
            scatter_data.append({
                'km': int(r['km_driven']),
                'price': round(float(r['selling_price']) / 100000, 2),
                'brand': str(r['brand']),
                'fuel': str(r['fuel'])
            })
            
    return render_template(
        "explorer.html",
        active_page="explorer",
        brands=brands,
        fuels=fuels,
        transmissions=transmissions,
        total_count=len(DF_DATA) if DF_DATA is not None else 3577,
        sample_records=sample_records,
        boxplot_data=json.dumps(boxplot_data),
        scatter_data=json.dumps(scatter_data)
    )


@app.route("/performance")
def performance():
    """Model Performance page with model comparisons, scatter plot, and statistical guide."""
    metrics = METADATA.get("metrics", {})
    lr_m = metrics.get("Linear Regression", {})
    rf_m = metrics.get("Random Forest Regressor", {})
    
    # Reproduce deterministic test predictions for actual-vs-predicted scatter plot
    X_test, y_test, y_pred = get_test_predictions()
    
    scatter_points = []
    if y_test is not None and y_pred is not None:
        for act, pred, brand, yr, km in zip(y_test.values, y_pred, X_test['brand'].values, X_test['year'].values, X_test['km_driven'].values):
            if act <= 3500000:
                scatter_points.append({
                    'actual': round(float(act) / 100000, 2),
                    'predicted': round(float(pred) / 100000, 2),
                    'brand': str(brand),
                    'year': int(yr),
                    'km': f"{int(km):,} km"
                })
                
    return render_template(
        "performance.html",
        active_page="performance",
        lr_metrics=lr_m,
        rf_metrics=rf_m,
        scatter_points=json.dumps(scatter_points)
    )


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint to predict car resale price using the saved Random Forest pipeline.
    Expects JSON: { brand, year, km_driven, fuel, seller_type, transmission, owner }
    """
    if PIPELINE is None:
        return jsonify({"success": False, "error": "Model pipeline is not loaded on server."}), 500
        
    try:
        data = request.get_json(force=True)
        
        brand = str(data.get("brand", "")).strip()
        year = data.get("year")
        km_driven = data.get("km_driven")
        fuel = str(data.get("fuel", "")).strip()
        seller_type = str(data.get("seller_type", "")).strip()
        transmission = str(data.get("transmission", "")).strip()
        owner = str(data.get("owner", "")).strip()
        
        if not brand or not fuel or not seller_type or not transmission or not owner:
            return jsonify({"success": False, "error": "All vehicle specification fields are required."}), 400
            
        try:
            year = int(year)
            km_driven = int(km_driven)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "Year and kilometers driven must be valid integers."}), 400
            
        if km_driven < 0:
            return jsonify({"success": False, "error": "Kilometers driven cannot be negative."}), 400
            
        if year < 1990 or year > 2026:
            return jsonify({"success": False, "error": "Manufacturing year must be between 1990 and 2026."}), 400
            
        features_df = pd.DataFrame([{
            'brand': brand,
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner
        }])
        
        raw_prediction = float(PIPELINE.predict(features_df)[0])
        predicted_price = max(15000.0, raw_prediction)
        formatted_full, formatted_lakh = format_inr(predicted_price)
        
        return jsonify({
            "success": True,
            "price_raw": round(predicted_price, 2),
            "price_formatted": formatted_full,
            "price_lakhs": formatted_lakh,
            "summary": {
                "brand": brand,
                "year": year,
                "km_driven": f"{km_driven:,} km",
                "fuel": fuel,
                "transmission": transmission,
                "seller_type": seller_type,
                "owner": owner
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Valuation calculation error: {str(e)}"}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    API endpoint for What-If Price Simulator.
    Compares baseline vs modified vehicle configurations and generates a real 15-point sensitivity curve.
    Expects JSON: { baseline: {...}, modified: {...}, curve_type: "km" | "year" }
    """
    if PIPELINE is None:
        return jsonify({"success": False, "error": "Model pipeline is not loaded on server."}), 500
        
    try:
        data = request.get_json(force=True)
        base = data.get("baseline", {})
        mod = data.get("modified", {})
        curve_type = str(data.get("curve_type", "km")).lower()
        
        # Validate base & mod
        for obj, name in [(base, "baseline"), (mod, "modified")]:
            if not obj.get("brand") or not obj.get("fuel") or not obj.get("transmission"):
                return jsonify({"success": False, "error": f"Missing required fields in {name} configuration."}), 400
            try:
                obj["year"] = int(obj.get("year", 2017))
                obj["km_driven"] = max(0, int(obj.get("km_driven", 45000)))
            except (ValueError, TypeError):
                return jsonify({"success": False, "error": f"Invalid year or mileage in {name} configuration."}), 400
                
        # 1. Real predictions for baseline and modified
        df_base = pd.DataFrame([{
            'brand': base['brand'],
            'year': base['year'],
            'km_driven': base['km_driven'],
            'fuel': base['fuel'],
            'seller_type': base['seller_type'],
            'transmission': base['transmission'],
            'owner': base['owner']
        }])
        
        df_mod = pd.DataFrame([{
            'brand': mod['brand'],
            'year': mod['year'],
            'km_driven': mod['km_driven'],
            'fuel': mod['fuel'],
            'seller_type': mod['seller_type'],
            'transmission': mod['transmission'],
            'owner': mod['owner']
        }])
        
        pred_base = max(15000.0, float(PIPELINE.predict(df_base)[0]))
        pred_mod = max(15000.0, float(PIPELINE.predict(df_mod)[0]))
        
        delta_raw = pred_mod - pred_base
        delta_pct = (delta_raw / pred_base) * 100 if pred_base > 0 else 0.0
        
        base_full, base_lakh = format_inr(pred_base)
        mod_full, mod_lakh = format_inr(pred_mod)
        delta_full, delta_lakh = format_inr(abs(delta_raw))
        
        delta_state = "up" if delta_raw > 500 else ("down" if delta_raw < -500 else "unchanged")
        
        # 2. Generate 15-point Sensitivity Curve using real model predictions
        curve_points = []
        if curve_type == "year":
            year_range = list(range(2005, 2025))  # 20 points
            batch_list = []
            for y_val in year_range:
                row = mod.copy()
                row['year'] = y_val
                batch_list.append(row)
            batch_df = pd.DataFrame(batch_list)
            batch_preds = PIPELINE.predict(batch_df)
            for y_val, p_val in zip(year_range, batch_preds):
                curve_points.append({
                    'x': y_val,
                    'y': round(max(15000.0, float(p_val)) / 100000, 2)
                })
        else: # "km" mileage curve
            km_range = [int(k) for k in np.linspace(2000, 160000, 16)]
            batch_list = []
            for k_val in km_range:
                row = mod.copy()
                row['km_driven'] = k_val
                batch_list.append(row)
            batch_df = pd.DataFrame(batch_list)
            batch_preds = PIPELINE.predict(batch_df)
            for k_val, p_val in zip(km_range, batch_preds):
                curve_points.append({
                    'x': k_val,
                    'y': round(max(15000.0, float(p_val)) / 100000, 2)
                })
                
        return jsonify({
            "success": True,
            "baseline": {
                "price_raw": round(pred_base, 2),
                "price_formatted": base_full,
                "price_lakhs": base_lakh,
                "specs": base
            },
            "modified": {
                "price_raw": round(pred_mod, 2),
                "price_formatted": mod_full,
                "price_lakhs": mod_lakh,
                "specs": mod
            },
            "delta": {
                "raw": round(delta_raw, 2),
                "percentage": round(delta_pct, 1),
                "state": delta_state,
                "formatted": ("+" if delta_raw > 0 else ("-" if delta_raw < 0 else "")) + delta_full,
                "lakhs": ("+" if delta_raw > 0 else ("-" if delta_raw < 0 else "")) + delta_lakh
            },
            "curve": {
                "type": curve_type,
                "points": curve_points,
                "current_x": mod['year'] if curve_type == "year" else mod['km_driven'],
                "current_y": round(pred_mod / 100000, 2),
                "baseline_x": base['year'] if curve_type == "year" else base['km_driven'],
                "baseline_y": round(pred_base / 100000, 2)
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Simulation error: {str(e)}"}), 500


@app.route("/api/explain-prediction", methods=["POST"])
def api_explain_prediction():
    """
    API endpoint to compute feature sensitivity / contribution for an individual vehicle
    relative to the dataset reference vehicle (median/mode benchmark).
    """
    if PIPELINE is None:
        return jsonify({"success": False, "error": "Model pipeline is not loaded on server."}), 500
        
    try:
        data = request.get_json(force=True)
        car = data.get("vehicle", {})
        
        # Verify car fields
        for f in ['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']:
            if f not in car:
                return jsonify({"success": False, "error": f"Missing field {f} in vehicle."}), 400
                
        car['year'] = int(car['year'])
        car['km_driven'] = max(0, int(car['km_driven']))
        
        df_car = pd.DataFrame([car])
        df_ref = pd.DataFrame([REFERENCE_VEHICLE])
        
        pred_car = max(15000.0, float(PIPELINE.predict(df_car)[0]))
        pred_ref = max(15000.0, float(PIPELINE.predict(df_ref)[0]))
        
        contributions = []
        labels = {
            'brand': 'Manufacturer Brand',
            'transmission': 'Transmission Gearbox',
            'year': 'Manufacturing Year',
            'km_driven': 'Odometer Mileage',
            'fuel': 'Fuel Mechanism',
            'seller_type': 'Seller Channel',
            'owner': 'Ownership Tier'
        }
        
        # Feature ablation vs reference
        for feat, label in labels.items():
            ablated = car.copy()
            ablated[feat] = REFERENCE_VEHICLE[feat]
            pred_ablated = max(15000.0, float(PIPELINE.predict(pd.DataFrame([ablated]))[0]))
            
            # Impact: how much this car's feature value altered price vs reference value
            impact = pred_car - pred_ablated
            contributions.append({
                'feature': feat,
                'label': label,
                'vehicle_val': str(car[feat]),
                'reference_val': str(REFERENCE_VEHICLE[feat]),
                'impact_raw': round(impact, 2),
                'impact_lakhs': round(impact / 100000, 2),
                'direction': 'positive' if impact > 200 else ('negative' if impact < -200 else 'neutral')
            })
            
        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x['impact_raw']), reverse=True)
        
        p_car_full, p_car_lakh = format_inr(pred_car)
        p_ref_full, p_ref_lakh = format_inr(pred_ref)
        
        return jsonify({
            "success": True,
            "vehicle_price_full": p_car_full,
            "vehicle_price_lakh": p_car_lakh,
            "reference_price_full": p_ref_full,
            "reference_price_lakh": p_ref_lakh,
            "reference_vehicle": REFERENCE_VEHICLE,
            "contributions": contributions
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Explainability error: {str(e)}"}), 500


@app.route("/api/explorer/filter", methods=["POST"])
def api_explorer_filter():
    """
    API endpoint to filter dataset rows dynamically for the Data Explorer page.
    """
    if DF_DATA is None:
        return jsonify({"success": False, "records": [], "count": 0}), 200
        
    try:
        data = request.get_json(force=True) or {}
        brand = data.get("brand", "")
        fuel = data.get("fuel", "")
        transmission = data.get("transmission", "")
        year_min = int(data.get("year_min", 1992))
        year_max = int(data.get("year_max", 2026))
        max_price_lakh = float(data.get("max_price_lakh", 90))
        search_query = str(data.get("search", "")).strip().lower()
        
        filtered = DF_DATA.copy()
        
        if brand and brand != "All":
            filtered = filtered[filtered['brand'] == brand]
            
        if fuel and fuel != "All":
            filtered = filtered[filtered['fuel'] == fuel]
            
        if transmission and transmission != "All":
            filtered = filtered[filtered['transmission'] == transmission]
            
        filtered = filtered[(filtered['year'] >= year_min) & (filtered['year'] <= year_max)]
        filtered = filtered[filtered['selling_price'] <= max_price_lakh * 100000]
        
        if search_query:
            filtered = filtered[
                filtered['name'].str.lower().str.contains(search_query, na=False) |
                filtered['brand'].str.lower().str.contains(search_query, na=False)
            ]
            
        total_matches = len(filtered)
        
        records = []
        for _, row in filtered.head(25).iterrows():
            records.append({
                'name': row.get('name', ''),
                'brand': row.get('brand', ''),
                'year': int(row.get('year', 0)),
                'selling_price_fmt': f"₹ {row.get('selling_price', 0) / 100000:.2f} L",
                'km_driven_fmt': f"{int(row.get('km_driven', 0)):,} km",
                'fuel': row.get('fuel', ''),
                'transmission': row.get('transmission', ''),
                'owner': row.get('owner', '')
            })
            
        return jsonify({
            "success": True,
            "count": total_matches,
            "records": records
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ---------------------------------------------------------------------------
# MAIN EXECUTION ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[*] Starting Carlytics Flask Application...")
    print(f"[*] Pipeline loaded: {'Yes' if PIPELINE is not None else 'No'}")
    print(f"[*] Dataset records: {len(DF_DATA) if DF_DATA is not None else 0}")
    port = int(os.environ.get("PORT", 5001))
    print(f"[*] Access the web application at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
