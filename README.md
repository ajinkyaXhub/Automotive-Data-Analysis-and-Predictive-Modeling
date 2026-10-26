# 🚗 Carlytics — Machine Learning & Automotive Data Analytics

An end-to-end Machine Learning mini-project and enterprise automotive analytics SaaS application designed for college academic evaluation and industrial demonstration. This application cleans, explores, trains, and evaluates regression algorithms to predict estimated resale prices of pre-owned vehicles in Indian Rupees based on historical CarDekho secondary market transactions.

---

## 📌 Project Overview

Predicting used car valuations is challenging because vehicle prices depreciate non-linearly across age, brand prestige, mileage, and fuel type. **Carlytics** implements an end-to-end machine learning system with advanced interpretability and simulation capabilities:

1. **Exploratory Data Analysis & Cleaning:** Inspects the raw dataset, eliminates duplicate entries, extracts manufacturer brand badges, and validates column schemas.
2. **Preprocessing Pipeline:** Employs Scikit-Learn's `ColumnTransformer` with `StandardScaler` on continuous features and `OneHotEncoder(handle_unknown='ignore')` on categorical columns.
3. **Model Benchmarking:** Trains and benchmarks **Linear Regression** vs. **Random Forest Regressor** on identical 80/20 train-test partitions.
4. **Honest Evaluation:** Assesses performance on unseen test records using **MAE**, **RMSE**, and **R² Score**, saving the winning pipeline (`car_price_pipeline.joblib`).
5. **Interactive What-If Price Simulator (`/simulator`):** Live side-by-side scenario comparison, debounced range sliders, dynamic price-delta badges, and real 16-point sensitivity curves.
6. **Model Explainability & XAI (`/explainability`):** Global Mean Decrease in Impurity (MDI) feature importances and local marginal feature ablation diagnostics relative to a population median reference car.
7. **Production SaaS Dashboard:** Full-stack Flask web application with a modern deep navy and neutral canvas design system, responsive Plotly charts, and micro-interactions.

---

## 📂 Project Structure

```text
ML MINIPROJECT/
│
├── car data.csv                        # Cleaned secondary market vehicle dataset (3,577 records)
├── train_model.py                      # Training pipeline, benchmark evaluation, and artifact exporter
├── flask_app.py                        # Production Flask web application backend and REST endpoints
├── app.py                              # Preserved Streamlit web application dashboard
├── templates/                          # Semantic Jinja2 HTML5 templates
│   ├── base.html                       # Base layout with deep navy sidebar, navigation, and topbar
│   ├── overview.html                   # High-level KPI metrics, distribution, and depreciation charts
│   ├── estimator.html                  # Vehicle price estimator with quick benchmark presets
│   ├── simulator.html                  # FEATURE ONE: Interactive What-If Price Simulator
│   ├── explainability.html             # FEATURE TWO: Model Explainability (MDI & Local Ablation)
│   ├── explorer.html                   # Interactive dataset explorer with dynamic multi-filters
│   └── performance.html                # Model evaluation metrics, comparison tables, and diagnostics
├── static/                             # Frontend static assets
│   ├── css/
│   │   └── style.css                   # Premium automotive analytics SaaS design system
│   └── js/
│       ├── main.js                     # Global micro-interactions and count-up number animations
│       ├── simulator.js                # What-If Simulator client controller and Plotly sensitivity curve
│       └── explainability.js           # Global MDI chart toggle and local feature ablation rendering
├── car_price_pipeline.joblib           # Trained Scikit-Learn Random Forest pipeline artifact
├── model_metrics.json                  # Saved evaluation scores & feature categories
├── requirements.txt                    # Project Python dependencies
└── README.md                           # Comprehensive documentation and viva guide
```

---

## 📊 Dataset Schema & Columns Identified

The dataset consists of **4,340 raw records** (reduced to **3,577 records** after eliminating 763 duplicate entries):

| Column Name | Data Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `name` | String | Full vehicle name (make, model, trim) | `Maruti 800 AC`, `Hyundai Verna 1.6 SX` |
| `year` | Integer | Year of vehicle manufacture/registration | `2007`, `2012`, `2017` |
| `selling_price` | Integer | **Target Variable**: Resale price in INR | `₹ 60,000`, `₹ 6,00,000` |
| `km_driven` | Integer | Total distance recorded on odometer | `25,000`, `70,000`, `1,25,000` |
| `fuel` | Categorical | Type of fuel | `Petrol`, `Diesel`, `CNG`, `LPG`, `Electric` |
| `seller_type` | Categorical | Channel of sale | `Individual`, `Dealer`, `Trustmark Dealer` |
| `transmission` | Categorical | Gearbox mechanism | `Manual`, `Automatic` |
| `owner` | Categorical | Previous ownership tier | `First Owner`, `Second Owner`, `Third Owner`, etc. |

> **Feature Engineering:** The first word of the `name` column is extracted as `brand` (e.g., Maruti, Hyundai, Toyota, Honda), reducing 1,491 sparse model names into 29 distinct vehicle manufacturers.

---

## 📈 Model Comparison & Evaluation Results

Both models were trained on 80% of data (2,861 samples) and evaluated on an **unseen 20% test partition (716 samples)** using fixed `random_state=42`:

| Metric | Linear Regression | Random Forest Regressor | Winner |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | ₹ 1,79,641.77 | **₹ 1,59,985.64** | 🏆 **Random Forest** (lower error by ₹ 19.6k) |
| **Root Mean Squared Error (RMSE)** | ₹ 3,84,722.70 | **₹ 3,68,712.62** | 🏆 **Random Forest** (lower error by ₹ 16.0k) |
| **R² Score (Goodness of Fit)** | 0.5405 (54.05%) | **0.5780 (57.80%)** | 🏆 **Random Forest** (explains +3.75% more variance) |

### Why Random Forest Outperformed Linear Regression:
- Used car depreciation is **exponential and non-linear**; cars lose substantial value within the first 3–5 years and flatten out afterward.
- Certain brands (e.g., Toyota, BMW, Audi) maintain non-linear brand tier premiums that tree ensembles isolate through recursive partitioning.
- Interactions between features (such as Diesel + Automatic or Dealer + First Owner) are naturally captured by ensemble decision trees without requiring manual polynomial feature engineering.

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.9+ (Python 3.10, 3.11, or 3.12 recommended)

### 1. Clone or Open Workspace
Open your terminal in the project directory:
```bash
cd "c:\Users\DELL\Desktop\Projects\ML MINIPROJECT"
```

### 2. Activate or Create Virtual Environment
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt (CMD)
.\.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```
*(If you need to create a fresh virtual environment: `python -m venv .venv`)*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models
Run the training pipeline to clean data, train models, compare evaluation metrics, and export the winning pipeline:
```bash
python train_model.py
```
This generates:
- `car_price_pipeline.joblib` (Trained Scikit-Learn pipeline)
- `model_metrics.json` (Evaluation results and category metadata)

### 5. Launch the Custom Flask Web Application (Recommended)
```bash
python flask_app.py
```
Access the custom automotive valuation SaaS dashboard in your browser at:
`http://127.0.0.1:5001`

### 6. Launch the Streamlit Version (Preserved)
```bash
streamlit run app.py
```
Access the Streamlit application at:
`http://localhost:8501`

---

## 🖥️ Web Application Dashboard Modules

The **Carlytics** web application delivers a full-stack automotive analytics SaaS dashboard (`flask_app.py`), styled with a refined deep navy sidebar, clean white cards, subtle shadows, and interactive Plotly visual analytics:

1. **📊 Overview Dashboard (`/overview`):**
   - 4 high-level metric cards: Dataset Size (3,577 records), Production Pipeline (Random Forest), Test MAE (₹ 1,59,986), and R² Score (0.5780).
   - Quick-launch feature showcases for What-If Simulator and Model Explainability.
   - Market distribution histogram of selling prices with interactive tooltips.
   - Depreciation curve displaying median valuation against vehicle age.
   - Powertrain & gearbox volume breakdown chart.
   - Real-time Session Valuation Log tracking predictions made during the session.

2. **🎯 Car Price Estimator (`/estimator`):**
   - Structured 4-group inputs: Vehicle Identity, Odometer & Usage, Powertrain & Gearbox, and Ownership & Channel.
   - 1-click Benchmark presets (*Maruti Swift '17, Hyundai Creta '19, Toyota Fortuner '18, Honda City '15*) and clean input reset.
   - High-contrast Hero Valuation card formatted in Indian Rupees (`₹ 5,13,474` / `Approx. ₹ 5.13 Lakhs`).
   - 1-click **"Simulate What-If Scenarios for this Vehicle ⚡"** direct seamless handover to the simulator.
   - Clear disclaimer clarifying ML-generated estimation boundaries.

3. **⚡ Interactive What-If Price Simulator (`/simulator`) — FEATURE ONE:**
   - **Side-by-Side Comparison:** Baseline vehicle benchmark vs simulated configuration with live real-time price updates.
   - **Dynamic Delta Indicator:** Animated price-difference pill with directional arrows (▲ / ▼ / —), absolute difference (`+₹ 45,000`), and percentage variance (`+8.4%`).
   - **Responsive Control Deck:** Debounced continuous sliders (250ms) for Manufacturing Year and Odometer Mileage, plus categorical dropdowns.
   - **Quick Preset Scenarios:** 1-click scenarios (*Low Mileage -30k km, High Mileage +50k km, Newer Model +3 Yrs, Older Model -4 Yrs, Automatic Upgrade, Diesel Powertrain*).
   - **Interactive Sensitivity Curve:** Plotly spline curve evaluated by batch model prediction with toggle between *Price vs. Mileage* and *Price vs. Year*, complete with active marker and baseline reference point.
   - **ML Integrity Guarantee:** Zero hardcoded heuristics; all predictions run synchronously through `car_price_pipeline.joblib`.

4. **🧭 Model Explainability & XAI (`/explainability`) — FEATURE TWO:**
   - **Global Feature Importance (MDI):** Extracted directly from `rf.feature_importances_` and `preprocessor.get_feature_names_out()`.
   - **Dual Perspective Toggle:** Switch between 7 aggregated original feature groups and the top 12 individual transformed one-hot columns.
   - **Local Prediction Feature Sensitivity:** Marginal feature ablation engine calculating exact ₹ Lakhs valuation impacts relative to the population reference car (*Maruti 2014, 60k km, Diesel, Manual*).
   - **Diverging Waterfall / Contribution Chart:** Visual green/red bars showing positive and negative marginal contributions.
   - **Attribution Diagnostics Table:** Line-by-line comparison between candidate vehicle values and reference values.
   - **Academic Integrity Guardrails:** Distinguishes Gini impurity reduction from real-world causation, highlights MDI cardinality nuances, and clarifies $R^2$ as explained variance.

5. **🔍 Data Explorer (`/explorer`):**
   - Multi-dimensional filters (Manufacturer, Fuel, Gearbox, Max Price in Lakhs, Model Search).
   - Dynamic real-time table updates via `/api/explorer/filter`.
   - Interactive boxplots preserving real outlier valuations.
   - Mileage vs. Price scatter plot with fuel-type color coding.

6. **📈 Model Performance & Diagnostics (`/performance`):**
   - Direct comparison between Linear Regression and Random Forest using exact test set figures.
   - Graphical error comparison bar charts.
   - **Actual vs. Predicted Valuation Scatter Plot** generated on 716 unseen test records with an ideal $y = x$ reference diagonal.
   - Rigorous explanation of MAE, RMSE, and $R^2$.

---

## 🛡️ Input Validation & Robustness

- **Kilometers Driven:** Protected against negative values; displays a high-mileage advisory for readings > 6,00,000 km.
- **Year of Manufacture:** Constrained to realistic automotive timelines (1992 to current year); rejects future dates.
- **Missing / Unknown Categories:** The Scikit-Learn `OneHotEncoder` is configured with `handle_unknown='ignore'`, preventing runtime crashes if new or unseen categories are encountered.
- **Price Bounds:** Guarantees predicted prices do not dip below vehicle scrap salvage value (~ ₹ 15,000).

---

## 🎓 College Viva / Q&A Guide

**Q1: What is the target variable and is this a Classification or Regression problem?**  
*Answer:* The target variable is `selling_price` (in Indian Rupees), which is a continuous numerical quantity. Therefore, this is a **Regression** problem.

**Q2: What is the difference between MAE and RMSE?**  
*Answer:* 
- **MAE (Mean Absolute Error):** Takes the average of absolute differences between predictions and actual values. It is easy to interpret in Rupees (average error of ~₹1.60 Lakhs).
- **RMSE (Root Mean Squared Error):** Squares the errors before averaging and taking the root, thereby penalizing larger outlier prediction errors much more severely.

**Q3: What is R² score?**  
*Answer:* R² (Coefficient of Determination) represents the proportion of variance in the dependent variable that is predictable from the independent features. An R² of 0.578 means our model explains ~57.8% of the variability in car resale prices.

**Q4: Why was Random Forest chosen over Linear Regression?**  
*Answer:* Random Forest is an ensemble of decision trees that can model complex non-linear patterns (such as exponential depreciation over years and luxury brand premiums) without assuming a straight-line relationship.

---

## 📜 License & Acknowledgments
- Dataset source: CarDekho via Kaggle.
- Built for academic and demonstration purposes.
