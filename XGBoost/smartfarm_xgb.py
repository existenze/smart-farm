#!/usr/bin/env python3
"""
smartfarm_xgb.py

Complete XGBoost training + minimal Flask API for crop yield prediction.

Usage:
    1. Place your Kaggle CSV file in the same folder and name it smart_farm_data.csv
       or change DATA_PATH below.
    2. pip install -r requirements.txt
       (requirements: pandas, numpy, scikit-learn, xgboost, shap, flask, joblib, matplotlib)
    3. python smartfarm_xgb.py        # trains model and saves model.joblib, prints metrics
    4. python smartfarm_xgb.py --serve  # runs a small Flask API for predictions
"""

import argparse
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, request
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# ---------- CONFIG ----------
DATA_PATH = r"C:\Users\brand\OneDrive\Desktop\COMPE510\Smart_Farming_Crop_Yield_2024.csv"
MODEL_OUT = "xgb_smartfarm_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2   # used for random split (not time-split)
TIME_SPLIT = True  # if True, do a time-based split (recommended for time-series)
TARGET_COL = "yield"  # change to the actual target column name in your CSV
DATE_COL = "date"     # change if your dataset has a different datetime column
FARM_ID_COL = "farm_id"  # optional: unique farm identifier if present

# Columns we expect to exist (example). Adapt to your dataset.
EXPECTED_COLUMNS = [
    TARGET_COL, DATE_COL, FARM_ID_COL,
    "soil_moisture", "soil_ph", "temperature", "humidity", "rainfall",
    "ndvi", "evi", "satellite_cloud_fraction"
]
# ----------------------------

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path, parse_dates=[DATE_COL] if DATE_COL in pd.read_csv(path, nrows=0).columns else [])
    print("Initial shape:", df.shape)
    return df

def quick_inspect(df: pd.DataFrame):
    print("\n--- Data sample ---")
    print(df.head())
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Missing values (per column) ---")
    print(df.isna().sum().sort_values(ascending=False).head(20))

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure date column is datetime
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Example: drop rows with no target
    df = df[~df[TARGET_COL].isna()].copy()

    # Feature engineering: rolling/windowed features per farm (if farm_id exists)
    rolling_features = []
    if FARM_ID_COL in df.columns and DATE_COL in df.columns:
        df = df.sort_values([FARM_ID_COL, DATE_COL])
        # Example rolling means for soil_moisture, rainfall, ndvi
        for feat in ["soil_moisture", "rainfall", "ndvi"]:
            if feat in df.columns:
                col_name = f"{feat}_rolling_3"
                df[col_name] = df.groupby(FARM_ID_COL)[feat].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
                rolling_features.append(col_name)

    # Time features
    if DATE_COL in df.columns:
        df["month"] = df[DATE_COL].dt.month
        df["dayofyear"] = df[DATE_COL].dt.dayofyear
        df["year"] = df[DATE_COL].dt.year

    # Example categorical handling: if you have crop_type, one-hot encode
    categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c not in [FARM_ID_COL, DATE_COL]]
    if len(categorical_cols) > 0:
        print("One-hot encoding:", categorical_cols)
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[categorical_cols].fillna("NA"))
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    # Imputation for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df

def choose_features(df: pd.DataFrame):
    # Construct feature list automatically but exclude identifiers & target
    exclude = {TARGET_COL, DATE_COL, FARM_ID_COL}
    features = [c for c in df.columns if c not in exclude]
    return features

def split_data(df: pd.DataFrame, features):
    X = df[features]
    y = df[TARGET_COL].values

    if TIME_SPLIT and DATE_COL in df.columns:
        # time-based split: keep latest TEST_SIZE fraction as test
        df_sorted = df.sort_values(DATE_COL)
        split_idx = int(len(df_sorted) * (1 - TEST_SIZE))
        X_train = df_sorted.iloc[:split_idx][features].values
        y_train = df_sorted.iloc[:split_idx][TARGET_COL].values
        X_test = df_sorted.iloc[split_idx:][features].values
        y_test = df_sorted.iloc[split_idx:][TARGET_COL].values
        # For CV we can use TimeSeriesSplit later
        print(f"Time-based split: train={len(X_train)} test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f"Random split: train={len(X_train)} test={len(X_test)}")
        return X_train, X_test, y_train, y_test

def train_xgboost(X_train, y_train, X_val, y_val):
    print("\nTraining XGBoost with basic parameters + early stopping ...")
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Fit with early stopping on validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=50
    )
    return model

def hyperparameter_tune(X_train, y_train):
    print("\nRunning RandomizedSearchCV for XGBoost hyperparameters (this may take a while)...")
    param_dist = {
        "n_estimators": [100, 300, 500, 800],
        "max_depth": [3, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 1.5, 2, 3]
    }
    xgb = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1)
    # If time-series, use TimeSeriesSplit for CV
    cv = TimeSeriesSplit(n_splits=3) if TIME_SPLIT else 3
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=30,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=2
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_

def evaluate_model(model, X_test, y_test, show_plots=True):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print("\n--- Evaluation on Test Set ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    if show_plots:
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, preds, alpha=0.4)
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title("Actual vs Predicted Yield")
        lims = [min(min(y_test), min(preds)), max(max(y_test), max(preds))]
        plt.plot(lims, lims, 'r--')
        plt.grid(True)
        plt.show()

    return {"mae": mae, "rmse": rmse, "r2": r2, "preds": preds}

def plot_feature_importances(model, feature_names, top_n=20):
    # XGBoost has feature_importances_ attribute
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8,6))
    plt.barh(range(len(indices))[::-1], importances[indices])
    plt.yticks(range(len(indices))[::-1], [feature_names[i] for i in indices])
    plt.xlabel("Feature importance")
    plt.title("Top feature importances (XGBoost)")
    plt.tight_layout()
    plt.show()

def shap_explain(model, X_sample, feature_names):
    print("\nComputing SHAP values for model interpretability (may take time)...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=True)

# Minimal Flask app for predictions
def create_flask_app(model, feature_names):
    app = Flask("smartfarm_predictor")

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Expects JSON payload:
        {
          "features": {"soil_moisture": 20, "temperature": 30, ...}
        }
        """
        payload = request.get_json(force=True)
        features = payload.get("features", {})
        # Build feature vector in the same order as feature_names
        x = np.array([features.get(fn, 0.0) for fn in feature_names]).reshape(1, -1)
        pred = model.predict(x)[0]
        return jsonify({"predicted_yield": float(pred)})
    return app

def main(args):
    # 1) Load
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Please download the Kaggle CSV and place it as {DATA_PATH} (or update DATA_PATH).")
    df = load_data(DATA_PATH)
    quick_inspect(df)

    # 2) Preprocess & augment features
    df = basic_preprocess(df)
    print("After preprocessing shape:", df.shape)

    # 3) Pick features
    features = choose_features(df)
    print("Candidate feature count:", len(features))
    print("Example features:", features[:20])

    # 4) Split
    X_train, X_test, y_train, y_test = split_data(df, features)

    # 5) Option A: quick train with early stopping
    model = train_xgboost(X_train, y_train, X_test, y_test)

    # 6) Option B (optional but recommended): hyperparameter tuning
    if args.tune:
        print("\nStarting hyperparameter tuning (RandomizedSearchCV). This can take a long time.")
        # For speed, tune on a subset or on train only
        model = hyperparameter_tune(X_train, y_train)

    # 7) Final eval
    metrics = evaluate_model(model, X_test, y_test, show_plots=not args.no_plots)

    # 8) Feature importance plot
    if not args.no_plots:
        plot_feature_importances(model, features, top_n=20)

    # 9) SHAP explainability on a sample
    if not args.no_plots and args.shap:
        # use a small sample for SHAP
        sample_idx = np.random.choice(len(X_test), size=min(200, len(X_test)), replace=False)
        X_sample = pd.DataFrame(X_test[sample_idx], columns=features)
        shap_explain(model, X_sample, features)

    # 10) Save model and feature list
    print(f"Saving model to {MODEL_OUT} ...")
    joblib.dump({"model": model, "features": features}, MODEL_OUT)

    # 11) Optionally serve
    if args.serve:
        app = create_flask_app(model, features)
        print("Starting Flask app on http://127.0.0.1:5000 ...")
        app.run(debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an XGBoost model for SmartFarm yield prediction.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (RandomizedSearchCV).")
    parser.add_argument("--serve", action="store_true", help="Start a small Flask API after training.")
    parser.add_argument("--no-plots", action="store_true", help="Don't show plots (useful when running headless).")
    parser.add_argument("--shap", action="store_true", help="Run SHAP explainability after training (can be slow).")
    args = parser.parse_args()
    main(args)
