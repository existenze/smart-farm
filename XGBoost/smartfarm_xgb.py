#!/usr/bin/env python3
"""
smartfarm_xgb.py

Complete XGBoost training + minimal Flask API for crop yield prediction.

Compatible with XGBoost 3.1.1 (Booster + DMatrix API).

Usage:
    1. Place your Kaggle CSV file in the same folder and name it Smart_Farming_Crop_Yield_2024.csv
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
import xgboost as xgb

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# ---------- CONFIG ----------
DATA_PATH = Path(__file__).parent / "Smart_Farming_Crop_Yield_2024.csv"
MODEL_OUT = "xgb_smartfarm_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIME_SPLIT = True
TARGET_COL = "yield_kg_per_hectare"
DATE_COL = "date"
FARM_ID_COL = "farm_id"
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
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    df = df[~df[TARGET_COL].isna()].copy()

    # Rolling features
    rolling_features = []
    if FARM_ID_COL in df.columns and DATE_COL in df.columns:
        df = df.sort_values([FARM_ID_COL, DATE_COL])
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

    # One-hot encoding categorical columns
    categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c not in [FARM_ID_COL, DATE_COL]]
    if len(categorical_cols) > 0:
        print("One-hot encoding:", categorical_cols)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[categorical_cols].fillna("NA"))
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    # Impute numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df

def choose_features(df: pd.DataFrame):
    exclude = {TARGET_COL, DATE_COL, FARM_ID_COL}
    return [c for c in df.columns if c not in exclude]

def split_data(df: pd.DataFrame, features):
    X = df[features]
    y = df[TARGET_COL]

    if TIME_SPLIT and DATE_COL in df.columns:
        df_sorted = df.sort_values(DATE_COL)
        split_idx = int(len(df_sorted) * (1 - TEST_SIZE))
        X_train = df_sorted.iloc[:split_idx][features]
        y_train = df_sorted.iloc[:split_idx][TARGET_COL]
        X_test = df_sorted.iloc[split_idx:][features]
        y_test = df_sorted.iloc[split_idx:][TARGET_COL]
        print(f"Time-based split: train={len(X_train)} test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f"Random split: train={len(X_train)} test={len(X_test)}")
        return X_train, X_test, y_train, y_test

def train_xgboost(X_train, y_train, X_val, y_val):
    print("\nTraining XGBoost with early stopping (3.x compatible)...")
    # Make sure feature names are carried through
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=list(X_val.columns))

    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse",
        "seed": RANDOM_STATE
    }

    evals = [(dtrain, "train"), (dval, "eval")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    print("Training completed.")
    return model

def evaluate_model(model, X_test, y_test, show_plots=True):
    dtest = xgb.DMatrix(X_test)
    preds = model.predict(dtest)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
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
    """
    Works whether get_score() returns 'f0','f1',... or real feature names.
    """
    raw = model.get_score(importance_type="weight")
    if not raw:
        print("No feature importances available from model.")
        return

    def _is_index_key(k: str) -> bool:
        return k.startswith("f") and k[1:].isdigit()

    if all(_is_index_key(k) for k in raw.keys()):
        # Map f{idx} -> feature_names[idx]
        imp_items = [(feature_names[int(k[1:])], v) for k, v in raw.items()]
    else:
        # Keys are already names
        imp_items = list(raw.items())

    top_features = sorted(imp_items, key=lambda x: x[1], reverse=True)[:top_n]
    if not top_features:
        print("No top features to plot.")
        return

    names, scores = zip(*top_features)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(scores))[::-1], scores)
    plt.yticks(range(len(scores))[::-1], names)
    plt.xlabel("Feature importance (weight)")
    plt.title("Top feature importances (XGBoost)")
    plt.tight_layout()
    plt.show()


def shap_explain(model, X_sample, feature_names):
    print("\nComputing SHAP values for model interpretability (may take time)...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=True)

def create_flask_app(model, feature_names):
    app = Flask("smartfarm_predictor")

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(force=True)
        features = payload.get("features", {})
        x = np.array([features.get(fn, 0.0) for fn in feature_names]).reshape(1, -1)
        dmatrix = xgb.DMatrix(x)
        pred = model.predict(dmatrix)[0]
        return jsonify({"predicted_yield": float(pred)})

    return app

def main(args):
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Please download the Kaggle CSV and place it as {DATA_PATH} (or update DATA_PATH).")
    df = load_data(DATA_PATH)
    quick_inspect(df)

    df = basic_preprocess(df)
    print("After preprocessing shape:", df.shape)

    features = choose_features(df)
    print("Candidate feature count:", len(features))
    print("Example features:", features[:20])

    X_train, X_test, y_train, y_test = split_data(df, features)

    model = train_xgboost(X_train, y_train, X_test, y_test)

    metrics = evaluate_model(model, X_test, y_test, show_plots=not args.no_plots)

    if not args.no_plots:
        plot_feature_importances(model, features, top_n=20)

    if not args.no_plots and args.shap:
        sample_idx = np.random.choice(len(X_test), size=min(200, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_idx]
        shap_explain(model, X_sample, features)

    print(f"Saving model to {MODEL_OUT} ...")
    joblib.dump({"model": model, "features": features}, MODEL_OUT)

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





