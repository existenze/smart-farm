#!/usr/bin/env python3
"""
smartfarm_svr.py

Train an SVR (Support Vector Regression) model on the Smart Farming crop yield dataset.

Usage:
    1. Put Smart_Farming_Crop_Yield_2024.csv in the same folder (or update DATA_PATH).
    2. pip install -r requirements.txt
       (needs: pandas, numpy, scikit-learn, matplotlib, joblib, flask)
    3. python smartfarm_svr.py              # trains SVR and prints metrics
    4. python smartfarm_svr.py --tune       # runs a small RandomizedSearch on SVR
    5. python smartfarm_svr.py --serve      # starts a minimal Flask API for predictions
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# ---------- CONFIG ----------
DATA_PATH = "Smart_Farming_Crop_Yield_2024.csv"
MODEL_OUT = "svr_smartfarm_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIME_SPLIT = True
TARGET_COL = "yield_kg_per_hectare"
DATE_COL = "date"
FARM_ID_COL = "farm_id"
# ----------------------------


# ----------------- DATA LOADING / INSPECTION -----------------
def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    # Try to parse date column if present
    cols = pd.read_csv(path, nrows=0).columns
    parse_dates = [DATE_COL] if DATE_COL in cols else []
    df = pd.read_csv(path, parse_dates=parse_dates)
    print("Initial shape:", df.shape)
    return df


def quick_inspect(df: pd.DataFrame):
    print("\n--- Data sample ---")
    print(df.head())
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Missing values (top 20) ---")
    print(df.isna().sum().sort_values(ascending=False).head(20))


# ----------------- PREPROCESSING -----------------
def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature engineering + one-hot + median imputation, same style as XGB script."""
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Drop rows with missing target
    df = df[~df[TARGET_COL].isna()].copy()

    # Rolling features per farm
    rolling_features = []
    if FARM_ID_COL in df.columns and DATE_COL in df.columns:
        df = df.sort_values([FARM_ID_COL, DATE_COL])
        for feat in ["soil_moisture", "rainfall", "ndvi"]:
            if feat in df.columns:
                col_name = f"{feat}_rolling_3"
                df[col_name] = (
                    df.groupby(FARM_ID_COL)[feat]
                    .rolling(window=3, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                rolling_features.append(col_name)

    # Time-based features
    if DATE_COL in df.columns:
        df["month"] = df[DATE_COL].dt.month
        df["dayofyear"] = df[DATE_COL].dt.dayofyear
        df["year"] = df[DATE_COL].dt.year

    # One-hot encode categorical columns (except farm_id/date)
    categorical_cols = [
        c
        for c in df.columns
        if df[c].dtype == "object" and c not in [FARM_ID_COL, DATE_COL]
    ]
    if len(categorical_cols) > 0:
        print("One-hot encoding:", categorical_cols)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[categorical_cols].fillna("NA"))
        encoded_df = pd.DataFrame(
            encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index
        )
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    # Median impute numeric columns (except target)
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
        # Time-based split on date
        df_sorted = df.sort_values(DATE_COL)
        split_idx = int(len(df_sorted) * (1 - TEST_SIZE))
        X_train = df_sorted.iloc[:split_idx][features]
        y_train = df_sorted.iloc[:split_idx][TARGET_COL]
        X_test = df_sorted.iloc[split_idx:][features]
        y_test = df_sorted.iloc[split_idx:][TARGET_COL]
        print(f"Time-based split: train={len(X_train)} test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Random split: train={len(X_train)} test={len(X_test)}")
        return X_train, X_test, y_train, y_test


# ----------------- MODEL TRAINING -----------------
def build_svr_pipeline(C=10.0, epsilon=0.1, gamma="scale", kernel="rbf"):
    """
    StandardScaler + SVR pipeline.
    Scaling is important for SVR, so this pipeline keeps that bundled.
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)),
        ]
    )
    return pipe


def train_svr(X_train, y_train, tune=False):
    """
    Train an SVR model. If tune=True, run a small RandomizedSearchCV
    over C / gamma / epsilon.
    """
    base_pipe = build_svr_pipeline()

    if not tune:
        print("\nTraining SVR with default hyperparameters...")
        base_pipe.fit(X_train, y_train)
        return base_pipe

    print("\nRunning RandomizedSearchCV for SVR hyperparameters...")

    param_distributions = {
        "svr__C": np.logspace(-1, 3, 20),
        "svr__epsilon": np.logspace(-3, 0, 10),
        "svr__gamma": ["scale", "auto"]
        + list(np.logspace(-4, -1, 10)),
        "svr__kernel": ["rbf"],
    }

    if TIME_SPLIT:
        cv = TimeSeriesSplit(n_splits=5)
    else:
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_distributions,
        n_iter=30,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV RMSE:", np.sqrt(-search.best_score_))
    return search.best_estimator_


# ----------------- EVALUATION -----------------
def evaluate_model(model, X_test, y_test, show_plots=True):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n--- Evaluation on Test Set (SVR) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    if show_plots:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, preds, alpha=0.4)
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield (SVR)")
        plt.title("Actual vs Predicted Yield (SVR)")
        lims = [
            min(min(y_test), min(preds)),
            max(max(y_test), max(preds)),
        ]
        plt.plot(lims, lims, "r--")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {"mae": mae, "rmse": rmse, "r2": r2, "preds": preds}


# ----------------- FLASK API -----------------
def create_flask_app(model, feature_names):
    """
    Minimal API: expects a JSON with a 'features' dict mapping feature_name -> value.
    These should already correspond to the engineered features used during training.
    """
    app = Flask("smartfarm_svr_predictor")

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(force=True)
        features = payload.get("features", {})
        x = np.array([features.get(fn, 0.0) for fn in feature_names]).reshape(1, -1)
        pred = model.predict(x)[0]
        return jsonify({"predicted_yield": float(pred)})

    return app


# ----------------- MAIN ENTRY -----------------
def main(args):
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Please place the CSV in the same folder "
            f"or update DATA_PATH in smartfarm_svr.py."
        )

    df = load_data(DATA_PATH)
    quick_inspect(df)

    df = basic_preprocess(df)
    print("After preprocessing shape:", df.shape)

    features = choose_features(df)
    print("Number of features:", len(features))
    print("Example features:", features[:20])

    X_train, X_test, y_train, y_test = split_data(df, features)

    model = train_svr(X_train, y_train, tune=args.tune)

    metrics = evaluate_model(model, X_test, y_test, show_plots=not args.no_plots)

    print(f"\nSaving SVR model to {MODEL_OUT} ...")
    joblib.dump({"model": model, "features": features}, MODEL_OUT)
    print("Done.")

    if args.serve:
        app = create_flask_app(model, features)
        print("Starting Flask app on http://127.0.0.1:5000 ...")
        app.run(debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an SVR model for SmartFarm yield prediction."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (RandomizedSearchCV) for SVR.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a small Flask API after training.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Don't show plots (useful when running headless).",
    )
    args = parser.parse_args()
    main(args)
