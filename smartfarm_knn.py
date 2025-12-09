#!/usr/bin/env python3
"""
smartfarm_knn.py

Train a k-NN regressor for crop yield prediction and provide a simple Flask API for predictions.

Usage:
  1. Put your dataset CSV next to this script or update DATA_PATH below.
  2. Install dependencies:
       pip install pandas numpy scikit-learn flask joblib matplotlib
  3. Run training:
       python smartfarm_knn.py
     Optional flags:
       --tune      run GridSearchCV hyperparameter tuning
       --serve     start Flask API after training (serves the in-memory model)
       --no-plots  skip plotting (useful on headless servers)
  4. After run, model saved as knn_smartfarm_pipeline.joblib
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from flask import Flask, jsonify, request

# ------------------ CONFIG (edit these as needed) ------------------
# NOTE: This path was taken from the conversation history. Replace if your CSV is elsewhere.
DATA_PATH = "Smart_Farming_Crop_Yield_2024.csv"
MODEL_OUT = "knn_smartfarm_pipeline.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIME_SPLIT = False   # Set True if you have a date column and want a time-based split
TARGET_COL = "yield_kg_per_hectare"  # change to your actual target column name (e.g., "yield_t_ha")
DATE_COL = "date"     # change if your dataset has a date/datetime column, otherwise set to None or ''
FARM_ID_COL = "farm_id"  # optional farm identifier column, used for rolling features if desired
# ------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please update DATA_PATH to the CSV file.")
    # try reading CSV; if path is a zip or other, user must extract CSV first
    df = pd.read_csv(path, parse_dates=[DATE_COL] if DATE_COL and DATE_COL in pd.read_csv(path, nrows=0).columns else [])
    print(f"Loaded data shape: {df.shape}")
    return df

def quick_inspect(df: pd.DataFrame):
    print("\n--- HEAD ---")
    print(df.head(5))
    print("\n--- DTYPEs ---")
    print(df.dtypes)
    print("\n--- Missing values per column (top 20) ---")
    print(df.isna().sum().sort_values(ascending=False).head(20))

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # ensure date is datetime if present
    if DATE_COL and DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')

    # drop rows missing target
    df = df.loc[~df[TARGET_COL].isna()].copy()

    # Add simple time features if date present
    if DATE_COL and DATE_COL in df.columns:
        df['month'] = df[DATE_COL].dt.month
        df['dayofyear'] = df[DATE_COL].dt.dayofyear
        df['year'] = df[DATE_COL].dt.year

    # Optionally add rolling features per farm (if farm id and date present)
    if FARM_ID_COL in df.columns and DATE_COL and DATE_COL in df.columns:
        df = df.sort_values([FARM_ID_COL, DATE_COL])
        numeric_for_roll = ['soil_moisture', 'rainfall', 'ndvi']  # adapt to your columns
        for c in numeric_for_roll:
            if c in df.columns:
                df[f'{c}_roll3'] = df.groupby(FARM_ID_COL)[c].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    return df

def build_preprocessing_and_pipeline(df: pd.DataFrame):
    # Exclude id/date/target from feature set
    exclude = {TARGET_COL}
    if DATE_COL:
        exclude.add(DATE_COL)
    if FARM_ID_COL:
        exclude.add(FARM_ID_COL)

    # Detect numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclude]

    print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols[:10]}")
    print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols[:10]}")

    # Preprocessing for numeric features: impute then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features: impute then one-hot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop', verbose_feature_names_out=False)

    # KNeighbors regressor pipeline
    knn = KNeighborsRegressor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', knn)
    ])

    features = numeric_cols + categorical_cols
    return pipeline, features

def split_data(df: pd.DataFrame, features):
    X = df[features]
    y = df[TARGET_COL].values

    if TIME_SPLIT and DATE_COL and DATE_COL in df.columns:
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

def tune_knn(pipeline, X_train, y_train):
    print("Starting GridSearchCV for KNN (this may take a while depending on data size)...")
    param_grid = {
        'model__n_neighbors': [3, 5, 7, 9, 11, 15],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
    }
    # If time series, use TimeSeriesSplit; else simple 5-fold
    cv = TimeSeriesSplit(n_splits=3) if TIME_SPLIT else 5
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV score (neg RMSE):", search.best_score_)
    return search.best_estimator_

def evaluate_and_report(model, X_test, y_test, show_plots=True):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print("\n--- Test set evaluation ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    if show_plots:
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, preds, alpha=0.4)
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title("KNN: Actual vs Predicted Yield")
        lims = [min(min(y_test), min(preds)), max(max(y_test), max(preds))]
        plt.plot(lims, lims, 'r--')
        plt.grid(True)
        plt.show()

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'preds': preds}

def save_pipeline(pipeline, feature_names, outpath=MODEL_OUT):
    print(f"Saving pipeline to {outpath} ...")
    joblib.dump({'pipeline': pipeline, 'features': feature_names}, outpath)

def create_flask_app(pipeline, features):
    app = Flask("smartfarm_knn_api")

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        # Expect features as a dict
        feat_dict = data.get('features', {})
        # Build input row in the same order as features
        x = [feat_dict.get(f, np.nan) for f in features]
        df_row = pd.DataFrame([x], columns=features)
        # model expects same columns and types
        pred = pipeline.predict(df_row)[0]
        return jsonify({'predicted_yield': float(pred)})

    return app

def main(args):
    # 1) Load
    df = load_data(DATA_PATH)
    quick_inspect(df)

    # 2) Feature engineering
    df = basic_feature_engineering(df)
    print("After feature engineering shape:", df.shape)

    # 3) Build pipeline and feature list
    pipeline, features = build_preprocessing_and_pipeline(df)
    print(f"Feature count used: {len(features)}")

    # 4) Split
    X_train, X_test, y_train, y_test = split_data(df, features)

    # 5) Train base KNN quickly (k=5 default)
    pipeline.set_params(model__n_neighbors=5, model__weights='distance', model__p=2)
    print("\nFitting baseline KNN pipeline (k=5, distance weights, p=2)...")
    pipeline.fit(X_train, y_train)

    # 6) Optional tuning
    if args.tune:
        pipeline = tune_knn(pipeline, X_train, y_train)

    # 7) Evaluate
    metrics = evaluate_and_report(pipeline, X_test, y_test, show_plots=not args.no_plots)

    # 8) Save
    save_pipeline(pipeline, features, MODEL_OUT)

    # 9) Optionally serve
    if args.serve:
        app = create_flask_app(pipeline, features)
        print("Starting Flask app on http://127.0.0.1:5050 ...")
        app.run(debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KNN regressor for SmartFarm yield prediction.")
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV hyperparameter tuning.")
    parser.add_argument("--serve", action="store_true", help="Start Flask API after training.")
    parser.add_argument("--no-plots", action="store_true", help="Don't show matplotlib plots.")
    args = parser.parse_args()
    main(args)
