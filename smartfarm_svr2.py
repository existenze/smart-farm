"""
smartfarm_svr.py

Train a Support Vector Regression (SVR) model for crop yield prediction (SmartFarm).
- Replace DATA_PATH with your CSV dataset (default points to the uploaded file path from this conversation).
- Usage:
    python smartfarm_svr.py                # train + save model
    python smartfarm_svr.py --tune         # run hyperparameter tuning (GridSearchCV)
    python smartfarm_svr.py --serve        # start Flask server loading saved model (svr_smartfarm_pipeline.joblib)
    python smartfarm_svr.py --no-plots     # suppress matplotlib plots
"""

import argparse
import sys
from pathlib import Path
import joblib
import traceback

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt

# ------------------ CONFIG - EDIT THESE ------------------
# NOTE: The path below is taken from the conversation history (uploaded file path).
# If your dataset is a CSV, replace this path with its CSV path (e.g., "./smart_farm_data.csv")
DATA_PATH = "Smart_Farming_Crop_Yield_2024.csv"   # <-- likely not a CSV; replace as needed
MODEL_OUT = "svr_smartfarm_pipeline.joblib"
TARGET_COL = "yield_kg_per_hectare"     # change to your dataset's yield column name, e.g., "yield_t_ha"
DATE_COL = "date"        # change if different or set to None if not present
FARM_ID_COL = "farm_id"  # set if present, else None or remove in code
TEST_SIZE = 0.2
TIME_SPLIT = False       # set True if you want time-based split and have DATE_COL present
RANDOM_STATE = 42
# --------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Update DATA_PATH to point to your CSV.")
    # Try reading as CSV. If the file is not CSV (e.g., PNG), user must set correct CSV path.
    try:
        df = pd.read_csv(path, parse_dates=[DATE_COL] if DATE_COL and DATE_COL in pd.read_csv(path, nrows=0).columns else [])
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV at {path}: {e}")
    return df

def inspect_df(df: pd.DataFrame):
    print("\n--- Dataframe head ---")
    print(df.head())
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Missing counts (top 20) ---")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    print("\nTotal rows:", len(df))

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure date column type if present
    if DATE_COL and DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')

    # Drop rows missing the target
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe columns: {df.columns.tolist()[:20]}")
    df = df.loc[~df[TARGET_COL].isna()].copy()

    # Simple time features if date exists
    if DATE_COL and DATE_COL in df.columns:
        df['month'] = df[DATE_COL].dt.month
        df['dayofyear'] = df[DATE_COL].dt.dayofyear
        df['year'] = df[DATE_COL].dt.year

    # Example rolling features if farm id and date present
    if FARM_ID_COL in df.columns and DATE_COL and DATE_COL in df.columns:
        df = df.sort_values([FARM_ID_COL, DATE_COL])
        numeric_roll = ['soil_moisture', 'rainfall', 'ndvi']  # adapt to your columns
        for c in numeric_roll:
            if c in df.columns:
                df[f'{c}_roll3'] = df.groupby(FARM_ID_COL)[c].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    return df

def build_preprocessor_and_pipeline(df: pd.DataFrame):
    # Exclude target, date, id from features
    exclude = {TARGET_COL}
    if DATE_COL:
        exclude.add(DATE_COL)
    if FARM_ID_COL:
        exclude.add(FARM_ID_COL)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclude]

    print(f"Detected {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.")
    print("Numeric sample:", numeric_cols[:10])
    print("Categorical sample:", categorical_cols[:10])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop', verbose_feature_names_out=False)

    svr = SVR()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('svr', svr)
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
        print(f"Time-based split -> train: {len(X_train)}, test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f"Random split -> train: {len(X_train)}, test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

def tune_and_train(pipeline, X_train, y_train, do_tune=False):
    if not do_tune:
        print("Training SVR with default parameters (this will use pipeline scaling).")
        pipeline.set_params(svr__kernel='rbf', svr__C=10.0, svr__gamma='scale', svr__epsilon=0.1)
        pipeline.fit(X_train, y_train)
        return pipeline

    # Grid search params (can be slow)
    param_grid = {
        'svr__kernel': ['rbf', 'poly'],
        'svr__C': [1.0, 10.0, 50.0, 100.0],
        'svr__gamma': ['scale', 'auto', 0.01, 0.001],
        'svr__epsilon': [0.01, 0.05, 0.1, 0.2]
    }
    cv = TimeSeriesSplit(n_splits=3) if TIME_SPLIT else 5
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    print("Starting GridSearchCV for SVR (this may take time)...")
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    return search.best_estimator_

def evaluate_model(model, X_test, y_test, show_plots=True):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    print("\n--- Test set metrics ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    if show_plots:
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, preds, alpha=0.4)
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield (SVR)")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.grid(True)
        plt.title("SVR: Actual vs Predicted")
        plt.show()

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'preds': preds}

def save_pipeline(pipeline, out_path=MODEL_OUT):
    joblib.dump(pipeline, out_path)
    print(f"Saved pipeline to {out_path}")

# Serve-only Flask app: load saved model and serve predictions
def serve_model(saved_model_path=MODEL_OUT, host="0.0.0.0", port=5000):
    # load the pipeline (preprocessor + model)
    payload = joblib.load(saved_model_path)
    pipeline = payload if not isinstance(payload, dict) else payload.get('pipeline', payload)
    # attempt to extract feature names in training order
    try:
        feature_names = pipeline.named_steps['preprocessor'].transformers_[0][2] + \
                        list(pipeline.named_steps['preprocessor'].transformers_[1][2])
    except Exception:
        feature_names = None

    app = Flask("smartfarm_svr_api")

    @app.route('/', methods=['GET'])
    def index():
        return {
            "message": "SmartFarm SVR prediction API",
            "predict_endpoint": "/predict"
        }

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json(force=True)
            if data is None:
                return jsonify({"error": "JSON body expected"}), 400

            # If feature_names available, require them
            if feature_names is not None:
                missing = [f for f in feature_names if f not in data]
                if missing:
                    return jsonify({"error": "Missing fields", "missing": missing, "required": feature_names}), 400
                x = np.array([[data[f] for f in feature_names]])
                # build DataFrame with same columns before pipeline
                import pandas as pd
                X = pd.DataFrame(x, columns=feature_names)
            else:
                # fallback: expect direct array passed as "vector"
                vec = data.get('vector', None)
                if vec is None:
                    return jsonify({"error": "Model feature order unknown. Provide 'vector' list or retrain with known features."}), 400
                X = np.array([vec])

            pred = pipeline.predict(X)[0]
            return jsonify({"prediction": float(pred)})

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    print(f"Starting SVR serve at http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False)

def main(args):
    if args.serve:
        # serve-only mode: serve an existing saved model
        if not Path(MODEL_OUT).exists():
            print(f"Saved model {MODEL_OUT} not found. Train first or provide correct MODEL_OUT.")
            sys.exit(1)
        serve_model(MODEL_OUT, host="0.0.0.0", port=5000)
        return

    # Training flow
    print(f"Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    inspect_df(df)

    df = basic_feature_engineering(df)
    print("After feature engineering, shape:", df.shape)

    pipeline, features = build_preprocessor_and_pipeline(df)
    print("Feature list length:", len(features))
    print("Example feature names:", features[:20])

    X_train, X_test, y_train, y_test = split_data(df, features)

    model = tune_and_train(pipeline, X_train, y_train, do_tune=args.tune)

    metrics = evaluate_model(model, X_test, y_test, show_plots=not args.no_plots)

    # Save pipeline (store pipeline object)
    save_pipeline(model, MODEL_OUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or serve Support Vector Regression model for SmartFarm.")
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV tuning (slow)")
    parser.add_argument("--serve", action="store_true", help="Serve an existing saved model (svr_smartfarm_pipeline.joblib)")
    parser.add_argument("--no-plots", action="store_true", help="Don't show matplotlib plots")
    args = parser.parse_args()
    main(args)