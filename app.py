import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

# xgboost is optional; show a friendly message if missing
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

st.set_page_config(page_title="SmartFarm — Yield Predictor", page_icon="🌾", layout="wide")

# ---------- Helpers
@st.cache_data
def load_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)

def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre, num_cols, cat_cols

def build_models(lasso_alpha: float, knn_k: int, xgb_params: dict, preprocessor):
    models = {}

    lasso = Pipeline([
        ("pre", preprocessor),
        ("est", Lasso(alpha=lasso_alpha, random_state=42, max_iter=10000)),
    ])
    models["Lasso"] = lasso

    knn = Pipeline([
        ("pre", preprocessor),
        ("est", KNeighborsRegressor(n_neighbors=knn_k, weights="distance")),
    ])
    models["KNN"] = knn

    if XGB_OK:
        xgb = Pipeline([
            ("pre", preprocessor),
            ("est", XGBRegressor(
                n_estimators=xgb_params["n_estimators"],
                max_depth=xgb_params["max_depth"],
                learning_rate=xgb_params["learning_rate"],
                subsample=xgb_params["subsample"],
                colsample_bytree=xgb_params["colsample_bytree"],
                reg_lambda=xgb_params["reg_lambda"],
                objective="reg:squarederror",
                random_state=42,
                n_jobs=0
            ))
        ])
        models["XGBoost"] = xgb
    return models

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "R²": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False),
    }, preds, model

def make_single_input_form(X: pd.DataFrame, num_cols, cat_cols):
    st.subheader("Enter new farming data")
    cols = st.columns(2)
    user_row = {}

    # Numeric inputs
    for i, col in enumerate(num_cols):
        col_min = float(np.nanmin(X[col])) if col in X.columns and X[col].notna().any() else 0.0
        col_max = float(np.nanmax(X[col])) if col in X.columns and X[col].notna().any() else 100.0
        default = float(np.nanmedian(X[col])) if col in X.columns and X[col].notna().any() else 0.0
        user_row[col] = cols[i % 2].number_input(col, value=default, min_value=col_min, max_value=col_max, step=(col_max-col_min)/100 if col_max>col_min else 1.0)

    # Categorical inputs
    for i, col in enumerate(cat_cols):
        choices = sorted([str(x) for x in X[col].dropna().unique()][:200]) if col in X.columns else []
        if len(choices) == 0:
            val = cols[i % 2].text_input(col, value="")
        else:
            val = cols[i % 2].selectbox(col, options=choices, index=0)
        user_row[col] = val

    return pd.DataFrame([user_row])

# ---------- UI
st.title("🌾 SmartFarm — Yield Prediction from Sensor Data")
st.markdown(
    "Train **Lasso**, **K-Nearest Neighbors**, and **XGBoost** models on your farming sensor dataset, "
    "then enter new measurements to predict yield."
)

with st.expander("About the dataset & project", expanded=False):
    st.write(
        "This app works with the **Smart Farming Sensor Data for Yield Prediction** dataset from Kaggle, "
        "or any CSV with a numeric target column. "
        "Upload the CSV on the left, choose your target, tune models, and compare results."
    )
    st.caption("Source dataset on Kaggle.")

# Sidebar — data
st.sidebar.header("1) Upload / Load Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional) — otherwise the app will load a default dataset", type=["csv"])


@st.cache_data
def find_local_csv(root_path: str = ".") -> str | None:
    # search for a csv in the repo (data/ or top-level)
    for dirpath, dirnames, filenames in os.walk(root_path):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                # ignore obvious output files
                if fn.lower().startswith("pred") or fn.lower().startswith("out"):
                    continue
                return os.path.join(dirpath, fn)
    return None


def try_load_from_github_raw() -> pd.DataFrame | None:
    # Attempt to construct a raw GitHub URL to a likely dataset file using the origin remote.
    # This is best-effort and will fail silently if not available.
    try:
        import subprocess
        remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode().strip()
        if remote.endswith('.git'):
            remote = remote[:-4]
        # common candidate paths
        candidates = [
            "data/Smart_Farming_Crop_Yield_2024.csv",
            "data/smartfarm.csv",
            "Smart_Farming_Crop_Yield_2024.csv",
            "smartfarm.csv",
        ]
        for c in candidates:
            # construct raw url for GitHub
            if remote.startswith("https://github.com/"):
                raw = remote.replace("https://github.com/", "https://raw.githubusercontent.com/") + "/main/" + c
                try:
                    df = pd.read_csv(raw)
                    return df
                except Exception:
                    continue
    except Exception:
        return None
    return None


def generate_synthetic_dataset(n_samples: int = 500) -> pd.DataFrame:
    # Create a small synthetic dataset resembling sensor readings and a yield target.
    rng = np.random.default_rng(42)
    soil_moisture = rng.normal(loc=30, scale=8, size=n_samples)
    temp = rng.normal(loc=22, scale=5, size=n_samples)
    humidity = rng.normal(loc=55, scale=10, size=n_samples)
    ph = rng.normal(loc=6.5, scale=0.5, size=n_samples)
    fertilizer = rng.integers(0, 3, size=n_samples)  # 0,1,2 types
    # create a target with some noise
    yield_kg = (0.5 * soil_moisture) + (1.2 * temp) + (-0.3 * ph) + (2.5 * fertilizer) + rng.normal(0, 5, size=n_samples)
    df = pd.DataFrame({
        "soil_moisture": soil_moisture,
        "temperature": temp,
        "humidity": humidity,
        "ph": ph,
        "fertilizer_type": fertilizer,
        "yield": yield_kg,
    })
    return df


def load_default_dataset():
    # 1) check for local CSVs
    local = find_local_csv()
    if local:
        try:
            df = pd.read_csv(local)
            st.sidebar.success(f"Loaded local dataset: {os.path.relpath(local)}")
            return df
        except Exception:
            pass
    # 2) try GitHub raw
    df = try_load_from_github_raw()
    if df is not None:
        st.sidebar.success("Loaded dataset from GitHub (raw)")
        return df
    # 3) fallback: synthetic dataset
    st.sidebar.warning("No dataset found locally or on GitHub — using a synthetic demo dataset. Upload a CSV to override.")
    return generate_synthetic_dataset()


if uploaded is not None:
    try:
        df = load_csv(uploaded)
        st.sidebar.success(f"Loaded uploaded CSV ({uploaded.name}) — shape {df.shape}.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        df = load_default_dataset()
else:
    df = load_default_dataset()
    st.success(f"Dataset ready — shape {df.shape}.")

# Target selection
st.sidebar.header("2) Choose Target")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_target = "yield" if "yield" in [c.lower() for c in df.columns] else (numeric_cols[-1] if numeric_cols else None)
target_col = st.sidebar.selectbox("Target column (numeric)", options=numeric_cols, index=numeric_cols.index(default_target) if default_target in numeric_cols else 0)

# Feature preview
st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

# Split
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = 42
X, y = split_features_target(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Preprocessor
pre, num_cols, cat_cols = build_preprocessor(X)

# Sidebar — model settings
st.sidebar.header("3) Models & Hyperparameters")

lasso_alpha = st.sidebar.number_input("Lasso α (L1 strength)", min_value=0.0001, max_value=10.0, value=0.1, step=0.05, format="%.4f")
knn_k = st.sidebar.slider("KNN: n_neighbors (K)", min_value=1, max_value=50, value=7, step=1)

xgb_params = {
    "n_estimators": st.sidebar.slider("XGB: n_estimators", 50, 1000, 300, step=50),
    "max_depth": st.sidebar.slider("XGB: max_depth", 2, 12, 6, step=1),
    "learning_rate": st.sidebar.slider("XGB: learning_rate", 0.01, 0.5, 0.1, step=0.01),
    "subsample": st.sidebar.slider("XGB: subsample", 0.5, 1.0, 0.9, step=0.05),
    "colsample_bytree": st.sidebar.slider("XGB: colsample_bytree", 0.5, 1.0, 0.9, step=0.05),
    "reg_lambda": st.sidebar.slider("XGB: reg_lambda", 0.0, 5.0, 1.0, step=0.1),
}
if not XGB_OK:
    st.sidebar.warning("xgboost not installed. Run: pip install xgboost")

models = build_models(lasso_alpha, knn_k, xgb_params, pre)

# Train & evaluate
st.header("Model Training & Evaluation")
run = st.button("Train / Re-train Models", type="primary")

if "results" not in st.session_state or run:
    results = {}
    preds_store = {}
    fitted_store = {}
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            metrics, preds, fitted = evaluate(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        preds_store[name] = preds
        fitted_store[name] = fitted
    st.session_state["results"] = results
    st.session_state["preds"] = preds_store
    st.session_state["models"] = fitted_store

# Results table
res_df = pd.DataFrame(st.session_state["results"]).T.sort_values("RMSE")
st.write("### Test Set Performance")
st.dataframe(res_df.style.format({"R²": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}), use_container_width=True)

best_model_name = res_df.index[0]
st.success(f"Best model right now: **{best_model_name}** (lowest RMSE).")

# Single prediction UI
st.header("Predict on New Data")
new_row = make_single_input_form(X, num_cols, cat_cols)
predict_btn = st.button("Predict Yield", type="primary")
if predict_btn:
    mdl = st.session_state["models"][best_model_name]
    pred = float(mdl.predict(new_row)[0])
    st.metric(label=f"Predicted {target_col}", value=f"{pred:,.3f}")

# Batch prediction
st.subheader("Batch Prediction")
batch_file = st.file_uploader("Upload CSV with the SAME feature columns for batch predictions", type=["csv"], key="batch")
if batch_file:
    batch_df = pd.read_csv(batch_file)
    mdl = st.session_state["models"][best_model_name]
    batch_pred = mdl.predict(batch_df)
    out = batch_df.copy()
    out[f"pred_{target_col}"] = batch_pred
    st.download_button("Download Predictions CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="smartfarm_predictions.csv", mime="text/csv")

# Model card
with st.expander("Model Card (auto-generated)", expanded=False):
    st.write(f"""
**Project:** SmartFarm — Yield Prediction  
**Target:** `{target_col}`  
**Features:** {len(X.columns)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)  
**Best model:** {best_model_name}  
**Data split:** train/test = {1-test_size:.2f} / {test_size:.2f} (random_state=42)  
**Metrics (test):**
- R²: {st.session_state['results'][best_model_name]['R²']:.3f}
- MAE: {st.session_state['results'][best_model_name]['MAE']:.3f}
- RMSE: {st.session_state['results'][best_model_name]['RMSE']:.3f}
""")
