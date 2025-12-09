import os
import io
import json
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

DEFAULT_CSV = "Smart_Farming_Crop_Yield_2024.csv"

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

def build_single_model(name: str, preprocessor):
    """Build one model by name with sensible defaults."""
    if name == "KNN":
        return Pipeline([
            ("pre", preprocessor),
            ("est", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ])
    if name == "Random Forest":
        return Pipeline([
            ("pre", preprocessor),
            ("est", RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )),
        ])
    if name == "SVR":
        return Pipeline([
            ("pre", preprocessor),
            ("est", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        ])
    if name == "XGBoost" and XGB_OK:
        return Pipeline([
            ("pre", preprocessor),
            ("est", XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=0,
            )),
        ])
    return None

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "R²": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
    }, preds, model

def make_single_input_form(X: pd.DataFrame, num_cols, cat_cols, initial: Optional[pd.Series] = None):
    """Render inputs for one hypothetical farm row; optionally prefill with a Series."""
    st.subheader("Enter a hypothetical farm")
    cols = st.columns(2)
    user_row = {}

    initial = initial if initial is not None else pd.Series(dtype=object)

    # Numeric inputs
    for i, col in enumerate(num_cols):
        col_min = float(np.nanmin(X[col])) if col in X.columns and X[col].notna().any() else 0.0
        col_max = float(np.nanmax(X[col])) if col in X.columns and X[col].notna().any() else 100.0
        default = float(initial[col]) if col in initial else (float(np.nanmedian(X[col])) if col in X.columns and X[col].notna().any() else 0.0)
        step = (col_max - col_min) / 100 if col_max > col_min else 1.0
        user_row[col] = cols[i % 2].number_input(col, value=default, min_value=min(col_min, default), max_value=max(col_max, default + step), step=step)

    # Categorical inputs
    for i, col in enumerate(cat_cols):
        choices = sorted([str(x) for x in X[col].dropna().unique()][:200]) if col in X.columns else []
        if len(choices) == 0:
            val_default = str(initial[col]) if col in initial else ""
            val = cols[i % 2].text_input(col, value=val_default)
        else:
            default_choice = str(initial[col]) if col in initial and str(initial[col]) in choices else choices[0] if choices else ""
            val = cols[i % 2].selectbox(col, options=choices, index=choices.index(default_choice) if default_choice in choices else 0)
        user_row[col] = val

    return pd.DataFrame([user_row])

# ---------- UI
st.title("🌾 SmartFarm — Yield Prediction from Sensor Data")
st.markdown(
    "Train **KNN**, **Random Forest**, **SVR**, and **XGBoost** models on the provided Smart Farming dataset, "
    "then enter new measurements to predict yield."
)

with st.expander("About the dataset & project", expanded=False):
    st.write(
        "This app uses the **Smart Farming Sensor Data for Yield Prediction** dataset (bundled CSV). "
        "Choose your target, tune models, and compare results."
    )
    st.caption("Source dataset on Kaggle.")

def find_local_csv(root_path: str = ".") -> Optional[str]:
    # search for a csv in the repo (data/ or top-level)
    for dirpath, dirnames, filenames in os.walk(root_path):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                # ignore obvious output files
                if fn.lower().startswith("pred") or fn.lower().startswith("out"):
                    continue
                return os.path.join(dirpath, fn)
    return None


def try_load_from_github_raw() -> Optional[pd.DataFrame]:
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
    """Always load the bundled default CSV; fall back to synthetic if missing."""
    local = DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else find_local_csv()
    if local:
        try:
            df = pd.read_csv(local)
            st.success(f"Loaded dataset: {os.path.relpath(local)}")
            return df
        except Exception as e:
            st.error(f"Failed to read {local}: {e}")
    # 2) try GitHub raw
    df = try_load_from_github_raw()
    if df is not None:
        st.success("Loaded dataset from GitHub (raw)")
        return df
    # 3) fallback: synthetic dataset
    st.warning("Using synthetic demo dataset (default CSV missing).")
    return generate_synthetic_dataset()


# Always use the default dataset; no uploads
df = load_default_dataset()
st.success(f"Dataset ready — shape {df.shape}. Using `{os.path.basename(DEFAULT_CSV)}` when available.")

# Target auto-selection (no sidebar control)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
yield_candidates = [c for c in numeric_cols if "yield" in c.lower()]
default_target = yield_candidates[0] if yield_candidates else (numeric_cols[-1] if numeric_cols else None)
target_col = default_target
st.info(f"Target column: `{target_col}`")

# Quick dataset preview (small)
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Split
test_size = 0.2
random_state = 42
X, y = split_features_target(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Preprocessor
pre, num_cols, cat_cols = build_preprocessor(X)

# Sidebar — model choice
st.sidebar.header("Model")

# Re-try importing xgboost in case it was installed after app start
if not XGB_OK:
    try:
        from xgboost import XGBRegressor  # type: ignore
        XGB_OK = True
    except Exception as e:
        st.sidebar.info(f"xgboost unavailable: {e}")

model_options = ["KNN", "Random Forest", "SVR"] + (["XGBoost"] if XGB_OK else [])
model_name = st.sidebar.selectbox("Pick algorithm", options=model_options, index=0)
if not XGB_OK and "XGBoost" not in model_options:
    st.sidebar.info("xgboost not installed; skipping XGBoost.")

# Train & evaluate a single model
st.header("Train & Evaluate")
run = st.button("Train model", type="primary")

if "trained_model" not in st.session_state or run or st.session_state.get("trained_model_name") != model_name:
    model = build_single_model(model_name, pre)
    if model is None:
        st.error(f"Model {model_name} is unavailable. Install xgboost if needed.")
    else:
        with st.spinner(f"Training {model_name}..."):
            metrics, preds, fitted = evaluate(model, X_train, X_test, y_train, y_test)
        st.session_state["trained_model"] = fitted
        st.session_state["trained_model_name"] = model_name
        st.session_state["metrics"] = metrics
        st.session_state["preds"] = preds

if "trained_model" in st.session_state:
    m = st.session_state["metrics"]
    st.success(f"Trained {st.session_state['trained_model_name']} — test R² {m['R²']:.3f}, RMSE {m['RMSE']:.3f}, MAE {m['MAE']:.3f}")

    # Single prediction UI
    st.header("Predict on New Data")
    new_row = make_single_input_form(X, num_cols, cat_cols)
    col_pred, col_rand = st.columns([1,1])
    if col_pred.button("Predict yield", type="primary"):
        mdl = st.session_state["trained_model"]
        pred = float(mdl.predict(new_row)[0])
        st.session_state["last_pred"] = {
            "value": pred,
            "model": st.session_state["trained_model_name"],
            "source": "manual",
        }

    if col_rand.button("Generate random farm & predict"):
        random_row = X.sample(1, random_state=np.random.randint(0, 10_000)).iloc[0]
        random_df = pd.DataFrame([random_row])
        mdl = st.session_state["trained_model"]
        pred = float(mdl.predict(random_df)[0])
        st.info("Random farm sample (from dataset):")
        st.dataframe(random_df, use_container_width=True)
        st.session_state["last_pred"] = {
            "value": pred,
            "model": st.session_state["trained_model_name"],
            "source": "random",
        }

    # Render last prediction if available
    if "last_pred" in st.session_state:
        lp = st.session_state["last_pred"]
        st.markdown(
            f"""
            <div style="text-align:center; margin-top: 24px;">
                <div style="font-size:18px; color:#bbb;">Predicted {target_col} using {lp['model']}</div>
                <div class="glass-card">
                    <div class="big-yield">{lp['value']:,.3f}</div>
                    <div class="glass-emoji">🌾 🍃 🌻</div>
                </div>
            </div>
            <style>
            .glass-card {{
                margin: 14px auto;
                padding: 24px 32px;
                width: min(640px, 95vw);
                background: radial-gradient(80% 80% at 50% 20%, rgba(139,226,139,0.18), rgba(46,46,46,0.4)) rgba(30,30,30,0.55);
                border: 1px solid rgba(255,255,255,0.15);
                box-shadow: 0 20px 50px rgba(0,0,0,0.35);
                border-radius: 24px;
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                animation: glass-pop 0.6s ease;
            }}
            .big-yield {{
                font-size: 120px;
                font-weight: 900;
                color: #d2ffd2;
                text-shadow: 0 0 18px rgba(139,226,139,0.45);
                animation: shimmer 1.8s ease-in-out infinite alternate;
            }}
            .glass-emoji {{
                font-size: 48px;
                margin-top: 8px;
                animation: floaty 2.4s ease-in-out infinite;
            }}
            @keyframes glass-pop {{
                0% {{ transform: scale(0.85); opacity: 0.2; }}
                60% {{ transform: scale(1.05); opacity: 1; }}
                100% {{ transform: scale(1.0); opacity: 1; }}
            }}
            @keyframes shimmer {{
                from {{ filter: drop-shadow(0 0 8px rgba(139,226,139,0.3)); }}
                to {{ filter: drop-shadow(0 0 20px rgba(139,226,139,0.6)); }}
            }}
            @keyframes floaty {{
                0% {{ transform: translateY(0px); opacity: 0.85; }}
                50% {{ transform: translateY(-8px); opacity: 1; }}
                100% {{ transform: translateY(0px); opacity: 0.85; }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
