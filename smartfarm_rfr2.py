# Random Forest Regressor 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = "Smart_Farming_Crop_Yield_2024.csv" 

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from {DATA_PATH}")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

print("Dataset preview:")
print(df.head())
print("\nShape:", df.shape)

# -----------------------------
# 2. Identify Target Column
# -----------------------------
# IMPORTANT: replace this with your actual yield column
TARGET_COL = "yield_kg_per_hectare"  # <-- Change this to match your dataset

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Please update TARGET_COL.")

# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -----------------------------
# 3. Handle Missing Values
# -----------------------------
X = X.fillna(X.median())
y = y.fillna(y.median())

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Build Random Forest Pipeline
# -----------------------------
rf_model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# 6. Train Model
# -----------------------------
print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)
print("Training complete!")

# -----------------------------
# 7. Evaluate Model
# -----------------------------
preds = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n-----------------------------")
print("Model Performance (Random Forest)")
print("-----------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# -----------------------------
# 8. Feature Importance
# -----------------------------
try:
    importance = rf_model.named_steps["rf"].feature_importances_
    feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    
    print("\nTop Feature Importances:")
    print(feature_importance.head(10))

    # Plot
    feature_importance.head(15).plot(kind='bar', figsize=(10, 6))
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("\nCould not calculate feature importance:", e)

# -----------------------------
# 9. Save Model to Joblib
# -----------------------------
OUTPUT_MODEL = "rf_smartfarm_model.joblib"
joblib.dump(rf_model, OUTPUT_MODEL)
print(f"\nModel saved as {OUTPUT_MODEL}")
