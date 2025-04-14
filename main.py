import pandas as pd
import numpy as np
import joblib
import os
import logging
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from value_mapping import ValueMapping
from cleaner import clean_fuel_type

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === Konfigurace ===
csv_path = "car_data_all_final.csv"
model_dir = "models"
maps_dir = "maps"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(maps_dir, exist_ok=True)
model_paths = {
    "rf": os.path.join(model_dir, "model_rf.pkl"),
    "xgb": os.path.join(model_dir, "model_xgb.pkl"),
    "mlp": os.path.join(model_dir, "model_mlp.pkl"),
}
scaler_path = os.path.join(model_dir, "scaler.pkl")
features_path = os.path.join(model_dir, "features.pkl")
maps_path = os.path.join(maps_dir, "value_mapping.json")

# === Načtení a čištění dat ===
logging.info(f"Načítám data z {csv_path}")
df = pd.read_csv(csv_path)
logging.info(f"Načteno {len(df)} záznamů")

df.dropna(subset=["price", "make", "model", "fuel_type", "transmission", "body_type"], inplace=True)
df = df[df["price"] > 0]
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
df["engine_power"] = pd.to_numeric(df["engine_power"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df.dropna(subset=["year", "mileage", "engine_power"], inplace=True)
df["fuel_type"] = df["fuel_type"].apply(clean_fuel_type)
logging.info(f"Data po čištění: {len(df)} záznamů")

# === Příprava ===
features = ['make', 'model', 'year', 'mileage', 'fuel_type', 'transmission', 'engine_power', 'body_type']
target = 'price'
X = df[features]
y = df[target]

categorical = ['make', 'model', 'fuel_type', 'transmission', 'body_type']
numeric = ['year', 'mileage', 'engine_power']

X = pd.get_dummies(X, columns=categorical, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

scaler = StandardScaler()
X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric] = scaler.transform(X_test[numeric])

def evaluate_model(name, model, param_grid):
    logging.info(f"Trénuji {name}...")
    start = time.time()
    grid = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    end = time.time()

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logging.info(f"[{name}] Nejlepší parametry: {grid.best_params_}")
    logging.info(f"[{name}] Trénink: {end - start:.2f} s, MAE: {mae:.2f} Kč, RMSE: {rmse:.2f} Kč")

    return best

# === Param Gridy ===
rf_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 30, 50],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

xgb_grid = {
    "n_estimators": [300],
    "max_depth": [8],
    "learning_rate": [0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.7]
}

mlp_grid = {
    "hidden_layer_sizes": [(100,), (100, 50), (128, 64, 32)],
    "alpha": [0.0001, 0.001],
    "activation": ["relu"],
    "solver": ["adam"],
    "learning_rate": ["adaptive"],
    "max_iter": [1000]
}

# === Trénuj modely ===
#model_rf = evaluate_model("RandomForestRegressor", RandomForestRegressor(random_state=42), rf_grid)
model_xgb = evaluate_model("XGBRegressor", XGBRegressor(random_state=42, verbosity=0), xgb_grid)
#model_mlp = evaluate_model("MLPRegressor", MLPRegressor(random_state=42), mlp_grid)

# === Uložení ===
#joblib.dump(model_rf, model_paths["rf"])
joblib.dump(model_xgb, model_paths["xgb"])
#joblib.dump(model_mlp, model_paths["mlp"])
joblib.dump(scaler, scaler_path)
joblib.dump(X_train.columns.tolist(), features_path)
logging.info(f"Modely a scaler uloženy do složky {model_dir}")

# === Mapping ===
mapping = ValueMapping(csv_path)
mapping.generate()
mapping.save(maps_path)
logging.info(f"Value mapping uložen do {maps_path}")
