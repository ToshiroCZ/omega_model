import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from cleaner import merge_and_clean_datasets, clean_dataset
from value_mapping import ValueMapping

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === Paths ===
aaa_path = "aaa_auto_data_copy2.csv"
esa_path = "auto_esa_data.csv"
csv_path = "car_data_all.csv"
model_dir = "models"
maps_dir = "maps"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(maps_dir, exist_ok=True)

# === Příprava datasetu ===
try:
    if not os.path.exists(csv_path):
        logging.info("[INFO] Spojuji a čistím datasety...")
        merge_and_clean_datasets(aaa_path, esa_path, csv_path)
        logging.info("[INFO] Dataset vytvořen.")
    else:
        logging.info("[INFO] Dataset existuje, provádím nové čištění pro jistotu...")
        df = pd.read_csv(csv_path)
        df = clean_dataset(df)
        df.to_csv(csv_path, index=False)
        logging.info("[INFO] Dataset byl znovu vyčištěn a uložen.")
except Exception as e:
    logging.error(f"[CHYBA] Problém při přípravě datasetu: {e}")
    exit()

# === Načtení dat ===
try:
    logging.info(f"Načítám data z {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Načteno {len(df)} záznamů")
except Exception as e:
    logging.error(f"Chyba při načítání dat: {e}")
    exit()

# === Výběr atributů ===
features = ['make', 'model', 'year', 'mileage', 'fuel_type', 'transmission', 'engine_power', 'body_type']
target = 'price'

# === Rozdělení ===
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === One-hot encoding ===
categorical_features = ['make', 'model', 'fuel_type', 'transmission', 'body_type']
numeric_features = ['year', 'mileage', 'engine_power']

X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

# === Škálování ===
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# === Vyhodnocení modelu ===
def evaluate_model(name, model, param_grid):
    try:
        logging.info(f"Trénuji {name}...")
        grid = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logging.info(f"[{name}] Nejlepší parametry: {grid.best_params_}")
        logging.info(f"[{name}] MAE: {mae:.2f} Kč, RMSE: {rmse:.2f} Kč")
        return best
    except Exception as e:
        logging.error(f"Chyba u modelu {name}: {e}")
        return None

# === Gridy ===
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

# === Trénování modelů ===
models = {
    "RandomForestRegressor": (RandomForestRegressor(random_state=42), rf_grid),
    "XGBRegressor": (XGBRegressor(random_state=42, verbosity=0), xgb_grid)
    #"MLPRegressor": (MLPRegressor(random_state=42), mlp_grid)
}

trained_models = {}
for name, (model, grid) in models.items():
    best_model = evaluate_model(name, model, grid)
    if best_model:
        path = os.path.join(model_dir, f"model_{name.lower().replace('regressor', '')}.pkl")
        joblib.dump(best_model, path)
        trained_models[name] = path
        logging.info(f"{name} uložen do {path}")

# === Uložení scaleru a feature columns ===
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "features.pkl"))
logging.info("Scaler a seznam feature sloupců uložen.")

# === Uložení value mapping ===
mapper = ValueMapping(csv_path)
mapper.generate()
mapper.save(os.path.join(maps_dir, "value_mapping.json"))
logging.info("Value mapping uložen.")
