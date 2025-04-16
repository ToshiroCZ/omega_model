OMEGA_MODEL - Used Car Price Prediction
=======================================

Matouš Podaný
C4b
SPŠE Ječná

Project Description:
--------------------
This module is part of the OMEGA project and is responsible for preparing, cleaning, combining and modeling data from used car listings (e.g., AAA Auto, Auto ESA). It trains three regression models (Random Forest, XGBoost, MLP) to predict car prices based on several features.

Directory Structure:
--------------------
omega_model/
├── main.py                # Main script to train models
├── cleaner.py             # Functions for cleaning and normalizing datasets
├── value_mapping.py       # Generator for mapping values used in the frontend
├── models/                # Folder where trained models are stored
├── maps/                  # Folder with generated JSON value mapping
├── car_data_all.csv       # Cleaned and merged dataset (if already created)

Required Files:
---------------
- `aaa_auto_data_copy.csv`
- `auto_esa_data.csv`
OR
- `car_data_all.csv`

These files are automatically merged and cleaned. The merged version is saved as `car_data_all.csv`.

Installation:
-------------
1. Create and activate a virtual environment:
   - python -m venv venv
   - venv\Scripts\activate (Windows)

2. Install required packages:
   - pip install -r requirements.txt

Usage:
------
1. Place your source CSVs (`aaa_auto_data_copy.csv` and `auto_esa_data.csv`) in the root of `omega_model/`.
2. Run:
   - python main.py
3. The script will:
   - Clean and normalize brand/model names
   - Merge datasets (if not already merged)
   - Prepare the data (encoding, scaling)
   - Train three models (RandomForest, XGBoost, MLP)
   - Save trained models and additional data:
     - `model_rf.pkl`, `model_xgb.pkl`, `model_mlp.pkl`
     - `scaler.pkl`
     - `features.pkl`
     - `value_mapping.json`

Model Evaluation:
-----------------
After training, the script prints:
- Best hyperparameters for each model (using GridSearchCV)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Training time for each model

Features used:
--------------
- make (brand)
- model
- year
- mileage
- fuel_type
- transmission
- engine_power
- body_type

Code Highlights:
----------------
✔ Robust exception handling  
✔ Modular structure  
✔ Easy extensibility (e.g., new models or sources)  
✔ Automatic feature alignment and scaling  
✔ Cleaned and normalized data input  

Quality Criteria Fulfilled:
---------------------------
✓ Configurability & Universality  
✓ Architecture & Design  
✓ Usability & Control  
✓ Correctness & Efficiency  
✓ Error Handling  
✓ Code Readability & Documentation  
✓ Machine Learning Quality

Notes:
------
- `features.pkl` and `scaler.pkl` are required by the frontend (`omega_web`) to process user input in the same way as during training.
- If the merged file `car_data_all.csv` exists, the script does not re-merge but only re-cleans it.