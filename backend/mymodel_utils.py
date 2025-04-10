# -*- coding: utf-8 -*-
"""
mymodel_utils.py: Core ML logic for data processing, training, and prediction.
Adapted from mymodel.py, removing plotting and SHAP.
Adds Optuna caching and uses LightGBM feature importance.
"""

# Standard libraries
import pandas as pd
import numpy as np
import os
import joblib
import warnings
import time
import json
from collections import defaultdict
import threading # For locking during initialization

# Scikit-learn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Modeling
import lightgbm as lgb
import optuna

# Reduce verbosity
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration (Keep relevant parts) ---
DATA_FILE = 'modified_dataset.csv'
BASE_ARTIFACTS_DIR = "soil_artifacts_tuned_v3" # Keep same name for consistency

MODEL_SAVE_DIR = os.path.join(BASE_ARTIFACTS_DIR, "models")
SCALER_SAVE_DIR = os.path.join(BASE_ARTIFACTS_DIR, "scalers")
IMPUTE_SAVE_DIR = os.path.join(BASE_ARTIFACTS_DIR, "imputation")
PARAMS_CACHE_FILE = os.path.join(BASE_ARTIFACTS_DIR, "best_params.json") # For Optuna cache
PERFORMANCE_METRICS_FILE = os.path.join(BASE_ARTIFACTS_DIR, "performance_metrics.json") # To store metrics
FEATURE_RANKING_FILE = os.path.join(BASE_ARTIFACTS_DIR, "feature_rankings.json") # To store rankings

SPECTRAL_COLS = ['410', '435', '460', '485', '510', '535', '560', '585',
                 '610', '645', '680', '705', '730', '760', '810', '860',
                 '900', '940']
# Target keys MUST match the keys expected/used by the frontend if possible,
# or map them in the Flask app. Let's stick to original model keys for now.
TARGET_COLS = ['Ph', 'Nitro', 'Posh Nitro', 'Pota Nitro', 'Capacitity Moist', 'Temp', 'Moist', 'EC']
CONTEXT_COL = 'Water_Level'
ID_COLS = ['Records', 'Soil_Code']

WATER_LEVELS_TO_PROCESS = [0, 25, 50]
MIN_TRAIN_SAMPLES = 15
MIN_TEST_SAMPLES = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_OPTUNA_TRIALS = 50 # Can reduce for faster local testing if needed, e.g., 10
OPTUNA_CV_FOLDS = 3
OPTUNA_METRIC_LGBM = 'mae' # Evaluate with MAE during training
OPTUNA_OPTIMIZE_METRIC = 'rmse' # Optimize for RMSE in objective

# --- Global State (managed by Flask app, passed into functions) ---
# These will hold the loaded artifacts after initialization
_scalers = {}
_imputation_values = {}
_tuned_models = defaultdict(dict)
_performance_metrics = {}
_feature_rankings = {} # Structure: {target: [{'rank': 1, 'wavelength': 'X', 'importanceScore': Y}, ...]}

_is_initialized = False
_init_lock = threading.Lock()

# --- Helper Functions ---
def _create_dirs():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(SCALER_SAVE_DIR, exist_ok=True)
    os.makedirs(IMPUTE_SAVE_DIR, exist_ok=True)

def _load_data():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Data loaded successfully: {df.shape}")
        # Basic validation
        if CONTEXT_COL not in df.columns:
            raise ValueError(f"Context column '{CONTEXT_COL}' not found.")
        missing_spectral = [col for col in SPECTRAL_COLS if col not in df.columns]
        if missing_spectral:
            raise ValueError(f"Missing spectral columns: {missing_spectral}")
        missing_target = [col for col in TARGET_COLS if col not in df.columns]
        if missing_target:
             raise ValueError(f"Missing target columns: {missing_target}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file '{DATA_FILE}' not found.")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load or validate data: {e}")
        raise

def _split_data(df):
    print("Splitting data...")
    features = SPECTRAL_COLS + [CONTEXT_COL]
    X = df[features].copy()
    y = df[TARGET_COLS].copy()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=X[CONTEXT_COL]
        )
        print(f"Stratified split successful ({1-TEST_SIZE:.0%} Train / {TEST_SIZE:.0%} Test).")
    except ValueError as e:
         print(f"Warning: Could not stratify by Water_Level: {e}. Performing random split.")
         X_train, X_test, y_train, y_test = train_test_split(
             X, y,
             test_size=TEST_SIZE,
             random_state=RANDOM_STATE
         )
    print(f"Data Split Shapes: X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def _prepare_scalers_imputation(X_train):
    print("Preparing scalers and imputation values (from training data)...")
    local_scalers = {}
    local_imputation_values = defaultdict(dict)

    for wl in WATER_LEVELS_TO_PROCESS:
        train_indices_wl = X_train[CONTEXT_COL] == wl
        X_train_wl_spectral = X_train.loc[train_indices_wl, SPECTRAL_COLS]

        if not X_train_wl_spectral.empty and X_train_wl_spectral.shape[0] >= 2: # Need at least 2 samples for variance
            # Imputation
            means = X_train_wl_spectral.mean(axis=0).to_dict()
            local_imputation_values[wl] = means
            impute_filename = os.path.join(IMPUTE_SAVE_DIR, f"impute_means_wl{wl}.json")
            try:
                with open(impute_filename, 'w') as f: json.dump(means, f, indent=4)
                print(f"  WL {wl}ml: Saved imputation values.")
            except Exception as e: print(f"    Error saving imputation values for WL {wl}: {e}")

            # Scaler
            scaler = StandardScaler()
            scaler.fit(X_train_wl_spectral)
            local_scalers[wl] = scaler
            scaler_filename = os.path.join(SCALER_SAVE_DIR, f"scaler_wl{wl}.joblib")
            try:
                joblib.dump(scaler, scaler_filename)
                print(f"  WL {wl}ml: Saved scaler.")
            except Exception as e: print(f"    Error saving scaler for WL {wl}: {e}")
        else:
            print(f"  WL {wl}ml: Insufficient data ({X_train_wl_spectral.shape[0]} samples) to fit scaler/calculate means robustly. Skipping.")
            local_imputation_values[wl] = {col: np.nan for col in SPECTRAL_COLS}
            local_scalers[wl] = None
    return local_scalers, local_imputation_values

# --- Optuna Objective ---
def _objective(trial, X_train_fold, y_train_fold, X_val_fold, y_val_fold):
    """Optuna objective function for LightGBM Regressor."""
    param = {
        'objective': 'regression_l1', # MAE
        'metric': OPTUNA_METRIC_LGBM, # Evaluate with MAE or RMSE during training
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50), # Reduced max for speed
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50), # Reduced range
        'max_depth': trial.suggest_int('max_depth', 3, 10), # Reduced range
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-7, 5.0, log=True), # reg_alpha
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-7, 5.0, log=True), # reg_lambda
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0), # colsample_bytree
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0), # subsample
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5), # subsample_freq
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30), # Reduced range
    }

    model = lgb.LGBMRegressor(**param)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              eval_metric=OPTUNA_METRIC_LGBM,
              callbacks=[lgb.early_stopping(50, verbose=False)]) # Reduced patience

    preds = model.predict(X_val_fold)

    if OPTUNA_OPTIMIZE_METRIC == 'rmse':
        score = np.sqrt(mean_squared_error(y_val_fold, preds))
    elif OPTUNA_OPTIMIZE_METRIC == 'mae':
        score = mean_absolute_error(y_val_fold, preds)
    else: # Default to RMSE
        score = np.sqrt(mean_squared_error(y_val_fold, preds))
    return score

def _run_optuna_tuning(X_train_wl_scaled_df, y_train_target):
    print(f"    Running Optuna ({N_OPTUNA_TRIALS} trials, {OPTUNA_CV_FOLDS}-fold CV)...")
    study = optuna.create_study(direction='minimize') # Minimize RMSE or MAE
    kf = KFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective_cv_wrapper(trial):
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_wl_scaled_df, y_train_target)):
            X_train_fold = X_train_wl_scaled_df.iloc[train_idx]
            y_train_fold = y_train_target.iloc[train_idx]
            X_val_fold = X_train_wl_scaled_df.iloc[val_idx]
            y_val_fold = y_train_target.iloc[val_idx]
            score = _objective(trial, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            cv_scores.append(score)
        return np.mean(cv_scores)

    study.optimize(objective_cv_wrapper, n_trials=N_OPTUNA_TRIALS, timeout=300) # Shorter timeout
    best_params = study.best_params
    best_score = study.best_value
    print(f"    Optuna finished. Best CV {OPTUNA_OPTIMIZE_METRIC}: {best_score:.4f}")
    # print(f"    Best Params: {best_params}") # Keep this less verbose for server logs
    return best_params

def _train_and_evaluate(X_train, y_train, X_test, y_test, local_scalers):
    print("Training models and evaluating...")
    local_tuned_models = defaultdict(dict)
    local_performance_metrics = defaultdict(lambda: defaultdict(dict))
    local_best_params_dict = defaultdict(dict)
    local_feature_importances = defaultdict(lambda: defaultdict(list)) # Store importances per model

    # --- Load or Run Optuna ---
    if os.path.exists(PARAMS_CACHE_FILE):
        print(f"Loading cached best parameters from {PARAMS_CACHE_FILE}")
        try:
            with open(PARAMS_CACHE_FILE, 'r') as f:
                local_best_params_dict = json.load(f)
            # Convert keys back to int if needed (JSON saves keys as strings)
            local_best_params_dict = {int(k) if k.isdigit() else k: v for k, v in local_best_params_dict.items()}
            print("Cached parameters loaded.")
            run_tuning = False
        except Exception as e:
            print(f"Warning: Failed to load cached parameters: {e}. Re-running tuning.")
            local_best_params_dict = defaultdict(dict) # Reset if loading failed
            run_tuning = True
    else:
        print("No cached parameters file found. Running Optuna tuning...")
        run_tuning = True
        local_best_params_dict = defaultdict(dict)


    start_time_total = time.time()

    for wl in WATER_LEVELS_TO_PROCESS:
        print(f"\n--- Processing Models: Water Level = {wl} ml ---")
        start_time_wl = time.time()

        train_indices = X_train[CONTEXT_COL] == wl
        test_indices = X_test[CONTEXT_COL] == wl

        X_train_wl_orig = X_train.loc[train_indices, SPECTRAL_COLS].copy()
        y_train_wl = y_train.loc[train_indices].copy()
        X_test_wl_orig = X_test.loc[test_indices, SPECTRAL_COLS].copy()
        y_test_wl = y_test.loc[test_indices].copy()

        if X_train_wl_orig.shape[0] < MIN_TRAIN_SAMPLES:
            print(f"  Skipping WL {wl}: Insufficient training data ({X_train_wl_orig.shape[0]} < {MIN_TRAIN_SAMPLES}).")
            for target in TARGET_COLS: local_performance_metrics[wl][target]['Status'] = 'Skipped_Insufficient_Train_Data'
            continue

        scaler = local_scalers.get(wl)
        if scaler is None:
            print(f"  Skipping WL {wl}: Scaler not found/fitted.")
            for target in TARGET_COLS: local_performance_metrics[wl][target]['Status'] = 'Skipped_Scaler_Missing'
            continue

        # Apply Scaling
        try:
            X_train_wl_scaled = scaler.transform(X_train_wl_orig)
            X_test_wl_scaled = scaler.transform(X_test_wl_orig) if not X_test_wl_orig.empty else np.array([])

            # Keep as DF with column names for importance tracking
            X_train_wl_scaled_df = pd.DataFrame(X_train_wl_scaled, index=X_train_wl_orig.index, columns=SPECTRAL_COLS)
            X_test_wl_scaled_df = pd.DataFrame(X_test_wl_scaled, index=X_test_wl_orig.index, columns=SPECTRAL_COLS) if X_test_wl_scaled.size > 0 else pd.DataFrame(columns=SPECTRAL_COLS)
            print(f"  Applied StandardScaler. Train shape: {X_train_wl_scaled_df.shape}, Test shape: {X_test_wl_scaled_df.shape}")
        except Exception as e:
            print(f"  Error applying scaler for WL {wl}: {e}. Skipping.")
            for target in TARGET_COLS: local_performance_metrics[wl][target]['Status'] = 'Error_Scaling'
            continue

        # Loop through targets
        for target in TARGET_COLS:
            print(f"\n  --- Target: {target} (WL: {wl}ml) ---")
            y_train_target = y_train_wl[target]
            y_test_target = y_test_wl[target]

            if y_train_target.nunique() <= 1:
                print(f"    Skipping: Target '{target}' is constant in training data.")
                local_performance_metrics[wl][target]['Status'] = 'Skipped_Constant_Train'
                continue

            best_params = None
            # --- Optuna Tuning (if needed) ---
            if run_tuning:
                try:
                    # Optuna expects dict[str, dict], handle potential int key from loading
                    best_params = _run_optuna_tuning(X_train_wl_scaled_df, y_train_target)
                    local_best_params_dict[wl][target] = best_params # Store params found
                except Exception as e:
                    print(f"    Error during Optuna for {target} WL {wl}: {e}")
                    local_performance_metrics[wl][target]['Status'] = 'Error_Optuna'
                    continue # Skip training if tuning failed
            else:
                # Load pre-tuned params
                best_params = local_best_params_dict.get(str(wl), {}).get(target) # JSON keys are strings
                if best_params is None:
                     best_params = local_best_params_dict.get(wl, {}).get(target) # Try int key just in case
                if best_params is None:
                    print(f"    Warning: Cached parameters not found for {target} WL {wl}. Using default LGBM params.")
                    # Define some basic defaults or skip
                    best_params = {'random_state': RANDOM_STATE} # Minimal default
                    # Alternatively: continue -> skip training this model
                else:
                     print(f"    Using cached parameters for {target} WL {wl}.")


            # --- Train Final Model ---
            print(f"    Training final model...")
            final_model = lgb.LGBMRegressor(
                objective='regression_l1', metric=OPTUNA_METRIC_LGBM, verbosity=-1, boosting_type='gbdt',
                **best_params # Unpack best hyperparameters
            )
            try:
                final_model.fit(X_train_wl_scaled_df, y_train_target)
                local_tuned_models[wl][target] = final_model
                print(f"    Final model trained.")

                # Save model
                model_filename = os.path.join(MODEL_SAVE_DIR, f"model_tuned_{target.replace(' ', '_')}_WL{wl}ml.joblib")
                joblib.dump(final_model, model_filename)
                # print(f"    Saved tuned model: {model_filename}") # Less verbose

                # Store feature importances
                importances = final_model.feature_importances_
                feature_importance_map = {feature: imp for feature, imp in zip(SPECTRAL_COLS, importances)}
                local_feature_importances[target][wl] = feature_importance_map


            except Exception as e:
                 print(f"    Error training final model for {target} WL {wl}: {e}")
                 local_performance_metrics[wl][target]['Status'] = 'Error_Train_Final'
                 continue # Skip evaluation if final training failed

            # --- Evaluate ---
            if X_test_wl_scaled_df.empty or y_test_target.empty:
                 print("    Skipping evaluation: No test data.")
                 local_performance_metrics[wl][target].update({'Status': 'Success_TrainOnly_No_Test', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan})
            elif X_test_wl_scaled_df.shape[0] < MIN_TEST_SAMPLES:
                 print(f"    Warning: Evaluating on small test set ({X_test_wl_scaled_df.shape[0]} samples). Metrics may be unstable.")
                 # Proceed with evaluation but be aware
                 try:
                    y_pred = final_model.predict(X_test_wl_scaled_df)
                    r2 = r2_score(y_test_target, y_pred)
                    mae = mean_absolute_error(y_test_target, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
                    print(f"    Test Metrics: R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
                    local_performance_metrics[wl][target].update({'Status': 'Success', 'R2': r2, 'MAE': mae, 'RMSE': rmse})
                 except Exception as e:
                    print(f"    Error during evaluation for {target} WL {wl}: {e}")
                    local_performance_metrics[wl][target].update({'Status': 'Error_Eval_Final', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan})
            else:
                print(f"    Evaluating final model on test set ({X_test_wl_scaled_df.shape[0]} samples)...")
                try:
                    y_pred = final_model.predict(X_test_wl_scaled_df)
                    r2 = r2_score(y_test_target, y_pred)
                    mae = mean_absolute_error(y_test_target, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
                    print(f"    Test Metrics: R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
                    local_performance_metrics[wl][target].update({'Status': 'Success', 'R2': r2, 'MAE': mae, 'RMSE': rmse})
                except Exception as e:
                    print(f"    Error during evaluation for {target} WL {wl}: {e}")
                    local_performance_metrics[wl][target].update({'Status': 'Error_Eval_Final', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan})

        wl_elapsed = time.time() - start_time_wl
        print(f"--- Water Level {wl}ml processing time: {wl_elapsed:.2f} seconds ---")

    # Save cached parameters if tuning was run
    if run_tuning:
        print(f"Saving best parameters found to {PARAMS_CACHE_FILE}")
        try:
            # Convert defaultdicts to regular dicts for JSON serialization
            params_to_save = {
                str(k): {t: p for t, p in targets.items()}
                for k, targets in local_best_params_dict.items()
            }
            with open(PARAMS_CACHE_FILE, 'w') as f:
                json.dump(params_to_save, f, indent=4)
        except Exception as e:
            print(f"Error saving best parameters: {e}")

    total_elapsed = time.time() - start_time_total
    print(f"\n=== Total Training/Tuning Time: {total_elapsed / 60:.2f} minutes ===")

    # --- Post-process metrics and importances ---
    # Format performance metrics for easier JSON saving/loading
    final_performance_metrics = {}
    metric_keys = ['R2', 'MAE', 'RMSE']
    for metric_key in metric_keys:
        final_performance_metrics[metric_key] = {}
        for wl in WATER_LEVELS_TO_PROCESS:
             wl_key = f"{wl}ml" # Match frontend key format
             final_performance_metrics[metric_key][wl_key] = {}
             for target in TARGET_COLS:
                 # Map internal target names to potential frontend keys if needed
                 # For now, assume keys match or frontend maps them.
                 # Use target name directly as the key.
                 metric_value = local_performance_metrics[wl].get(target, {}).get(metric_key)
                 # Store None if NaN or missing, JSON handles null
                 final_performance_metrics[metric_key][wl_key][target] = None if (metric_value is None or np.isnan(metric_value)) else float(metric_value)

    # Save metrics
    try:
        with open(PERFORMANCE_METRICS_FILE, 'w') as f:
            json.dump(final_performance_metrics, f, indent=4)
        print(f"Saved performance metrics to {PERFORMANCE_METRICS_FILE}")
    except Exception as e:
        print(f"Error saving performance metrics: {e}")


    # --- Calculate Aggregated Feature Rankings ---
    print("\nCalculating aggregated feature importance rankings...")
    final_rankings = {}
    for target in TARGET_COLS:
        target_importances = local_feature_importances.get(target, {})
        if not target_importances:
            print(f"  Skipping ranking for {target}: No importance scores found.")
            final_rankings[target] = []
            continue

        # Average importance across water levels for this target
        aggregated_importance = defaultdict(float)
        count = 0
        for wl, importance_map in target_importances.items():
            if importance_map: # Check if importance map is not empty
                for feature, imp in importance_map.items():
                    aggregated_importance[feature] += imp
                count += 1

        if count > 0:
            avg_importance = {feat: total_imp / count for feat, total_imp in aggregated_importance.items()}
            # Sort features by average importance (descending)
            sorted_features = sorted(avg_importance.items(), key=lambda item: item[1], reverse=True)

            # Create ranking list [{rank, wavelength, importanceScore}, ...]
            target_ranking = [
                {'rank': i + 1, 'wavelength': feat, 'importanceScore': float(score)}
                for i, (feat, score) in enumerate(sorted_features)
            ]
            final_rankings[target] = target_ranking
            print(f"  Ranked features for {target} (Top 3): {target_ranking[:3]}")
        else:
             print(f"  Skipping ranking for {target}: No valid importance scores to aggregate.")
             final_rankings[target] = []

    # Save rankings
    try:
        with open(FEATURE_RANKING_FILE, 'w') as f:
            json.dump(final_rankings, f, indent=4)
        print(f"Saved feature rankings to {FEATURE_RANKING_FILE}")
    except Exception as e:
        print(f"Error saving feature rankings: {e}")


    return local_tuned_models, final_performance_metrics, final_rankings


# --- Prediction Function (Adapted for Flask context) ---
def predict_soil_properties_flexible_internal(
    input_spectral_data,
    water_level,
    loaded_models, # Pass loaded models
    loaded_scalers, # Pass loaded scalers
    loaded_imputation_values # Pass loaded imputation values
):
    """Internal prediction logic, assumes artifacts are loaded."""
    predictions = {
        'Prediction_Status': 'Pending',
        'Input_Water_Level': water_level,
        'Provided_Features': list(input_spectral_data.keys()),
        'Imputed_Features': []
    }
    target_predictions = {target: None for target in TARGET_COLS} # Initialize target predictions

    # --- Validation (Basic - more in Flask route) ---
    if water_level not in WATER_LEVELS_TO_PROCESS:
        predictions['Prediction_Status'] = f"Error: Invalid water_level '{water_level}'."
        return predictions, target_predictions

    print(f"\n--- Predicting for Water Level: {water_level} ml ---")
    num_provided = len(input_spectral_data)
    print(f"  Provided {num_provided} spectral features.")

    # --- Get Imputation Values for WL ---
    wl_impute_means = loaded_imputation_values.get(water_level)
    if wl_impute_means is None or not isinstance(wl_impute_means, dict) or any(v is None or np.isnan(v) for v in wl_impute_means.values()):
        predictions['Prediction_Status'] = f"Error: Imputation values missing or invalid for WL {water_level}."
        print(f"  {predictions['Prediction_Status']}")
        return predictions, target_predictions

    # --- Prepare Full Feature Set (Impute Missing) ---
    input_full = {}
    imputed_features_list = []
    all_features_valid = True
    for col in SPECTRAL_COLS:
        if col in input_spectral_data:
            value = input_spectral_data[col]
            # Basic check if value seems numeric (more robust checks can be added)
            if not isinstance(value, (int, float)) or np.isnan(value):
                 predictions['Prediction_Status'] = f"Error: Invalid numeric value provided for feature '{col}' ({value})."
                 print(f"  {predictions['Prediction_Status']}")
                 all_features_valid = False
                 break # Stop processing if one value is bad
            input_full[col] = value
        else:
            impute_val = wl_impute_means.get(col)
            if impute_val is None or np.isnan(impute_val):
                 predictions['Prediction_Status'] = f"Error: Missing imputation value for feature '{col}' at WL {water_level}."
                 print(f"  {predictions['Prediction_Status']}")
                 all_features_valid = False
                 break
            input_full[col] = impute_val
            imputed_features_list.append(col)

    if not all_features_valid:
        return predictions, target_predictions # Return early if imputation or input values failed

    predictions['Imputed_Features'] = imputed_features_list
    if imputed_features_list:
        print(f"  Imputed values for: {imputed_features_list}")

    # Create DataFrame in correct order
    try:
        input_df = pd.DataFrame([input_full])[SPECTRAL_COLS]
    except Exception as e:
         predictions['Prediction_Status'] = f"Error creating input DataFrame: {e}"
         print(f"  {predictions['Prediction_Status']}")
         return predictions, target_predictions

    # --- Load and Apply Scaler ---
    scaler = loaded_scalers.get(water_level)
    if scaler is None:
         predictions['Prediction_Status'] = f"Error: Scaler not found for WL {water_level}."
         print(f"  {predictions['Prediction_Status']}")
         return predictions, target_predictions
    try:
        input_scaled = scaler.transform(input_df)
        print(f"  Applied scaler for WL {water_level}.")
    except Exception as e:
         predictions['Prediction_Status'] = f"Error applying scaler for WL {water_level}: {e}"
         print(f"  {predictions['Prediction_Status']}")
         return predictions, target_predictions

    # --- Load Models and Predict ---
    all_preds_successful = True
    models_for_wl = loaded_models.get(water_level, {})

    for target in TARGET_COLS:
        target_pred_val = None # Use None for missing/error
        model = models_for_wl.get(target)

        if model is None:
            print(f"  Warning: Model not found/loaded for '{target}' at WL {water_level}. Skipping.")
            target_pred_val = None # Indicate model missing
            all_preds_successful = False # Mark as partial if any model is missing
        else:
            try:
                pred = model.predict(input_scaled)[0]
                # Rounding for cleaner output (optional, frontend can also format)
                # Adjust precision based on target variable nature
                if target in ['Ph', 'Temp']:
                    target_pred_val = round(pred, 2)
                elif target in ['Nitro', 'Posh Nitro', 'Pota Nitro', 'EC']:
                     target_pred_val = round(pred, 3)
                else: # Moist, Cap Moist
                     target_pred_val = round(pred, 1)

                print(f"  Predicted {target:<18}: {target_pred_val}")
            except Exception as e:
                print(f"  Error predicting '{target}' for WL {water_level}: {e}")
                target_pred_val = None # Indicate prediction error
                all_preds_successful = False

        target_predictions[target] = target_pred_val

    # Final status update
    if all_preds_successful and all(v is not None for v in target_predictions.values()):
         predictions['Prediction_Status'] = 'Success'
    elif any(v is not None for v in target_predictions.values()):
         predictions['Prediction_Status'] = 'Partial Success (Some models/predictions failed or missing)'
    else:
         predictions['Prediction_Status'] = 'Failed (All predictions failed or critical error)'

    return predictions, target_predictions


# --- Main Initialization Function (Called by Flask app) ---
def initialize_application():
    """
    Loads data, trains/loads models & artifacts.
    This should run only once.
    """
    global _is_initialized, _scalers, _imputation_values, _tuned_models, _performance_metrics, _feature_rankings
    with _init_lock: # Ensure thread safety during init
        if _is_initialized:
            print("Application already initialized.")
            return True

        print("="*30 + " Initializing Application " + "="*30)
        start_init_time = time.time()
        try:
            _create_dirs()
            df = _load_data()
            X_train, X_test, y_train, y_test = _split_data(df)

            # Try loading artifacts first
            artifacts_loaded = _load_artifacts()

            if not artifacts_loaded:
                print("Artifacts not found or incomplete. Running training and evaluation...")
                # Prepare scalers and imputation (always based on current train split)
                _scalers, _imputation_values = _prepare_scalers_imputation(X_train)

                # Train/Evaluate (loads/runs Optuna, trains models, evaluates, calculates rankings)
                _tuned_models, _performance_metrics, _feature_rankings = _train_and_evaluate(
                    X_train, y_train, X_test, y_test, _scalers
                )
                # Ensure models are loaded into the global state correctly
                # (The return value _tuned_models should be assigned globally)

            else:
                print("Successfully loaded all required artifacts.")

            _is_initialized = True
            init_duration = time.time() - start_init_time
            print(f"Application Initialization Complete. Duration: {init_duration:.2f} seconds.")
            return True

        except Exception as e:
            print(f"FATAL: Application initialization failed: {e}")
            _is_initialized = False # Mark as not initialized on failure
            # Consider raising the exception or returning False to signal failure
            return False

def _load_artifacts():
    """Attempts to load all necessary artifacts from disk."""
    global _scalers, _imputation_values, _tuned_models, _performance_metrics, _feature_rankings
    print("Attempting to load pre-existing artifacts...")
    all_loaded = True

    # 1. Scalers
    _scalers = {}
    for wl in WATER_LEVELS_TO_PROCESS:
        scaler_filename = os.path.join(SCALER_SAVE_DIR, f"scaler_wl{wl}.joblib")
        if os.path.exists(scaler_filename):
            try:
                _scalers[wl] = joblib.load(scaler_filename)
            except Exception as e:
                print(f"  Error loading scaler for WL {wl}: {e}")
                all_loaded = False
                _scalers[wl] = None # Mark as missing
        else:
            print(f"  Scaler file missing for WL {wl}")
            all_loaded = False
            _scalers[wl] = None

    # 2. Imputation Values
    _imputation_values = {}
    for wl in WATER_LEVELS_TO_PROCESS:
        impute_filename = os.path.join(IMPUTE_SAVE_DIR, f"impute_means_wl{wl}.json")
        if os.path.exists(impute_filename):
            try:
                with open(impute_filename, 'r') as f:
                    means = json.load(f)
                 # Quick check for validity (optional but good)
                if not isinstance(means, dict) or any(v is None or np.isnan(v) for v in means.values()):
                     raise ValueError("Invalid format or NaN values in imputation file.")
                _imputation_values[wl] = means
            except Exception as e:
                print(f"  Error loading imputation values for WL {wl}: {e}")
                all_loaded = False
                _imputation_values[wl] = {col: np.nan for col in SPECTRAL_COLS} # Mark as invalid
        else:
             print(f"  Imputation file missing for WL {wl}")
             all_loaded = False
             _imputation_values[wl] = {col: np.nan for col in SPECTRAL_COLS}

    # 3. Models
    _tuned_models = defaultdict(dict)
    for wl in WATER_LEVELS_TO_PROCESS:
        for target in TARGET_COLS:
            model_filename = os.path.join(MODEL_SAVE_DIR, f"model_tuned_{target.replace(' ', '_')}_WL{wl}ml.joblib")
            if os.path.exists(model_filename):
                try:
                    _tuned_models[wl][target] = joblib.load(model_filename)
                except Exception as e:
                    print(f"  Error loading model for WL {wl}, Target {target}: {e}")
                    # Don't necessarily set all_loaded to False, prediction can handle missing models
                    _tuned_models[wl][target] = None # Mark as missing
            else:
                 # It's okay if some models don't exist (e.g., constant target)
                 # print(f"  Model file missing for WL {wl}, Target {target} (Might be expected)")
                 _tuned_models[wl][target] = None

    # Check if *any* model was loaded (useful if training failed entirely before)
    if not any(_tuned_models[wl].get(target) for wl in WATER_LEVELS_TO_PROCESS for target in TARGET_COLS):
         print("  Warning: No trained models were loaded successfully.")
         all_loaded = False # Consider this a failure if *no* models are available

    # 4. Performance Metrics
    if os.path.exists(PERFORMANCE_METRICS_FILE):
        try:
            with open(PERFORMANCE_METRICS_FILE, 'r') as f:
                _performance_metrics = json.load(f)
            # Basic check
            if not isinstance(_performance_metrics, dict) or not all(k in _performance_metrics for k in ['R2', 'MAE', 'RMSE']):
                raise ValueError("Invalid performance metrics file format.")
        except Exception as e:
            print(f"  Error loading performance metrics: {e}")
            all_loaded = False
            _performance_metrics = {}
    else:
        print(f"  Performance metrics file missing: {PERFORMANCE_METRICS_FILE}")
        all_loaded = False
        _performance_metrics = {}

    # 5. Feature Rankings
    if os.path.exists(FEATURE_RANKING_FILE):
        try:
            with open(FEATURE_RANKING_FILE, 'r') as f:
                _feature_rankings = json.load(f)
            # Basic check
            if not isinstance(_feature_rankings, dict):
                 raise ValueError("Invalid feature ranking file format.")
        except Exception as e:
            print(f"  Error loading feature rankings: {e}")
            all_loaded = False
            _feature_rankings = {}
    else:
        print(f"  Feature ranking file missing: {FEATURE_RANKING_FILE}")
        all_loaded = False
        _feature_rankings = {}

    print(f"Artifact loading attempt finished. Overall success: {all_loaded}")
    return all_loaded


# --- Getter Functions for Flask App ---
def get_status():
    """Returns the initialization status."""
    return _is_initialized

def get_performance_metrics():
    """Returns the loaded performance metrics."""
    if not _is_initialized: return {"error": "Application not initialized"}
    return _performance_metrics

def get_feature_rankings():
    """Returns the loaded feature rankings."""
    if not _is_initialized: return {"error": "Application not initialized"}
    return _feature_rankings

def run_prediction(input_spectral_data, water_level):
    """Runs prediction using loaded artifacts."""
    if not _is_initialized:
         # Should not happen if app ensures init before handling requests
        return {"Prediction_Status": "Error: Application not initialized"}, {}
    return predict_soil_properties_flexible_internal(
        input_spectral_data,
        water_level,
        _tuned_models,
        _scalers,
        _imputation_values
    )