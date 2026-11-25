
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.stats import randint, uniform, loguniform
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "processed" / "district_splits"

OUTPUT_ROOT = ROOT / "results" / "district_xgb_arma_rebuilt"

METRICS_DIR = OUTPUT_ROOT / "metrics"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
IMPORTANCE_DIR = OUTPUT_ROOT / "feature_importance"
CV_DIR = OUTPUT_ROOT / "cv_results"

REGIONS = ["north_goa", "south_goa"]
TEST_YEARS = [2018, 2019]

TARGET = "Malaria_Cases"

CLIMATE_FEATURES = [
    "max_temperature",
    "min_temperature",
    "total_rainfall",
    "avg_relative_humidity",
    "max_pressure",
    "min_wind_speed",
]

SEASONAL_PERIOD = 12
RANDOM_STATE = 42

# Candidate SARIMA specifications.
# The first is the hybrid order reported by Hussain et al.
SARIMA_CANDIDATES = [
    ((1, 1, 1), (0, 1, 1, 12)),
    ((2, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 2), (0, 1, 1, 12)),
    ((1, 1, 1), (1, 1, 1, 12)),
]

# Three expanding-window folds, each forecasting 12 months.
N_SPLITS = 3
VALIDATION_HORIZON = 12

# Number of XGBoost configurations evaluated.
N_PARAMETER_CONFIGURATIONS = 30

PARAM_DISTRIBUTIONS = {
    "max_depth": randint(1, 7),
    "n_estimators": randint(100, 801),
    "learning_rate": loguniform(0.01, 0.30),
    "gamma": uniform(0.0, 1.0),
    "subsample": uniform(0.60, 0.40),
    "colsample_bytree": uniform(0.60, 0.40),
    "min_child_weight": randint(1, 9),
    "reg_alpha": loguniform(1e-4, 10.0),
    "reg_lambda": loguniform(1e-2, 100.0),
}


# ============================================================
# BOX-COX LAMBDA ESTIMATION
# ============================================================

def estimate_guerrero_lambda(values, period=12):

    values = np.asarray(values, dtype=float)

    if np.any(values <= 0):
        raise ValueError(
            "Box-Cox requires strictly positive observations."
        )

    seasonal_means = []
    seasonal_stds = []

    for month in range(period):

        subseries = values[month::period]

        if len(subseries) >= 2:
            seasonal_means.append(np.mean(subseries))
            seasonal_stds.append(np.std(subseries, ddof=1))

    means = np.asarray(seasonal_means)
    stds = np.asarray(seasonal_stds)

    if (
        len(means) < 2
        or np.any(means <= 0)
        or np.any(stds <= 0)
    ):
        raise ValueError(
            "Insufficient valid seasonal subseries for Box-Cox lambda."
        )

    def objective(lmbda):

        adjusted = stds / (means ** (1 - lmbda))
        mean_adjusted = np.mean(adjusted)

        if mean_adjusted <= 0:
            return np.inf

        return np.std(adjusted, ddof=1) / mean_adjusted

    result = minimize_scalar(
        objective,
        bounds=(-1.0, 2.0),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError("Box-Cox lambda estimation failed.")

    return float(result.x)


# ============================================================
# DATA LOADING
# ============================================================

def load_split(region, year):

    train_path = SPLIT_DIR / f"{region}_{year}_train.csv"
    test_path = SPLIT_DIR / f"{region}_{year}_test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    train = train.sort_values("Date").reset_index(drop=True)
    test = test.sort_values("Date").reset_index(drop=True)

    required = [TARGET] + CLIMATE_FEATURES

    for column in required:

        if column not in train.columns or column not in test.columns:
            raise KeyError(f"Missing required column: {column}")

    if train[required].isna().any().any():
        raise ValueError("Missing values in training data.")

    if test[required].isna().any().any():
        raise ValueError("Missing values in test data.")

    if len(test) != 12:
        raise ValueError(
            f"Expected 12 test observations, found {len(test)}."
        )

    if np.any(train[TARGET].to_numpy(dtype=float) <= 0):
        raise ValueError(
            "Training malaria counts must be positive for Box-Cox."
        )

    return train, test


# ============================================================
# SARIMA ORDER SELECTION
# Select by AIC using only the supplied training observations.
# ============================================================

def select_sarima(values, verbose=False):

    values = np.asarray(values, dtype=float)

    best_fit = None
    best_order = None
    best_seasonal_order = None
    best_aic = np.inf

    for order, seasonal_order in SARIMA_CANDIDATES:

        try:

            model = SARIMAX(
                values,
                order=order,
                seasonal_order=seasonal_order,
                trend="n",
                enforce_stationarity=True,
                enforce_invertibility=True,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fitted = model.fit(
                    disp=False,
                    maxiter=300,
                )

            if (
                np.isfinite(fitted.aic)
                and fitted.aic < best_aic
            ):
                best_fit = fitted
                best_order = order
                best_seasonal_order = seasonal_order
                best_aic = float(fitted.aic)

        except Exception as error:

            if verbose:
                print(
                    f"SARIMA {order} x {seasonal_order} failed: "
                    f"{repr(error)}"
                )

    if best_fit is None:
        raise RuntimeError(
            "None of the candidate SARIMA models fitted successfully."
        )

    return (
        best_fit,
        best_order,
        best_seasonal_order,
        best_aic,
    )


# ============================================================
# FEATURE CONSTRUCTION
#
# p=1: one response lag
# q=1: one non-seasonal residual lag
# Q=1: one seasonal residual lag
#
# The residuals are one-step SARIMA innovations.
# ============================================================

def build_training_features(
    transformed_y,
    sarima_residuals,
    climate,
):

    rows = []
    targets = []

    for t in range(SEASONAL_PERIOD, len(transformed_y)):

        row = {
            "response_lag_1": transformed_y[t - 1],
            "residual_lag_1": sarima_residuals[t - 1],
            "residual_lag_12": sarima_residuals[t - 12],
        }

        for feature in CLIMATE_FEATURES:
            row[feature] = float(climate.iloc[t][feature])

        rows.append(row)
        targets.append(transformed_y[t])

    X = pd.DataFrame(rows)
    y = np.asarray(targets, dtype=float)

    if len(X) == 0:
        raise ValueError("No hybrid training rows were generated.")

    if not np.all(np.isfinite(X.to_numpy(dtype=float))):
        raise ValueError("Non-finite hybrid training features.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite hybrid training targets.")

    return X, y


# ============================================================
# XGBOOST MODEL
# ============================================================

def create_model(parameters):

    xgb_pipeline = Pipeline([
        ("scale", StandardScaler()),
        (
            "xgb",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **parameters,
            ),
        ),
    ])

    return TransformedTargetRegressor(
        regressor=xgb_pipeline,
        transformer=StandardScaler(),
    )


# ============================================================
# RECURSIVE WALK-FORWARD FORECAST
#
# The model predicts transformed malaria incidence.
# The generated response is appended to the SARIMA state.
# The next residual feature is the generated response minus
# the corresponding one-step SARIMA forecast.
#
# No actual validation/test malaria values enter recursion.
# ============================================================

def recursive_forecast(
    fitted_model,
    sarima_fit,
    transformed_history,
    residual_history,
    future_climate,
):

    sarima_state = sarima_fit

    response_history = list(
        np.asarray(transformed_history, dtype=float)
    )

    residual_history = list(
        np.asarray(residual_history, dtype=float)
    )

    forecasts = []
    sarima_forecasts = []

    for step in range(len(future_climate)):

        sarima_prediction = float(
            np.asarray(
                sarima_state.get_forecast(steps=1).predicted_mean
            ).reshape(-1)[0]
        )

        feature_row = {
            "response_lag_1": response_history[-1],
            "residual_lag_1": residual_history[-1],
            "residual_lag_12": residual_history[-12],
        }

        for feature in CLIMATE_FEATURES:
            feature_row[feature] = float(
                future_climate.iloc[step][feature]
            )

        X_current = pd.DataFrame([feature_row])

        hybrid_prediction = float(
            fitted_model.predict(X_current)[0]
        )

        forecasts.append(hybrid_prediction)
        sarima_forecasts.append(sarima_prediction)

        generated_innovation = (
            hybrid_prediction - sarima_prediction
        )

        sarima_state = sarima_state.append(
            endog=np.asarray([hybrid_prediction]),
            refit=False,
        )

        response_history.append(hybrid_prediction)
        residual_history.append(generated_innovation)

    return (
        np.asarray(forecasts, dtype=float),
        np.asarray(sarima_forecasts, dtype=float),
    )


# ============================================================
# PREPARE ONE CHRONOLOGICAL VALIDATION FOLD
#
# Every fold independently:
#   1. estimates Box-Cox lambda
#   2. selects SARIMA by AIC
#   3. fits SARIMA
#   4. generates training residuals and hybrid features
#
# Validation forecasts are then generated recursively.
# ============================================================

def prepare_validation_fold(train, validation):

    y_raw = train[TARGET].to_numpy(dtype=float)

    lambda_value = estimate_guerrero_lambda(
        y_raw,
        period=SEASONAL_PERIOD,
    )

    y_transformed = boxcox(
        y_raw,
        lmbda=lambda_value,
    )

    (
        sarima_fit,
        order,
        seasonal_order,
        aic,
    ) = select_sarima(y_transformed)

    residuals = np.asarray(
        sarima_fit.resid,
        dtype=float,
    )

    X_train, y_hybrid = build_training_features(
        transformed_y=y_transformed,
        sarima_residuals=residuals,
        climate=train,
    )

    return {
        "lambda": lambda_value,
        "sarima_fit": sarima_fit,
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": aic,
        "X_train": X_train,
        "y_train": y_hybrid,
        "transformed_history": y_transformed,
        "residual_history": residuals,
        "validation": validation.reset_index(drop=True),
    }


# ============================================================
# VALIDATION SCORE
# Original-scale RMSE over all rolling validation forecasts.
# ============================================================

def evaluate_parameters(parameters, fold_artifacts):

    all_actual = []
    all_predicted = []

    for fold in fold_artifacts:

        model = create_model(parameters)

        model.fit(
            fold["X_train"],
            fold["y_train"],
        )

        transformed_predictions, _ = recursive_forecast(
            fitted_model=model,
            sarima_fit=fold["sarima_fit"],
            transformed_history=fold["transformed_history"],
            residual_history=fold["residual_history"],
            future_climate=fold["validation"],
        )

        lambda_value = fold["lambda"]

        inverse_domain = (
            1.0 + lambda_value * transformed_predictions
        )

        if (
            np.any(inverse_domain <= 0)
            or not np.all(np.isfinite(transformed_predictions))
        ):
            return np.inf

        with np.errstate(invalid="ignore", over="ignore"):

            predictions = inv_boxcox(
                transformed_predictions,
                lambda_value,
            )

        if not np.all(np.isfinite(predictions)):
            return np.inf

        actual = fold["validation"][TARGET].to_numpy(dtype=float)

        all_actual.extend(actual.tolist())
        all_predicted.extend(predictions.tolist())

    return float(
        np.sqrt(
            mean_squared_error(
                np.asarray(all_actual),
                np.asarray(all_predicted),
            )
        )
    )


# ============================================================
# BUILD ROLLING VALIDATION FOLDS
# ============================================================

def build_validation_folds(train):

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_HORIZON,
    )

    artifacts = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(train),
        start=1,
    ):

        fold_train = (
            train.iloc[train_indices]
            .reset_index(drop=True)
        )

        fold_validation = (
            train.iloc[validation_indices]
            .reset_index(drop=True)
        )

        print(
            f"Preparing fold {fold_number}: "
            f"{len(fold_train)} training months, "
            f"{len(fold_validation)} validation months"
        )

        artifact = prepare_validation_fold(
            fold_train,
            fold_validation,
        )

        artifacts.append(artifact)

    return artifacts


# ============================================================
# HYPERPARAMETER OPTIMISATION
# ============================================================

def optimise_xgboost(train, region, year):

    fold_artifacts = build_validation_folds(train)

    parameter_candidates = list(
        ParameterSampler(
            PARAM_DISTRIBUTIONS,
            n_iter=N_PARAMETER_CONFIGURATIONS,
            random_state=RANDOM_STATE,
        )
    )

    results = []

    best_parameters = None
    best_rmse = np.inf

    print(
        f"\nEvaluating {len(parameter_candidates)} XGBoost "
        f"configurations across {len(fold_artifacts)} "
        "rolling-origin folds."
    )

    for candidate_number, parameters in enumerate(
        parameter_candidates,
        start=1,
    ):

        score = evaluate_parameters(
            parameters,
            fold_artifacts,
        )

        results.append({
            "candidate": candidate_number,
            "validation_rmse_original_scale": score,
            **parameters,
        })

        print(
            f"Candidate {candidate_number:02d}/"
            f"{len(parameter_candidates)} | "
            f"Rolling RMSE: {score:.4f}"
        )

        if score < best_rmse:
            best_rmse = score
            best_parameters = parameters.copy()

    if best_parameters is None or not np.isfinite(best_rmse):
        raise RuntimeError(
            "No valid XGBoost configuration was found."
        )

    cv_results = pd.DataFrame(results).sort_values(
        "validation_rmse_original_scale"
    )

    cv_path = (
        CV_DIR / f"xgb_arma_{region}_{year}_rolling_cv.csv"
    )

    cv_results.to_csv(cv_path, index=False)

    print("\nBEST XGBOOST PARAMETERS")
    print(best_parameters)
    print(f"Rolling validation RMSE: {best_rmse:.4f}")

    return best_parameters, best_rmse


# ============================================================
# FINAL DISTRICT/YEAR EXPERIMENT
# ============================================================

def run_experiment(region, year):

    print("\n" + "=" * 78)
    print(f"REBUILT XGB-ARMA: {region.upper()} | {year}")
    print("=" * 78)

    train, test = load_split(region, year)

    print(f"Outer training observations: {len(train)}")
    print(f"Outer test observations: {len(test)}")

    # --------------------------------------------------------
    # 1. OPTIMISE XGBOOST USING TRAINING DATA ONLY
    # --------------------------------------------------------

    best_parameters, best_cv_rmse = optimise_xgboost(
        train,
        region,
        year,
    )

    # --------------------------------------------------------
    # 2. FIT FINAL BOX-COX TRANSFORMATION
    # --------------------------------------------------------

    y_train_raw = train[TARGET].to_numpy(dtype=float)

    lambda_value = estimate_guerrero_lambda(
        y_train_raw,
        period=SEASONAL_PERIOD,
    )

    y_train_transformed = boxcox(
        y_train_raw,
        lmbda=lambda_value,
    )

    print(f"\nFinal training-only Box-Cox lambda: {lambda_value:.6f}")

    # --------------------------------------------------------
    # 3. SELECT AND FIT FINAL SARIMA MODEL
    # --------------------------------------------------------

    (
        sarima_fit,
        sarima_order,
        seasonal_order,
        sarima_aic,
    ) = select_sarima(
        y_train_transformed,
        verbose=True,
    )

    residuals = np.asarray(
        sarima_fit.resid,
        dtype=float,
    )

    print(f"Selected SARIMA order: {sarima_order}")
    print(f"Selected seasonal order: {seasonal_order}")
    print(f"Selected SARIMA AIC: {sarima_aic:.4f}")

    # --------------------------------------------------------
    # 4. BUILD FINAL HYBRID TRAINING MATRIX
    # --------------------------------------------------------

    X_train, y_hybrid_train = build_training_features(
        transformed_y=y_train_transformed,
        sarima_residuals=residuals,
        climate=train,
    )

    print(f"Final hybrid training rows: {len(X_train)}")
    print(f"Hybrid features: {list(X_train.columns)}")

    # --------------------------------------------------------
    # 5. FIT FINAL XGBOOST
    # --------------------------------------------------------

    final_model = create_model(best_parameters)

    final_model.fit(
        X_train,
        y_hybrid_train,
    )

    # --------------------------------------------------------
    # 6. RECURSIVE 12-MONTH TEST FORECAST
    # --------------------------------------------------------

    hybrid_transformed, sarima_transformed = recursive_forecast(
        fitted_model=final_model,
        sarima_fit=sarima_fit,
        transformed_history=y_train_transformed,
        residual_history=residuals,
        future_climate=test,
    )

    # --------------------------------------------------------
    # 7. INVERSE BOX-COX
    # --------------------------------------------------------

    hybrid_domain = (
        1.0 + lambda_value * hybrid_transformed
    )

    sarima_domain = (
        1.0 + lambda_value * sarima_transformed
    )

    with np.errstate(invalid="ignore", over="ignore"):

        hybrid_original = inv_boxcox(
            hybrid_transformed,
            lambda_value,
        )

        sarima_original = inv_boxcox(
            sarima_transformed,
            lambda_value,
        )

    hybrid_valid = (
        np.isfinite(hybrid_original)
        & (hybrid_domain > 0)
    )

    sarima_valid = (
        np.isfinite(sarima_original)
        & (sarima_domain > 0)
    )

    actual = test[TARGET].to_numpy(dtype=float)

    if hybrid_valid.all():

        rmse = float(
            np.sqrt(
                mean_squared_error(actual, hybrid_original)
            )
        )

        mae = float(
            mean_absolute_error(actual, hybrid_original)
        )

        status = "valid"

        print("\nHYBRID TEST PERFORMANCE")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

    else:

        rmse = np.nan
        mae = np.nan
        status = "invalid_inverse_boxcox_domain"

        invalid_dates = test.loc[
            ~hybrid_valid, "Date"
        ].dt.strftime("%Y-%m").tolist()

        print("\nInvalid hybrid inverse Box-Cox forecasts:")
        print(invalid_dates)

    # --------------------------------------------------------
    # 8. SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "Date": test["Date"],
        "Actual": actual,
        "SARIMA_Predicted": sarima_original,
        "Hybrid_Predicted": hybrid_original,
        "SARIMA_Predicted_Transformed": sarima_transformed,
        "Hybrid_Predicted_Transformed": hybrid_transformed,
        "SARIMA_Valid": sarima_valid,
        "Hybrid_Valid": hybrid_valid,
    })

    prediction_path = (
        PREDICTIONS_DIR
        / f"xgb_arma_{region}_{year}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # 9. SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    fitted_xgb = (
        final_model.regressor_
        .named_steps["xgb"]
    )

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": fitted_xgb.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False,
    )

    importance_path = (
        IMPORTANCE_DIR
        / f"xgb_arma_{region}_{year}_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    print("\nFeature importance:")
    print(importance_df.to_string(index=False))

    print(f"\nPredictions saved: {prediction_path}")
    print(f"Feature importance saved: {importance_path}")

    return {
        "region": region,
        "test_year": year,
        "training_size": len(train),
        "hybrid_training_size": len(X_train),
        "test_size": len(test),
        "lambda": lambda_value,
        "sarima_order": str(sarima_order),
        "sarima_seasonal_order": str(seasonal_order),
        "sarima_aic": sarima_aic,
        "rolling_cv_rmse_original_scale": best_cv_rmse,
        "best_parameters": str(best_parameters),
        "RMSE": rmse,
        "MAE": mae,
        "status": status,
        "invalid_months": ";".join(
            test.loc[~hybrid_valid, "Date"]
            .dt.strftime("%Y-%m").tolist()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    for directory in [
        METRICS_DIR,
        PREDICTIONS_DIR,
        IMPORTANCE_DIR,
        CV_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    results = []

    for region in REGIONS:

        for year in TEST_YEARS:

            try:

                result = run_experiment(region, year)
                results.append(result)

            except Exception as error:

                print("\nEXPERIMENT FAILED")
                print(f"{region} | {year}")
                print(repr(error))

                results.append({
                    "region": region,
                    "test_year": year,
                    "status": "failed",
                    "error": repr(error),
                })

    metrics = pd.DataFrame(results)

    metrics_path = (
        METRICS_DIR / "district_xgb_arma_rebuilt_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    print("\n" + "=" * 78)
    print("FINAL REBUILT DISTRICT XGB-ARMA RESULTS")
    print("=" * 78)

    columns = [
        "region",
        "test_year",
        "training_size",
        "hybrid_training_size",
        "sarima_order",
        "sarima_seasonal_order",
        "rolling_cv_rmse_original_scale",
        "RMSE",
        "MAE",
        "status",
    ]

    print(
        metrics[
            [column for column in columns if column in metrics.columns]
        ].to_string(index=False)
    )

    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
