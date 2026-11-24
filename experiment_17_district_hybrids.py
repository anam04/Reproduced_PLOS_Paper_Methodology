
from pathlib import Path
from itertools import product
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.stats import boxcox, randint, uniform, loguniform
from scipy.special import inv_boxcox

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


# ============================================================
# EXPERIMENT 17: DISTRICT-SPECIFIC HYBRID MODELS
#
# North Goa: RBF-SVR-ARMA
# South Goa: XGBoost-ARMA
#
# Four independent evaluations:
# North 2018, North 2019, South 2018, South 2019.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "processed" / "district_splits"
SOURCE_DATA = (
    ROOT / "data" / "processed"
    / "goa_district_malaria_climate_2010_2019.csv"
)

OUTPUT_ROOT = ROOT / "results" / "experiment_17_district_hybrids"

METRICS_DIR = OUTPUT_ROOT / "metrics"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
CV_DIR = OUTPUT_ROOT / "cv_results"
IMPORTANCE_DIR = OUTPUT_ROOT / "feature_importance"

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

FEATURE_COLUMNS = [
    "response_lag_1",
    "innovation_lag_1",
    "innovation_lag_12",
] + CLIMATE_FEATURES

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)

SEASONAL_PERIOD = 12
RANDOM_STATE = 42

N_SPLITS = 3
VALIDATION_HORIZON = 12

N_XGB_CANDIDATES = 30

XGB_PARAMETER_DISTRIBUTIONS = {
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

SVR_PARAMETER_GRID = list(product(
    [0.1, 1.0, 10.0, 100.0],
    ["scale", 0.01, 0.1, 1.0],
    [0.01, 0.1, 0.2],
))


# ============================================================
# DATA LOADING AND PRESSURE VARIABLE
# ============================================================

def normalize_region(value):
    value = str(value).strip().lower()
    value = value.replace("_", "").replace(" ", "")
    return value


def load_pressure_lookup():

    if not SOURCE_DATA.exists():
        raise FileNotFoundError(
            f"Processed district dataset not found: {SOURCE_DATA}"
        )

    source = pd.read_csv(SOURCE_DATA)

    required = {"Date", "Region", "max_pressure"}
    missing = required - set(source.columns)

    if missing:
        raise KeyError(
            f"Source dataset is missing required fields: {sorted(missing)}"
        )

    source["Date"] = pd.to_datetime(source["Date"])
    source["region_key"] = source["Region"].map(normalize_region)

    lookup = source[
        ["Date", "region_key", "max_pressure"]
    ].copy()

    if lookup.duplicated(["Date", "region_key"]).any():
        raise ValueError(
            "Duplicate district/date entries in pressure lookup."
        )

    return lookup


def add_minimum_pressure(frame, region, pressure_lookup):

    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])

    if "max_pressure" in frame.columns:
        if frame["max_pressure"].isna().any():
            raise ValueError("Existing max_pressure contains missing values.")
        return frame

    region_key = normalize_region(region)

    lookup = pressure_lookup[
        pressure_lookup["region_key"] == region_key
    ][["Date", "max_pressure"]]

    frame = frame.merge(
        lookup,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    if frame["max_pressure"].isna().any():
        missing_dates = frame.loc[
            frame["max_pressure"].isna(), "Date"
        ].dt.strftime("%Y-%m-%d").tolist()

        raise ValueError(
            f"Could not match max_pressure for {region}: {missing_dates}"
        )

    return frame


def load_split(region, year, pressure_lookup):

    train_path = SPLIT_DIR / f"{region}_{year}_train.csv"
    test_path = SPLIT_DIR / f"{region}_{year}_test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = add_minimum_pressure(train, region, pressure_lookup)
    test = add_minimum_pressure(test, region, pressure_lookup)

    for frame in (train, test):
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)

        required = [TARGET] + CLIMATE_FEATURES
        missing = set(required) - set(frame.columns)

        if missing:
            raise KeyError(
                f"{region} {year}: missing columns {sorted(missing)}"
            )

        if frame[required].isna().any().any():
            raise ValueError(
                f"{region} {year}: missing values in model inputs."
            )

    if len(test) != 12:
        raise ValueError(
            f"{region} {year}: expected 12 test months, got {len(test)}."
        )

    if np.any(train[TARGET].to_numpy(dtype=float) <= 0):
        raise ValueError(
            f"{region} {year}: Box-Cox requires positive training counts."
        )

    return train, test


# ============================================================
# BOX-COX TRANSFORMATION
# ============================================================

def estimate_guerrero_lambda(values, period=12):

    values = np.asarray(values, dtype=float)

    if np.any(values <= 0):
        raise ValueError("Box-Cox requires strictly positive values.")

    means = []
    stds = []

    for month in range(period):
        subseries = values[month::period]

        if len(subseries) >= 2:
            means.append(np.mean(subseries))
            stds.append(np.std(subseries, ddof=1))

    means = np.asarray(means)
    stds = np.asarray(stds)

    if (
        len(means) < 2
        or np.any(means <= 0)
        or np.any(stds <= 0)
    ):
        raise ValueError("Invalid seasonal subseries for Box-Cox.")

    def objective(lmbda):
        adjusted = stds / (means ** (1.0 - lmbda))
        return np.std(adjusted, ddof=1) / np.mean(adjusted)

    result = minimize_scalar(
        objective,
        bounds=(-1.0, 2.0),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError("Box-Cox lambda estimation failed.")

    return float(result.x)


# ============================================================
# SARIMA AND ONE-STEP INNOVATIONS
# ============================================================

def fit_sarima(transformed_y):

    model = SARIMAX(
        np.asarray(transformed_y, dtype=float),
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp=False, maxiter=300)

    return fitted


def extract_one_step_innovations(fitted, transformed_y):

    transformed_y = np.asarray(transformed_y, dtype=float)

    one_step_predictions = np.asarray(
        fitted.get_prediction(
            start=0,
            end=len(transformed_y) - 1,
            dynamic=False,
        ).predicted_mean,
        dtype=float,
    ).reshape(-1)

    innovations = transformed_y - one_step_predictions

    if not np.all(np.isfinite(innovations)):
        raise ValueError("Non-finite SARIMA innovations.")

    return innovations


# ============================================================
# HYBRID TRAINING FEATURES
# ============================================================

def build_training_features(transformed_y, innovations, climate):

    rows = []
    targets = []

    for t in range(SEASONAL_PERIOD, len(transformed_y)):

        row = {
            "response_lag_1": transformed_y[t - 1],
            "innovation_lag_1": innovations[t - 1],
            "innovation_lag_12": innovations[t - 12],
        }

        for feature in CLIMATE_FEATURES:
            row[feature] = float(climate.iloc[t][feature])

        rows.append(row)
        targets.append(transformed_y[t])

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = np.asarray(targets, dtype=float)

    if not np.all(np.isfinite(X.to_numpy(dtype=float))):
        raise ValueError("Non-finite hybrid training features.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite hybrid training targets.")

    return X, y


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def create_svr_model(parameters):

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        (
            "svr",
            SVR(
                kernel="rbf",
                C=float(parameters["C"]),
                gamma=parameters["gamma"],
                epsilon=float(parameters["epsilon"]),
            ),
        ),
    ])

    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


def create_xgb_model(parameters):

    pipeline = Pipeline([
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
        regressor=pipeline,
        transformer=StandardScaler(),
    )


def create_model(region, parameters):

    if region == "north_goa":
        return create_svr_model(parameters)

    if region == "south_goa":
        return create_xgb_model(parameters)

    raise ValueError(f"Unknown region: {region}")


# ============================================================
# RECURSIVE FORECAST
#
# Two independent SARIMA states are maintained:
# 1. Hybrid state receives hybrid-generated predictions.
# 2. Baseline state receives SARIMA-generated predictions.
#
# This ensures the SARIMA baseline is a genuine standalone
# recursive forecast, not a forecast from the hybrid-updated state.
# ============================================================

def recursive_forecast(
    model,
    sarima_fit,
    transformed_history,
    innovation_history,
    future_climate,
):

    hybrid_state = sarima_fit
    baseline_state = sarima_fit

    response_history = list(
        np.asarray(transformed_history, dtype=float)
    )
    innovations = list(
        np.asarray(innovation_history, dtype=float)
    )

    hybrid_predictions = []
    baseline_predictions = []
    generated_innovations = []

    for step in range(len(future_climate)):

        # Standalone SARIMA forecast.
        baseline_prediction = float(
            np.asarray(
                baseline_state.get_forecast(steps=1).predicted_mean
            ).reshape(-1)[0]
        )

        # SARIMA forecast from the hybrid-updated state.
        hybrid_backbone_prediction = float(
            np.asarray(
                hybrid_state.get_forecast(steps=1).predicted_mean
            ).reshape(-1)[0]
        )

        row = {
            "response_lag_1": response_history[-1],
            "innovation_lag_1": innovations[-1],
            "innovation_lag_12": innovations[-12],
        }

        for feature in CLIMATE_FEATURES:
            row[feature] = float(future_climate.iloc[step][feature])

        X_current = pd.DataFrame(
            [row],
            columns=FEATURE_COLUMNS,
        )

        hybrid_prediction = float(model.predict(X_current)[0])

        hybrid_predictions.append(hybrid_prediction)
        baseline_predictions.append(baseline_prediction)

        # Update the hybrid SARIMA state using the generated hybrid response.
        updated_hybrid_state = hybrid_state.append(
            endog=np.asarray([hybrid_prediction]),
            refit=False,
        )

        # The updated filter's forecast error is the generated innovation.
        filter_errors = np.asarray(
            updated_hybrid_state.filter_results.forecasts_error[0],
            dtype=float,
        ).reshape(-1)

        generated_innovation = float(filter_errors[-1])

        if not np.isfinite(generated_innovation):
            raise ValueError("Non-finite generated innovation.")

        generated_innovations.append(generated_innovation)

        hybrid_state = updated_hybrid_state
        response_history.append(hybrid_prediction)
        innovations.append(generated_innovation)

        # Independently update the baseline with its own forecast.
        baseline_state = baseline_state.append(
            endog=np.asarray([baseline_prediction]),
            refit=False,
        )

    return (
        np.asarray(hybrid_predictions),
        np.asarray(baseline_predictions),
        np.asarray(generated_innovations),
    )


# ============================================================
# VALIDATION FOLD PREPARATION
# ============================================================

def prepare_fold(train):

    raw_y = train[TARGET].to_numpy(dtype=float)

    lmbda = estimate_guerrero_lambda(raw_y)

    transformed_y = boxcox(
        raw_y,
        lmbda=lmbda,
    )

    sarima_fit = fit_sarima(transformed_y)

    innovations = extract_one_step_innovations(
        sarima_fit,
        transformed_y,
    )

    X, y = build_training_features(
        transformed_y,
        innovations,
        train,
    )

    return {
        "lambda": lmbda,
        "sarima_fit": sarima_fit,
        "transformed_y": transformed_y,
        "innovations": innovations,
        "X": X,
        "y": y,
    }


def prepare_validation_folds(train):

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_HORIZON,
    )

    folds = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(train),
        start=1,
    ):

        fold_train = train.iloc[train_idx].reset_index(drop=True)
        fold_valid = train.iloc[valid_idx].reset_index(drop=True)

        print(
            f"  Preparing fold {fold_number}: "
            f"{len(fold_train)} training / "
            f"{len(fold_valid)} validation observations"
        )

        prepared = prepare_fold(fold_train)
        prepared["validation"] = fold_valid

        folds.append(prepared)

    return folds


# ============================================================
# VALIDATION SCORING
# ============================================================

def evaluate_parameters(region, parameters, folds):

    all_actual = []
    all_predicted = []

    for fold in folds:

        model = create_model(region, parameters)
        model.fit(fold["X"], fold["y"])

        transformed_predictions, _, _ = recursive_forecast(
            model=model,
            sarima_fit=fold["sarima_fit"],
            transformed_history=fold["transformed_y"],
            innovation_history=fold["innovations"],
            future_climate=fold["validation"],
        )

        lmbda = fold["lambda"]

        if (
            not np.all(np.isfinite(transformed_predictions))
            or np.any(1.0 + lmbda * transformed_predictions <= 0)
        ):
            return np.inf

        predictions = inv_boxcox(
            transformed_predictions,
            lmbda,
        )

        if not np.all(np.isfinite(predictions)):
            return np.inf

        actual = fold["validation"][TARGET].to_numpy(dtype=float)

        all_actual.extend(actual.tolist())
        all_predicted.extend(predictions.tolist())

    return float(
        np.sqrt(
            mean_squared_error(all_actual, all_predicted)
        )
    )


# ============================================================
# HYPERPARAMETER OPTIMISATION
# ============================================================

def optimise_model(region, train, year):

    print("\nPreparing rolling-origin validation folds.")

    folds = prepare_validation_folds(train)

    if region == "north_goa":

        candidates = [
            {
                "C": C,
                "gamma": gamma,
                "epsilon": epsilon,
            }
            for C, gamma, epsilon in SVR_PARAMETER_GRID
        ]

        model_name = "SVR"

    else:

        candidates = list(
            ParameterSampler(
                XGB_PARAMETER_DISTRIBUTIONS,
                n_iter=N_XGB_CANDIDATES,
                random_state=RANDOM_STATE,
            )
        )

        model_name = "XGBoost"

    print(
        f"\nTuning {model_name}: {len(candidates)} configurations, "
        f"{len(folds)} rolling-origin folds."
    )

    results = []
    best_parameters = None
    best_score = np.inf

    for number, parameters in enumerate(candidates, start=1):

        score = evaluate_parameters(
            region,
            parameters,
            folds,
        )

        results.append({
            "candidate": number,
            "rolling_validation_RMSE": score,
            **parameters,
        })

        print(
            f"  {model_name} {number:02d}/{len(candidates)} "
            f"| RMSE={score:.4f}"
        )

        if score < best_score:
            best_score = score
            best_parameters = parameters.copy()

    if best_parameters is None or not np.isfinite(best_score):
        raise RuntimeError(
            f"No valid {model_name} configuration for {region} {year}."
        )

    results_df = pd.DataFrame(results).sort_values(
        "rolling_validation_RMSE"
    )

    results_df.to_csv(
        CV_DIR / f"experiment_17_{region}_{year}_cv.csv",
        index=False,
    )

    print(f"\nBest {model_name} parameters: {best_parameters}")
    print(f"Best rolling validation RMSE: {best_score:.4f}")

    return best_parameters, best_score


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted, valid):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(actual)
        & np.isfinite(predicted)
    )

    if valid.sum() == 0:
        return np.nan, np.nan, 0

    rmse = float(
        np.sqrt(
            mean_squared_error(actual[valid], predicted[valid])
        )
    )

    mae = float(
        mean_absolute_error(actual[valid], predicted[valid])
    )

    return rmse, mae, int(valid.sum())


# ============================================================
# ONE DISTRICT/YEAR RUN
# ============================================================

def run_experiment(region, year, pressure_lookup):

    print("\n" + "=" * 78)
    print(f"EXPERIMENT 17 | {region.upper()} | {year}")
    print("=" * 78)

    train, test = load_split(
        region,
        year,
        pressure_lookup,
    )

    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print("Climate predictors:", ", ".join(CLIMATE_FEATURES))

    # 1. Select model parameters using training data only.
    best_parameters, best_cv_rmse = optimise_model(
        region,
        train,
        year,
    )

    # 2. Fit final Box-Cox transformation.
    raw_y = train[TARGET].to_numpy(dtype=float)

    lmbda = estimate_guerrero_lambda(raw_y)

    transformed_y = boxcox(
        raw_y,
        lmbda=lmbda,
    )

    # 3. Fit the final SARIMA model.
    sarima_fit = fit_sarima(transformed_y)

    innovations = extract_one_step_innovations(
        sarima_fit,
        transformed_y,
    )

    print(f"\nBox-Cox lambda: {lmbda:.6f}")
    print(f"SARIMA order: {SARIMA_ORDER}")
    print(f"SARIMA seasonal order: {SARIMA_SEASONAL_ORDER}")
    print(f"SARIMA AIC: {sarima_fit.aic:.4f}")

    # 4. Construct hybrid training data.
    X_train, y_train = build_training_features(
        transformed_y,
        innovations,
        train,
    )

    print(f"Hybrid training rows: {len(X_train)}")

    # 5. Fit the district-specific learner.
    final_model = create_model(
        region,
        best_parameters,
    )

    final_model.fit(X_train, y_train)

    # 6. Produce recursive forecasts.
    (
        hybrid_transformed,
        sarima_transformed,
        generated_innovations,
    ) = recursive_forecast(
        model=final_model,
        sarima_fit=sarima_fit,
        transformed_history=transformed_y,
        innovation_history=innovations,
        future_climate=test,
    )

    # 7. Inverse Box-Cox transformation.
    with np.errstate(invalid="ignore", over="ignore"):
        hybrid_original = inv_boxcox(
            hybrid_transformed,
            lmbda,
        )

        sarima_original = inv_boxcox(
            sarima_transformed,
            lmbda,
        )

    hybrid_valid = (
        np.isfinite(hybrid_original)
        & (1.0 + lmbda * hybrid_transformed > 0)
    )

    sarima_valid = (
        np.isfinite(sarima_original)
        & (1.0 + lmbda * sarima_transformed > 0)
    )

    actual = test[TARGET].to_numpy(dtype=float)

    hybrid_rmse, hybrid_mae, hybrid_n = calculate_metrics(
        actual,
        hybrid_original,
        hybrid_valid,
    )

    sarima_rmse, sarima_mae, sarima_n = calculate_metrics(
        actual,
        sarima_original,
        sarima_valid,
    )

    common_valid = hybrid_valid & sarima_valid

    hybrid_common_rmse, hybrid_common_mae, common_n = calculate_metrics(
        actual,
        hybrid_original,
        common_valid,
    )

    sarima_common_rmse, sarima_common_mae, _ = calculate_metrics(
        actual,
        sarima_original,
        common_valid,
    )

    print("\nTEST PERFORMANCE")
    print(
        f"SARIMA: {sarima_rmse:.4f} RMSE | "
        f"{sarima_mae:.4f} MAE | n={sarima_n}"
    )
    print(
        f"{'SVM-ARMA' if region == 'north_goa' else 'XGB-ARMA'}: "
        f"{hybrid_rmse:.4f} RMSE | {hybrid_mae:.4f} MAE | n={hybrid_n}"
    )
    print(
        f"Common months ({common_n}): "
        f"SARIMA RMSE={sarima_common_rmse:.4f} | "
        f"Hybrid RMSE={hybrid_common_rmse:.4f}"
    )

    # 8. Save forecasts.
    prediction_df = pd.DataFrame({
        "Date": test["Date"],
        "Actual": actual,
        "SARIMA_Predicted": sarima_original,
        "Hybrid_Predicted": hybrid_original,
        "SARIMA_Transformed": sarima_transformed,
        "Hybrid_Transformed": hybrid_transformed,
        "Generated_Innovation": generated_innovations,
        "SARIMA_Valid": sarima_valid,
        "Hybrid_Valid": hybrid_valid,
    })

    prediction_df.to_csv(
        PREDICTIONS_DIR
        / f"experiment_17_{region}_{year}_predictions.csv",
        index=False,
    )

    # 9. Save feature importance for XGBoost.
    if region == "south_goa":

        fitted_xgb = (
            final_model.regressor_
            .named_steps["xgb"]
        )

        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": fitted_xgb.feature_importances_,
        }).sort_values("Importance", ascending=False)

        importance_df.to_csv(
            IMPORTANCE_DIR
            / f"experiment_17_{region}_{year}_feature_importance.csv",
            index=False,
        )

        print("\nXGBoost feature importance:")
        print(importance_df.to_string(index=False))

    return {
        "region": region,
        "test_year": year,
        "model": "SVM-ARMA" if region == "north_goa" else "XGB-ARMA",
        "training_size": len(train),
        "hybrid_training_size": len(X_train),
        "test_size": len(test),
        "boxcox_lambda": lmbda,
        "sarima_order": str(SARIMA_ORDER),
        "sarima_seasonal_order": str(SARIMA_SEASONAL_ORDER),
        "sarima_aic": float(sarima_fit.aic),
        "rolling_validation_RMSE": best_cv_rmse,
        "SARIMA_RMSE": sarima_rmse,
        "SARIMA_MAE": sarima_mae,
        "SARIMA_valid_months": sarima_n,
        "Hybrid_RMSE": hybrid_rmse,
        "Hybrid_MAE": hybrid_mae,
        "Hybrid_valid_months": hybrid_n,
        "Common_months": common_n,
        "SARIMA_common_RMSE": sarima_common_rmse,
        "SARIMA_common_MAE": sarima_common_mae,
        "Hybrid_common_RMSE": hybrid_common_rmse,
        "Hybrid_common_MAE": hybrid_common_mae,
        "status": "valid" if hybrid_valid.all() else "partial",
        "best_parameters": str(best_parameters),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    for directory in [
        METRICS_DIR,
        PREDICTIONS_DIR,
        CV_DIR,
        IMPORTANCE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    pressure_lookup = load_pressure_lookup()

    results = []

    for region in REGIONS:
        for year in TEST_YEARS:

            try:
                result = run_experiment(
                    region,
                    year,
                    pressure_lookup,
                )

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
        METRICS_DIR / "experiment_17_final_metrics.csv"
    )

    metrics.to_csv(metrics_path, index=False)

    print("\n" + "=" * 78)
    print("EXPERIMENT 17: FINAL DISTRICT-WISE RESULTS")
    print("=" * 78)

    display_columns = [
        "region",
        "test_year",
        "model",
        "SARIMA_RMSE",
        "Hybrid_RMSE",
        "SARIMA_common_RMSE",
        "Hybrid_common_RMSE",
        "SARIMA_MAE",
        "Hybrid_MAE",
        "status",
    ]

    available_columns = [
        column for column in display_columns
        if column in metrics.columns
    ]

    print(metrics[available_columns].to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
