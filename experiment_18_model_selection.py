
from pathlib import Path
from itertools import product
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.stats import boxcox, randint, loguniform
from scipy.special import inv_boxcox
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "district_splits"

OUTPUT = ROOT / "results" / "experiment_18_model_selection"

METRICS_DIR = OUTPUT / "metrics"
PREDICTIONS_DIR = OUTPUT / "predictions"
CV_DIR = OUTPUT / "cv_results"

REGIONS = ["north_goa", "south_goa"]
YEARS = [2018, 2019]

TARGET = "Malaria_Cases"

CLIMATE = [
    "max_temperature",
    "min_temperature",
    "total_rainfall",
    "avg_relative_humidity",
    "max_pressure",
    "min_wind_speed",
]

SARIMA_ORDERS = [
    ((1, 1, 1), (0, 1, 1, 12)),
    ((2, 1, 1), (0, 1, 1, 12)),
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 2), (1, 1, 1, 12)),
]

N_FOLDS = 3
HORIZON = 12
RANDOM_STATE = 42

SVR_CONFIGS = [
    {"C": c, "gamma": g, "epsilon": e}
    for c, g, e in product(
        [1, 10, 100],
        ["scale", 0.1],
        [0.1, 1.0],
    )
]

XGB_CONFIGS = list(
    ParameterSampler(
        {
            "n_estimators": randint(100, 501),
            "max_depth": randint(1, 6),
            "learning_rate": loguniform(0.015, 0.2),
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": randint(1, 6),
            "reg_lambda": loguniform(0.1, 20),
        },
        n_iter=20,
        random_state=RANDOM_STATE,
    )
)


# ============================================================
# DATA
# ============================================================

def load_data(region, year):

    train = pd.read_csv(
        SPLIT_DIR / f"{region}_{year}_train.csv"
    )
    test = pd.read_csv(
        SPLIT_DIR / f"{region}_{year}_test.csv"
    )

    for frame in (train, test):
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)

        required = [TARGET] + CLIMATE

        missing = set(required) - set(frame.columns)
        if missing:
            raise KeyError(f"Missing columns: {sorted(missing)}")

        if frame[required].isna().any().any():
            raise ValueError("Missing values in model inputs.")

    if len(test) != 12:
        raise ValueError("Test period must contain exactly 12 months.")

    if np.any(train[TARGET].to_numpy() <= 0):
        raise ValueError("Training malaria counts must be positive.")

    return train, test


# ============================================================
# BOX-COX
# ============================================================

def guerrero_lambda(y):

    y = np.asarray(y, dtype=float)

    means, stds = [], []

    for month in range(12):
        subseries = y[month::12]

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
        raise ValueError("Invalid seasonal subseries.")

    def objective(lmbda):
        adjusted = stds / means ** (1 - lmbda)
        return np.std(adjusted, ddof=1) / np.mean(adjusted)

    result = minimize_scalar(
        objective,
        bounds=(-1, 2),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError("Box-Cox lambda estimation failed.")

    return float(result.x)


# ============================================================
# SARIMA / SARIMAX
# ============================================================

def fit_statistical_model(
    train,
    candidate,
    use_climate,
):

    y = train[TARGET].to_numpy(dtype=float)
    lmbda = guerrero_lambda(y)
    transformed = boxcox(y, lmbda=lmbda)

    order, seasonal_order = candidate

    if use_climate:
        scaler = StandardScaler()

        exog = scaler.fit_transform(
            train[CLIMATE].to_numpy(dtype=float)
        )
    else:
        scaler = None
        exog = None

    model = SARIMAX(
        transformed,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit(disp=False, maxiter=300)

    return fitted, lmbda, scaler


def forecast_statistical(
    train,
    future,
    candidate,
    use_climate,
):

    fitted, lmbda, scaler = fit_statistical_model(
        train,
        candidate,
        use_climate,
    )

    if use_climate:
        future_exog = scaler.transform(
            future[CLIMATE].to_numpy(dtype=float)
        )
    else:
        future_exog = None

    transformed_prediction = np.asarray(
        fitted.get_forecast(
            steps=len(future),
            exog=future_exog,
        ).predicted_mean,
        dtype=float,
    ).reshape(-1)

    if (
        not np.all(np.isfinite(transformed_prediction))
        or np.any(1 + lmbda * transformed_prediction <= 0)
    ):
        raise ValueError("Invalid inverse Box-Cox forecast.")

    prediction = inv_boxcox(
        transformed_prediction,
        lmbda,
    )

    if not np.all(np.isfinite(prediction)):
        raise ValueError("Non-finite statistical forecasts.")

    return prediction


# ============================================================
# DIRECT ML FEATURE ENGINEERING
# ============================================================

ML_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_12",
    "month_sin",
    "month_cos",
] + CLIMATE


def make_ml_row(history, climate_row, date):

    history = np.asarray(history, dtype=float)

    month = pd.Timestamp(date).month

    row = {
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "lag_6": history[-6],
        "lag_12": history[-12],
        "rolling_mean_3": np.mean(history[-3:]),
        "rolling_mean_12": np.mean(history[-12:]),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }

    for feature in CLIMATE:
        row[feature] = float(climate_row[feature])

    return row


def build_ml_training_data(train):

    y = train[TARGET].to_numpy(dtype=float)

    rows, targets = [], []

    for t in range(12, len(train)):

        row = make_ml_row(
            y[:t],
            train.iloc[t],
            train.iloc[t]["Date"],
        )

        rows.append(row)
        targets.append(y[t])

    return (
        pd.DataFrame(rows, columns=ML_FEATURES),
        np.asarray(targets, dtype=float),
    )


def create_ml_model(name, parameters):

    if name == "SVR":

        pipeline = Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                SVR(
                    kernel="rbf",
                    C=parameters["C"],
                    gamma=parameters["gamma"],
                    epsilon=parameters["epsilon"],
                ),
            ),
        ])

        return TransformedTargetRegressor(
            regressor=pipeline,
            transformer=StandardScaler(),
        )

    if name == "XGBoost":

        pipeline = Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
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

    raise ValueError(name)


def forecast_ml(train, future, name, parameters):

    X_train, y_train = build_ml_training_data(train)

    model = create_ml_model(name, parameters)
    model.fit(X_train, y_train)

    history = train[TARGET].to_numpy(dtype=float).tolist()
    predictions = []

    for _, row in future.iterrows():

        features = make_ml_row(
            history,
            row,
            row["Date"],
        )

        X_current = pd.DataFrame(
            [features],
            columns=ML_FEATURES,
        )

        prediction = float(model.predict(X_current)[0])

        # Malaria incidence cannot be negative.
        prediction = max(0.0, prediction)

        predictions.append(prediction)
        history.append(prediction)

    return np.asarray(predictions), model


# ============================================================
# SEASONAL NAIVE
# ============================================================

def forecast_seasonal_naive(train, future):

    history = train[TARGET].to_numpy(dtype=float)
    return np.asarray(history[-12:], dtype=float)


# ============================================================
# GENERIC FORECAST DISPATCH
# ============================================================

def forecast_candidate(
    name,
    parameters,
    train,
    future,
):

    if name == "SARIMA":
        return forecast_statistical(
            train,
            future,
            parameters["order"],
            use_climate=False,
        )

    if name == "SARIMAX":
        return forecast_statistical(
            train,
            future,
            parameters["order"],
            use_climate=True,
        )

    if name in ("SVR", "XGBoost"):
        prediction, _ = forecast_ml(
            train,
            future,
            name,
            parameters,
        )
        return prediction

    if name == "SeasonalNaive":
        return forecast_seasonal_naive(train, future)

    raise ValueError(name)


# ============================================================
# CANDIDATE DEFINITIONS
# ============================================================

def get_candidates():

    candidates = []

    for order in SARIMA_ORDERS:
        candidates.append(("SARIMA", {"order": order}))
        candidates.append(("SARIMAX", {"order": order}))

    for parameters in SVR_CONFIGS:
        candidates.append(("SVR", parameters))

    for parameters in XGB_CONFIGS:
        candidates.append(("XGBoost", parameters))

    candidates.append(("SeasonalNaive", {}))

    return candidates


# ============================================================
# ROLLING-ORIGIN VALIDATION
# ============================================================

def evaluate_candidate(name, parameters, folds):

    actual_all = []
    predicted_all = []

    for train, validation in folds:

        try:
            prediction = forecast_candidate(
                name,
                parameters,
                train,
                validation,
            )

            actual = validation[TARGET].to_numpy(dtype=float)

            if (
                len(prediction) != len(actual)
                or not np.all(np.isfinite(prediction))
            ):
                return np.inf

            actual_all.extend(actual.tolist())
            predicted_all.extend(prediction.tolist())

        except Exception:
            return np.inf

    return float(
        np.sqrt(
            mean_squared_error(actual_all, predicted_all)
        )
    )


def build_folds(train):

    splitter = TimeSeriesSplit(
        n_splits=N_FOLDS,
        test_size=HORIZON,
    )

    folds = []

    for number, (train_idx, validation_idx) in enumerate(
        splitter.split(train),
        start=1,
    ):

        fold_train = train.iloc[train_idx].reset_index(drop=True)
        fold_validation = train.iloc[validation_idx].reset_index(drop=True)

        print(
            f"  Fold {number}: "
            f"{len(fold_train)} train / "
            f"{len(fold_validation)} validation"
        )

        folds.append((fold_train, fold_validation))

    return folds


def select_best_model(train, region, year):

    folds = build_folds(train)
    candidates = get_candidates()

    results = []

    print(f"\nEvaluating {len(candidates)} candidate configurations.")

    for number, (name, parameters) in enumerate(candidates, start=1):

        score = evaluate_candidate(
            name,
            parameters,
            folds,
        )

        results.append({
            "candidate": number,
            "model": name,
            "validation_RMSE": score,
            "parameters": str(parameters),
        })

        print(
            f"  {number:03d}/{len(candidates)} "
            f"{name:12s} RMSE={score:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "validation_RMSE"
    )

    results_df.to_csv(
        CV_DIR / f"experiment_18_{region}_{year}_cv.csv",
        index=False,
    )

    valid_results = results_df[
        np.isfinite(results_df["validation_RMSE"])
    ]

    if valid_results.empty:
        raise RuntimeError("Every candidate failed validation.")

    best_row = valid_results.iloc[0]

    # Recover the selected parameter dictionary from the
    # original candidate list, preserving Python data types.
    selected_number = int(best_row["candidate"])

    best_name, best_parameters = candidates[selected_number - 1]

    print("\nTop five candidates:")
    print(results_df.head(5).to_string(index=False))

    print("\nSelected model:", best_name)
    print("Selected parameters:", best_parameters)
    print("Validation RMSE:", best_row["validation_RMSE"])

    return best_name, best_parameters, float(best_row["validation_RMSE"])


# ============================================================
# FINAL METRICS
# ============================================================

def metrics(actual, prediction):

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(prediction)

    if not valid.any():
        return np.nan, np.nan

    return (
        float(np.sqrt(mean_squared_error(actual[valid], prediction[valid]))),
        float(mean_absolute_error(actual[valid], prediction[valid])),
    )


# ============================================================
# ONE DISTRICT/YEAR EXPERIMENT
# ============================================================

def run_one(region, year):

    print("\n" + "=" * 78)
    print(f"EXPERIMENT 18 | {region.upper()} | {year}")
    print("=" * 78)

    train, test = load_data(region, year)

    print(f"Training: {len(train)} months")
    print(f"Testing: {len(test)} months")

    best_name, best_parameters, cv_rmse = select_best_model(
        train,
        region,
        year,
    )

    prediction = forecast_candidate(
        best_name,
        best_parameters,
        train,
        test,
    )

    actual = test[TARGET].to_numpy(dtype=float)

    rmse, mae = metrics(actual, prediction)

    print("\nFINAL TEST RESULT")
    print(f"Selected model: {best_name}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    prediction_df = pd.DataFrame({
        "Date": test["Date"],
        "Actual": actual,
        "Predicted": prediction,
        "Selected_Model": best_name,
    })

    prediction_df.to_csv(
        PREDICTIONS_DIR
        / f"experiment_18_{region}_{year}_predictions.csv",
        index=False,
    )

    return {
        "region": region,
        "test_year": year,
        "selected_model": best_name,
        "validation_RMSE": cv_rmse,
        "test_RMSE": rmse,
        "test_MAE": mae,
        "training_size": len(train),
        "test_size": len(test),
        "parameters": str(best_parameters),
        "status": "valid",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    for directory in [
        METRICS_DIR,
        PREDICTIONS_DIR,
        CV_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    results = []

    for region in REGIONS:
        for year in YEARS:

            try:
                results.append(
                    run_one(region, year)
                )

            except Exception as error:

                print("\nFAILED:", region, year)
                print(repr(error))

                results.append({
                    "region": region,
                    "test_year": year,
                    "status": "failed",
                    "error": repr(error),
                })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        METRICS_DIR / "experiment_18_final_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 18: FINAL RESULTS")
    print("=" * 78)

    print(
        results_df[
            [
                "region",
                "test_year",
                "selected_model",
                "validation_RMSE",
                "test_RMSE",
                "test_MAE",
                "status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
