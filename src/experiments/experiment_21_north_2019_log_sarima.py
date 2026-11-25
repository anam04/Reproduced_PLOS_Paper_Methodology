
from pathlib import Path
from itertools import product
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "processed" / "district_splits"

OUTPUT = ROOT / "results" / "experiment_21_north_2019_log_sarima"

METRICS_DIR = OUTPUT / "metrics"
PREDICTIONS_DIR = OUTPUT / "predictions"
CV_DIR = OUTPUT / "cv_results"

REGION = "north_goa"
YEAR = 2019
TARGET = "Malaria_Cases"

PERIOD = 12
N_SPLITS = 3
VALIDATION_HORIZON = 12

CANDIDATES = [
    (
        (p, 1, q),
        (P, 1, Q, PERIOD),
    )
    for p, q, P, Q in product(
        [0, 1, 2],
        [0, 1, 2],
        [0, 1],
        [0, 1],
    )
]


# ============================================================
# DATA
# ============================================================

def load_data():

    train = pd.read_csv(
        SPLIT_DIR / f"{REGION}_{YEAR}_train.csv"
    )

    test = pd.read_csv(
        SPLIT_DIR / f"{REGION}_{YEAR}_test.csv"
    )

    for frame in (train, test):
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)

        if frame[TARGET].isna().any():
            raise ValueError("Missing malaria observations.")

    y_train = train[TARGET].to_numpy(dtype=float)

    if np.any(y_train <= 0):
        raise ValueError("Log transformation requires positive counts.")

    if len(test) != 12:
        raise ValueError("Expected exactly 12 test months.")

    return train, test


# ============================================================
# LOG-SARIMA FITTING
# ============================================================

def fit_log_sarima(y, order, seasonal_order):

    y = np.asarray(y, dtype=float)

    if np.any(y <= 0):
        raise ValueError("Log-SARIMA requires positive observations.")

    transformed = np.log(y)

    model = SARIMAX(
        transformed,
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

    if not np.isfinite(fitted.aic):
        raise ValueError("Non-finite SARIMA AIC.")

    return fitted


def forecast_log_sarima(train, future, order, seasonal_order):

    y = train[TARGET].to_numpy(dtype=float)

    fitted = fit_log_sarima(
        y,
        order,
        seasonal_order,
    )

    log_prediction = np.asarray(
        fitted.get_forecast(
            steps=len(future)
        ).predicted_mean,
        dtype=float,
    ).reshape(-1)

    if not np.all(np.isfinite(log_prediction)):
        raise ValueError("Non-finite log-scale forecast.")

    with np.errstate(over="ignore", invalid="ignore"):
        prediction = np.exp(log_prediction)

    if (
        not np.all(np.isfinite(prediction))
        or np.any(prediction <= 0)
    ):
        raise ValueError("Invalid exponential inverse forecast.")

    return prediction, fitted


# ============================================================
# ROLLING-ORIGIN VALIDATION
# ============================================================

def build_folds(train):

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_HORIZON,
    )

    folds = []

    for number, (train_idx, valid_idx) in enumerate(
        splitter.split(train),
        start=1,
    ):

        fold_train = train.iloc[train_idx].reset_index(drop=True)
        fold_valid = train.iloc[valid_idx].reset_index(drop=True)

        print(
            f"Fold {number}: "
            f"{len(fold_train)} training / "
            f"{len(fold_valid)} validation observations"
        )

        folds.append((fold_train, fold_valid))

    return folds


def evaluate_candidate(order, seasonal_order, folds):

    actual_all = []
    predicted_all = []

    fold_scores = []

    for fold_train, fold_valid in folds:

        try:
            prediction, _ = forecast_log_sarima(
                fold_train,
                fold_valid,
                order,
                seasonal_order,
            )

            actual = fold_valid[TARGET].to_numpy(dtype=float)

            if (
                len(prediction) != len(actual)
                or not np.all(np.isfinite(prediction))
            ):
                return np.inf, np.inf

            fold_scores.append(
                np.sqrt(mean_squared_error(actual, prediction))
            )

            actual_all.extend(actual.tolist())
            predicted_all.extend(prediction.tolist())

        except Exception:
            return np.inf, np.inf

    pooled_rmse = float(
        np.sqrt(
            mean_squared_error(actual_all, predicted_all)
        )
    )

    mean_fold_rmse = float(np.mean(fold_scores))

    return pooled_rmse, mean_fold_rmse


# ============================================================
# ORDER OPTIMISATION
# ============================================================

def optimise_order(train):

    folds = build_folds(train)

    results = []

    best_order = None
    best_seasonal_order = None
    best_score = np.inf

    print(f"\nEvaluating {len(CANDIDATES)} configurations.")

    for number, (order, seasonal_order) in enumerate(
        CANDIDATES,
        start=1,
    ):

        pooled_rmse, mean_fold_rmse = evaluate_candidate(
            order,
            seasonal_order,
            folds,
        )

        results.append({
            "candidate": number,
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "pooled_validation_RMSE": pooled_rmse,
            "mean_fold_RMSE": mean_fold_rmse,
        })

        print(
            f"{number:02d}/{len(CANDIDATES)} "
            f"ARIMA{order} x {seasonal_order} "
            f"| RMSE={pooled_rmse:.4f}"
        )

        if pooled_rmse < best_score:
            best_score = pooled_rmse
            best_order = order
            best_seasonal_order = seasonal_order

    results_df = pd.DataFrame(results).sort_values(
        "pooled_validation_RMSE"
    )

    results_df.to_csv(
        CV_DIR / "experiment_21_north_2019_log_cv.csv",
        index=False,
    )

    valid = results_df[
        np.isfinite(results_df["pooled_validation_RMSE"])
    ]

    if valid.empty:
        raise RuntimeError("All log-SARIMA configurations failed.")

    print("\nTop five configurations:")
    print(valid.head(5).to_string(index=False))

    best_row = valid.iloc[0]

    best_order = eval(best_row["order"])
    best_seasonal_order = eval(best_row["seasonal_order"])

    best_score = float(best_row["pooled_validation_RMSE"])

    print("\nSelected order:", best_order)
    print("Selected seasonal order:", best_seasonal_order)
    print(f"Best validation RMSE: {best_score:.4f}")

    return best_order, best_seasonal_order, best_score


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def main():

    for directory in [
        METRICS_DIR,
        PREDICTIONS_DIR,
        CV_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    train, test = load_data()

    print("=" * 75)
    print("EXPERIMENT 21: NORTH GOA 2019 LOG-SARIMA")
    print("=" * 75)

    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")

    order, seasonal_order, validation_rmse = optimise_order(train)

    # Refit the selected configuration on all 2010-2018 observations.
    prediction, fitted = forecast_log_sarima(
        train,
        test,
        order,
        seasonal_order,
    )

    actual = test[TARGET].to_numpy(dtype=float)

    rmse = float(
        np.sqrt(mean_squared_error(actual, prediction))
    )

    mae = float(
        mean_absolute_error(actual, prediction)
    )

    prediction_df = pd.DataFrame({
        "Date": test["Date"],
        "Actual": actual,
        "Predicted": prediction,
        "Residual": actual - prediction,
    })

    prediction_df.to_csv(
        PREDICTIONS_DIR / "north_goa_2019_log_predictions.csv",
        index=False,
    )

    result = {
        "region": REGION,
        "test_year": YEAR,
        "transformation": "log",
        "order": str(order),
        "seasonal_order": str(seasonal_order),
        "aic": float(fitted.aic),
        "validation_RMSE": validation_rmse,
        "test_RMSE": rmse,
        "test_MAE": mae,
        "status": "valid",
    }

    pd.DataFrame([result]).to_csv(
        METRICS_DIR / "experiment_21_final_metrics.csv",
        index=False,
    )

    print("\nFINAL RESULT")
    print(f"SARIMA order: {order}")
    print(f"Seasonal order: {seasonal_order}")
    print(f"AIC: {fitted.aic:.4f}")
    print(f"Validation RMSE: {validation_rmse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    print("\nPrevious Box-Cox result:")
    print("Experiment 20 North Goa 2019: invalid inverse Box-Cox forecast.")

    print("\nResults saved:", OUTPUT)


if __name__ == "__main__":
    main()
