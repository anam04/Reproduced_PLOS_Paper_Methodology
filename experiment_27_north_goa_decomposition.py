from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "district_splits"
OUTPUT_DIR = ROOT / "results" / "experiment_27"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = (2018, 2019)
HORIZON = 12
SEASONAL_PERIOD = 12
MIN_TRAIN_SIZE = 60

TARGET_ALIASES = (
    "malaria_cases",
    "indigenous_malaria_cases",
    "indigenous_cases",
    "malaria_incidence",
    "incidence",
    "cases",
    "total_cases",
    "indigenous malaria cases",
)

DATE_ALIASES = ("date", "month_date", "datetime", "ds", "time")


def normalise_name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def find_column(frame, aliases):
    lookup = {normalise_name(col): col for col in frame.columns}

    for alias in aliases:
        key = normalise_name(alias)
        if key in lookup:
            return lookup[key]

    return None


def load_split(year):
    train_path = SPLIT_DIR / f"north_goa_{year}_train.csv"
    test_path = SPLIT_DIR / f"north_goa_{year}_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing split files for {year}:\n"
            f"{train_path}\n{test_path}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target_col = find_column(train, TARGET_ALIASES)

    if target_col is None:
        numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
        excluded = {
            "year", "month", "month_sin", "month_cos",
            "max_temperature", "min_temperature",
            "total_rainfall", "avg_relative_humidity",
            "max_pressure", "min_wind_speed",
        }
        candidates = [
            col for col in numeric_cols
            if normalise_name(col) not in excluded
        ]

        if len(candidates) != 1:
            raise ValueError(
                "Could not identify the malaria target column. "
                f"Numeric columns found: {numeric_cols}. "
                "Set the correct target name in TARGET_ALIASES."
            )

        target_col = candidates[0]

    test_target_col = find_column(test, TARGET_ALIASES)

    if test_target_col is None:
        if target_col not in test.columns:
            raise ValueError(
                f"Target column {target_col!r} is missing from test data."
            )
        test_target_col = target_col

    date_col = find_column(train, DATE_ALIASES)
    test_date_col = find_column(test, DATE_ALIASES)

    y_train = pd.to_numeric(train[target_col], errors="coerce").to_numpy()
    y_test = pd.to_numeric(test[test_target_col], errors="coerce").to_numpy()

    if np.isnan(y_train).any() or np.isnan(y_test).any():
        raise ValueError("Missing or non-numeric values found in target.")

    if len(y_test) != HORIZON:
        raise ValueError(
            f"Expected {HORIZON} test observations for {year}, "
            f"found {len(y_test)}."
        )

    if date_col is not None and test_date_col is not None:
        dates_train = pd.to_datetime(train[date_col])
        dates_test = pd.to_datetime(test[test_date_col])
    else:
        # The existing splits contain consecutive monthly observations
        # beginning in January 2010.
        dates = pd.date_range(
            start="2010-01-01",
            periods=len(train) + len(test),
            freq="MS",
        )
        dates_train = dates[:len(train)]
        dates_test = dates[len(train):]

    train_frame = pd.DataFrame({
        "date": dates_train,
        "y": y_train,
    }).sort_values("date").reset_index(drop=True)

    test_frame = pd.DataFrame({
        "date": dates_test,
        "y": y_test,
    }).sort_values("date").reset_index(drop=True)

    if train_frame["date"].duplicated().any():
        raise ValueError("Duplicate dates found in training data.")

    if test_frame["date"].duplicated().any():
        raise ValueError("Duplicate dates found in test data.")

    return train_frame, test_frame, target_col


def forecast_trend(trend, horizon, method):
    trend = np.asarray(trend, dtype=float)
    steps = np.arange(len(trend))
    future_steps = np.arange(len(trend), len(trend) + horizon)

    if method == "linear":
        slope, intercept = np.polyfit(steps, trend, deg=1)
        return intercept + slope * future_steps

    if method == "holt":
        model = ExponentialSmoothing(
            trend,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.asarray(model.forecast(horizon), dtype=float)

    if method == "constant":
        return np.repeat(trend[-1], horizon)

    raise ValueError(f"Unknown trend method: {method}")


def decomposition_forecast(y, dates, future_dates, transform, trend_method):
    """
    STL is fitted only on the supplied historical observations.
    No validation or test targets enter the decomposition.
    """
    y = np.asarray(y, dtype=float)

    if transform == "log1p":
        if np.any(y < 0):
            raise ValueError("log1p requires non-negative observations.")
        transformed = np.log1p(y)
    elif transform == "identity":
        transformed = y.copy()
    else:
        raise ValueError(f"Unknown transformation: {transform}")

    decomposition = STL(
        transformed,
        period=SEASONAL_PERIOD,
        robust=True,
    ).fit()

    trend = np.asarray(decomposition.trend)
    seasonal = np.asarray(decomposition.seasonal)

    predicted_trend = forecast_trend(
        trend,
        len(future_dates),
        trend_method,
    )

    # Estimate each calendar month's seasonal effect using training data.
    seasonal_frame = pd.DataFrame({
        "month": pd.DatetimeIndex(dates).month,
        "seasonal": seasonal,
    })

    monthly_seasonality = seasonal_frame.groupby("month")["seasonal"].mean()

    overall_seasonality = float(seasonal.mean())
    predicted_seasonality = np.array([
        monthly_seasonality.get(
            month,
            overall_seasonality,
        )
        for month in pd.DatetimeIndex(future_dates).month
    ])

    prediction = predicted_trend + predicted_seasonality

    if transform == "log1p":
        prediction = np.expm1(prediction)

    return np.maximum(prediction, 0.0)


def seasonal_naive_forecast(y, horizon):
    y = np.asarray(y, dtype=float)

    if len(y) < SEASONAL_PERIOD:
        raise ValueError("Insufficient observations for seasonal naive.")

    return np.array([
        y[-SEASONAL_PERIOD + (i % SEASONAL_PERIOD)]
        for i in range(horizon)
    ])


def predict(y, dates, future_dates, model_name):
    if model_name == "seasonal_naive":
        return seasonal_naive_forecast(y, len(future_dates))

    transform, trend_method = model_name.split("__")

    return decomposition_forecast(
        y=y,
        dates=dates,
        future_dates=future_dates,
        transform=transform,
        trend_method=trend_method,
    )


def calculate_metrics(actual, predicted, training):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    training = np.asarray(training, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))

    seasonal_errors = np.abs(
        training[SEASONAL_PERIOD:] - training[:-SEASONAL_PERIOD]
    )
    scale = float(np.mean(seasonal_errors))

    mase = mae / scale if scale > 0 else np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MASE": mase,
    }


def validation_origins(n):
    """
    Three expanding-window folds, each forecasting 12 months.
    The final fold ends at the end of the supplied training series.
    """
    first_origin = n - 3 * HORIZON

    origins = [
        first_origin,
        first_origin + HORIZON,
        first_origin + 2 * HORIZON,
    ]

    if min(origins) < MIN_TRAIN_SIZE:
        raise ValueError(
            f"Not enough training observations for three folds: n={n}."
        )

    return origins


def evaluate_validation(train_frame):
    y = train_frame["y"].to_numpy()
    dates = train_frame["date"].to_numpy()

    models = [
        "seasonal_naive",
        "identity__linear",
        "identity__holt",
        "identity__constant",
        "log1p__linear",
        "log1p__holt",
        "log1p__constant",
    ]

    origins = validation_origins(len(y))
    records = []

    for model_name in models:
        fold_actual = []
        fold_predicted = []
        fold_metrics = []

        for fold_number, origin in enumerate(origins, start=1):
            historical_y = y[:origin]
            historical_dates = dates[:origin]

            validation_y = y[origin:origin + HORIZON]
            validation_dates = dates[origin:origin + HORIZON]

            prediction = predict(
                historical_y,
                historical_dates,
                validation_dates,
                model_name,
            )

            metrics = calculate_metrics(
                validation_y,
                prediction,
                historical_y,
            )

            fold_actual.extend(validation_y)
            fold_predicted.extend(prediction)

            fold_metrics.append(metrics["RMSE"])

            records.append({
                "model": model_name,
                "fold": fold_number,
                "train_size": origin,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MASE": metrics["MASE"],
            })

        pooled_rmse = float(np.sqrt(mean_squared_error(
            fold_actual,
            fold_predicted,
        )))

        records.append({
            "model": model_name,
            "fold": "pooled",
            "train_size": len(y),
            "RMSE": pooled_rmse,
            "MAE": float(mean_absolute_error(
                fold_actual,
                fold_predicted,
            )),
            "MASE": np.nan,
        })

    validation_results = pd.DataFrame(records)

    pooled_results = (
        validation_results[
            validation_results["fold"] == "pooled"
        ]
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    return validation_results, pooled_results


def run_experiment(year):
    train_frame, test_frame, target_col = load_split(year)

    print(f"\n{'=' * 65}")
    print(f"NORTH GOA | EXPERIMENT 27 | {year}")
    print(f"{'=' * 65}")
    print(f"Target column: {target_col}")
    print(f"Training observations: {len(train_frame)}")
    print(f"Test observations: {len(test_frame)}")

    validation_results, pooled_results = evaluate_validation(train_frame)

    selected_model = str(pooled_results.iloc[0]["model"])
    validation_rmse = float(pooled_results.iloc[0]["RMSE"])

    print("\nValidation ranking:")
    print(
        pooled_results[
            ["model", "RMSE", "MAE"]
        ].to_string(index=False)
    )

    print(f"\nSelected model: {selected_model}")
    print(f"Pooled validation RMSE: {validation_rmse:.4f}")

    prediction = predict(
        train_frame["y"].to_numpy(),
        train_frame["date"].to_numpy(),
        test_frame["date"].to_numpy(),
        selected_model,
    )

    metrics = calculate_metrics(
        test_frame["y"].to_numpy(),
        prediction,
        train_frame["y"].to_numpy(),
    )

    forecast_frame = pd.DataFrame({
        "date": test_frame["date"],
        "actual": test_frame["y"],
        "predicted": prediction,
        "error": test_frame["y"] - prediction,
        "absolute_error": np.abs(test_frame["y"] - prediction),
    })

    forecast_frame.to_csv(
        OUTPUT_DIR / f"north_goa_{year}_forecasts.csv",
        index=False,
    )

    validation_results.to_csv(
        OUTPUT_DIR / f"north_goa_{year}_validation_folds.csv",
        index=False,
    )

    print("\nTest performance:")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"MASE: {metrics['MASE']:.4f}")

    print("\nMonthly forecasts:")
    print(forecast_frame.to_string(index=False))

    return {
        "district": "North Goa",
        "year": year,
        "selected_model": selected_model,
        "validation_RMSE": validation_rmse,
        **metrics,
    }


def main():
    results = []

    for year in YEARS:
        results.append(run_experiment(year))

    summary = pd.DataFrame(results)
    summary.to_csv(
        OUTPUT_DIR / "experiment_27_summary.csv",
        index=False,
    )

    print(f"\n{'=' * 65}")
    print("EXPERIMENT 27 SUMMARY")
    print(f"{'=' * 65}")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
