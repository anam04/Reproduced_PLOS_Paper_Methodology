
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "district_splits"
OUTPUT_DIR = ROOT / "results" / "experiment_28"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = (2018, 2019)
HORIZON = 12
SEASONAL_PERIOD = 12
MIN_TRAIN_SIZE = 60
RANDOM_STATE = 42

CLIMATE_ALIASES = {
    "max_temperature": ["max_temperature", "maximum_temperature"],
    "min_temperature": ["min_temperature", "minimum_temperature"],
    "total_rainfall": ["total_rainfall", "rainfall", "precipitation"],
    "avg_relative_humidity": [
        "avg_relative_humidity",
        "average_relative_humidity",
        "relative_humidity",
    ],
    "max_pressure": ["max_pressure", "maximum_pressure"],
    "min_wind_speed": ["min_wind_speed", "minimum_wind_speed"],
}

TARGET_ALIASES = [
    "malaria_cases",
    "indigenous_malaria_cases",
    "indigenous_cases",
    "malaria_incidence",
    "incidence",
    "cases",
    "total_cases",
]

DATE_ALIASES = ["date", "month_date", "datetime", "ds", "time"]

LAGS = (1, 2, 3, 6, 12)


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

def normalise_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_column(frame, aliases):
    lookup = {
        normalise_name(col): col
        for col in frame.columns
    }

    for alias in aliases:
        key = normalise_name(alias)
        if key in lookup:
            return lookup[key]

    return None


def build_dates(frame, fallback_start=None):
    date_col = find_column(frame, DATE_ALIASES)

    if date_col is not None:
        return pd.to_datetime(frame[date_col])

    year_col = find_column(frame, ["year"])
    month_col = find_column(frame, ["month"])

    if year_col is not None and month_col is not None:
        return pd.to_datetime({
            "year": pd.to_numeric(frame[year_col]),
            "month": pd.to_numeric(frame[month_col]),
            "day": 1,
        })

    if fallback_start is None:
        raise ValueError("Could not identify date, year, or month columns.")

    return pd.date_range(
        start=fallback_start,
        periods=len(frame),
        freq="MS",
    )


def load_split(year):
    train_path = SPLIT_DIR / f"north_goa_{year}_train.csv"
    test_path = SPLIT_DIR / f"north_goa_{year}_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing split files for {year}."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target_col = find_column(train, TARGET_ALIASES)

    if target_col is None:
        raise ValueError(
            f"Could not identify target column. "
            f"Available columns: {train.columns.tolist()}"
        )

    test_target_col = find_column(test, TARGET_ALIASES)

    if test_target_col is None:
        if target_col not in test.columns:
            raise ValueError("Target column missing from test data.")
        test_target_col = target_col

    train_y = pd.to_numeric(
        train[target_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    test_y = pd.to_numeric(
        test[test_target_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    if np.isnan(train_y).any() or np.isnan(test_y).any():
        raise ValueError("Missing or non-numeric target observations.")

    if np.any(train_y < 0) or np.any(test_y < 0):
        raise ValueError("Malaria case counts must be non-negative.")

    if len(test_y) != HORIZON:
        raise ValueError(
            f"Expected {HORIZON} test observations, got {len(test_y)}."
        )

    train_dates = build_dates(
        train,
        fallback_start="2010-01-01",
    )

    test_dates = build_dates(
        test,
        fallback_start=(
            pd.Timestamp(train_dates.iloc[-1])
            + pd.offsets.MonthBegin(1)
        ),
    )

    train = train.copy()
    test = test.copy()

    train["__date"] = pd.to_datetime(train_dates).to_numpy()
    test["__date"] = pd.to_datetime(test_dates).to_numpy()

    climate_train = {}
    climate_test = {}

    for canonical_name, aliases in CLIMATE_ALIASES.items():
        train_col = find_column(train, aliases)
        test_col = find_column(test, aliases)

        if train_col is None or test_col is None:
            raise ValueError(
                f"Missing climate feature: {canonical_name}. "
                f"Training columns: {train.columns.tolist()}"
            )

        climate_train[canonical_name] = pd.to_numeric(
            train[train_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        climate_test[canonical_name] = pd.to_numeric(
            test[test_col],
            errors="coerce",
        ).to_numpy(dtype=float)

    climate_frame = pd.DataFrame({
        "__date": np.concatenate([
            train["__date"].to_numpy(),
            test["__date"].to_numpy(),
        ])
    })

    for name in CLIMATE_ALIASES:
        climate_frame[name] = np.concatenate([
            climate_train[name],
            climate_test[name],
        ])

    if climate_frame.isna().any().any():
        raise ValueError("Missing climate or date values detected.")

    climate_frame = climate_frame.sort_values(
        "__date"
    ).reset_index(drop=True)

    train_frame = pd.DataFrame({
        "date": train["__date"],
        "y": train_y,
    }).sort_values("date").reset_index(drop=True)

    test_frame = pd.DataFrame({
        "date": test["__date"],
        "y": test_y,
    }).sort_values("date").reset_index(drop=True)

    if not train_frame["date"].is_monotonic_increasing:
        raise ValueError("Training dates are not chronological.")

    if not test_frame["date"].is_monotonic_increasing:
        raise ValueError("Test dates are not chronological.")

    return train_frame, test_frame, climate_frame, target_col


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

def make_features(y_history, climate_frame, origin, target_idx):
    """
    origin is the first unobserved month at the forecast origin.
    All malaria lags use observations strictly before origin.
    Climate features belong to the target month.
    """
    y_history = np.asarray(y_history, dtype=float)

    if origin < 12:
        raise ValueError("At least 12 historical observations are required.")

    features = {}

    for lag in LAGS:
        features[f"lag_{lag}"] = float(y_history[origin - lag])

    features["rolling_mean_3"] = float(
        np.mean(y_history[origin - 3:origin])
    )

    target_date = pd.Timestamp(
        climate_frame.iloc[target_idx]["__date"]
    )

    month = target_date.month

    features["month_sin"] = float(
        np.sin(2 * np.pi * month / 12)
    )

    features["month_cos"] = float(
        np.cos(2 * np.pi * month / 12)
    )

    for name in CLIMATE_ALIASES:
        features[name] = float(
            climate_frame.iloc[target_idx][name]
        )

    return features


def build_supervised_data(y, climate_frame, cutoff, horizon):
    """
    Construct horizon-specific training examples.

    Each training target occurs strictly before cutoff.
    Every feature row uses only malaria observations available
    at its historical forecast origin.
    """
    rows = []
    targets = []

    first_target = horizon + 12

    for target_idx in range(first_target, cutoff):
        historical_origin = target_idx - horizon

        row = make_features(
            y_history=y[:historical_origin],
            climate_frame=climate_frame,
            origin=historical_origin,
            target_idx=target_idx,
        )

        rows.append(row)
        targets.append(y[target_idx])

    if not rows:
        raise ValueError(
            f"No supervised samples for horizon {horizon}, "
            f"cutoff {cutoff}."
        )

    return (
        pd.DataFrame(rows),
        np.asarray(targets, dtype=float),
    )


# ------------------------------------------------------------
# MODEL CANDIDATES
# ------------------------------------------------------------

def make_model(model_name):
    if model_name == "ridge_10":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        )

    if model_name == "ridge_100":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0),
        )

    if model_name == "svr_01":
        return make_pipeline(
            StandardScaler(),
            SVR(
                C=0.1,
                gamma="scale",
                epsilon=0.1,
            ),
        )

    if model_name == "svr_1":
        return make_pipeline(
            StandardScaler(),
            SVR(
                C=1.0,
                gamma="scale",
                epsilon=0.1,
            ),
        )

    if model_name == "svr_10":
        return make_pipeline(
            StandardScaler(),
            SVR(
                C=10.0,
                gamma="scale",
                epsilon=0.1,
            ),
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "hist_gradient":
        return HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model: {model_name}")


MODEL_NAMES = [
    "ridge_10",
    "ridge_100",
    "svr_01",
    "svr_1",
    "svr_10",
    "random_forest",
    "hist_gradient",
]

CANDIDATES = [
    (model, weighting)
    for model in MODEL_NAMES
    for weighting in ("unweighted", "peak_weighted")
]


def calculate_sample_weights(y):
    """
    Higher incidence observations receive greater weight.
    The 75th percentile is calculated from the current training
    targets only, never from validation or test targets.
    """
    y = np.asarray(y, dtype=float)

    q75 = float(np.quantile(y, 0.75))

    if q75 <= 0:
        return np.ones(len(y), dtype=float)

    relative_incidence = y / q75

    return 1.0 + np.minimum(relative_incidence, 3.0)


def fit_model(model, x, y, weighting):
    if weighting == "unweighted":
        model.fit(x, y)
        return model

    sample_weights = calculate_sample_weights(y)

    if hasattr(model, "steps"):
        final_step_name = model.steps[-1][0]
        model.fit(
            x,
            y,
            **{
                f"{final_step_name}__sample_weight": sample_weights
            },
        )
    else:
        model.fit(
            x,
            y,
            sample_weight=sample_weights,
        )

    return model


# ------------------------------------------------------------
# BASELINES
# ------------------------------------------------------------

def seasonal_naive_forecast(y, horizon):
    y = np.asarray(y, dtype=float)

    return np.array([
        y[-12 + (step % 12)]
        for step in range(horizon)
    ])


def decomposition_forecast(y, dates, future_dates):
    """
    Experiment 27 baseline:
    log1p STL decomposition, constant trend extrapolation,
    and month-specific seasonal averages.

    STL is fitted only on the available historical observations.
    """
    y = np.asarray(y, dtype=float)

    transformed = np.log1p(y)

    decomposition = STL(
        transformed,
        period=12,
        robust=True,
    ).fit()

    trend = np.asarray(decomposition.trend)
    seasonal = np.asarray(decomposition.seasonal)

    seasonal_frame = pd.DataFrame({
        "month": pd.DatetimeIndex(dates).month,
        "seasonal": seasonal,
    })

    monthly_seasonality = (
        seasonal_frame.groupby("month")["seasonal"].mean()
    )

    overall_seasonality = float(seasonal.mean())

    future_seasonality = np.array([
        monthly_seasonality.get(
            pd.Timestamp(date).month,
            overall_seasonality,
        )
        for date in future_dates
    ])

    forecast_log = trend[-1] + future_seasonality

    return np.maximum(np.expm1(forecast_log), 0.0)


def predict_candidate(
    y,
    climate_frame,
    cutoff,
    horizon,
    model_name,
    weighting,
):
    x_train, y_train = build_supervised_data(
        y=y,
        climate_frame=climate_frame,
        cutoff=cutoff,
        horizon=horizon,
    )

    model = make_model(model_name)
    model = fit_model(
        model,
        x_train,
        y_train,
        weighting,
    )

    target_idx = cutoff + horizon - 1

    x_future = pd.DataFrame([
        make_features(
            y_history=y[:cutoff],
            climate_frame=climate_frame,
            origin=cutoff,
            target_idx=target_idx,
        )
    ])

    prediction = float(model.predict(x_future)[0])

    return max(prediction, 0.0)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

def validation_origins(n):
    origins = [
        n - 3 * HORIZON,
        n - 2 * HORIZON,
        n - HORIZON,
    ]

    if min(origins) < MIN_TRAIN_SIZE:
        raise ValueError(
            f"Insufficient observations for validation: n={n}."
        )

    return origins


def calculate_metrics(actual, predicted, training):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    training = np.asarray(training, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))

    seasonal_errors = np.abs(
        training[12:] - training[:-12]
    )

    scale = float(np.mean(seasonal_errors))

    mase = mae / scale if scale > 0 else np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MASE": mase,
    }


def evaluate_candidate(
    y,
    dates,
    climate_frame,
    model_name,
    weighting,
):
    all_actual = []
    all_predictions = []
    fold_records = []

    for fold, cutoff in enumerate(
        validation_origins(len(y)),
        start=1,
    ):
        fold_actual = y[cutoff:cutoff + HORIZON]
        fold_dates = dates[cutoff:cutoff + HORIZON]

        predictions = []

        for horizon in range(1, HORIZON + 1):
            prediction = predict_candidate(
                y=y,
                climate_frame=climate_frame,
                cutoff=cutoff,
                horizon=horizon,
                model_name=model_name,
                weighting=weighting,
            )

            predictions.append(prediction)

        predictions = np.asarray(predictions)

        metrics = calculate_metrics(
            fold_actual,
            predictions,
            y[:cutoff],
        )

        all_actual.extend(fold_actual)
        all_predictions.extend(predictions)

        fold_records.append({
            "model": model_name,
            "weighting": weighting,
            "fold": fold,
            "train_size": cutoff,
            **metrics,
        })

    pooled_metrics = calculate_metrics(
        all_actual,
        all_predictions,
        y,
    )

    return pooled_metrics, fold_records


def evaluate_baseline(
    y,
    dates,
    climate_frame,
    baseline_name,
):
    all_actual = []
    all_predictions = []
    fold_records = []

    for fold, cutoff in enumerate(
        validation_origins(len(y)),
        start=1,
    ):
        history = y[:cutoff]
        future_dates = dates[cutoff:cutoff + HORIZON]

        if baseline_name == "seasonal_naive":
            prediction = seasonal_naive_forecast(
                history,
                HORIZON,
            )

        elif baseline_name == "stl_log_constant":
            prediction = decomposition_forecast(
                history,
                dates[:cutoff],
                future_dates,
            )

        else:
            raise ValueError(baseline_name)

        actual = y[cutoff:cutoff + HORIZON]

        metrics = calculate_metrics(
            actual,
            prediction,
            history,
        )

        all_actual.extend(actual)
        all_predictions.extend(prediction)

        fold_records.append({
            "model": baseline_name,
            "weighting": "not_applicable",
            "fold": fold,
            "train_size": cutoff,
            **metrics,
        })

    pooled_metrics = calculate_metrics(
        all_actual,
        all_predictions,
        y,
    )

    return pooled_metrics, fold_records


# ------------------------------------------------------------
# EXPERIMENT EXECUTION
# ------------------------------------------------------------

def run_experiment(year):
    train_frame, test_frame, climate_frame, target_col = load_split(year)

    y_train = train_frame["y"].to_numpy(dtype=float)
    y_test = test_frame["y"].to_numpy(dtype=float)

    dates_train = train_frame["date"].to_numpy()
    dates_test = test_frame["date"].to_numpy()

    print("\n" + "=" * 68)
    print(f"NORTH GOA | EXPERIMENT 28 | {year}")
    print("=" * 68)

    print(f"Target: {target_col}")
    print(f"Training observations: {len(y_train)}")
    print(f"Test observations: {len(y_test)}")

    ranking_records = []
    fold_records = []

    # Direct forecasting candidates
    for model_name, weighting in CANDIDATES:
        print(
            f"Validating {model_name} | {weighting}",
            flush=True,
        )

        metrics, folds = evaluate_candidate(
            y=y_train,
            dates=dates_train,
            climate_frame=climate_frame,
            model_name=model_name,
            weighting=weighting,
        )

        ranking_records.append({
            "model": model_name,
            "weighting": weighting,
            **metrics,
        })

        fold_records.extend(folds)

    # Baselines
    for baseline_name in (
        "seasonal_naive",
        "stl_log_constant",
    ):
        metrics, folds = evaluate_baseline(
            y=y_train,
            dates=dates_train,
            climate_frame=climate_frame,
            baseline_name=baseline_name,
        )

        ranking_records.append({
            "model": baseline_name,
            "weighting": "not_applicable",
            **metrics,
        })

        fold_records.extend(folds)

    ranking = (
        pd.DataFrame(ranking_records)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    fold_results = pd.DataFrame(fold_records)

    ranking.to_csv(
        OUTPUT_DIR / f"north_goa_{year}_validation_ranking.csv",
        index=False,
    )

    fold_results.to_csv(
        OUTPUT_DIR / f"north_goa_{year}_validation_folds.csv",
        index=False,
    )

    print("\nValidation ranking:")
    print(ranking.to_string(index=False))

    selected = ranking.iloc[0]
    selected_model = str(selected["model"])
    selected_weighting = str(selected["weighting"])

    print(f"\nSelected model: {selected_model}")
    print(f"Weighting: {selected_weighting}")
    print(f"Validation RMSE: {selected['RMSE']:.4f}")

    # Final test forecast
    if selected_model == "seasonal_naive":
        test_prediction = seasonal_naive_forecast(
            y_train,
            HORIZON,
        )

    elif selected_model == "stl_log_constant":
        test_prediction = decomposition_forecast(
            y_train,
            dates_train,
            dates_test,
        )

    else:
        test_prediction = []

        for horizon in range(1, HORIZON + 1):
            prediction = predict_candidate(
                y=y_train,
                climate_frame=climate_frame,
                cutoff=len(y_train),
                horizon=horizon,
                model_name=selected_model,
                weighting=selected_weighting,
            )

            test_prediction.append(prediction)

        test_prediction = np.asarray(test_prediction)

    test_metrics = calculate_metrics(
        y_test,
        test_prediction,
        y_train,
    )

    forecast_frame = pd.DataFrame({
        "date": dates_test,
        "actual": y_test,
        "predicted": test_prediction,
        "error": y_test - test_prediction,
        "absolute_error": np.abs(y_test - test_prediction),
    })

    forecast_frame.to_csv(
        OUTPUT_DIR / f"north_goa_{year}_forecasts.csv",
        index=False,
    )

    print("\nTest performance:")
    print(f"RMSE: {test_metrics['RMSE']:.4f}")
    print(f"MAE:  {test_metrics['MAE']:.4f}")
    print(f"MASE: {test_metrics['MASE']:.4f}")

    print("\nMonthly forecasts:")
    print(forecast_frame.to_string(index=False))

    return {
        "district": "North Goa",
        "year": year,
        "selected_model": selected_model,
        "weighting": selected_weighting,
        "validation_RMSE": float(selected["RMSE"]),
        **test_metrics,
    }


def main():
    results = []

    for year in YEARS:
        results.append(run_experiment(year))

    summary = pd.DataFrame(results)

    summary.to_csv(
        OUTPUT_DIR / "experiment_28_summary.csv",
        index=False,
    )

    print("\n" + "=" * 68)
    print("EXPERIMENT 28 SUMMARY")
    print("=" * 68)
    print(summary.to_string(index=False))

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
