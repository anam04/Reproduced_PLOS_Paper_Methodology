
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data/processed/district_splits"
OUTPUT_DIR = ROOT / "results/experiment_23_direct"

TARGET = "Malaria_Cases"
HORIZON = 12
N_SPLITS = 3
SEASON = 12
RANDOM_STATE = 42

CLIMATE = [
    "max_temperature",
    "min_temperature",
    "total_rainfall",
    "avg_relative_humidity",
    "max_pressure",
    "min_wind_speed",
]

FEATURES = [
    "lag_1", "lag_2", "lag_3",
    "lag_6", "lag_12", "rolling_mean_3",
    "month_sin", "month_cos",
] + CLIMATE


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

def load_split(region, year):

    train = pd.read_csv(
        SPLIT_DIR / f"{region}_{year}_train.csv"
    )
    test = pd.read_csv(
        SPLIT_DIR / f"{region}_{year}_test.csv"
    )

    for frame in (train, test):
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.sort_values("Date").reset_index(drop=True)

        # Replace the original frame with its sorted version.
        if frame is train:
            train = frame
        else:
            test = frame

    required = [TARGET] + CLIMATE

    for frame in (train, test):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if frame[required].isna().any().any():
            raise ValueError("Missing values in model inputs.")

    if len(test) != HORIZON:
        raise ValueError("Expected 12 test observations.")

    if len(train) < 36:
        raise ValueError("Insufficient training history.")

    return train, test


# ---------------------------------------------------------
# DIRECT HORIZON FEATURES
# ---------------------------------------------------------

def make_feature_row(history, climate_row, target_date):

    history = np.asarray(history, dtype=float)
    month = pd.Timestamp(target_date).month

    row = {
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "lag_6": history[-6],
        "lag_12": history[-12],
        "rolling_mean_3": np.mean(history[-3:]),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }

    for col in CLIMATE:
        row[col] = float(climate_row[col])

    return pd.DataFrame([row], columns=FEATURES)


def build_horizon_training_data(train, horizon):

    y = train[TARGET].to_numpy(dtype=float)
    rows = []
    targets = []

    for target_idx in range(SEASON + horizon - 1, len(train)):

        origin = target_idx - horizon + 1
        history = y[:origin]

        row = make_feature_row(
            history,
            train.iloc[target_idx],
            train.iloc[target_idx]["Date"],
        )

        rows.append(row.iloc[0].to_dict())
        targets.append(y[target_idx])

    if not rows:
        raise ValueError(f"No training samples for horizon {horizon}")

    return pd.DataFrame(rows, columns=FEATURES), np.asarray(targets)


# ---------------------------------------------------------
# CANDIDATES
# ---------------------------------------------------------

def get_candidates():

    candidates = [
        ("SeasonalNaive", {}),
    ]

    for alpha in [0.1, 1.0, 10.0, 100.0]:
        candidates.append(("Ridge", {"alpha": alpha}))

    for C, gamma in [
        (0.1, "scale"),
        (1.0, "scale"),
        (10.0, "scale"),
        (100.0, "scale"),
        (1.0, 0.01),
        (10.0, 0.01),
    ]:
        candidates.append((
            "SVR",
            {"C": C, "gamma": gamma, "epsilon": 0.1},
        ))

    candidates.extend([
        ("XGB", {
            "max_depth": 2,
            "n_estimators": 200,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_lambda": 10,
        }),
        ("XGB", {
            "max_depth": 2,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 20,
        }),
    ])

    return candidates


def create_model(kind, parameters):

    if kind == "Ridge":
        estimator = Ridge(**parameters)

    elif kind == "SVR":
        estimator = SVR(
            kernel="rbf",
            **parameters,
        )

    elif kind == "XGB":
        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
            **parameters,
        )

    else:
        raise ValueError(kind)

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", estimator),
    ])

    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


# ---------------------------------------------------------
# DIRECT 12-MONTH FORECAST
# ---------------------------------------------------------

def direct_forecast(kind, parameters, train, future):

    history = train[TARGET].to_numpy(dtype=float)
    predictions = []

    if kind == "SeasonalNaive":
        return np.maximum(0.0, history[-12:])

    for horizon in range(1, HORIZON + 1):

        X_train, y_train = build_horizon_training_data(
            train,
            horizon,
        )

        model = create_model(kind, parameters)
        model.fit(X_train, y_train)

        target_row = future.iloc[horizon - 1]

        X_future = make_feature_row(
            history,
            target_row,
            target_row["Date"],
        )

        prediction = float(model.predict(X_future)[0])

        if not np.isfinite(prediction):
            raise ValueError("Non-finite direct forecast.")

        predictions.append(max(0.0, prediction))

    return np.asarray(predictions)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def evaluate_candidate(candidate, train):

    kind, parameters = candidate

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=HORIZON,
    )

    actual_all = []
    predicted_all = []

    for train_idx, valid_idx in splitter.split(train):

        fold_train = train.iloc[train_idx].reset_index(drop=True)
        fold_valid = train.iloc[valid_idx].reset_index(drop=True)

        predictions = direct_forecast(
            kind,
            parameters,
            fold_train,
            fold_valid,
        )

        actual = fold_valid[TARGET].to_numpy(dtype=float)

        if (
            len(predictions) != HORIZON
            or not np.all(np.isfinite(predictions))
        ):
            return np.inf

        actual_all.extend(actual.tolist())
        predicted_all.extend(predictions.tolist())

    return float(
        np.sqrt(
            mean_squared_error(actual_all, predicted_all)
        )
    )


# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------

def run_one(region, year):

    print("\n" + "=" * 70)
    print(f"DIRECT FORECASTING | {region.upper()} | {year}")
    print("=" * 70)

    train, test = load_split(region, year)
    candidates = get_candidates()

    results = []
    best_candidate = None
    best_score = np.inf

    for number, candidate in enumerate(candidates, start=1):

        kind, parameters = candidate

        try:
            score = evaluate_candidate(candidate, train)
        except Exception as error:
            print(f"{kind} failed: {error}")
            score = np.inf

        results.append({
            "model": kind,
            "parameters": str(parameters),
            "validation_RMSE": score,
        })

        print(
            f"{number:02d}/{len(candidates)} "
            f"{kind:13s} RMSE={score:.4f}"
        )

        if score < best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None or not np.isfinite(best_score):
        raise RuntimeError("No valid candidate found.")

    best_kind, best_parameters = best_candidate

    predictions = direct_forecast(
        best_kind,
        best_parameters,
        train,
        test,
    )

    actual = test[TARGET].to_numpy(dtype=float)

    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(mean_absolute_error(actual, predictions))

    seasonal_denominator = np.mean(
        np.abs(
            train[TARGET].to_numpy(dtype=float)[12:]
            - train[TARGET].to_numpy(dtype=float)[:-12]
        )
    )

    mase = (
        mae / seasonal_denominator
        if seasonal_denominator > 0
        else np.nan
    )

    print("\nSELECTED MODEL")
    print(best_kind, best_parameters)
    print(f"Validation RMSE: {best_score:.4f}")

    print("\nTEST RESULTS")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MASE: {mase:.4f}")

    prediction_dir = OUTPUT_DIR / "predictions"
    cv_dir = OUTPUT_DIR / "cv_results"

    prediction_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "Date": test["Date"],
        "Actual": actual,
        "Predicted": predictions,
        "Residual": actual - predictions,
    }).to_csv(
        prediction_dir / f"{region}_{year}_predictions.csv",
        index=False,
    )

    pd.DataFrame(results).sort_values(
        "validation_RMSE"
    ).to_csv(
        cv_dir / f"{region}_{year}_leaderboard.csv",
        index=False,
    )

    return {
        "region": region,
        "year": year,
        "selected_model": best_kind,
        "parameters": str(best_parameters),
        "validation_RMSE": best_score,
        "test_RMSE": rmse,
        "test_MAE": mae,
        "test_MASE": mase,
    }


def main():

    results = []

    for region in ("north_goa", "south_goa"):
        for year in (2018, 2019):
            results.append(run_one(region, year))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(results)
    summary.to_csv(
        OUTPUT_DIR / "experiment_23_summary.csv",
        index=False,
    )

    print("\nFINAL RESULTS")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
