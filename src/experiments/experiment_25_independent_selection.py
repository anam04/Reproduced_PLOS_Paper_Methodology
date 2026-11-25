
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =========================================================
# CONFIGURATION
# =========================================================

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data/processed/district_splits"
OUTPUT_DIR = ROOT / "results/experiment_25_independent_selection"

TARGET = "Malaria_Cases"
HORIZON = 12
N_SPLITS = 3
MIN_TRAIN_SIZE = 48
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
    "lag_6", "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "month_sin", "month_cos",
] + CLIMATE


# =========================================================
# DATA
# =========================================================

def load_split(region, year):

    train = pd.read_csv(
        SPLIT_DIR / f"{region}_{year}_train.csv"
    )

    test = pd.read_csv(
        SPLIT_DIR / f"{region}_{year}_test.csv"
    )

    for frame in (train, test):
        frame["Date"] = pd.to_datetime(frame["Date"])

    train = train.sort_values("Date").reset_index(drop=True)
    test = test.sort_values("Date").reset_index(drop=True)

    required = [TARGET] + CLIMATE

    for frame in (train, test):

        missing = set(required) - set(frame.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if frame[required].isna().any().any():
            raise ValueError("Missing values in model inputs.")

        if (frame[TARGET] < 0).any():
            raise ValueError("Negative malaria counts detected.")

    if len(test) != HORIZON:
        raise ValueError("Expected 12 test observations.")

    if len(train) < MIN_TRAIN_SIZE + N_SPLITS * HORIZON:
        raise ValueError("Insufficient training history.")

    return train, test


# =========================================================
# DIRECT FORECAST FEATURES
# =========================================================

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
        "rolling_mean_6": np.mean(history[-6:]),
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
        raise ValueError("No training samples available.")

    return (
        pd.DataFrame(rows, columns=FEATURES),
        np.asarray(targets, dtype=float),
    )


# =========================================================
# CANDIDATES
# =========================================================

def get_candidates(region):

    candidates = [
        ("SeasonalNaive", {}),
    ]

    # Statistical models: ordinary and seasonal specifications.
    sarima_specs = [
        ((1, 1, 1), (0, 0, 0, 12)),
        ((2, 1, 1), (0, 0, 0, 12)),
        ((1, 1, 1), (1, 0, 0, 12)),
        ((2, 1, 1), (1, 0, 0, 12)),
        ((1, 1, 1), (0, 1, 1, 12)),
    ]

    for order, seasonal_order in sarima_specs:
        for transform in ("identity", "log1p"):
            candidates.append((
                "SARIMA",
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "transform": transform,
                },
            ))

    # Climate-informed statistical models.
    for order in [(1, 1, 1), (2, 1, 1)]:
        candidates.append((
            "SARIMAX",
            {
                "order": order,
                "seasonal_order": (0, 0, 0, 12),
                "transform": "identity",
            },
        ))

    if region == "north_goa":

        for C, gamma in [
            (0.1, "scale"),
            (1.0, "scale"),
            (1.0, 0.01),
            (10.0, 0.01),
        ]:
            candidates.append((
                "LogSVR",
                {"C": C, "gamma": gamma, "epsilon": 0.1},
            ))

        for alpha in [1.0, 10.0, 100.0]:
            candidates.append(("LogRidge", {"alpha": alpha}))

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

        for alpha in [1.0, 10.0]:
            candidates.append(("Poisson", {"alpha": alpha}))

    elif region == "south_goa":

        for alpha in [0.01, 0.1, 1.0, 10.0]:
            candidates.append(("Poisson", {"alpha": alpha}))

        for leaves in [2, 3, 5]:
            candidates.append((
                "HistPoisson",
                {
                    "max_iter": 100,
                    "learning_rate": 0.05,
                    "max_leaf_nodes": leaves,
                    "l2_regularization": 10.0,
                },
            ))

        for C, gamma in [
            (0.1, "scale"),
            (1.0, "scale"),
            (1.0, 0.01),
        ]:
            candidates.append((
                "LogSVR",
                {"C": C, "gamma": gamma, "epsilon": 0.1},
            ))

        candidates.append(("LogRidge", {"alpha": 10.0}))

        candidates.append((
            "XGB",
            {
                "max_depth": 2,
                "n_estimators": 200,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_lambda": 20,
            },
        ))

    return candidates


# =========================================================
# MODEL FACTORY
# =========================================================

def create_ml_model(kind, parameters):

    if kind == "LogSVR":

        base = Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", **parameters)),
        ])

        return TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    if kind == "LogRidge":

        base = Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(**parameters)),
        ])

        return TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    if kind == "Poisson":

        return Pipeline([
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(
                max_iter=1000,
                **parameters,
            )),
        ])

    if kind == "HistPoisson":

        return HistGradientBoostingRegressor(
            loss="poisson",
            random_state=RANDOM_STATE,
            **parameters,
        )

    if kind == "XGB":

        base = Pipeline([
            ("scale", StandardScaler()),
            ("model", XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
                **parameters,
            )),
        ])

        return TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    raise ValueError(kind)


# =========================================================
# STATISTICAL FORECASTING
# =========================================================

def statistical_forecast(kind, parameters, train, future):

    y = train[TARGET].to_numpy(dtype=float)

    transform = parameters["transform"]

    if transform == "log1p":
        y_model = np.log1p(y)
    else:
        y_model = y

    order = parameters["order"]
    seasonal_order = parameters["seasonal_order"]

    exog_train = None
    exog_future = None

    if kind == "SARIMAX":

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        exog_train = scaler.fit_transform(train[CLIMATE])
        exog_future = scaler.transform(future[CLIMATE])

    model = SARIMAX(
        y_model,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(
        disp=False,
        maxiter=200,
    )

    forecast = fitted.get_forecast(
        steps=HORIZON,
        exog=exog_future,
    ).predicted_mean

    forecast = np.asarray(forecast, dtype=float)

    if transform == "log1p":
        forecast = np.expm1(np.clip(forecast, -20, 20))

    return forecast


# =========================================================
# DIRECT MACHINE-LEARNING FORECAST
# =========================================================

def direct_forecast(kind, parameters, train, future):

    history = train[TARGET].to_numpy(dtype=float)

    if kind == "SeasonalNaive":
        return np.maximum(0.0, history[-HORIZON:])

    if kind in ("SARIMA", "SARIMAX"):
        return statistical_forecast(
            kind,
            parameters,
            train,
            future,
        )

    predictions = []

    for horizon in range(1, HORIZON + 1):

        X_train, y_train = build_horizon_training_data(
            train,
            horizon,
        )

        model = create_ml_model(kind, parameters)
        model.fit(X_train, y_train)

        target_row = future.iloc[horizon - 1]

        X_future = make_feature_row(
            history,
            target_row,
            target_row["Date"],
        )

        prediction = float(model.predict(X_future)[0])

        predictions.append(prediction)

    return np.asarray(predictions, dtype=float)


# =========================================================
# VALIDATION
# =========================================================

def get_validation_folds(train):

    initial_size = len(train) - N_SPLITS * HORIZON

    if initial_size < MIN_TRAIN_SIZE:
        raise ValueError("Insufficient initial training window.")

    folds = []

    for fold in range(N_SPLITS):

        train_end = initial_size + fold * HORIZON
        valid_end = train_end + HORIZON

        fold_train = train.iloc[:train_end].reset_index(drop=True)
        fold_valid = train.iloc[train_end:valid_end].reset_index(drop=True)

        folds.append((fold_train, fold_valid))

    return folds


def evaluate_candidate(candidate, train):

    kind, parameters = candidate

    actual_all = []
    predicted_all = []

    for fold_train, fold_valid in get_validation_folds(train):

        predictions = direct_forecast(
            kind,
            parameters,
            fold_train,
            fold_valid,
        )

        actual = fold_valid[TARGET].to_numpy(dtype=float)

        if len(predictions) != HORIZON:
            return np.inf

        if not np.all(np.isfinite(predictions)):
            return np.inf

        # Reject unstable or implausibly explosive numerical forecasts.
        upper_limit = max(
            500.0,
            10.0 * float(fold_train[TARGET].max()),
        )

        if np.max(np.abs(predictions)) > upper_limit:
            return np.inf

        predictions = np.maximum(0.0, predictions)

        actual_all.extend(actual.tolist())
        predicted_all.extend(predictions.tolist())

    return float(np.sqrt(mean_squared_error(
        actual_all,
        predicted_all,
    )))


def select_model(region, year, train):

    candidates = get_candidates(region)
    results = []

    print("\n" + "=" * 75)
    print(f"MODEL SELECTION: {region.upper()} | {year}")
    print("=" * 75)

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

    leaderboard = pd.DataFrame(results).sort_values(
        "validation_RMSE"
    )

    valid = leaderboard[
        np.isfinite(leaderboard["validation_RMSE"])
    ]

    if valid.empty:
        raise RuntimeError("No valid candidate model.")

    best_row = valid.iloc[0]

    # Recover the exact candidate object without parsing its string.
    best_candidate = None

    for candidate in candidates:
        if (
            candidate[0] == best_row["model"]
            and str(candidate[1]) == best_row["parameters"]
        ):
            best_candidate = candidate
            break

    if best_candidate is None:
        raise RuntimeError("Could not recover selected candidate.")

    print("\nSELECTED MODEL")
    print(best_candidate)
    print(f"Validation RMSE: {best_row['validation_RMSE']:.4f}")

    return best_candidate, float(best_row["validation_RMSE"]), leaderboard


# =========================================================
# FINAL TEST EVALUATION
# =========================================================

def evaluate_test(region, year, train, test, candidate, validation_score):

    kind, parameters = candidate

    predictions = direct_forecast(
        kind,
        parameters,
        train,
        test,
    )

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Non-finite test predictions.")

    predictions = np.maximum(0.0, predictions)
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

    prediction_dir = OUTPUT_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "Date": test["Date"],
        "Actual": actual,
        "Predicted": predictions,
        "Residual": actual - predictions,
    }).to_csv(
        prediction_dir / f"{region}_{year}_predictions.csv",
        index=False,
    )

    print("\nTEST RESULTS")
    print(f"Region:          {region}")
    print(f"Year:            {year}")
    print(f"Model:           {kind}")
    print(f"Parameters:      {parameters}")
    print(f"Validation RMSE: {validation_score:.4f}")
    print(f"Test RMSE:       {rmse:.4f}")
    print(f"Test MAE:        {mae:.4f}")
    print(f"Test MASE:       {mase:.4f}")

    return {
        "region": region,
        "year": year,
        "selected_model": kind,
        "parameters": str(parameters),
        "validation_RMSE": validation_score,
        "test_RMSE": rmse,
        "test_MAE": mae,
        "test_MASE": mase,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for region in ("north_goa", "south_goa"):

        for year in (2018, 2019):

            train, test = load_split(region, year)

            candidate, validation_score, leaderboard = select_model(
                region,
                year,
                train,
            )

            cv_dir = OUTPUT_DIR / "cv_results"
            cv_dir.mkdir(parents=True, exist_ok=True)

            leaderboard.to_csv(
                cv_dir / f"{region}_{year}_leaderboard.csv",
                index=False,
            )

            result = evaluate_test(
                region,
                year,
                train,
                test,
                candidate,
                validation_score,
            )

            all_results.append(result)

    summary = pd.DataFrame(all_results)

    summary.to_csv(
        OUTPUT_DIR / "experiment_25_summary.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 25 FINAL SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
