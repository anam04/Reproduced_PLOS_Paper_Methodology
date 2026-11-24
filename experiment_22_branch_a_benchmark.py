

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data/processed/district_splits"
OUTPUT_DIR = ROOT / "results/experiment_22_branch_a"

TARGET = "Malaria_Cases"

CLIMATE = [
    "max_temperature",
    "min_temperature",
    "total_rainfall",
    "avg_relative_humidity",
    "max_pressure",
    "min_wind_speed",
]

SEASONAL_ORDER = (0, 1, 1, 12)
SARIMA_ORDERS = [
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 1),
]

HORIZON = 12
N_SPLITS = 3
SEASON = 12
RANDOM_STATE = 42


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
        frame.sort_values("Date", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    required = [TARGET] + CLIMATE

    for frame in (train, test):
        missing = set(required) - set(frame.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if frame[required].isna().any().any():
            raise ValueError("Missing model inputs.")

    if len(test) != HORIZON:
        raise ValueError("Expected 12 test observations.")

    return train, test


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "rolling_mean_3",
] + CLIMATE


def make_features(history, climate_row):

    history = np.asarray(history, dtype=float)

    features = {
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "lag_12": history[-12],
        "rolling_mean_3": np.mean(history[-3:]),
    }

    for col in CLIMATE:
        features[col] = float(climate_row[col])

    return pd.DataFrame([features], columns=FEATURES)


def build_ml_training_data(train):

    y = train[TARGET].to_numpy(dtype=float)
    rows = []

    for t in range(SEASON, len(train)):

        row = {
            "lag_1": y[t - 1],
            "lag_2": y[t - 2],
            "lag_3": y[t - 3],
            "lag_12": y[t - 12],
            "rolling_mean_3": np.mean(y[t - 3:t]),
        }

        for col in CLIMATE:
            row[col] = float(train.iloc[t][col])

        rows.append(row)

    X = pd.DataFrame(rows, columns=FEATURES)
    target = y[SEASON:]

    return X, target


# ---------------------------------------------------------
# ML MODELS
# ---------------------------------------------------------

def create_ml_model(kind, parameters):

    if kind == "SVR":
        estimator = SVR(
            kernel="rbf",
            C=parameters["C"],
            gamma=parameters["gamma"],
            epsilon=parameters["epsilon"],
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
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


SVR_CONFIGS = [
    {"C": 1, "gamma": "scale", "epsilon": 0.1},
    {"C": 10, "gamma": "scale", "epsilon": 0.1},
    {"C": 100, "gamma": 0.1, "epsilon": 0.1},
    {"C": 10, "gamma": 0.01, "epsilon": 0.2},
]

XGB_CONFIGS = [
    {
        "max_depth": 2,
        "n_estimators": 200,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 10,
    },
    {
        "max_depth": 2,
        "n_estimators": 400,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 10,
    },
    {
        "max_depth": 3,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 10,
    },
]


# ---------------------------------------------------------
# FORECASTING
# ---------------------------------------------------------

def forecast_candidate(kind, parameters, train, future):

    y = train[TARGET].to_numpy(dtype=float)

    # Seasonal naive
    if kind == "SeasonalNaive":
        return np.maximum(0, y[-12:])

    # SARIMA
    if kind == "SARIMA":

        model = SARIMAX(
            y,
            order=parameters["order"],
            seasonal_order=SEASONAL_ORDER,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = model.fit(disp=False, maxiter=300)

        predictions = fitted.get_forecast(
            steps=HORIZON
        ).predicted_mean

        return np.maximum(0, np.asarray(predictions))

    # SARIMAX with climate predictors
    if kind == "SARIMAX":

        scaler = StandardScaler()

        exog_train = scaler.fit_transform(
            train[CLIMATE]
        )

        exog_future = scaler.transform(
            future[CLIMATE]
        )

        model = SARIMAX(
            y,
            exog=exog_train,
            order=parameters["order"],
            seasonal_order=SEASONAL_ORDER,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = model.fit(disp=False, maxiter=300)

        predictions = fitted.get_forecast(
            steps=HORIZON,
            exog=exog_future,
        ).predicted_mean

        return np.maximum(0, np.asarray(predictions))

    # SVR / XGBoost
    if kind in ("SVR", "XGB"):

        X, target = build_ml_training_data(train)

        model = create_ml_model(kind, parameters)
        model.fit(X, target)

        history = list(y)
        predictions = []

        for step in range(HORIZON):

            X_current = make_features(
                history,
                future.iloc[step],
            )

            prediction = float(model.predict(X_current)[0])
            prediction = max(0.0, prediction)

            predictions.append(prediction)
            history.append(prediction)

        return np.asarray(predictions)

    raise ValueError(f"Unknown model: {kind}")


# ---------------------------------------------------------
# MODEL CANDIDATES
# ---------------------------------------------------------

def get_candidates():

    candidates = [
        ("SeasonalNaive", {}),
    ]

    for order in SARIMA_ORDERS:
        candidates.append(("SARIMA", {"order": order}))
        candidates.append(("SARIMAX", {"order": order}))

    for parameters in SVR_CONFIGS:
        candidates.append(("SVR", parameters))

    for parameters in XGB_CONFIGS:
        candidates.append(("XGB", parameters))

    return candidates


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

        predictions = forecast_candidate(
            kind,
            parameters,
            fold_train,
            fold_valid,
        )

        actual = fold_valid[TARGET].to_numpy(dtype=float)

        if (
            len(predictions) != len(actual)
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
# METRICS
# ---------------------------------------------------------

def calculate_metrics(actual, predicted, training):

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    training_y = np.asarray(training, dtype=float)

    mase_denominator = np.mean(
        np.abs(training_y[12:] - training_y[:-12])
    )

    mase = (
        mae / mase_denominator
        if mase_denominator > 0
        else np.nan
    )

    return float(rmse), float(mae), float(mase)


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def run_one(region, year):

    print("\n" + "=" * 70)
    print(f"BRANCH A BENCHMARK | {region.upper()} | {year}")
    print("=" * 70)

    train, test = load_split(region, year)

    candidates = get_candidates()
    results = []

    best_candidate = None
    best_score = np.inf

    print(f"Training observations: {len(train)}")
    print(f"Candidates: {len(candidates)}")

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
        raise RuntimeError("All candidate models failed.")

    # Final untouched test evaluation
    best_kind, best_parameters = best_candidate

    predictions = forecast_candidate(
        best_kind,
        best_parameters,
        train,
        test,
    )

    actual = test[TARGET].to_numpy(dtype=float)

    rmse, mae, mase = calculate_metrics(
        actual,
        predictions,
        train[TARGET],
    )

    print("\nSELECTED MODEL")
    print(best_kind)
    print(best_parameters)

    print("\nFINAL TEST RESULTS")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MASE: {mase:.4f}")
    print(f"Validation RMSE: {best_score:.4f}")

    # Save predictions
    prediction_df = pd.DataFrame({
        "Date": test["Date"],
        "Actual": actual,
        "Predicted": predictions,
        "Residual": actual - predictions,
    })

    prediction_dir = OUTPUT_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(
        prediction_dir / f"{region}_{year}_predictions.csv",
        index=False,
    )

    # Save leaderboard
    results_df = pd.DataFrame(results).sort_values(
        "validation_RMSE"
    )

    cv_dir = OUTPUT_DIR / "cv_results"
    cv_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
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

    summary = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / "experiment_22_summary.csv",
        index=False,
    )

    print("\nFINAL BRANCH A SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
