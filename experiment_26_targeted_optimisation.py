
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.special import boxcox1p, inv_boxcox1p

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data/processed/district_splits"
OUTPUT_DIR = ROOT / "results/experiment_26_targeted_optimisation"

TARGET = "Malaria_Cases"
HORIZON = 12
N_SPLITS = 3
SEASON = 12
MIN_TRAIN_SIZE = 48
RANDOM_STATE = 42

CLIMATE = [
    "max_temperature",
    "min_temperature",
    "total_rainfall",
    "avg_relative_humidity",
    "max_pressure",
    "min_wind_speed",
]

AUTOREG_FEATURES = [
    "lag_1", "lag_2", "lag_3",
    "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6",
    "month_sin", "month_cos",
]

FEATURE_SETS = {
    "autoregressive": AUTOREG_FEATURES,
    "climate_only": [
        "month_sin", "month_cos",
    ] + CLIMATE,
    "combined": AUTOREG_FEATURES + CLIMATE,
}


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
            raise ValueError("Missing values detected.")
        if (frame[TARGET] < 0).any():
            raise ValueError("Negative malaria counts detected.")

    if len(test) != HORIZON:
        raise ValueError("Expected 12 test observations.")

    if len(train) < MIN_TRAIN_SIZE + N_SPLITS * HORIZON:
        raise ValueError("Insufficient training history.")

    return train, test


# =========================================================
# FEATURES
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

    return pd.DataFrame([row])


def build_training_data(train, horizon, feature_set):

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

    features = FEATURE_SETS[feature_set]

    return (
        pd.DataFrame(rows)[features],
        np.asarray(targets, dtype=float),
    )


# =========================================================
# CANDIDATE GENERATION
# =========================================================

def add_candidate(candidates, kind, parameters):
    candidates.append((kind, parameters))


def get_candidates(region, year):

    candidates = [("SeasonalNaive", {})]

    if region == "north_goa" and year == 2018:

        # Primary: LogRidge refinement.
        for alpha in [10, 30, 50, 75, 100, 150, 200, 300]:
            for features in ["autoregressive", "combined"]:
                add_candidate(candidates, "LogRidge", {
                    "alpha": alpha,
                    "feature_set": features,
                })

        # Secondary: XGBoost.
        for depth in [1, 2]:
            for estimators in [100, 200, 300]:
                add_candidate(candidates, "LogXGB", {
                    "max_depth": depth,
                    "n_estimators": estimators,
                    "learning_rate": 0.03,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "reg_lambda": 10,
                    "feature_set": "combined",
                })

        # Retain LogSVR as a comparator.
        for C in [0.05, 0.1, 0.2, 0.5, 1.0]:
            add_candidate(candidates, "LogSVR", {
                "C": C,
                "gamma": "scale",
                "epsilon": 0.1,
                "feature_set": "combined",
            })

    elif region == "north_goa" and year == 2019:

        # Primary: LogSVR refinement.
        for C in [0.025, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for gamma in ["scale", 0.005, 0.01, 0.02]:
                for features in ["autoregressive", "combined"]:
                    add_candidate(candidates, "LogSVR", {
                        "C": C,
                        "gamma": gamma,
                        "epsilon": 0.1,
                        "feature_set": features,
                    })

        # Secondary statistical candidates.
        add_candidate(candidates, "LogRidge", {
            "alpha": 100,
            "feature_set": "combined",
        })

    elif region == "south_goa":

        # Focus on SARIMA order and transformation.
        orders = [
            (1, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
            (2, 1, 1),
            (2, 1, 0),
        ]

        seasonal_orders = [
            (0, 0, 0, 12),
            (1, 0, 0, 12),
            (0, 0, 1, 12),
            (0, 1, 1, 12),
            (1, 1, 0, 12),
        ]

        transforms = [
            ("identity", None),
            ("log1p", 0.0),
            ("boxcox1p", 0.25),
            ("boxcox1p", 0.5),
        ]

        for order in orders:
            for seasonal_order in seasonal_orders:
                for transform, lam in transforms:

                    # Avoid an unnecessarily large search.
                    if (
                        transform == "identity"
                        and seasonal_order in [
                            (0, 0, 0, 12),
                            (1, 0, 0, 12),
                        ]
                    ) or transform != "identity":

                        add_candidate(candidates, "SARIMA", {
                            "order": order,
                            "seasonal_order": seasonal_order,
                            "transform": transform,
                            "lambda": lam,
                        })

        # Climate-informed statistical comparator.
        for order in [(1, 1, 1), (2, 1, 1)]:
            add_candidate(candidates, "SARIMAX", {
                "order": order,
                "seasonal_order": (0, 0, 0, 12),
                "transform": "log1p",
                "lambda": 0.0,
            })

        # Retain the previously competitive ML candidate.
        for C in [0.1, 1.0, 10.0]:
            add_candidate(candidates, "LogSVR", {
                "C": C,
                "gamma": "scale",
                "epsilon": 0.1,
                "feature_set": "combined",
            })

    return candidates


# =========================================================
# TRANSFORMATIONS
# =========================================================

def transform_target(y, transform, lam):

    if transform == "identity":
        return np.asarray(y, dtype=float)

    if transform == "log1p":
        return np.log1p(y)

    if transform == "boxcox1p":
        return boxcox1p(y, lam)

    raise ValueError(transform)


def inverse_target(y, transform, lam):

    y = np.asarray(y, dtype=float)

    if transform == "identity":
        return y

    if transform == "log1p":
        return np.expm1(np.clip(y, -20, 20))

    if transform == "boxcox1p":
        return inv_boxcox1p(np.clip(y, -20, 20), lam)

    raise ValueError(transform)


# =========================================================
# MACHINE LEARNING MODELS
# =========================================================

def create_ml_model(kind, parameters):

    if kind == "LogRidge":

        estimator = Ridge(alpha=parameters["alpha"])

    elif kind == "LogSVR":

        estimator = SVR(
            kernel="rbf",
            C=parameters["C"],
            gamma=parameters["gamma"],
            epsilon=parameters["epsilon"],
        )

    elif kind == "LogXGB":

        xgb_params = {
            key: value
            for key, value in parameters.items()
            if key != "feature_set"
        }

        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
            **xgb_params,
        )

    else:
        raise ValueError(kind)

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", estimator),
    ])

    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def ml_forecast(kind, parameters, train, future):

    history = train[TARGET].to_numpy(dtype=float)
    feature_set = parameters["feature_set"]
    features = FEATURE_SETS[feature_set]

    predictions = []

    for horizon in range(1, HORIZON + 1):

        X_train, y_train = build_training_data(
            train,
            horizon,
            feature_set,
        )

        model = create_ml_model(kind, parameters)
        model.fit(X_train, y_train)

        target_row = future.iloc[horizon - 1]

        X_future = make_feature_row(
            history,
            target_row,
            target_row["Date"],
        )[features]

        prediction = float(model.predict(X_future)[0])

        predictions.append(prediction)

    return np.asarray(predictions, dtype=float)


# =========================================================
# SARIMA / SARIMAX
# =========================================================

def statistical_forecast(kind, parameters, train, future):

    y = train[TARGET].to_numpy(dtype=float)

    transform = parameters["transform"]
    lam = parameters["lambda"]

    y_transformed = transform_target(y, transform, lam)

    exog_train = None
    exog_future = None

    if kind == "SARIMAX":

        scaler = StandardScaler()
        exog_train = scaler.fit_transform(train[CLIMATE])
        exog_future = scaler.transform(future[CLIMATE])

    model = SARIMAX(
        y_transformed,
        exog=exog_train,
        order=parameters["order"],
        seasonal_order=parameters["seasonal_order"],
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

    forecast = inverse_target(
        np.asarray(forecast, dtype=float),
        transform,
        lam,
    )

    return forecast


# =========================================================
# FORECAST DISPATCH
# =========================================================

def forecast(candidate, train, future):

    kind, parameters = candidate

    if kind == "SeasonalNaive":
        return np.maximum(
            0.0,
            train[TARGET].to_numpy(dtype=float)[-HORIZON:],
        )

    if kind in ("SARIMA", "SARIMAX"):
        return statistical_forecast(
            kind,
            parameters,
            train,
            future,
        )

    return ml_forecast(
        kind,
        parameters,
        train,
        future,
    )


# =========================================================
# VALIDATION
# =========================================================

def validation_folds(train):

    initial_size = len(train) - N_SPLITS * HORIZON

    if initial_size < MIN_TRAIN_SIZE:
        raise ValueError("Insufficient initial training window.")

    for fold in range(N_SPLITS):

        train_end = initial_size + fold * HORIZON
        valid_end = train_end + HORIZON

        yield (
            train.iloc[:train_end].reset_index(drop=True),
            train.iloc[train_end:valid_end].reset_index(drop=True),
        )


def evaluate_candidate(candidate, train):

    actual_all = []
    predicted_all = []

    for fold_train, fold_valid in validation_folds(train):

        predictions = forecast(
            candidate,
            fold_train,
            fold_valid,
        )

        if not np.all(np.isfinite(predictions)):
            return np.inf

        upper_limit = max(
            500.0,
            10.0 * float(fold_train[TARGET].max()),
        )

        if np.max(np.abs(predictions)) > upper_limit:
            return np.inf

        predictions = np.maximum(0.0, predictions)

        actual_all.extend(
            fold_valid[TARGET].to_numpy(dtype=float).tolist()
        )
        predicted_all.extend(predictions.tolist())

    return float(np.sqrt(mean_squared_error(
        actual_all,
        predicted_all,
    )))


# =========================================================
# SELECTION
# =========================================================

def select_model(region, year, train):

    candidates = get_candidates(region, year)
    results = []

    print("\n" + "=" * 75)
    print(f"TARGETED OPTIMISATION: {region.upper()} | {year}")
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
            f"{number:03d}/{len(candidates)} "
            f"{kind:12s} RMSE={score:.4f}"
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

    best_candidate = next(
        candidate
        for candidate in candidates
        if (
            candidate[0] == best_row["model"]
            and str(candidate[1]) == best_row["parameters"]
        )
    )

    print("\nSELECTED MODEL")
    print(best_candidate)
    print(f"Validation RMSE: {best_row['validation_RMSE']:.4f}")

    return (
        best_candidate,
        float(best_row["validation_RMSE"]),
        leaderboard,
    )


# =========================================================
# FINAL TEST EVALUATION
# =========================================================

def evaluate_test(region, year, train, test, candidate, validation_score):

    predictions = forecast(candidate, train, test)

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Non-finite test predictions.")

    predictions = np.maximum(0.0, predictions)

    actual = test[TARGET].to_numpy(dtype=float)

    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(mean_absolute_error(actual, predictions))

    denominator = np.mean(
        np.abs(
            train[TARGET].to_numpy(dtype=float)[12:]
            - train[TARGET].to_numpy(dtype=float)[:-12]
        )
    )

    mase = mae / denominator if denominator > 0 else np.nan

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

    kind, parameters = candidate

    print("\nTEST RESULTS")
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
        OUTPUT_DIR / "experiment_26_summary.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 26 FINAL RESULTS")
    print("=" * 75)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
