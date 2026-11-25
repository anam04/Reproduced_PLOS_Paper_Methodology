
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# EXPERIMENT 29: BRANCH A RESULTS CONSOLIDATION
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "experiment_29"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These are results transcribed from the completed experiment logs.
# Missing values are intentionally left as None, not estimated.

CANDIDATE_RESULTS = [
    # North Goa 2018
    ("North Goa", 2018, 22, "SVR", 18.6393, 36.4431, 31.2421, 0.9756),
    ("North Goa", 2018, 25, "LogRidge alpha=100", 21.5589, 26.2322, 22.6793, 0.7082),
    ("North Goa", 2018, 26, "LogXGB depth=2", 20.2369, 30.9712, 24.8237, 0.7752),
    ("North Goa", 2018, 27, "log1p STL constant trend", 23.7038, 11.8354, 9.4342, 0.2946),
    ("North Goa", 2018, 28, "SVR C=10 unweighted", 22.3659, 23.8821, 19.7027, 0.6153),

    # North Goa 2019
    ("North Goa", 2019, 22, "SARIMA(2,1,1)", 24.7841, 13.4659, 10.0251, 0.3276),
    ("North Goa", 2019, 25, "LogSVR", 23.1184, 20.3803, 19.1903, 0.6271),
    ("North Goa", 2019, 26, "LogSVR C=0.2", 22.7569, 20.6742, 19.0290, 0.6218),
    ("North Goa", 2019, 27, "log1p STL constant trend", 23.7144, 17.2873, 9.6453, 0.3152),
    ("North Goa", 2019, 28, "SVR C=10 unweighted", 23.7014, 23.4223, 22.2408, 0.7267),

    # South Goa 2018
    ("South Goa", 2018, 22, "SVR", 11.4294, 17.1515, 14.5786, 0.6703),
    ("South Goa", 2018, 25, "log-SARIMA", 11.4793, 11.1966, 10.0698, 0.4630),
    ("South Goa", 2018, 26, "BoxCox1p SARIMA", 11.4689, 12.6829, 11.8503, 0.5448),

    # South Goa 2019
    ("South Goa", 2019, 22, "SARIMA(2,1,1)", 10.9140, 5.5024, 4.3991, 0.2201),
    ("South Goa", 2019, 25, "log-SARIMA", 10.2406, 2.5234, 1.9704, 0.0986),
    ("South Goa", 2019, 26, "BoxCox1p SARIMA", 10.1355, 2.4399, 2.0130, 0.1007),
]

COLUMNS = [
    "district",
    "year",
    "experiment",
    "model",
    "validation_RMSE",
    "test_RMSE",
    "test_MAE",
    "test_MASE",
]

# Historical best test results across the experiment series.
# These are retrospective benchmarks, NOT model-selection criteria.
HISTORICAL_BEST = [
    {
        "district": "North Goa",
        "year": 2018,
        "experiment": 27,
        "model": "log1p STL constant trend",
        "test_RMSE": 11.8354,
        "test_MAE": 9.4342,
        "test_MASE": 0.2946,
    },
    {
        "district": "North Goa",
        "year": 2019,
        "experiment": 22,
        "model": "SARIMA(2,1,1)",
        "test_RMSE": 13.4659,
        "test_MAE": 10.0251,
        "test_MASE": 0.3276,
    },
    {
        "district": "South Goa",
        "year": 2018,
        "experiment": 7,
        "model": "Raw SARIMA",
        "test_RMSE": 7.85,
        "test_MAE": None,
        "test_MASE": None,
    },
    {
        "district": "South Goa",
        "year": 2019,
        "experiment": 17,
        "model": "Box-Cox SARIMA",
        "test_RMSE": 2.373,
        "test_MAE": None,
        "test_MASE": None,
    },
]


def build_candidate_table():
    frame = pd.DataFrame(
        CANDIDATE_RESULTS,
        columns=COLUMNS,
    )

    frame = frame.sort_values(
        ["district", "year", "validation_RMSE"]
    ).reset_index(drop=True)

    return frame


def build_validation_leaders(candidates):
    """
    Select the lowest recorded validation RMSE for each
    district-year among the experiments with recorded
    validation metrics.

    This is a provisional cross-experiment ranking.
    Validation designs and candidate pools were not identical.
    """
    indices = (
        candidates.groupby(
            ["district", "year"]
        )["validation_RMSE"].idxmin()
    )

    leaders = candidates.loc[indices].copy()

    leaders = leaders.rename(columns={
        "experiment": "validation_leader_experiment",
        "model": "validation_leader_model",
        "test_RMSE": "leader_test_RMSE",
        "test_MAE": "leader_test_MAE",
        "test_MASE": "leader_test_MASE",
    })

    return leaders.sort_values(
        ["district", "year"]
    ).reset_index(drop=True)


def build_historical_benchmarks():
    return pd.DataFrame(HISTORICAL_BEST)


def build_generalisation_comparison(leaders, historical):
    comparison = leaders.merge(
        historical[
            [
                "district",
                "year",
                "experiment",
                "model",
                "test_RMSE",
            ]
        ].rename(columns={
            "experiment": "historical_best_experiment",
            "model": "historical_best_model",
            "test_RMSE": "historical_best_test_RMSE",
        }),
        on=["district", "year"],
        how="left",
    )

    comparison["validation_to_test_gap"] = (
        comparison["leader_test_RMSE"]
        - comparison["validation_RMSE"]
    )

    comparison["historical_test_RMSE_difference"] = (
        comparison["leader_test_RMSE"]
        - comparison["historical_best_test_RMSE"]
    )

    return comparison


def collect_saved_forecasts():
    """
    Collect saved monthly forecast files from experiment folders.
    The experiment number is inferred from the folder name.
    """
    forecast_files = list(
        RESULTS_DIR.glob("experiment_*/north_goa_*_forecasts.csv")
    )

    frames = []

    for path in forecast_files:
        try:
            frame = pd.read_csv(path)

            if not {"date", "actual", "predicted"}.issubset(
                frame.columns
            ):
                continue

            experiment_name = path.parent.name
            experiment_number = experiment_name.replace(
                "experiment_", ""
            )

            frame["experiment"] = experiment_number
            frame["source_file"] = str(path)

            year_token = path.stem.split("_")[-2]
            frame["year"] = int(year_token)
            frame["district"] = "North Goa"

            frames.append(frame)

        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def write_markdown_report(
    candidates,
    leaders,
    historical,
    comparison,
):
    lines = [
        "# Branch A: Consolidated Malaria Forecasting Results",
        "",
        "## 1. Validation-based candidate ranking",
        "",
        "The following candidates have the lowest recorded validation "
        "RMSE for each district-year among the included experiments.",
        "",
        "| District | Year | Experiment | Model | Validation RMSE | Test RMSE |",
        "|---|---:|---:|---|---:|---:|",
    ]

    for _, row in leaders.iterrows():
        lines.append(
            f"| {row['district']} "
            f"| {row['year']} "
            f"| {int(row['validation_leader_experiment'])} "
            f"| {row['validation_leader_model']} "
            f"| {row['validation_RMSE']:.4f} "
            f"| {row['leader_test_RMSE']:.4f} |"
        )

    lines.extend([
        "",
        "**Interpretation:** These are provisional validation leaders "
        "across recorded experiments. Candidate pools and validation "
        "designs differed, so this ranking is descriptive rather than "
        "a strictly controlled model-selection comparison.",
        "",
        "## 2. Historical best observed test performance",
        "",
        "| District | Year | Experiment | Model | Test RMSE | Test MAE | Test MASE |",
        "|---|---:|---:|---|---:|---:|---:|",
    ])

    for _, row in historical.iterrows():
        mae = (
            f"{row['test_MAE']:.4f}"
            if pd.notna(row["test_MAE"])
            else "Not recorded"
        )
        mase = (
            f"{row['test_MASE']:.4f}"
            if pd.notna(row["test_MASE"])
            else "Not recorded"
        )

        lines.append(
            f"| {row['district']} "
            f"| {row['year']} "
            f"| {int(row['experiment'])} "
            f"| {row['model']} "
            f"| {row['test_RMSE']:.4f} "
            f"| {mae} "
            f"| {mase} |"
        )

    lines.extend([
        "",
        "**Important:** Historical best test scores are retrospective "
        "benchmarks. They must not be used to select models or presented "
        "as unbiased estimates after repeated experimentation.",
        "",
        "## 3. Generalisation comparison",
        "",
        "| District | Year | Validation RMSE | Leader test RMSE | Historical best test RMSE |",
        "|---|---:|---:|---:|---:|",
    ])

    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['district']} "
            f"| {row['year']} "
            f"| {row['validation_RMSE']:.4f} "
            f"| {row['leader_test_RMSE']:.4f} "
            f"| {row['historical_best_test_RMSE']:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Methodological limitations",
        "",
        "- Each test set contains only 12 monthly observations.",
        "- Validation protocols and candidate pools varied across experiments.",
        "- Climate predictors are observed test-period climate values. "
        "Forecasts are therefore conditional on known climate, not "
        "fully operational forecasts using predicted weather.",
        "- Historical best test results are descriptive and should not "
        "be treated as unbiased model-selection outcomes.",
        "- North Goa exhibits sharp incidence peaks and substantial "
        "forecasting uncertainty.",
        "- Missing metrics are left unreported rather than reconstructed.",
        "",
        "## 5. Reproducibility",
        "",
        "This report consolidates recorded experiment outputs. It does "
        "not retrain models or modify the original experiment results.",
        "",
    ])

    report_path = OUTPUT_DIR / "branch_a_consolidated_report.md"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path


def main():
    candidates = build_candidate_table()
    leaders = build_validation_leaders(candidates)
    historical = build_historical_benchmarks()

    comparison = build_generalisation_comparison(
        leaders,
        historical,
    )

    candidates.to_csv(
        OUTPUT_DIR / "all_recorded_candidates.csv",
        index=False,
    )

    leaders.to_csv(
        OUTPUT_DIR / "validation_leaders.csv",
        index=False,
    )

    historical.to_csv(
        OUTPUT_DIR / "historical_best_test_benchmarks.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "generalisation_comparison.csv",
        index=False,
    )

    forecasts = collect_saved_forecasts()

    if not forecasts.empty:
        forecasts.to_csv(
            OUTPUT_DIR / "north_goa_monthly_forecast_comparison.csv",
            index=False,
        )

    report_path = write_markdown_report(
        candidates,
        leaders,
        historical,
        comparison,
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT 29: BRANCH A CONSOLIDATION")
    print("=" * 72)

    print("\nValidation leaders:")
    print(
        leaders[
            [
                "district",
                "year",
                "validation_leader_experiment",
                "validation_leader_model",
                "validation_RMSE",
                "leader_test_RMSE",
            ]
        ].to_string(index=False)
    )

    print("\nHistorical best test benchmarks:")
    print(
        historical[
            [
                "district",
                "year",
                "experiment",
                "model",
                "test_RMSE",
            ]
        ].to_string(index=False)
    )

    print("\nGeneralisation comparison:")
    print(
        comparison[
            [
                "district",
                "year",
                "validation_RMSE",
                "leader_test_RMSE",
                "historical_best_test_RMSE",
                "validation_to_test_gap",
            ]
        ].to_string(index=False)
    )

    print(f"\nMonthly forecast files collected: {len(forecasts)}")
    print(f"Report saved to: {report_path}")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
