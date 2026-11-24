# Corrected Branch A Results Audit

## 1. Historical best observed test performance

| District | Year | Experiment | Forecast file | RMSE | MAE | MASE |
|---|---:|---|---|---:|---:|---:|
| North Goa | 2018 | experiment_27 | north_goa_2018_forecasts | 11.8354 | 9.4342 | 0.2946 |
| North Goa | 2019 | experiment_22_branch_a | north_goa_2019_predictions | 13.4659 | 10.0251 | 0.3276 |
| South Goa | 2018 | district_sarima | sarima_south_goa_2018_predictions | 7.8540 | 6.3037 | 0.2898 |
| South Goa | 2019 | experiment_18_model_selection | experiment_18_south_goa_2019_predictions | 2.3726 | 1.7608 | 0.0881 |

**Interpretation:** These are the lowest recomputed test RMSE values among the discovered, district-labelled, 12-observation forecast files. They are retrospective exploratory benchmarks, not unbiased final test estimates.

## 2. Validation leaders within each experiment

| District | Year | Experiment | Candidate | Validation RMSE |
|---|---:|---|---|---:|
| North Goa | 2018.0 | district_xgb_arma_rebuilt | 9.0 | 18.4370 |
| North Goa | 2018.0 | experiment_22_branch_a | SVR | 18.6393 |
| North Goa | 2018.0 | experiment_18_model_selection | SVR | 18.9919 |
| North Goa | 2018.0 | experiment_17_district_hybrids | 28 | 19.2020 |
| North Goa | 2018.0 | experiment_26_targeted_optimisation | LogXGB | 20.2369 |
| North Goa | 2018.0 | experiment_16_district_hybrids | 29 | 21.2628 |
| North Goa | 2018.0 | experiment_24_district_specific | LogRidge | 21.4031 |
| North Goa | 2018.0 | experiment_25_independent_selection | LogRidge | 21.5589 |
| North Goa | 2018.0 | experiment_23_direct | SVR | 21.7090 |
| North Goa | 2019.0 | experiment_26_targeted_optimisation | LogSVR | 22.7569 |
| North Goa | 2019.0 | experiment_17_district_hybrids | 17 | 22.9499 |
| North Goa | 2019.0 | experiment_24_district_specific | LogSVR | 23.0407 |
| North Goa | 2019.0 | experiment_25_independent_selection | LogSVR | 23.1184 |
| North Goa | 2019.0 | experiment_16_district_hybrids | 17 | 23.3782 |
| North Goa | 2019.0 | experiment_23_direct | SVR | 23.5555 |
| North Goa | 2019.0 | experiment_22_branch_a | SARIMA | 24.7841 |
| North Goa | 2019.0 | district_xgb_arma_rebuilt | 9.0 | 25.1034 |
| North Goa | 2019.0 | experiment_18_model_selection | SVR | 25.5675 |
| South Goa | 2018.0 | experiment_18_model_selection | SARIMA | 10.7372 |
| South Goa | 2018.0 | experiment_22_branch_a | SVR | 11.4294 |
| South Goa | 2018.0 | experiment_26_targeted_optimisation | SARIMA | 11.4689 |
| South Goa | 2018.0 | experiment_25_independent_selection | SARIMA | 11.4793 |
| South Goa | 2018.0 | district_xgb_arma_rebuilt | 11.0 | 14.6502 |
| South Goa | 2018.0 | experiment_17_district_hybrids | 11.0 | 14.6502 |
| South Goa | 2018.0 | experiment_16_district_hybrids | 8.0 | 15.3005 |
| South Goa | 2018.0 | experiment_23_direct | SeasonalNaive | 15.5009 |
| South Goa | 2018.0 | experiment_24_district_specific | SeasonalNaive | 15.5009 |
| South Goa | 2019.0 | experiment_26_targeted_optimisation | SARIMA | 10.1355 |
| South Goa | 2019.0 | experiment_18_model_selection | SARIMA | 10.1466 |
| South Goa | 2019.0 | experiment_25_independent_selection | SARIMA | 10.2406 |
| South Goa | 2019.0 | experiment_22_branch_a | SARIMA | 10.9140 |
| South Goa | 2019.0 | experiment_24_district_specific | HistPoisson | 11.0653 |
| South Goa | 2019.0 | experiment_23_direct | SeasonalNaive | 11.3456 |
| South Goa | 2019.0 | district_xgb_arma_rebuilt | 29.0 | 12.5957 |
| South Goa | 2019.0 | experiment_17_district_hybrids | 29.0 | 12.5957 |
| South Goa | 2019.0 | experiment_16_district_hybrids | 8.0 | 14.3256 |

Each row is the lowest recorded candidate validation RMSE within that experiment's saved leaderboard. The validation protocol may differ across experiments.

## 3. Descriptive lowest validation scores across discovered files

| District | Year | Experiment | Candidate | Validation RMSE |
|---|---:|---|---|---:|
| North Goa | 2018.0 | district_xgb_arma_rebuilt | 9.0 | 18.4370 |
| North Goa | 2019.0 | experiment_26_targeted_optimisation | LogSVR | 22.7569 |
| South Goa | 2018.0 | experiment_18_model_selection | SARIMA | 10.7372 |
| South Goa | 2019.0 | experiment_26_targeted_optimisation | SARIMA | 10.1355 |

**Warning:** This is not a definitive model-selection result. The candidate pools, transformations, training windows and validation procedures differ. The smallest number must not automatically be declared the final model.

## 4. Evaluation coverage

- Saved district-specific forecast files included: 50
- Candidate validation records collected: 1295
- Forecast files skipped: 0
- Validation files skipped: 12

## 5. Methodological limitations

- Each test evaluation contains 12 monthly observations.
- Repeated inspection of test results makes historical best test scores exploratory rather than untouched final estimates.
- Validation procedures and candidate pools vary across experiments.
- MASE uses the seasonal-naïve scale calculated from the corresponding district-year training series.
- Climate predictors are observed test-period climate values. The forecasts are conditional on known climate, not operational forecasts using predicted weather.
- The audit includes only forecast files whose filenames explicitly identify a district and year and which contain 12 valid actual-predicted pairs.
- Missing or ambiguous files are not silently assigned to a district or evaluation year.

## 6. Reproducibility

All test metrics in this audit are recomputed from saved prediction CSVs. Original experiment files are not modified.
