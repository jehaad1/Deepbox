# Fraud Detection Platform

> **View online:** https://deepbox.dev/projects/07-fraud-detection-platform

A production-style transaction monitoring workflow built around the biggest new v1.0.0 additions: calibrated classification, anomaly detection, feature inspection, and operations-ready reporting.

## Features

- **Synthetic transaction stream** with merchant, channel, country, velocity, and device-risk signals
- **Supervised fraud scoring** using `LogisticRegression`
- **Probability calibration** with `CalibratedClassifierCV`
- **Unsupervised anomaly detection** with `IsolationForest`, `LocalOutlierFactor`, and `OneClassSVM`
- **Inspection** via `permutationImportance`
- **Operational reporting** with DataFrame querying, grouping, and JSON export

## Deepbox Modules Used

| Module               | Features Used                                                                 |
| -------------------- | ----------------------------------------------------------------------------- |
| `deepbox/ml`         | LogisticRegression, CalibratedClassifierCV, IsolationForest, LocalOutlierFactor, OneClassSVM, calibrationCurve, permutationImportance |
| `deepbox/preprocess` | trainTestSplit, StandardScaler                                                |
| `deepbox/metrics`    | accuracy, precision, recall, f1Score                                          |
| `deepbox/dataframe`  | DataFrame, query, groupBy, to_datetime                                        |
| `deepbox/plot`       | plotCalibrationCurve, plotFeatureImportance, saveFig                          |
| `deepbox/ndarray`    | tensor                                                                        |

## Usage

```bash
npm run project:07
```

## Output

- `output/calibration-curve.svg`
- `output/feature-importance.svg`
- `output/model-report.json`

## Architecture

```text
07-fraud-detection-platform/
├── index.ts
├── README.md
└── output/
```
