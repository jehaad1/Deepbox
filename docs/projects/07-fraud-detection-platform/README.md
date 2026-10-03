# Fraud Detection Platform

> **View online:** https://deepbox.dev/projects/07-fraud-detection-platform

Scores 1,500 synthetic card transactions in four parts: reporting with a DataFrame, a supervised classifier with calibrated probabilities, three anomaly detectors trained on legitimate rows only, and model inspection.

## Features

- Synthetic transaction stream with merchant, channel, country, velocity and device-risk signals
- Supervised fraud scoring with `LogisticRegression`
- Probability calibration with `CalibratedClassifierCV`
- Anomaly detection with `IsolationForest`, `LocalOutlierFactor` and `OneClassSVM`
- Inspection with `permutationImportance`
- Reporting with DataFrame `query`, `groupBy` and JSON export

## Deepbox Modules Used

| Module               | Features Used                                                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/ml`         | `LogisticRegression`, `CalibratedClassifierCV`, `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `calibrationCurve`, `permutationImportance` |
| `deepbox/preprocess` | `trainTestSplit`, `StandardScaler`                                                                                                                  |
| `deepbox/metrics`    | `accuracy`, `precision`, `recall`, `f1Score`                                                                                                        |
| `deepbox/dataframe`  | `DataFrame`, `query`, `groupBy`, `toDatetime`                                                                                                       |
| `deepbox/plot`       | `figure`, `plotCalibrationCurve`, `plotFeatureImportance`, `saveFig`                                                                                |
| `deepbox/ndarray`    | `tensor`                                                                                                                                            |

## Usage

```bash
npm run project:07
```

## Output

- `output/calibration-curve.svg`
- `output/feature-importance.svg`
- `output/model-report.json`

## Reading the anomaly detector numbers

The detectors never see a fraud label. Each one is fitted on legitimate training rows, then asked to flag outliers in the test set. Two numbers matter together:

- Fraud recall: the share of fraud rows the detector flagged.
- False alarm rate: the share of legitimate rows it flagged.

A detector that flags every row has a recall of 1.0 and is useless. Always read recall next to the false alarm rate.

Current results on the seeded data (about 15% of rows are fraud):

| Detector           | Fraud recall | False alarm rate |
| ------------------ | ------------ | ---------------- |
| IsolationForest    | 0.339        | 0.179            |
| LocalOutlierFactor | 0.321        | 0.141            |
| OneClassSVM        | 0.429        | 0.238            |

### OneClassSVM recall in 1.0.0 vs 1.5.0

Deepbox 1.0.0 printed a OneClassSVM recall of about 0.97 for this project. That value came from a bug, not from a good model. The 1.0.0 solver flagged most rows as outliers: on the same data with `nu = 0.1` it flagged 88.3% of the training set, although `nu` caps that share at about 10%. Nearly every transaction was flagged, so nearly every fraud row was too.

Since 1.5.0 the fit is correct. With `nu = 0.1` it flags 10.3% of the training set, the same as scikit-learn. On this project's data (`nu = 0.181`) it flags 18.3% of its training rows, and the recall is 0.429.

The 1.5.0 numbers were checked against scikit-learn 1.8: `OneClassSVM(nu=0.1808, kernel="rbf", gamma="scale")` fitted on the same scaled training rows predicts the same label as Deepbox for every test row, with the same recall (0.429) and false alarm rate (0.238).

A recall near 0.43 is the honest figure for this data. The fraud rows overlap legitimate rows in feature space, so an unsupervised detector has to flag many legitimate transactions to catch about 4 in 10 frauds. For comparison, the supervised `LogisticRegression`, which does use the labels, reaches a recall of 0.571 with a false alarm rate of 0.201 at its 0.18 threshold.

## Architecture

```text
07-fraud-detection-platform/
├── index.ts
├── README.md
└── output/
```
