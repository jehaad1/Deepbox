# Kernel SVM & Anomaly Detection

> **View online:** https://deepbox.dev/examples/36-svm-anomaly-detection

Kernel support vector machines (`SVC`, `NuSVC`, `SVR`) and anomaly detectors (`IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`). The SVMs compare kernels and values of `C` on synthetic data. The detectors look for three planted outliers in a small 2-D dataset.

## Deepbox Modules Used

| Module               | Features Used                                                     |
| -------------------- | ----------------------------------------------------------------- |
| `deepbox/ml`         | SVC, SVR, NuSVC, OneClassSVM, IsolationForest, LocalOutlierFactor |
| `deepbox/metrics`    | accuracy, r2Score                                                 |
| `deepbox/datasets`   | makeClassification, makeRegression                                |
| `deepbox/preprocess` | trainTestSplit, StandardScaler                                    |

## Usage

```bash
npm run example:36
```

## Output

- Console output only: test accuracy per kernel and per `C`, R² for `SVR`, and the rows each anomaly detector flags.

## Files

```
36-svm-anomaly-detection/
├── index.ts     # Example script
└── README.md    # This file
```
