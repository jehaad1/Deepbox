# Advanced ML Models

> **View online:** https://deepbox.dev/examples/10-advanced-ml-models

Five models on tiny datasets: KMeans clustering, K-Nearest Neighbors for classification and regression, PCA, and Gaussian Naive Bayes.

## Deepbox Modules Used

| Module               | Features Used                                                           |
| -------------------- | ----------------------------------------------------------------------- |
| `deepbox/ml`         | `KMeans`, `KNeighborsClassifier`, `KNeighborsRegressor`, `PCA`, `GaussianNB` |
| `deepbox/ndarray`    | `tensor`, `sum`, `item`                                                 |
| `deepbox/metrics`    | `accuracy`                                                              |
| `deepbox/preprocess` | `trainTestSplit`                                                        |

## What It Shows

- `KMeans` exposes `clusterCenters`, `inertia` and `nIter` after fitting.
- `KNeighborsClassifier` returns class probabilities with `predictProba`.
- `clone()` returns an unfitted estimator with the same settings.
- `PCA` reduces 3 features to 2, reports `explainedVarianceRatio` and maps back with `inverseTransform`. The total explained variance is read with `sum().item()`.
- Supervised models share `fit(X, y)`, `predict(X)` and `score(X, y)`.

## Usage

```bash
npm run example:10
```

## Output

Console output only.

## Files

```
10-advanced-ml-models/
├── index.ts     # Main entry point
└── README.md    # This file
```
