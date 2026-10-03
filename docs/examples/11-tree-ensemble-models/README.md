# Tree-Based & Ensemble Models

> **View online:** https://deepbox.dev/examples/11-tree-ensemble-models

Decision trees, random forests, gradient boosting and linear SVMs, for classification and regression.

## Deepbox Modules Used

| Module               | Features Used                                                                        |
| -------------------- | ------------------------------------------------------------------------------------ |
| `deepbox/datasets`   | `loadIris`                                                                           |
| `deepbox/ml`         | `DecisionTree*`, `RandomForest*`, `GradientBoosting*` (classifier and regressor), `LinearSVC`, `LinearSVR` |
| `deepbox/ndarray`    | `tensor`, `slice`, `eq`, `astype`                                                    |
| `deepbox/metrics`    | `accuracy`, `mse`, `r2Score`                                                         |
| `deepbox/preprocess` | `trainTestSplit`                                                                     |

## What It Shows

- Eight models, each fitted and scored on a held-out split.
- Gradient boosting and the linear SVC are run on a binary subset of Iris (classes 0 and 1).
- Part 9 compares tree options: `maxLeafNodes` (best-first growth), `ccpAlpha` (cost-complexity pruning), `classWeight`, and a `sampleWeight` argument to `fit`. It prints `getNLeaves()` and `getDepth()` so the effect on tree size is visible. Forests also expose `featureImportances`.
- `clone()` returns an unfitted copy of an estimator.
- `LinearSVR` gets a larger `maxIter` and a `randomState` so it converges without a warning and gives the same numbers on every run.

## Usage

```bash
npm run example:11
```

## Output

Console output only: accuracy for classifiers, MSE and R² for regressors.

## Files

```
11-tree-ensemble-models/
├── index.ts     # Main entry point
└── README.md    # This file
```
