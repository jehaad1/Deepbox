# Cross-Validation

> **View online:** https://deepbox.dev/examples/23-cross-validation

Cross-validation scores a model on several different train/test splits, so one lucky or unlucky split does not decide the result. The example covers the splitters and `crossValScore`, which runs the whole loop.

## Deepbox Modules Used

| Module               | Features Used                                           |
| -------------------- | ------------------------------------------------------- |
| `deepbox/preprocess` | `KFold`, `StratifiedKFold`, `LeaveOneOut`               |
| `deepbox/ml`         | `LinearRegression`, `crossValScore`                     |
| `deepbox/ndarray`    | `tensor`, `arange`                                      |
| `deepbox/random`     | `setSeed`, `rand`                                       |

## What It Shows

- `KFold` yields `{ trainIndex, testIndex }` row indices for each fold. `shuffle` and `randomState` control the order.
- `StratifiedKFold` keeps class proportions. The script counts the test rows of each class per fold to show it.
- `LeaveOneOut` makes one fold per row.
- `crossValScore(model, X, y, 5)` fits and scores the model on each fold and returns one score per fold. For a regressor the score is R². For classifiers the folds are stratified.
- `setSeed` makes the synthetic data, and so the printed scores, the same on every run.

## Usage

```bash
npm run example:23
```

## Output

Console output only.

## Files

```
23-cross-validation/
├── index.ts     # Main entry point
└── README.md    # This file
```
