# Logistic Regression

> **View online:** https://deepbox.dev/examples/08-logistic-regression

Binary classification with logistic regression on the Iris dataset: setosa vs the other two species.

## Deepbox Modules Used

| Module               | Features Used                                               |
| -------------------- | ----------------------------------------------------------- |
| `deepbox/datasets`   | `loadIris`                                                  |
| `deepbox/ml`         | `LogisticRegression`                                        |
| `deepbox/metrics`    | `accuracy`, `precision`, `recall`, `f1Score`, `confusionMatrix` |
| `deepbox/preprocess` | `trainTestSplit`, `StandardScaler`                          |

## What It Shows

- `iris.target.clip(0, 1)` maps the labels 0, 1, 2 to 0, 1, 1.
- Features are standardized with a scaler fitted on the training set only.
- Metrics are plain numbers. `confusionMatrix` returns a tensor with true classes as rows and predicted classes as columns.
- Setosa is linearly separable from the other species, so scores of 100% on the test split are expected here. They say little about harder problems.

## Usage

```bash
npm run example:08
```

## Output

Console output only: accuracy, precision, recall, F1 and the confusion matrix.

## Files

```
08-logistic-regression/
├── index.ts     # Main entry point
└── README.md    # This file
```
