# Machine Learning Pipeline

> **View online:** https://deepbox.dev/examples/06-ml-pipeline

Two end-to-end workflows. Part 1 classifies Iris (setosa vs the rest). Part 2 compares four linear regression models on the Housing-Mini dataset, cross-validates one and plots its predictions.

## Deepbox Modules Used

| Module               | Features Used                                                        |
| -------------------- | -------------------------------------------------------------------- |
| `deepbox/datasets`   | `loadIris`, `loadHousingMini`                                        |
| `deepbox/ml`         | `LogisticRegression`, `LinearRegression`, `Ridge`, `Lasso`, `crossValidate` |
| `deepbox/metrics`    | `accuracy`, `precision`, `recall`, `f1Score`, `confusionMatrix`, `r2Score`, `mse`, `mae`, `rmse` |
| `deepbox/preprocess` | `trainTestSplit`, `StandardScaler`                                   |
| `deepbox/plot`       | `Figure`, `scatter`, `plot`, `renderSVG`                             |

## What It Shows

- `iris.target.clip(0, 1)` turns the three class labels into two without a loop.
- The scaler is fitted on the training split only and then applied to both splits.
- Metric functions return plain numbers, so no `Number(...)` wrapper is needed.
- `crossValidate` takes a `scoring` object of functions and returns per-fold scores in `testScores`.
- The plot passes tensors directly. The red line marks where predictions equal true values.
- The closing summary is computed from the results.

## Usage

```bash
npm run example:06
```

## Output

One SVG file is written to `output/`: `predictions-vs-actual.svg`, a scatter plot of test predictions against true values.

## Files

```
06-ml-pipeline/
├── index.ts     # Main entry point
├── README.md    # This file
└── output/      # Generated SVG chart
```
