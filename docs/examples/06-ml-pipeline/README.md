# Complete Machine Learning Pipeline

End-to-end ML pipeline demonstrating classification with the Iris dataset and regression with the Housing-Mini dataset. Includes data preprocessing, model comparison, cross-validation, and visualization.

## Deepbox Modules Used

| Module               | Features Used                                      |
| -------------------- | -------------------------------------------------- |
| `deepbox/datasets`   | loadIris, loadHousingMini                          |
| `deepbox/ml`         | LogisticRegression, LinearRegression, Ridge, Lasso |
| `deepbox/metrics`    | accuracy, precision, recall, f1Score, r2Score, mse |
| `deepbox/preprocess` | trainTestSplit, StandardScaler, KFold              |
| `deepbox/plot`       | Figure, scatter, plot                              |

## Usage

```bash
npm run example:06
```

## Output

- 1 SVG visualization in `output/`:
  - `predictions-vs-actual.svg` — Scatter plot of predictions vs actual values
