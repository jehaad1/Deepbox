# Model Evaluation Metrics

> **View online:** https://deepbox.dev/examples/24-metrics

Metrics for classification, regression and clustering, including the multiclass and probability-based metrics.

## Deepbox Modules Used

| Module            | Features Used                                                                                                                            |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/ndarray` | `tensor`                                                                                                                                 |
| `deepbox/metrics` | `accuracy`, `precision`, `recall`, `f1Score`, `jaccardScore`, `matthewsCorrcoef`, `confusionMatrix`, `rocAucScore`, `logLoss`, `r2Score`, `mse`, `rmse`, `mae`, `mape`, `meanAbsolutePercentageError`, `silhouetteScore` |

## What It Shows

- Metrics return plain numbers.
- `precision`, `recall`, `f1Score` and `jaccardScore` take `{ average: "macro" | "micro" | "weighted" }`. `{ average: null }` returns one value per class. Other options include `labels`, `zeroDivision` and `sampleWeight`.
- `rocAucScore` and `logLoss` take probabilities. For more than two classes pass an `[n, classes]` matrix. `rocAucScore` scores one-vs-rest by default (`multiClass: "ovr"`) and macro-averages the classes. `multiClass: "ovo"` is also available.
- `confusionMatrix` has true classes as rows and predicted classes as columns.
- `mape` returns a percentage, unlike scikit-learn, which returns a fraction. `meanAbsolutePercentageError` returns the fraction and accepts `sampleWeight`.
- `silhouetteScore(X, labels)` ranges from -1 to 1. Higher means better separated clusters.

## Usage

```bash
npm run example:24
```

## Output

Console output only.

## Files

```
24-metrics/
├── index.ts     # Main entry point
└── README.md    # This file
```
