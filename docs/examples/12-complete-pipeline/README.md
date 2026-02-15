# Complete ML Pipeline

> **View online:** https://deepbox.dev/examples/12-complete-pipeline

Bring everything together in a comprehensive machine learning workflow. From data loading to model evaluation and visualization.

## Deepbox Modules Used

| Module               | Features Used                  |
| -------------------- | ------------------------------ |
| `deepbox/datasets`   | loadHousingMini                |
| `deepbox/ml`         | Ridge                          |
| `deepbox/metrics`    | r2Score, mse, mae              |
| `deepbox/preprocess` | trainTestSplit, StandardScaler |
| `deepbox/stats`      | mean, std                      |
| `deepbox/plot`       | Figure, scatter, plot          |

## Usage

```bash
npm run example:12
```

## Output

- 1 SVG visualization in `output/`:
  - `predictions.svg` — Predictions vs actual values scatter plot
