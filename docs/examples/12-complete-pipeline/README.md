# Complete ML Pipeline

> **View online:** https://deepbox.dev/examples/12-complete-pipeline

A regression workflow from start to finish on the Housing-Mini dataset: load, explore, split, scale, fit a Ridge model, evaluate and plot.

## Deepbox Modules Used

| Module               | Features Used                      |
| -------------------- | ---------------------------------- |
| `deepbox/datasets`   | `loadHousingMini`                  |
| `deepbox/ml`         | `Ridge`                            |
| `deepbox/metrics`    | `r2Score`, `mse`, `mae`            |
| `deepbox/preprocess` | `trainTestSplit`, `StandardScaler` |
| `deepbox/stats`      | `mean`, `std`                      |
| `deepbox/plot`       | `Figure`, `scatter`, `plot`, `renderSVG` |

## What It Shows

- `dataset.data.slice({}, 0)` selects the first feature column: `{}` keeps every row, `0` picks column 0.
- The scaler is fitted on the training split only, which avoids leaking test information.
- Metrics are plain numbers. Results of `mean(...)` are tensors, read with `item()`.
- The plot receives tensors directly. The red line marks perfect predictions, drawn from the minimum to the maximum of the data.

## Usage

```bash
npm run example:12
```

## Output

One SVG file is written to `output/`: `predictions.svg`, a scatter plot of predicted against actual values.

## Files

```
12-complete-pipeline/
├── index.ts     # Main entry point
├── README.md    # This file
└── output/      # Generated SVG chart
```
