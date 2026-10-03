# Linear Regression

> **View online:** https://deepbox.dev/examples/07-linear-regression

Fit a linear regression to noisy synthetic data (`y = 2x + 3`) and measure it on a held-out test set.

## Deepbox Modules Used

| Module               | Features Used                               |
| -------------------- | ------------------------------------------- |
| `deepbox/ml`         | `LinearRegression`                          |
| `deepbox/ndarray`    | `arange`, chained `div`, `mul`, `add`, `reshape` |
| `deepbox/random`     | `setSeed`, `rand`                           |
| `deepbox/metrics`    | `r2Score`, `mse`, `mae`                     |
| `deepbox/preprocess` | `trainTestSplit`                            |

## What It Shows

- The data is built with chained tensor methods instead of loops.
- `setSeed(42)` makes the noise, and so the printed numbers, the same on every run.
- `coef` and `intercept` are close to 2 and 3, the values used to generate the data.
- `r2Score`, `mse` and `mae` return plain numbers.

## Usage

```bash
npm run example:07
```

## Output

Console output only: coefficient, intercept and the three test metrics.

## Files

```
07-linear-regression/
├── index.ts     # Main entry point
└── README.md    # This file
```
