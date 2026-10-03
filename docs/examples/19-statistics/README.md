# Statistics

> **View online:** https://deepbox.dev/examples/19-statistics

Descriptive statistics, percentiles and correlation on small samples. For hypothesis tests see example 43.

## Deepbox Modules Used

| Module            | Features Used                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------ |
| `deepbox/ndarray` | `tensor`, `stack`                                                                          |
| `deepbox/stats`   | `mean`, `median`, `std`, `variance`, `skewness`, `kurtosis`, `percentile`, `pearsonr`, `corrcoef` |

## What It Shows

- The statistics functions return tensors. `item()` reads a one-element result as a number.
- `std` and `variance` divide by `n` by default, like NumPy. Pass `{ ddof: 1 }` for the sample value, like pandas.
- `kurtosis` returns excess kurtosis by default, so a normal distribution gives 0.
- `percentile(data, [25, 50, 75])` returns several percentiles in one tensor.
- `pearsonr(x, y)` returns `[r, p]`. The `alternative` option (`"two-sided"`, `"less"`, `"greater"`) selects the test direction.
- `corrcoef` on a matrix treats rows as observations and columns as variables. `stack([x, y, z], 1)` builds such a matrix from three vectors.

## Usage

```bash
npm run example:19
```

## Output

Console output only.

## Files

```
19-statistics/
├── index.ts     # Main entry point
└── README.md    # This file
```
