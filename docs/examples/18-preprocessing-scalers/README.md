# Preprocessing: Scalers

> **View online:** https://deepbox.dev/examples/18-preprocessing-scalers

Seven scalers applied to the same small table. The table has a large outlier in the first and last columns, so you can see how each scaler reacts to it.

## Deepbox Modules Used

| Module               | Features Used                                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `deepbox/ndarray`    | `tensor`, `slice`, `mean`, `square`, `sum`                                                                  |
| `deepbox/preprocess` | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`, `PowerTransformer`, `QuantileTransformer` |

## What It Shows

- `StandardScaler` gives zero mean and unit variance per column. `inverseTransform` undoes it.
- `MinMaxScaler` maps each column to [0, 1]. The outlier squeezes the other rows toward 0.
- `RobustScaler` uses the median and the interquartile range, so the outlier has less influence.
- `MaxAbsScaler` divides by the largest absolute value per column.
- `Normalizer` scales each row, not each column, to unit norm. It needs no `fit`.
- `PowerTransformer` makes data more Gaussian. Its `standardize` option defaults to `false` here, while scikit-learn defaults to `true`. The script shows both.
- `QuantileTransformer` maps values to their rank in the column.
- Fit a scaler on training data only, then reuse it with `transform()`.

## Usage

```bash
npm run example:18
```

## Output

Console output only: the first three scaled rows for each scaler.

## Files

```
18-preprocessing-scalers/
├── index.ts     # Main entry point
└── README.md    # This file
```
