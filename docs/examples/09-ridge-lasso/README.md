# Ridge & Lasso Regression

> **View online:** https://deepbox.dev/examples/09-ridge-lasso

Compare L2 (Ridge) and L1 (Lasso) regularization against plain linear regression on the diabetes dataset.

## Deepbox Modules Used

| Module               | Features Used                      |
| -------------------- | ---------------------------------- |
| `deepbox/datasets`   | `loadDiabetes`                     |
| `deepbox/ml`         | `LinearRegression`, `Ridge`, `Lasso` |
| `deepbox/metrics`    | `r2Score`, `mse`                   |
| `deepbox/preprocess` | `trainTestSplit`, `StandardScaler` |

## What It Shows

- Ridge penalizes the sum of squared coefficients. It shrinks every coefficient but rarely makes one exactly zero.
- Lasso penalizes the sum of absolute values. With a large enough `alpha` it sets some coefficients exactly to zero, which drops those features.
- The script prints the number of zero coefficients per model, computed with `model.coef.eq(0).sum().item()`.
- Features are standardized first, because both penalties depend on the scale of the coefficients.
- On this dataset the regularized models do not beat plain linear regression on the test split. Regularization helps most when there are many features or little data.

## Usage

```bash
npm run example:09
```

## Output

Console output only: one line per model with test R², test MSE and the zero-coefficient count.

## Files

```
09-ridge-lasso/
├── index.ts     # Main entry point
└── README.md    # This file
```
