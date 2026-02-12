# Ridge & Lasso Regression

Compare L1 (Lasso) and L2 (Ridge) regularization techniques. Learn when to use each regularization method.

## Deepbox Modules Used

| Module               | Features Used                  |
| -------------------- | ------------------------------ |
| `deepbox/datasets`   | loadDiabetes                   |
| `deepbox/ml`         | LinearRegression, Ridge, Lasso |
| `deepbox/metrics`    | r2Score, mse                   |
| `deepbox/preprocess` | trainTestSplit, StandardScaler |

## Usage

```bash
npm run example:09
```

## Output

- Console output comparing R² and MSE across Linear, Ridge, and Lasso regression with varying alpha values
