# Time Series Stock Price Forecasting

> **View online:** https://deepbox.dev/projects/04-stock-price-forecasting

Forecasts next-day returns on 500 days of synthetic prices with lagged returns and technical indicators as features. Linear regression and ridge regression are compared with a mean baseline.

## Features

- Synthetic prices with a small upward trend, mean reversion to 100 and random noise
- Indicators: 20 and 50 day moving averages, 20 day volatility, 14 day RSI, 10 day momentum
- Time-ordered train and test split, with no shuffling
- Metrics: MSE, RMSE, MAE, R2 and directional accuracy
- Autocorrelation of returns with `pearsonr`

## What to expect

The returns are close to random noise, so no model has much to find. R2 stays near zero, the models barely beat the mean baseline, and directional accuracy stays near 50%. The project shows the workflow. It does not show a profitable strategy.

## Deepbox Modules Used

| Module               | Features Used                                    |
| -------------------- | ------------------------------------------------ |
| `deepbox/ndarray`    | `tensor`, `toArray`, `item`                      |
| `deepbox/stats`      | `mean`, `std`, `pearsonr`                        |
| `deepbox/ml`         | `LinearRegression`, `Ridge`                      |
| `deepbox/preprocess` | `StandardScaler`                                 |
| `deepbox/metrics`    | `mse`, `rmse`, `mae`, `r2Score`                  |
| `deepbox/dataframe`  | `DataFrame` for the console tables               |
| `deepbox/plot`       | `Figure`, price line chart and returns histogram |

## Usage

```bash
npm run project:04
```

## Output

- Return statistics, indicator values and the model comparison table on the console
- `output/price-chart.svg`
- `output/returns-distribution.svg`
