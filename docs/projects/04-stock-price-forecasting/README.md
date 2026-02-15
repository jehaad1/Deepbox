# Time Series Stock Price Forecasting

> **View online:** https://deepbox.dev/projects/04-stock-price-forecasting

A time series forecasting project for stock prices using synthetic market data, feature engineering, and regression baselines.

## Features

- **Data Generation**: Synthetic stock price data with realistic patterns
- **Feature Engineering**: Technical indicators (MA, RSI, volatility)
- **Statistical Analysis**: Return distribution and correlation analysis
- **Forecasting Models**: Linear Regression and Ridge Regression baselines
- **Evaluation**: RMSE, MAE, directional accuracy

## Deepbox Modules Used

| Module              | Features Used                             |
| ------------------- | ----------------------------------------- |
| `deepbox/ndarray`   | Tensor operations, time series processing |
| `deepbox/stats`     | Statistical tests, correlation analysis   |
| `deepbox/ml`        | LinearRegression, Ridge for baseline      |
| `deepbox/metrics`   | mse, rmse, mae, r2Score                   |
| `deepbox/dataframe` | Data manipulation                         |
| `deepbox/plot`      | Time series visualization                 |

## Usage

```bash
npm run project:04
```
