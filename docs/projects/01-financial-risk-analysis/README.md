# Financial Portfolio Risk Analysis System

> **View online:** https://deepbox.dev/projects/01-financial-risk-analysis

A production-grade financial risk analysis system built with Deepbox, demonstrating comprehensive use of statistics, linear algebra, and DataFrame operations.

## Features

- **Portfolio Construction**: Build diversified portfolios from asset data
- **Risk Metrics**: Calculate VaR (Value at Risk), CVaR, Sharpe Ratio, Sortino Ratio
- **Correlation Analysis**: Asset correlation matrices and heatmaps
- **Optimization**: Mean-Variance optimization using eigenvalue decomposition
- **Statistical Tests**: Normality tests, stationarity analysis
- **Monte Carlo Simulation**: Portfolio return simulations

## Deepbox Modules Used

| Module              | Features Used                              |
| ------------------- | ------------------------------------------ |
| `deepbox/ndarray`   | Tensor operations, matrix math             |
| `deepbox/linalg`    | SVD, eigenvalues, matrix inverse, solve    |
| `deepbox/stats`     | Correlation, covariance, statistical tests |
| `deepbox/dataframe` | Data manipulation, grouping                |
| `deepbox/random`    | Monte Carlo simulations                    |
| `deepbox/plot`      | Visualization                              |

## Usage

```bash
npm run project:01
```

## Output

- Risk metrics report
- Correlation heatmap (SVG)
- Efficient frontier plot (SVG)
- Portfolio allocation recommendations

## Architecture

```
01-financial-risk-analysis/
├── index.ts              # Main entry point
├── README.md             # This file
└── src/
    ├── portfolio.ts      # Portfolio class and operations
    ├── risk-metrics.ts   # VaR, CVaR, Sharpe calculations
    ├── optimization.ts   # Mean-variance optimization
    └── monte-carlo.ts    # Monte Carlo simulations
```
