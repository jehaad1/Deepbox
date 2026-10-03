# Financial Portfolio Risk Analysis

> **View online:** https://deepbox.dev/projects/01-financial-risk-analysis

Builds a synthetic eight-asset portfolio and reports its risk: VaR and CVaR, Sharpe and Sortino ratios, drawdown, correlations, optimized weights, a Monte Carlo forecast and a few stress scenarios.

## Features

- Portfolio construction from asset return series
- Risk metrics: historical, parametric and Cornish-Fisher VaR, CVaR, Sharpe, Sortino, Calmar, maximum drawdown
- Correlation and covariance matrices, with a Pearson, Spearman and Shapiro-Wilk check on the returns
- Optimization: minimum variance, maximum Sharpe and risk parity weights, plus an efficient frontier. The first two use the inverse of the covariance matrix (`inv`), the closed form of Markowitz mean-variance optimization
- Monte Carlo simulation of 12-month portfolio returns, using the Cholesky factor of the covariance matrix to correlate the shocks
- Bootstrap confidence interval for the mean return, stress scenarios and a rolling VaR backtest

## Deepbox Modules Used

| Module              | Features Used                                              |
| ------------------- | ---------------------------------------------------------- |
| `deepbox/ndarray`   | `tensor`, `toArray`, `item`                                |
| `deepbox/linalg`    | `inv`, `cholesky`, `det`, `trace`                          |
| `deepbox/stats`     | `cov`, `mean`, `std`, `pearsonr`, `spearmanr`, `shapiro`   |
| `deepbox/dataframe` | `DataFrame` for the console tables                         |
| `deepbox/plot`      | `Figure`, `plot`, `scatter`, `bar`, `heatmap`, `renderSVG` |

The Monte Carlo and bootstrap code uses a small seeded generator in `src/monte-carlo.ts`, so results are the same on every run.

## Usage

```bash
npm run project:01
```

## Output

- Risk report on the console
- `output/efficient-frontier.svg`
- `output/monte-carlo-distribution.svg`
- `output/correlation-heatmap.svg`

## Architecture

```text
01-financial-risk-analysis/
├── index.ts              # Main entry point
├── README.md             # This file
├── output/               # Generated SVGs
└── src/
    ├── portfolio.ts      # Portfolio class and synthetic asset data
    ├── risk-metrics.ts   # VaR, CVaR, Sharpe, drawdown, backtest
    ├── optimization.ts   # Mean-variance, risk parity, efficient frontier
    └── monte-carlo.ts    # Monte Carlo, bootstrap, stress scenarios
```
