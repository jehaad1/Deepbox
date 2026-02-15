# Data Analysis & Visualization

> **View online:** https://deepbox.dev/examples/03-data-analysis

Comprehensive data analysis workflow using DataFrames, statistics, and plotting. Explores, analyzes, and visualizes employee data.

## Deepbox Modules Used

| Module              | Features Used                           |
| ------------------- | --------------------------------------- |
| `deepbox/dataframe` | DataFrame, groupBy, agg, filter, select |
| `deepbox/ndarray`   | tensor                                  |
| `deepbox/stats`     | mean, std, corrcoef                     |
| `deepbox/plot`      | Figure, scatter, hist, bar, heatmap     |

## Usage

```bash
npm run example:03
```

## Output

- 4 SVG visualizations in `output/`:
  - `salary-vs-experience.svg` — Scatter plot
  - `salary-distribution.svg` — Histogram
  - `dept-salaries.svg` — Bar chart
  - `correlation-heatmap.svg` — Heatmap

## Architecture

```
03-data-analysis/
├── index.ts     # Main entry point
├── README.md    # This file
└── output/      # Generated SVG visualizations
```
