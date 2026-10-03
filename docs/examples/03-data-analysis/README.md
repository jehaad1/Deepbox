# Data Analysis & Visualization

> **View online:** https://deepbox.dev/examples/03-data-analysis

An exploratory workflow on a 20-row employee table: summarize it, compute statistics, group and filter rows, and draw four charts.

## Deepbox Modules Used

| Module              | Features Used                                                                |
| ------------------- | ---------------------------------------------------------------------------- |
| `deepbox/dataframe` | `DataFrame`, `groupBy`, `agg`, `filter`, `select`, `head`                    |
| `deepbox/ndarray`   | `tensor`, `item`                                                             |
| `deepbox/stats`     | `mean`, `std`, `corrcoef`                                                    |
| `deepbox/plot`      | `Figure`, `scatter`, `hist`, `bar`, `heatmap`, `renderSVG`                   |

## What It Shows

- Pull DataFrame columns out as arrays with `get(name).toArray()` and turn them into tensors.
- `mean(t).item()` and `std(t).item()` return plain numbers.
- `groupBy("department").agg({ salary: "mean", experience: "mean" })` gives one row per department. Groups appear in order of first occurrence.
- `corrcoef` expects rows to be observations and columns to be variables.
- The closing summary is computed from the data, not typed in.

## Usage

```bash
npm run example:03
```

## Output

Four SVG files are written to `output/`:

- `salary-vs-experience.svg`: scatter plot
- `salary-distribution.svg`: histogram
- `dept-salaries.svg`: bar chart
- `correlation-heatmap.svg`: heatmap of the correlation matrix

## Files

```
03-data-analysis/
├── index.ts     # Main entry point
├── README.md    # This file
└── output/      # Generated SVG charts
```
