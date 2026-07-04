# Experimentation Platform

> **View online:** https://deepbox.dev/projects/09-experimentation-platform

A production-style experimentation and rollout analysis workflow built around the v1.0.0 inference surface: confidence intervals, bootstrap uplift analysis, multiple-comparison correction, KDE diagnostics, and power planning.

## Features

- **Synthetic experiment traffic** across variants, devices, segments, and regions
- **Operational scorecards** with `DataFrame` grouping for conversion, retention, revenue, and latency
- **Inference** via confidence intervals, pairwise tests, and Benjamini-Hochberg correction
- **Bootstrap uplift estimation** for winner-vs-control decision support
- **Power analysis** to plan the next confirmatory experiment
- **Artifacts** with grouped rate charts, KDE comparisons, and JSON decision reports

## Deepbox Modules Used

| Module               | Features Used |
| -------------------- | ------------- |
| `deepbox/dataframe`  | `DataFrame`, `groupBy` |
| `deepbox/stats`      | `meanConfidenceInterval`, `meanConfidenceIntervalZ`, `meanDiffConfidenceInterval`, `proportionConfidenceInterval`, `bootstrap`, `cohenD`, `tTestPower`, `ttest_ind`, `benjaminiHochberg` |
| `deepbox/plot`       | `figure`, `groupedBar`, `kdeplot`, `axhline`, `legend`, `saveFig` |
| `deepbox/random`     | `Generator` |
| `deepbox/ndarray`    | `tensor` |

## Usage

```bash
npm run project:09
```

## Output

- `output/variant-scorecard.json`
- `output/decision-report.json`
- `output/variant-rates.svg`
- `output/winner-order-value-density.svg`
- `output/revenue-significance.svg`

## Architecture

```text
09-experimentation-platform/
├── index.ts
├── README.md
└── output/
```
