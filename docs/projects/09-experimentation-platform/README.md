# Experimentation Platform

> **View online:** https://deepbox.dev/projects/09-experimentation-platform

Analyzes a synthetic checkout experiment with three variants (control, streamlined checkout, smart bundle) and 3,600 sessions. It builds scorecards, tests each variant against control, estimates uplift with a bootstrap and plans the sample size for a follow-up test.

## Features

- Synthetic sessions across variants, devices, customer segments and regions
- Scorecards built with `DataFrame.groupBy` for conversion, retention, revenue and latency
- Confidence intervals for proportions and means
- Pairwise t-tests of revenue and latency against control, with Benjamini-Hochberg correction
- Bootstrap of the revenue uplift of the winning variant over control. The resampled values are the uplifts in each segment and device cell, so the interval reflects how much the uplift varies across cells
- Power analysis: current power and the sample size per arm needed for 90% power
- Charts: grouped rates, order-value densities (`kdeplot`) and raw vs corrected p-values

## Deepbox Modules Used

| Module              | Features Used                                                                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/dataframe` | `DataFrame`, `groupBy`                                                                                                                                                                  |
| `deepbox/stats`     | `meanConfidenceInterval`, `meanConfidenceIntervalZ`, `meanDiffConfidenceInterval`, `proportionConfidenceInterval`, `bootstrap`, `cohenD`, `tTestPower`, `ttestInd`, `benjaminiHochberg` |
| `deepbox/plot`      | `figure`, `groupedBar`, `kdeplot`, `axhline`, `legend`, `saveFig`                                                                                                                       |
| `deepbox/random`    | `Generator`                                                                                                                                                                             |
| `deepbox/ndarray`   | `tensor`                                                                                                                                                                                |

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
