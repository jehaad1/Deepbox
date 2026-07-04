# Statistical Inference Playbook

> **View online:** https://deepbox.dev/examples/48-statistical-inference-playbook

A focused walkthrough for the v1.0.0 inference APIs that were still missing from the runnable docs: confidence intervals, bootstrap resampling, Gaussian KDE, multiple-comparison correction, and statistical power planning.

## Deepbox Modules Used

| Module           | Features Used |
| ---------------- | ------------- |
| `deepbox/stats`  | `meanConfidenceInterval`, `meanConfidenceIntervalZ`, `meanDiffConfidenceInterval`, `proportionConfidenceInterval`, `bootstrap`, `gaussian_kde`, `benjaminiHochberg`, `bonferroni`, `cohenD`, `tTestPower`, `ttest_ind` |
| `deepbox/plot`   | `figure`, `kdeplot`, `groupedBar`, `axhline`, `legend`, `saveFig` |
| `deepbox/ndarray`| `tensor` |

## Usage

```bash
npm run example:48
```

## Output

- Console walkthrough of interval estimation, uplift resampling, correction, and power planning
- `output/revenue-density.svg`
- `output/multiple-comparisons.svg`

## Architecture

```text
48-statistical-inference-playbook/
├── index.ts
├── README.md
└── output/
```
