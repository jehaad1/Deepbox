# Statistical Inference Playbook

> **View online:** https://deepbox.dev/examples/48-statistical-inference-playbook

Applies the inference functions in `deepbox/stats` to a small A/B experiment: confidence intervals, a bootstrap of the mean uplift, a Gaussian kernel density estimate, p-value correction for four metrics, and a power analysis.

## Deepbox Modules Used

| Module            | Features Used                                                                                                                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `deepbox/stats`   | `meanConfidenceInterval`, `meanConfidenceIntervalZ`, `meanDiffConfidenceInterval`, `proportionConfidenceInterval`, `bootstrap`, `gaussianKde`, `benjaminiHochberg`, `benjaminiYekutieli`, `bonferroni`, `hochberg`, `cohenD`, `tTestPower`, `ttestInd` |
| `deepbox/plot`    | `figure`, `kdeplot`, `groupedBar`, `axhline`, `legend`, `saveFig`                                                                                                                                                                                      |
| `deepbox/ndarray` | `tensor`                                                                                                                                                                                                                                               |

## Usage

```bash
npm run example:48
```

## Output

- Console output: confidence intervals, the bootstrap estimate, KDE bandwidths and densities, raw and corrected p-values for four metrics, and the power analysis.
- `output/revenue-density.svg`
- `output/multiple-comparisons.svg`
- The older names `gaussian_kde`, `ttest_ind` and the `bw_method` option (in `gaussianKde` and `kdeplot`) still work. Use `gaussianKde`, `ttestInd` and `bwMethod`.

## Files

```text
48-statistical-inference-playbook/
├── index.ts
├── README.md
└── output/
```
