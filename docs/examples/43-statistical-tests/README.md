# Statistical Distributions & Hypothesis Tests

> **View online:** https://deepbox.dev/examples/43-statistical-tests

Evaluates probability distributions (`norm`, `t`, `chi2`, `expon`, `beta`, `uniform`, `binom`, `poisson`) and runs hypothesis tests (t-tests, chi-square, Shapiro-Wilk, Kolmogorov-Smirnov, one-way ANOVA) and correlation tests (Pearson, Spearman, Kendall). The correlation tests take an `alternative` option, and `kendalltau` also takes `variant` and `method`.

## Deepbox Modules Used

| Module            | Features Used                                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/stats`   | norm, t, chi2, expon, beta, uniform, binom, poisson, ttest1samp, ttestInd, ttestRel, chisquare, kstest, shapiro, fOneway, pearsonr, spearmanr, kendalltau |
| `deepbox/ndarray` | tensor                                                                                                                                                    |

## Usage

```bash
npm run example:43
```

## Output

- Console output only: density, cumulative and quantile values, test statistics with p-values, and correlation coefficients.
- The older snake_case names (`ttest_1samp`, `ttest_ind`, `ttest_rel`, `f_oneway`) still work. Use the camelCase names.

## Files

```
43-statistical-tests/
├── index.ts     # Example script
└── README.md    # This file
```
