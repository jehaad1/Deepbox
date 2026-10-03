/**
 * Example 43: Statistical Distributions & Hypothesis Tests
 *
 * Probability distributions (normal, t, chi-square, exponential, beta, uniform,
 * binomial, Poisson), hypothesis tests (t-tests, chi-square, Shapiro-Wilk, KS, ANOVA)
 * and correlations (Pearson, Spearman, Kendall).
 */

import { tensor } from "deepbox/ndarray";
import {
  beta,
  binom,
  chi2,
  chisquare,
  expon,
  fOneway,
  kendalltau,
  kstest,
  norm,
  pearsonr,
  poisson,
  shapiro,
  spearmanr,
  t,
  ttest1samp,
  ttestInd,
  ttestRel,
  uniform,
} from "deepbox/stats";

console.log("=".repeat(60));
console.log("Example 43: Statistical Distributions & Hypothesis Tests");
console.log("=".repeat(60));

// ============================================================================
// Part 1: Continuous Distributions
// ============================================================================
console.log("\nPart 1: Continuous Distributions");
console.log("-".repeat(60));

// Normal distribution: norm(mean, standardDeviation)
const normal = norm(0, 1);
console.log("Normal(0, 1):");
console.log(`  PDF at x=0:   ${normal.pdf(0).toFixed(6)}`);
console.log(`  CDF at x=0:   ${normal.cdf(0).toFixed(6)}`);
console.log(`  PPF at p=0.975: ${normal.ppf(0.975).toFixed(6)} (z critical value)`);
console.log(`  Mean:  ${normal.mean().toFixed(4)}, Var: ${normal.variance().toFixed(4)}`);

// Student's t distribution
const tDist = t(10); // 10 degrees of freedom
console.log("\nStudent's t(df=10):");
console.log(`  PDF at x=0:   ${tDist.pdf(0).toFixed(6)}`);
console.log(`  CDF at x=2:   ${tDist.cdf(2).toFixed(6)}`);
console.log(`  PPF at p=0.975: ${tDist.ppf(0.975).toFixed(6)}`);

// Chi-squared distribution
const chi2Dist = chi2(5); // 5 degrees of freedom
console.log("\nChi-squared(df=5):");
console.log(`  PDF at x=5:   ${chi2Dist.pdf(5).toFixed(6)}`);
console.log(`  CDF at x=5:   ${chi2Dist.cdf(5).toFixed(6)}`);
console.log(`  Mean: ${chi2Dist.mean().toFixed(4)}`);

// Exponential distribution
const expDist = expon(0.5); // rate=0.5
console.log("\nExponential(rate=0.5):");
console.log(`  PDF at x=1:   ${expDist.pdf(1).toFixed(6)}`);
console.log(`  CDF at x=2:   ${expDist.cdf(2).toFixed(6)}`);
console.log(`  Mean: ${expDist.mean().toFixed(4)}`);

// Beta distribution
const betaDist = beta(2, 5);
console.log("\nBeta(α=2, β=5):");
console.log(`  PDF at x=0.3: ${betaDist.pdf(0.3).toFixed(6)}`);
console.log(`  Mean: ${betaDist.mean().toFixed(4)}`);

// Uniform distribution
const uniformDist = uniform(0, 10);
console.log("\nUniform(0, 10):");
console.log(`  PDF at x=5:   ${uniformDist.pdf(5).toFixed(6)}`);
console.log(`  Mean: ${uniformDist.mean().toFixed(4)}`);

// ============================================================================
// Part 2: Discrete Distributions
// ============================================================================
console.log("\nPart 2: Discrete Distributions");
console.log("-".repeat(60));

// Binomial distribution
const binDist = binom(20, 0.3); // 20 trials, p=0.3
console.log("Binomial(n=20, p=0.3):");
console.log(`  PMF at k=6:   ${binDist.pmf(6).toFixed(6)}`);
console.log(`  CDF at k=6:   ${binDist.cdf(6).toFixed(6)}`);
console.log(`  Mean: ${binDist.mean().toFixed(4)}, Var: ${binDist.variance().toFixed(4)}`);

// Poisson distribution
const poisDist = poisson(5); // lambda=5
console.log("\nPoisson(λ=5):");
console.log(`  PMF at k=5:   ${poisDist.pmf(5).toFixed(6)}`);
console.log(`  PMF at k=3:   ${poisDist.pmf(3).toFixed(6)}`);
console.log(`  CDF at k=7:   ${poisDist.cdf(7).toFixed(6)}`);
console.log(`  Mean: ${poisDist.mean().toFixed(4)}, Var: ${poisDist.variance().toFixed(4)}`);

// ============================================================================
// Part 3: T-Tests
// ============================================================================
console.log("\nPart 3: T-Tests");
console.log("-".repeat(60));

// One-sample t-test: is the mean different from 0?
const sample1 = tensor([2.1, 2.5, 2.3, 2.8, 2.2, 2.6, 2.4, 2.7, 2.3, 2.5]);
const onesamp = ttest1samp(sample1, 0);
console.log("One-sample t-test (H0: mean = 0):");
console.log(`  t-statistic: ${onesamp.statistic.toFixed(4)}`);
console.log(`  p-value:     ${onesamp.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${onesamp.pvalue < 0.05 ? "Yes" : "No"}`);

// Independent two-sample t-test
const groupA = tensor([5.1, 5.3, 5.0, 5.5, 5.2, 5.4, 5.1, 5.3]);
const groupB = tensor([4.8, 4.6, 4.9, 4.5, 4.7, 4.6, 4.8, 4.5]);
const twosamp = ttestInd(groupA, groupB);
console.log("\nIndependent two-sample t-test (H0: mean_A = mean_B):");
console.log(`  t-statistic: ${twosamp.statistic.toFixed(4)}`);
console.log(`  p-value:     ${twosamp.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${twosamp.pvalue < 0.05 ? "Yes" : "No"}`);

// Paired t-test
const before = tensor([85, 90, 78, 92, 88, 76, 95, 89]);
const after = tensor([88, 93, 82, 95, 91, 80, 97, 92]);
const paired = ttestRel(before, after);
console.log("\nPaired t-test (H0: no difference before/after):");
console.log(`  t-statistic: ${paired.statistic.toFixed(4)}`);
console.log(`  p-value:     ${paired.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${paired.pvalue < 0.05 ? "Yes" : "No"}`);

// ============================================================================
// Part 4: Chi-Square Test
// ============================================================================
console.log("\nPart 4: Chi-Square Goodness of Fit");
console.log("-".repeat(60));

// Test if observed frequencies match expected (uniform)
const observed = tensor([18, 22, 20, 15, 25]);
const expected = tensor([20, 20, 20, 20, 20]);
const chiResult = chisquare(observed, expected);
console.log("Chi-square test (H0: observed matches expected):");
console.log("  Observed: [18, 22, 20, 15, 25]");
console.log("  Expected: [20, 20, 20, 20, 20]");
console.log(`  χ² statistic: ${chiResult.statistic.toFixed(4)}`);
console.log(`  p-value:      ${chiResult.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${chiResult.pvalue < 0.05 ? "Yes" : "No"}`);

// ============================================================================
// Part 5: Normality Tests
// ============================================================================
console.log("\nPart 5: Normality Tests");
console.log("-".repeat(60));

// Shapiro-Wilk test for normality
const normalSample = tensor([
  -0.2, 0.5, 1.1, -0.8, 0.3, -0.1, 0.7, -0.4, 0.9, -0.6, 0.2, 0.8, -0.3, 0.4, -0.7, 0.1, 0.6, -0.5,
  0.0, 0.3,
]);
const shapResult = shapiro(normalSample);
console.log("Shapiro-Wilk test (H0: data is normally distributed):");
console.log(`  W-statistic: ${shapResult.statistic.toFixed(4)}`);
console.log(`  p-value:     ${shapResult.pvalue.toFixed(6)}`);
console.log(
  `  Normal at α=0.05: ${shapResult.pvalue > 0.05 ? "Yes (fail to reject)" : "No (rejected)"}`
);

// KS test against normal distribution
const ksResult = kstest(normalSample, "norm");
console.log("\nKolmogorov-Smirnov test (H0: data follows standard normal):");
console.log(`  KS statistic: ${ksResult.statistic.toFixed(4)}`);
console.log(`  p-value:      ${ksResult.pvalue.toFixed(6)}`);

// ============================================================================
// Part 6: ANOVA (One-way)
// ============================================================================
console.log("\nPart 6: One-way ANOVA");
console.log("-".repeat(60));

// Test if three groups have the same mean
const group1 = tensor([6.1, 5.8, 6.3, 5.9, 6.0]);
const group2 = tensor([7.2, 7.0, 7.4, 7.1, 7.3]);
const group3 = tensor([5.5, 5.3, 5.7, 5.4, 5.6]);
const anovaResult = fOneway(group1, group2, group3);
console.log("One-way ANOVA (H0: all group means are equal):");
console.log("  Group means: about 6.0, 7.2 and 5.5");
console.log(`  F-statistic: ${anovaResult.statistic.toFixed(4)}`);
console.log(`  p-value:     ${anovaResult.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${anovaResult.pvalue < 0.05 ? "Yes" : "No"}`);

// ============================================================================
// Part 7: Correlation Analysis
// ============================================================================
console.log("\nPart 7: Correlation Analysis");
console.log("-".repeat(60));

const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const y = tensor([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.8, 16.1, 18.2, 19.9]);

// Pearson correlation (linear relationship)
const [rPearson, pPearson] = pearsonr(x, y);
console.log("Pearson correlation (linear relationship):");
console.log(`  r = ${rPearson.toFixed(4)}, p-value = ${pPearson.toFixed(6)}`);

// Spearman rank correlation (monotonic relationship)
const [rSpearman, pSpearman] = spearmanr(x, y);
console.log("\nSpearman rank correlation (monotonic relationship):");
console.log(`  ρ = ${rSpearman.toFixed(4)}, p-value = ${pSpearman.toFixed(6)}`);

// Kendall's tau (ordinal association). The p-value is exact for small samples without ties.
const [tauKendall, pKendall] = kendalltau(x, y);
console.log("\nKendall's tau (ordinal association):");
console.log(`  τ = ${tauKendall.toFixed(4)}, p-value = ${pKendall.toFixed(6)}`);

// The correlation tests take an alternative: "two-sided" (default), "less" or "greater".
// kendalltau also takes variant ("b" or "c") and method ("auto", "exact" or "asymptotic").
const [, pGreater] = pearsonr(x, y, { alternative: "greater" });
console.log(`\nPearson, alternative "greater" (H1: r > 0): p-value = ${pGreater.toExponential(2)}`);
const [tauAsym, pAsym] = kendalltau(x, y, { method: "asymptotic" });
console.log(
  `Kendall, asymptotic p-value: τ = ${tauAsym.toFixed(4)}, p-value = ${pAsym.toFixed(6)}`
);

// Non-linear relationship example
const xNonlin = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const yNonlin = tensor([1, 4, 9, 16, 25, 36, 49, 64, 81, 100]); // y = x²
const [rLin] = pearsonr(xNonlin, yNonlin);
const [rMono] = spearmanr(xNonlin, yNonlin);
console.log(
  `\nNon-linear (y=x²): Pearson r = ${rLin.toFixed(4)}, Spearman ρ = ${rMono.toFixed(4)}`
);
console.log("  (Spearman is 1 for any monotonic relationship, Pearson is lower for a curve)");

// ============================================================================
// Part 8: Distribution Sampling & Quantiles
// ============================================================================
console.log("\nPart 8: Distribution Quantiles");
console.log("-".repeat(60));

const stdNorm = norm(0, 1);
console.log("Standard normal quantiles:");
for (const p of [0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99]) {
  console.log(`  P(X < ${stdNorm.ppf(p).toFixed(4).padStart(7)}) = ${p}`);
}

console.log("\n68-95-99.7 Rule Verification:");
console.log(`  P(-1 < X < 1) = ${(stdNorm.cdf(1) - stdNorm.cdf(-1)).toFixed(4)} (expect ~0.6827)`);
console.log(`  P(-2 < X < 2) = ${(stdNorm.cdf(2) - stdNorm.cdf(-2)).toFixed(4)} (expect ~0.9545)`);
console.log(`  P(-3 < X < 3) = ${(stdNorm.cdf(3) - stdNorm.cdf(-3)).toFixed(4)} (expect ~0.9973)`);

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• Continuous distributions: norm, t, chi2, f, expon, beta, uniform, gamma and more");
console.log("• Discrete distributions: binom, poisson, geom, hypergeom, nbinom");
console.log("• Each distribution has pdf or pmf, cdf, ppf (quantile), mean and variance");
console.log("• T-tests (ttest1samp, ttestInd, ttestRel): compare means");
console.log("• chisquare, chi2Contingency: goodness of fit and contingency tables");
console.log("• Normality: shapiro, kstest, anderson");
console.log("• fOneway: compare the means of three or more groups");
console.log("• Correlations: pearsonr (linear), spearmanr (monotonic), kendalltau (ordinal)");
console.log("• Correlation tests accept alternative: two-sided, less or greater");

console.log("\nStatistical Distributions & Tests Example Complete!");
console.log("=".repeat(60));
