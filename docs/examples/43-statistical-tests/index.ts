/**
 * Example 43: Statistical Distributions & Hypothesis Tests
 *
 * New in v1.0.0: Full stats module with probability distributions (normal, t,
 * chi2, F, binomial, Poisson, etc.), hypothesis tests (t-test, chi-square,
 * KS, Shapiro-Wilk, ANOVA), correlations, and confidence intervals.
 */

import { tensor } from "deepbox/ndarray";
import {
  beta,
  binom,
  chi2,
  chisquare,
  expon,
  f_oneway,
  kendalltau,
  kstest,
  norm,
  pearsonr,
  poisson,
  shapiro,
  spearmanr,
  t,
  ttest_1samp,
  ttest_ind,
  ttest_rel,
  uniform,
} from "deepbox/stats";

console.log("=".repeat(60));
console.log("Example 43: Statistical Distributions & Hypothesis Tests");
console.log("=".repeat(60));

// ============================================================================
// Part 1: Continuous Distributions
// ============================================================================
console.log("\n📊 Part 1: Continuous Distributions");
console.log("-".repeat(60));

// Normal distribution
const normal = norm(0, 1); // mean=0, std=1
console.log("Normal(0, 1):");
console.log(`  PDF at x=0:   ${normal.pdf(0).toFixed(6)}`);
console.log(`  CDF at x=0:   ${normal.cdf(0).toFixed(6)}`);
console.log(`  PPF at p=0.975: ${normal.ppf(0.975).toFixed(6)} (z-critical)`);
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
console.log("\n🎲 Part 2: Discrete Distributions");
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
console.log("\n🧪 Part 3: T-Tests");
console.log("-".repeat(60));

// One-sample t-test: is the mean significantly different from 0?
const sample1 = tensor([2.1, 2.5, 2.3, 2.8, 2.2, 2.6, 2.4, 2.7, 2.3, 2.5]);
const onesamp = ttest_1samp(sample1, 0);
console.log("One-sample t-test (H0: mean = 0):");
console.log(`  t-statistic: ${onesamp.statistic.toFixed(4)}`);
console.log(`  p-value:     ${onesamp.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${onesamp.pvalue < 0.05 ? "Yes" : "No"}`);

// Independent two-sample t-test
const groupA = tensor([5.1, 5.3, 5.0, 5.5, 5.2, 5.4, 5.1, 5.3]);
const groupB = tensor([4.8, 4.6, 4.9, 4.5, 4.7, 4.6, 4.8, 4.5]);
const twosamp = ttest_ind(groupA, groupB);
console.log("\nIndependent two-sample t-test (H0: mean_A = mean_B):");
console.log(`  t-statistic: ${twosamp.statistic.toFixed(4)}`);
console.log(`  p-value:     ${twosamp.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${twosamp.pvalue < 0.05 ? "Yes" : "No"}`);

// Paired t-test
const before = tensor([85, 90, 78, 92, 88, 76, 95, 89]);
const after = tensor([88, 93, 82, 95, 91, 80, 97, 92]);
const paired = ttest_rel(before, after);
console.log("\nPaired t-test (H0: no difference before/after):");
console.log(`  t-statistic: ${paired.statistic.toFixed(4)}`);
console.log(`  p-value:     ${paired.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${paired.pvalue < 0.05 ? "Yes" : "No"}`);

// ============================================================================
// Part 4: Chi-Square Test
// ============================================================================
console.log("\n📐 Part 4: Chi-Square Goodness of Fit");
console.log("-".repeat(60));

// Test if observed frequencies match expected (uniform)
const observed = tensor([18, 22, 20, 15, 25]);
const expected = tensor([20, 20, 20, 20, 20]);
const chiResult = chisquare(observed, expected);
console.log("Chi-square test (H0: observed matches expected):");
console.log(`  Observed: [18, 22, 20, 15, 25]`);
console.log(`  Expected: [20, 20, 20, 20, 20]`);
console.log(`  χ² statistic: ${chiResult.statistic.toFixed(4)}`);
console.log(`  p-value:      ${chiResult.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${chiResult.pvalue < 0.05 ? "Yes" : "No"}`);

// ============================================================================
// Part 5: Normality Tests
// ============================================================================
console.log("\n📏 Part 5: Normality Tests");
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
console.log("\n📊 Part 6: One-way ANOVA");
console.log("-".repeat(60));

// Test if three groups have the same mean
const group1 = tensor([6.1, 5.8, 6.3, 5.9, 6.0]);
const group2 = tensor([7.2, 7.0, 7.4, 7.1, 7.3]);
const group3 = tensor([5.5, 5.3, 5.7, 5.4, 5.6]);
const anovaResult = f_oneway(group1, group2, group3);
console.log("One-way ANOVA (H0: all group means are equal):");
console.log(`  Groups: [~6.0], [~7.2], [~5.5]`);
console.log(`  F-statistic: ${anovaResult.statistic.toFixed(4)}`);
console.log(`  p-value:     ${anovaResult.pvalue.toFixed(6)}`);
console.log(`  Reject H0 at α=0.05: ${anovaResult.pvalue < 0.05 ? "Yes" : "No"}`);

// ============================================================================
// Part 7: Correlation Analysis
// ============================================================================
console.log("\n🔗 Part 7: Correlation Analysis");
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

// Kendall's tau (ordinal association)
const [tauKendall, pKendall] = kendalltau(x, y);
console.log("\nKendall's tau (ordinal association):");
console.log(`  τ = ${tauKendall.toFixed(4)}, p-value = ${pKendall.toFixed(6)}`);

// Non-linear relationship example
const xNonlin = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const yNonlin = tensor([1, 4, 9, 16, 25, 36, 49, 64, 81, 100]); // y = x²
const [rLin] = pearsonr(xNonlin, yNonlin);
const [rMono] = spearmanr(xNonlin, yNonlin);
console.log(
  `\nNon-linear (y=x²): Pearson r = ${rLin.toFixed(4)}, Spearman ρ = ${rMono.toFixed(4)}`
);
console.log("  (Spearman captures monotonic relationship better)");

// ============================================================================
// Part 8: Distribution Sampling & Quantiles
// ============================================================================
console.log("\n🎯 Part 8: Distribution Quantiles & Summary");
console.log("-".repeat(60));

const stdNorm = norm(0, 1);
console.log("Standard Normal — Key Quantiles:");
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
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• Continuous distributions: norm, t, chi2, F, expon, beta, uniform, gamma, etc.");
console.log("• Discrete distributions: binom, poisson, geom, hypergeom, nbinom");
console.log("• Each distribution has: pdf/pmf, cdf, ppf (quantile), mean, variance");
console.log("• T-tests: one-sample, independent, paired — compare means");
console.log("• Chi-square: goodness of fit and contingency tables");
console.log("• Normality: Shapiro-Wilk, KS test, Anderson-Darling");
console.log("• ANOVA: compare means across 3+ groups");
console.log("• Correlations: Pearson (linear), Spearman (monotonic), Kendall (ordinal)");

console.log("\n✅ Statistical Distributions & Tests Example Complete!");
console.log("=".repeat(60));
