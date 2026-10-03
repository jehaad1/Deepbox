/**
 * Statistics: descriptive measures, correlation, confidence intervals, probability
 * distributions, hypothesis tests, kernel density estimation, multiple comparison
 * corrections and power analysis.
 *
 * @module stats
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox documentation}
 */

// Confidence Intervals
export type { ConfidenceInterval } from "./confidence";
export {
  meanConfidenceInterval,
  meanConfidenceIntervalZ,
  meanDiffConfidenceInterval,
  proportionConfidenceInterval,
} from "./confidence";
// Correlation
export {
  corrcoef,
  cov,
  kendalltau,
  partialcorr,
  pearsonr,
  pointbiserialr,
  spearmanr,
} from "./correlation";
// Descriptive
export type {
  KurtosisOptions,
  MeanOptions,
  SkewnessOptions,
  VarianceOptions,
} from "./descriptive";
export {
  bootstrap,
  cohenD,
  geometricMean,
  harmonicMean,
  iqr,
  kurtosis,
  mean,
  median,
  mode,
  moment,
  percentile,
  quantile,
  sem,
  skewness,
  std,
  trimMean,
  variance,
  zscore,
} from "./descriptive";
// Distributions
export type {
  ContinuousDistribution,
  DiscreteDistribution,
} from "./distributions";
export {
  beta,
  binom,
  cauchy,
  chi2,
  expon,
  f,
  gamma,
  geom,
  hypergeom,
  laplace,
  lognorm,
  nbinom,
  norm,
  pareto,
  poisson,
  t,
  uniform,
  weibull,
} from "./distributions";
// Kernel Density Estimation
export type { BandwidthMethod, GaussianKDEOptions } from "./kde";
export { GaussianKDE, gaussian_kde, gaussianKde } from "./kde";

// Multiple Comparison Corrections
export type { MultipleComparisonResult } from "./multiple";
export {
  benjaminiHochberg,
  benjaminiYekutieli,
  bonferroni,
  hochberg,
  holm,
  sidak,
} from "./multiple";

// Power Analysis
export type { PowerAnalysisResult, TTestPowerOptions } from "./power";
export { tTestPower } from "./power";
// Tests
export type {
  AndersonResult,
  ContingencyResult,
  KsTestOptions,
  MedianTestOptions,
  RankTestOptions,
  TestAlternative,
  TestResult,
  TTestIndOptions,
  TwoWayAnovaResult,
  VarianceTestCenter,
  VarianceTestOptions,
  WilcoxonOptions,
} from "./tests";
export {
  anderson,
  bartlett,
  chi2_contingency,
  chi2Contingency,
  chisquare,
  f_oneway,
  f_twoway,
  fisher_exact,
  fisherExact,
  fligner,
  fOneway,
  friedmanchisquare,
  fTwoway,
  kruskal,
  ks_2samp,
  ks2samp,
  kstest,
  levene,
  lilliefors,
  mannwhitneyu,
  median_test,
  medianTest,
  normaltest,
  runs_test,
  runsTest,
  shapiro,
  ttest_1samp,
  ttest_ind,
  ttest_rel,
  ttest1samp,
  ttestInd,
  ttestRel,
  wilcoxon,
} from "./tests";
