/**
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox documentation}
 */

// Correlation

// Confidence Intervals
export type { ConfidenceInterval } from "./confidence";
export {
  meanConfidenceInterval,
  meanConfidenceIntervalZ,
  meanDiffConfidenceInterval,
  proportionConfidenceInterval,
} from "./confidence";
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
export { GaussianKDE, gaussian_kde } from "./kde";

// Multiple Comparison Corrections
export type { MultipleComparisonResult } from "./multiple";
export { benjaminiHochberg, bonferroni, holm, sidak } from "./multiple";

// Power Analysis
export type { PowerAnalysisResult, TTestPowerOptions } from "./power";
export { tTestPower } from "./power";
// Tests
export type { ContingencyResult, TestResult, TwoWayAnovaResult } from "./tests";
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
  kruskal,
  ks_2samp,
  ks2samp,
  kstest,
  levene,
  lilliefors,
  mannwhitneyu,
  median_test,
  normaltest,
  runs_test,
  shapiro,
  ttest_1samp,
  ttest_ind,
  ttest_rel,
  ttestInd,
  ttestRel,
  wilcoxon,
} from "./tests";
