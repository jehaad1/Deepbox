/**
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox documentation}
 */

export type { SampleWeightInput, WeightedMetricOptions } from "./_internal";
export type {
  AveragedMetricOptions,
  ConfusionMatrixOptions,
  LogLossOptions,
  RocAucOptions,
} from "./classification";
export {
  accuracy,
  averagePrecisionScore,
  balancedAccuracyScore,
  classificationReport,
  cohenKappaScore,
  confusionMatrix,
  f1Score,
  fbetaScore,
  hammingLoss,
  jaccardScore,
  logLoss,
  matthewsCorrcoef,
  precision,
  precisionRecallCurve,
  recall,
  rocAucScore,
  rocCurve,
} from "./classification";
export type { AverageMethod, SilhouetteMetric, SilhouetteScoreOptions } from "./clustering";
export {
  adjustedMutualInfoScore,
  adjustedRandScore,
  calinskiHarabaszScore,
  completenessScore,
  daviesBouldinScore,
  fowlkesMallowsScore,
  homogeneityScore,
  mutualInfoScore,
  normalizedMutualInfoScore,
  randScore,
  silhouetteSamples,
  silhouetteScore,
  vMeasureScore,
} from "./clustering";
export type { DetCurveResult, MultilabelConfusionMatrixEntry } from "./extra";
export {
  brierScoreLoss,
  coverageError,
  d2TweedieScore,
  detCurve,
  hingeLoss,
  labelRankingLoss,
  meanGammaDeviance,
  meanPinballLoss,
  meanPoissonDeviance,
  meanSquaredLogError,
  multilabelConfusionMatrix,
  smape,
  topKAccuracyScore,
  zeroOneLoss,
} from "./extra";
export {
  ndcgScore,
  pairwiseCosine,
  pairwiseEuclidean,
  pairwiseManhattan,
  reciprocalRank,
} from "./pairwise";
export {
  adjustedR2Score,
  explainedVarianceScore,
  mae,
  mape,
  maxError,
  meanAbsoluteError,
  meanAbsolutePercentageError,
  meanSquaredError,
  medianAbsoluteError,
  mse,
  r2Score,
  rmse,
  rootMeanSquaredError,
} from "./regression";
