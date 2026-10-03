/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

// Base types for ML estimators

// Anomaly detection
export { IsolationForest } from "./anomaly/IsolationForest";
export { LocalOutlierFactor } from "./anomaly/LocalOutlierFactor";
export type {
  Classifier,
  Clusterer,
  Estimator,
  EstimatorTags,
  OutlierDetector,
  OutputType,
  Regressor,
  Transformer,
} from "./base";
export {
  assertEstimator,
  get_output,
  getEstimatorTags,
  getOutput,
  reset_output,
  resetOutput,
  set_output,
  setOutput,
} from "./base";
// Calibration
export { CalibratedClassifierCV, calibrationCurve } from "./calibration";
// Clustering models
export { AffinityPropagation } from "./clustering/AffinityPropagation";
export { AgglomerativeClustering } from "./clustering/AgglomerativeClustering";
export { Birch } from "./clustering/Birch";
export { DBSCAN } from "./clustering/DBSCAN";
export { GaussianMixture } from "./clustering/GaussianMixture";
export { KMeans } from "./clustering/KMeans";
export { MeanShift } from "./clustering/MeanShift";
export { MiniBatchKMeans } from "./clustering/MiniBatchKMeans";
export { OPTICS } from "./clustering/OPTICS";
export { SpectralClustering } from "./clustering/SpectralClustering";
// Dimensionality reduction
export {
  FastICA,
  LatentDirichletAllocation,
  NMF,
  PCA,
  TruncatedSVD,
} from "./decomposition";
// Discriminant analysis
export {
  LinearDiscriminantAnalysis,
  QuadraticDiscriminantAnalysis,
} from "./discriminant_analysis";
// Ensemble methods
export type {
  AdaBoostClassifierOptions,
  AdaBoostRegressorOptions,
  BaggingOptions,
  GradientBoostingClassifierOptions,
  GradientBoostingLoss,
  GradientBoostingRegressorOptions,
  StackingClassifierOptions,
  StackingMethod,
  StackingRegressorOptions,
  VotingClassifierOptions,
  VotingRegressorOptions,
} from "./ensemble";
export {
  AdaBoostClassifier,
  AdaBoostRegressor,
  BaggingClassifier,
  BaggingRegressor,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  StackingClassifier,
  StackingRegressor,
  VotingClassifier,
  VotingRegressor,
} from "./ensemble";
// Gaussian Processes
export type {
  GaussianProcessClassifierOptions,
  GaussianProcessRegressorOptions,
} from "./gaussian_process";
export {
  GaussianProcessClassifier,
  GaussianProcessRegressor,
} from "./gaussian_process";
// Inspection
export type { PermutationImportanceOptions, PermutationImportanceResult } from "./inspection";
export { permutationImportance } from "./inspection";
// Linear models - regression and classification
export type { BayesianRidgeOptions } from "./linear/BayesianRidge";
export { BayesianRidge } from "./linear/BayesianRidge";
export type { ElasticNetOptions } from "./linear/ElasticNet";
export { ElasticNet } from "./linear/ElasticNet";
export type { HuberRegressorOptions } from "./linear/HuberRegressor";
export { HuberRegressor } from "./linear/HuberRegressor";
export type {
  IsotonicOutOfBounds,
  IsotonicRegressionOptions,
} from "./linear/IsotonicRegression";
export { IsotonicRegression } from "./linear/IsotonicRegression";
export type { KernelRidgeKernel, KernelRidgeOptions } from "./linear/KernelRidge";
export { KernelRidge } from "./linear/KernelRidge";
export type { LassoOptions } from "./linear/Lasso";
export { Lasso } from "./linear/Lasso";
export { LinearRegression } from "./linear/LinearRegression";
export { LogisticRegression } from "./linear/LogisticRegression";
export { QuantileRegressor } from "./linear/QuantileRegressor";
export { RANSACRegressor } from "./linear/RANSACRegressor";
export { Ridge } from "./linear/Ridge";
export { SGDClassifier, SGDRegressor } from "./linear/SGD";
// Manifold learning
export { Isomap, MDS, SpectralEmbedding, TSNE } from "./manifold";
export type { TSNEOptions } from "./manifold/TSNE";
// Multi-layer Perceptron
export { MLPClassifier, MLPRegressor } from "./mlp";
// Model selection
export type { CrossValidateResult, GridSearchResult } from "./model_selection";
export {
  cross_val_score,
  cross_validate,
  crossValidate,
  crossValScore,
  GridSearchCV,
  RandomizedSearchCV,
} from "./model_selection";
// Multiclass meta-estimators
export { OneVsOneClassifier, OneVsRestClassifier } from "./multiclass";
// Naive Bayes
export { GaussianNB } from "./naive_bayes";
export { BernoulliNB } from "./naive_bayes/BernoulliNB";
export { CategoricalNB } from "./naive_bayes/CategoricalNB";
export { ComplementNB } from "./naive_bayes/ComplementNB";
export { MultinomialNB } from "./naive_bayes/MultinomialNB";
// Neighbors
export {
  KNeighborsClassifier,
  KNeighborsRegressor,
  NearestNeighbors,
} from "./neighbors";
export { BallTree } from "./neighbors/BallTree";
export { KDTree } from "./neighbors/KDTree";
export { NearestCentroid } from "./neighbors/NearestCentroid";
export {
  RadiusNeighborsClassifier,
  RadiusNeighborsRegressor,
} from "./neighbors/RadiusNeighbors";
// Pipeline
export {
  ColumnTransformer,
  FeatureUnion,
  makePipeline,
  Pipeline,
} from "./pipeline";
// Random Projection
export { GaussianRandomProjection, johnsonLindenstraussMinDim } from "./random_projection";
// Semi-supervised
export type { SelfTrainingTermination } from "./semi_supervised";
export {
  LabelPropagation,
  LabelSpreading,
  SelfTrainingClassifier,
} from "./semi_supervised";
// Support Vector Machines
export type {
  ClassWeightOption,
  GammaOption,
  KernelType,
  LinearSVCLoss,
  LinearSVCOptions,
  LinearSVRLoss,
  LinearSVROptions,
  NuSVCOptions,
  NuSVROptions,
  OneClassSVMOptions,
  SVCOptions,
  SVROptions,
} from "./svm";
export {
  LinearSVC,
  LinearSVR,
  NuSVC,
  NuSVR,
  OneClassSVM,
  SVC,
  SVR,
} from "./svm";
// Tree-based models
export type {
  ClassificationCriterion,
  DecisionTreeClassifierOptions,
  DecisionTreeRegressorOptions,
  ForestClassWeight,
  ForestMaxFeatures,
  RandomForestClassifierOptions,
  RandomForestOptions,
  RandomForestRegressorOptions,
  TreeClassWeight,
  TreeGrowthOptions,
  TreeMaxFeatures,
} from "./tree";
export {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  ExtraTreesClassifier,
  ExtraTreesRegressor,
  export_text,
  exportText,
  RandomForestClassifier,
  RandomForestRegressor,
} from "./tree";
export type { ExtraTreesClassifierOptions, ExtraTreesOptions } from "./tree/ExtraTrees";
