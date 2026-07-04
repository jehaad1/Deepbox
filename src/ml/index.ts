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
  reset_output,
  set_output,
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
export {
  GaussianProcessClassifier,
  GaussianProcessRegressor,
} from "./gaussian_process";
// Inspection
export type { PermutationImportanceResult } from "./inspection";
export { permutationImportance } from "./inspection";
// Linear models - regression and classification
export { BayesianRidge } from "./linear/BayesianRidge";
export { ElasticNet } from "./linear/ElasticNet";
export { HuberRegressor } from "./linear/HuberRegressor";
export { IsotonicRegression } from "./linear/IsotonicRegression";
export { KernelRidge } from "./linear/KernelRidge";
export { Lasso } from "./linear/Lasso";
export { LinearRegression } from "./linear/LinearRegression";
export { LogisticRegression } from "./linear/LogisticRegression";
export { QuantileRegressor } from "./linear/QuantileRegressor";
export { RANSACRegressor } from "./linear/RANSACRegressor";
export { Ridge } from "./linear/Ridge";
export { SGDClassifier, SGDRegressor } from "./linear/SGD";
// Manifold learning
export { Isomap, MDS, SpectralEmbedding, TSNE } from "./manifold";
// Multi-layer Perceptron
export { MLPClassifier, MLPRegressor } from "./mlp";
// Model selection
export type { CrossValidateResult, GridSearchResult } from "./model_selection";
export {
  cross_val_score,
  cross_validate,
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
export { GaussianRandomProjection } from "./random_projection";
// Semi-supervised
export {
  LabelPropagation,
  LabelSpreading,
  SelfTrainingClassifier,
} from "./semi_supervised";
// Support Vector Machines
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
export type { ClassificationCriterion } from "./tree";
export {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  ExtraTreesClassifier,
  ExtraTreesRegressor,
  export_text,
  RandomForestClassifier,
  RandomForestRegressor,
} from "./tree";
