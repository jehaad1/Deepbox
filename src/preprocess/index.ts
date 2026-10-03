/**
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

// Encoders

// Discretization
export { KBinsDiscretizer } from "./discretizer";
export {
  LabelBinarizer,
  LabelEncoder,
  MultiLabelBinarizer,
  OneHotEncoder,
  OrdinalEncoder,
  TargetEncoder,
} from "./encoders";
// Feature selection
export {
  f_classif,
  f_regression,
  fClassif,
  fRegression,
  type ImportanceEstimator,
  RFE,
  RFECV,
  type RFECVScoring,
  type ScoreFunc,
  type ScoringEstimator,
  SelectFromModel,
  SelectKBest,
  VarianceThreshold,
} from "./feature_selection";
// Imputation
export { KNNImputer, MissingIndicator, SimpleImputer } from "./impute";
// Mutual information scoring
export {
  type MutualInfoOptions,
  mutual_info_classif,
  mutual_info_regression,
  mutualInfoClassif,
  mutualInfoRegression,
} from "./mutual_info";
// Feature transformers
export {
  Binarizer,
  FunctionTransformer,
  PolynomialFeatures,
} from "./polynomial";
// Scalers
export {
  MaxAbsScaler,
  MinMaxScaler,
  Normalizer,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
} from "./scalers";

// Spline features
export { type SplineExtrapolation, type SplineKnots, SplineTransformer } from "./spline";
// Splitting
export {
  GroupKFold,
  GroupShuffleSplit,
  KFold,
  LeaveOneGroupOut,
  LeaveOneOut,
  LeavePGroupsOut,
  LeavePOut,
  PredefinedSplit,
  RepeatedKFold,
  RepeatedStratifiedKFold,
  ShuffleSplit,
  type SplitResult,
  StratifiedGroupKFold,
  StratifiedKFold,
  StratifiedShuffleSplit,
  TimeSeriesSplit,
  trainTestSplit,
} from "./split";
// Text feature extraction
export {
  CountVectorizer,
  type CountVectorizerOptions,
  HashingVectorizer,
  type HashingVectorizerOptions,
  TfidfVectorizer,
  type TfidfVectorizerOptions,
} from "./text";
