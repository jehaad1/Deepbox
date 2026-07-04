import { InvalidParameterError } from "../core";
import type { Tensor } from "../ndarray";

/**
 * Output format for estimator transform/predict results.
 *
 * - "default": Return Tensor (the standard behavior)
 * - "array": Return plain nested number arrays
 */
export type OutputType = "default" | "array";

let globalOutputType: OutputType = "default";

/**
 * Set the global output type for all estimators.
 *
 * Controls whether `transform()`, `predict()`, etc. return
 * Tensors ("default") or plain arrays ("array").
 *
 * @param outputType - "default" for Tensor, "array" for number[][]
 *
 * @example
 * ```ts
 * import { set_output } from 'deepbox/ml';
 *
 * set_output('array');   // All outputs as plain arrays
 * set_output('default'); // Back to Tensor outputs
 * ```
 */
export function set_output(outputType: OutputType): void {
  if (outputType !== "default" && outputType !== "array") {
    throw new InvalidParameterError(
      `outputType must be "default" or "array"; received "${String(outputType)}"`,
      "outputType",
      outputType
    );
  }
  globalOutputType = outputType;
}

/**
 * Get the current global output type setting.
 *
 * @returns Current output type
 */
export function get_output(): OutputType {
  return globalOutputType;
}

/**
 * Reset the output type to default (Tensor).
 */
export function reset_output(): void {
  globalOutputType = "default";
}

/**
 * Metadata tags describing estimator capabilities and requirements.
 *
 * Inspired by scikit-learn's estimator tags, these provide machine-readable
 * metadata about what an estimator supports or requires.
 *
 * @example
 * ```ts
 * const tags = getEstimatorTags(myClassifier);
 * if (tags.multiOutput) {
 *   // Can handle multi-output targets
 * }
 * ```
 */
export type EstimatorTags = {
  /** Estimator type: "classifier", "regressor", "clusterer", "transformer", "outlier_detector" */
  readonly estimatorType:
    | "classifier"
    | "regressor"
    | "clusterer"
    | "transformer"
    | "outlier_detector";
  /** Whether the estimator supports multi-output targets */
  readonly multiOutput: boolean;
  /** Whether the estimator requires all features to be non-negative */
  readonly requiresPositiveX: boolean;
  /** Whether the estimator requires the target to be non-negative */
  readonly requiresPositiveY: boolean;
  /** Whether the estimator supports sparse input */
  readonly supportsSparse: boolean;
  /** Whether the estimator supports sample weights */
  readonly supportsSampleWeight: boolean;
  /** Whether the estimator has a predict_proba method */
  readonly hasPredictProba: boolean;
  /** Whether the estimator has a decision_function method */
  readonly hasDecisionFunction: boolean;
  /** Whether fit requires y (false for unsupervised) */
  readonly requiresY: boolean;
};

const DEFAULT_TAGS: EstimatorTags = {
  estimatorType: "classifier",
  multiOutput: false,
  requiresPositiveX: false,
  requiresPositiveY: false,
  supportsSparse: false,
  supportsSampleWeight: false,
  hasPredictProba: false,
  hasDecisionFunction: false,
  requiresY: true,
};

/**
 * Get estimator tags from an estimator, using sensible defaults
 * based on the estimator's interface.
 *
 * If the estimator implements `_getTags()`, those are used.
 * Otherwise, tags are inferred from the estimator's methods.
 *
 * @param estimator - The estimator to get tags for
 * @returns Estimator tags
 */
export function getEstimatorTags(estimator: Estimator): EstimatorTags {
  // Check if the estimator provides its own tags
  const tagged = estimator as Record<string, unknown>;
  if (typeof tagged["_getTags"] === "function") {
    const custom = (tagged["_getTags"] as () => Partial<EstimatorTags>)();
    return { ...DEFAULT_TAGS, ...custom };
  }

  // Infer from interface
  const hasPredict = typeof tagged["predict"] === "function";
  const hasPredictProba = typeof tagged["predictProba"] === "function";
  const hasTransform = typeof tagged["transform"] === "function";
  const hasFitPredict = typeof tagged["fitPredict"] === "function";
  const hasScoreSamples = typeof tagged["scoreSamples"] === "function";

  let estimatorType: EstimatorTags["estimatorType"] = "classifier";
  let requiresY = true;

  if (hasTransform && !hasPredict) {
    estimatorType = "transformer";
    requiresY = false;
  } else if (hasScoreSamples) {
    estimatorType = "outlier_detector";
    requiresY = false;
  } else if (hasFitPredict && !hasPredictProba) {
    estimatorType = "clusterer";
    requiresY = false;
  } else if (hasPredictProba) {
    estimatorType = "classifier";
  } else if (hasPredict) {
    // Could be regressor or classifier; default to regressor if no predictProba
    estimatorType = "regressor";
  }

  return {
    ...DEFAULT_TAGS,
    estimatorType,
    requiresY,
    hasPredictProba: hasPredictProba,
  };
}

/**
 * Base type for all estimators (models) in Deepbox.
 *
 * Base estimator type for all ML models.
 *
 * @template FitParams - Type of parameters passed to fit method
 *
 * References:
 * - Deepbox ML: https://deepbox.dev/docs/ml-linear
 */
export type Estimator<FitParams = void> = {
  /**
   * Fit the model to training data.
   *
   * @param X - Training features of shape (n_samples, n_features)
   * @param y - Training targets (optional for unsupervised learning)
   * @param params - Additional fitting parameters
   * @returns The fitted estimator (for method chaining)
   */
  fit(X: Tensor, y?: Tensor, params?: FitParams): Estimator<FitParams>;

  /**
   * Get parameters for this estimator.
   *
   * @returns Object containing all parameters
   */
  getParams(): Record<string, unknown>;

  /**
   * Set parameters for this estimator.
   *
   * @param params - Parameters to set
   * @returns The estimator (for method chaining)
   */
  setParams(params: Record<string, unknown>): Estimator<FitParams>;

  /**
   * Create a fresh unfitted clone of this estimator with the same parameters.
   *
   * @returns A new estimator instance with identical configuration but no fitted state
   */
  clone?(): Estimator<FitParams>;
};

/**
 * Type for classification models.
 *
 * Classifiers predict discrete class labels.
 */
export type Classifier = Estimator & {
  /**
   * Fit the model to training data.
   *
   * @param X - Training features of shape (n_samples, n_features)
   * @param y - Training targets
   * @returns The fitted estimator
   */
  fit(X: Tensor, y: Tensor): Classifier;

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,)
   */
  predict(X: Tensor): Tensor;

  /**
   * Predict class probabilities for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probabilities of shape (n_samples, n_classes)
   */
  predictProba(X: Tensor): Tensor;

  /**
   * Compute the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples
   * @param y - True labels
   * @returns Mean accuracy score
   */
  score(X: Tensor, y: Tensor): number;

  /** Array of unique class labels seen during fit */
  readonly classes?: Tensor | undefined;
};

/**
 * Type for regression models.
 *
 * Regressors predict continuous values.
 */
export type Regressor = Estimator & {
  /**
   * Fit the model to training data.
   *
   * @param X - Training features of shape (n_samples, n_features)
   * @param y - Training targets
   * @returns The fitted estimator
   */
  fit(X: Tensor, y: Tensor): Regressor;

  /**
   * Predict target values for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,) or (n_samples, n_targets)
   */
  predict(X: Tensor): Tensor;

  /**
   * Compute the coefficient of determination R^2 of the prediction.
   *
   * @param X - Test samples
   * @param y - True target values
   * @returns R^2 score
   */
  score(X: Tensor, y: Tensor): number;
};

/**
 * Type for clustering models.
 *
 * Clusterers group similar samples together.
 */
export type Clusterer = Estimator<void> & {
  /**
   * Fit the model to training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns The fitted estimator
   */
  fit(X: Tensor, y?: Tensor): Clusterer;

  /**
   * Predict cluster labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   */
  predict(X: Tensor): Tensor;

  /**
   * Fit the model and predict cluster labels.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns Cluster labels of shape (n_samples,)
   */
  fitPredict(X: Tensor, y?: Tensor): Tensor;

  /** Cluster centers after fitting */
  readonly clusterCenters?: Tensor;

  /** Labels of each point after fitting */
  readonly labels?: Tensor;
};

/**
 * Type for transformer models.
 *
 * Transformers modify or transform the input data.
 */
export type Transformer = Estimator<void> & {
  /**
   * Fit the model to training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns The fitted estimator
   */
  fit(X: Tensor, y?: Tensor): Transformer;

  /**
   * Transform the input data.
   *
   * @param X - Data to transform
   * @returns Transformed data
   */
  transform(X: Tensor): Tensor;

  /**
   * Fit to data, then transform it.
   *
   * @param X - Training data
   * @param y - Target values (optional)
   * @returns Transformed data
   */
  fitTransform(X: Tensor, y?: Tensor): Tensor;

  /**
   * Inverse transform the data back to original representation.
   *
   * @param X - Transformed data
   * @returns Original representation
   */
  inverseTransform?(X: Tensor): Tensor;
};

/**
 * Type for outlier/anomaly detection models.
 */
export type OutlierDetector = Estimator<void> & {
  /**
   * Fit the model to training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns The fitted estimator
   */
  fit(X: Tensor, y?: Tensor): OutlierDetector;

  /**
   * Predict if samples are outliers or inliers.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels: +1 for inliers, -1 for outliers
   */
  predict(X: Tensor): Tensor;

  /**
   * Fit the model and predict outliers.
   *
   * @param X - Training data
   * @param y - Ignored
   * @returns Labels: +1 for inliers, -1 for outliers
   */
  fitPredict(X: Tensor, y?: Tensor): Tensor;

  /**
   * Compute anomaly scores for samples.
   *
   * @param X - Samples to score
   * @returns Anomaly scores (lower = more abnormal)
   */
  scoreSamples(X: Tensor): Tensor;
};

/**
 * Runtime helper to validate estimator-like objects.
 */
export function assertEstimator<T extends Estimator>(value: T): T {
  if (!value || typeof value !== "object") {
    throw new InvalidParameterError("Estimator must be an object", "value", value);
  }
  const required: Array<keyof Estimator> = ["fit", "getParams", "setParams"];
  for (const name of required) {
    if (typeof value[name] !== "function") {
      throw new InvalidParameterError(`Estimator is missing ${name}()`, "value", value);
    }
  }
  return value;
}
