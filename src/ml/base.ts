import { InvalidParameterError } from "../core";
import type { Tensor } from "../ndarray";

/**
 * Base type for all estimators (models) in Deepbox.
 *
 * This follows the scikit-learn estimator type design.
 *
 * @template FitParams - Type of parameters passed to fit method
 *
 * References:
 * - scikit-learn API: https://scikit-learn.org/stable/developers/develop.html
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
