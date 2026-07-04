/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import { LinearRegression } from "../linear/LinearRegression";
import { LogisticRegression } from "../linear/LogisticRegression";

/**
 * Stacking Classifier.
 *
 * Combines multiple classifiers using a meta-learner (final estimator)
 * that learns to combine the base estimators' predictions.
 *
 * The base estimators produce predictions on the training set via
 * cross-validation (or direct fitting), then the meta-learner is trained
 * on those predictions to produce the final output.
 *
 * @example
 * ```ts
 * import { StackingClassifier, DecisionTreeClassifier, LogisticRegression } from 'deepbox/ml';
 *
 * const clf = new StackingClassifier({
 *   estimators: [
 *     new DecisionTreeClassifier({ maxDepth: 3 }),
 *     new DecisionTreeClassifier({ maxDepth: 5 }),
 *   ],
 *   finalEstimator: new LogisticRegression(),
 * });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(X_test);
 * ```
 *
 * @category Ensemble
 */
export class StackingClassifier implements Classifier {
  private readonly estimators: Classifier[];
  private readonly finalEstimator: Classifier;
  private passthrough: boolean;

  private classLabels: number[] = [];
  private nFeatures = 0;
  private fitted = false;

  constructor(options: {
    readonly estimators: Classifier[];
    readonly finalEstimator?: Classifier;
    readonly passthrough?: boolean;
  }) {
    if (!options.estimators || options.estimators.length === 0) {
      throw new InvalidParameterError(
        "StackingClassifier requires at least one base estimator",
        "estimators",
        options.estimators
      );
    }
    this.estimators = options.estimators;
    this.finalEstimator = (options.finalEstimator ?? new LogisticRegression()) as Classifier;
    this.passthrough = options.passthrough ?? false;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    // Extract class labels
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);

    // Fit all base estimators on full training data
    for (const est of this.estimators) {
      est.fit(X, y);
    }

    // Generate meta-features: predictions from each base estimator
    const metaCols: number[][] = [];
    for (const est of this.estimators) {
      const pred = est.predict(X);
      const col: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        col.push(Number(pred.data[pred.offset + i]));
      }
      metaCols.push(col);
    }

    // Build meta-feature matrix
    const metaData: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (const col of metaCols) {
        row.push(col[i] ?? 0);
      }
      if (this.passthrough) {
        for (let f = 0; f < nFeatures; f++) {
          row.push(Number(X.data[X.offset + i * nFeatures + f] ?? 0));
        }
      }
      metaData.push(row);
    }

    // Fit the final estimator on meta-features
    this.finalEstimator.fit(tensor(metaData), y);

    this.fitted = true;
    return this;
  }

  private buildMetaFeatures(X: Tensor): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const metaCols: number[][] = [];
    for (const est of this.estimators) {
      const pred = est.predict(X);
      const col: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        col.push(Number(pred.data[pred.offset + i]));
      }
      metaCols.push(col);
    }

    const metaData: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (const col of metaCols) {
        row.push(col[i] ?? 0);
      }
      if (this.passthrough) {
        for (let f = 0; f < nFeatures; f++) {
          row.push(Number(X.data[X.offset + i * nFeatures + f] ?? 0));
        }
      }
      metaData.push(row);
    }

    return tensor(metaData);
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StackingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "StackingClassifier");
    const meta = this.buildMetaFeatures(X);
    return this.finalEstimator.predict(meta);
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StackingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "StackingClassifier");
    const meta = this.buildMetaFeatures(X);
    return this.finalEstimator.predictProba(meta);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.estimators.length,
      passthrough: this.passthrough,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "passthrough":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("passthrough must be a boolean", "passthrough", value);
          }
          this.passthrough = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * Stacking Regressor.
 *
 * Combines multiple regressors using a meta-learner (final estimator)
 * that learns to combine the base estimators' predictions.
 *
 * @example
 * ```ts
 * import { StackingRegressor, DecisionTreeRegressor, LinearRegression } from 'deepbox/ml';
 *
 * const reg = new StackingRegressor({
 *   estimators: [
 *     new DecisionTreeRegressor({ maxDepth: 3 }),
 *     new DecisionTreeRegressor({ maxDepth: 5 }),
 *   ],
 *   finalEstimator: new LinearRegression(),
 * });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(X_test);
 * ```
 *
 * @category Ensemble
 */
export class StackingRegressor implements Regressor {
  private readonly estimators: Regressor[];
  private readonly finalEstimator: Regressor;
  private passthrough: boolean;

  private nFeatures = 0;
  private fitted = false;

  constructor(options: {
    readonly estimators: Regressor[];
    readonly finalEstimator?: Regressor;
    readonly passthrough?: boolean;
  }) {
    if (!options.estimators || options.estimators.length === 0) {
      throw new InvalidParameterError(
        "StackingRegressor requires at least one base estimator",
        "estimators",
        options.estimators
      );
    }
    this.estimators = options.estimators;
    this.finalEstimator = (options.finalEstimator ?? new LinearRegression()) as Regressor;
    this.passthrough = options.passthrough ?? false;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    // Fit all base estimators on full training data
    for (const est of this.estimators) {
      est.fit(X, y);
    }

    // Generate meta-features: predictions from each base estimator
    const metaCols: number[][] = [];
    for (const est of this.estimators) {
      const pred = est.predict(X);
      const col: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        col.push(Number(pred.data[pred.offset + i]));
      }
      metaCols.push(col);
    }

    // Build meta-feature matrix
    const metaData: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (const col of metaCols) {
        row.push(col[i] ?? 0);
      }
      if (this.passthrough) {
        for (let f = 0; f < nFeatures; f++) {
          row.push(Number(X.data[X.offset + i * nFeatures + f] ?? 0));
        }
      }
      metaData.push(row);
    }

    // Fit the final estimator on meta-features
    this.finalEstimator.fit(tensor(metaData), y);

    this.fitted = true;
    return this;
  }

  private buildMetaFeatures(X: Tensor): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const metaCols: number[][] = [];
    for (const est of this.estimators) {
      const pred = est.predict(X);
      const col: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        col.push(Number(pred.data[pred.offset + i]));
      }
      metaCols.push(col);
    }

    const metaData: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (const col of metaCols) {
        row.push(col[i] ?? 0);
      }
      if (this.passthrough) {
        for (let f = 0; f < nFeatures; f++) {
          row.push(Number(X.data[X.offset + i * nFeatures + f] ?? 0));
        }
      }
      metaData.push(row);
    }

    return tensor(metaData);
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StackingRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "StackingRegressor");
    const meta = this.buildMetaFeatures(X);
    return this.finalEstimator.predict(meta);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i]);
    }
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i]);
      const pVal = Number(predictions.data[predictions.offset + i]);
      ssRes += (yVal - pVal) ** 2;
      ssTot += (yVal - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.estimators.length,
      passthrough: this.passthrough,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "passthrough":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("passthrough must be a boolean", "passthrough", value);
          }
          this.passthrough = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
