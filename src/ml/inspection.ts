/**
 * Model inspection and explanation utilities.
 *
 * @module ml/inspection
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, ShapeError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { Generator } from "../random/Generator";
import { __random, __randomBelow } from "../random/random";
import { toFloat64View } from "./_validation";

/**
 * Result of a permutation importance computation.
 *
 * Note that `importances` is laid out as (n_repeats, n_features), the transpose of
 * scikit-learn's `(n_features, n_repeats)`.
 */
export type PermutationImportanceResult = {
  /** Mean importance for each feature (shape: [n_features]). */
  readonly importancesMean: Tensor;
  /** Standard deviation of the importance for each feature, with ddof = 0 (shape: [n_features]). */
  readonly importancesStd: Tensor;
  /** Raw importances matrix (shape: [n_repeats, n_features]). */
  readonly importances: Tensor;
};

/**
 * Estimator-like object that has a `score(X, y)` method.
 */
interface Scorable {
  score(X: Tensor, y: Tensor): number;
}

/**
 * Options for {@link permutationImportance}.
 */
export type PermutationImportanceOptions = {
  /** Number of shuffles per feature (default: 5). Must be an integer >= 1. */
  readonly nRepeats?: number;
  /**
   * Seed for the shuffles. Equal seeds give identical results. When omitted, the global
   * seed set with `setSeed` is used, and unseeded runs are random.
   */
  readonly randomState?: number;
  /**
   * Custom score function where larger is better (default: `estimator.score(X, y)`).
   * Use it to measure importance with a metric other than accuracy or R^2.
   */
  readonly scoring?: (estimator: Scorable, X: Tensor, y: Tensor) => number;
};

/**
 * Permutation feature importance.
 *
 * Measures the decrease in a model's score when a single feature
 * value is randomly shuffled. A large decrease indicates the model
 * relies heavily on that feature.
 *
 * `X` is never modified. Each importance is `baselineScore - permutedScore`, so a
 * feature the model ignores has an importance near 0 and a negative value means the
 * shuffled feature happened to score slightly better.
 *
 * @param estimator - A fitted estimator with a `score(X, y)` method
 * @param X - Test features of shape (n_samples, n_features)
 * @param y - Test labels/targets of shape (n_samples,)
 * @param options - Configuration options
 * @returns Permutation importance result
 * @throws {ShapeError} If `X` is not 2-D, `y` is not 1-D, or their sample counts differ
 * @throws {DataValidationError} If `X` has no samples or no features
 * @throws {InvalidParameterError} If `nRepeats` is not an integer >= 1, or `estimator` has
 * no `score` method and no `scoring` function is given
 *
 * @example
 * ```ts
 * import { permutationImportance, LinearRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 5], [2, 3], [3, 8], [4, 1], [5, 9], [6, 2]]);
 * const y = tensor([2, 4, 6, 8, 10, 12]);
 * const model = new LinearRegression();
 * model.fit(X, y);
 * const result = permutationImportance(model, X, y, { nRepeats: 10, randomState: 0 });
 * console.log(result.importancesMean);
 * ```
 */
export function permutationImportance(
  estimator: Scorable,
  X: Tensor,
  y: Tensor,
  options: PermutationImportanceOptions = {}
): PermutationImportanceResult {
  const nRepeats = options.nRepeats ?? 5;

  if (!Number.isInteger(nRepeats) || nRepeats < 1) {
    throw new InvalidParameterError("nRepeats must be an integer >= 1", "nRepeats", nRepeats);
  }
  const scoring = options.scoring;
  if (scoring !== undefined && typeof scoring !== "function") {
    throw new InvalidParameterError("scoring must be a function", "scoring", scoring);
  }
  if (scoring === undefined && typeof estimator?.score !== "function") {
    throw new InvalidParameterError(
      "estimator must have a score(X, y) method or a scoring function must be given",
      "estimator",
      estimator
    );
  }
  if (X.ndim !== 2) {
    throw new ShapeError(`X must be 2-dimensional; got ndim=${X.ndim}`);
  }
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }

  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.shape[1] ?? 0;
  if (nSamples === 0) {
    throw new DataValidationError("X must have at least one sample");
  }
  if (nFeatures === 0) {
    throw new DataValidationError("X must have at least one feature");
  }
  if (y.shape[0] !== nSamples) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X.shape[0]=${nSamples}, y.shape[0]=${y.shape[0]}`
    );
  }

  const evaluate = (Xs: Tensor): number =>
    scoring !== undefined ? scoring(estimator, Xs, y) : estimator.score(Xs, y);

  // Private mutable copy of X (toFloat64View can return a view of X's own memory).
  const flat = Float64Array.from(toFloat64View(X));
  const toTensor = (): Tensor =>
    tensor(flat.slice(), { dtype: "float64" }).reshape([nSamples, nFeatures]);

  const rng = options.randomState !== undefined ? new Generator(options.randomState) : null;
  const next = rng !== null ? () => rng.random() : __random;

  // Baseline score on the unmodified data.
  const baselineScore = evaluate(toTensor());

  const importances = new Float64Array(nRepeats * nFeatures);
  const original = new Float64Array(nSamples);
  const perm = new Int32Array(nSamples);
  for (let rep = 0; rep < nRepeats; rep++) {
    for (let f = 0; f < nFeatures; f++) {
      // Save the column, then write back a Fisher-Yates permutation of it.
      for (let i = 0; i < nSamples; i++) {
        original[i] = flat[i * nFeatures + f] ?? 0;
        perm[i] = i;
      }
      for (let i = nSamples - 1; i > 0; i--) {
        const j = __randomBelow(next, i + 1);
        const tmp = perm[i] ?? 0;
        perm[i] = perm[j] ?? 0;
        perm[j] = tmp;
      }
      for (let i = 0; i < nSamples; i++) {
        flat[i * nFeatures + f] = original[perm[i] ?? 0] ?? 0;
      }

      const permutedScore = evaluate(toTensor());
      importances[rep * nFeatures + f] = baselineScore - permutedScore;

      // Restore the column before moving to the next feature.
      for (let i = 0; i < nSamples; i++) {
        flat[i * nFeatures + f] = original[i] ?? 0;
      }
    }
  }

  // Mean and (population) standard deviation per feature.
  const mean = new Float64Array(nFeatures);
  const std = new Float64Array(nFeatures);
  for (let f = 0; f < nFeatures; f++) {
    let sum = 0;
    for (let rep = 0; rep < nRepeats; rep++) {
      sum += importances[rep * nFeatures + f] ?? 0;
    }
    const m = sum / nRepeats;
    mean[f] = m;

    let variance = 0;
    for (let rep = 0; rep < nRepeats; rep++) {
      const diff = (importances[rep * nFeatures + f] ?? 0) - m;
      variance += diff * diff;
    }
    std[f] = Math.sqrt(variance / nRepeats);
  }

  return {
    importancesMean: tensor(mean, { dtype: "float64" }),
    importancesStd: tensor(std, { dtype: "float64" }),
    importances: tensor(importances, { dtype: "float64" }).reshape([nRepeats, nFeatures]),
  };
}
