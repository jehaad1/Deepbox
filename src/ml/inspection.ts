/**
 * Model inspection and explanation utilities.
 *
 * @module ml/inspection
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";
import { type Tensor, tensor } from "../ndarray";

/**
 * Result of a permutation importance computation.
 */
export type PermutationImportanceResult = {
  /** Mean importance for each feature (shape: [n_features]). */
  readonly importancesMean: Tensor;
  /** Std deviation of importance for each feature (shape: [n_features]). */
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
 * Permutation feature importance.
 *
 * Measures the decrease in a model's score when a single feature
 * value is randomly shuffled. A large decrease indicates the model
 * relies heavily on that feature.
 *
 * @param estimator - A fitted estimator with a `score(X, y)` method
 * @param X - Test features of shape (n_samples, n_features)
 * @param y - Test labels/targets of shape (n_samples,)
 * @param options - Configuration options
 * @returns Permutation importance result
 *
 * @example
 * ```ts
 * import { permutationImportance, LinearRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const model = new LinearRegression();
 * model.fit(X_train, y_train);
 * const result = permutationImportance(model, X_test, y_test, { nRepeats: 10 });
 * console.log(result.importancesMean);
 * ```
 */
export function permutationImportance(
  estimator: Scorable,
  X: Tensor,
  y: Tensor,
  options: { readonly nRepeats?: number; readonly randomState?: number } = {}
): PermutationImportanceResult {
  const nRepeats = options.nRepeats ?? 5;

  if (!Number.isInteger(nRepeats) || nRepeats < 1) {
    throw new InvalidParameterError("nRepeats must be an integer >= 1", "nRepeats", nRepeats);
  }

  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.shape[1] ?? 0;

  // Baseline score
  const baselineScore = estimator.score(X, y);

  // Build a mutable copy of X data
  const flat = new Float64Array(nSamples * nFeatures);
  for (let i = 0; i < nSamples * nFeatures; i++) {
    flat[i] = Number(X.data[X.offset + i]);
  }

  // Seeded RNG (simple LCG)
  let rngState = options.randomState ?? Date.now() | 0;
  function nextRng(): number {
    rngState = (rngState * 1664525 + 1013904223) | 0;
    return (rngState >>> 0) / 0x100000000;
  }

  const importances: number[][] = [];
  for (let rep = 0; rep < nRepeats; rep++) {
    const row: number[] = [];
    for (let f = 0; f < nFeatures; f++) {
      // Save original column
      const original = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        original[i] = flat[i * nFeatures + f] ?? 0;
      }

      // Fisher-Yates shuffle on the column
      const shuffled = new Float64Array(original);
      for (let i = nSamples - 1; i > 0; i--) {
        const j = Math.floor(nextRng() * (i + 1));
        const tmp = shuffled[i] ?? 0;
        shuffled[i] = shuffled[j] ?? 0;
        shuffled[j] = tmp;
      }

      // Apply shuffled column
      for (let i = 0; i < nSamples; i++) {
        flat[i * nFeatures + f] = shuffled[i] ?? 0;
      }

      // Create tensor and score
      const permutedRows: number[][] = [];
      for (let i = 0; i < nSamples; i++) {
        const r: number[] = [];
        for (let j = 0; j < nFeatures; j++) {
          r.push(flat[i * nFeatures + j] ?? 0);
        }
        permutedRows.push(r);
      }
      const XPermuted = tensor(permutedRows);
      const permutedScore = estimator.score(XPermuted, y);
      row.push(baselineScore - permutedScore);

      // Restore original column
      for (let i = 0; i < nSamples; i++) {
        flat[i * nFeatures + f] = original[i] ?? 0;
      }
    }
    importances.push(row);
  }

  // Compute mean and std per feature
  const mean: number[] = [];
  const std: number[] = [];
  for (let f = 0; f < nFeatures; f++) {
    let sum = 0;
    for (let rep = 0; rep < nRepeats; rep++) {
      sum += importances[rep]![f]!;
    }
    const m = sum / nRepeats;
    mean.push(m);

    let variance = 0;
    for (let rep = 0; rep < nRepeats; rep++) {
      const diff = importances[rep]![f]! - m;
      variance += diff * diff;
    }
    std.push(Math.sqrt(variance / nRepeats));
  }

  return {
    importancesMean: tensor(mean),
    importancesStd: tensor(std),
    importances: tensor(importances),
  };
}
