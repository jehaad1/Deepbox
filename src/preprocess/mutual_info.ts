/**
 * Mutual information scoring functions for feature selection.
 *
 * Implements k-nearest-neighbor based mutual information estimation
 * following the approach of Kraskov, Stögbauer & Grassberger (2004).
 *
 * @module preprocess/mutual_info
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import { DTypeError, InvalidParameterError } from "../core/errors";
import type { Tensor } from "../ndarray";
import { assert2D, getShape2D, getStrides2D } from "./_internal";

/**
 * Digamma (psi) function using asymptotic expansion with recurrence.
 * Accurate to ~1e-12 for positive arguments.
 */
function digamma(x: number): number {
  let result = 0;
  // Use recurrence to shift x into range where asymptotic expansion is accurate
  while (x < 6) {
    result -= 1 / x;
    x += 1;
  }
  // Asymptotic expansion for large x
  const invX = 1 / x;
  const invX2 = invX * invX;
  result +=
    Math.log(x) -
    0.5 * invX -
    invX2 * (1 / 12 - invX2 * (1 / 120 - invX2 * (1 / 252 - invX2 * (1 / 240))));
  return result;
}

/**
 * Read a numeric value from a tensor at a flat index.
 */
function readValue(t: Tensor, offset: number): number {
  const raw = t.data[offset];
  if (typeof raw === "number") return raw;
  if (typeof raw === "bigint") return Number(raw);
  return NaN;
}

/**
 * Count points within a given radius for 1D data.
 */
function countWithinRadius1D(
  values: Float64Array,
  sampleIdx: number,
  radius: number,
  nSamples: number
): number {
  let count = 0;
  const center = values[sampleIdx] ?? 0;
  for (let j = 0; j < nSamples; j++) {
    if (j === sampleIdx) continue;
    if (Math.abs((values[j] ?? 0) - center) <= radius) {
      count++;
    }
  }
  return count;
}

/**
 * Extract a 2D tensor into a flat Float64Array in row-major order.
 */
function extractFeatureData(X: Tensor, nSamples: number, nFeatures: number): Float64Array {
  const [rowStride, colStride] = getStrides2D(X);
  const data = new Float64Array(nSamples * nFeatures);
  for (let i = 0; i < nSamples; i++) {
    for (let j = 0; j < nFeatures; j++) {
      data[i * nFeatures + j] = readValue(X, X.offset + i * rowStride + j * colStride);
    }
  }
  return data;
}

/**
 * Extract a 1D target tensor into Float64Array.
 */
function extractTargetData(y: Tensor, nSamples: number): Float64Array {
  const data = new Float64Array(nSamples);
  const stride = y.strides[0] ?? 1;
  for (let i = 0; i < nSamples; i++) {
    data[i] = readValue(y, y.offset + i * stride);
  }
  return data;
}

/**
 * Add small noise to continuous features to break ties.
 * Noise magnitude is proportional to the feature's range.
 */
function addNoise(data: Float64Array, nSamples: number, nFeatures: number, seed: number): void {
  // Simple LCG for reproducible noise
  const a = 1103515245;
  const c = 12345;
  const m = 2 ** 31;
  let state = ((seed % m) + m) % m;
  const nextRand = (): number => {
    state = (a * state + c) % m;
    return state / m;
  };

  for (let f = 0; f < nFeatures; f++) {
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < nSamples; i++) {
      const v = data[i * nFeatures + f] ?? 0;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    const range = maxVal - minVal;
    const noiseScale = range > 0 ? range * 1e-10 : 1e-10;
    for (let i = 0; i < nSamples; i++) {
      const idx = i * nFeatures + f;
      data[idx] = (data[idx] ?? 0) + (nextRand() - 0.5) * noiseScale;
    }
  }
}

/**
 * Estimate mutual information between each continuous feature and a discrete target.
 * Uses the method from Kraskov, Stögbauer & Grassberger (2004), adapted for
 * mixed continuous-discrete variables (Ross, 2014).
 */
function estimateMIClassif(
  xData: Float64Array,
  yData: Float64Array,
  nSamples: number,
  nFeatures: number,
  k: number
): number[] {
  const scores: number[] = [];

  for (let f = 0; f < nFeatures; f++) {
    // Extract single feature column
    const featureData = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      featureData[i] = xData[i * nFeatures + f] ?? 0;
    }

    // For each class, count members
    const classMap = new Map<number, number[]>();
    for (let i = 0; i < nSamples; i++) {
      const label = yData[i] ?? 0;
      let arr = classMap.get(label);
      if (!arr) {
        arr = [];
        classMap.set(label, arr);
      }
      arr.push(i);
    }

    // MI estimation using KSG estimator adapted for classification
    // I(X;Y) = psi(k) - <psi(n_c)> + psi(N) - <psi(m)>
    // where n_c = number of samples in same class as point i,
    // m = number of points within epsilon distance in X-space
    let sumPsiNc = 0;
    let sumPsiM = 0;

    for (let i = 0; i < nSamples; i++) {
      const label = yData[i] ?? 0;
      const classMembers = classMap.get(label);
      if (!classMembers || classMembers.length === 0) continue;
      const nc = classMembers.length;
      sumPsiNc += digamma(nc);

      // Find the k-th nearest neighbor among same-class points via an
      // O(n*k) selection scan (full sort per point was the hot spot).
      const xi = featureData[i] ?? 0;
      const kCap = Math.min(k, nc - 1);
      const kBest = new Float64Array(Math.max(kCap, 1));
      let filled = 0;
      for (const j of classMembers) {
        if (j === i) continue;
        const d = Math.abs(xi - (featureData[j] ?? 0));
        if (filled < kCap) {
          let pos = filled;
          while (pos > 0 && (kBest[pos - 1] ?? 0) > d) {
            kBest[pos] = kBest[pos - 1] ?? 0;
            pos--;
          }
          kBest[pos] = d;
          filled++;
        } else if (kCap > 0 && d < (kBest[kCap - 1] ?? 0)) {
          let pos = kCap - 1;
          while (pos > 0 && (kBest[pos - 1] ?? 0) > d) {
            kBest[pos] = kBest[pos - 1] ?? 0;
            pos--;
          }
          kBest[pos] = d;
        }
      }
      const epsilon = filled > 0 ? (kBest[filled - 1] ?? 0) : 0;

      // Count all points (regardless of class) within epsilon in X-space
      const m = countWithinRadius1D(featureData, i, epsilon, nSamples);
      sumPsiM += digamma(Math.max(1, m + 1));
    }

    const mi = Math.max(
      0,
      digamma(nSamples) - sumPsiNc / nSamples + digamma(k) - sumPsiM / nSamples
    );
    scores.push(mi);
  }

  return scores;
}

/**
 * Estimate mutual information between each continuous feature and a continuous target.
 * Uses the KSG estimator (Kraskov, Stögbauer & Grassberger, 2004), algorithm 1.
 */
function estimateMIRegression(
  xData: Float64Array,
  yData: Float64Array,
  nSamples: number,
  nFeatures: number,
  k: number
): number[] {
  const scores: number[] = [];

  for (let f = 0; f < nFeatures; f++) {
    // Extract single feature column
    const featureData = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      featureData[i] = xData[i * nFeatures + f] ?? 0;
    }

    // For each point, find k-th nearest neighbor in joint (x, y) space using Chebyshev
    let sumPsiNx = 0;
    let sumPsiNy = 0;

    const kBest = new Float64Array(k);
    for (let i = 0; i < nSamples; i++) {
      // Find the k-th nearest neighbor distance in joint space with an
      // O(n*k) selection scan (sorting all n distances per point made the
      // estimator ~25x slower).
      let filled = 0;
      const xi = featureData[i] ?? 0;
      const yi = yData[i] ?? 0;
      for (let j = 0; j < nSamples; j++) {
        if (i === j) continue;
        const dx = Math.abs(xi - (featureData[j] ?? 0));
        const dy = Math.abs(yi - (yData[j] ?? 0));
        const d = dx > dy ? dx : dy;
        if (filled < k) {
          // Insert into the sorted k-buffer.
          let pos = filled;
          while (pos > 0 && (kBest[pos - 1] ?? 0) > d) {
            kBest[pos] = kBest[pos - 1] ?? 0;
            pos--;
          }
          kBest[pos] = d;
          filled++;
        } else if (d < (kBest[k - 1] ?? 0)) {
          let pos = k - 1;
          while (pos > 0 && (kBest[pos - 1] ?? 0) > d) {
            kBest[pos] = kBest[pos - 1] ?? 0;
            pos--;
          }
          kBest[pos] = d;
        }
      }
      const epsilon = kBest[k - 1] ?? 0;

      // Count points within epsilon in X-marginal
      const nx = countWithinRadius1D(featureData, i, epsilon, nSamples);
      // Count points within epsilon in Y-marginal
      const ny = countWithinRadius1D(yData, i, epsilon, nSamples);

      sumPsiNx += digamma(Math.max(1, nx + 1));
      sumPsiNy += digamma(Math.max(1, ny + 1));
    }

    const mi = Math.max(
      0,
      digamma(k) + digamma(nSamples) - sumPsiNx / nSamples - sumPsiNy / nSamples
    );
    scores.push(mi);
  }

  return scores;
}

/**
 * Estimate mutual information between each feature and a discrete target variable.
 *
 * Mutual information (MI) measures the dependency between variables.
 * It equals zero iff two random variables are independent, and higher values
 * mean higher dependency. This function uses a nearest-neighbor based estimator.
 *
 * Can be used as a `scoreFunc` for `SelectKBest`.
 *
 * @param X - Feature matrix of shape [n_samples, n_features]
 * @param y - Target vector of shape [n_samples] (discrete class labels)
 * @param options - Optional parameters
 * @param options.nNeighbors - Number of neighbors for MI estimation (default: 3)
 * @param options.randomState - Seed for noise added to break ties (default: 0)
 * @returns Array of MI scores, one per feature (in nats, not bits)
 *
 * @example
 * ```ts
 * import { mutual_info_classif, SelectKBest } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const y = tensor([0, 0, 1, 1]);
 * const scores = mutual_info_classif(X, y);
 *
 * // Use with SelectKBest
 * const skb = new SelectKBest({ scoreFunc: mutual_info_classif, k: 1 });
 * skb.fit(X, y);
 * ```
 * @deprecated Prefer {@link mutualInfoClassif}.
 */
export function mutual_info_classif(
  X: Tensor,
  y: Tensor,
  options: { nNeighbors?: number; randomState?: number } = {}
): number[] {
  if (X.dtype === "string") {
    throw new DTypeError("mutual_info_classif requires numeric features");
  }
  if (y.dtype === "string") {
    throw new DTypeError("mutual_info_classif requires numeric target");
  }
  assert2D(X, "X");
  const [nSamples, nFeatures] = getShape2D(X);

  if (y.ndim !== 1 || (y.shape[0] ?? -1) !== nSamples) {
    throw new InvalidParameterError(
      `y must be a 1D tensor with ${nSamples} elements`,
      "y",
      y.shape
    );
  }

  const k = options.nNeighbors ?? 3;
  if (!Number.isInteger(k) || k < 1) {
    throw new InvalidParameterError("nNeighbors must be a positive integer", "nNeighbors", k);
  }
  if (k >= nSamples) {
    throw new InvalidParameterError(
      `nNeighbors (${k}) must be less than n_samples (${nSamples})`,
      "nNeighbors",
      k
    );
  }

  const xData = extractFeatureData(X, nSamples, nFeatures);
  const yData = extractTargetData(y, nSamples);

  // Add small noise to break ties
  addNoise(xData, nSamples, nFeatures, options.randomState ?? 0);

  return estimateMIClassif(xData, yData, nSamples, nFeatures, k);
}

/**
 * Estimate mutual information between each feature and a continuous target variable.
 *
 * Mutual information (MI) measures the dependency between variables.
 * It equals zero iff two random variables are independent, and higher values
 * mean higher dependency. This function uses the KSG nearest-neighbor estimator.
 *
 * Can be used as a `scoreFunc` for `SelectKBest`.
 *
 * @param X - Feature matrix of shape [n_samples, n_features]
 * @param y - Target vector of shape [n_samples] (continuous values)
 * @param options - Optional parameters
 * @param options.nNeighbors - Number of neighbors for MI estimation (default: 3)
 * @param options.randomState - Seed for noise added to break ties (default: 0)
 * @returns Array of MI scores, one per feature (in nats, not bits)
 *
 * @example
 * ```ts
 * import { mutual_info_regression, SelectKBest } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 0.5], [2, 1.5], [3, 2.5], [4, 3.5]]);
 * const y = tensor([1.1, 2.1, 3.1, 4.1]);
 * const scores = mutual_info_regression(X, y);
 *
 * // Use with SelectKBest
 * const skb = new SelectKBest({ scoreFunc: mutual_info_regression, k: 1 });
 * skb.fit(X, y);
 * ```
 * @deprecated Prefer {@link mutualInfoRegression}.
 */
export function mutual_info_regression(
  X: Tensor,
  y: Tensor,
  options: { nNeighbors?: number; randomState?: number } = {}
): number[] {
  if (X.dtype === "string") {
    throw new DTypeError("mutual_info_regression requires numeric features");
  }
  if (y.dtype === "string") {
    throw new DTypeError("mutual_info_regression requires numeric target");
  }
  assert2D(X, "X");
  const [nSamples, nFeatures] = getShape2D(X);

  if (y.ndim !== 1 || (y.shape[0] ?? -1) !== nSamples) {
    throw new InvalidParameterError(
      `y must be a 1D tensor with ${nSamples} elements`,
      "y",
      y.shape
    );
  }

  const k = options.nNeighbors ?? 3;
  if (!Number.isInteger(k) || k < 1) {
    throw new InvalidParameterError("nNeighbors must be a positive integer", "nNeighbors", k);
  }
  if (k >= nSamples) {
    throw new InvalidParameterError(
      `nNeighbors (${k}) must be less than n_samples (${nSamples})`,
      "nNeighbors",
      k
    );
  }

  const xData = extractFeatureData(X, nSamples, nFeatures);
  const yData = extractTargetData(y, nSamples);

  // Add small noise to break ties in both X and y
  addNoise(xData, nSamples, nFeatures, options.randomState ?? 0);
  addNoise(yData, nSamples, 1, (options.randomState ?? 0) + 1);

  return estimateMIRegression(xData, yData, nSamples, nFeatures, k);
}

/** Canonical camelCase alias of {@link mutual_info_classif}. */
export const mutualInfoClassif = mutual_info_classif;
/** Canonical camelCase alias of {@link mutual_info_regression}. */
export const mutualInfoRegression = mutual_info_regression;
