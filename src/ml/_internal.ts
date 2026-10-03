/**
 * Numeric helpers shared by several `deepbox/ml` estimators.
 * This file is not exported from the public API.
 *
 * @internal
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError } from "../core";

/** What `cloneEstimator` needs from an estimator. */
type CloneableEstimator = {
  getParams(): Record<string, unknown>;
  setParams(params: Record<string, unknown>): unknown;
  clone?(): unknown;
};

/**
 * Fresh unfitted copy of an estimator, optionally with parameters overridden.
 *
 * An estimator with a `clone()` method is cloned with it and the overrides are applied with
 * `setParams`. Any other estimator is rebuilt through its constructor from `getParams()` merged
 * with the overrides. Every meta-estimator that fits copies of its base estimator uses this
 * function, so they all accept and reject the same estimators.
 *
 * @param estimator - Estimator to copy
 * @param owner - Name of the caller, used in the error message
 * @param overrides - Parameters that replace the ones of `estimator`
 * @throws {InvalidParameterError} If a parameter override is invalid, or the estimator can be
 *   neither cloned nor rebuilt from its parameters
 *
 * @internal
 */
export function cloneEstimator<T extends CloneableEstimator>(
  estimator: T,
  owner: string,
  overrides: Record<string, unknown> = {}
): T {
  const hasOverrides = Object.keys(overrides).length > 0;
  if (typeof estimator.clone === "function") {
    const cloned = estimator.clone() as T;
    if (hasOverrides) cloned.setParams(overrides);
    return cloned;
  }
  const EstimatorClass = estimator.constructor as new (params: Record<string, unknown>) => T;
  try {
    return new EstimatorClass({ ...estimator.getParams(), ...overrides });
  } catch (cause) {
    if (hasOverrides && cause instanceof InvalidParameterError) throw cause;
    throw new InvalidParameterError(
      `${owner} cannot clone the estimator: implement clone() or make its constructor accept the object returned by getParams()`,
      "estimator",
      estimator,
      { cause }
    );
  }
}

/**
 * Value of rank `k` (0-based) in `a`. Partially reorders `a` in place.
 * Average O(n); `a` must not contain NaN.
 *
 * @internal
 */
export function kthSmallest(a: Float64Array, k: number): number {
  let lo = 0;
  let hi = a.length - 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1] as number;
    let i = lo;
    let j = hi;
    while (i <= j) {
      while ((a[i] as number) < pivot) i++;
      while ((a[j] as number) > pivot) j--;
      if (i <= j) {
        const t = a[i] as number;
        a[i] = a[j] as number;
        a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k] as number;
  }
  return a[k] as number;
}

/**
 * Coefficient of determination of one-dimensional predictions, with the conventions of
 * `sklearn.metrics.r2_score`: a constant target scores 1 for a perfect fit and 0 otherwise.
 *
 * @param yTrue - True targets
 * @param yPred - Predictions, same length as `yTrue`
 *
 * @internal
 */
export function r2Score(yTrue: ArrayLike<number>, yPred: ArrayLike<number>): number {
  const n = yTrue.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += yTrue[i] as number;
  mean /= n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const t = yTrue[i] as number;
    const r = t - (yPred[i] as number);
    const c = t - mean;
    ssRes += r * r;
    ssTot += c * c;
  }
  return ssTot === 0 ? (ssRes === 0 ? 1 : 0) : 1 - ssRes / ssTot;
}

/**
 * Eigen-decomposition of a real symmetric matrix with the cyclic Jacobi method.
 *
 * @param A - Row-major (n x n) symmetric matrix; it is not modified
 * @param n - Matrix size
 * @returns Eigenvalues in descending order and the matching eigenvectors as the
 *   columns of a row-major (n x n) matrix (`vectors[i * n + j]` is component `i` of eigenvector `j`)
 *
 * @internal
 */
export function jacobiEigenSymmetric(
  A: Float64Array,
  n: number
): { values: Float64Array; vectors: Float64Array } {
  const a = Float64Array.from(A);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    let total = 0;
    for (let p = 0; p < n; p++) {
      total += (a[p * n + p] ?? 0) ** 2;
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q] ?? 0;
        off += apq * apq;
        total += 2 * apq * apq;
      }
    }
    if (off === 0 || off <= 1e-32 * total) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q] ?? 0;
        if (apq === 0) continue;
        const app = a[p * n + p] ?? 0;
        const aqq = a[q * n + q] ?? 0;
        const theta = (aqq - app) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let r = 0; r < n; r++) {
          if (r === p || r === q) continue;
          const arp = a[r * n + p] ?? 0;
          const arq = a[r * n + q] ?? 0;
          const np = c * arp - s * arq;
          const nq = s * arp + c * arq;
          a[r * n + p] = np;
          a[p * n + r] = np;
          a[r * n + q] = nq;
          a[q * n + r] = nq;
        }
        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0;
        a[q * n + p] = 0;
        for (let r = 0; r < n; r++) {
          const vrp = v[r * n + p] ?? 0;
          const vrq = v[r * n + q] ?? 0;
          v[r * n + p] = c * vrp - s * vrq;
          v[r * n + q] = s * vrp + c * vrq;
        }
      }
    }
  }

  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((x, y) => (a[y * n + y] ?? 0) - (a[x * n + x] ?? 0));
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const src = order[j] ?? j;
    values[j] = a[src * n + src] ?? 0;
    for (let i = 0; i < n; i++) vectors[i * n + j] = v[i * n + src] ?? 0;
  }
  return { values, vectors };
}

/**
 * Digamma function psi(x) for x > 0. The argument is shifted above 10 with the
 * recurrence psi(x) = psi(x + 1) - 1 / x and the asymptotic series is applied
 * (absolute error below 1e-12).
 *
 * @internal
 */
export function digamma(x: number): number {
  if (!(x > 0)) return Number.NEGATIVE_INFINITY;
  let result = 0;
  let val = x;
  while (val < 10) {
    result -= 1 / val;
    val += 1;
  }
  const inv = 1 / val;
  const inv2 = inv * inv;
  return (
    result +
    Math.log(val) -
    0.5 * inv -
    inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))))
  );
}
