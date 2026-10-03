/**
 * Mutual information scoring functions for feature selection.
 *
 * Implements k-nearest-neighbor based mutual information estimation
 * following Kraskov, Stögbauer & Grassberger (2004) and, for a continuous
 * variable against a discrete one, Ross (2014). The estimators, the feature
 * scaling and the tie-breaking noise follow `sklearn.feature_selection`.
 *
 * @module preprocess/mutual_info
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import { DataValidationError, DTypeError, InvalidParameterError } from "../core/errors";
import type { Tensor } from "../ndarray";
import { assert2D, assertNumericTensor, getShape2D, getStrides2D } from "./_internal";

/** Options shared by {@link mutual_info_classif} and {@link mutual_info_regression}. */
export type MutualInfoOptions = {
  /** Number of neighbors used by the estimator (default 3). */
  nNeighbors?: number;
  /** Seed of the small noise added to continuous variables to break ties (default 0). */
  randomState?: number;
  /**
   * Which features are discrete: `true` for all, `false` for none (default), a boolean mask with
   * one entry per feature, or an array of feature indices. Discrete features are not scaled and
   * get no noise.
   */
  discreteFeatures?: boolean | readonly boolean[] | readonly number[];
};

/**
 * Digamma (psi) function: recurrence up to x >= 8, then the asymptotic series.
 * Absolute error about 1e-14 or less for positive arguments (checked against SciPy).
 */
function digamma(x: number): number {
  let result = 0;
  while (x < 8) {
    result -= 1 / x;
    x += 1;
  }
  const invX = 1 / x;
  const invX2 = invX * invX;
  // ln x - 1/(2x) - sum B_2n / (2n x^2n)
  const series =
    invX2 *
    (1 / 12 -
      invX2 *
        (1 / 120 -
          invX2 * (1 / 252 - invX2 * (1 / 240 - invX2 * (1 / 132 - invX2 * (691 / 32760))))));
  result += Math.log(x) - 0.5 * invX - series;
  return result;
}

/** Seeded generator of standard normal values (mulberry32 + Box-Muller). */
function createNormalRng(seed: number): () => number {
  let state = (Math.trunc(seed) ^ Math.trunc(seed / 2 ** 32)) >>> 0;
  const uniform = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare: number | undefined;
  return () => {
    if (spare !== undefined) {
      const s = spare;
      spare = undefined;
      return s;
    }
    let u = uniform();
    while (u === 0) u = uniform();
    const v = uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

/**
 * Center and scale a continuous variable to unit population standard deviation
 * (a constant variable keeps scale 1), then add `1e-10 * max(1, mean|x|)` times
 * standard normal noise, as scikit-learn does before the nearest-neighbor search.
 */
function scaleAndJitter(values: Float64Array, rng: () => number): void {
  const n = values.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += values[i] as number;
  mean /= n;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = (values[i] as number) - mean;
    values[i] = d;
    ss += d * d;
  }
  const std = Math.sqrt(ss / n);
  const scale = std > 0 ? 1 / std : 1;
  let meanAbs = 0;
  for (let i = 0; i < n; i++) {
    const v = (values[i] as number) * scale;
    values[i] = v;
    meanAbs += Math.abs(v);
  }
  meanAbs /= n;
  const noise = 1e-10 * Math.max(1, meanAbs);
  for (let i = 0; i < n; i++) values[i] = (values[i] as number) + noise * rng();
}

/**
 * Number of values in the ascending array `sorted` within `radius` of `center`,
 * the point itself included. The distance must be strictly below `radius`; for
 * a zero radius, equal values count (scikit-learn queries with the radius just
 * below the k-th neighbor distance, which is the same thing).
 */
function countWithin(sorted: Float64Array, center: number, radius: number): number {
  const closed = radius === 0;
  // First index whose value is not too far to the left.
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    const d = center - (sorted[mid] as number);
    if (closed ? d <= radius : d < radius) hi = mid;
    else lo = mid + 1;
  }
  const first = lo;
  // First index whose value is too far to the right.
  hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    const d = (sorted[mid] as number) - center;
    if (closed ? d > radius : d >= radius) hi = mid;
    else lo = mid + 1;
  }
  return lo - first;
}

/**
 * MI between a continuous variable `c` and a discrete variable `d`
 * (sklearn `_compute_mi_cd`). Points whose label occurs once are ignored.
 */
function mutualInfoContinuousDiscrete(c: Float64Array, d: Float64Array, k: number): number {
  const n = c.length;
  const groups = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const label = d[i] as number;
    let g = groups.get(label);
    if (!g) {
      g = [];
      groups.set(label, g);
    }
    g.push(i);
  }

  const radius = new Float64Array(n);
  const keptIdx: number[] = [];
  let sumPsiK = 0;
  let sumPsiCount = 0;
  for (const members of groups.values()) {
    const count = members.length;
    if (count < 2) continue;
    const kk = Math.min(k, count - 1);
    const order = members.slice().sort((a, b) => (c[a] as number) - (c[b] as number));
    const vals = Float64Array.from(order, (i) => c[i] as number);
    for (let p = 0; p < count; p++) {
      let l = p - 1;
      let r = p + 1;
      let dk = 0;
      for (let step = 0; step < kk; step++) {
        const dl = l >= 0 ? (vals[p] as number) - (vals[l] as number) : Number.POSITIVE_INFINITY;
        const dr = r < count ? (vals[r] as number) - (vals[p] as number) : Number.POSITIVE_INFINITY;
        if (dl <= dr) {
          dk = dl;
          l--;
        } else {
          dk = dr;
          r++;
        }
      }
      const i = order[p] as number;
      radius[i] = dk;
      keptIdx.push(i);
    }
    sumPsiK += count * digamma(kk);
    sumPsiCount += count * digamma(count);
  }

  const m = keptIdx.length;
  if (m === 0) return 0;
  const sortedKept = Float64Array.from(keptIdx, (i) => c[i] as number).sort();
  let sumPsiM = 0;
  for (const i of keptIdx) {
    sumPsiM += digamma(countWithin(sortedKept, c[i] as number, radius[i] as number));
  }
  const mi = digamma(m) + sumPsiK / m - sumPsiCount / m - sumPsiM / m;
  return Math.max(0, mi);
}

/**
 * MI between two continuous variables (KSG estimator, algorithm 1, Chebyshev
 * distance; sklearn `_compute_mi_cc`).
 */
function mutualInfoContinuousContinuous(x: Float64Array, y: Float64Array, k: number): number {
  const n = x.length;
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => (x[a] as number) - (x[b] as number)
  );
  const xs = Float64Array.from(order, (i) => x[i] as number);
  const ysByX = Float64Array.from(order, (i) => y[i] as number);
  const ys = Float64Array.from(y).sort();

  const kBest = new Float64Array(k);
  let sumPsiX = 0;
  let sumPsiY = 0;
  for (let p = 0; p < n; p++) {
    const xp = xs[p] as number;
    const yp = ysByX[p] as number;
    let filled = 0;
    let l = p - 1;
    let r = p + 1;
    // Walk outwards in x order; the x-gap is a lower bound of the joint distance,
    // so stop once it reaches the current k-th best distance.
    while (l >= 0 || r < n) {
      const dxl = l >= 0 ? xp - (xs[l] as number) : Number.POSITIVE_INFINITY;
      const dxr = r < n ? (xs[r] as number) - xp : Number.POSITIVE_INFINITY;
      let dx: number;
      let j: number;
      if (dxl <= dxr) {
        dx = dxl;
        j = l--;
      } else {
        dx = dxr;
        j = r++;
      }
      if (filled === k && dx >= (kBest[k - 1] as number)) break;
      const dy = Math.abs(yp - (ysByX[j] as number));
      const dist = dx > dy ? dx : dy;
      if (filled < k) {
        let pos = filled++;
        while (pos > 0 && (kBest[pos - 1] as number) > dist) {
          kBest[pos] = kBest[pos - 1] as number;
          pos--;
        }
        kBest[pos] = dist;
      } else if (dist < (kBest[k - 1] as number)) {
        let pos = k - 1;
        while (pos > 0 && (kBest[pos - 1] as number) > dist) {
          kBest[pos] = kBest[pos - 1] as number;
          pos--;
        }
        kBest[pos] = dist;
      }
    }
    const eps = kBest[k - 1] as number;
    sumPsiX += digamma(countWithin(xs, xp, eps));
    sumPsiY += digamma(countWithin(ys, yp, eps));
  }
  return Math.max(0, digamma(k) + digamma(n) - sumPsiX / n - sumPsiY / n);
}

/** MI between two discrete variables from their contingency table (nats). */
function mutualInfoDiscreteDiscrete(a: Float64Array, b: Float64Array): number {
  const n = a.length;
  const aIdx = new Map<number, number>();
  const bIdx = new Map<number, number>();
  const aCount: number[] = [];
  const bCount: number[] = [];
  const joint = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const av = a[i] as number;
    const bv = b[i] as number;
    let ia = aIdx.get(av);
    if (ia === undefined) {
      ia = aCount.length;
      aIdx.set(av, ia);
      aCount.push(0);
    }
    let ib = bIdx.get(bv);
    if (ib === undefined) {
      ib = bCount.length;
      bIdx.set(bv, ib);
      bCount.push(0);
    }
    aCount[ia] = (aCount[ia] as number) + 1;
    bCount[ib] = (bCount[ib] as number) + 1;
    const key = ia * 2 ** 26 + ib;
    joint.set(key, (joint.get(key) ?? 0) + 1);
  }
  let mi = 0;
  for (const [key, nij] of joint) {
    const ia = Math.floor(key / 2 ** 26);
    const ib = key - ia * 2 ** 26;
    mi += (nij / n) * Math.log((nij * n) / ((aCount[ia] as number) * (bCount[ib] as number)));
  }
  return Math.max(0, mi);
}

function resolveDiscrete(
  spec: MutualInfoOptions["discreteFeatures"],
  nFeatures: number
): boolean[] {
  if (spec === undefined || spec === false) return new Array<boolean>(nFeatures).fill(false);
  if (spec === true) return new Array<boolean>(nFeatures).fill(true);
  if (!Array.isArray(spec)) {
    throw new InvalidParameterError(
      "discreteFeatures must be a boolean, a boolean mask or an array of feature indices",
      "discreteFeatures",
      spec
    );
  }
  const mask = new Array<boolean>(nFeatures).fill(false);
  if (spec.length === 0) return mask;
  if (typeof spec[0] === "boolean") {
    if (spec.length !== nFeatures || spec.some((v) => typeof v !== "boolean")) {
      throw new InvalidParameterError(
        `discreteFeatures mask must have ${nFeatures} boolean entries`,
        "discreteFeatures",
        spec.length
      );
    }
    return [...(spec as readonly boolean[])];
  }
  for (const idx of spec as readonly unknown[]) {
    if (typeof idx !== "number" || !Number.isInteger(idx) || idx < 0 || idx >= nFeatures) {
      throw new InvalidParameterError(
        `discreteFeatures index ${String(idx)} is out of range for ${nFeatures} features`,
        "discreteFeatures",
        idx
      );
    }
    mask[idx] = true;
  }
  return mask;
}

type Prepared = {
  nSamples: number;
  nFeatures: number;
  k: number;
  columns: Float64Array[];
  yv: Float64Array;
  discrete: boolean[];
  rng: () => number;
};

/** Validate the inputs shared by both estimators and extract dense columns. */
function prepare(X: Tensor, y: Tensor, options: MutualInfoOptions, who: string): Prepared {
  if (X.dtype === "string") {
    throw new DTypeError(`${who} requires numeric features`);
  }
  if (y.dtype === "string") {
    throw new DTypeError(`${who} requires numeric target`);
  }
  assertNumericTensor(X, "X");
  assertNumericTensor(y, "y");
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
  const seed = options.randomState ?? 0;
  if (!Number.isSafeInteger(seed)) {
    throw new InvalidParameterError("randomState must be a safe integer", "randomState", seed);
  }
  const discrete = resolveDiscrete(options.discreteFeatures, nFeatures);

  const [rs, cs] = getStrides2D(X);
  const src = X.data as ArrayLike<number | bigint>;
  const columns: Float64Array[] = [];
  for (let j = 0; j < nFeatures; j++) {
    const col = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      const v = Number(src[X.offset + i * rs + j * cs]);
      if (!Number.isFinite(v)) {
        throw new DataValidationError(`${who}: X contains NaN or infinity (row ${i}, column ${j})`);
      }
      col[i] = v;
    }
    columns.push(col);
  }

  const yv = new Float64Array(nSamples);
  const ySrc = y.data as ArrayLike<number | bigint>;
  const yStride = y.strides[0] ?? 1;
  for (let i = 0; i < nSamples; i++) {
    const v = Number(ySrc[y.offset + i * yStride]);
    if (!Number.isFinite(v)) {
      throw new DataValidationError(`${who}: y contains NaN or infinity (index ${i})`);
    }
    yv[i] = v;
  }

  return { nSamples, nFeatures, k, columns, yv, discrete, rng: createNormalRng(seed) };
}

/**
 * Estimate mutual information between each feature and a discrete target variable.
 *
 * Mutual information (MI) measures the dependency between variables.
 * It equals zero iff two random variables are independent, and higher values
 * mean higher dependency. Continuous features are scaled to unit standard
 * deviation, jittered by a tiny seeded noise to break ties, and scored with the
 * nearest-neighbor estimator of Ross (2014), as in scikit-learn. Features
 * listed in `discreteFeatures` are scored from the contingency table with the
 * target instead. Samples whose class occurs only once are ignored by the
 * continuous estimator.
 *
 * Can be used as a `scoreFunc` for `SelectKBest`.
 *
 * @param X - Feature matrix of shape [n_samples, n_features]; must be finite
 * @param y - Target vector of shape [n_samples] (discrete class labels)
 * @param options - Optional parameters
 * @param options.nNeighbors - Number of neighbors for MI estimation (default: 3, must be below n_samples)
 * @param options.randomState - Seed for noise added to break ties (default: 0)
 * @param options.discreteFeatures - Discrete features: boolean, mask or indices (default: none)
 * @returns Array of MI scores, one per feature (in nats, not bits)
 * @throws {DTypeError} If X or y is a string tensor
 * @throws {DataValidationError} If X or y contain NaN or infinity
 *
 * @example
 * ```ts
 * import { mutualInfoClassif, SelectKBest } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const y = tensor([0, 0, 1, 1]);
 * const scores = mutualInfoClassif(X, y);
 *
 * // Use with SelectKBest
 * const skb = new SelectKBest({ scoreFunc: mutualInfoClassif, k: 1 });
 * skb.fit(X, y);
 * ```
 * @deprecated Prefer {@link mutualInfoClassif}.
 */
export function mutual_info_classif(
  X: Tensor,
  y: Tensor,
  options: MutualInfoOptions = {}
): number[] {
  const { nFeatures, k, columns, yv, discrete, rng } = prepare(
    X,
    y,
    options,
    "mutual_info_classif"
  );
  // Continuous features are jittered in order, as scikit-learn draws its noise.
  for (let j = 0; j < nFeatures; j++) {
    if (!discrete[j]) scaleAndJitter(columns[j] as Float64Array, rng);
  }
  return columns.map((col, j) =>
    discrete[j] ? mutualInfoDiscreteDiscrete(col, yv) : mutualInfoContinuousDiscrete(col, yv, k)
  );
}

/**
 * Estimate mutual information between each feature and a continuous target variable.
 *
 * Mutual information (MI) measures the dependency between variables.
 * It equals zero iff two random variables are independent, and higher values
 * mean higher dependency. Continuous features and the target are scaled to unit
 * standard deviation (the joint nearest-neighbor search uses the Chebyshev
 * distance, so the scale matters), jittered by a tiny seeded noise, and scored
 * with the KSG estimator, as in scikit-learn. Features listed in
 * `discreteFeatures` are scored against the target with the continuous-discrete
 * estimator.
 *
 * Can be used as a `scoreFunc` for `SelectKBest`.
 *
 * @param X - Feature matrix of shape [n_samples, n_features]; must be finite
 * @param y - Target vector of shape [n_samples] (continuous values)
 * @param options - Optional parameters
 * @param options.nNeighbors - Number of neighbors for MI estimation (default: 3, must be below n_samples)
 * @param options.randomState - Seed for noise added to break ties (default: 0)
 * @param options.discreteFeatures - Discrete features: boolean, mask or indices (default: none)
 * @returns Array of MI scores, one per feature (in nats, not bits)
 * @throws {DTypeError} If X or y is a string tensor
 * @throws {DataValidationError} If X or y contain NaN or infinity
 *
 * @example
 * ```ts
 * import { mutualInfoRegression, SelectKBest } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 0.5], [2, 1.5], [3, 2.5], [4, 3.5]]);
 * const y = tensor([1.1, 2.1, 3.1, 4.1]);
 * const scores = mutualInfoRegression(X, y);
 *
 * // Use with SelectKBest
 * const skb = new SelectKBest({ scoreFunc: mutualInfoRegression, k: 1 });
 * skb.fit(X, y);
 * ```
 * @deprecated Prefer {@link mutualInfoRegression}.
 */
export function mutual_info_regression(
  X: Tensor,
  y: Tensor,
  options: MutualInfoOptions = {}
): number[] {
  const { nFeatures, k, columns, yv, discrete, rng } = prepare(
    X,
    y,
    options,
    "mutual_info_regression"
  );
  for (let j = 0; j < nFeatures; j++) {
    if (!discrete[j]) scaleAndJitter(columns[j] as Float64Array, rng);
  }
  scaleAndJitter(yv, rng);
  return columns.map((col, j) =>
    discrete[j]
      ? mutualInfoContinuousDiscrete(yv, col, k)
      : mutualInfoContinuousContinuous(col, yv, k)
  );
}

/**
 * Estimate mutual information between each feature and a discrete target variable.
 *
 * Mutual information (MI) measures the dependency between variables.
 * It equals zero iff two random variables are independent, and higher values
 * mean higher dependency. Continuous features are scaled to unit standard
 * deviation, jittered by a tiny seeded noise to break ties, and scored with the
 * nearest-neighbor estimator of Ross (2014), as in scikit-learn. Features
 * listed in `discreteFeatures` are scored from the contingency table with the
 * target instead. Samples whose class occurs only once are ignored by the
 * continuous estimator.
 *
 * Can be used as a `scoreFunc` for `SelectKBest`.
 *
 * @param X - Feature matrix of shape [n_samples, n_features]; must be finite
 * @param y - Target vector of shape [n_samples] (discrete class labels)
 * @param options - Optional parameters
 * @param options.nNeighbors - Number of neighbors for MI estimation (default: 3, must be below n_samples)
 * @param options.randomState - Seed for noise added to break ties (default: 0)
 * @param options.discreteFeatures - Discrete features: boolean, mask or indices (default: none)
 * @returns Array of MI scores, one per feature (in nats, not bits)
 * @throws {DTypeError} If X or y is a string tensor
 * @throws {DataValidationError} If X or y contain NaN or infinity
 *
 * @example
 * ```ts
 * import { mutualInfoClassif, SelectKBest } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const y = tensor([0, 0, 1, 1]);
 * const scores = mutualInfoClassif(X, y);
 *
 * // Use with SelectKBest
 * const skb = new SelectKBest({ scoreFunc: mutualInfoClassif, k: 1 });
 * skb.fit(X, y);
 * ```
 */
export const mutualInfoClassif = mutual_info_classif;
/**
 * Estimate mutual information between each feature and a continuous target variable.
 *
 * Mutual information (MI) measures the dependency between variables.
 * It equals zero iff two random variables are independent, and higher values
 * mean higher dependency. Continuous features and the target are scaled to unit
 * standard deviation (the joint nearest-neighbor search uses the Chebyshev
 * distance, so the scale matters), jittered by a tiny seeded noise, and scored
 * with the KSG estimator, as in scikit-learn. Features listed in
 * `discreteFeatures` are scored against the target with the continuous-discrete
 * estimator.
 *
 * Can be used as a `scoreFunc` for `SelectKBest`.
 *
 * @param X - Feature matrix of shape [n_samples, n_features]; must be finite
 * @param y - Target vector of shape [n_samples] (continuous values)
 * @param options - Optional parameters
 * @param options.nNeighbors - Number of neighbors for MI estimation (default: 3, must be below n_samples)
 * @param options.randomState - Seed for noise added to break ties (default: 0)
 * @param options.discreteFeatures - Discrete features: boolean, mask or indices (default: none)
 * @returns Array of MI scores, one per feature (in nats, not bits)
 * @throws {DTypeError} If X or y is a string tensor
 * @throws {DataValidationError} If X or y contain NaN or infinity
 *
 * @example
 * ```ts
 * import { mutualInfoRegression, SelectKBest } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 0.5], [2, 1.5], [3, 2.5], [4, 3.5]]);
 * const y = tensor([1.1, 2.1, 3.1, 4.1]);
 * const scores = mutualInfoRegression(X, y);
 *
 * // Use with SelectKBest
 * const skb = new SelectKBest({ scoreFunc: mutualInfoRegression, k: 1 });
 * skb.fit(X, y);
 * ```
 */
export const mutualInfoRegression = mutual_info_regression;
