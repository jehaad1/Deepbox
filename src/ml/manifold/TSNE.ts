import { ConvergenceError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { fromDenseMatrix2D } from "../../linalg/_internal";
import type { Tensor } from "../../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../../random/random";
import { toFloat64View, validateUnsupervisedFitInputs } from "../_validation";

/**
 * Build a uniform [0, 1) generator. A seed gives a private deterministic stream;
 * without one the global Deepbox generator is used, so `setSeed` still applies.
 */
function createRng(seed: number | undefined): () => number {
  if (seed === undefined) return __random;
  const gen = new __SeededRandom(__seedToUint64(seed));
  return () => gen.next();
}

/** Entropy tolerance of the perplexity search (same as scikit-learn). */
const PERPLEXITY_TOL = 1e-5;
const PERPLEXITY_MAX_STEPS = 100;
/** Smallest allowed value of an adaptive gain. */
const MIN_GAIN = 0.01;
/** Probabilities are clamped to this value inside logarithms. */
const LOG_EPS = 2.220446049250313e-16;

/**
 * Conditional probabilities `P(j|i)` of one point from its squared distances to
 * `m` other points, calibrated so that the Shannon entropy of the row equals
 * `log(perplexity)`.
 *
 * The precision `beta = 1 / (2 sigma^2)` is found by bisection on the log scale
 * (doubling or halving until the target is bracketed). Distances are shifted by
 * their minimum before exponentiation, which leaves the probabilities unchanged
 * but keeps the normalizer at or above 1, so neither very small nor very large
 * distance scales underflow.
 *
 * @param dist - Squared distances of the point to its `m` candidates
 * @param p - Output buffer of at least `m` entries; receives the row (sums to 1)
 */
function conditionalProbabilities(
  dist: Float64Array,
  m: number,
  logPerplexity: number,
  p: Float64Array
): void {
  let dMin = Infinity;
  for (let k = 0; k < m; k++) {
    const d = dist[k] as number;
    if (d < dMin) dMin = d;
  }

  let beta = 1;
  let betaMin = -Infinity;
  let betaMax = Infinity;
  let sum = 1;
  for (let step = 0; step < PERPLEXITY_MAX_STEPS; step++) {
    sum = 0;
    let sumShiftedDist = 0;
    for (let k = 0; k < m; k++) {
      const shifted = (dist[k] as number) - dMin;
      const e = Math.exp(-shifted * beta);
      p[k] = e;
      sum += e;
      sumShiftedDist += shifted * e;
    }
    // H = log(sum) + beta * E[shifted distance] under the normalized row.
    const diff = Math.log(sum) + (beta * sumShiftedDist) / sum - logPerplexity;
    if (Math.abs(diff) <= PERPLEXITY_TOL) break;
    if (diff > 0) {
      betaMin = beta;
      beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
    } else {
      betaMax = beta;
      beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
    }
  }
  for (let k = 0; k < m; k++) p[k] = (p[k] as number) / sum;
}

/**
 * Symmetric joint probabilities of exact t-SNE, `P = (P(j|i) + P(i|j)) / (2n)`.
 *
 * @param X - Row-major `n x d` data
 * @returns Row-major `n x n` matrix with a zero diagonal that sums to 1
 * @internal
 */
export function tsneJointProbabilities(
  X: Float64Array,
  n: number,
  d: number,
  perplexity: number
): Float64Array {
  const P = new Float64Array(n * n);
  const logPerplexity = Math.log(perplexity);
  const dist = new Float64Array(n - 1);
  const row = new Float64Array(n - 1);
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let sq = 0;
      for (let f = 0; f < d; f++) {
        const diff = (X[i * d + f] as number) - (X[j * d + f] as number);
        sq += diff * diff;
      }
      dist[m++] = sq;
    }
    conditionalProbabilities(dist, m, logPerplexity, row);
    m = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i) P[i * n + j] = row[m++] as number;
    }
  }
  const scale = 1 / (2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const v = ((P[i * n + j] as number) + (P[j * n + i] as number)) * scale;
      P[i * n + j] = v;
      P[j * n + i] = v;
    }
  }
  return P;
}

/** Sparse symmetric probabilities: neighbor indices and values of every row. */
type SparseRows = { indices: Int32Array[]; values: Float64Array[] };

/** Scratch buffers of the sparse gradient, allocated once per fit. */
type ApproxWorkspace = {
  /** Maximum number of partners (P neighbors plus negatives) per row. */
  readonly stride: number;
  readonly partners: Int32Array;
  readonly count: Int32Array;
  readonly weights: Float64Array;
  readonly kernel: Float64Array;
  readonly mark: Float64Array;
  stamp: number;
};

/** Constructor options of {@link TSNE}. */
export type TSNEOptions = {
  readonly nComponents?: number;
  readonly perplexity?: number;
  readonly learningRate?: number | "auto";
  readonly nIter?: number;
  readonly earlyExaggeration?: number;
  readonly earlyExaggerationIter?: number;
  readonly randomState?: number;
  readonly minGradNorm?: number;
  /** "exact" uses full pairwise interactions; "approximate" uses sampling for large datasets. */
  readonly method?: "exact" | "approximate";
  /** Maximum samples allowed in exact mode before requiring approximate. */
  readonly maxExactSamples?: number;
  /** Number of nearest neighbors per point in approximate mode. */
  readonly approximateNeighbors?: number;
  /** Number of negative samples per point in approximate mode. */
  readonly negativeSamples?: number;
};

type ResolvedTSNEOptions = {
  readonly nComponents: number;
  readonly perplexity: number;
  readonly learningRate: number | "auto";
  readonly nIter: number;
  readonly earlyExaggeration: number;
  readonly earlyExaggerationIter: number;
  readonly randomState: number | undefined;
  readonly minGradNorm: number;
  readonly method: "exact" | "approximate";
  readonly maxExactSamples: number;
  readonly approximateNeighbors: number;
  readonly negativeSamples: number;
};

const TSNE_PARAM_NAMES: ReadonlySet<string> = new Set([
  "nComponents",
  "perplexity",
  "learningRate",
  "nIter",
  "earlyExaggeration",
  "earlyExaggerationIter",
  "randomState",
  "minGradNorm",
  "method",
  "maxExactSamples",
  "approximateNeighbors",
  "negativeSamples",
]);

/**
 * t-Distributed Stochastic Neighbor Embedding (t-SNE).
 *
 * A nonlinear dimensionality reduction technique for embedding high-dimensional
 * data into a low-dimensional space (typically 2D or 3D) for visualization.
 *
 * **Algorithm**: Exact t-SNE with an optional sparse approximation
 * - Computes pairwise affinities in high-dimensional space using a Gaussian kernel
 *   whose bandwidth is calibrated per point to the requested perplexity
 * - Computes pairwise affinities in low-dimensional space using a Student-t distribution
 * - Minimizes the KL divergence between the two distributions by gradient descent
 *   with momentum and adaptive per-coordinate gains
 *
 * **Scalability Note**:
 * Exact t-SNE is O(n^2) in time and memory. With `method: "approximate"` the input
 * affinities are restricted to the `approximateNeighbors` true nearest neighbors of
 * every point and the repulsive forces are estimated from `negativeSamples` random
 * points per sample and iteration, so every iteration costs O(n * (neighbors + negatives)).
 * Finding the neighbors is still an O(n^2 d) brute-force pass.
 *
 * t-SNE is non-parametric: it embeds exactly the samples it was fitted on and cannot
 * project new data.
 *
 * @example
 * ```ts
 * import { TSNE } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [2, 1, 0]]);
 *
 * const tsne = new TSNE({ nComponents: 2, perplexity: 2, randomState: 0 });
 * const embedding = tsne.fitTransform(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-manifold | Deepbox Manifold Learning}
 * @see van der Maaten, L.J.P.; Hinton, G.E. (2008). "Visualizing High-Dimensional Data Using t-SNE"
 */
export class TSNE {
  /** Number of dimensions in the embedding */
  private nComponents: number;

  /** Perplexity parameter (related to number of nearest neighbors) */
  private perplexity: number;

  /** Learning rate for gradient descent, or "auto" */
  private learningRate: number | "auto";

  /** Number of iterations */
  private nIter: number;

  /** Early exaggeration factor */
  private earlyExaggeration: number;

  /** Number of iterations with early exaggeration */
  private earlyExaggerationIter: number;

  /** Random seed for reproducibility */
  private randomState: number | undefined;

  /** Minimum gradient norm for convergence */
  private minGradNorm: number;

  /** Method for computing affinities and gradients */
  private method: "exact" | "approximate";

  /** Maximum samples allowed for exact mode */
  private maxExactSamples: number;

  /** Neighbor count for approximate mode */
  private approximateNeighbors: number;

  /** Negative samples per point for approximate mode */
  private negativeSamples: number;

  /** Options as given by the caller; defaults that depend on other options are derived from them. */
  private userOptions: TSNEOptions;

  /** The embedded points after fitting, row-major `n x nComponents` */
  private embedding_: Float64Array = new Float64Array(0);

  /** Number of fitted samples */
  private nSamples_ = 0;

  /** Embedding dimension used by the last fit (`setParams` may change `nComponents` afterwards) */
  private nComponentsFitted_ = 0;

  /** Final KL divergence (exact mode only) */
  private klDivergence_: number | undefined;

  /** Whether the model has been fitted */
  private fitted = false;

  /**
   * @param options.nComponents - Dimension of the embedding (default: 2)
   * @param options.perplexity - Effective number of neighbors, must be < n_samples (default: 30)
   * @param options.learningRate - Step size, or `"auto"` for `max(n_samples / earlyExaggeration / 4, 50)` (default: "auto")
   * @param options.nIter - Total number of optimization iterations, including early exaggeration (default: 1000)
   * @param options.earlyExaggeration - Multiplier of the input affinities during the first phase (default: 12)
   * @param options.earlyExaggerationIter - Length of the first phase, capped at `nIter` (default: 250)
   * @param options.randomState - Seed of the initialization and sampling; without it the global Deepbox generator is used
   * @param options.minGradNorm - Gradient norm below which a phase stops early (default: 1e-7)
   * @param options.method - `"exact"` or `"approximate"` (default: "exact")
   * @param options.maxExactSamples - Largest n_samples accepted in exact mode (default: 2000)
   * @param options.approximateNeighbors - Nearest neighbors per point in approximate mode (default: `max(5, 3 * perplexity)`)
   * @param options.negativeSamples - Random repulsion partners per point and iteration in approximate mode (default: `max(10, 2 * perplexity)`)
   */
  constructor(options: TSNEOptions = {}) {
    const resolved = TSNE.resolveOptions(options);
    this.nComponents = resolved.nComponents;
    this.perplexity = resolved.perplexity;
    this.learningRate = resolved.learningRate;
    this.nIter = resolved.nIter;
    this.earlyExaggeration = resolved.earlyExaggeration;
    this.earlyExaggerationIter = resolved.earlyExaggerationIter;
    this.randomState = resolved.randomState;
    this.minGradNorm = resolved.minGradNorm;
    this.method = resolved.method;
    this.maxExactSamples = resolved.maxExactSamples;
    this.approximateNeighbors = resolved.approximateNeighbors;
    this.negativeSamples = resolved.negativeSamples;
    this.userOptions = { ...options };
  }

  /** Apply defaults and validate every option; nothing is stored here. */
  private static resolveOptions(options: TSNEOptions): ResolvedTSNEOptions {
    const nComponents = options.nComponents ?? 2;
    const perplexity = options.perplexity ?? 30;
    const learningRate = options.learningRate ?? "auto";
    const nIter = options.nIter ?? 1000;
    const earlyExaggeration = options.earlyExaggeration ?? 12;
    const earlyExaggerationRequested = options.earlyExaggerationIter ?? 250;
    const earlyExaggerationIter = Math.min(earlyExaggerationRequested, nIter);
    const minGradNorm = options.minGradNorm ?? 1e-7;
    const method = options.method ?? "exact";
    const maxExactSamples = options.maxExactSamples ?? 2000;
    const approximateNeighbors =
      options.approximateNeighbors ?? Math.max(5, Math.floor(perplexity * 3));
    const negativeSamples = options.negativeSamples ?? Math.max(10, Math.floor(perplexity * 2));

    if (!Number.isInteger(nComponents) || nComponents <= 0) {
      throw new InvalidParameterError("nComponents must be positive", "nComponents", nComponents);
    }
    if (!Number.isFinite(perplexity) || perplexity <= 0) {
      throw new InvalidParameterError("perplexity must be positive", "perplexity", perplexity);
    }
    if (
      learningRate !== "auto" &&
      (typeof learningRate !== "number" || !Number.isFinite(learningRate) || learningRate <= 0)
    ) {
      throw new InvalidParameterError(
        "learningRate must be positive or 'auto'",
        "learningRate",
        learningRate
      );
    }
    if (!Number.isInteger(nIter) || nIter <= 0) {
      throw new InvalidParameterError("nIter must be a positive integer", "nIter", nIter);
    }
    if (!Number.isFinite(earlyExaggeration) || earlyExaggeration <= 0) {
      throw new InvalidParameterError(
        "earlyExaggeration must be positive",
        "earlyExaggeration",
        earlyExaggeration
      );
    }
    if (!Number.isInteger(earlyExaggerationRequested) || earlyExaggerationRequested < 0) {
      throw new InvalidParameterError(
        "earlyExaggerationIter must be an integer >= 0",
        "earlyExaggerationIter",
        earlyExaggerationRequested
      );
    }
    if (!Number.isFinite(minGradNorm) || minGradNorm <= 0) {
      throw new InvalidParameterError("minGradNorm must be positive", "minGradNorm", minGradNorm);
    }
    if (method !== "exact" && method !== "approximate") {
      throw new InvalidParameterError("method must be 'exact' or 'approximate'", "method", method);
    }
    if (!Number.isInteger(maxExactSamples) || maxExactSamples <= 0) {
      throw new InvalidParameterError(
        "maxExactSamples must be a positive integer",
        "maxExactSamples",
        maxExactSamples
      );
    }
    if (!Number.isInteger(approximateNeighbors) || approximateNeighbors <= 0) {
      throw new InvalidParameterError(
        "approximateNeighbors must be a positive integer",
        "approximateNeighbors",
        approximateNeighbors
      );
    }
    if (!Number.isInteger(negativeSamples) || negativeSamples <= 0) {
      throw new InvalidParameterError(
        "negativeSamples must be a positive integer",
        "negativeSamples",
        negativeSamples
      );
    }
    if (options.randomState !== undefined && !Number.isFinite(options.randomState)) {
      throw new InvalidParameterError(
        "randomState must be a finite number",
        "randomState",
        options.randomState
      );
    }
    return {
      nComponents,
      perplexity,
      learningRate,
      nIter,
      earlyExaggeration,
      earlyExaggerationIter,
      randomState: options.randomState,
      minGradNorm,
      method,
      maxExactSamples,
      approximateNeighbors,
      negativeSamples,
    };
  }

  /**
   * Random initialization: independent normal coordinates with standard
   * deviation 1e-4, as in scikit-learn's `init="random"`.
   */
  private initializeEmbedding(n: number, rng: () => number): Float64Array {
    const Y = new Float64Array(n * this.nComponents);
    for (let i = 0; i < Y.length; i++) {
      const u1 = 1 - rng(); // in (0, 1], keeps log finite
      const u2 = rng();
      Y[i] = 1e-4 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return Y;
  }

  /**
   * The `k` nearest neighbors of every row (brute force), nearest first.
   * Distances are squared Euclidean.
   */
  private nearestNeighbors(
    X: Float64Array,
    n: number,
    d: number,
    k: number
  ): { indices: Int32Array; dist: Float64Array } {
    const indices = new Int32Array(n * k);
    const dist = new Float64Array(n * k);
    for (let i = 0; i < n; i++) {
      let count = 0;
      const base = i * k;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        let sq = 0;
        for (let f = 0; f < d; f++) {
          const diff = (X[i * d + f] as number) - (X[j * d + f] as number);
          sq += diff * diff;
        }
        if (count === k && sq >= (dist[base + k - 1] as number)) continue;
        let pos = count < k ? count : k - 1;
        while (pos > 0 && (dist[base + pos - 1] as number) > sq) {
          dist[base + pos] = dist[base + pos - 1] as number;
          indices[base + pos] = indices[base + pos - 1] as number;
          pos--;
        }
        dist[base + pos] = sq;
        indices[base + pos] = j;
        if (count < k) count++;
      }
    }
    return { indices, dist };
  }

  /**
   * Sparse symmetric joint probabilities on the k-nearest-neighbor graph.
   * A pair appears in both rows whenever either point lists the other, so the
   * attractive forces are symmetric.
   */
  private sparseJointProbabilities(X: Float64Array, n: number, d: number, k: number): SparseRows {
    const { indices, dist } = this.nearestNeighbors(X, n, d, k);
    const logPerplexity = Math.log(this.perplexity);
    const maps: Array<Map<number, number>> = [];
    for (let i = 0; i < n; i++) maps.push(new Map<number, number>());
    const scale = 1 / (2 * n);
    const p = new Float64Array(k);
    for (let i = 0; i < n; i++) {
      conditionalProbabilities(dist.subarray(i * k, (i + 1) * k), k, logPerplexity, p);
      for (let t = 0; t < k; t++) {
        const j = indices[i * k + t] as number;
        const v = (p[t] as number) * scale;
        const mi = maps[i] as Map<number, number>;
        const mj = maps[j] as Map<number, number>;
        mi.set(j, (mi.get(j) ?? 0) + v);
        mj.set(i, (mj.get(i) ?? 0) + v);
      }
    }
    const rows: SparseRows = { indices: [], values: [] };
    for (let i = 0; i < n; i++) {
      const map = maps[i] as Map<number, number>;
      const idx = new Int32Array(map.size);
      const val = new Float64Array(map.size);
      let t = 0;
      for (const [j, v] of map) {
        idx[t] = j;
        val[t] = v;
        t++;
      }
      rows.indices.push(idx);
      rows.values.push(val);
    }
    return rows;
  }

  /**
   * Gradient of the KL divergence for exact t-SNE, written into `grad`.
   *
   * Uses the symmetry of P and Q to visit every pair once. `QBuf` is scratch
   * space for the unnormalized Student-t kernel values (upper triangle).
   */
  private exactGradient(
    P: Float64Array,
    exaggeration: number,
    Y: Float64Array,
    n: number,
    QBuf: Float64Array,
    grad: Float64Array
  ): void {
    const dims = this.nComponents;
    let sumQ = 0;
    for (let i = 0; i < n; i++) {
      const bi = i * dims;
      for (let j = i + 1; j < n; j++) {
        const bj = j * dims;
        let sq = 0;
        for (let k = 0; k < dims; k++) {
          const diff = (Y[bi + k] as number) - (Y[bj + k] as number);
          sq += diff * diff;
        }
        const q = 1 / (1 + sq);
        QBuf[i * n + j] = q;
        sumQ += 2 * q;
      }
    }
    const invSumQ = 1 / sumQ;
    grad.fill(0);
    for (let i = 0; i < n; i++) {
      const bi = i * dims;
      for (let j = i + 1; j < n; j++) {
        const bj = j * dims;
        const q = QBuf[i * n + j] as number;
        const mult = 4 * ((P[i * n + j] as number) * exaggeration - q * invSumQ) * q;
        for (let k = 0; k < dims; k++) {
          const g = mult * ((Y[bi + k] as number) - (Y[bj + k] as number));
          grad[bi + k] = (grad[bi + k] as number) + g;
          grad[bj + k] = (grad[bj + k] as number) - g;
        }
      }
    }
  }

  /** KL(P || Q) of the exact model, summed over all ordered pairs. */
  private exactKL(P: Float64Array, Y: Float64Array, n: number): number {
    const dims = this.nComponents;
    let sumQ = 0;
    const Q = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let sq = 0;
        for (let k = 0; k < dims; k++) {
          const diff = (Y[i * dims + k] as number) - (Y[j * dims + k] as number);
          sq += diff * diff;
        }
        const q = 1 / (1 + sq);
        Q[i * n + j] = q;
        sumQ += 2 * q;
      }
    }
    let kl = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const p = P[i * n + j] as number;
        const q = (Q[i * n + j] as number) / sumQ;
        kl += 2 * p * Math.log(Math.max(p, LOG_EPS) / Math.max(q, LOG_EPS));
      }
    }
    return kl;
  }

  /**
   * Gradient for the sparse approximation, written into `grad`.
   *
   * Row `i` sums the attraction over its P neighbors and the repulsion over
   * those neighbors plus up to `negativeCount` random points. Negatives are
   * weighted by `candidates / sampled` so that both the repulsive force and the
   * normalizer are unbiased estimates of their full sums.
   */
  private approximateGradient(
    P: SparseRows,
    exaggeration: number,
    Y: Float64Array,
    n: number,
    negativeCount: number,
    rng: () => number,
    ws: ApproxWorkspace,
    grad: Float64Array
  ): void {
    const dims = this.nComponents;
    const { stride, partners, count, weights, kernel, mark } = ws;
    // Pass 1: choose partners and accumulate the normalizer.
    let sumQ = 0;
    for (let i = 0; i < n; i++) {
      const nbr = P.indices[i] as Int32Array;
      const stamp = ++ws.stamp;
      mark[i] = stamp;
      const base = i * stride;
      for (let t = 0; t < nbr.length; t++) {
        const j = nbr[t] as number;
        mark[j] = stamp;
        partners[base + t] = j;
      }
      const candidates = n - 1 - nbr.length;
      const wantNeg = Math.min(negativeCount, candidates);
      let got = 0;
      let attempts = 0;
      const maxAttempts = wantNeg * 12 + 200;
      while (got < wantNeg && attempts < maxAttempts) {
        const idx = __randomBelow(rng, n);
        attempts++;
        if (mark[idx] !== stamp) {
          mark[idx] = stamp;
          partners[base + nbr.length + got] = idx;
          got++;
        }
      }
      for (let c = 0; got < wantNeg && c < n; c++) {
        if (mark[c] !== stamp) {
          mark[c] = stamp;
          partners[base + nbr.length + got] = c;
          got++;
        }
      }
      const total = nbr.length + got;
      count[i] = total;
      const negWeight = got > 0 ? candidates / got : 0;
      const bi = i * dims;
      for (let t = 0; t < total; t++) {
        const bj = (partners[base + t] as number) * dims;
        let sq = 0;
        for (let k = 0; k < dims; k++) {
          const diff = (Y[bi + k] as number) - (Y[bj + k] as number);
          sq += diff * diff;
        }
        const qv = 1 / (1 + sq);
        const wt = t < nbr.length ? 1 : negWeight;
        kernel[base + t] = qv;
        weights[base + t] = wt;
        sumQ += wt * qv;
      }
    }

    // Pass 2: gradient rows.
    const invSumQ = 1 / sumQ;
    grad.fill(0);
    for (let i = 0; i < n; i++) {
      const pv = P.values[i] as Float64Array;
      const base = i * stride;
      const bi = i * dims;
      const total = count[i] as number;
      for (let t = 0; t < total; t++) {
        const qu = kernel[base + t] as number;
        const pij = t < pv.length ? (pv[t] as number) * exaggeration : 0;
        const mult = 4 * (weights[base + t] as number) * (pij - qu * invSumQ) * qu;
        const bj = (partners[base + t] as number) * dims;
        for (let k = 0; k < dims; k++) {
          grad[bi + k] =
            (grad[bi + k] as number) + mult * ((Y[bi + k] as number) - (Y[bj + k] as number));
        }
      }
    }
  }

  /**
   * Fit the t-SNE model and return the embedding.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @returns Low-dimensional embedding of shape (n_samples, n_components)
   * @throws {InvalidParameterError} If there are fewer than 4 samples, `perplexity >= n_samples`,
   *   or exact mode is asked to handle more than `maxExactSamples` samples
   * @throws {ConvergenceError} If the optimization produces non-finite values
   */
  fitTransform(X: Tensor): Tensor {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    if (nSamples < 4) {
      throw new InvalidParameterError("t-SNE requires at least 4 samples", "nSamples", nSamples);
    }
    if (this.perplexity >= nSamples) {
      throw new InvalidParameterError(
        `perplexity must be less than n_samples; received perplexity=${this.perplexity}, n_samples=${nSamples}`,
        "perplexity",
        this.perplexity
      );
    }
    if (this.method === "exact" && nSamples > this.maxExactSamples) {
      throw new InvalidParameterError(
        `Exact t-SNE is O(n^2) and limited to n_samples <= ${this.maxExactSamples}; received n_samples=${nSamples}. Use method="approximate" or increase maxExactSamples.`,
        "nSamples",
        nSamples
      );
    }

    const data = toFloat64View(X);
    const rng = createRng(this.randomState);
    const dims = this.nComponents;

    const useApproximate = this.method === "approximate";
    const neighborCount = Math.min(this.approximateNeighbors, nSamples - 1);
    if (useApproximate && neighborCount < 2) {
      throw new InvalidParameterError(
        "approximateNeighbors must be at least 2 for approximate mode",
        "approximateNeighbors",
        neighborCount
      );
    }
    if (useApproximate && this.perplexity >= neighborCount) {
      throw new InvalidParameterError(
        `perplexity must be less than approximateNeighbors; received perplexity=${this.perplexity}, approximateNeighbors=${neighborCount}`,
        "perplexity",
        this.perplexity
      );
    }

    // Joint probabilities P
    const PExact = useApproximate
      ? null
      : tsneJointProbabilities(data, nSamples, nFeatures, this.perplexity);
    const PSparse = useApproximate
      ? this.sparseJointProbabilities(data, nSamples, nFeatures, neighborCount)
      : null;
    const QBuf = PExact ? new Float64Array(nSamples * nSamples) : null;
    const negatives = Math.min(this.negativeSamples, Math.max(0, nSamples - 1 - neighborCount));
    let workspace: ApproxWorkspace | null = null;
    if (PSparse) {
      let maxRow = 0;
      for (const idx of PSparse.indices) maxRow = Math.max(maxRow, idx.length);
      const stride = maxRow + negatives;
      workspace = {
        stride,
        partners: new Int32Array(nSamples * stride),
        count: new Int32Array(nSamples),
        weights: new Float64Array(nSamples * stride),
        kernel: new Float64Array(nSamples * stride),
        mark: new Float64Array(nSamples).fill(-1),
        stamp: 0,
      };
    }

    const learningRate =
      this.learningRate === "auto"
        ? Math.max(nSamples / this.earlyExaggeration / 4, 50)
        : this.learningRate;

    const Y = this.initializeEmbedding(nSamples, rng);
    const update = new Float64Array(Y.length);
    const gains = new Float64Array(Y.length).fill(1);
    const grad = new Float64Array(Y.length);
    const center = new Float64Array(dims);

    const momentum = 0.5;
    const finalMomentum = 0.8;
    let exaggerationUntil = this.earlyExaggerationIter;

    for (let iter = 0; iter < this.nIter; iter++) {
      const exaggerating = iter < exaggerationUntil;
      const exaggeration = exaggerating ? this.earlyExaggeration : 1;

      if (PExact && QBuf) {
        this.exactGradient(PExact, exaggeration, Y, nSamples, QBuf, grad);
      } else if (PSparse && workspace) {
        this.approximateGradient(
          PSparse,
          exaggeration,
          Y,
          nSamples,
          negatives,
          rng,
          workspace,
          grad
        );
      }

      let gradNorm = 0;
      for (let i = 0; i < grad.length; i++) gradNorm += (grad[i] as number) ** 2;
      gradNorm = Math.sqrt(gradNorm);
      if (!Number.isFinite(gradNorm)) {
        throw new ConvergenceError(
          `t-SNE diverged at iteration ${iter}: the gradient is not finite. Try a smaller learningRate.`,
          { iterations: iter }
        );
      }

      // Momentum update with adaptive gains (delta-bar-delta).
      const currentMomentum = exaggerating ? momentum : finalMomentum;
      for (let i = 0; i < Y.length; i++) {
        const g = grad[i] as number;
        const u = update[i] as number;
        let gain = u * g < 0 ? (gains[i] as number) + 0.2 : (gains[i] as number) * 0.8;
        if (gain < MIN_GAIN) gain = MIN_GAIN;
        gains[i] = gain;
        const nu = currentMomentum * u - learningRate * gain * g;
        update[i] = nu;
        Y[i] = (Y[i] as number) + nu;
      }

      // Keep the embedding centered.
      center.fill(0);
      for (let i = 0; i < nSamples; i++) {
        for (let k = 0; k < dims; k++) {
          center[k] = (center[k] as number) + (Y[i * dims + k] as number);
        }
      }
      for (let k = 0; k < dims; k++) center[k] = (center[k] as number) / nSamples;
      for (let i = 0; i < nSamples; i++) {
        for (let k = 0; k < dims; k++) {
          Y[i * dims + k] = (Y[i * dims + k] as number) - (center[k] as number);
        }
      }

      if (gradNorm < this.minGradNorm) {
        // Converged: finish the exaggeration phase early, or stop in the final phase.
        if (exaggerating) exaggerationUntil = iter + 1;
        else break;
      }
    }

    this.klDivergence_ = PExact ? this.exactKL(PExact, Y, nSamples) : undefined;
    this.embedding_ = Y;
    this.nSamples_ = nSamples;
    this.nComponentsFitted_ = dims;
    this.fitted = true;

    return fromDenseMatrix2D(nSamples, dims, Float64Array.from(Y));
  }

  /**
   * Fit the model (same as fitTransform for t-SNE).
   */
  fit(X: Tensor): this {
    this.fitTransform(X);
    return this;
  }

  /**
   * Return the fitted embedding. For t-SNE, transform is equivalent to
   * returning the already-computed embedding (t-SNE is non-parametric and cannot
   * embed new samples).
   *
   * @param X - Optional; if given it must have the same number of rows as the data passed to `fit`
   * @returns Low-dimensional embedding of shape (n_samples, n_components)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If `X` has a different number of samples than the fitted data
   */
  transform(X?: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("TSNE must be fitted before transform");
    }
    if (X !== undefined) {
      const rows = X.shape[0] ?? 0;
      if (X.ndim !== 2 || rows !== this.nSamples_) {
        throw new ShapeError(
          `TSNE cannot embed new samples; transform only returns the embedding of the ` +
            `${this.nSamples_} fitted samples, but X has shape [${X.shape.join(", ")}]`
        );
      }
    }
    return fromDenseMatrix2D(
      this.nSamples_,
      this.nComponentsFitted_,
      Float64Array.from(this.embedding_)
    );
  }

  /**
   * Embedding of the fitted data, shape (n_samples, n_components).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get embedding(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("TSNE must be fitted before accessing embedding");
    }
    return fromDenseMatrix2D(
      this.nSamples_,
      this.nComponentsFitted_,
      Float64Array.from(this.embedding_)
    );
  }

  /**
   * Get the embedding.
   *
   * @deprecated Use {@link TSNE.embedding}, which matches the other manifold estimators.
   * @throws {NotFittedError} If the model has not been fitted
   */
  get embeddingResult(): Tensor {
    return this.embedding;
  }

  /**
   * KL divergence between the input and embedding affinities after the last fit.
   * Only available for `method: "exact"`; `undefined` in approximate mode.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get klDivergence(): number | undefined {
    if (!this.fitted) {
      throw new NotFittedError("TSNE must be fitted before accessing klDivergence");
    }
    return this.klDivergence_;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nComponents: this.nComponents,
      perplexity: this.perplexity,
      learningRate: this.learningRate,
      nIter: this.nIter,
      earlyExaggeration: this.earlyExaggeration,
      earlyExaggerationIter: this.earlyExaggerationIter,
      randomState: this.randomState,
      minGradNorm: this.minGradNorm,
      method: this.method,
      maxExactSamples: this.maxExactSamples,
      approximateNeighbors: this.approximateNeighbors,
      negativeSamples: this.negativeSamples,
    };
  }

  /**
   * Update hyperparameters. The model must be refitted afterwards; the embedding of the last
   * fit stays available until then.
   *
   * Defaults that depend on `perplexity` (`approximateNeighbors`, `negativeSamples`) are derived
   * again unless they were given explicitly.
   *
   * @param params - Any subset of the constructor options
   * @returns this
   * @throws {InvalidParameterError} On an unknown or invalid parameter (the estimator is left unchanged)
   */
  setParams(params: Record<string, unknown>): this {
    const merged: Record<string, unknown> = { ...this.userOptions };
    for (const [key, value] of Object.entries(params)) {
      if (!TSNE_PARAM_NAMES.has(key)) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
      merged[key] = value;
    }
    const options = merged as TSNEOptions;
    const resolved = TSNE.resolveOptions(options);
    this.nComponents = resolved.nComponents;
    this.perplexity = resolved.perplexity;
    this.learningRate = resolved.learningRate;
    this.nIter = resolved.nIter;
    this.earlyExaggeration = resolved.earlyExaggeration;
    this.earlyExaggerationIter = resolved.earlyExaggerationIter;
    this.randomState = resolved.randomState;
    this.minGradNorm = resolved.minGradNorm;
    this.method = resolved.method;
    this.maxExactSamples = resolved.maxExactSamples;
    this.approximateNeighbors = resolved.approximateNeighbors;
    this.negativeSamples = resolved.negativeSamples;
    this.userOptions = options;
    return this;
  }
}
