/**
 * Mini-Batch K-Means clustering.
 *
 * @module ml/clustering/MiniBatchKMeans
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, ShapeError, warn } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../../random/random";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";

/** Result of one mini-batch run (flat, row-major buffers). */
type MiniBatchRun = {
  readonly centers: Float64Array;
  readonly counts: Float64Array;
  readonly labels: Int32Array;
  readonly inertia: number;
  readonly nSteps: number;
};

/**
 * Build a uniform [0, 1) generator. A seed gives a private deterministic stream;
 * without one the global Deepbox generator is used (so `setSeed` still applies).
 */
function createRng(seed: number | undefined): () => number {
  if (seed === undefined) return __random;
  const gen = new __SeededRandom(__seedToUint64(seed));
  return () => gen.next();
}

function checkPositiveInt(name: string, value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError(`${name} must be an integer >= 1`, name, value);
  }
  return value;
}

function checkTol(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError("tol must be a finite number >= 0", "tol", value);
  }
  return value;
}

function checkInit(value: unknown): "random" | "kmeans++" {
  if (value !== "random" && value !== "kmeans++") {
    throw new InvalidParameterError(`init must be "random" or "kmeans++"`, "init", value);
  }
  return value;
}

function checkRandomState(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new InvalidParameterError("randomState must be a finite number", "randomState", value);
  }
  return value;
}

function checkMaxNoImprovement(value: unknown): number {
  if (value === Infinity) return value;
  return checkPositiveInt("maxNoImprovement", value);
}

/**
 * Mini-Batch K-Means clustering.
 *
 * A faster variant of KMeans that updates the centroids from small random batches of
 * data (Sculley, 2010): each batch sample is assigned to its nearest centroid, then
 * every centroid moves towards its samples with a per-centroid learning rate of
 * `1 / (number of samples it has seen)`, so a centroid is the running mean of all
 * samples assigned to it. This trades a small amount of quality for much lower cost
 * on large datasets.
 *
 * `maxIter` counts mini-batch steps (one batch per step), not passes over the data
 * as in scikit-learn. Training also stops early when the smoothed batch inertia has
 * not improved for `maxNoImprovement` consecutive steps, or when the squared centroid
 * movement of a step is at most `tol * mean(var(X))` (disabled for the default `tol: 0`).
 * With `nInit > 1` the whole procedure is repeated and the run with the lowest inertia
 * on the full data is kept.
 *
 * @example
 * ```ts
 * import { MiniBatchKMeans } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]);
 * const km = new MiniBatchKMeans({ nClusters: 2, batchSize: 3, randomState: 42 });
 * km.fit(X);
 * console.log(km.clusterCenters);
 * ```
 */
export class MiniBatchKMeans implements Clusterer {
  private nClusters: number;
  private maxIter: number;
  private batchSize: number;
  private tol: number;
  private nInit: number;
  private init: "random" | "kmeans++";
  private randomState: number | undefined;
  private maxNoImprovement: number;

  private clusterCenters_?: Tensor;
  private labels_?: Tensor;
  private inertia_?: number;
  private counts_?: Float64Array;
  private nSteps_ = 0;
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * Create a new Mini-Batch K-Means model.
   *
   * @param options - Configuration options
   * @param options.nClusters - Number of clusters (default: 8)
   * @param options.maxIter - Maximum number of mini-batch steps per initialization (default: 100)
   * @param options.batchSize - Samples drawn (with replacement) per step; capped at the number of samples (default: 100)
   * @param options.tol - Stop when the squared centroid movement of a step is at most `tol * mean(var(X))`; 0 disables this check (default: 0)
   * @param options.nInit - Number of initializations, keeping the best by final inertia (default: 3)
   * @param options.init - Initialization method: 'random' or 'kmeans++' (default: 'kmeans++')
   * @param options.randomState - Random seed for reproducibility
   * @param options.maxNoImprovement - Stop after this many consecutive steps without improvement of the smoothed batch inertia; `Infinity` disables this check (default: 10)
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly nClusters?: number;
      readonly maxIter?: number;
      readonly batchSize?: number;
      readonly tol?: number;
      readonly nInit?: number;
      readonly init?: "random" | "kmeans++";
      readonly randomState?: number;
      readonly maxNoImprovement?: number;
    } = {}
  ) {
    this.nClusters = checkPositiveInt("nClusters", options.nClusters ?? 8);
    this.maxIter = checkPositiveInt("maxIter", options.maxIter ?? 100);
    this.batchSize = checkPositiveInt("batchSize", options.batchSize ?? 100);
    this.tol = checkTol(options.tol ?? 0);
    this.nInit = checkPositiveInt("nInit", options.nInit ?? 3);
    this.init = checkInit(options.init ?? "kmeans++");
    this.randomState = checkRandomState(options.randomState);
    this.maxNoImprovement = checkMaxNoImprovement(options.maxNoImprovement ?? 10);
  }

  /** Mean of the per-feature population variances of `data` (0 for a single sample). */
  private static meanVariance(data: Float64Array, n: number, d: number): number {
    let total = 0;
    for (let j = 0; j < d; j++) {
      let mean = 0;
      for (let i = 0; i < n; i++) mean += data[i * d + j] as number;
      mean /= n;
      let ss = 0;
      for (let i = 0; i < n; i++) {
        const diff = (data[i * d + j] as number) - mean;
        ss += diff * diff;
      }
      total += ss / n;
    }
    return total / d;
  }

  /** Squared Euclidean distance from the row at `base` of `x` to the row at `cBase` of `c`. */
  private static sqDist(
    x: Float64Array,
    base: number,
    c: Float64Array,
    cBase: number,
    d: number
  ): number {
    let s = 0;
    for (let j = 0; j < d; j++) {
      const diff = (x[base + j] as number) - (c[cBase + j] as number);
      s += diff * diff;
    }
    return s;
  }

  /** Nearest center of the row at `base` (lowest index wins ties); writes the squared distance to `out[0]`. */
  private static nearest(
    x: Float64Array,
    base: number,
    centers: Float64Array,
    k: number,
    d: number,
    out: Float64Array
  ): number {
    let bestK = 0;
    let bestDist = Infinity;
    for (let c = 0; c < k; c++) {
      const cBase = c * d;
      let dist = 0;
      for (let j = 0; j < d; j++) {
        const diff = (x[base + j] as number) - (centers[cBase + j] as number);
        dist += diff * diff;
        if (dist >= bestDist) break;
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestK = c;
      }
    }
    out[0] = bestDist;
    return bestK;
  }

  private static initRandom(
    data: Float64Array,
    n: number,
    d: number,
    k: number,
    rng: () => number
  ): Float64Array {
    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    const centers = new Float64Array(k * d);
    for (let c = 0; c < k; c++) {
      const j = c + Math.min(n - c - 1, __randomBelow(rng, n - c));
      const picked = order[j] as number;
      order[j] = order[c] as number;
      order[c] = picked;
      for (let f = 0; f < d; f++) centers[c * d + f] = data[picked * d + f] as number;
    }
    return centers;
  }

  /**
   * Greedy k-means++ seeding (as in scikit-learn). Each new center is chosen among
   * `2 + floor(ln k)` candidates drawn with probability proportional to the squared
   * distance to the closest chosen center, keeping the candidate that lowers the total
   * squared distance the most.
   */
  private static initKMeansPlusPlus(
    data: Float64Array,
    n: number,
    d: number,
    k: number,
    rng: () => number
  ): Float64Array {
    const centers = new Float64Array(k * d);
    const minDist = new Float64Array(n).fill(Infinity);
    const cumulative = new Float64Array(n);
    const candidateDist = new Float64Array(n);
    const bestDist = new Float64Array(n);
    const nTrials = 2 + Math.floor(Math.log(k));

    const setCenter = (c: number, idx: number): void => {
      for (let f = 0; f < d; f++) centers[c * d + f] = data[idx * d + f] as number;
    };

    const first = Math.min(n - 1, __randomBelow(rng, n));
    setCenter(0, first);
    for (let i = 0; i < n; i++) minDist[i] = MiniBatchKMeans.sqDist(data, i * d, centers, 0, d);

    for (let c = 1; c < k; c++) {
      let total = 0;
      for (let i = 0; i < n; i++) {
        total += minDist[i] as number;
        cumulative[i] = total;
      }

      let bestPotential = Infinity;
      let bestIdx = -1;
      for (let trial = 0; trial < nTrials; trial++) {
        let idx: number;
        if (total > 0) {
          // First sample whose cumulative weight reaches the draw.
          const r = rng() * total;
          let lo = 0;
          let hi = n - 1;
          while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if ((cumulative[mid] as number) >= r) hi = mid;
            else lo = mid + 1;
          }
          idx = lo;
          // A draw can land on a sample that already is a center (weight 0); move to a neighbor with weight.
          while (idx > 0 && (minDist[idx] as number) === 0) idx--;
          while (idx < n - 1 && (minDist[idx] as number) === 0) idx++;
        } else {
          // Every sample coincides with a chosen center (fewer distinct points than clusters).
          idx = Math.min(n - 1, __randomBelow(rng, n));
        }
        let potential = 0;
        for (let i = 0; i < n; i++) {
          const dist = MiniBatchKMeans.sqDist(data, i * d, data, idx * d, d);
          const m = dist < (minDist[i] as number) ? dist : (minDist[i] as number);
          candidateDist[i] = m;
          potential += m;
        }
        if (potential < bestPotential) {
          bestPotential = potential;
          bestIdx = idx;
          bestDist.set(candidateDist);
        }
      }
      setCenter(c, bestIdx);
      minDist.set(bestDist);
    }
    return centers;
  }

  private initialCenters(
    data: Float64Array,
    n: number,
    d: number,
    rng: () => number
  ): Float64Array {
    return this.init === "random"
      ? MiniBatchKMeans.initRandom(data, n, d, this.nClusters, rng)
      : MiniBatchKMeans.initKMeansPlusPlus(data, n, d, this.nClusters, rng);
  }

  /**
   * One mini-batch step. All `m` batch samples are assigned using the centers as they were
   * at the start of the step, then every center is moved towards its samples with learning
   * rate `1 / counts[center]` (a running mean).
   *
   * @param rows - Row indices of the batch, or `null` to use rows `0..m-1`
   * @returns The summed squared distance of the batch to its assigned centers, and the squared
   *   movement of all centers during the step
   */
  private static step(
    data: Float64Array,
    d: number,
    k: number,
    rows: Int32Array | null,
    m: number,
    centers: Float64Array,
    counts: Float64Array,
    assignment: Int32Array,
    before: Float64Array
  ): { readonly batchInertia: number; readonly shiftSq: number } {
    before.set(centers);
    const dist = new Float64Array(1);
    let batchInertia = 0;
    for (let b = 0; b < m; b++) {
      const i = rows === null ? b : (rows[b] as number);
      assignment[b] = MiniBatchKMeans.nearest(data, i * d, centers, k, d, dist);
      batchInertia += dist[0] as number;
    }
    for (let b = 0; b < m; b++) {
      const i = rows === null ? b : (rows[b] as number);
      const c = assignment[b] as number;
      const seen = (counts[c] as number) + 1;
      counts[c] = seen;
      const eta = 1 / seen;
      for (let f = 0; f < d; f++) {
        const old = centers[c * d + f] as number;
        centers[c * d + f] = old + eta * ((data[i * d + f] as number) - old);
      }
    }
    let shiftSq = 0;
    for (let q = 0; q < centers.length; q++) {
      const diff = (centers[q] as number) - (before[q] as number);
      shiftSq += diff * diff;
    }
    return { batchInertia, shiftSq };
  }

  /** Labels every row with its nearest center and returns the total inertia. */
  private static labelAll(
    data: Float64Array,
    n: number,
    d: number,
    k: number,
    centers: Float64Array,
    labels: Int32Array
  ): number {
    const dist = new Float64Array(1);
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      labels[i] = MiniBatchKMeans.nearest(data, i * d, centers, k, d, dist);
      inertia += dist[0] as number;
    }
    return inertia;
  }

  /**
   * Fit Mini-Batch K-Means on training data.
   *
   * Calling `fit` again replaces the previous model, including anything learned through
   * {@link MiniBatchKMeans.partialFit}.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   * @throws {InvalidParameterError} If there are fewer samples than clusters
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const k = this.nClusters;

    if (n < k) {
      throw new InvalidParameterError(
        `n_samples=${n} should be >= n_clusters=${k}`,
        "nClusters",
        k
      );
    }

    const data = toFloat64View(X);
    const tolScaled = this.tol > 0 ? this.tol * MiniBatchKMeans.meanVariance(data, n, d) : 0;
    const batch = Math.min(this.batchSize, n);
    const rng = createRng(this.randomState);

    const rows = new Int32Array(batch);
    const assignment = new Int32Array(batch);
    const before = new Float64Array(k * d);

    let best: MiniBatchRun | undefined;
    for (let run = 0; run < this.nInit; run++) {
      const centers = this.initialCenters(data, n, d, rng);
      const counts = new Float64Array(k);
      let ewaInertia: number | undefined;
      let ewaMin = Infinity;
      let noImprovement = 0;
      let steps = 0;

      for (let iter = 0; iter < this.maxIter; iter++) {
        for (let b = 0; b < batch; b++) rows[b] = Math.min(n - 1, __randomBelow(rng, n));
        const { batchInertia, shiftSq } = MiniBatchKMeans.step(
          data,
          d,
          k,
          rows,
          batch,
          centers,
          counts,
          assignment,
          before
        );
        steps++;

        // The first step only measures the inertia of the initial centers.
        if (iter === 0) continue;
        const normalized = batchInertia / batch;
        const alpha = Math.min(1, (batch * 2) / (n + 1));
        ewaInertia =
          ewaInertia === undefined ? normalized : ewaInertia * (1 - alpha) + normalized * alpha;
        if (tolScaled > 0 && shiftSq <= tolScaled) break;
        if (ewaInertia < ewaMin) {
          noImprovement = 0;
          ewaMin = ewaInertia;
        } else {
          noImprovement++;
        }
        if (noImprovement >= this.maxNoImprovement) break;
      }

      const labels = new Int32Array(n);
      const inertia = MiniBatchKMeans.labelAll(data, n, d, k, centers, labels);
      if (best === undefined || inertia < best.inertia) {
        best = { centers, counts, labels, inertia, nSteps: steps };
      }
    }

    // `best` is always set because nInit >= 1.
    const final = best as MiniBatchRun;

    const sizes = new Int32Array(k);
    for (let i = 0; i < n; i++) sizes[final.labels[i] as number]!++;
    let distinct = 0;
    for (let c = 0; c < k; c++) if ((sizes[c] as number) > 0) distinct++;
    if (distinct < k) {
      warn(
        `Number of distinct clusters (${distinct}) found is smaller than n_clusters (${k}). ` +
          "This can happen when X contains duplicate points.",
        "ConvergenceWarning",
        "MiniBatchKMeans"
      );
    }

    this.nFeaturesIn_ = d;
    this.clusterCenters_ = tensor(final.centers).reshape([k, d]);
    this.labels_ = tensor(final.labels, { dtype: "int32" });
    this.inertia_ = final.inertia;
    this.counts_ = final.counts;
    this.nSteps_ = final.nSteps;
    this.fitted = true;
    return this;
  }

  /**
   * Update the model with a single mini-batch (online learning).
   *
   * On the first call the centroids are initialized from `X`, so it needs at least
   * `nClusters` samples. Later calls move the existing centroids towards the new samples
   * and keep the per-centroid sample counts, so a centroid stays the running mean of
   * everything it has seen. `labels` and `inertia` describe the latest batch.
   *
   * @param X - Batch of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns this - The updated estimator
   * @throws {ShapeError} If X is not 2D or has a different number of features than earlier batches
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   * @throws {InvalidParameterError} If this is the first batch and it has fewer samples than clusters
   */
  partialFit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const k = this.nClusters;
    const data = toFloat64View(X);

    let centers: Float64Array;
    let counts: Float64Array;
    if (this.fitted && this.clusterCenters_ && this.counts_) {
      if (d !== this.nFeaturesIn_) {
        throw new ShapeError(
          `X has ${d} features but MiniBatchKMeans was fitted with ${this.nFeaturesIn_} features`
        );
      }
      if ((this.clusterCenters_.shape[0] ?? 0) !== k) {
        throw new InvalidParameterError(
          `nClusters was changed from ${this.clusterCenters_.shape[0]} to ${k} after fitting; call fit again`,
          "nClusters",
          k
        );
      }
      centers = toFloat64View(this.clusterCenters_).slice();
      counts = this.counts_.slice();
    } else {
      if (n < k) {
        throw new InvalidParameterError(
          `n_samples=${n} should be >= n_clusters=${k}`,
          "nClusters",
          k
        );
      }
      centers = this.initialCenters(data, n, d, createRng(this.randomState));
      counts = new Float64Array(k);
    }

    MiniBatchKMeans.step(
      data,
      d,
      k,
      null,
      n,
      centers,
      counts,
      new Int32Array(n),
      new Float64Array(k * d)
    );
    const labels = new Int32Array(n);
    const inertia = MiniBatchKMeans.labelAll(data, n, d, k, centers, labels);

    this.nFeaturesIn_ = d;
    this.clusterCenters_ = tensor(centers).reshape([k, d]);
    this.labels_ = tensor(labels, { dtype: "int32" });
    this.inertia_ = inertia;
    this.counts_ = counts;
    this.nSteps_ += 1;
    this.fitted = true;
    return this;
  }

  /**
   * Predict the closest cluster for each sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MiniBatchKMeans must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MiniBatchKMeans");

    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const k = this.clusterCenters_.shape[0] ?? 0;
    const labels = new Int32Array(n);
    MiniBatchKMeans.labelAll(
      toFloat64View(X),
      n,
      d,
      k,
      toFloat64View(this.clusterCenters_),
      labels
    );
    return tensor(labels, { dtype: "int32" });
  }

  /**
   * Fit the model and return the training labels.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns Cluster labels of shape (n_samples,)
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels;
  }

  /**
   * Transform X to cluster-distance space.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Euclidean distance from each sample to each centroid, shape (n_samples, n_clusters)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MiniBatchKMeans must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MiniBatchKMeans");

    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const k = this.clusterCenters_.shape[0] ?? 0;
    const data = toFloat64View(X);
    const centers = toFloat64View(this.clusterCenters_);
    const out = new Float64Array(n * k);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < k; c++) {
        out[i * k + c] = Math.sqrt(MiniBatchKMeans.sqDist(data, i * d, centers, c * d, d));
      }
    }
    return tensor(out).reshape([n, k]);
  }

  /**
   * Fit the model and return the distances of the training samples to every centroid.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns Distances of shape (n_samples, n_clusters)
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /**
   * Opposite of the inertia of `X` under the fitted centroids (higher is better).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Negative sum of squared distances to the closest centroid
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  score(X: Tensor): number {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MiniBatchKMeans must be fitted before score");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MiniBatchKMeans");

    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const k = this.clusterCenters_.shape[0] ?? 0;
    return -MiniBatchKMeans.labelAll(
      toFloat64View(X),
      n,
      d,
      k,
      toFloat64View(this.clusterCenters_),
      new Int32Array(n)
    );
  }

  /** Cluster centers of shape (n_clusters, n_features). */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MiniBatchKMeans must be fitted to access cluster centers");
    }
    return this.clusterCenters_;
  }

  /** Labels of the training data (of the latest batch after `partialFit`). */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("MiniBatchKMeans must be fitted to access labels");
    }
    return this.labels_;
  }

  /** Sum of squared distances of the samples to their closest centroid. */
  get inertia(): number {
    if (!this.fitted || this.inertia_ === undefined) {
      throw new NotFittedError("MiniBatchKMeans must be fitted to access inertia");
    }
    return this.inertia_;
  }

  /** Number of mini-batch steps performed by the selected run (plus any `partialFit` calls). */
  get nSteps(): number {
    if (!this.fitted) {
      throw new NotFittedError("MiniBatchKMeans must be fitted to access the step count");
    }
    return this.nSteps_;
  }

  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClusters,
      maxIter: this.maxIter,
      batchSize: this.batchSize,
      tol: this.tol,
      nInit: this.nInit,
      init: this.init,
      randomState: this.randomState,
      maxNoImprovement: this.maxNoImprovement,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nClusters":
          this.nClusters = checkPositiveInt("nClusters", value);
          break;
        case "maxIter":
          this.maxIter = checkPositiveInt("maxIter", value);
          break;
        case "batchSize":
          this.batchSize = checkPositiveInt("batchSize", value);
          break;
        case "tol":
          this.tol = checkTol(value);
          break;
        case "nInit":
          this.nInit = checkPositiveInt("nInit", value);
          break;
        case "init":
          this.init = checkInit(value);
          break;
        case "randomState":
          this.randomState = checkRandomState(value);
          break;
        case "maxNoImprovement":
          this.maxNoImprovement = checkMaxNoImprovement(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
