import { InvalidParameterError, NotFittedError, ShapeError, warn } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../../random/random";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";

/** Result of one K-Means run (flat, row-major buffers). */
type KMeansRun = {
  readonly centers: Float64Array;
  readonly labels: Int32Array;
  readonly inertia: number;
  readonly nIter: number;
};

/** Above this many `n_samples * n_clusters` bound entries `algorithm: "auto"` stays with Lloyd. */
const ELKAN_AUTO_MAX_BOUNDS = 2e7;

/**
 * Build a uniform [0, 1) generator. A seed gives a private deterministic stream;
 * without one the global Deepbox generator is used (so `setSeed` still applies).
 */
function createRng(seed: number | undefined): () => number {
  if (seed === undefined) return __random;
  const gen = new __SeededRandom(__seedToUint64(seed));
  return () => gen.next();
}

/**
 * K-Means clustering algorithm.
 *
 * Partitions n samples into k clusters by minimizing the within-cluster
 * sum of squared distances to cluster centroids.
 *
 * **Algorithm**: Lloyd's algorithm (iterative refinement), optionally with
 * Elkan's triangle-inequality bounds
 * 1. Initialize k centroids (random or greedy k-means++)
 * 2. Assign each point to nearest centroid
 * 3. Update centroids as mean of assigned points (empty clusters are moved to
 *    the points that are farthest from their centroid)
 * 4. Stop when the assignment no longer changes, when the squared centroid
 *    shift is at most `tol * mean(var(X))`, or after `maxIter` iterations
 *
 * `tol` is relative to the average per-feature variance of `X`, like
 * scikit-learn, so the stopping rule does not depend on the scale of the data.
 *
 * **Time Complexity**: O(n * k * i * d) where:
 * - n = number of samples
 * - k = number of clusters
 * - i = number of iterations
 * - d = number of features
 *
 * @example
 * ```ts
 * import { KMeans } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]);
 * const kmeans = new KMeans({ nClusters: 2, randomState: 42 });
 * kmeans.fit(X);
 *
 * const labels = kmeans.predict(X);
 * console.log('Cluster labels:', labels);
 * console.log('Centroids:', kmeans.clusterCenters);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox Clustering}
 */
export class KMeans implements Clusterer {
  private nClusters: number;
  private maxIter: number;
  private tol: number;
  private init: "random" | "kmeans++";
  private nInit: number;
  private warmStart: boolean;
  private randomState: number | undefined;
  private algorithm: "lloyd" | "elkan" | "auto";

  private clusterCenters_?: Tensor;
  private labels_?: Tensor;
  private inertia_?: number;
  private nIter_?: number;
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * Create a new K-Means clustering model.
   *
   * @param options - Configuration options
   * @param options.nClusters - Number of clusters (default: 8)
   * @param options.maxIter - Maximum number of iterations per run (default: 300)
   * @param options.tol - Convergence tolerance on the squared centroid shift, relative to the mean feature variance of `X` (default: 1e-4)
   * @param options.init - Initialization method: 'random' or 'kmeans++' (default: 'kmeans++'; 'k-means++' is accepted as an alias)
   * @param options.nInit - Number of times to run with different seeds, keeping the best result (default: 10)
   * @param options.warmStart - If true, a refit starts from the centroids of the previous fit and runs once (default: false)
   * @param options.randomState - Random seed for reproducibility
   * @param options.algorithm - Algorithm: 'lloyd', 'elkan', or 'auto' (default: 'auto', which uses Elkan for 4 or more clusters)
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly nClusters?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly init?: "random" | "kmeans++" | "k-means++";
      readonly nInit?: number;
      readonly warmStart?: boolean;
      readonly randomState?: number;
      readonly algorithm?: "lloyd" | "elkan" | "auto";
    } = {}
  ) {
    this.nClusters = options.nClusters ?? 8;
    this.maxIter = options.maxIter ?? 300;
    this.tol = options.tol ?? 1e-4;
    const requestedInit = options.init ?? "kmeans++";
    this.init = requestedInit === "k-means++" ? "kmeans++" : requestedInit;
    this.nInit = options.nInit ?? 10;
    this.warmStart = options.warmStart ?? false;
    this.algorithm = options.algorithm ?? "auto";
    if (options.randomState !== undefined) {
      this.randomState = options.randomState;
    }

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new InvalidParameterError(
        "nClusters must be an integer >= 1",
        "nClusters",
        this.nClusters
      );
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", this.maxIter);
    }
    if (!Number.isFinite(this.tol) || this.tol < 0) {
      throw new InvalidParameterError("tol must be a finite number >= 0", "tol", this.tol);
    }
    if (this.init !== "random" && this.init !== "kmeans++") {
      throw new InvalidParameterError(
        `init must be "random" or "kmeans++"; received ${String(this.init)}`,
        "init",
        this.init
      );
    }
    if (!Number.isInteger(this.nInit) || this.nInit < 1) {
      throw new InvalidParameterError("nInit must be an integer >= 1", "nInit", this.nInit);
    }
    if (typeof this.warmStart !== "boolean") {
      throw new InvalidParameterError("warmStart must be a boolean", "warmStart", this.warmStart);
    }
    if (options.randomState !== undefined && !Number.isFinite(options.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(options.randomState)}`,
        "randomState",
        options.randomState
      );
    }
    if (this.algorithm !== "lloyd" && this.algorithm !== "elkan" && this.algorithm !== "auto") {
      throw new InvalidParameterError(
        `algorithm must be 'lloyd', 'elkan', or 'auto'; received ${String(this.algorithm)}`,
        "algorithm",
        this.algorithm
      );
    }
  }

  /**
   * Fit K-Means clustering on training data.
   *
   * Calling `fit` again replaces the previous model. With `warmStart: true` the
   * previous centroids seed a single run instead of a fresh initialization.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, or a warm start does not match the previous centroids
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   * @throws {InvalidParameterError} If there are fewer samples than clusters
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const k = this.nClusters;

    if (nSamples < k) {
      throw new InvalidParameterError(
        `n_samples=${nSamples} should be >= n_clusters=${k}`,
        "nClusters",
        k
      );
    }

    const data = toFloat64View(X);
    const tolScaled = this.tol * KMeans.meanVariance(data, nSamples, nFeatures);
    const useElkan =
      this.algorithm === "elkan" ||
      (this.algorithm === "auto" && k >= 4 && nSamples * k <= ELKAN_AUTO_MAX_BOUNDS);
    const runOnce = (init: Float64Array): KMeansRun =>
      useElkan
        ? this.runElkan(data, nSamples, nFeatures, init, tolScaled)
        : this.runLloyd(data, nSamples, nFeatures, init, tolScaled);

    let best: KMeansRun | undefined;

    if (this.warmStart && this.fitted && this.clusterCenters_) {
      const prev = this.clusterCenters_;
      const prevK = prev.shape[0] ?? 0;
      const prevD = prev.shape[1] ?? 0;
      if (prevK !== k || prevD !== nFeatures) {
        throw new ShapeError(
          `warmStart needs the previous centroids to have shape [${k}, ${nFeatures}]; got [${prevK}, ${prevD}]`
        );
      }
      best = runOnce(toFloat64View(prev).slice());
    } else {
      const rng = createRng(this.randomState);
      for (let run = 0; run < this.nInit; run++) {
        const init =
          this.init === "random"
            ? KMeans.initRandom(data, nSamples, nFeatures, k, rng)
            : KMeans.initKMeansPlusPlus(data, nSamples, nFeatures, k, rng);
        const result = runOnce(init);
        if (best === undefined || result.inertia < best.inertia) best = result;
      }
    }

    // `best` is always set: nInit >= 1 on the cold path, and the warm path assigns it.
    const final = best as KMeansRun;

    const sizes = new Int32Array(k);
    for (let i = 0; i < nSamples; i++) {
      const label = final.labels[i] as number;
      sizes[label] = (sizes[label] as number) + 1;
    }
    let distinct = 0;
    for (let c = 0; c < k; c++) if ((sizes[c] as number) > 0) distinct++;
    if (distinct < k) {
      warn(
        `Number of distinct clusters (${distinct}) found is smaller than n_clusters (${k}). ` +
          "This can happen when X contains duplicate points.",
        "ConvergenceWarning",
        "KMeans"
      );
    }

    this.nFeaturesIn_ = nFeatures;
    this.clusterCenters_ = tensor(final.centers).reshape([k, nFeatures]);
    this.labels_ = tensor(final.labels);
    this.inertia_ = final.inertia;
    this.nIter_ = final.nIter;
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
      throw new NotFittedError("KMeans must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "KMeans");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const labels = new Int32Array(nSamples);
    KMeans.assign(
      toFloat64View(X),
      nSamples,
      nFeatures,
      toFloat64View(this.clusterCenters_),
      this.clusterCenters_.shape[0] ?? 0,
      labels,
      undefined,
      false
    );
    return tensor(labels);
  }

  /**
   * Fit and predict in one step.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns Cluster labels of shape (n_samples,)
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    if (!this.labels_) {
      throw new NotFittedError("KMeans fit did not produce labels");
    }
    return this.labels_;
  }

  /**
   * Transform X to cluster-distance space.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Euclidean distance from each sample to each centroid, shape (n_samples, n_clusters)
   * @throws {NotFittedError} If the model has not been fitted
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("KMeans must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "KMeans");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const k = this.clusterCenters_.shape[0] ?? 0;
    const data = toFloat64View(X);
    const centers = toFloat64View(this.clusterCenters_);
    const out = new Float64Array(nSamples * k);
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < k; c++) {
        out[i * k + c] = Math.sqrt(
          KMeans.sqDist(data, i * nFeatures, centers, c * nFeatures, nFeatures)
        );
      }
    }
    return tensor(out).reshape([nSamples, k]);
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
   */
  score(X: Tensor): number {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("KMeans must be fitted before score");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "KMeans");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const dist2 = new Float64Array(nSamples);
    KMeans.assign(
      toFloat64View(X),
      nSamples,
      nFeatures,
      toFloat64View(this.clusterCenters_),
      this.clusterCenters_.shape[0] ?? 0,
      new Int32Array(nSamples),
      dist2,
      false
    );
    let total = 0;
    for (let i = 0; i < nSamples; i++) total += dist2[i] as number;
    return -total;
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

  /** Squared Euclidean distance between row at `xBase` of `x` and row at `cBase` of `c`. */
  private static sqDist(
    x: Float64Array,
    xBase: number,
    c: Float64Array,
    cBase: number,
    d: number
  ): number {
    let s = 0;
    for (let j = 0; j < d; j++) {
      const diff = (x[xBase + j] as number) - (c[cBase + j] as number);
      s += diff * diff;
    }
    return s;
  }

  /**
   * E-step: label every sample with its nearest centroid (lowest index wins ties).
   *
   * @returns Number of samples whose label changed
   */
  private static assign(
    data: Float64Array,
    n: number,
    d: number,
    centers: Float64Array,
    k: number,
    labels: Int32Array,
    dist2: Float64Array | undefined,
    trackChanges: boolean
  ): number {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const xBase = i * d;
      let minDist = Infinity;
      let minLabel = 0;
      for (let c = 0; c < k; c++) {
        const dist = KMeans.sqDist(data, xBase, centers, c * d, d);
        if (dist < minDist) {
          minDist = dist;
          minLabel = c;
        }
      }
      if (trackChanges && labels[i] !== minLabel) changed++;
      labels[i] = minLabel;
      if (dist2) dist2[i] = minDist;
    }
    return changed;
  }

  /** Sample `k` distinct row indices uniformly (Floyd's algorithm) and copy those rows. */
  private static initRandom(
    data: Float64Array,
    n: number,
    d: number,
    k: number,
    rng: () => number
  ): Float64Array {
    const chosen = new Set<number>();
    for (let j = n - k; j < n; j++) {
      const t = Math.min(__randomBelow(rng, j + 1), j);
      chosen.add(chosen.has(t) ? j : t);
    }
    const centers = new Float64Array(k * d);
    let c = 0;
    for (const idx of chosen) {
      for (let f = 0; f < d; f++) centers[c * d + f] = data[idx * d + f] as number;
      c++;
    }
    return centers;
  }

  /**
   * Greedy k-means++ seeding (Arthur & Vassilvitskii 2007), as in scikit-learn: every
   * new centroid is the best of `2 + floor(ln k)` candidates drawn with probability
   * proportional to the squared distance to the closest chosen centroid.
   */
  private static initKMeansPlusPlus(
    data: Float64Array,
    n: number,
    d: number,
    k: number,
    rng: () => number
  ): Float64Array {
    const centers = new Float64Array(k * d);
    const nTrials = 2 + Math.floor(Math.log(k));

    const first = Math.min(__randomBelow(rng, n), n - 1);
    for (let f = 0; f < d; f++) centers[f] = data[first * d + f] as number;

    const closest = new Float64Array(n);
    const candidateDist = new Float64Array(n);
    let potential = 0;
    for (let i = 0; i < n; i++) {
      const dist = KMeans.sqDist(data, i * d, data, first * d, d);
      closest[i] = dist;
      potential += dist;
    }

    const cumulative = new Float64Array(n);
    const bestDist = new Float64Array(n);
    for (let c = 1; c < k; c++) {
      let run = 0;
      for (let i = 0; i < n; i++) {
        run += closest[i] as number;
        cumulative[i] = run;
      }

      let bestPotential = Infinity;
      let bestCandidate = -1;
      for (let t = 0; t < nTrials; t++) {
        // First index whose cumulative weight reaches the draw (binary search).
        const target = rng() * potential;
        let lo = 0;
        let hi = n - 1;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if ((cumulative[mid] as number) < target) lo = mid + 1;
          else hi = mid;
        }
        const cand = lo;

        let pot = 0;
        for (let i = 0; i < n; i++) {
          const dist = KMeans.sqDist(data, i * d, data, cand * d, d);
          const m = dist < (closest[i] as number) ? dist : (closest[i] as number);
          candidateDist[i] = m;
          pot += m;
        }
        if (pot < bestPotential) {
          bestPotential = pot;
          bestCandidate = cand;
          bestDist.set(candidateDist);
        }
      }

      for (let f = 0; f < d; f++) centers[c * d + f] = data[bestCandidate * d + f] as number;
      potential = bestPotential;
      closest.set(bestDist);
    }
    return centers;
  }

  /**
   * Give every empty cluster the sample that is farthest from its own centroid,
   * adjusting the running sums and counts (scikit-learn's empty-cluster handling).
   */
  private static relocateEmptyClusters(
    data: Float64Array,
    n: number,
    d: number,
    k: number,
    labels: Int32Array,
    dist2: Float64Array,
    sums: Float64Array,
    counts: Int32Array
  ): void {
    let hasEmpty = false;
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        hasEmpty = true;
        break;
      }
    }
    if (!hasEmpty) return;

    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => (dist2[b] as number) - (dist2[a] as number) || a - b);

    let next = 0;
    for (let c = 0; c < k; c++) {
      if (counts[c] !== 0) continue;
      while (next < n) {
        const p = order[next++] as number;
        const old = labels[p] as number;
        // Never empty another cluster to fill this one.
        if ((counts[old] as number) <= 1) continue;
        for (let f = 0; f < d; f++) {
          const v = data[p * d + f] as number;
          sums[old * d + f] = (sums[old * d + f] as number) - v;
          sums[c * d + f] = v;
        }
        counts[old] = (counts[old] as number) - 1;
        counts[c] = 1;
        break;
      }
    }
  }

  /**
   * M-step: write the per-cluster means of the labelled samples to `out`. Empty
   * clusters are relocated first (see `relocateEmptyClusters`); if none can be
   * filled the old centroid is kept.
   */
  private static updateCenters(
    data: Float64Array,
    n: number,
    d: number,
    k: number,
    labels: Int32Array,
    dist2: Float64Array,
    centers: Float64Array,
    out: Float64Array
  ): void {
    const sums = new Float64Array(k * d);
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const c = labels[i] as number;
      counts[c] = (counts[c] as number) + 1;
      const cBase = c * d;
      const xBase = i * d;
      for (let j = 0; j < d; j++) {
        sums[cBase + j] = (sums[cBase + j] as number) + (data[xBase + j] as number);
      }
    }
    KMeans.relocateEmptyClusters(data, n, d, k, labels, dist2, sums, counts);
    for (let c = 0; c < k; c++) {
      const cnt = counts[c] as number;
      const cBase = c * d;
      for (let j = 0; j < d; j++) {
        out[cBase + j] =
          cnt > 0 ? (sums[cBase + j] as number) / cnt : (centers[cBase + j] as number);
      }
    }
  }

  /** Lloyd's algorithm from the given initial centroids. */
  private runLloyd(
    data: Float64Array,
    n: number,
    d: number,
    init: Float64Array,
    tolScaled: number
  ): KMeansRun {
    const k = this.nClusters;
    let centers = init.slice();
    let next = new Float64Array(k * d);
    const labels = new Int32Array(n).fill(-1);
    const dist2 = new Float64Array(n);

    let nIter = 0;
    let strict = false;
    for (let iter = 0; iter < this.maxIter; iter++) {
      nIter = iter + 1;
      const changed = KMeans.assign(data, n, d, centers, k, labels, dist2, true);
      if (changed === 0) {
        strict = true;
        break;
      }
      KMeans.updateCenters(data, n, d, k, labels, dist2, centers, next);
      let shift = 0;
      for (let i = 0; i < k * d; i++) {
        const diff = (next[i] as number) - (centers[i] as number);
        shift += diff * diff;
      }
      const swap = centers;
      centers = next;
      next = swap;
      if (shift <= tolScaled) break;
    }

    // Centroids moved after the last E-step unless the labels were stable, so relabel.
    if (!strict) KMeans.assign(data, n, d, centers, k, labels, dist2, false);
    let inertia = 0;
    for (let i = 0; i < n; i++) inertia += dist2[i] as number;
    return { centers, labels, inertia, nIter };
  }

  /**
   * Elkan's algorithm: Lloyd iterations that skip distance computations ruled out by
   * the triangle inequality. Produces the same partition as Lloyd for the same start.
   */
  private runElkan(
    data: Float64Array,
    n: number,
    d: number,
    init: Float64Array,
    tolScaled: number
  ): KMeansRun {
    const K = this.nClusters;
    let centers = init.slice();
    let next = new Float64Array(K * d);

    // lower[i * K + c]: lower bound on d(x_i, center_c); upper[i]: upper bound on d(x_i, assigned center)
    const lower = new Float64Array(n * K);
    const upper = new Float64Array(n);
    const labels = new Int32Array(n);
    const dist2 = new Float64Array(n);
    const centerDist = new Float64Array(K * K);
    const halfMin = new Float64Array(K);
    const movement = new Float64Array(K);

    for (let i = 0; i < n; i++) {
      let minD = Infinity;
      let minC = 0;
      for (let c = 0; c < K; c++) {
        const dist = Math.sqrt(KMeans.sqDist(data, i * d, centers, c * d, d));
        lower[i * K + c] = dist;
        if (dist < minD) {
          minD = dist;
          minC = c;
        }
      }
      labels[i] = minC;
      upper[i] = minD;
    }

    let nIter = 0;
    for (let iter = 0; iter < this.maxIter; iter++) {
      nIter = iter + 1;

      halfMin.fill(Infinity);
      for (let a = 0; a < K; a++) {
        for (let b = a + 1; b < K; b++) {
          const dist = Math.sqrt(KMeans.sqDist(centers, a * d, centers, b * d, d));
          centerDist[a * K + b] = dist;
          centerDist[b * K + a] = dist;
          const half = 0.5 * dist;
          if (half < (halfMin[a] as number)) halfMin[a] = half;
          if (half < (halfMin[b] as number)) halfMin[b] = half;
        }
      }

      // The first pass keeps the initial full assignment, so it always counts as a change.
      let changed = iter === 0;
      for (let i = 0; i < n; i++) {
        let a = labels[i] as number;
        let u = upper[i] as number;
        if (u <= (halfMin[a] as number)) continue;

        const iBase = i * K;
        let tight = false;
        for (let c = 0; c < K; c++) {
          if (c === a) continue;
          if (u <= (lower[iBase + c] as number)) continue;
          if (u <= 0.5 * (centerDist[a * K + c] as number)) continue;

          if (!tight) {
            u = Math.sqrt(KMeans.sqDist(data, i * d, centers, a * d, d));
            lower[iBase + a] = u;
            tight = true;
            if (u <= (lower[iBase + c] as number)) continue;
            if (u <= 0.5 * (centerDist[a * K + c] as number)) continue;
          }

          const dc = Math.sqrt(KMeans.sqDist(data, i * d, centers, c * d, d));
          lower[iBase + c] = dc;
          if (dc < u) {
            a = c;
            u = dc;
          }
        }
        if (a !== labels[i]) {
          labels[i] = a;
          changed = true;
        }
        upper[i] = u;
      }

      if (!changed) break;

      // Exact squared distances are only needed to pick relocation candidates.
      let hasEmpty = false;
      {
        const seen = new Uint8Array(K);
        for (let i = 0; i < n; i++) seen[labels[i] as number] = 1;
        for (let c = 0; c < K; c++) if (seen[c] === 0) hasEmpty = true;
      }
      if (hasEmpty) {
        for (let i = 0; i < n; i++) {
          dist2[i] = KMeans.sqDist(data, i * d, centers, (labels[i] as number) * d, d);
        }
      }
      KMeans.updateCenters(data, n, d, K, labels, dist2, centers, next);

      let shift = 0;
      for (let c = 0; c < K; c++) {
        const s = KMeans.sqDist(next, c * d, centers, c * d, d);
        shift += s;
        movement[c] = Math.sqrt(s);
      }
      for (let i = 0; i < n; i++) {
        const iBase = i * K;
        for (let c = 0; c < K; c++) {
          const v = (lower[iBase + c] as number) - (movement[c] as number);
          lower[iBase + c] = v > 0 ? v : 0;
        }
        upper[i] = (upper[i] as number) + (movement[labels[i] as number] as number);
      }

      const swap = centers;
      centers = next;
      next = swap;
      if (shift <= tolScaled) break;
    }

    // The bounds only prune work; the reported labels come from an exact E-step.
    KMeans.assign(data, n, d, centers, K, labels, dist2, false);
    let inertia = 0;
    for (let i = 0; i < n; i++) inertia += dist2[i] as number;
    return { centers, labels, inertia, nIter };
  }

  /**
   * Get cluster centers.
   *
   * @returns Tensor of shape (n_clusters, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("KMeans must be fitted to access cluster centers");
    }
    return this.clusterCenters_;
  }

  /**
   * Get training labels.
   *
   * @returns Tensor of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("KMeans must be fitted to access labels");
    }
    return this.labels_;
  }

  /**
   * Get inertia (sum of squared distances of samples to their closest centroid).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get inertia(): number {
    if (!this.fitted || this.inertia_ === undefined) {
      throw new NotFittedError("KMeans must be fitted to access inertia");
    }
    return this.inertia_;
  }

  /**
   * Get number of iterations run by the best run.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    if (!this.fitted || this.nIter_ === undefined) {
      throw new NotFittedError("KMeans must be fitted to access n_iter");
    }
    return this.nIter_;
  }

  /**
   * Number of features seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("KMeans must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClusters,
      maxIter: this.maxIter,
      tol: this.tol,
      init: this.init,
      nInit: this.nInit,
      warmStart: this.warmStart,
      randomState: this.randomState,
      algorithm: this.algorithm,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set (nClusters, maxIter, tol, init, nInit, warmStart, randomState, algorithm)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nClusters":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nClusters must be an integer >= 1",
              "nClusters",
              value
            );
          }
          this.nClusters = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
            throw new InvalidParameterError("tol must be a finite number >= 0", "tol", value);
          }
          this.tol = value;
          break;
        case "init":
          if (value !== "random" && value !== "kmeans++" && value !== "k-means++") {
            throw new InvalidParameterError(`init must be "random" or "kmeans++"`, "init", value);
          }
          this.init = value === "random" ? "random" : "kmeans++";
          break;
        case "nInit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("nInit must be an integer >= 1", "nInit", value);
          }
          this.nInit = value;
          break;
        case "warmStart":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("warmStart must be a boolean", "warmStart", value);
          }
          this.warmStart = value;
          break;
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value;
          break;
        case "algorithm":
          if (value !== "lloyd" && value !== "elkan" && value !== "auto") {
            throw new InvalidParameterError(
              `algorithm must be 'lloyd', 'elkan', or 'auto'`,
              "algorithm",
              value
            );
          }
          this.algorithm = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   *
   * @returns A new KMeans instance
   */
  clone(): KMeans {
    const params = this.getParams() as {
      nClusters: number;
      maxIter: number;
      tol: number;
      init: "random" | "kmeans++";
      nInit: number;
      warmStart: boolean;
      randomState: number | undefined;
      algorithm: "lloyd" | "elkan" | "auto";
    };
    const { randomState, ...rest } = params;
    return new KMeans(randomState === undefined ? rest : { ...rest, randomState });
  }
}
