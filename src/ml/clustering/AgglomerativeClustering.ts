/**
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox documentation}
 */

import { InvalidParameterError, MemoryError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";

type Linkage = "single" | "complete" | "average" | "ward";

/** Largest condensed distance matrix (in doubles) the implementation will allocate (4 GiB). */
const MAX_CONDENSED_ENTRIES = 2 ** 29;

/**
 * Dendrogram of a hierarchical clustering. Row `i` of `children` holds the two
 * nodes merged at step `i`; ids below `n` are samples, id `n + i` is the cluster
 * created by step `i`. `distances[i]` is the merge height (non-decreasing).
 */
type Dendrogram = {
  readonly children: Int32Array;
  readonly distances: Float64Array;
};

/**
 * Build the dendrogram with the nearest-neighbor-chain algorithm (O(n^2) time,
 * O(n^2 / 2) memory). Valid for the four supported linkages because all of them
 * are reducible. Merge order and node ids follow SciPy's `linkage` output.
 */
function buildDendrogram(data: Float64Array, n: number, d: number, linkage: Linkage): Dendrogram {
  const nMerges = n - 1;
  const children = new Int32Array(nMerges * 2);
  const distances = new Float64Array(nMerges);
  if (nMerges === 0) return { children, distances };

  const condensedSize = (n * (n - 1)) / 2;
  if (condensedSize > MAX_CONDENSED_ENTRIES) {
    throw new MemoryError(
      `AgglomerativeClustering needs a distance matrix of ${condensedSize} entries for n_samples=${n}, which is too large`,
      { requestedBytes: condensedSize * 8 }
    );
  }

  const ward = linkage === "ward";
  // Index of the pair (i, j), i < j, in the condensed matrix.
  const pos = (i: number, j: number): number =>
    i < j ? n * i - (i * (i + 1)) / 2 + j - i - 1 : n * j - (j * (j + 1)) / 2 + i - j - 1;

  // Ward works on squared Euclidean distances (Lance-Williams); the others on Euclidean distances.
  let dist: Float64Array;
  try {
    dist = new Float64Array(condensedSize);
  } catch (error) {
    throw new MemoryError(
      `AgglomerativeClustering could not allocate a distance matrix of ${condensedSize} entries for n_samples=${n}`,
      { requestedBytes: condensedSize * 8, cause: error }
    );
  }
  let p = 0;
  for (let i = 0; i < n - 1; i++) {
    const iBase = i * d;
    for (let j = i + 1; j < n; j++) {
      const jBase = j * d;
      let s = 0;
      for (let f = 0; f < d; f++) {
        const diff = (data[iBase + f] as number) - (data[jBase + f] as number);
        s += diff * diff;
      }
      dist[p++] = ward ? s : Math.sqrt(s);
    }
  }

  const size = new Float64Array(n).fill(1);
  const active = new Uint8Array(n).fill(1);
  const chain = new Int32Array(n);
  let chainLength = 0;
  const mx = new Int32Array(nMerges);
  const my = new Int32Array(nMerges);
  const md = new Float64Array(nMerges);

  for (let step = 0; step < nMerges; step++) {
    if (chainLength === 0) {
      for (let i = 0; i < n; i++) {
        if (active[i] === 1) {
          chain[0] = i;
          chainLength = 1;
          break;
        }
      }
    }

    let x = 0;
    let y = 0;
    let current = Infinity;
    for (;;) {
      x = chain[chainLength - 1] as number;
      y = -1;
      current = Infinity;
      // Prefer the previous chain element on ties so the chain terminates.
      if (chainLength > 1) {
        y = chain[chainLength - 2] as number;
        current = dist[pos(x, y)] as number;
      }
      for (let i = 0; i < n; i++) {
        if (active[i] === 0 || i === x) continue;
        const v = dist[pos(x, i)] as number;
        if (v < current) {
          current = v;
          y = i;
        }
      }
      if (chainLength > 1 && y === chain[chainLength - 2]) break;
      chain[chainLength++] = y;
    }
    chainLength -= 2;

    if (x > y) {
      const t = x;
      x = y;
      y = t;
    }
    const nx = size[x] as number;
    const ny = size[y] as number;
    mx[step] = x;
    my[step] = y;
    md[step] = ward ? Math.sqrt(Math.max(0, current)) : current;

    // The merged cluster keeps slot y; slot x is retired.
    active[x] = 0;
    size[y] = nx + ny;
    for (let i = 0; i < n; i++) {
      if (active[i] === 0 || i === y) continue;
      const dxi = dist[pos(x, i)] as number;
      const dyi = dist[pos(y, i)] as number;
      let updated: number;
      if (linkage === "single") {
        updated = dxi < dyi ? dxi : dyi;
      } else if (linkage === "complete") {
        updated = dxi > dyi ? dxi : dyi;
      } else if (linkage === "average") {
        updated = (nx * dxi + ny * dyi) / (nx + ny);
      } else {
        const ni = size[i] as number;
        const t = nx + ny + ni;
        updated = ((ni + nx) * dxi + (ni + ny) * dyi - ni * current) / t;
        if (updated < 0) updated = 0;
      }
      dist[pos(y, i)] = updated;
    }
  }

  // Order merges by height (stable) and give clusters their dendrogram ids.
  const order = Array.from({ length: nMerges }, (_, i) => i);
  order.sort((a, b) => (md[a] as number) - (md[b] as number) || a - b);

  const parent = new Int32Array(2 * n - 1);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (v: number): number => {
    let root = v;
    while (parent[root] !== root) root = parent[root] as number;
    while (parent[v] !== root) {
      const nextNode = parent[v] as number;
      parent[v] = root;
      v = nextNode;
    }
    return root;
  };

  for (let k = 0; k < nMerges; k++) {
    const m = order[k] as number;
    const a = find(mx[m] as number);
    const b = find(my[m] as number);
    children[2 * k] = a < b ? a : b;
    children[2 * k + 1] = a < b ? b : a;
    distances[k] = md[m] as number;
    parent[a] = n + k;
    parent[b] = n + k;
  }
  return { children, distances };
}

/**
 * Cut the dendrogram into `nClusters` clusters (same procedure and label order as
 * scikit-learn's `_hc_cut`: repeatedly split the most recently created cluster).
 */
function cutDendrogram(nClusters: number, children: Int32Array, nLeaves: number): Int32Array {
  const labels = new Int32Array(nLeaves);
  if (nLeaves === 1 || nClusters === 1) return labels;

  // Min-heap of negated node ids, laid out exactly like Python's heapq.
  const heap: number[] = [-(2 * nLeaves - 2)];
  const siftDown = (startPos: number, startItemPos: number): void => {
    let position = startItemPos;
    const item = heap[position] as number;
    while (position > startPos) {
      const parentPos = (position - 1) >> 1;
      const parentVal = heap[parentPos] as number;
      if (item < parentVal) {
        heap[position] = parentVal;
        position = parentPos;
        continue;
      }
      break;
    }
    heap[position] = item;
  };
  const siftUp = (startItemPos: number): void => {
    const end = heap.length;
    let position = startItemPos;
    const item = heap[position] as number;
    let child = 2 * position + 1;
    while (child < end) {
      const right = child + 1;
      if (right < end && !((heap[child] as number) < (heap[right] as number))) child = right;
      heap[position] = heap[child] as number;
      position = child;
      child = 2 * position + 1;
    }
    heap[position] = item;
    siftDown(startItemPos, position);
  };
  const push = (item: number): void => {
    heap.push(item);
    siftDown(0, heap.length - 1);
  };
  const pushPop = (item: number): void => {
    if (heap.length > 0 && (heap[0] as number) < item) {
      heap[0] = item;
      siftUp(0);
    }
  };

  for (let i = 0; i < nClusters - 1; i++) {
    const row = -(heap[0] as number) - nLeaves;
    push(-(children[2 * row] as number));
    pushPop(-(children[2 * row + 1] as number));
  }

  const stack: number[] = [];
  for (let c = 0; c < heap.length; c++) {
    stack.push(-(heap[c] as number));
    while (stack.length > 0) {
      const node = stack.pop() as number;
      if (node < nLeaves) {
        labels[node] = c;
      } else {
        const row = node - nLeaves;
        stack.push(children[2 * row] as number, children[2 * row + 1] as number);
      }
    }
  }
  return labels;
}

/**
 * Agglomerative (hierarchical) clustering.
 *
 * Bottom-up clustering that starts with each sample as its own cluster
 * and iteratively merges the closest pair of clusters until `nClusters`
 * clusters remain (or, with `distanceThreshold`, until the closest pair is
 * at least that far apart).
 *
 * Supported linkage criteria:
 * - **single**: minimum distance between clusters
 * - **complete**: maximum distance between clusters
 * - **average**: average distance between clusters
 * - **ward**: minimizes the total within-cluster variance (Euclidean only)
 *
 * Distances are Euclidean. The dendrogram is built in O(n^2) time and
 * O(n^2) memory, so very large inputs (tens of thousands of samples) are
 * not practical.
 *
 * @example
 * ```ts
 * import { AgglomerativeClustering } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [1.5, 1.8], [5, 8], [8, 8]]);
 * const agg = new AgglomerativeClustering({ nClusters: 2 });
 * agg.fit(X);
 * console.log(agg.labels);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox Clustering}
 */
export class AgglomerativeClustering implements Clusterer {
  private nClustersParam: number | null;
  private linkage: Linkage;
  private distanceThreshold: number | undefined;

  private labels_?: Tensor;
  private children_?: Int32Array;
  private distances_?: Float64Array;
  private nClustersFitted_ = 0;
  private nFeaturesIn_ = 0;
  // Training samples and per-cluster centroids (flat, row-major), kept for predict().
  private fitData_?: Float64Array;
  private centers_?: Float64Array;
  private fitted = false;

  /**
   * Create an agglomerative clustering model.
   *
   * @param options - Configuration options
   * @param options.nClusters - Number of clusters to find, integer >= 1 (default: 2). Pass `null` to use `distanceThreshold` instead.
   * @param options.linkage - "single", "complete", "average" or "ward" (default: "ward")
   * @param options.distanceThreshold - Merge distance at or above which clusters are not joined. When set, `nClusters` defaults to `null`; setting both is an error.
   * @throws {InvalidParameterError} If an option is out of range, or if both or neither of `nClusters` and `distanceThreshold` are set
   */
  constructor(
    options: {
      readonly nClusters?: number | null;
      readonly linkage?: Linkage;
      readonly distanceThreshold?: number;
    } = {}
  ) {
    this.distanceThreshold = options.distanceThreshold;
    this.nClustersParam =
      options.nClusters === undefined
        ? options.distanceThreshold === undefined
          ? 2
          : null
        : options.nClusters;
    this.linkage = options.linkage ?? "ward";

    if (
      this.nClustersParam !== null &&
      (!Number.isInteger(this.nClustersParam) || this.nClustersParam < 1)
    ) {
      throw new InvalidParameterError(
        "nClusters must be an integer >= 1",
        "nClusters",
        this.nClustersParam
      );
    }
    AgglomerativeClustering.checkLinkage(this.linkage);
    if (
      this.distanceThreshold !== undefined &&
      (!Number.isFinite(this.distanceThreshold) || this.distanceThreshold < 0)
    ) {
      throw new InvalidParameterError(
        "distanceThreshold must be a finite number >= 0",
        "distanceThreshold",
        this.distanceThreshold
      );
    }
    this.checkClusterTarget();
  }

  private static checkLinkage(value: unknown): void {
    if (value !== "single" && value !== "complete" && value !== "average" && value !== "ward") {
      throw new InvalidParameterError(
        `linkage must be "single", "complete", "average", or "ward"`,
        "linkage",
        value
      );
    }
  }

  /** Exactly one of `nClusters` and `distanceThreshold` must be set. */
  private checkClusterTarget(): void {
    if (this.nClustersParam === null && this.distanceThreshold === undefined) {
      throw new InvalidParameterError(
        "Set either nClusters or distanceThreshold",
        "nClusters",
        this.nClustersParam
      );
    }
    if (this.nClustersParam !== null && this.distanceThreshold !== undefined) {
      throw new InvalidParameterError(
        "nClusters and distanceThreshold cannot both be set; pass nClusters: null to use distanceThreshold",
        "distanceThreshold",
        this.distanceThreshold
      );
    }
  }

  /**
   * Build the cluster hierarchy of X and cut it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf values
   * @throws {InvalidParameterError} If there are fewer samples than `nClusters`
   * @throws {MemoryError} If the pairwise distance matrix would be too large
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    this.checkClusterTarget();
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    if (this.nClustersParam !== null && nSamples < this.nClustersParam) {
      throw new InvalidParameterError(
        `n_samples=${nSamples} should be >= n_clusters=${this.nClustersParam}`,
        "nClusters",
        this.nClustersParam
      );
    }

    // Own the training rows so later edits of X cannot change predict().
    const data = toFloat64View(X).slice();
    const { children, distances } = buildDendrogram(data, nSamples, nFeatures, this.linkage);

    let nClusters: number;
    if (this.nClustersParam !== null) {
      nClusters = this.nClustersParam;
    } else {
      const threshold = this.distanceThreshold as number;
      nClusters = 1;
      for (let i = 0; i < distances.length; i++) {
        if ((distances[i] as number) >= threshold) nClusters++;
      }
    }

    const labels = cutDendrogram(nClusters, children, nSamples);

    const centers = new Float64Array(nClusters * nFeatures);
    const counts = new Float64Array(nClusters);
    for (let i = 0; i < nSamples; i++) {
      const c = labels[i] as number;
      counts[c] = (counts[c] as number) + 1;
      for (let f = 0; f < nFeatures; f++) {
        centers[c * nFeatures + f] =
          (centers[c * nFeatures + f] as number) + (data[i * nFeatures + f] as number);
      }
    }
    for (let c = 0; c < nClusters; c++) {
      const cnt = counts[c] as number;
      for (let f = 0; f < nFeatures; f++) {
        centers[c * nFeatures + f] = (centers[c * nFeatures + f] as number) / cnt;
      }
    }

    this.labels_ = tensor(labels);
    this.children_ = children;
    this.distances_ = distances;
    this.nClustersFitted_ = nClusters;
    this.nFeaturesIn_ = nFeatures;
    this.fitData_ = data;
    this.centers_ = centers;
    this.fitted = true;
    return this;
  }

  /**
   * Predict cluster labels for new samples using nearest-neighbor assignment.
   *
   * Each new point is assigned the label of its nearest training sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.labels_ || !this.fitData_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "AgglomerativeClustering");

    const nSamples = X.shape[0] ?? 0;
    const d = this.nFeaturesIn_;
    const data = toFloat64View(X);
    const train = this.fitData_;
    const nTrain = train.length / d;
    const trainLabels = this.labels_.data as Int32Array;
    const trainOffset = this.labels_.offset;
    const result = new Int32Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let bestDist = Infinity;
      let bestLabel = 0;
      for (let t = 0; t < nTrain; t++) {
        let dist = 0;
        for (let f = 0; f < d; f++) {
          const diff = (data[i * d + f] as number) - (train[t * d + f] as number);
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestLabel = trainLabels[trainOffset + t] as number;
        }
      }
      result[i] = bestLabel;
    }

    return tensor(result);
  }

  /**
   * Fit the model and return the cluster labels of the training samples.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns Cluster labels of shape (n_samples,)
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_ as Tensor;
  }

  /**
   * Cluster labels of the training samples.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted to access labels");
    }
    return this.labels_;
  }

  /**
   * Centroid (mean of the training samples) of every cluster, indexed by label.
   *
   * @returns Tensor of shape (n_clusters, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.centers_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted to access cluster centers");
    }
    return tensor(this.centers_.slice()).reshape([this.nClustersFitted_, this.nFeaturesIn_]);
  }

  /**
   * Number of clusters found.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nClusters(): number {
    if (!this.fitted) {
      throw new NotFittedError("AgglomerativeClustering must be fitted to access nClusters");
    }
    return this.nClustersFitted_;
  }

  /**
   * Number of training samples (leaves of the dendrogram).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nLeaves(): number {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted to access nLeaves");
    }
    return this.labels_.size;
  }

  /**
   * Children of every non-leaf node of the dendrogram. Row `i` lists the two
   * nodes merged at step `i`; ids below `n_samples` are samples and id
   * `n_samples + i` is the cluster formed at step `i`.
   *
   * @returns Int32 tensor of shape (n_samples - 1, 2)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get children(): Tensor {
    if (!this.fitted || !this.children_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted to access children");
    }
    return tensor(this.children_.slice()).reshape([this.children_.length / 2, 2]);
  }

  /**
   * Distance at which every merge of the dendrogram happened (for ward linkage,
   * the Ward merge height sqrt(2 * n_a * n_b / (n_a + n_b)) * ||c_a - c_b||).
   *
   * @returns Float64 tensor of shape (n_samples - 1,), non-decreasing
   * @throws {NotFittedError} If the model has not been fitted
   */
  get distances(): Tensor {
    if (!this.fitted || !this.distances_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted to access distances");
    }
    return tensor(this.distances_.slice());
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClustersParam,
      linkage: this.linkage,
      distanceThreshold: this.distanceThreshold,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set (nClusters, linkage, distanceThreshold)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nClusters":
          if (
            value !== null &&
            (typeof value !== "number" || !Number.isInteger(value) || value < 1)
          ) {
            throw new InvalidParameterError(
              "nClusters must be an integer >= 1 or null",
              "nClusters",
              value
            );
          }
          this.nClustersParam = value;
          break;
        case "linkage":
          AgglomerativeClustering.checkLinkage(value);
          this.linkage = value as Linkage;
          break;
        case "distanceThreshold":
          if (
            value !== undefined &&
            value !== null &&
            (typeof value !== "number" || !Number.isFinite(value) || value < 0)
          ) {
            throw new InvalidParameterError(
              "distanceThreshold must be a finite number >= 0",
              "distanceThreshold",
              value
            );
          }
          this.distanceThreshold = value ?? undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
