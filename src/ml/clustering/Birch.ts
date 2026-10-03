/**
 * BIRCH: Balanced Iterative Reducing and Clustering using Hierarchies.
 *
 * Incrementally builds a CF (Clustering Feature) tree to summarize
 * data, then applies a global clustering step. Memory-efficient for
 * large datasets.
 *
 * @module ml/clustering/Birch
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, ShapeError, warn } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";
import { AgglomerativeClustering } from "./AgglomerativeClustering";

/**
 * Clustering feature of a group of points: count, linear sum and total squared
 * norm. Entries of inner nodes summarize everything below their child node.
 */
class CFSubcluster {
  n: number;
  readonly ls: Float64Array;
  ss: number;
  readonly centroid: Float64Array;
  child: CFNode | null = null;

  constructor(d: number, point?: Float64Array) {
    this.ls = new Float64Array(d);
    this.centroid = new Float64Array(d);
    this.n = 0;
    this.ss = 0;
    if (point) {
      this.n = 1;
      this.ls.set(point);
      this.centroid.set(point);
      let ss = 0;
      for (let j = 0; j < d; j++) ss += (point[j] as number) * (point[j] as number);
      this.ss = ss;
    }
  }

  /** Add the clustering feature of `other` to this one. */
  update(other: CFSubcluster): void {
    this.n += other.n;
    this.ss += other.ss;
    const d = this.ls.length;
    for (let j = 0; j < d; j++) {
      const sum = (this.ls[j] as number) + (other.ls[j] as number);
      this.ls[j] = sum;
      this.centroid[j] = sum / this.n;
    }
  }

  /**
   * Absorb `nominee` if the merged subcluster keeps a radius of at most `threshold`.
   *
   * @returns Whether the nominee was absorbed
   */
  merge(nominee: CFSubcluster, threshold: number): boolean {
    const d = this.ls.length;
    const n = this.n + nominee.n;
    const ss = this.ss + nominee.ss;
    let sqNorm = 0;
    for (let j = 0; j < d; j++) {
      const c = ((this.ls[j] as number) + (nominee.ls[j] as number)) / n;
      sqNorm += c * c;
    }
    // Mean squared distance of the members to the merged centroid.
    if (ss / n - sqNorm > threshold * threshold) return false;
    this.update(nominee);
    return true;
  }
}

/** Node of the CF tree. Leaves are chained so that all leaf entries can be listed in order. */
class CFNode {
  readonly isLeaf: boolean;
  subclusters: CFSubcluster[] = [];
  prevLeaf: CFNode | null = null;
  nextLeaf: CFNode | null = null;

  constructor(isLeaf: boolean) {
    this.isLeaf = isLeaf;
  }

  /**
   * Insert a subcluster below this node.
   *
   * @returns True if this node now holds too many entries and must be split by the caller
   */
  insert(sub: CFSubcluster, threshold: number, branching: number): boolean {
    if (this.subclusters.length === 0) {
      this.subclusters.push(sub);
      return false;
    }

    const d = sub.centroid.length;
    let closest = 0;
    let bestDist = Infinity;
    for (let idx = 0; idx < this.subclusters.length; idx++) {
      const c = (this.subclusters[idx] as CFSubcluster).centroid;
      let dist = 0;
      for (let j = 0; j < d; j++) {
        const diff = (c[j] as number) - (sub.centroid[j] as number);
        dist += diff * diff;
      }
      if (dist < bestDist) {
        bestDist = dist;
        closest = idx;
      }
    }
    const nearest = this.subclusters[closest] as CFSubcluster;

    if (nearest.child !== null) {
      const needSplit = nearest.child.insert(sub, threshold, branching);
      if (!needSplit) {
        nearest.update(sub);
        return false;
      }
      const [first, second] = splitNode(nearest.child, d);
      this.subclusters[closest] = first;
      this.subclusters.push(second);
      return this.subclusters.length > branching;
    }

    if (nearest.merge(sub, threshold)) return false;
    this.subclusters.push(sub);
    return this.subclusters.length > branching;
  }
}

/**
 * Split an over-full node in two. The two entries that are farthest apart seed the
 * new nodes and every other entry joins the seed it is closer to (ties go to the
 * second node).
 *
 * @returns Subclusters summarizing the two new nodes
 */
function splitNode(node: CFNode, d: number): [CFSubcluster, CFSubcluster] {
  const first = new CFSubcluster(d);
  const second = new CFSubcluster(d);
  const nodeA = new CFNode(node.isLeaf);
  const nodeB = new CFNode(node.isLeaf);
  first.child = nodeA;
  second.child = nodeB;

  if (node.isLeaf) {
    if (node.prevLeaf) node.prevLeaf.nextLeaf = nodeA;
    nodeA.prevLeaf = node.prevLeaf;
    nodeA.nextLeaf = nodeB;
    nodeB.prevLeaf = nodeA;
    nodeB.nextLeaf = node.nextLeaf;
    if (node.nextLeaf) node.nextLeaf.prevLeaf = nodeB;
  }

  const subs = node.subclusters;
  const m = subs.length;
  const sqDist = (a: CFSubcluster, b: CFSubcluster): number => {
    let s = 0;
    for (let j = 0; j < d; j++) {
      const diff = (a.centroid[j] as number) - (b.centroid[j] as number);
      s += diff * diff;
    }
    return s;
  };

  let seedA = 0;
  let seedB = 0;
  let farthest = -Infinity;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const dist = sqDist(subs[i] as CFSubcluster, subs[j] as CFSubcluster);
      if (dist > farthest) {
        farthest = dist;
        seedA = i;
        seedB = j;
      }
    }
  }

  for (let idx = 0; idx < m; idx++) {
    const sub = subs[idx] as CFSubcluster;
    const closerToA =
      idx === seedA ||
      sqDist(sub, subs[seedA] as CFSubcluster) < sqDist(sub, subs[seedB] as CFSubcluster);
    if (closerToA) {
      nodeA.subclusters.push(sub);
      first.update(sub);
    } else {
      nodeB.subclusters.push(sub);
      second.update(sub);
    }
  }
  return [first, second];
}

/**
 * BIRCH clustering algorithm.
 *
 * Phase 1: Build a CF tree by scanning the data. Each sample joins the closest
 * leaf entry if the merged entry keeps a radius (root mean squared distance of its
 * members to their centroid) of at most `threshold`; otherwise it starts a new
 * entry. A node with more than `branchingFactor` entries is split.
 * Phase 2: Apply Ward agglomerative clustering on the leaf entries
 * to produce final `nClusters` clusters. With `nClusters: null` this step is
 * skipped and every leaf entry is its own cluster.
 *
 * The tree is built on mean-centered data, so large constant offsets in `X` do not
 * degrade the radius test. Samples are labeled by their closest leaf entry.
 *
 * @example
 * ```ts
 * import { Birch } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, 0], [10, 10], [10.1, 10]]);
 * const birch = new Birch({ nClusters: 2, threshold: 0.5 });
 * birch.fit(X);
 * console.log(birch.labels);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox Clustering}
 */
export class Birch implements Clusterer {
  private nClusters: number | null;
  private threshold: number;
  private branchingFactor: number;

  private labels_?: Tensor;
  private nFeaturesIn_ = 0;
  // CF tree (built in coordinates shifted by `shift_`), kept for partialFit().
  private root_: CFNode | null = null;
  private dummyLeaf_: CFNode | null = null;
  private treeThreshold_ = 0;
  private treeBranching_ = 0;
  private shift_?: Float64Array;
  // Leaf entries in shifted coordinates, their labels, and the final cluster centers.
  private subclusterCenters_?: Float64Array;
  private subclusterLabels_?: Int32Array;
  private clusterCenters_?: Float64Array;
  private nClustersFitted_ = 0;
  private fitted = false;

  /**
   * Create a BIRCH model.
   *
   * @param options - Configuration options
   * @param options.nClusters - Number of final clusters, integer >= 1, or `null` to keep every leaf entry as a cluster (default: 3)
   * @param options.threshold - Maximum radius of a leaf entry after absorbing a sample, > 0 (default: 0.5)
   * @param options.branchingFactor - Maximum number of entries per tree node, integer >= 2 (default: 50)
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly nClusters?: number | null;
      readonly threshold?: number;
      readonly branchingFactor?: number;
    } = {}
  ) {
    this.nClusters = options.nClusters === undefined ? 3 : options.nClusters;
    this.threshold = options.threshold ?? 0.5;
    this.branchingFactor = options.branchingFactor ?? 50;

    if (this.nClusters !== null && (!Number.isInteger(this.nClusters) || this.nClusters < 1)) {
      throw new InvalidParameterError(
        "nClusters must be an integer >= 1 or null",
        "nClusters",
        this.nClusters
      );
    }
    if (!Number.isFinite(this.threshold) || this.threshold <= 0) {
      throw new InvalidParameterError(
        "threshold must be a finite number > 0",
        "threshold",
        this.threshold
      );
    }
    if (!Number.isInteger(this.branchingFactor) || this.branchingFactor < 2) {
      throw new InvalidParameterError(
        "branchingFactor must be an integer >= 2",
        "branchingFactor",
        this.branchingFactor
      );
    }
  }

  /**
   * Build the CF tree from X (discarding any previous tree) and cluster its leaf entries.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    this.root_ = null;
    this.dummyLeaf_ = null;
    return this.partialFit(X);
  }

  /**
   * Add the samples of X to the existing CF tree (or start one) and recompute the
   * global clustering. Unlike `fit`, previously inserted samples are kept.
   *
   * Without an argument only the global clustering step is repeated, for example
   * after changing `nClusters` with `setParams`. The tree keeps the `threshold` and
   * `branchingFactor` it was started with.
   *
   * @param X - New samples of shape (n_samples, n_features); `labels` then holds the labels of these samples
   * @returns this - The updated estimator
   * @throws {ShapeError} If X is not 2D or has a different number of features than earlier batches
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   * @throws {NotFittedError} If called without X before any data was seen
   */
  partialFit(X?: Tensor, _y?: Tensor): this {
    if (X === undefined) {
      if (!this.root_ || !this.dummyLeaf_) {
        throw new NotFittedError("Birch needs data before partialFit() can be called without X");
      }
      this.finalize();
      return this;
    }

    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    if (this.root_ !== null && d !== this.nFeaturesIn_) {
      throw new ShapeError(
        `X has ${d} features but the CF tree was built with ${this.nFeaturesIn_} features`
      );
    }
    const data = toFloat64View(X);

    if (this.root_ === null || this.dummyLeaf_ === null) {
      const shift = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) shift[j] = (shift[j] as number) + (data[i * d + j] as number);
      }
      for (let j = 0; j < d; j++) shift[j] = (shift[j] as number) / n;
      this.shift_ = shift;
      this.root_ = new CFNode(true);
      this.dummyLeaf_ = new CFNode(true);
      this.dummyLeaf_.nextLeaf = this.root_;
      this.root_.prevLeaf = this.dummyLeaf_;
      this.treeThreshold_ = this.threshold;
      this.treeBranching_ = this.branchingFactor;
      this.nFeaturesIn_ = d;
    }

    const shift = this.shift_ as Float64Array;
    const point = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) point[j] = (data[i * d + j] as number) - (shift[j] as number);
      const sub = new CFSubcluster(d, point);
      const split = this.root_.insert(sub, this.treeThreshold_, this.treeBranching_);
      if (split) {
        const [first, second] = splitNode(this.root_, d);
        const newRoot = new CFNode(false);
        newRoot.subclusters.push(first, second);
        this.root_ = newRoot;
      }
    }

    this.finalize();
    this.labels_ = tensor(this.nearestSubclusterLabels(data, n, d));
    return this;
  }

  /** Collect the leaf entries and run the global clustering step. */
  private finalize(): void {
    const d = this.nFeaturesIn_;
    const entries: CFSubcluster[] = [];
    let leaf = (this.dummyLeaf_ as CFNode).nextLeaf;
    while (leaf !== null) {
      for (const sub of leaf.subclusters) entries.push(sub);
      leaf = leaf.nextLeaf;
    }
    const m = entries.length;
    const centers = new Float64Array(m * d);
    const counts = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const sub = entries[i] as CFSubcluster;
      centers.set(sub.centroid, i * d);
      counts[i] = sub.n;
    }

    let labels: Int32Array;
    const requested = this.nClusters;
    if (requested === null || m < requested) {
      if (requested !== null) {
        warn(
          `Number of subclusters found (${m}) by BIRCH is less than (${requested}). ` +
            "Decrease the threshold.",
          "ConvergenceWarning",
          "Birch"
        );
      }
      labels = new Int32Array(m);
      for (let i = 0; i < m; i++) labels[i] = i;
    } else {
      const agglomerative = new AgglomerativeClustering({ nClusters: requested, linkage: "ward" });
      agglomerative.fit(tensor(centers.slice()).reshape([m, d]));
      labels = (agglomerative.labels.data as Int32Array).slice(
        agglomerative.labels.offset,
        agglomerative.labels.offset + m
      );
    }

    let k = 0;
    for (let i = 0; i < m; i++) if ((labels[i] as number) + 1 > k) k = (labels[i] as number) + 1;
    const clusterCenters = new Float64Array(k * d);
    const clusterCounts = new Float64Array(k);
    for (let i = 0; i < m; i++) {
      const c = labels[i] as number;
      const w = counts[i] as number;
      clusterCounts[c] = (clusterCounts[c] as number) + w;
      for (let j = 0; j < d; j++) {
        clusterCenters[c * d + j] =
          (clusterCenters[c * d + j] as number) + w * (centers[i * d + j] as number);
      }
    }
    for (let c = 0; c < k; c++) {
      for (let j = 0; j < d; j++) {
        clusterCenters[c * d + j] =
          (clusterCenters[c * d + j] as number) / (clusterCounts[c] as number);
      }
    }

    this.subclusterCenters_ = centers;
    this.subclusterLabels_ = labels;
    this.clusterCenters_ = clusterCenters;
    this.nClustersFitted_ = k;
    this.fitted = true;
  }

  /** Label rows of `data` (original coordinates) with the label of the closest leaf entry. */
  private nearestSubclusterLabels(data: Float64Array, n: number, d: number): Int32Array {
    const centers = this.subclusterCenters_ as Float64Array;
    const subLabels = this.subclusterLabels_ as Int32Array;
    const shift = this.shift_ as Float64Array;
    const m = subLabels.length;
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let best = 0;
      for (let s = 0; s < m; s++) {
        let dist = 0;
        for (let j = 0; j < d; j++) {
          const diff =
            (data[i * d + j] as number) - (shift[j] as number) - (centers[s * d + j] as number);
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          best = s;
        }
      }
      out[i] = subLabels[best] as number;
    }
    return out;
  }

  /**
   * Assign each sample the label of its closest leaf entry.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.subclusterCenters_) {
      throw new NotFittedError("Birch must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "Birch");
    const n = X.shape[0] ?? 0;
    return tensor(this.nearestSubclusterLabels(toFloat64View(X), n, this.nFeaturesIn_));
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
   * Cluster labels of the samples passed to the last `fit` or `partialFit` call.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("Birch must be fitted to access labels");
    }
    return this.labels_;
  }

  /**
   * Centers of the final clusters: the sample-count weighted mean of their leaf entries.
   *
   * @returns Float64 tensor of shape (n_clusters, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_ || !this.shift_) {
      throw new NotFittedError("Birch must be fitted to access cluster centers");
    }
    return this.unshifted(this.clusterCenters_, this.nClustersFitted_);
  }

  /**
   * Centroids of the leaf entries of the CF tree.
   *
   * @returns Float64 tensor of shape (n_subclusters, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get subclusterCenters(): Tensor {
    if (!this.fitted || !this.subclusterCenters_ || !this.subclusterLabels_) {
      throw new NotFittedError("Birch must be fitted to access subcluster centers");
    }
    return this.unshifted(this.subclusterCenters_, this.subclusterLabels_.length);
  }

  /**
   * Final cluster of every leaf entry.
   *
   * @returns Int32 tensor of shape (n_subclusters,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get subclusterLabels(): Tensor {
    if (!this.fitted || !this.subclusterLabels_) {
      throw new NotFittedError("Birch must be fitted to access subcluster labels");
    }
    return tensor(this.subclusterLabels_.slice());
  }

  /** Convert flat shifted centers back to the coordinates of the input data. */
  private unshifted(flat: Float64Array, rows: number): Tensor {
    const d = this.nFeaturesIn_;
    const shift = this.shift_ as Float64Array;
    const out = new Float64Array(rows * d);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < d; j++) {
        out[i * d + j] = (flat[i * d + j] as number) + (shift[j] as number);
      }
    }
    return tensor(out).reshape([rows, d]);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClusters,
      threshold: this.threshold,
      branchingFactor: this.branchingFactor,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * `threshold` and `branchingFactor` apply to the next `fit`; `nClusters` applies
   * to the next `fit` or `partialFit()`.
   *
   * @param params - Parameters to set (nClusters, threshold, branchingFactor)
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
          this.nClusters = value;
          break;
        case "threshold":
          if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new InvalidParameterError(
              "threshold must be a finite number > 0",
              "threshold",
              value
            );
          }
          this.threshold = value;
          break;
        case "branchingFactor":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              "branchingFactor must be an integer >= 2",
              "branchingFactor",
              value
            );
          }
          this.branchingFactor = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
