/**
 * OPTICS: Ordering Points To Identify the Clustering Structure.
 *
 * A density-based clustering algorithm related to DBSCAN that creates
 * a reachability ordering of the data points. Unlike DBSCAN, OPTICS
 * can detect clusters of varying density.
 *
 * @module ml/clustering/OPTICS
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { kthSmallest } from "../_internal";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";

/** Writes the squared Euclidean distance from row `i` to every row of `data` into `out`. */
function squaredDistanceRow(
  data: Float64Array,
  i: number,
  n: number,
  d: number,
  out: Float64Array
): void {
  const iBase = i * d;
  for (let j = 0; j < n; j++) {
    const jBase = j * d;
    let s = 0;
    for (let f = 0; f < d; f++) {
      const diff = (data[iBase + f] as number) - (data[jBase + f] as number);
      s += diff * diff;
    }
    out[j] = s;
  }
}

/** A steep downward area of the reachability plot (positions are indices into the ordering). */
type SteepDownArea = { readonly start: number; readonly end: number; mib: number };

/**
 * Extend a steep region from `start` as far as it goes. Mirrors `_extend_region` of
 * scikit-learn: at most `minSamples` consecutive points that are neither steep nor
 * moving in the opposite direction may be absorbed.
 */
function extendRegion(
  steep: Uint8Array,
  opposite: Uint8Array,
  start: number,
  minSamples: number
): number {
  const n = steep.length;
  let nonSteep = 0;
  let end = start;
  for (let index = start; index < n; index++) {
    if (steep[index]) {
      nonSteep = 0;
      end = index;
    } else if (!opposite[index]) {
      nonSteep++;
      if (nonSteep > minSamples) break;
    } else {
      return end;
    }
  }
  return end;
}

/**
 * Predecessor correction (Algorithm 2 of Schubert and Gertz, 2018). Returns the corrected
 * `[start, end]` or `null` when nothing is left of the cluster.
 */
function correctPredecessor(
  plot: Float64Array,
  predecessorPlot: Int32Array,
  ordering: Int32Array,
  start: number,
  end: number
): [number, number] | null {
  let e = end;
  while (start < e) {
    if ((plot[start] as number) > (plot[e] as number)) return [start, e];
    const pe = predecessorPlot[e] as number;
    for (let i = start; i < e; i++) {
      if (pe === ordering[i]) return [start, e];
    }
    e--;
  }
  return null;
}

/**
 * Xi-steep cluster extraction (Figure 19 of the OPTICS paper, with the corrections made by
 * scikit-learn). Works on the reachability plot, i.e. reachability in OPTICS order.
 *
 * @returns Clusters as inclusive `[start, end]` positions in the ordering, smaller
 *   clusters before the larger clusters that contain them
 */
function extractXiHierarchy(
  reachabilityInOrder: Float64Array,
  predecessorInOrder: Int32Array,
  ordering: Int32Array,
  xi: number,
  minSamples: number,
  minClusterSize: number,
  predecessorCorrection: boolean
): Array<[number, number]> {
  const n = reachabilityInOrder.length;
  // The trailing infinity lets a cluster that runs to the end of the plot be closed.
  const plot = new Float64Array(n + 1);
  plot.set(reachabilityInOrder);
  plot[n] = Infinity;

  const xiComplement = 1 - xi;
  const steepUp = new Uint8Array(n);
  const steepDown = new Uint8Array(n);
  const down = new Uint8Array(n);
  const up = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    // Inf / Inf and 0 / 0 are NaN, which fails every comparison below (as in numpy).
    const ratio = (plot[i] as number) / (plot[i + 1] as number);
    steepUp[i] = ratio <= xiComplement ? 1 : 0;
    steepDown[i] = ratio >= 1 / xiComplement ? 1 : 0;
    down[i] = ratio > 1 ? 1 : 0;
    up[i] = ratio < 1 ? 1 : 0;
  }

  let sdas: SteepDownArea[] = [];
  const clusters: Array<[number, number]> = [];
  let index = 0;
  let mib = 0;

  const updateFilterSdas = (): void => {
    if (mib === Infinity) {
      sdas = [];
      return;
    }
    sdas = sdas.filter((sda) => mib <= (plot[sda.start] as number) * xiComplement);
    for (const sda of sdas) sda.mib = Math.max(sda.mib, mib);
  };

  for (let steepIndex = 0; steepIndex < n; steepIndex++) {
    if (!steepUp[steepIndex] && !steepDown[steepIndex]) continue;
    // Part of an area that was already discovered.
    if (steepIndex < index) continue;

    for (let i = index; i <= steepIndex; i++) mib = Math.max(mib, plot[i] as number);

    if (steepDown[steepIndex]) {
      updateFilterSdas();
      const dEnd = extendRegion(steepDown, up, steepIndex, minSamples);
      sdas.push({ start: steepIndex, end: dEnd, mib: 0 });
      index = dEnd + 1;
      mib = plot[index] as number;
    } else {
      updateFilterSdas();
      const uStart = steepIndex;
      const uEnd = extendRegion(steepUp, down, uStart, minSamples);
      index = uEnd + 1;
      mib = plot[index] as number;

      const found: Array<[number, number]> = [];
      for (const sda of sdas) {
        let cStart = sda.start;
        let cEnd = uEnd;

        // Line (**), sc2*
        if ((plot[cEnd + 1] as number) * xiComplement < sda.mib) continue;

        // Definition 11, criterion 4
        const dMax = plot[sda.start] as number;
        if (dMax * xiComplement >= (plot[cEnd + 1] as number)) {
          while ((plot[cStart + 1] as number) > (plot[cEnd + 1] as number) && cStart < sda.end) {
            cStart++;
          }
        } else if ((plot[cEnd + 1] as number) * xiComplement >= dMax) {
          while ((plot[cEnd - 1] as number) > dMax && cEnd > uStart) cEnd--;
        }

        if (predecessorCorrection) {
          const corrected = correctPredecessor(plot, predecessorInOrder, ordering, cStart, cEnd);
          if (corrected === null) continue;
          [cStart, cEnd] = corrected;
        }

        // Definition 11, criterion 3a
        if (cEnd - cStart + 1 < minClusterSize) continue;
        // Criterion 1
        if (cStart > sda.end) continue;
        // Criterion 2
        if (cEnd < uStart) continue;

        found.push([cStart, cEnd]);
      }
      // Smaller clusters first.
      found.reverse();
      for (const c of found) clusters.push(c);
    }
  }
  return clusters;
}

function checkMinSamples(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("minSamples must be an integer >= 1", "minSamples", value);
  }
  return value;
}

function checkMaxEps(value: unknown): number {
  if (typeof value !== "number" || !(value > 0)) {
    throw new InvalidParameterError("maxEps must be > 0", "maxEps", value);
  }
  return value;
}

function checkClusterMethod(value: unknown): "xi" | "dbscan" {
  if (value !== "xi" && value !== "dbscan") {
    throw new InvalidParameterError(
      `clusterMethod must be "xi" or "dbscan"`,
      "clusterMethod",
      value
    );
  }
  return value;
}

function checkXi(value: unknown): number {
  if (typeof value !== "number" || !(value > 0 && value < 1)) {
    throw new InvalidParameterError("xi must be in (0, 1)", "xi", value);
  }
  return value;
}

function checkEps(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !(value > 0))) {
    throw new InvalidParameterError("eps must be a positive number or undefined", "eps", value);
  }
  return value;
}

function checkMinClusterSize(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isInteger(value) || value < 2)) {
    throw new InvalidParameterError(
      "minClusterSize must be an integer >= 2 or undefined",
      "minClusterSize",
      value
    );
  }
  return value;
}

function checkBoolean(name: string, value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(`${name} must be a boolean`, name, value);
  }
  return value;
}

/**
 * OPTICS clustering algorithm.
 *
 * Computes the core distances, the reachability distances and the visiting order of
 * every sample (Euclidean metric), then extracts flat clusters from the reachability
 * plot with one of two methods:
 *
 * - `"dbscan"` (default): cut the plot at `eps`, like DBSCAN with that radius. When `eps`
 *   is not given it defaults to `maxEps`; if `maxEps` is infinite, the 75th percentile of
 *   the finite reachability distances is used instead (a Deepbox heuristic; scikit-learn
 *   would put everything in one cluster).
 * - `"xi"`: the Xi-steep method of the paper, with predecessor correction, as in
 *   `sklearn.cluster.OPTICS(cluster_method="xi")`. Cluster boundaries are the steep
 *   regions of the plot whose reachability ratio is at least `xi`.
 *
 * Noise samples get label -1. As in scikit-learn, `minSamples` counts the sample itself,
 * and samples are visited in index order, with ties in reachability broken by the lower index.
 *
 * Distances are computed on the fly, so memory use is linear in the number of samples;
 * run time is quadratic.
 *
 * @example
 * ```ts
 * import { OPTICS } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, 0], [10, 10], [10.1, 10]]);
 * const optics = new OPTICS({ minSamples: 2 });
 * optics.fit(X);
 * console.log(optics.labels);
 * console.log(optics.reachability);
 * ```
 */
export class OPTICS implements Clusterer {
  private minSamples: number;
  private maxEps: number;
  private clusterMethod: "xi" | "dbscan";
  private xi: number;
  private eps: number | undefined;
  private minClusterSize: number | undefined;
  private predecessorCorrection: boolean;

  private labels_?: Tensor;
  private reachability_?: Float64Array;
  private ordering_?: Int32Array;
  private coreDistances_?: Float64Array;
  private predecessor_?: Int32Array;
  private clusterHierarchy_: Array<readonly [number, number]> = [];
  private fitData_?: Float64Array;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * Create a new OPTICS model.
   *
   * @param options - Configuration options
   * @param options.minSamples - Number of samples in a neighborhood (the sample itself included) for a point to be a core point (default: 5)
   * @param options.maxEps - Largest neighborhood radius considered; smaller values run faster (default: Infinity)
   * @param options.clusterMethod - Cluster extraction method: 'dbscan' or 'xi' (default: 'dbscan')
   * @param options.xi - Minimum steepness on the reachability plot that marks a cluster boundary, in (0, 1); only for 'xi' (default: 0.05)
   * @param options.eps - Radius at which the reachability plot is cut; only for 'dbscan', must not exceed `maxEps` (default: `maxEps`, see the class description)
   * @param options.minClusterSize - Smallest cluster accepted by 'xi' (default: `max(minSamples, 2)`)
   * @param options.predecessorCorrection - Correct 'xi' clusters using the predecessor of each point (default: true)
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly minSamples?: number;
      readonly maxEps?: number;
      readonly clusterMethod?: "xi" | "dbscan";
      readonly xi?: number;
      readonly eps?: number;
      readonly minClusterSize?: number;
      readonly predecessorCorrection?: boolean;
    } = {}
  ) {
    this.minSamples = checkMinSamples(options.minSamples ?? 5);
    this.maxEps = checkMaxEps(options.maxEps ?? Infinity);
    this.clusterMethod = checkClusterMethod(options.clusterMethod ?? "dbscan");
    this.xi = checkXi(options.xi ?? 0.05);
    this.eps = checkEps(options.eps);
    this.minClusterSize = checkMinClusterSize(options.minClusterSize);
    this.predecessorCorrection = checkBoolean(
      "predecessorCorrection",
      options.predecessorCorrection ?? true
    );
  }

  /**
   * Compute the OPTICS ordering and extract clusters.
   *
   * Calling `fit` again replaces the previous model.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   * @throws {InvalidParameterError} If `minSamples` exceeds the number of samples, or `eps`
   *   exceeds `maxEps` with the 'dbscan' method
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;

    if (this.minSamples > n) {
      throw new InvalidParameterError(
        `minSamples=${this.minSamples} must be <= n_samples=${n}`,
        "minSamples",
        this.minSamples
      );
    }
    if (this.clusterMethod === "dbscan" && this.eps !== undefined && this.eps > this.maxEps) {
      throw new InvalidParameterError(
        `eps=${this.eps} must not exceed maxEps=${this.maxEps}`,
        "eps",
        this.eps
      );
    }

    const data = toFloat64View(X);
    const row = new Float64Array(n);
    const scratch = new Float64Array(n);

    // Core distance: distance to the minSamples-th nearest sample, the sample itself
    // being the first. Infinity when that distance exceeds maxEps.
    const coreDistances = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      squaredDistanceRow(data, i, n, d, row);
      scratch.set(row);
      const core = Math.sqrt(kthSmallest(scratch, this.minSamples - 1));
      coreDistances[i] = core > this.maxEps ? Infinity : core;
    }

    const reachability = new Float64Array(n).fill(Infinity);
    const predecessor = new Int32Array(n).fill(-1);
    const ordering = new Int32Array(n);
    const processed = new Uint8Array(n);

    for (let orderIdx = 0; orderIdx < n; orderIdx++) {
      // Next point: the unprocessed one with the smallest reachability, lowest index on ties.
      // Points never reached keep Infinity, so a new region starts at the lowest unvisited index.
      let point = -1;
      for (let i = 0; i < n; i++) {
        if (processed[i]) continue;
        if (point < 0 || (reachability[i] as number) < (reachability[point] as number)) point = i;
      }
      processed[point] = 1;
      ordering[orderIdx] = point;

      const core = coreDistances[point] as number;
      if (core === Infinity) continue;
      squaredDistanceRow(data, point, n, d, row);
      for (let j = 0; j < n; j++) {
        if (processed[j]) continue;
        const dist = Math.sqrt(row[j] as number);
        if (dist > this.maxEps) continue;
        const reach = Math.max(core, dist);
        if (reach < (reachability[j] as number)) {
          reachability[j] = reach;
          predecessor[j] = point;
        }
      }
    }

    let labels: Int32Array;
    let hierarchy: Array<readonly [number, number]> = [];
    if (this.clusterMethod === "dbscan") {
      labels = this.extractDBSCANClusters(n, ordering, reachability, coreDistances);
    } else {
      const result = this.extractXiClusters(n, ordering, reachability, predecessor);
      labels = result.labels;
      hierarchy = result.hierarchy;
    }

    this.nFeaturesIn_ = d;
    // Copy: the data view may share memory with the caller's tensor.
    this.fitData_ = data.slice();
    this.coreDistances_ = coreDistances;
    this.reachability_ = reachability;
    this.predecessor_ = predecessor;
    this.ordering_ = ordering;
    this.clusterHierarchy_ = hierarchy;
    this.labels_ = tensor(labels, { dtype: "int32" });
    this.fitted = true;
    return this;
  }

  private extractDBSCANClusters(
    n: number,
    ordering: Int32Array,
    reachability: Float64Array,
    coreDistances: Float64Array
  ): Int32Array {
    const cutEps =
      this.eps ?? (this.maxEps === Infinity ? OPTICS.estimateEps(reachability) : this.maxEps);
    const labels = new Int32Array(n).fill(-1);
    let currentCluster = -1;

    for (let i = 0; i < n; i++) {
      const idx = ordering[i] as number;
      const farReach = (reachability[idx] as number) > cutEps;
      const nearCore = (coreDistances[idx] as number) <= cutEps;
      if (farReach) {
        // Out of reach of the previous points: a core point starts a new cluster, anything else is noise.
        if (nearCore) {
          currentCluster++;
          labels[idx] = currentCluster;
        }
      } else {
        labels[idx] = currentCluster;
      }
    }
    return labels;
  }

  private extractXiClusters(
    n: number,
    ordering: Int32Array,
    reachability: Float64Array,
    predecessor: Int32Array
  ): { readonly labels: Int32Array; readonly hierarchy: Array<readonly [number, number]> } {
    const minSamples = Math.max(2, this.minSamples);
    const minClusterSize = this.minClusterSize ?? minSamples;

    const reachInOrder = new Float64Array(n);
    const predInOrder = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const idx = ordering[i] as number;
      reachInOrder[i] = reachability[idx] as number;
      predInOrder[i] = predecessor[idx] as number;
    }
    const clusters = extractXiHierarchy(
      reachInOrder,
      predInOrder,
      ordering,
      this.xi,
      minSamples,
      minClusterSize,
      this.predecessorCorrection
    );

    // Smaller clusters come first, so a nested cluster wins over the one that contains it.
    const byPosition = new Int32Array(n).fill(-1);
    let label = 0;
    for (const [start, end] of clusters) {
      let free = true;
      for (let p = start; p <= end; p++) {
        if (byPosition[p] !== -1) {
          free = false;
          break;
        }
      }
      if (!free) continue;
      byPosition.fill(label, start, end + 1);
      label++;
    }
    const labels = new Int32Array(n);
    for (let p = 0; p < n; p++) labels[ordering[p] as number] = byPosition[p] as number;
    return { labels, hierarchy: clusters };
  }

  /** 75th percentile of the finite reachability distances (1 when there are none). */
  private static estimateEps(reachability: Float64Array): number {
    const finite: number[] = [];
    for (let i = 0; i < reachability.length; i++) {
      const r = reachability[i] as number;
      if (r < Infinity) finite.push(r);
    }
    if (finite.length === 0) return 1;
    finite.sort((a, b) => a - b);
    return finite[Math.floor(finite.length * 0.75)] ?? 1;
  }

  /**
   * Predict cluster labels for new samples using nearest-neighbor assignment.
   *
   * Each new point is assigned the label of its nearest training sample, which is -1
   * when that sample is noise. OPTICS itself has no inductive prediction rule; this
   * is a Deepbox convenience.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.labels_ || !this.fitData_) {
      throw new NotFittedError("OPTICS must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "OPTICS");

    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const data = toFloat64View(X);
    const train = this.fitData_;
    const nTrain = train.length / d;
    const trainLabels = toFloat64View(this.labels_);
    const result = new Int32Array(n);

    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let bestLabel = -1;
      for (let t = 0; t < nTrain; t++) {
        let dist = 0;
        for (let f = 0; f < d; f++) {
          const diff = (data[i * d + f] as number) - (train[t * d + f] as number);
          dist += diff * diff;
          if (dist >= bestDist) break;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestLabel = trainLabels[t] as number;
        }
      }
      result[i] = bestLabel;
    }
    return tensor(result, { dtype: "int32" });
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

  /** Training labels of shape (n_samples,); -1 marks noise. */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("OPTICS must be fitted to access labels");
    }
    return this.labels_;
  }

  /** Reachability distance of every sample (indexed by sample, `Infinity` for unreachable ones). */
  get reachability(): Float64Array {
    if (!this.fitted || !this.reachability_) {
      throw new NotFittedError("OPTICS must be fitted to access reachability");
    }
    return this.reachability_;
  }

  /** Sample indices in the order OPTICS visited them. */
  get ordering(): Int32Array {
    if (!this.fitted || !this.ordering_) {
      throw new NotFittedError("OPTICS must be fitted to access ordering");
    }
    return this.ordering_;
  }

  /** Core distance of every sample; `Infinity` for samples that are not core points within `maxEps`. */
  get coreDistances(): Float64Array {
    if (!this.fitted || !this.coreDistances_) {
      throw new NotFittedError("OPTICS must be fitted to access core distances");
    }
    return this.coreDistances_;
  }

  /** For every sample, the sample it was reached from (-1 for samples that start a new region). */
  get predecessor(): Int32Array {
    if (!this.fitted || !this.predecessor_) {
      throw new NotFittedError("OPTICS must be fitted to access predecessors");
    }
    return this.predecessor_;
  }

  /**
   * Clusters found by the 'xi' method as inclusive `[start, end]` positions in `ordering`,
   * with nested clusters listed before the clusters that contain them. Always empty for
   * the 'dbscan' method.
   */
  get clusterHierarchy(): ReadonlyArray<readonly [number, number]> {
    if (!this.fitted) {
      throw new NotFittedError("OPTICS must be fitted to access the cluster hierarchy");
    }
    return this.clusterHierarchy_;
  }

  /** Mean of every cluster found, shape (n_clusters, n_features); noise is excluded. */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.fitData_ || !this.labels_) {
      throw new NotFittedError("OPTICS must be fitted to access cluster centers");
    }
    const d = this.nFeaturesIn_;
    const labels = toFloat64View(this.labels_);
    let nClusters = 0;
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i] as number;
      if (label >= nClusters) nClusters = label + 1;
    }
    const sums = new Float64Array(nClusters * d);
    const counts = new Float64Array(nClusters);
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i] as number;
      if (label < 0) continue;
      counts[label] = (counts[label] as number) + 1;
      for (let f = 0; f < d; f++) {
        sums[label * d + f] =
          (sums[label * d + f] as number) + (this.fitData_[i * d + f] as number);
      }
    }
    for (let c = 0; c < nClusters; c++) {
      for (let f = 0; f < d; f++) {
        sums[c * d + f] = (sums[c * d + f] as number) / (counts[c] as number);
      }
    }
    return tensor(sums).reshape([nClusters, d]);
  }

  getParams(): Record<string, unknown> {
    return {
      minSamples: this.minSamples,
      maxEps: this.maxEps,
      clusterMethod: this.clusterMethod,
      xi: this.xi,
      eps: this.eps,
      minClusterSize: this.minClusterSize,
      predecessorCorrection: this.predecessorCorrection,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "minSamples":
          this.minSamples = checkMinSamples(value);
          break;
        case "maxEps":
          this.maxEps = checkMaxEps(value);
          break;
        case "clusterMethod":
          this.clusterMethod = checkClusterMethod(value);
          break;
        case "xi":
          this.xi = checkXi(value);
          break;
        case "eps":
          this.eps = checkEps(value);
          break;
        case "minClusterSize":
          this.minClusterSize = checkMinClusterSize(value);
          break;
        case "predecessorCorrection":
          this.predecessorCorrection = checkBoolean("predecessorCorrection", value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
