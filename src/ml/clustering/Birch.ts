/**
 * BIRCH — Balanced Iterative Reducing and Clustering using Hierarchies.
 *
 * Incrementally builds a CF (Clustering Feature) tree to summarize
 * data, then applies a global clustering step. Memory-efficient for
 * large datasets.
 *
 * @module ml/clustering/Birch
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validatePredictInputs, validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

/** Clustering Feature triple (n, LS, SS). */
interface CF {
  n: number;
  ls: Float64Array; // linear sum
  ss: Float64Array; // squared sum
}

/** CF tree leaf node. */
interface CFLeaf {
  entries: CF[];
}

/**
 * BIRCH clustering algorithm.
 *
 * Phase 1: Build a CF tree by scanning the data.
 * Phase 2: Apply agglomerative clustering on the CF leaf entries
 * to produce final `nClusters` clusters.
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
 */
export class Birch implements Clusterer {
  private nClusters: number;
  private threshold: number;
  private branchingFactor: number;

  private labels_?: Tensor;
  private clusterCenters_?: Float64Array;
  private nFeaturesIn_ = 0;
  // Actual cluster count after clamping to the number of CF entries; the
  // getters/predict must use this, not the requested nClusters, when there
  // are fewer CF entries than requested clusters (otherwise reshape/loop
  // reads past the centers buffer).
  private nClustersActual_ = 0;
  private leaves_: CFLeaf[] = [];
  private fitted = false;

  constructor(
    options: {
      readonly nClusters?: number;
      readonly threshold?: number;
      readonly branchingFactor?: number;
    } = {}
  ) {
    this.nClusters = options.nClusters ?? 3;
    this.threshold = options.threshold ?? 0.5;
    this.branchingFactor = options.branchingFactor ?? 50;

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new InvalidParameterError("nClusters must be >= 1", "nClusters", this.nClusters);
    }
    if (this.threshold <= 0) {
      throw new InvalidParameterError("threshold must be > 0", "threshold", this.threshold);
    }
    if (!Number.isInteger(this.branchingFactor) || this.branchingFactor < 2) {
      throw new InvalidParameterError(
        "branchingFactor must be >= 2",
        "branchingFactor",
        this.branchingFactor
      );
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Phase 1: Build CF tree (simplified — single level of leaves)
    this.leaves_ = [];
    const allCFs: CF[] = [];

    for (let i = 0; i < nSamples; i++) {
      const point = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        point[j] = Number(X.data[X.offset + i * nFeatures + j]);
      }

      // Find closest CF entry within threshold
      let bestLeafIdx = -1;
      let bestEntryIdx = -1;
      let bestDist = Infinity;

      for (let li = 0; li < this.leaves_.length; li++) {
        const leaf = this.leaves_[li]!;
        for (let ei = 0; ei < leaf.entries.length; ei++) {
          const cf = leaf.entries[ei]!;
          const centroid = this.cfCentroid(cf, nFeatures);
          const d = this.euclideanDist(point, centroid, nFeatures);
          if (d < bestDist) {
            bestDist = d;
            bestLeafIdx = li;
            bestEntryIdx = ei;
          }
        }
      }

      if (bestDist <= this.threshold && bestLeafIdx >= 0 && bestEntryIdx >= 0) {
        // Absorb into existing CF
        const cf = this.leaves_[bestLeafIdx]!.entries[bestEntryIdx]!;
        this.absorbPoint(cf, point, nFeatures);
      } else {
        // Create new CF entry
        const newCF: CF = {
          n: 1,
          ls: new Float64Array(point),
          ss: new Float64Array(nFeatures),
        };
        for (let j = 0; j < nFeatures; j++) {
          newCF.ss[j] = (point[j] ?? 0) * (point[j] ?? 0);
        }

        // Find leaf with room, or create new leaf
        let added = false;
        for (const leaf of this.leaves_) {
          if (leaf.entries.length < this.branchingFactor) {
            leaf.entries.push(newCF);
            added = true;
            break;
          }
        }
        if (!added) {
          this.leaves_.push({ entries: [newCF] });
        }
      }
    }

    // Collect all CF entries
    for (const leaf of this.leaves_) {
      for (const cf of leaf.entries) {
        allCFs.push(cf);
      }
    }

    // Phase 2: Agglomerative clustering on CF centroids
    const nEntries = allCFs.length;
    const k = Math.min(this.nClusters, nEntries);

    // Compute centroids
    const centroids: Float64Array[] = [];
    for (const cf of allCFs) {
      centroids.push(this.cfCentroid(cf, nFeatures));
    }

    // Agglomerative: assign each CF entry to a cluster
    const clusterAssignment = new Int32Array(nEntries);
    for (let i = 0; i < nEntries; i++) clusterAssignment[i] = i;

    let numClusters = nEntries;
    while (numClusters > k) {
      // Find two closest clusters to merge
      let bestI = 0;
      let bestJ = 1;
      let bestMergeDist = Infinity;

      const clusterIds = [...new Set(Array.from(clusterAssignment))];
      for (let ci = 0; ci < clusterIds.length; ci++) {
        for (let cj = ci + 1; cj < clusterIds.length; cj++) {
          const idI = clusterIds[ci] ?? 0;
          const idJ = clusterIds[cj] ?? 0;
          const centI = this.clusterMeanCentroid(centroids, clusterAssignment, idI, nFeatures);
          const centJ = this.clusterMeanCentroid(centroids, clusterAssignment, idJ, nFeatures);
          const d = this.euclideanDist(centI, centJ, nFeatures);
          if (d < bestMergeDist) {
            bestMergeDist = d;
            bestI = idI;
            bestJ = idJ;
          }
        }
      }

      // Merge bestJ into bestI
      for (let i = 0; i < nEntries; i++) {
        if (clusterAssignment[i] === bestJ) {
          clusterAssignment[i] = bestI;
        }
      }
      numClusters--;
    }

    // Relabel clusters to 0..k-1
    const uniqueLabels = [...new Set(Array.from(clusterAssignment))];
    const labelMap = new Map<number, number>();
    for (let idx = 0; idx < uniqueLabels.length; idx++) {
      labelMap.set(uniqueLabels[idx] ?? 0, idx);
    }

    // Phase 3: Assign original points to nearest cluster center
    // First compute final cluster centers from CF entries
    const finalCenters = new Float64Array(k * nFeatures);
    const finalCounts = new Float64Array(k);
    for (let i = 0; i < nEntries; i++) {
      const cf = allCFs[i]!;
      const clusterIdx = labelMap.get(clusterAssignment[i] ?? 0) ?? 0;
      const cent = this.cfCentroid(cf, nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const idx = clusterIdx * nFeatures + j;
        finalCenters[idx] = (finalCenters[idx] ?? 0) + (cent[j] ?? 0) * cf.n;
      }
      finalCounts[clusterIdx] = (finalCounts[clusterIdx] ?? 0) + cf.n;
    }
    for (let c = 0; c < k; c++) {
      const cnt = finalCounts[c] ?? 1;
      for (let j = 0; j < nFeatures; j++) {
        finalCenters[c * nFeatures + j] = (finalCenters[c * nFeatures + j] ?? 0) / cnt;
      }
    }

    this.clusterCenters_ = finalCenters;
    this.nClustersActual_ = k;

    // Assign each original point to nearest center
    const labels = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        let d = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff =
            Number(X.data[X.offset + i * nFeatures + j]) - (finalCenters[c * nFeatures + j] ?? 0);
          d += diff * diff;
        }
        d = Math.sqrt(d);
        if (d < bestD) {
          bestD = d;
          bestC = c;
        }
      }
      labels[i] = bestC;
    }

    this.labels_ = tensor(Array.from(labels), { dtype: "int32" });
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("Birch must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "Birch");
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const k = this.nClustersActual_;
    const centers = this.clusterCenters_;

    const labels = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        let d = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff =
            Number(X.data[X.offset + i * nFeatures + j]) - (centers[c * nFeatures + j] ?? 0);
          d += diff * diff;
        }
        if (d < bestD) {
          bestD = d;
          bestC = c;
        }
      }
      labels[i] = bestC;
    }

    return tensor(Array.from(labels), { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_!;
  }

  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("Birch must be fitted to access labels");
    }
    return this.labels_;
  }

  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("Birch must be fitted to access cluster centers");
    }
    const k = this.nClustersActual_;
    const nF = this.nFeaturesIn_;
    return tensor(Array.from(this.clusterCenters_)).reshape([k, nF]);
  }

  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClusters,
      threshold: this.threshold,
      branchingFactor: this.branchingFactor,
    };
  }

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
        case "threshold":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("threshold must be > 0", "threshold", value);
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

  private cfCentroid(cf: CF, nFeatures: number): Float64Array {
    const centroid = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      centroid[j] = (cf.ls[j] ?? 0) / cf.n;
    }
    return centroid;
  }

  private absorbPoint(cf: CF, point: Float64Array, nFeatures: number): void {
    cf.n += 1;
    for (let j = 0; j < nFeatures; j++) {
      cf.ls[j] = (cf.ls[j] ?? 0) + (point[j] ?? 0);
      cf.ss[j] = (cf.ss[j] ?? 0) + (point[j] ?? 0) * (point[j] ?? 0);
    }
  }

  private euclideanDist(a: Float64Array, b: Float64Array, n: number): number {
    let d = 0;
    for (let j = 0; j < n; j++) {
      const diff = (a[j] ?? 0) - (b[j] ?? 0);
      d += diff * diff;
    }
    return Math.sqrt(d);
  }

  private clusterMeanCentroid(
    centroids: Float64Array[],
    assignments: Int32Array,
    clusterId: number,
    nFeatures: number
  ): Float64Array {
    const mean = new Float64Array(nFeatures);
    let count = 0;
    for (let i = 0; i < assignments.length; i++) {
      if (assignments[i] === clusterId) {
        const cent = centroids[i];
        for (let j = 0; j < nFeatures; j++) {
          mean[j] = (mean[j] ?? 0) + (cent ? (cent[j] ?? 0) : 0);
        }
        count++;
      }
    }
    if (count > 0) {
      for (let j = 0; j < nFeatures; j++) mean[j] = (mean[j] ?? 0) / count;
    }
    return mean;
  }
}
