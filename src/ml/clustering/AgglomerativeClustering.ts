/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

type Linkage = "single" | "complete" | "average" | "ward";

/**
 * Agglomerative (hierarchical) clustering.
 *
 * Bottom-up clustering that starts with each sample as its own cluster
 * and iteratively merges the closest pair of clusters until `nClusters`
 * clusters remain.
 *
 * Supported linkage criteria:
 * - **single**: minimum distance between clusters
 * - **complete**: maximum distance between clusters
 * - **average**: average distance between clusters
 * - **ward**: minimizes the total within-cluster variance (Euclidean only)
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
 */
export class AgglomerativeClustering implements Clusterer {
  private nClusters: number;
  private linkage: Linkage;

  private labels_?: Tensor;
  private fitData_?: number[][];
  private fitted = false;

  constructor(
    options: {
      readonly nClusters?: number;
      readonly linkage?: Linkage;
    } = {}
  ) {
    this.nClusters = options.nClusters ?? 2;
    this.linkage = options.linkage ?? "ward";

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new InvalidParameterError(
        "nClusters must be an integer >= 1",
        "nClusters",
        this.nClusters
      );
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    if (nSamples < this.nClusters) {
      throw new InvalidParameterError(
        `n_samples=${nSamples} should be >= n_clusters=${this.nClusters}`,
        "nClusters",
        this.nClusters
      );
    }

    // Extract data
    const data: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      data.push(row);
    }

    // Each sample starts as its own cluster
    // clusterMembers[c] = array of original sample indices in cluster c
    let clusters: number[][] = data.map((_, i) => [i]);

    // Precompute pairwise distances
    const distMatrix = new Float64Array(nSamples * nSamples);
    for (let i = 0; i < nSamples; i++) {
      for (let j = i + 1; j < nSamples; j++) {
        let d = 0;
        for (let k = 0; k < nFeatures; k++) {
          const diff = (data[i]![k] ?? 0) - (data[j]![k] ?? 0);
          d += diff * diff;
        }
        d = Math.sqrt(d);
        distMatrix[i * nSamples + j] = d;
        distMatrix[j * nSamples + i] = d;
      }
    }

    // Merge until we reach the desired number of clusters
    while (clusters.length > this.nClusters) {
      let bestI = 0;
      let bestJ = 1;
      let bestDist = Infinity;

      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          const d = this.clusterDistance(
            clusters[i]!,
            clusters[j]!,
            distMatrix,
            nSamples,
            data,
            nFeatures
          );
          if (d < bestDist) {
            bestDist = d;
            bestI = i;
            bestJ = j;
          }
        }
      }

      // Merge bestJ into bestI
      const merged = clusters[bestI]!.concat(clusters[bestJ]!);
      const newClusters: number[][] = [];
      for (let i = 0; i < clusters.length; i++) {
        if (i === bestI) {
          newClusters.push(merged);
        } else if (i !== bestJ) {
          newClusters.push(clusters[i]!);
        }
      }
      clusters = newClusters;
    }

    // Assign labels
    const labelArr = new Array<number>(nSamples);
    for (let c = 0; c < clusters.length; c++) {
      for (const idx of clusters[c]!) {
        labelArr[idx] = c;
      }
    }

    this.labels_ = tensor(labelArr, { dtype: "int32" });
    this.fitData_ = data;
    this.fitted = true;
    return this;
  }

  private clusterDistance(
    a: number[],
    b: number[],
    distMatrix: Float64Array,
    n: number,
    data: number[][],
    nFeatures: number
  ): number {
    if (this.linkage === "single") {
      let minD = Infinity;
      for (const i of a) {
        for (const j of b) {
          const d = distMatrix[i * n + j] ?? 0;
          if (d < minD) minD = d;
        }
      }
      return minD;
    }

    if (this.linkage === "complete") {
      let maxD = -Infinity;
      for (const i of a) {
        for (const j of b) {
          const d = distMatrix[i * n + j] ?? 0;
          if (d > maxD) maxD = d;
        }
      }
      return maxD;
    }

    if (this.linkage === "average") {
      let total = 0;
      let count = 0;
      for (const i of a) {
        for (const j of b) {
          total += distMatrix[i * n + j] ?? 0;
          count++;
        }
      }
      return count > 0 ? total / count : 0;
    }

    // Ward's linkage: increase in total within-cluster variance
    const centroidA = this.centroid(a, data, nFeatures);
    const centroidB = this.centroid(b, data, nFeatures);
    const merged = a.concat(b);
    const centroidM = this.centroid(merged, data, nFeatures);

    let varA = 0;
    for (const i of a) {
      for (let k = 0; k < nFeatures; k++) {
        const d = (data[i]![k] ?? 0) - (centroidA[k] ?? 0);
        varA += d * d;
      }
    }
    let varB = 0;
    for (const i of b) {
      for (let k = 0; k < nFeatures; k++) {
        const d = (data[i]![k] ?? 0) - (centroidB[k] ?? 0);
        varB += d * d;
      }
    }
    let varM = 0;
    for (const i of merged) {
      for (let k = 0; k < nFeatures; k++) {
        const d = (data[i]![k] ?? 0) - (centroidM[k] ?? 0);
        varM += d * d;
      }
    }

    return varM - varA - varB;
  }

  private centroid(indices: number[], data: number[][], nFeatures: number): number[] {
    const c = new Array<number>(nFeatures).fill(0);
    for (const i of indices) {
      for (let k = 0; k < nFeatures; k++) {
        c[k] = (c[k] ?? 0) + (data[i]![k] ?? 0);
      }
    }
    const n = indices.length;
    for (let k = 0; k < nFeatures; k++) {
      c[k] = (c[k] ?? 0) / n;
    }
    return c;
  }

  /**
   * Predict cluster labels for new samples using nearest-neighbor assignment.
   *
   * Each new point is assigned the label of its nearest training sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.labels_ || !this.fitData_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted before prediction");
    }

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const trainData = this.fitData_;
    const trainLabels = this.labels_;
    const nTrain = trainData.length;
    const result: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      let bestDist = Infinity;
      let bestLabel = 0;
      for (let t = 0; t < nTrain; t++) {
        const trainRow = trainData[t];
        if (!trainRow) continue;
        let dist = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff = (row[f] ?? 0) - (trainRow[f] ?? 0);
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestLabel = Number(trainLabels.data[trainLabels.offset + t]);
        }
      }
      result.push(bestLabel);
    }

    return tensor(result, { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_!;
  }

  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted to access labels");
    }
    return this.labels_;
  }

  get clusterCenters(): Tensor {
    if (!this.fitted || !this.fitData_ || !this.labels_) {
      throw new NotFittedError("AgglomerativeClustering must be fitted to access cluster centers");
    }
    // Compute centroids from fitted data
    const nFeatures = this.fitData_[0]?.length ?? 0;
    const clusterMap = new Map<number, number[][]>();
    for (let i = 0; i < this.fitData_.length; i++) {
      const label = Number(this.labels_.data[this.labels_.offset + i]);
      const existing = clusterMap.get(label);
      const row = this.fitData_[i];
      if (row) {
        if (existing) {
          existing.push(row);
        } else {
          clusterMap.set(label, [row]);
        }
      }
    }
    const sortedLabels = [...clusterMap.keys()].sort((a, b) => a - b);
    const centers: number[][] = [];
    for (const label of sortedLabels) {
      const members = clusterMap.get(label);
      if (!members || members.length === 0) continue;
      const centroid = new Array<number>(nFeatures).fill(0);
      for (const m of members) {
        for (let f = 0; f < nFeatures; f++) {
          centroid[f] = (centroid[f] ?? 0) + (m[f] ?? 0);
        }
      }
      for (let f = 0; f < nFeatures; f++) {
        centroid[f] = (centroid[f] ?? 0) / members.length;
      }
      centers.push(centroid);
    }
    return tensor(centers);
  }

  getParams(): Record<string, unknown> {
    return { nClusters: this.nClusters, linkage: this.linkage };
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
        case "linkage":
          if (
            value !== "single" &&
            value !== "complete" &&
            value !== "average" &&
            value !== "ward"
          ) {
            throw new InvalidParameterError(
              `linkage must be "single", "complete", "average", or "ward"`,
              "linkage",
              value
            );
          }
          this.linkage = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
