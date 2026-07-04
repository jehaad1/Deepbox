/**
 * Missing value imputation for ML pipelines.
 *
 * @module preprocess/impute
 * @see {@link https://deepbox.dev/docs/preprocess-features | Deepbox Imputation}
 */

import { InvalidParameterError, NotFittedError } from "../core";
import type { Transformer } from "../ml/base";
import { type Tensor, Tensor as TensorClass, tensor } from "../ndarray";
import { getStrides2D } from "./_internal";

/**
 * Simple imputation transformer for completing missing values.
 *
 * Replace NaN values using a descriptive statistic (mean, median, most_frequent)
 * or a constant value computed from each feature column.
 *
 * @example
 * ```ts
 * import { SimpleImputer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const imp = new SimpleImputer({ strategy: 'mean' });
 * const X = tensor([[1, 2], [NaN, 3], [7, NaN]]);
 * imp.fit(X);
 * const X_filled = imp.transform(X);
 * ```
 */
export class SimpleImputer implements Transformer {
  private readonly strategy: "mean" | "median" | "most_frequent" | "constant";
  private readonly fillValue: number;
  private statistics_?: number[];
  private nFeatures_?: number;
  private fitted = false;

  /**
   * @param options - Configuration options
   * @param options.strategy - Imputation strategy (default: 'mean')
   * @param options.fillValue - Value to use when strategy='constant' (default: 0)
   */
  constructor(
    options: {
      readonly strategy?: "mean" | "median" | "most_frequent" | "constant";
      readonly fillValue?: number;
    } = {}
  ) {
    this.strategy = options.strategy ?? "mean";
    this.fillValue = options.fillValue ?? 0;

    const validStrategies = ["mean", "median", "most_frequent", "constant"];
    if (!validStrategies.includes(this.strategy)) {
      throw new InvalidParameterError(
        `strategy must be one of ${validStrategies.join(", ")}; got '${this.strategy}'`,
        "strategy",
        this.strategy
      );
    }
  }

  /**
   * Compute the imputation statistics from training data.
   *
   * @param X - Training data of shape (n_samples, n_features), may contain NaN
   * @returns this
   */
  fit(X: Tensor): this {
    if (X.ndim !== 2) {
      throw new InvalidParameterError(`X must be 2D; got ndim=${X.ndim}`, "X", X.shape);
    }

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const [__s0, __s1] = getStrides2D(X);
    this.nFeatures_ = nFeatures;
    this.statistics_ = [];

    for (let j = 0; j < nFeatures; j++) {
      const colValues: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const val = Number(X.data[X.offset + i * __s0 + j * __s1]);
        if (!Number.isNaN(val)) {
          colValues.push(val);
        }
      }

      let stat: number;
      if (this.strategy === "constant") {
        stat = this.fillValue;
      } else if (colValues.length === 0) {
        stat = 0;
      } else if (this.strategy === "mean") {
        stat = colValues.reduce((a, b) => a + b, 0) / colValues.length;
      } else if (this.strategy === "median") {
        colValues.sort((a, b) => a - b);
        const mid = Math.floor(colValues.length / 2);
        stat =
          colValues.length % 2 === 0
            ? ((colValues[mid - 1] ?? 0) + (colValues[mid] ?? 0)) / 2
            : (colValues[mid] ?? 0);
      } else {
        // most_frequent
        const counts = new Map<number, number>();
        for (const v of colValues) {
          counts.set(v, (counts.get(v) ?? 0) + 1);
        }
        let bestVal = colValues[0] ?? 0;
        let bestCount = 0;
        for (const [v, c] of counts) {
          // On ties, prefer the smallest value to match scikit-learn's behavior.
          if (c > bestCount || (c === bestCount && v < bestVal)) {
            bestCount = c;
            bestVal = v;
          }
        }
        stat = bestVal;
      }

      this.statistics_.push(stat);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Impute missing values in X using the fitted statistics.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Imputed data tensor of same shape
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.statistics_ || this.nFeatures_ === undefined) {
      throw new NotFittedError("SimpleImputer must be fitted before transform");
    }
    if (X.ndim !== 2) {
      throw new InvalidParameterError(`X must be 2D; got ndim=${X.ndim}`, "X", X.shape);
    }
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const [__s0, __s1] = getStrides2D(X);
    if (nFeatures !== this.nFeatures_) {
      throw new InvalidParameterError(
        `X has ${nFeatures} features, expected ${this.nFeatures_}`,
        "X",
        X.shape
      );
    }

    const result: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const val = Number(X.data[X.offset + i * __s0 + j * __s1]);
        result.push(Number.isNaN(val) ? (this.statistics_[j] ?? 0) : val);
      }
    }

    return tensor(result).reshape([nSamples, nFeatures]);
  }

  /**
   * Fit and transform in one step.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Imputed data tensor
   */
  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  getParams(): Record<string, unknown> {
    return {
      strategy: this.strategy,
      fillValue: this.fillValue,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    throw new InvalidParameterError(
      "SimpleImputer does not support setParams after construction",
      "params",
      _params
    );
  }

  /**
   * Get the computed imputation statistics per feature.
   */
  get statistics(): number[] {
    if (!this.fitted || !this.statistics_) {
      throw new NotFittedError("SimpleImputer must be fitted first");
    }
    return [...this.statistics_];
  }
}

/**
 * KNN-based imputation for completing missing values.
 *
 * Each missing value is imputed using the weighted mean of the
 * k nearest neighbors found in the training set. Distance is
 * computed only over features that are present (non-NaN) in both
 * the query row and the candidate row.
 *
 * @example
 * ```ts
 * import { KNNImputer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const imp = new KNNImputer({ nNeighbors: 3 });
 * const X = tensor([[1, 2], [NaN, 3], [7, NaN], [4, 5]]);
 * const filled = imp.fitTransform(X);
 * ```
 */
export class KNNImputer implements Transformer {
  private readonly nNeighbors: number;
  private readonly weights: "uniform" | "distance";
  private trainFlat_?: Float64Array;
  private nTrain_ = 0;
  private nFeatures_?: number;
  private fitted = false;

  /**
   * @param options - Configuration options
   * @param options.nNeighbors - Number of neighbors to use (default: 5)
   * @param options.weights - Weight function: 'uniform' or 'distance' (default: 'uniform')
   */
  constructor(
    options: {
      readonly nNeighbors?: number;
      readonly weights?: "uniform" | "distance";
    } = {}
  ) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.weights = options.weights ?? "uniform";

    if (
      !Number.isFinite(this.nNeighbors) ||
      !Number.isInteger(this.nNeighbors) ||
      this.nNeighbors < 1
    ) {
      throw new InvalidParameterError(
        "nNeighbors must be a positive integer",
        "nNeighbors",
        this.nNeighbors
      );
    }
    if (this.weights !== "uniform" && this.weights !== "distance") {
      throw new InvalidParameterError(
        "weights must be 'uniform' or 'distance'",
        "weights",
        this.weights
      );
    }
  }

  fit(X: Tensor): this {
    if (X.ndim !== 2) {
      throw new InvalidParameterError(`X must be 2D; got ndim=${X.ndim}`, "X", X.shape);
    }
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const [__s0, __s1] = getStrides2D(X);
    this.nFeatures_ = nFeatures;
    this.nTrain_ = nSamples;

    // Store training data as a single row-major Float64Array — cache-friendly
    // and avoids the array-of-arrays access in the O(nTrain·nFeatures) distance
    // sweep of transform().
    const flat = new Float64Array(nSamples * nFeatures);
    const src = X.data;
    const offset = X.offset;
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = offset + i * __s0;
      for (let j = 0; j < nFeatures; j++) flat[pos++] = Number(src[rowBase + j * __s1]);
    }
    this.trainFlat_ = flat;

    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    const trainFlat = this.trainFlat_;
    if (!this.fitted || !trainFlat || this.nFeatures_ === undefined) {
      throw new NotFittedError("KNNImputer must be fitted before transform");
    }
    if (X.ndim !== 2) {
      throw new InvalidParameterError(`X must be 2D; got ndim=${X.ndim}`, "X", X.shape);
    }
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const [__s0, __s1] = getStrides2D(X);
    if (nFeatures !== this.nFeatures_) {
      throw new InvalidParameterError(
        `X has ${nFeatures} features, expected ${this.nFeatures_}`,
        "X",
        X.shape
      );
    }

    const nTrain = this.nTrain_;
    const out = new Float64Array(nSamples * nFeatures);
    const useDistance = this.weights === "distance";

    // Reusable scratch buffers for the neighbour search (per query).
    const queryRow = new Float64Array(nFeatures);
    const dist = new Float64Array(nTrain);
    const nbrIdx = new Int32Array(Math.min(this.nNeighbors, nTrain));
    const nbrDist = new Float64Array(nbrIdx.length);

    const src = X.data;
    const offset = X.offset;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = offset + i * __s0;
      const outBase = i * nFeatures;
      let hasMissing = false;
      for (let j = 0; j < nFeatures; j++) {
        const val = Number(src[rowBase + j * __s1]);
        queryRow[j] = val;
        out[outBase + j] = val;
        if (Number.isNaN(val)) hasMissing = true;
      }
      if (!hasMissing) continue;

      const k = this.findNeighbors(queryRow, nFeatures, trainFlat, nTrain, dist, nbrIdx, nbrDist);

      for (let j = 0; j < nFeatures; j++) {
        if (!Number.isNaN(queryRow[j] as number)) continue; // present — already copied
        let weightSum = 0;
        let valueSum = 0;
        for (let n = 0; n < k; n++) {
          const nVal = trainFlat[(nbrIdx[n] as number) * nFeatures + j] as number;
          if (Number.isNaN(nVal)) continue;
          const w = useDistance ? 1 / ((nbrDist[n] as number) + 1e-10) : 1;
          weightSum += w;
          valueSum += w * nVal;
        }
        out[outBase + j] = weightSum > 0 ? valueSum / weightSum : 0;
      }
    }

    return TensorClass.fromTypedArray({
      data: out,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: X.device,
    });
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  getParams(): Record<string, unknown> {
    return {
      nNeighbors: this.nNeighbors,
      weights: this.weights,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    throw new InvalidParameterError(
      "KNNImputer does not support setParams after construction",
      "params",
      _params
    );
  }

  /**
   * Find k nearest neighbors for a query row, computing distance
   * only over mutually present (non-NaN) features.
   */
  /**
   * Nan-aware Euclidean k-NN over the flat training buffer. Writes the k
   * nearest (index, distance) into nbrIdx/nbrDist and returns the count
   * found (≤ k). Uses a bounded insertion into the small result arrays
   * instead of allocating an object per train row and full-sorting them.
   */
  private findNeighbors(
    queryRow: Float64Array,
    nFeatures: number,
    trainFlat: Float64Array,
    nTrain: number,
    dist: Float64Array,
    nbrIdx: Int32Array,
    nbrDist: Float64Array
  ): number {
    const k = nbrIdx.length;
    if (k === 0) return 0;

    // Pass 1: nan-Euclidean distance to every training row.
    let valid = 0;
    for (let t = 0; t < nTrain; t++) {
      const tBase = t * nFeatures;
      let sumSq = 0;
      let shared = 0;
      for (let j = 0; j < nFeatures; j++) {
        const qVal = queryRow[j] as number;
        const tVal = trainFlat[tBase + j] as number;
        if (Number.isNaN(qVal) || Number.isNaN(tVal)) continue;
        const diff = qVal - tVal;
        sumSq += diff * diff;
        shared++;
      }
      dist[t] = shared === 0 ? Number.POSITIVE_INFINITY : Math.sqrt((sumSq / shared) * nFeatures);
      if (shared !== 0) valid++;
    }

    // Pass 2: bounded selection of the k smallest (k is small, so an
    // insertion into a size-k sorted list beats sorting all nTrain).
    const kEff = Math.min(k, valid);
    let filled = 0;
    for (let t = 0; t < nTrain; t++) {
      const d = dist[t] as number;
      if (!Number.isFinite(d)) continue;
      if (filled < kEff) {
        // Insert into the sorted prefix [0, filled).
        let p = filled - 1;
        while (p >= 0 && (nbrDist[p] as number) > d) {
          nbrDist[p + 1] = nbrDist[p] as number;
          nbrIdx[p + 1] = nbrIdx[p] as number;
          p--;
        }
        nbrDist[p + 1] = d;
        nbrIdx[p + 1] = t;
        filled++;
      } else if (d < (nbrDist[kEff - 1] as number)) {
        let p = kEff - 2;
        while (p >= 0 && (nbrDist[p] as number) > d) {
          nbrDist[p + 1] = nbrDist[p] as number;
          nbrIdx[p + 1] = nbrIdx[p] as number;
          p--;
        }
        nbrDist[p + 1] = d;
        nbrIdx[p + 1] = t;
      }
    }
    return kEff;
  }
}

/**
 * Binary indicator for missing values.
 *
 * Generates a boolean matrix where 1 indicates a missing value (NaN)
 * and 0 indicates a present value. Useful as an additional feature
 * set to complement imputation.
 *
 * @example
 * ```ts
 * import { MissingIndicator } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const mi = new MissingIndicator();
 * const X = tensor([[1, NaN], [NaN, 3], [7, 6]]);
 * mi.fit(X);
 * const indicators = mi.transform(X);
 * // indicators shape: [3, 2], values: [[0, 1], [1, 0], [0, 0]]
 * ```
 */
export class MissingIndicator implements Transformer {
  private readonly features: "missing-only" | "all";
  private missingFeatures_?: number[];
  private nFeatures_?: number;
  private fitted = false;

  /**
   * @param options - Configuration options
   * @param options.features - Which features to include:
   *   - 'missing-only': only features that had missing values during fit (default)
   *   - 'all': all features
   */
  constructor(
    options: {
      readonly features?: "missing-only" | "all";
    } = {}
  ) {
    this.features = options.features ?? "missing-only";
    if (this.features !== "missing-only" && this.features !== "all") {
      throw new InvalidParameterError(
        "features must be 'missing-only' or 'all'",
        "features",
        this.features
      );
    }
  }

  fit(X: Tensor): this {
    if (X.ndim !== 2) {
      throw new InvalidParameterError(`X must be 2D; got ndim=${X.ndim}`, "X", X.shape);
    }
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const [__s0, __s1] = getStrides2D(X);
    this.nFeatures_ = nFeatures;

    if (this.features === "all") {
      this.missingFeatures_ = Array.from({ length: nFeatures }, (_, i) => i);
    } else {
      // Identify columns that have at least one NaN
      this.missingFeatures_ = [];
      for (let j = 0; j < nFeatures; j++) {
        let hasMissing = false;
        for (let i = 0; i < nSamples; i++) {
          if (Number.isNaN(Number(X.data[X.offset + i * __s0 + j * __s1]))) {
            hasMissing = true;
            break;
          }
        }
        if (hasMissing) {
          this.missingFeatures_.push(j);
        }
      }
    }

    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.missingFeatures_ || this.nFeatures_ === undefined) {
      throw new NotFittedError("MissingIndicator must be fitted before transform");
    }
    if (X.ndim !== 2) {
      throw new InvalidParameterError(`X must be 2D; got ndim=${X.ndim}`, "X", X.shape);
    }
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const [__s0, __s1] = getStrides2D(X);
    if (nFeatures !== this.nFeatures_) {
      throw new InvalidParameterError(
        `X has ${nFeatures} features, expected ${this.nFeatures_}`,
        "X",
        X.shape
      );
    }

    const outCols = this.missingFeatures_.length;
    const result = new Float64Array(nSamples * outCols);

    for (let i = 0; i < nSamples; i++) {
      for (let k = 0; k < outCols; k++) {
        const j = this.missingFeatures_[k] ?? 0;
        const val = Number(X.data[X.offset + i * __s0 + j * __s1]);
        result[i * outCols + k] = Number.isNaN(val) ? 1 : 0;
      }
    }

    return tensor(result, { dtype: "float64" }).reshape([nSamples, outCols]);
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  getParams(): Record<string, unknown> {
    return { features: this.features };
  }

  setParams(_params: Record<string, unknown>): this {
    throw new InvalidParameterError(
      "MissingIndicator does not support setParams after construction",
      "params",
      _params
    );
  }

  /**
   * Get the indices of features that were detected as having missing values.
   */
  get features_(): number[] {
    if (!this.fitted || !this.missingFeatures_) {
      throw new NotFittedError("MissingIndicator must be fitted first");
    }
    return [...this.missingFeatures_];
  }
}
