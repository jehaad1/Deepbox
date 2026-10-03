/**
 * Missing value imputation for ML pipelines.
 *
 * @module preprocess/impute
 * @see {@link https://deepbox.dev/docs/preprocess-features | Deepbox Imputation}
 */

import { DataValidationError, DTypeError, InvalidParameterError, NotFittedError } from "../core";
import type { Transformer } from "../ml/base";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { assert2D, assertNumericTensor, getShape2D, getStrides2D } from "./_internal";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function assertKnownParams(
  params: Record<string, unknown>,
  known: readonly string[],
  who: string
): void {
  for (const key of Object.keys(params)) {
    if (!known.includes(key)) {
      throw new InvalidParameterError(
        `Invalid parameter '${key}' for ${who}; valid parameters are ${known.join(", ")}`,
        key,
        params[key]
      );
    }
  }
}

function definedEntries(params: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out;
}

function validateMissingValues(value: number): void {
  if (typeof value !== "number") {
    throw new InvalidParameterError(
      "missingValues must be a number (NaN by default)",
      "missingValues",
      value
    );
  }
}

/**
 * Read a 2D numeric tensor into a dense row-major Float64Array in which every
 * missing entry (`missingValues`, NaN by default) is stored as NaN.
 *
 * When `missingValues` is a number other than NaN, a genuine NaN in X is an
 * error, because it could not be told apart from a valid observation.
 */
function readDense(
  X: Tensor,
  who: string,
  missingValues: number
): { data: Float64Array; nSamples: number; nFeatures: number } {
  if (X.dtype === "string") {
    throw new DTypeError(`${who} requires numeric input`);
  }
  assertNumericTensor(X, "X");
  assert2D(X, "X");
  const [nSamples, nFeatures] = getShape2D(X);
  const [rs, cs] = getStrides2D(X);
  const src = X.data as ArrayLike<number | bigint>;
  const base = X.offset;
  const data = new Float64Array(nSamples * nFeatures);
  const nanIsMissing = Number.isNaN(missingValues);
  let pos = 0;
  for (let i = 0; i < nSamples; i++) {
    const rowBase = base + i * rs;
    for (let j = 0; j < nFeatures; j++) {
      const v = Number(src[rowBase + j * cs]);
      if (nanIsMissing) {
        data[pos++] = v;
      } else if (v === missingValues) {
        data[pos++] = Number.NaN;
      } else if (Number.isNaN(v)) {
        throw new DataValidationError(
          `${who}: X contains NaN, but missingValues is ${missingValues}; ` +
            "NaN is only allowed when missingValues is NaN"
        );
      } else {
        data[pos++] = v;
      }
    }
  }
  return { data, nSamples, nFeatures };
}

function toTensor(data: Float64Array, nSamples: number, nCols: number, X: Tensor): Tensor {
  return TensorClass.fromTypedArray({
    data,
    shape: [nSamples, nCols],
    dtype: "float64",
    device: X.device,
  });
}

/** Keep the listed columns of a dense row-major matrix. */
function projectColumns(
  data: Float64Array,
  nSamples: number,
  nFeatures: number,
  cols: readonly number[]
): Float64Array {
  const out = new Float64Array(nSamples * cols.length);
  let pos = 0;
  for (let i = 0; i < nSamples; i++) {
    const rowBase = i * nFeatures;
    for (const c of cols) out[pos++] = data[rowBase + c] as number;
  }
  return out;
}

/** Neumaier compensated sum. */
function compensatedSum(values: ArrayLike<number>, count: number): number {
  let sum = 0;
  let comp = 0;
  for (let i = 0; i < count; i++) {
    const v = values[i] as number;
    const t = sum + v;
    comp += Math.abs(sum) >= Math.abs(v) ? sum - t + v : v - t + sum;
    sum = t;
  }
  return sum + comp;
}

function assertFeatureCount(nFeatures: number, expected: number, X: Tensor): void {
  if (nFeatures !== expected) {
    throw new InvalidParameterError(
      `X has ${nFeatures} features, expected ${expected}`,
      "X",
      X.shape
    );
  }
}

// ---------------------------------------------------------------------------
// SimpleImputer
// ---------------------------------------------------------------------------

type ImputeStrategy = "mean" | "median" | "most_frequent" | "constant";

/**
 * Simple imputation transformer for completing missing values.
 *
 * Replaces missing values (NaN, or the value given by `missingValues`) using a
 * statistic of each feature column (mean, median, most frequent value) or a
 * constant. The statistics ignore missing entries. For `most_frequent`, ties are
 * broken towards the smallest value. The output is always float64.
 *
 * A column without any observed value gets 0 for `mean`, `median` and
 * `most_frequent` when `keepEmptyFeatures` is true (the default), and is
 * dropped from the output when it is false (its statistic is then NaN).
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
  private strategy: ImputeStrategy;
  private fillValue: number;
  private missingValues: number;
  private keepEmptyFeatures: boolean;
  private statistics_?: number[];
  private keptColumns_?: number[];
  private nFeatures_?: number;
  private fitted = false;

  /**
   * @param options - Configuration options
   * @param options.strategy - Imputation strategy (default: 'mean')
   * @param options.fillValue - Value to use when strategy='constant' (default: 0)
   * @param options.missingValues - Placeholder that marks a missing value (default: NaN)
   * @param options.keepEmptyFeatures - Keep columns that are entirely missing in fit,
   *   filled with 0 (default: true). When false they are dropped from the output.
   */
  constructor(
    options: {
      readonly strategy?: ImputeStrategy;
      readonly fillValue?: number;
      readonly missingValues?: number;
      readonly keepEmptyFeatures?: boolean;
    } = {}
  ) {
    this.strategy = options.strategy ?? "mean";
    this.fillValue = options.fillValue ?? 0;
    this.missingValues = options.missingValues ?? Number.NaN;
    this.keepEmptyFeatures = options.keepEmptyFeatures ?? true;

    const validStrategies = ["mean", "median", "most_frequent", "constant"];
    if (!validStrategies.includes(this.strategy)) {
      throw new InvalidParameterError(
        `strategy must be one of ${validStrategies.join(", ")}; got '${this.strategy}'`,
        "strategy",
        this.strategy
      );
    }
    if (typeof this.fillValue !== "number") {
      throw new InvalidParameterError("fillValue must be a number", "fillValue", this.fillValue);
    }
    validateMissingValues(this.missingValues);
    if (typeof this.keepEmptyFeatures !== "boolean") {
      throw new InvalidParameterError(
        "keepEmptyFeatures must be a boolean",
        "keepEmptyFeatures",
        this.keepEmptyFeatures
      );
    }
  }

  /**
   * Compute the imputation statistics from training data.
   *
   * @param X - Training data of shape (n_samples, n_features), may contain missing values
   * @returns this
   * @throws {ShapeError} If X is not 2D
   * @throws {InvalidParameterError} If X has no samples
   */
  fit(X: Tensor): this {
    const { data, nSamples, nFeatures } = readDense(X, "SimpleImputer", this.missingValues);
    if (nSamples === 0) {
      throw new InvalidParameterError("Cannot fit on empty array", "X", nSamples);
    }

    const statistics: number[] = [];
    const kept: number[] = [];
    const column = new Float64Array(nSamples);

    for (let j = 0; j < nFeatures; j++) {
      let count = 0;
      for (let i = 0; i < nSamples; i++) {
        const v = data[i * nFeatures + j] as number;
        if (!Number.isNaN(v)) column[count++] = v;
      }

      let stat: number;
      let keep = true;
      if (this.strategy === "constant") {
        stat = this.fillValue;
      } else if (count === 0) {
        stat = this.keepEmptyFeatures ? 0 : Number.NaN;
        keep = this.keepEmptyFeatures;
      } else if (this.strategy === "mean") {
        stat = compensatedSum(column, count) / count;
      } else if (this.strategy === "median") {
        const sorted = column.slice(0, count).sort();
        const mid = Math.floor(count / 2);
        stat =
          count % 2 === 0
            ? ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2
            : (sorted[mid] as number);
      } else {
        // most_frequent: scan runs of the sorted values; the first (smallest)
        // value reaching the highest count wins ties.
        const sorted = column.slice(0, count).sort();
        let bestVal = sorted[0] as number;
        let bestCount = 0;
        let runStart = 0;
        for (let i = 1; i <= count; i++) {
          if (i === count || sorted[i] !== sorted[runStart]) {
            if (i - runStart > bestCount) {
              bestCount = i - runStart;
              bestVal = sorted[runStart] as number;
            }
            runStart = i;
          }
        }
        stat = bestVal;
      }

      statistics.push(stat);
      if (keep) kept.push(j);
    }

    this.statistics_ = statistics;
    this.keptColumns_ = kept;
    this.nFeatures_ = nFeatures;
    this.fitted = true;
    return this;
  }

  /**
   * Impute missing values in X using the fitted statistics.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of shape (n_samples, n_kept_features); equal to the
   *   input shape unless columns were dropped (`keepEmptyFeatures: false`)
   */
  transform(X: Tensor): Tensor {
    const stats = this.statistics_;
    const kept = this.keptColumns_;
    if (!this.fitted || !stats || !kept || this.nFeatures_ === undefined) {
      throw new NotFittedError("SimpleImputer must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readDense(X, "SimpleImputer", this.missingValues);
    assertFeatureCount(nFeatures, this.nFeatures_, X);

    for (let i = 0; i < nSamples; i++) {
      const rowBase = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        if (Number.isNaN(data[rowBase + j] as number)) data[rowBase + j] = stats[j] as number;
      }
    }

    if (kept.length === nFeatures) return toTensor(data, nSamples, nFeatures, X);
    return toTensor(projectColumns(data, nSamples, nFeatures, kept), nSamples, kept.length, X);
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
      missingValues: this.missingValues,
      keepEmptyFeatures: this.keepEmptyFeatures,
    };
  }

  /**
   * Update parameters. The fitted state is discarded, so call `fit` again.
   *
   * @throws {InvalidParameterError} On an unknown parameter or invalid value
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(
      params,
      ["strategy", "fillValue", "missingValues", "keepEmptyFeatures"],
      "SimpleImputer"
    );
    const next = new SimpleImputer({
      ...this.getParams(),
      ...definedEntries(params),
    } as ConstructorParameters<typeof SimpleImputer>[0]);
    this.strategy = next.strategy;
    this.fillValue = next.fillValue;
    this.missingValues = next.missingValues;
    this.keepEmptyFeatures = next.keepEmptyFeatures;
    delete this.statistics_;
    delete this.keptColumns_;
    delete this.nFeatures_;
    this.fitted = false;
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): SimpleImputer {
    return new SimpleImputer(this.getParams() as ConstructorParameters<typeof SimpleImputer>[0]);
  }

  /**
   * Get the computed imputation statistics per feature (NaN for a dropped column).
   */
  get statistics(): number[] {
    if (!this.fitted || !this.statistics_) {
      throw new NotFittedError("SimpleImputer must be fitted first");
    }
    return [...this.statistics_];
  }
}

// ---------------------------------------------------------------------------
// KNNImputer
// ---------------------------------------------------------------------------

/**
 * KNN-based imputation for completing missing values.
 *
 * Follows `sklearn.impute.KNNImputer`. For every row with missing values, the
 * distance to each training row is the NaN-aware Euclidean distance
 * `sqrt(n_features / n_shared * sum((a - b)^2))` over the features present in
 * both rows. Each missing value is then the mean of that feature over the
 * `nNeighbors` nearest training rows that have the feature observed (a
 * distance-weighted mean for `weights: 'distance'`; if a donor is at distance 0
 * only the donors at distance 0 are used). When no donor exists, the training
 * column mean is used (0 if the column has no observed value). The output is
 * always float64.
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
  private nNeighbors: number;
  private weights: "uniform" | "distance";
  private missingValues: number;
  private keepEmptyFeatures: boolean;
  private trainFlat_?: Float64Array;
  private colMeans_?: Float64Array;
  private keptColumns_?: number[];
  private nTrain_ = 0;
  private nFeatures_?: number;
  private fitted = false;

  /**
   * @param options - Configuration options
   * @param options.nNeighbors - Number of neighbors to use (default: 5)
   * @param options.weights - Weight function: 'uniform' or 'distance' (default: 'uniform')
   * @param options.missingValues - Placeholder that marks a missing value (default: NaN)
   * @param options.keepEmptyFeatures - Keep columns that are entirely missing in fit,
   *   filled with 0 (default: true). When false they are dropped from the output.
   */
  constructor(
    options: {
      readonly nNeighbors?: number;
      readonly weights?: "uniform" | "distance";
      readonly missingValues?: number;
      readonly keepEmptyFeatures?: boolean;
    } = {}
  ) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.weights = options.weights ?? "uniform";
    this.missingValues = options.missingValues ?? Number.NaN;
    this.keepEmptyFeatures = options.keepEmptyFeatures ?? true;

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
    validateMissingValues(this.missingValues);
    if (typeof this.keepEmptyFeatures !== "boolean") {
      throw new InvalidParameterError(
        "keepEmptyFeatures must be a boolean",
        "keepEmptyFeatures",
        this.keepEmptyFeatures
      );
    }
  }

  /**
   * Store the training data used to find neighbors.
   *
   * @param X - Training data of shape (n_samples, n_features), may contain missing values
   * @throws {ShapeError} If X is not 2D
   * @throws {InvalidParameterError} If X has no samples
   */
  fit(X: Tensor): this {
    const { data, nSamples, nFeatures } = readDense(X, "KNNImputer", this.missingValues);
    if (nSamples === 0) {
      throw new InvalidParameterError("Cannot fit on empty array", "X", nSamples);
    }

    const means = new Float64Array(nFeatures);
    const kept: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      let count = 0;
      let sum = 0;
      for (let i = 0; i < nSamples; i++) {
        const v = data[i * nFeatures + j] as number;
        if (!Number.isNaN(v)) {
          sum += v;
          count++;
        }
      }
      means[j] = count > 0 ? sum / count : 0;
      if (count > 0 || this.keepEmptyFeatures) kept.push(j);
    }

    this.trainFlat_ = data;
    this.colMeans_ = means;
    this.keptColumns_ = kept;
    this.nTrain_ = nSamples;
    this.nFeatures_ = nFeatures;
    this.fitted = true;
    return this;
  }

  /**
   * Impute the missing values of X from the training data.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of shape (n_samples, n_kept_features); equal to the
   *   input shape unless columns were dropped (`keepEmptyFeatures: false`)
   */
  transform(X: Tensor): Tensor {
    const trainFlat = this.trainFlat_;
    const colMeans = this.colMeans_;
    const kept = this.keptColumns_;
    if (!this.fitted || !trainFlat || !colMeans || !kept || this.nFeatures_ === undefined) {
      throw new NotFittedError("KNNImputer must be fitted before transform");
    }
    const { data: out, nSamples, nFeatures } = readDense(X, "KNNImputer", this.missingValues);
    assertFeatureCount(nFeatures, this.nFeatures_, X);

    const nTrain = this.nTrain_;
    const useDistance = this.weights === "distance";
    const kMax = Math.min(this.nNeighbors, nTrain);

    // Scratch buffers reused across query rows.
    const query = new Float64Array(nFeatures);
    const dist = new Float64Array(nTrain);
    const nbrIdx = new Int32Array(kMax);
    const nbrDist = new Float64Array(kMax);
    const colIdx = new Int32Array(kMax);
    const colDist = new Float64Array(kMax);

    for (let i = 0; i < nSamples; i++) {
      const outBase = i * nFeatures;
      let hasMissing = false;
      for (let j = 0; j < nFeatures; j++) {
        const v = out[outBase + j] as number;
        query[j] = v;
        if (Number.isNaN(v)) hasMissing = true;
      }
      if (!hasMissing) continue;

      this.computeDistances(query, nFeatures, trainFlat, nTrain, dist);
      const nGlobal = selectNearest(dist, nTrain, -1, trainFlat, nFeatures, nbrIdx, nbrDist);

      for (let j = 0; j < nFeatures; j++) {
        if (!Number.isNaN(query[j] as number)) continue;

        // The k nearest rows overall serve when all of them have feature j;
        // otherwise search again among the rows that do.
        let idx = nbrIdx;
        let dst = nbrDist;
        let count = nGlobal;
        for (let n = 0; n < nGlobal; n++) {
          if (Number.isNaN(trainFlat[(nbrIdx[n] as number) * nFeatures + j] as number)) {
            count = selectNearest(dist, nTrain, j, trainFlat, nFeatures, colIdx, colDist);
            idx = colIdx;
            dst = colDist;
            break;
          }
        }

        if (count === 0) {
          out[outBase + j] = colMeans[j] as number;
          continue;
        }

        let zeroWeights = false;
        if (useDistance) {
          for (let n = 0; n < count; n++) {
            if ((dst[n] as number) === 0) {
              zeroWeights = true;
              break;
            }
          }
        }
        let weightSum = 0;
        let valueSum = 0;
        for (let n = 0; n < count; n++) {
          const d = dst[n] as number;
          let w = 1;
          if (useDistance) w = zeroWeights ? (d === 0 ? 1 : 0) : 1 / d;
          weightSum += w;
          valueSum += w * (trainFlat[(idx[n] as number) * nFeatures + j] as number);
        }
        out[outBase + j] = valueSum / weightSum;
      }
    }

    if (kept.length === nFeatures) return toTensor(out, nSamples, nFeatures, X);
    return toTensor(projectColumns(out, nSamples, nFeatures, kept), nSamples, kept.length, X);
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  getParams(): Record<string, unknown> {
    return {
      nNeighbors: this.nNeighbors,
      weights: this.weights,
      missingValues: this.missingValues,
      keepEmptyFeatures: this.keepEmptyFeatures,
    };
  }

  /**
   * Update parameters. The fitted state is discarded, so call `fit` again.
   *
   * @throws {InvalidParameterError} On an unknown parameter or invalid value
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(
      params,
      ["nNeighbors", "weights", "missingValues", "keepEmptyFeatures"],
      "KNNImputer"
    );
    const next = new KNNImputer({
      ...this.getParams(),
      ...definedEntries(params),
    } as ConstructorParameters<typeof KNNImputer>[0]);
    this.nNeighbors = next.nNeighbors;
    this.weights = next.weights;
    this.missingValues = next.missingValues;
    this.keepEmptyFeatures = next.keepEmptyFeatures;
    delete this.trainFlat_;
    delete this.colMeans_;
    delete this.keptColumns_;
    delete this.nFeatures_;
    this.nTrain_ = 0;
    this.fitted = false;
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): KNNImputer {
    return new KNNImputer(this.getParams() as ConstructorParameters<typeof KNNImputer>[0]);
  }

  /**
   * NaN-aware Euclidean distance from `query` to every training row.
   * Rows that share no observed feature with the query, and rows whose distance
   * is not finite, get `Infinity` (they are never neighbors).
   */
  private computeDistances(
    query: Float64Array,
    nFeatures: number,
    trainFlat: Float64Array,
    nTrain: number,
    dist: Float64Array
  ): void {
    for (let t = 0; t < nTrain; t++) {
      const tBase = t * nFeatures;
      let sumSq = 0;
      let shared = 0;
      for (let j = 0; j < nFeatures; j++) {
        const qVal = query[j] as number;
        const tVal = trainFlat[tBase + j] as number;
        if (Number.isNaN(qVal) || Number.isNaN(tVal)) continue;
        const diff = qVal - tVal;
        sumSq += diff * diff;
        shared++;
      }
      if (shared === 0) {
        dist[t] = Number.POSITIVE_INFINITY;
        continue;
      }
      const d = Math.sqrt((sumSq / shared) * nFeatures);
      dist[t] = Number.isFinite(d) ? d : Number.POSITIVE_INFINITY;
    }
  }
}

/**
 * Select the nearest training rows by a bounded insertion into the small
 * result arrays (cheaper than sorting all rows). Only rows with a finite
 * distance qualify; when `column >= 0` the row must also have that feature
 * observed. Ties keep the earlier row. Returns the number of neighbors found.
 */
function selectNearest(
  dist: Float64Array,
  nTrain: number,
  column: number,
  trainFlat: Float64Array,
  nFeatures: number,
  outIdx: Int32Array,
  outDist: Float64Array
): number {
  const k = outIdx.length;
  let filled = 0;
  for (let t = 0; t < nTrain; t++) {
    const d = dist[t] as number;
    if (d === Number.POSITIVE_INFINITY) continue;
    if (column >= 0 && Number.isNaN(trainFlat[t * nFeatures + column] as number)) continue;
    if (filled < k) {
      let p = filled - 1;
      while (p >= 0 && (outDist[p] as number) > d) {
        outDist[p + 1] = outDist[p] as number;
        outIdx[p + 1] = outIdx[p] as number;
        p--;
      }
      outDist[p + 1] = d;
      outIdx[p + 1] = t;
      filled++;
    } else if (d < (outDist[k - 1] as number)) {
      let p = k - 2;
      while (p >= 0 && (outDist[p] as number) > d) {
        outDist[p + 1] = outDist[p] as number;
        outIdx[p + 1] = outIdx[p] as number;
        p--;
      }
      outDist[p + 1] = d;
      outIdx[p + 1] = t;
    }
  }
  return filled;
}

// ---------------------------------------------------------------------------
// MissingIndicator
// ---------------------------------------------------------------------------

/**
 * Binary indicator for missing values.
 *
 * Generates a matrix where 1 marks a missing value (NaN, or the value given by
 * `missingValues`) and 0 a present value, as float64. Useful as an additional
 * feature set to complement imputation.
 *
 * With `features: 'missing-only'` (the default) only columns that had missing
 * values during fit are output. If `errorOnNew` is true (the default, as in
 * scikit-learn), transform throws when another column contains missing values,
 * because the indicator would silently ignore them.
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
  private features: "missing-only" | "all";
  private missingValues: number;
  private errorOnNew: boolean;
  private missingFeatures_?: number[];
  private nFeatures_?: number;
  private fitted = false;

  /**
   * @param options - Configuration options
   * @param options.features - Which features to include:
   *   - 'missing-only': only features that had missing values during fit (default)
   *   - 'all': all features
   * @param options.missingValues - Placeholder that marks a missing value (default: NaN)
   * @param options.errorOnNew - With 'missing-only', throw in transform when a column that had
   *   no missing values in fit has some (default: true)
   */
  constructor(
    options: {
      readonly features?: "missing-only" | "all";
      readonly missingValues?: number;
      readonly errorOnNew?: boolean;
    } = {}
  ) {
    this.features = options.features ?? "missing-only";
    this.missingValues = options.missingValues ?? Number.NaN;
    this.errorOnNew = options.errorOnNew ?? true;
    if (this.features !== "missing-only" && this.features !== "all") {
      throw new InvalidParameterError(
        "features must be 'missing-only' or 'all'",
        "features",
        this.features
      );
    }
    validateMissingValues(this.missingValues);
    if (typeof this.errorOnNew !== "boolean") {
      throw new InvalidParameterError(
        "errorOnNew must be a boolean",
        "errorOnNew",
        this.errorOnNew
      );
    }
  }

  /**
   * Find the columns that contain missing values.
   *
   * @throws {ShapeError} If X is not 2D
   * @throws {InvalidParameterError} If X has no samples
   */
  fit(X: Tensor): this {
    const { data, nSamples, nFeatures } = readDense(X, "MissingIndicator", this.missingValues);
    if (nSamples === 0) {
      throw new InvalidParameterError("Cannot fit on empty array", "X", nSamples);
    }

    if (this.features === "all") {
      this.missingFeatures_ = Array.from({ length: nFeatures }, (_, i) => i);
    } else {
      this.missingFeatures_ = columnsWithMissing(data, nSamples, nFeatures);
    }
    this.nFeatures_ = nFeatures;
    this.fitted = true;
    return this;
  }

  /**
   * Build the indicator matrix.
   *
   * @returns Float64 tensor of shape (n_samples, n_indicator_features)
   * @throws {DataValidationError} With `errorOnNew`, if a column that had no missing values
   *   during fit has some now
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.missingFeatures_ || this.nFeatures_ === undefined) {
      throw new NotFittedError("MissingIndicator must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readDense(X, "MissingIndicator", this.missingValues);
    assertFeatureCount(nFeatures, this.nFeatures_, X);

    if (this.features === "missing-only" && this.errorOnNew) {
      const known = new Set(this.missingFeatures_);
      const unexpected = columnsWithMissing(data, nSamples, nFeatures).filter((c) => !known.has(c));
      if (unexpected.length > 0) {
        throw new DataValidationError(
          `The columns [${unexpected.join(", ")}] have missing values in transform but had none ` +
            "during fit; set errorOnNew to false to ignore them"
        );
      }
    }

    const cols = this.missingFeatures_;
    const outCols = cols.length;
    const result = new Float64Array(nSamples * outCols);
    for (let i = 0; i < nSamples; i++) {
      for (let k = 0; k < outCols; k++) {
        result[i * outCols + k] = Number.isNaN(data[i * nFeatures + (cols[k] as number)] as number)
          ? 1
          : 0;
      }
    }
    return toTensor(result, nSamples, outCols, X);
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  getParams(): Record<string, unknown> {
    return {
      features: this.features,
      missingValues: this.missingValues,
      errorOnNew: this.errorOnNew,
    };
  }

  /**
   * Update parameters. The fitted state is discarded, so call `fit` again.
   *
   * @throws {InvalidParameterError} On an unknown parameter or invalid value
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(params, ["features", "missingValues", "errorOnNew"], "MissingIndicator");
    const next = new MissingIndicator({
      ...this.getParams(),
      ...definedEntries(params),
    } as ConstructorParameters<typeof MissingIndicator>[0]);
    this.features = next.features;
    this.missingValues = next.missingValues;
    this.errorOnNew = next.errorOnNew;
    delete this.missingFeatures_;
    delete this.nFeatures_;
    this.fitted = false;
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): MissingIndicator {
    return new MissingIndicator(
      this.getParams() as ConstructorParameters<typeof MissingIndicator>[0]
    );
  }

  /**
   * Get the indices of the features that produce an indicator column.
   */
  get features_(): number[] {
    if (!this.fitted || !this.missingFeatures_) {
      throw new NotFittedError("MissingIndicator must be fitted first");
    }
    return [...this.missingFeatures_];
  }
}

function columnsWithMissing(data: Float64Array, nSamples: number, nFeatures: number): number[] {
  const cols: number[] = [];
  for (let j = 0; j < nFeatures; j++) {
    for (let i = 0; i < nSamples; i++) {
      if (Number.isNaN(data[i * nFeatures + j] as number)) {
        cols.push(j);
        break;
      }
    }
  }
  return cols;
}
