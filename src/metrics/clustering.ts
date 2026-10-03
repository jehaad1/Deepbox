import { isNumericTypedArray, isTypedArray, type NumericTypedArray } from "../core";
import { DataValidationError, DTypeError, InvalidParameterError, ShapeError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../random/random";
import {
  assertFiniteNumber,
  assertSameSize,
  compensatedSum,
  createFlatOffsetter,
  denseFloat64,
  euclideanDistance,
  tryDenseNumeric,
} from "./_internal";

/** Distance input accepted by {@link silhouetteScore} and {@link silhouetteSamples}. */
export type SilhouetteMetric = "euclidean" | "precomputed";

/** How the two label entropies are combined into the normalizer of NMI and AMI. */
export type AverageMethod = "min" | "geometric" | "arithmetic" | "max";

/** Sampling options of {@link silhouetteScore}. */
export type SilhouetteScoreOptions = {
  /** Number of samples drawn without replacement; required when n_samples > 2000. */
  sampleSize?: number;
  /** Seed of the sampler. Without it the global Deepbox random state is used. */
  randomState?: number;
};

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

function getNumericTensorData(t: Tensor, name: string): NumericTypedArray {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} must be numeric (string tensors not supported)`);
  }
  if (t.dtype === "int64") {
    throw new DTypeError(`${name} must be numeric (int64 tensors not supported)`);
  }
  const data = t.data;
  if (!isTypedArray(data) || !isNumericTypedArray(data)) {
    throw new DTypeError(`${name} must be a numeric tensor`);
  }
  return data;
}

type FeatureLayout = {
  readonly data: NumericTypedArray;
  readonly nSamples: number;
  readonly nFeatures: number;
};

/** Validate dtype and shape of a feature matrix (1D is read as a single feature). */
function getFeatureLayout(X: Tensor): FeatureLayout {
  const data = getNumericTensorData(X, "X");
  if (X.ndim === 0 || X.ndim > 2) {
    throw new ShapeError("X must be a 1D or 2D tensor");
  }
  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.ndim === 1 ? 1 : (X.shape[1] ?? 0);
  if (nFeatures === 0) {
    throw new ShapeError("X must have at least one feature");
  }
  return { data, nSamples, nFeatures };
}

function assertFiniteRow(v: number, sample: number, feature: number): void {
  assertFiniteNumber(v, "X", `sample ${sample}, feature ${feature}`);
}

/**
 * Copy features into a dense row-major Float64Array, validating finiteness
 * once up front so distance loops run validation-free. With `indices`, only
 * those rows are gathered (and validated).
 */
function toDenseFeatures(
  X: Tensor,
  layout: FeatureLayout,
  indices: Int32Array | null
): Float64Array {
  const { data, nSamples, nFeatures } = layout;
  if (indices === null) {
    const dense = denseFloat64(X, "X", false);
    for (let i = 0; i < dense.length; i++) {
      const v = dense[i] as number;
      if (!Number.isFinite(v)) assertFiniteRow(v, Math.floor(i / nFeatures), i % nFeatures);
    }
    return dense;
  }
  const n = indices.length;
  const dense = new Float64Array(n * nFeatures);
  const sampleStride = X.strides[0] ?? nFeatures;
  const featureStride = X.ndim === 1 ? 0 : (X.strides[1] ?? 1);
  for (let i = 0; i < n; i++) {
    const src = indices[i] as number;
    if (src < 0 || src >= nSamples) {
      throw new DataValidationError(`sample index out of range: ${src}`);
    }
    const base = X.offset + src * sampleStride;
    for (let k = 0; k < nFeatures; k++) {
      const v = data[base + k * featureStride];
      if (v === undefined) {
        throw new DataValidationError(`X is missing a value at sample ${src}, feature ${k}`);
      }
      assertFiniteRow(v, src, k);
      dense[i * nFeatures + k] = v;
    }
  }
  return dense;
}

type EncodedLabels = {
  readonly codes: Int32Array;
  readonly nClusters: number;
};

/**
 * Map arbitrary discrete labels to consecutive integer codes (in order of first
 * appearance). Floats must hold integral values; strings are rejected.
 */
function encodeLabels(labels: Tensor, name: string): EncodedLabels {
  if (labels.dtype === "string") {
    throw new DTypeError(`${name}: string labels not supported for clustering metrics`);
  }

  const n = labels.size;
  const codes = new Int32Array(n);
  const data = labels.data;
  let next = 0;

  if (data instanceof BigInt64Array || data instanceof BigUint64Array) {
    const offsetter = createFlatOffsetter(labels);
    const map = new Map<bigint, number>();
    for (let i = 0; i < n; i++) {
      const v = data[offsetter(i)];
      if (v === undefined) {
        throw new DataValidationError(`${name} must contain a value for index ${i}`);
      }
      let code = map.get(v);
      if (code === undefined) {
        code = next++;
        map.set(v, code);
      }
      codes[i] = code;
    }
    return { codes, nClusters: next };
  }

  if (!isTypedArray(data)) {
    throw new DTypeError(`${name} has unsupported backing storage for labels`);
  }

  const values = tryDenseNumeric(labels);
  if (values === null) {
    throw new DTypeError(`${name} has unsupported backing storage for labels`);
  }
  const mustBeIntegral = labels.dtype === "float32" || labels.dtype === "float64";
  const map = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const v = values[i] as number;
    assertFiniteNumber(v, name, `index ${i}`);
    if (mustBeIntegral && !Number.isInteger(v)) {
      throw new DataValidationError(
        `${name} must contain discrete labels; found non-integer ${String(v)} at index ${i}`
      );
    }
    let code = map.get(v);
    if (code === undefined) {
      code = next++;
      map.set(v, code);
    }
    codes[i] = code;
  }
  return { codes, nClusters: next };
}

// ---------------------------------------------------------------------------
// Contingency statistics (ARI, MI family, FMI, homogeneity)
// ---------------------------------------------------------------------------

function comb2(x: number): number {
  return x <= 1 ? 0 : (x * (x - 1)) / 2;
}

type ContingencyStats = {
  readonly contingencyDense: Int32Array | null;
  readonly contingencySparse: Map<number, number> | null;
  readonly trueCount: Int32Array;
  readonly predCount: Int32Array;
  readonly nTrue: number;
  readonly nPred: number;
  readonly n: number;
};

const MAX_DENSE_CONTINGENCY_CELLS = 4_000_000; // 16 MB of Int32

function buildContingencyStats(labelsTrue: Tensor, labelsPred: Tensor): ContingencyStats {
  assertSameSize(labelsTrue, labelsPred, "labelsTrue", "labelsPred");

  const n = labelsTrue.size;
  const encT = encodeLabels(labelsTrue, "labelsTrue");
  const encP = encodeLabels(labelsPred, "labelsPred");

  const trueCodes = encT.codes;
  const predCodes = encP.codes;
  const nTrue = encT.nClusters;
  const nPred = encP.nClusters;

  const trueCount = new Int32Array(nTrue);
  const predCount = new Int32Array(nPred);
  for (let i = 0; i < n; i++) {
    trueCount[trueCodes[i] as number]!++;
    predCount[predCodes[i] as number]!++;
  }

  const denseSize = nTrue * nPred;
  if (denseSize > 0 && denseSize <= MAX_DENSE_CONTINGENCY_CELLS) {
    const contingency = new Int32Array(denseSize);
    for (let i = 0; i < n; i++) {
      contingency[(trueCodes[i] as number) * nPred + (predCodes[i] as number)]!++;
    }
    return {
      contingencyDense: contingency,
      contingencySparse: null,
      trueCount,
      predCount,
      nTrue,
      nPred,
      n,
    };
  }

  const contingency = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const key = (trueCodes[i] as number) * nPred + (predCodes[i] as number);
    contingency.set(key, (contingency.get(key) ?? 0) + 1);
  }
  return {
    contingencyDense: null,
    contingencySparse: contingency,
    trueCount,
    predCount,
    nTrue,
    nPred,
    n,
  };
}

/** Visit every non-empty contingency cell as (trueCode, predCode, count). */
function forEachCell(
  stats: ContingencyStats,
  visit: (t: number, p: number, nij: number) => void
): void {
  const { contingencyDense, contingencySparse, nPred } = stats;
  if (contingencyDense) {
    for (let idx = 0; idx < contingencyDense.length; idx++) {
      const nij = contingencyDense[idx] as number;
      if (nij <= 0) continue;
      const t = Math.floor(idx / nPred);
      visit(t, idx - t * nPred, nij);
    }
  } else if (contingencySparse) {
    for (const [key, nij] of contingencySparse) {
      if (nij <= 0) continue;
      const t = Math.floor(key / nPred);
      visit(t, key - t * nPred, nij);
    }
  }
}

/** Sum of C(count, 2) over cells, rows and columns (pair-counting indices). */
function pairCountSums(stats: ContingencyStats): {
  sumCells: number;
  sumTrue: number;
  sumPred: number;
} {
  let sumCells = 0;
  forEachCell(stats, (_t, _p, nij) => {
    sumCells += comb2(nij);
  });
  let sumTrue = 0;
  for (let i = 0; i < stats.trueCount.length; i++) sumTrue += comb2(stats.trueCount[i] as number);
  let sumPred = 0;
  for (let j = 0; j < stats.predCount.length; j++) sumPred += comb2(stats.predCount[j] as number);
  return { sumCells, sumTrue, sumPred };
}

function entropyFromCountArray(counts: Int32Array, n: number): number {
  if (n === 0) return 0;
  let h = 0;
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i] as number;
    if (c > 0) {
      const p = c / n;
      h -= p * Math.log(p);
    }
  }
  return h;
}

/** Mutual information in nats, clipped at 0 like scikit-learn (rounding can push it below). */
function mutualInformationFromContingency(stats: ContingencyStats): number {
  const { trueCount, predCount, n } = stats;
  if (n === 0) return 0;
  let mi = 0;
  forEachCell(stats, (t, p, nij) => {
    const ni = trueCount[t] as number;
    const nj = predCount[p] as number;
    mi += (nij / n) * Math.log((n * nij) / (ni * nj));
  });
  return mi > 0 ? mi : 0;
}

function buildLogFactorials(n: number): Float64Array {
  const out = new Float64Array(n + 1);
  for (let i = 1; i <= n; i++) {
    out[i] = (out[i - 1] as number) + Math.log(i);
  }
  return out;
}

const LOG_EXP_UNDERFLOW_CUTOFF = -745;

/** Multiplicity of each distinct positive cluster size. */
function sizeMultiplicities(counts: Int32Array): Map<number, number> {
  const out = new Map<number, number>();
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i] as number;
    if (c > 0) out.set(c, (out.get(c) ?? 0) + 1);
  }
  return out;
}

/**
 * Expected mutual information under the hypergeometric model of random
 * labelings with the observed cluster sizes (Vinh et al., 2010).
 *
 * Clusters of equal size contribute identical terms, so each distinct size pair
 * is evaluated once and weighted by its multiplicity.
 */
function expectedMutualInformation(stats: ContingencyStats): number {
  const { trueCount, predCount, n } = stats;
  if (n <= 1) return 0;

  const rows = sizeMultiplicities(trueCount);
  const cols = sizeMultiplicities(predCount);
  const lf = buildLogFactorials(n);
  const lfN = lf[n] as number;

  let emi = 0;
  let comp = 0;

  for (const [a, rowWeight] of rows) {
    const lfA = (lf[a] as number) + (lf[n - a] as number);
    for (const [b, colWeight] of cols) {
      const nijMin = Math.max(1, a + b - n);
      const nijMax = Math.min(a, b);
      if (nijMin > nijMax) continue;

      const logConst = lfA + (lf[b] as number) + (lf[n - b] as number) - lfN;
      const weight = rowWeight * colWeight;

      for (let nij = nijMin; nij <= nijMax; nij++) {
        const logProbability =
          logConst -
          (lf[nij] as number) -
          (lf[a - nij] as number) -
          (lf[b - nij] as number) -
          (lf[n - a - b + nij] as number);
        if (logProbability < LOG_EXP_UNDERFLOW_CUTOFF) continue;

        const probability = Math.exp(logProbability);
        if (!Number.isFinite(probability) || probability === 0) continue;

        const miTerm = (nij / n) * Math.log((n * nij) / (a * b));
        const y = weight * probability * miTerm - comp;
        const t = emi + y;
        comp = t - emi - y;
        emi = t;
      }
    }
  }

  return emi;
}

function assertAverageMethod(method: AverageMethod): void {
  if (method !== "min" && method !== "max" && method !== "geometric" && method !== "arithmetic") {
    throw new InvalidParameterError(
      `Unsupported averageMethod: '${String(method)}'. Must be 'min', 'geometric', 'arithmetic' or 'max'`,
      "averageMethod",
      method
    );
  }
}

function averageEntropy(hTrue: number, hPred: number, method: AverageMethod): number {
  switch (method) {
    case "min":
      return Math.min(hTrue, hPred);
    case "max":
      return Math.max(hTrue, hPred);
    case "geometric":
      return Math.sqrt(hTrue * hPred);
    default:
      return (hTrue + hPred) / 2;
  }
}

// ---------------------------------------------------------------------------
// Silhouette
// ---------------------------------------------------------------------------

function validateSilhouetteLabels(labels: Tensor, nSamples: number): EncodedLabels {
  if (labels.size !== nSamples) {
    throw new ShapeError("labels length must match number of samples");
  }
  const enc = encodeLabels(labels, "labels");
  const k = enc.nClusters;
  if (k < 2 || k > nSamples - 1) {
    throw new InvalidParameterError(
      "silhouette requires 2 <= n_clusters <= n_samples - 1",
      "n_clusters",
      k
    );
  }
  return enc;
}

/**
 * Draw `k` distinct indices from [0, n) uniformly (reservoir sampling), sorted
 * ascending. A given `seed` uses its own generator and leaves the global state
 * untouched; otherwise the global Deepbox generator is consumed.
 */
function sampleIndices(n: number, k: number, seed: number | undefined): Int32Array {
  let random: () => number = __random;
  if (seed !== undefined) {
    const rng = new __SeededRandom(__seedToUint64(seed));
    random = () => rng.next();
  }

  const out = new Int32Array(k);
  for (let i = 0; i < k; i++) out[i] = i;
  for (let i = k; i < n; i++) {
    const j = __randomBelow(random, i + 1);
    if (j < k) out[j] = i;
  }
  out.sort();
  return out;
}

function reencodeSubset(codes: Int32Array): EncodedLabels {
  const out = new Int32Array(codes.length);
  const map = new Map<number, number>();
  let next = 0;
  for (let i = 0; i < codes.length; i++) {
    const v = codes[i] as number;
    let code = map.get(v);
    if (code === undefined) {
      code = next++;
      map.set(v, code);
    }
    out[i] = code;
  }
  return { codes: out, nClusters: next };
}

function atolForPrecomputed(dtype: Tensor["dtype"]): number {
  // scikit-learn allows 100 * machine epsilon of the matrix dtype on the diagonal.
  return dtype === "float32" || dtype === "float16" || dtype === "bfloat16" ? 1.1920929e-5 : 1e-12;
}

/**
 * Copy a precomputed distance matrix (or, with `indices`, only the sub-matrix of the
 * selected samples) into a dense row-major Float64Array, requiring every copied entry to
 * be finite and non-negative.
 */
function gatherDistances(
  X: Tensor,
  data: NumericTypedArray,
  indices: Int32Array | null
): Float64Array {
  const nTotal = X.shape[0] ?? 0;
  const rejectEntry = (v: number, row: number, col: number): never => {
    const where = `[${row},${col}]`;
    assertFiniteNumber(v, "X", `distance${where}`);
    throw new DataValidationError(
      `Precomputed distances must be non-negative; found ${String(v)} at ${where}`
    );
  };

  if (indices === null) {
    const D = denseFloat64(X, "X", false);
    for (let idx = 0; idx < D.length; idx++) {
      const v = D[idx] as number;
      if (!(Number.isFinite(v) && v >= 0)) rejectEntry(v, Math.floor(idx / nTotal), idx % nTotal);
    }
    return D;
  }

  const m = indices.length;
  const rowStride = X.strides[0] ?? nTotal;
  const colStride = X.strides[1] ?? 1;
  const D = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    const srcRow = indices[i] as number;
    const base = X.offset + srcRow * rowStride;
    for (let j = 0; j < m; j++) {
      const srcCol = indices[j] as number;
      const v = data[base + srcCol * colStride] as number;
      if (!(Number.isFinite(v) && v >= 0)) rejectEntry(v, srcRow, srcCol);
      D[i * m + j] = v;
    }
  }
  return D;
}

/**
 * Per-sample silhouette coefficients for the (optionally sub-sampled) data.
 *
 * `indices` selects the samples to use; `null` uses all of them. The cluster
 * count is re-checked after sub-sampling because a sample may miss clusters.
 */
function silhouetteValues(
  X: Tensor,
  labels: Tensor,
  metric: SilhouetteMetric,
  indices: Int32Array | null
): Float64Array {
  const nTotal = labels.size;
  if (nTotal < 2) {
    throw new InvalidParameterError("silhouette requires at least 2 samples", "n_samples", nTotal);
  }

  let fillRowSums: (i: number, codes: Int32Array, sums: Float64Array) => void;
  let enc: EncodedLabels;

  if (metric === "euclidean") {
    const layout = getFeatureLayout(X);
    if (layout.nSamples < 2) {
      throw new InvalidParameterError(
        "silhouette requires at least 2 samples",
        "n_samples",
        layout.nSamples
      );
    }
    enc = validateSilhouetteLabels(labels, layout.nSamples);
    const dense = toDenseFeatures(X, layout, indices);
    const d = layout.nFeatures;
    const n = indices ? indices.length : layout.nSamples;
    fillRowSums = (i, codes, sums) => {
      const baseI = i * d;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        sums[codes[j] as number]! += euclideanDistance(dense, baseI, dense, j * d, d);
      }
    };
  } else {
    const matrixData = getNumericTensorData(X, "X");
    if (X.ndim !== 2) {
      throw new ShapeError("X must be a 2D tensor for metric='precomputed'");
    }
    if (X.shape[0] !== nTotal || X.shape[1] !== nTotal) {
      throw new ShapeError(
        "For metric='precomputed', X must be a square [n_samples, n_samples] matrix"
      );
    }
    enc = validateSilhouetteLabels(labels, nTotal);
    const n = indices ? indices.length : nTotal;
    const D = gatherDistances(X, matrixData, indices);
    const atol = atolForPrecomputed(X.dtype);
    for (let i = 0; i < n; i++) {
      const d0 = D[i * n + i] as number;
      if (Math.abs(d0) > atol) {
        const src = indices ? (indices[i] as number) : i;
        throw new DataValidationError(
          `Precomputed distance matrix diagonal must be ~0; found ${String(d0)} at [${src},${src}]`
        );
      }
    }
    fillRowSums = (i, codes, sums) => {
      const row = i * n;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        sums[codes[j] as number]! += D[row + j] as number;
      }
    };
  }

  // Cluster ids of the used samples, renumbered 0..k-1 after sub-sampling.
  const n = indices ? indices.length : nTotal;
  let codes: Int32Array;
  let k: number;
  if (indices) {
    const subsetCodes = new Int32Array(n);
    for (let i = 0; i < n; i++) subsetCodes[i] = enc.codes[indices[i] as number] as number;
    ({ codes, nClusters: k } = reencodeSubset(subsetCodes));
  } else {
    ({ codes, nClusters: k } = enc);
  }

  if (k < 2 || k > n - 1) {
    throw new InvalidParameterError(
      "silhouette requires 2 <= n_clusters <= n_samples - 1",
      "n_clusters",
      k
    );
  }

  const clusterSizes = new Int32Array(k);
  for (let i = 0; i < n; i++) clusterSizes[codes[i] as number]!++;

  const sums = new Float64Array(k);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ci = codes[i] as number;
    const sizeOwn = clusterSizes[ci] as number;
    if (sizeOwn <= 1) continue; // singleton clusters score 0

    sums.fill(0);
    fillRowSums(i, codes, sums);

    const a = (sums[ci] as number) / (sizeOwn - 1);
    let b = Number.POSITIVE_INFINITY;
    for (let cl = 0; cl < k; cl++) {
      if (cl === ci) continue;
      const mean = (sums[cl] as number) / (clusterSizes[cl] as number);
      if (mean < b) b = mean;
    }

    const denom = Math.max(a, b);
    out[i] = denom > 0 ? (b - a) / denom : 0;
  }
  return out;
}

/**
 * Computes the mean Silhouette Coefficient over all samples.
 *
 * The Silhouette Coefficient for a sample measures how similar it is to its own
 * cluster compared to other clusters. Values range from -1 to 1, where higher
 * values indicate better-defined clusters. Samples in singleton clusters score 0.
 *
 * **Formula**: s(i) = (b(i) - a(i)) / max(a(i), b(i))
 * - a(i): mean intra-cluster distance for sample i
 * - b(i): mean nearest-cluster distance for sample i
 *
 * **Time Complexity**: O(n²) where n is the number of samples
 * **Space Complexity**: O(n + k) where k is the number of clusters
 *
 * @param X - Feature matrix of shape [n_samples, n_features], or a precomputed
 *   [n_samples, n_samples] distance matrix when `metric` is `'precomputed'`
 * @param labels - Cluster labels for each sample
 * @param metric - Distance metric: 'euclidean' (default) or 'precomputed'
 * @param options - Optional parameters
 * @param options.sampleSize - Number of samples to use for approximation (required when n > 2000)
 * @param options.randomState - Seed for reproducible sampling
 * @returns Mean silhouette coefficient in range [-1, 1]
 *
 * @throws {InvalidParameterError} If fewer than 2 samples, invalid sampleSize or randomState,
 *   a cluster count outside [2, n_samples - 1], or unsupported metric
 * @throws {ShapeError} If labels length doesn't match samples, or X shape is invalid
 * @throws {DTypeError} If labels are string or X is non-numeric
 * @throws {DataValidationError} If X contains non-finite values, or a precomputed matrix has
 *   negative entries or a non-zero diagonal
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function silhouetteScore(
  X: Tensor,
  labels: Tensor,
  metric: SilhouetteMetric = "euclidean",
  options?: SilhouetteScoreOptions
): number {
  if (metric !== "euclidean" && metric !== "precomputed") {
    throw new InvalidParameterError(
      `Unsupported metric: '${String(metric)}'. Must be 'euclidean' or 'precomputed'`,
      "metric",
      metric
    );
  }

  const sampleSize = options?.sampleSize;
  const randomState = options?.randomState;

  const nSamples = labels.size;
  if (nSamples < 2) {
    throw new InvalidParameterError(
      "silhouette requires at least 2 samples",
      "n_samples",
      nSamples
    );
  }

  const maxFull = 2000;
  if (sampleSize === undefined && nSamples > maxFull) {
    throw new InvalidParameterError(
      `silhouetteScore is O(n²) and n_samples=${nSamples} is too large for full computation; provide options.sampleSize`,
      "sampleSize",
      sampleSize
    );
  }

  if (sampleSize !== undefined) {
    if (!Number.isFinite(sampleSize) || !Number.isInteger(sampleSize)) {
      throw new InvalidParameterError("sampleSize must be an integer", "sampleSize", sampleSize);
    }
    if (sampleSize < 2 || sampleSize > nSamples) {
      throw new InvalidParameterError(
        "sampleSize must satisfy 2 <= sampleSize <= n_samples",
        "sampleSize",
        sampleSize
      );
    }
  }
  if (randomState !== undefined && !Number.isFinite(randomState)) {
    throw new InvalidParameterError(
      "randomState must be a finite number",
      "randomState",
      randomState
    );
  }

  const indices =
    sampleSize !== undefined && sampleSize < nSamples
      ? sampleIndices(nSamples, sampleSize, randomState)
      : null;

  const values = silhouetteValues(X, labels, metric, indices);
  return compensatedSum(values) / values.length;
}

/**
 * Computes the Silhouette Coefficient for each sample.
 *
 * Returns a tensor of per-sample silhouette values. Useful for identifying
 * samples that are well-clustered vs. poorly-clustered. Samples in singleton
 * clusters score 0.
 *
 * **Time Complexity**: O(n²) where n is the number of samples
 * **Space Complexity**: O(n + k) where k is the number of clusters
 *
 * @param X - Feature matrix of shape [n_samples, n_features], or a precomputed
 *   [n_samples, n_samples] distance matrix when `metric` is `'precomputed'`
 * @param labels - Cluster labels for each sample
 * @param metric - Distance metric: 'euclidean' (default) or 'precomputed'
 * @returns Float64 tensor of shape [n_samples] with coefficients in range [-1, 1]
 *
 * @throws {InvalidParameterError} If fewer than 2 samples, a cluster count outside
 *   [2, n_samples - 1], or unsupported metric
 * @throws {ShapeError} If labels length doesn't match samples, or X shape is invalid
 * @throws {DTypeError} If labels are string or X is non-numeric
 * @throws {DataValidationError} If X contains non-finite values, or a precomputed matrix has
 *   negative entries or a non-zero diagonal
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function silhouetteSamples(
  X: Tensor,
  labels: Tensor,
  metric: SilhouetteMetric = "euclidean"
): Tensor {
  if (metric !== "euclidean" && metric !== "precomputed") {
    throw new InvalidParameterError(
      `Unsupported metric: '${String(metric)}'. Must be 'euclidean' or 'precomputed'`,
      "metric",
      metric
    );
  }
  return tensor(silhouetteValues(X, labels, metric, null));
}

// ---------------------------------------------------------------------------
// Internal (centroid based) indices
// ---------------------------------------------------------------------------

/**
 * Multiply `values` in place by a power of two so that the largest magnitude is about 1,
 * when it is far from 1. The factor is exact, and indices that are ratios of squared
 * distances do not change, but squares of the scaled values can no longer overflow or
 * underflow.
 */
function scaleIntoSafeRange(values: Float64Array): void {
  let maxAbs = 0;
  for (let i = 0; i < values.length; i++) {
    const a = Math.abs(values[i] as number);
    if (a > maxAbs) maxAbs = a;
  }
  if (maxAbs === 0 || (maxAbs < 1e100 && maxAbs > 1e-100)) return;
  const factor = 2 ** -Math.max(-1000, Math.min(1000, Math.round(Math.log2(maxAbs))));
  for (let i = 0; i < values.length; i++) values[i] = (values[i] as number) * factor;
}

/** Per-cluster centroids as a flat [k * d] array plus cluster sizes. */
function clusterCentroids(
  dense: Float64Array,
  codes: Int32Array,
  k: number,
  nSamples: number,
  d: number
): { centroids: Float64Array; sizes: Int32Array } {
  const centroids = new Float64Array(k * d);
  const sizes = new Int32Array(k);
  for (let i = 0; i < nSamples; i++) {
    const c = codes[i] as number;
    sizes[c]!++;
    const cBase = c * d;
    const base = i * d;
    for (let f = 0; f < d; f++) centroids[cBase + f]! += dense[base + f] as number;
  }
  for (let c = 0; c < k; c++) {
    const sz = sizes[c] as number;
    if (sz <= 0) continue;
    const cBase = c * d;
    for (let f = 0; f < d; f++) centroids[cBase + f] = (centroids[cBase + f] as number) / sz;
  }
  return { centroids, sizes };
}

/**
 * Computes the Davies-Bouldin index.
 *
 * The Davies-Bouldin index measures the average similarity ratio of each cluster
 * with its most similar cluster. Lower values indicate better clustering.
 * Returns 0 when k < 2 or n_samples == 0. Returns Infinity when two clusters
 * with non-zero scatter share the same centroid.
 *
 * **Time Complexity**: O(n * d + k² * d) where n is samples, d is features, k is clusters
 * **Space Complexity**: O(n * d)
 *
 * @param X - Feature matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns Davies-Bouldin index (lower is better, 0 is minimum)
 *
 * @throws {ShapeError} If labels length doesn't match samples, or X shape is invalid
 * @throws {DTypeError} If labels are string or X is non-numeric
 * @throws {DataValidationError} If X contains non-finite values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function daviesBouldinScore(X: Tensor, labels: Tensor): number {
  const layout = getFeatureLayout(X);
  const { nSamples, nFeatures: d } = layout;
  if (nSamples === 0) return 0;

  if (labels.size !== nSamples) {
    throw new ShapeError("labels length must match number of samples");
  }

  const dense = toDenseFeatures(X, layout, null);
  const { codes, nClusters: k } = encodeLabels(labels, "labels");
  if (k < 2) return 0;

  const { centroids, sizes } = clusterCentroids(dense, codes, k, nSamples, d);

  const scatterSum = new Float64Array(k);
  for (let i = 0; i < nSamples; i++) {
    const c = codes[i] as number;
    scatterSum[c]! += euclideanDistance(dense, i * d, centroids, c * d, d);
  }
  const S = new Float64Array(k);
  for (let c = 0; c < k; c++) {
    const sz = sizes[c] as number;
    S[c] = sz > 0 ? (scatterSum[c] as number) / sz : 0;
  }

  let db = 0;
  for (let i = 0; i < k; i++) {
    let maxRatio = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      const dist = euclideanDistance(centroids, i * d, centroids, j * d, d);
      const spread = (S[i] as number) + (S[j] as number);
      // Coincident centroids of clusters with no scatter at all have nothing to separate.
      const ratio = dist === 0 ? (spread === 0 ? 0 : Number.POSITIVE_INFINITY) : spread / dist;
      if (ratio > maxRatio) maxRatio = ratio;
    }
    db += maxRatio;
  }

  return db / k;
}

/**
 * Computes the Calinski-Harabasz index (Variance Ratio Criterion).
 *
 * The score is the ratio of between-cluster dispersion to within-cluster
 * dispersion. Higher values indicate better-defined clusters.
 * Returns 0 when k < 2, n_samples == 0, or within-group sum of squares is 0.
 *
 * **Time Complexity**: O(n * d) where n is samples and d is features
 * **Space Complexity**: O(n * d)
 *
 * @param X - Feature matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns Calinski-Harabasz index (higher is better)
 *
 * @throws {ShapeError} If labels length doesn't match samples, or X shape is invalid
 * @throws {DTypeError} If labels are string or X is non-numeric
 * @throws {DataValidationError} If X contains non-finite values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function calinskiHarabaszScore(X: Tensor, labels: Tensor): number {
  const layout = getFeatureLayout(X);
  const { nSamples, nFeatures: d } = layout;
  if (nSamples === 0) return 0;

  if (labels.size !== nSamples) {
    throw new ShapeError("labels length must match number of samples");
  }

  const dense = toDenseFeatures(X, layout, null);
  scaleIntoSafeRange(dense);
  const { codes, nClusters: k } = encodeLabels(labels, "labels");

  const overallMean = new Float64Array(d);
  for (let i = 0; i < nSamples; i++) {
    const base = i * d;
    for (let f = 0; f < d; f++) overallMean[f]! += dense[base + f] as number;
  }
  for (let f = 0; f < d; f++) overallMean[f] = (overallMean[f] as number) / nSamples;

  const { centroids, sizes } = clusterCentroids(dense, codes, k, nSamples, d);

  let bgss = 0;
  for (let c = 0; c < k; c++) {
    const sz = sizes[c] as number;
    if (sz <= 0) continue;
    let distSq = 0;
    for (let f = 0; f < d; f++) {
      const diff = (centroids[c * d + f] as number) - (overallMean[f] as number);
      distSq += diff * diff;
    }
    bgss += sz * distSq;
  }

  let wgss = 0;
  for (let i = 0; i < nSamples; i++) {
    const cBase = (codes[i] as number) * d;
    const base = i * d;
    for (let f = 0; f < d; f++) {
      const diff = (dense[base + f] as number) - (centroids[cBase + f] as number);
      wgss += diff * diff;
    }
  }

  if (k < 2 || wgss === 0) return 0;
  return bgss / (k - 1) / (wgss / (nSamples - k));
}

// ---------------------------------------------------------------------------
// External (label based) indices
// ---------------------------------------------------------------------------

/**
 * Computes the Rand Index between two clusterings.
 *
 * The Rand Index is the fraction of sample pairs on which the two clusterings
 * agree (both put the pair in the same cluster, or both separate it). It is not
 * adjusted for chance; see {@link adjustedRandScore} for the adjusted version.
 *
 * **Time Complexity**: O(n + k₁ * k₂) where k₁, k₂ are cluster counts
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth cluster labels
 * @param labelsPred - Predicted cluster labels
 * @returns Rand Index in range [0, 1]; 1 for fewer than two samples
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @example
 * ```ts
 * import { randScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * randScore(tensor([0, 0, 1, 1]), tensor([0, 0, 1, 2])); // 0.8333333333333334
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function randScore(labelsTrue: Tensor, labelsPred: Tensor): number {
  const stats = buildContingencyStats(labelsTrue, labelsPred);
  const totalPairs = comb2(stats.n);
  if (totalPairs === 0) return 1;
  const { sumCells, sumTrue, sumPred } = pairCountSums(stats);
  // Pairs split by exactly one of the two clusterings disagree.
  const disagreements = sumTrue + sumPred - 2 * sumCells;
  return 1 - disagreements / totalPairs;
}

/**
 * Computes the Adjusted Rand Index (ARI).
 *
 * The ARI measures the similarity between two clusterings, adjusted for chance.
 * It ranges from -0.5 to 1, where 1 indicates perfect agreement, 0 indicates
 * random labeling, and negative values indicate worse than random.
 *
 * **Time Complexity**: O(n + k₁ * k₂) where k₁, k₂ are cluster counts
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth cluster labels
 * @param labelsPred - Predicted cluster labels
 * @returns Adjusted Rand Index (1 for identical partitions and for fewer than two samples)
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function adjustedRandScore(labelsTrue: Tensor, labelsPred: Tensor): number {
  const stats = buildContingencyStats(labelsTrue, labelsPred);

  const totalPairs = comb2(stats.n);
  if (totalPairs === 0) return 1;

  const { sumCells: sumComb, sumTrue: sumCombTrue, sumPred: sumCombPred } = pairCountSums(stats);

  const expectedIndex = (sumCombTrue * sumCombPred) / totalPairs;
  const maxIndex = (sumCombTrue + sumCombPred) / 2;

  const denom = maxIndex - expectedIndex;
  if (denom === 0) return 1;

  return (sumComb - expectedIndex) / denom;
}

/**
 * Computes the Mutual Information between two clusterings, in nats.
 *
 * **Time Complexity**: O(n + k₁ * k₂)
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth cluster labels
 * @param labelsPred - Predicted cluster labels
 * @returns Mutual information (non-negative; 0 for independent labelings)
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @example
 * ```ts
 * import { mutualInfoScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * mutualInfoScore(tensor([0, 0, 1, 1]), tensor([0, 0, 1, 1])); // ln 2 = 0.6931471805599453
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function mutualInfoScore(labelsTrue: Tensor, labelsPred: Tensor): number {
  return mutualInformationFromContingency(buildContingencyStats(labelsTrue, labelsPred));
}

/**
 * Computes the Adjusted Mutual Information (AMI) between two clusterings.
 *
 * AMI adjusts the Mutual Information score to account for chance, providing
 * a normalized measure of agreement between two clusterings. Identical
 * partitions score 1, independent labelings score about 0, and the score can
 * be slightly negative for labelings that agree less than chance.
 *
 * **Time Complexity**: O(n + k₁ * k₂ * min(k₁, k₂))
 * **Space Complexity**: O(k₁ * k₂ + n)
 *
 * @param labelsTrue - Ground truth cluster labels
 * @param labelsPred - Predicted cluster labels
 * @param averageMethod - Method to compute the normalizer: 'min', 'geometric', 'arithmetic' (default), or 'max'
 * @returns Adjusted Mutual Information score, at most 1
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {InvalidParameterError} If averageMethod is not one of the supported values
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function adjustedMutualInfoScore(
  labelsTrue: Tensor,
  labelsPred: Tensor,
  averageMethod: AverageMethod = "arithmetic"
): number {
  const stats = buildContingencyStats(labelsTrue, labelsPred);
  const { n, trueCount, predCount, nTrue, nPred } = stats;
  assertAverageMethod(averageMethod);
  if (n <= 1) return 1;
  // Neither labeling splits the data: perfect agreement by convention.
  if (nTrue === 1 && nPred === 1) return 1;
  // Identical partitions (up to relabeling) have exactly one non-empty cell per
  // cluster. Decide this combinatorially: the 0/0 limit is otherwise lost in rounding.
  let nonEmptyCells = 0;
  forEachCell(stats, () => {
    nonEmptyCells++;
  });
  if (nonEmptyCells === nTrue && nTrue === nPred) return 1;

  const mi = mutualInformationFromContingency(stats);
  const hTrue = entropyFromCountArray(trueCount, n);
  const hPred = entropyFromCountArray(predCount, n);
  const emi = expectedMutualInformation(stats);

  const normalizer = averageEntropy(hTrue, hPred, averageMethod);
  // The expected MI reaches the normalizer when chance alone can attain the maximum
  // (for example a labeling with a single cluster under 'min'); the score is then 0.
  const denom = normalizer - emi;
  if (Math.abs(denom) < 1e-12) return 0;

  const ami = (mi - emi) / denom;
  if (!Number.isFinite(ami)) return 0;
  if (ami > 1) return 1;
  if (ami < -1) return -1;
  return ami;
}

/**
 * Computes the Normalized Mutual Information (NMI) between two clusterings.
 *
 * NMI normalizes the Mutual Information score to scale between 0 and 1,
 * where 1 indicates perfect correlation between clusterings. It is not
 * adjusted for chance; see {@link adjustedMutualInfoScore}.
 *
 * **Time Complexity**: O(n + k₁ * k₂)
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth cluster labels
 * @param labelsPred - Predicted cluster labels
 * @param averageMethod - Method to compute the normalizer: 'min', 'geometric', 'arithmetic' (default), or 'max'
 * @returns Normalized Mutual Information score in range [0, 1]
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {InvalidParameterError} If averageMethod is not one of the supported values
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function normalizedMutualInfoScore(
  labelsTrue: Tensor,
  labelsPred: Tensor,
  averageMethod: AverageMethod = "arithmetic"
): number {
  const stats = buildContingencyStats(labelsTrue, labelsPred);
  const { n, trueCount, predCount } = stats;
  assertAverageMethod(averageMethod);

  const ht = entropyFromCountArray(trueCount, n);
  const hp = entropyFromCountArray(predCount, n);

  if (ht === 0 || hp === 0) {
    return ht === 0 && hp === 0 ? 1.0 : 0.0;
  }

  const mi = mutualInformationFromContingency(stats);
  const normalizer = averageEntropy(ht, hp, averageMethod);
  if (normalizer === 0) return 0;

  const nmi = mi / normalizer;
  if (nmi > 1) return 1;
  if (nmi < 0) return 0;
  return nmi;
}

/**
 * Computes the Fowlkes-Mallows Index (FMI).
 *
 * The FMI is the geometric mean of pairwise precision and recall.
 * It ranges from 0 to 1, where 1 indicates perfect agreement.
 *
 * **Time Complexity**: O(n + k₁ * k₂)
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth cluster labels
 * @param labelsPred - Predicted cluster labels
 * @returns Fowlkes-Mallows score in range [0, 1]
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function fowlkesMallowsScore(labelsTrue: Tensor, labelsPred: Tensor): number {
  const stats = buildContingencyStats(labelsTrue, labelsPred);
  if (stats.n === 0) return 1.0;

  const { sumCells: tk, sumTrue: pk, sumPred: qk } = pairCountSums(stats);
  if (pk === 0 || qk === 0) return 0.0;
  return Math.sqrt(tk / pk) * Math.sqrt(tk / qk);
}

/** Homogeneity and completeness from one contingency table. */
function homogeneityAndCompleteness(
  labelsTrue: Tensor,
  labelsPred: Tensor
): { homogeneity: number; completeness: number } {
  const stats = buildContingencyStats(labelsTrue, labelsPred);
  const { trueCount, predCount, n } = stats;
  if (n === 0) return { homogeneity: 1.0, completeness: 1.0 };

  const mi = mutualInformationFromContingency(stats);
  const hTrue = entropyFromCountArray(trueCount, n);
  const hPred = entropyFromCountArray(predCount, n);
  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
  return {
    homogeneity: hTrue === 0 ? 1.0 : clamp01(mi / hTrue),
    completeness: hPred === 0 ? 1.0 : clamp01(mi / hPred),
  };
}

/**
 * Computes the homogeneity score of a clustering.
 *
 * A clustering result satisfies homogeneity if all of its clusters contain
 * only data points which are members of a single class. Score ranges from
 * 0 to 1, where 1 indicates perfectly homogeneous clustering.
 *
 * **Time Complexity**: O(n + k₁ * k₂)
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth class labels
 * @param labelsPred - Predicted cluster labels
 * @returns Homogeneity score in range [0, 1]
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function homogeneityScore(labelsTrue: Tensor, labelsPred: Tensor): number {
  return homogeneityAndCompleteness(labelsTrue, labelsPred).homogeneity;
}

/**
 * Computes the completeness score of a clustering.
 *
 * A clustering result satisfies completeness if all data points that are
 * members of a given class are assigned to the same cluster. Score ranges
 * from 0 to 1, where 1 indicates perfectly complete clustering.
 *
 * **Time Complexity**: O(n + k₁ * k₂)
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth class labels
 * @param labelsPred - Predicted cluster labels
 * @returns Completeness score in range [0, 1]
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function completenessScore(labelsTrue: Tensor, labelsPred: Tensor): number {
  return homogeneityAndCompleteness(labelsTrue, labelsPred).completeness;
}

/**
 * Computes the V-measure score of a clustering.
 *
 * V-measure is the weighted harmonic mean of homogeneity and completeness. With
 * beta > 1 completeness is weighted more; with beta < 1 homogeneity is weighted more.
 *
 * **Formula**: v = (1 + beta) * h * c / (beta * h + c)
 *
 * **Time Complexity**: O(n + k₁ * k₂)
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth class labels
 * @param labelsPred - Predicted cluster labels
 * @param beta - Weight of completeness relative to homogeneity (default: 1.0)
 * @returns V-measure score in range [0, 1]
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {InvalidParameterError} If beta is not a positive finite number
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function vMeasureScore(labelsTrue: Tensor, labelsPred: Tensor, beta = 1.0): number {
  if (!Number.isFinite(beta) || beta <= 0) {
    throw new InvalidParameterError("beta must be a positive finite number", "beta", beta);
  }

  const { homogeneity: h, completeness: c } = homogeneityAndCompleteness(labelsTrue, labelsPred);

  if (h + c === 0) return 0;
  return ((1 + beta) * h * c) / (beta * h + c);
}
