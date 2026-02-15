import {
  getNumericElement,
  isNumericTypedArray,
  isTypedArray,
  type NumericTypedArray,
} from "../core";
import { DataValidationError, DTypeError, InvalidParameterError, ShapeError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";
import {
  assertFiniteNumber,
  assertSameSize,
  computeLogicalStrides,
  createFlatOffsetter,
  type FlatOffsetter,
} from "./_internal";

function isBigIntTypedArray(x: unknown): x is BigInt64Array | BigUint64Array {
  return x instanceof BigInt64Array || x instanceof BigUint64Array;
}

function readIndex<T>(arr: ArrayLike<T>, index: number, name: string): T {
  const v = arr[index];
  if (v === undefined) {
    throw new DataValidationError(`${name} index out of range: ${index}`);
  }
  return v;
}

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

function getFeatureAccessor(X: Tensor) {
  const data = getNumericTensorData(X, "X");

  if (X.ndim === 0 || X.ndim > 2) {
    throw new ShapeError("X must be a 1D or 2D tensor");
  }

  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.ndim === 1 ? 1 : (X.shape[1] ?? 0);
  if (nFeatures === 0) {
    throw new ShapeError("X must have at least one feature");
  }

  const logicalStrides = computeLogicalStrides(X.shape);
  const logical0 = logicalStrides[0];
  if (logical0 === undefined) throw new ShapeError("Invalid logical strides");

  const sampleStride = X.strides[0] ?? logical0 ?? nFeatures;

  const featureStride = X.ndim === 1 ? 0 : (X.strides[1] ?? logicalStrides[1] ?? 1);

  if (X.ndim === 2 && featureStride === 0) {
    throw new ShapeError("X must have a non-degenerate feature stride");
  }

  return {
    data,
    nSamples,
    nFeatures,
    sampleStride,
    featureStride,
    offset: X.offset,
  };
}

type EncodedLabels = {
  readonly codes: Int32Array;
  readonly nClusters: number;
};

function readLabelValue(
  labels: Tensor,
  offsetter: FlatOffsetter,
  index: number,
  name: string
): string | number | boolean | bigint {
  const data = labels.data;
  const flat = offsetter(index);

  if (isBigIntTypedArray(data)) {
    const v = data[flat];
    if (v === undefined) {
      throw new DataValidationError(`${name} must contain a value for index ${index}`);
    }
    return v;
  }

  if (isTypedArray(data) && isNumericTypedArray(data)) {
    const v = getNumericElement(data, flat);
    assertFiniteNumber(v, name, `index ${index}`);

    if ((labels.dtype === "float32" || labels.dtype === "float64") && !Number.isInteger(v)) {
      throw new DataValidationError(
        `${name} must contain discrete labels; found non-integer ${String(v)} at index ${index}`
      );
    }

    if (labels.dtype === "int64" && (!Number.isInteger(v) || !Number.isSafeInteger(v))) {
      throw new DataValidationError(
        `${name} contains an int64 value that cannot be represented safely as a number at index ${index}`
      );
    }

    return v;
  }

  if (Array.isArray(data)) {
    const v = data[flat];
    if (v === undefined || v === null) {
      throw new DataValidationError(`${name} must contain a value for index ${index}`);
    }

    if (typeof v === "string") return v;
    if (typeof v === "boolean") return v;

    if (typeof v === "number") {
      assertFiniteNumber(v, name, `index ${index}`);

      if ((labels.dtype === "float32" || labels.dtype === "float64") && !Number.isInteger(v)) {
        throw new DataValidationError(
          `${name} must contain discrete labels; found non-integer ${String(v)} at index ${index}`
        );
      }

      if (labels.dtype === "int64" && (!Number.isInteger(v) || !Number.isSafeInteger(v))) {
        throw new DataValidationError(
          `${name} contains an int64 value that cannot be represented safely as a number at index ${index}`
        );
      }

      return v;
    }

    throw new DTypeError(`${name} must contain primitive labels (string, boolean, number, bigint)`);
  }

  throw new DTypeError(`${name} has unsupported backing storage for labels`);
}

function encodeLabels(labels: Tensor, name: string): EncodedLabels {
  if (labels.dtype === "string") {
    throw new DTypeError(`${name}: string labels not supported for clustering metrics`);
  }

  const n = labels.size;
  const codes = new Int32Array(n);
  const offsetter = createFlatOffsetter(labels);

  const map = new Map<string | number | boolean | bigint, number>();
  let next = 0;

  for (let i = 0; i < n; i++) {
    const raw = readLabelValue(labels, offsetter, i, name);

    const existing = map.get(raw);
    if (existing === undefined) {
      map.set(raw, next);
      codes[i] = next;
      next++;
    } else {
      codes[i] = existing;
    }
  }

  return { codes, nClusters: next };
}

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
    const t = readIndex(trueCodes, i, "trueCodes");
    const p = readIndex(predCodes, i, "predCodes");
    trueCount[t] = (trueCount[t] ?? 0) + 1;
    predCount[p] = (predCount[p] ?? 0) + 1;
  }

  const denseSize = nTrue * nPred;
  const maxDenseCells = 4_000_000; // ~16MB Int32

  if (denseSize > 0 && denseSize <= maxDenseCells) {
    const contingency = new Int32Array(denseSize);
    for (let i = 0; i < n; i++) {
      const t = readIndex(trueCodes, i, "trueCodes");
      const p = readIndex(predCodes, i, "predCodes");
      const idx = t * nPred + p;
      contingency[idx] = (contingency[idx] ?? 0) + 1;
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
    const t = readIndex(trueCodes, i, "trueCodes");
    const p = readIndex(predCodes, i, "predCodes");
    const key = t * nPred + p;
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

function entropyFromCountArray(counts: Int32Array, n: number): number {
  if (n === 0) return 0;
  let h = 0;

  for (let i = 0; i < counts.length; i++) {
    const c = readIndex(counts, i, "counts");
    if (c > 0) {
      const p = c / n;
      h -= p * Math.log(p);
    }
  }
  return h;
}

function mutualInformationFromContingency(stats: ContingencyStats): number {
  const { contingencyDense, contingencySparse, trueCount, predCount, nPred, n } = stats;
  if (n === 0) return 0;

  let mi = 0;

  if (contingencyDense) {
    for (let idx = 0; idx < contingencyDense.length; idx++) {
      const nij = readIndex(contingencyDense, idx, "contingencyDense");
      if (nij <= 0) continue;

      const t = Math.floor(idx / nPred);
      const p = idx - t * nPred;

      const ni = readIndex(trueCount, t, "trueCount");
      const nj = readIndex(predCount, p, "predCount");
      if (ni > 0 && nj > 0) {
        mi += (nij / n) * Math.log((n * nij) / (ni * nj));
      }
    }
    return mi;
  }

  if (contingencySparse) {
    for (const [key, nij] of contingencySparse) {
      if (nij <= 0) continue;

      const t = Math.floor(key / nPred);
      const p = key - t * nPred;

      const ni = readIndex(trueCount, t, "trueCount");
      const nj = readIndex(predCount, p, "predCount");
      if (ni > 0 && nj > 0) {
        mi += (nij / n) * Math.log((n * nij) / (ni * nj));
      }
    }
    return mi;
  }

  return 0;
}

function buildLogFactorials(n: number): Float64Array {
  const out = new Float64Array(n + 1);
  for (let i = 1; i <= n; i++) {
    out[i] = (out[i - 1] ?? 0) + Math.log(i);
  }
  return out;
}

function logCombination(n: number, k: number, logFactorials: Float64Array): number {
  if (k < 0 || k > n) return Number.NEGATIVE_INFINITY;
  const a = readIndex(logFactorials, n, "logFactorials");
  const b = readIndex(logFactorials, k, "logFactorials");
  const c = readIndex(logFactorials, n - k, "logFactorials");
  return a - b - c;
}

const LOG_EXP_UNDERFLOW_CUTOFF = -745;

function expectedMutualInformation(stats: ContingencyStats): number {
  const { trueCount, predCount, n } = stats;
  if (n <= 1) return 0;

  const rowSums = Array.from(trueCount);
  const colSums = Array.from(predCount);
  const logFactorials = buildLogFactorials(n);

  let emi = 0;
  let comp = 0;

  for (const a of rowSums) {
    if (a <= 0) continue;
    for (const b of colSums) {
      if (b <= 0) continue;

      const nijMin = Math.max(1, a + b - n);
      const nijMax = Math.min(a, b);
      if (nijMin > nijMax) continue;

      const logDenominator = logCombination(n, b, logFactorials);

      for (let nij = nijMin; nij <= nijMax; nij++) {
        const logProbability =
          logCombination(a, nij, logFactorials) +
          logCombination(n - a, b - nij, logFactorials) -
          logDenominator;

        if (logProbability < LOG_EXP_UNDERFLOW_CUTOFF) continue;

        const probability = Math.exp(logProbability);
        if (!Number.isFinite(probability) || probability === 0) continue;

        const miTerm = (nij / n) * Math.log((n * nij) / (a * b));
        const y = probability * miTerm - comp;
        const t = emi + y;
        comp = t - emi - y;
        emi = t;
      }
    }
  }

  return emi;
}

type AverageMethod = "min" | "geometric" | "arithmetic" | "max";

function averageEntropy(hTrue: number, hPred: number, method: AverageMethod): number {
  if (method === "min") return Math.min(hTrue, hPred);
  if (method === "max") return Math.max(hTrue, hPred);
  if (method === "geometric") return Math.sqrt(hTrue * hPred);
  return (hTrue + hPred) / 2;
}

function euclideanDistance(
  data: NumericTypedArray,
  offset: number,
  sampleStride: number,
  featureStride: number,
  nFeatures: number,
  i: number,
  j: number
): number {
  let sum = 0;
  const baseI = offset + i * sampleStride;
  const baseJ = offset + j * sampleStride;

  for (let k = 0; k < nFeatures; k++) {
    const vi = getNumericElement(data, baseI + k * featureStride);
    const vj = getNumericElement(data, baseJ + k * featureStride);
    assertFiniteNumber(vi, "X", `sample ${i}, feature ${k}`);
    assertFiniteNumber(vj, "X", `sample ${j}, feature ${k}`);
    const d = vi - vj;
    sum += d * d;
  }
  return Math.sqrt(sum);
}

type SilhouetteMetric = "euclidean" | "precomputed";

function getPrecomputedDistanceAccessor(X: Tensor, n: number) {
  const data = getNumericTensorData(X, "X");

  if (X.ndim !== 2) {
    throw new ShapeError("X must be a 2D tensor for metric='precomputed'");
  }
  const rows = X.shape[0] ?? 0;
  const cols = X.shape[1] ?? 0;
  if (rows !== n || cols !== n) {
    throw new ShapeError(
      "For metric='precomputed', X must be a square [n_samples, n_samples] matrix"
    );
  }

  const logicalStrides = computeLogicalStrides(X.shape);
  const rowStride = X.strides[0] ?? logicalStrides[0] ?? n;
  const colStride = X.strides[1] ?? logicalStrides[1] ?? 1;

  if (rowStride === 0 || colStride === 0) {
    throw new ShapeError("Precomputed distance matrix must have non-degenerate strides");
  }

  const base = X.offset;

  return (i: number, j: number) => {
    const v = getNumericElement(data, base + i * rowStride + j * colStride);
    assertFiniteNumber(v, "X", `distance[${i},${j}]`);
    if (v < 0) {
      throw new DataValidationError(
        `Precomputed distances must be non-negative; found ${String(v)} at [${i},${j}]`
      );
    }
    return v;
  };
}

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

function reservoirSampleIndices(n: number, k: number, seed: number | undefined): Int32Array {
  let state = (seed ?? 0) >>> 0;
  const hasSeed = seed !== undefined;

  const randU32 = () => {
    if (!hasSeed) return (Math.random() * 0x1_0000_0000) >>> 0;
    state = (1664525 * state + 1013904223) >>> 0;
    return state;
  };

  const randInt = (exclusiveMax: number) => randU32() % exclusiveMax;

  const out = new Int32Array(k);
  for (let i = 0; i < k; i++) out[i] = i;

  for (let i = k; i < n; i++) {
    const j = randInt(i + 1);
    if (j < k) out[j] = i;
  }
  return out;
}

function reencodeSubset(codes: Int32Array): {
  codes: Int32Array;
  nClusters: number;
} {
  const out = new Int32Array(codes.length);
  const map = new Map<number, number>();
  let next = 0;

  for (let i = 0; i < codes.length; i++) {
    const v = readIndex(codes, i, "codes");
    const existing = map.get(v);
    if (existing === undefined) {
      map.set(v, next);
      out[i] = next;
      next++;
    } else {
      out[i] = existing;
    }
  }

  return { codes: out, nClusters: next };
}

function silhouetteMeanEuclidean(X: Tensor, labels: Tensor, indices: Int32Array | null): number {
  const { data, nSamples, nFeatures, sampleStride, featureStride, offset } = getFeatureAccessor(X);

  const n = indices ? indices.length : nSamples;
  if (nSamples < 2 || n < 2) {
    throw new InvalidParameterError("silhouette requires at least 2 samples", "n_samples", n);
  }

  const encAll = validateSilhouetteLabels(labels, nSamples);

  const subsetCodesRaw = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const src = indices ? readIndex(indices, i, "indices") : i;
    const c = readIndex(encAll.codes, src, "labels.codes");
    subsetCodesRaw[i] = c;
  }

  const re = reencodeSubset(subsetCodesRaw);
  const codes = re.codes;
  const k = re.nClusters;

  if (k < 2 || k > n - 1) {
    throw new InvalidParameterError(
      "silhouette requires 2 <= n_clusters <= n_samples - 1",
      "n_clusters",
      k
    );
  }

  const clusterSizes = new Int32Array(k);
  for (let i = 0; i < n; i++) {
    const ci = readIndex(codes, i, "codes");
    clusterSizes[ci] = (clusterSizes[ci] ?? 0) + 1;
  }

  const sumsToClusters = new Float64Array(k);

  let sum = 0;
  let comp = 0;

  for (let i = 0; i < n; i++) {
    const ci = readIndex(codes, i, "codes");
    const sizeOwn = readIndex(clusterSizes, ci, "clusterSizes");

    if (sizeOwn <= 1) continue; // singleton => 0

    sumsToClusters.fill(0);

    const srcI = indices ? readIndex(indices, i, "indices") : i;

    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const srcJ = indices ? readIndex(indices, j, "indices") : j;
      const cj = readIndex(codes, j, "codes");

      const d = euclideanDistance(data, offset, sampleStride, featureStride, nFeatures, srcI, srcJ);
      sumsToClusters[cj] = (sumsToClusters[cj] ?? 0) + d;
    }

    const a = readIndex(sumsToClusters, ci, "sumsToClusters") / (sizeOwn - 1);

    let b = Infinity;
    for (let cl = 0; cl < k; cl++) {
      if (cl === ci) continue;
      const sz = readIndex(clusterSizes, cl, "clusterSizes");
      if (sz <= 0) continue;
      const mean = readIndex(sumsToClusters, cl, "sumsToClusters") / sz;
      if (mean < b) b = mean;
    }

    if (!Number.isFinite(b) || b === Infinity) continue;

    const denom = Math.max(a, b);
    const s = denom > 0 ? (b - a) / denom : 0;

    const y = s - comp;
    const t = sum + y;
    comp = t - sum - y;
    sum = t;
  }

  return sum / n;
}

function silhouetteMeanPrecomputed(X: Tensor, labels: Tensor, indices: Int32Array | null): number {
  const nSamples = labels.size;
  if (nSamples < 2) {
    throw new InvalidParameterError(
      "silhouette requires at least 2 samples",
      "n_samples",
      nSamples
    );
  }

  const encAll = validateSilhouetteLabels(labels, nSamples);
  const dist = getPrecomputedDistanceAccessor(X, nSamples);

  const n = indices ? indices.length : nSamples;
  if (n < 2) {
    throw new InvalidParameterError("silhouette requires at least 2 samples", "n_samples", n);
  }

  // Validate diagonal ~ 0 for relevant indices
  for (let i = 0; i < n; i++) {
    const src = indices ? readIndex(indices, i, "indices") : i;
    const d0 = dist(src, src);
    if (!Number.isFinite(d0) || Math.abs(d0) > 1e-12) {
      throw new DataValidationError(
        `Precomputed distance matrix diagonal must be ~0; found ${String(d0)} at [${src},${src}]`
      );
    }
  }

  const subsetCodesRaw = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const src = indices ? readIndex(indices, i, "indices") : i;
    const c = readIndex(encAll.codes, src, "labels.codes");
    subsetCodesRaw[i] = c;
  }

  const re = reencodeSubset(subsetCodesRaw);
  const codes = re.codes;
  const k = re.nClusters;

  if (k < 2 || k > n - 1) {
    throw new InvalidParameterError(
      "silhouette requires 2 <= n_clusters <= n_samples - 1",
      "n_clusters",
      k
    );
  }

  const clusterSizes = new Int32Array(k);
  for (let i = 0; i < n; i++) {
    const ci = readIndex(codes, i, "codes");
    clusterSizes[ci] = (clusterSizes[ci] ?? 0) + 1;
  }

  const sumsToClusters = new Float64Array(k);

  let sum = 0;
  let comp = 0;

  for (let i = 0; i < n; i++) {
    const ci = readIndex(codes, i, "codes");
    const sizeOwn = readIndex(clusterSizes, ci, "clusterSizes");
    if (sizeOwn <= 1) continue;

    sumsToClusters.fill(0);

    const srcI = indices ? readIndex(indices, i, "indices") : i;

    for (let j = 0; j < n; j++) {
      if (i === j) continue;

      const srcJ = indices ? readIndex(indices, j, "indices") : j;
      const cj = readIndex(codes, j, "codes");

      const d = dist(srcI, srcJ);
      sumsToClusters[cj] = (sumsToClusters[cj] ?? 0) + d;
    }

    const a = readIndex(sumsToClusters, ci, "sumsToClusters") / (sizeOwn - 1);

    let b = Infinity;
    for (let cl = 0; cl < k; cl++) {
      if (cl === ci) continue;
      const sz = readIndex(clusterSizes, cl, "clusterSizes");
      if (sz <= 0) continue;
      const mean = readIndex(sumsToClusters, cl, "sumsToClusters") / sz;
      if (mean < b) b = mean;
    }

    if (!Number.isFinite(b) || b === Infinity) continue;

    const denom = Math.max(a, b);
    const s = denom > 0 ? (b - a) / denom : 0;

    const y = s - comp;
    const t = sum + y;
    comp = t - sum - y;
    sum = t;
  }

  return sum / n;
}

/**
 * Computes the mean Silhouette Coefficient over all samples.
 *
 * The Silhouette Coefficient for a sample measures how similar it is to its own
 * cluster compared to other clusters. Values range from -1 to 1, where higher
 * values indicate better-defined clusters.
 *
 * **Formula**: s(i) = (b(i) - a(i)) / max(a(i), b(i))
 * - a(i): mean intra-cluster distance for sample i
 * - b(i): mean nearest-cluster distance for sample i
 *
 * **Time Complexity**: O(n²) where n is the number of samples
 * **Space Complexity**: O(n + k) where k is the number of clusters
 *
 * @param X - Feature matrix of shape [n_samples, n_features], or a precomputed distance matrix
 * @param labels - Cluster labels for each sample
 * @param metric - Distance metric: 'euclidean' (default) or 'precomputed'
 * @param options - Optional parameters
 * @param options.sampleSize - Number of samples to use for approximation (required when n > 2000)
 * @param options.randomState - Seed for reproducible sampling
 * @returns Mean silhouette coefficient in range [-1, 1]
 *
 * @throws {InvalidParameterError} If fewer than 2 samples, invalid sampleSize, or unsupported metric
 * @throws {ShapeError} If labels length doesn't match samples, or X shape is invalid
 * @throws {DTypeError} If labels are string or X is non-numeric
 * @throws {DataValidationError} If X contains non-finite values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function silhouetteScore(
  X: Tensor,
  labels: Tensor,
  metric: SilhouetteMetric = "euclidean",
  options?: { sampleSize?: number; randomState?: number }
): number {
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

  const indices =
    sampleSize !== undefined && sampleSize < nSamples
      ? reservoirSampleIndices(nSamples, sampleSize, randomState)
      : null;

  if (metric === "euclidean") return silhouetteMeanEuclidean(X, labels, indices);
  if (metric === "precomputed") return silhouetteMeanPrecomputed(X, labels, indices);

  throw new InvalidParameterError(
    `Unsupported metric: '${String(metric)}'. Must be 'euclidean' or 'precomputed'`,
    "metric",
    metric
  );
}

/**
 * Computes the Silhouette Coefficient for each sample.
 *
 * Returns a tensor of per-sample silhouette values. Useful for identifying
 * samples that are well-clustered vs. poorly-clustered.
 *
 * **Time Complexity**: O(n²) where n is the number of samples
 * **Space Complexity**: O(n + k) where k is the number of clusters
 *
 * @param X - Feature matrix of shape [n_samples, n_features], or a precomputed distance matrix
 * @param labels - Cluster labels for each sample
 * @param metric - Distance metric: 'euclidean' (default) or 'precomputed'
 * @returns Tensor of silhouette coefficients in range [-1, 1] for each sample
 *
 * @throws {InvalidParameterError} If fewer than 2 samples or unsupported metric
 * @throws {ShapeError} If labels length doesn't match samples, or X shape is invalid
 * @throws {DTypeError} If labels are string or X is non-numeric
 * @throws {DataValidationError} If X contains non-finite values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function silhouetteSamples(
  X: Tensor,
  labels: Tensor,
  metric: SilhouetteMetric = "euclidean"
): Tensor {
  const nSamples = labels.size;
  if (nSamples < 2) {
    throw new InvalidParameterError(
      "silhouette requires at least 2 samples",
      "n_samples",
      nSamples
    );
  }

  const enc = validateSilhouetteLabels(labels, nSamples);
  const codes = enc.codes;
  const k = enc.nClusters;

  const silhouettes = new Float64Array(nSamples);

  const clusterSizes = new Int32Array(k);
  for (let i = 0; i < nSamples; i++) {
    const ci = readIndex(codes, i, "codes");
    clusterSizes[ci] = (clusterSizes[ci] ?? 0) + 1;
  }

  const sumsToClusters = new Float64Array(k);

  if (metric === "euclidean") {
    const { data, nFeatures, sampleStride, featureStride, offset } = getFeatureAccessor(X);

    for (let i = 0; i < nSamples; i++) {
      const ci = readIndex(codes, i, "codes");
      const sizeOwn = readIndex(clusterSizes, ci, "clusterSizes");
      if (sizeOwn <= 1) {
        silhouettes[i] = 0;
        continue;
      }

      sumsToClusters.fill(0);

      for (let j = 0; j < nSamples; j++) {
        if (i === j) continue;
        const cj = readIndex(codes, j, "codes");

        const d = euclideanDistance(data, offset, sampleStride, featureStride, nFeatures, i, j);
        sumsToClusters[cj] = (sumsToClusters[cj] ?? 0) + d;
      }

      const a = readIndex(sumsToClusters, ci, "sumsToClusters") / (sizeOwn - 1);

      let b = Infinity;
      for (let cl = 0; cl < k; cl++) {
        if (cl === ci) continue;
        const sz = readIndex(clusterSizes, cl, "clusterSizes");
        if (sz <= 0) continue;
        const mean = readIndex(sumsToClusters, cl, "sumsToClusters") / sz;
        if (mean < b) b = mean;
      }

      if (!Number.isFinite(b) || b === Infinity) {
        silhouettes[i] = 0;
        continue;
      }

      const denom = Math.max(a, b);
      silhouettes[i] = denom > 0 ? (b - a) / denom : 0;
    }

    return tensor(silhouettes);
  }

  if (metric === "precomputed") {
    const dist = getPrecomputedDistanceAccessor(X, nSamples);

    for (let i = 0; i < nSamples; i++) {
      const d0 = dist(i, i);
      if (!Number.isFinite(d0) || Math.abs(d0) > 1e-12) {
        throw new DataValidationError(
          `Precomputed distance matrix diagonal must be ~0; found ${String(d0)} at [${i},${i}]`
        );
      }
    }

    for (let i = 0; i < nSamples; i++) {
      const ci = readIndex(codes, i, "codes");
      const sizeOwn = readIndex(clusterSizes, ci, "clusterSizes");
      if (sizeOwn <= 1) {
        silhouettes[i] = 0;
        continue;
      }

      sumsToClusters.fill(0);

      for (let j = 0; j < nSamples; j++) {
        if (i === j) continue;
        const cj = readIndex(codes, j, "codes");

        const d = dist(i, j);
        sumsToClusters[cj] = (sumsToClusters[cj] ?? 0) + d;
      }

      const a = readIndex(sumsToClusters, ci, "sumsToClusters") / (sizeOwn - 1);

      let b = Infinity;
      for (let cl = 0; cl < k; cl++) {
        if (cl === ci) continue;
        const sz = readIndex(clusterSizes, cl, "clusterSizes");
        if (sz <= 0) continue;
        const mean = readIndex(sumsToClusters, cl, "sumsToClusters") / sz;
        if (mean < b) b = mean;
      }

      if (!Number.isFinite(b) || b === Infinity) {
        silhouettes[i] = 0;
        continue;
      }

      const denom = Math.max(a, b);
      silhouettes[i] = denom > 0 ? (b - a) / denom : 0;
    }

    return tensor(silhouettes);
  }

  throw new InvalidParameterError(
    `Unsupported metric: '${String(metric)}'. Must be 'euclidean' or 'precomputed'`,
    "metric",
    metric
  );
}

/**
 * Computes the Davies-Bouldin index.
 *
 * The Davies-Bouldin index measures the average similarity ratio of each cluster
 * with its most similar cluster. Lower values indicate better clustering.
 * Returns 0 when k < 2 or n_samples == 0.
 *
 * **Time Complexity**: O(n * d + k² * d) where n is samples, d is features, k is clusters
 * **Space Complexity**: O(k * d)
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
  const { data, nSamples, nFeatures, sampleStride, featureStride, offset } = getFeatureAccessor(X);
  if (nSamples === 0) return 0;

  if (labels.size !== nSamples) {
    throw new ShapeError("labels length must match number of samples");
  }

  const enc = encodeLabels(labels, "labels");
  const codes = enc.codes;
  const k = enc.nClusters;
  if (k < 2) return 0;

  const centroids = new Array<Float64Array>(k);
  for (let c = 0; c < k; c++) centroids[c] = new Float64Array(nFeatures);

  const clusterSizes = new Int32Array(k);

  for (let i = 0; i < nSamples; i++) {
    const c = readIndex(codes, i, "codes");
    clusterSizes[c] = (clusterSizes[c] ?? 0) + 1;

    const base = offset + i * sampleStride;
    const centroid = readIndex(centroids, c, "centroids");
    for (let f = 0; f < nFeatures; f++) {
      const v = getNumericElement(data, base + f * featureStride);
      assertFiniteNumber(v, "X", `sample ${i}, feature ${f}`);
      centroid[f] = (centroid[f] ?? 0) + v;
    }
  }

  for (let c = 0; c < k; c++) {
    const sz = readIndex(clusterSizes, c, "clusterSizes");
    if (sz <= 0) continue;
    const centroid = readIndex(centroids, c, "centroids");
    for (let f = 0; f < nFeatures; f++) {
      centroid[f] = (centroid[f] ?? 0) / sz;
    }
  }

  const scatterSum = new Float64Array(k);

  for (let i = 0; i < nSamples; i++) {
    const c = readIndex(codes, i, "codes");
    const centroid = readIndex(centroids, c, "centroids");

    let distSq = 0;
    const base = offset + i * sampleStride;
    for (let f = 0; f < nFeatures; f++) {
      const v = getNumericElement(data, base + f * featureStride);
      assertFiniteNumber(v, "X", `sample ${i}, feature ${f}`);
      const d = v - (centroid[f] ?? 0);
      distSq += d * d;
    }

    scatterSum[c] = (scatterSum[c] ?? 0) + Math.sqrt(distSq);
  }

  const S = new Float64Array(k);
  for (let c = 0; c < k; c++) {
    const sz = readIndex(clusterSizes, c, "clusterSizes");
    const sc = readIndex(scatterSum, c, "scatterSum");
    S[c] = sz > 0 ? sc / sz : 0;
  }

  let db = 0;

  for (let i = 0; i < k; i++) {
    let maxRatio = Number.NEGATIVE_INFINITY;
    const ci = readIndex(centroids, i, "centroids");

    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      const cj = readIndex(centroids, j, "centroids");

      let distSq = 0;
      for (let f = 0; f < nFeatures; f++) {
        const d = (ci[f] ?? 0) - (cj[f] ?? 0);
        distSq += d * d;
      }

      const dist = Math.sqrt(distSq);
      const si = readIndex(S, i, "S");
      const sj = readIndex(S, j, "S");
      const ratio = dist === 0 ? Number.POSITIVE_INFINITY : (si + sj) / dist;
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
 * **Space Complexity**: O(k * d) where k is clusters
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
  const { data, nSamples, nFeatures, sampleStride, featureStride, offset } = getFeatureAccessor(X);
  if (nSamples === 0) return 0;

  if (labels.size !== nSamples) {
    throw new ShapeError("labels length must match number of samples");
  }

  const enc = encodeLabels(labels, "labels");
  const codes = enc.codes;
  const k = enc.nClusters;

  const overallMean = new Float64Array(nFeatures);

  for (let i = 0; i < nSamples; i++) {
    const base = offset + i * sampleStride;
    for (let f = 0; f < nFeatures; f++) {
      const v = getNumericElement(data, base + f * featureStride);
      assertFiniteNumber(v, "X", `sample ${i}, feature ${f}`);
      overallMean[f] = (overallMean[f] ?? 0) + v;
    }
  }
  for (let f = 0; f < nFeatures; f++) {
    overallMean[f] = (overallMean[f] ?? 0) / nSamples;
  }

  const centroids = new Array<Float64Array>(k);
  for (let c = 0; c < k; c++) centroids[c] = new Float64Array(nFeatures);
  const clusterSizes = new Int32Array(k);

  for (let i = 0; i < nSamples; i++) {
    const c = readIndex(codes, i, "codes");
    clusterSizes[c] = (clusterSizes[c] ?? 0) + 1;

    const base = offset + i * sampleStride;
    const centroid = readIndex(centroids, c, "centroids");
    for (let f = 0; f < nFeatures; f++) {
      const v = getNumericElement(data, base + f * featureStride);
      assertFiniteNumber(v, "X", `sample ${i}, feature ${f}`);
      centroid[f] = (centroid[f] ?? 0) + v;
    }
  }

  for (let c = 0; c < k; c++) {
    const sz = readIndex(clusterSizes, c, "clusterSizes");
    if (sz <= 0) continue;
    const centroid = readIndex(centroids, c, "centroids");
    for (let f = 0; f < nFeatures; f++) {
      centroid[f] = (centroid[f] ?? 0) / sz;
    }
  }

  let bgss = 0;
  for (let c = 0; c < k; c++) {
    const sz = readIndex(clusterSizes, c, "clusterSizes");
    if (sz <= 0) continue;

    const centroid = readIndex(centroids, c, "centroids");
    let distSq = 0;
    for (let f = 0; f < nFeatures; f++) {
      const d = (centroid[f] ?? 0) - (overallMean[f] ?? 0);
      distSq += d * d;
    }
    bgss += sz * distSq;
  }

  let wgss = 0;
  for (let i = 0; i < nSamples; i++) {
    const c = readIndex(codes, i, "codes");
    const centroid = readIndex(centroids, c, "centroids");

    const base = offset + i * sampleStride;
    for (let f = 0; f < nFeatures; f++) {
      const v = getNumericElement(data, base + f * featureStride);
      assertFiniteNumber(v, "X", `sample ${i}, feature ${f}`);
      const d = v - (centroid[f] ?? 0);
      wgss += d * d;
    }
  }

  if (k < 2 || wgss === 0) return 0;
  return bgss / (k - 1) / (wgss / (nSamples - k));
}

/**
 * Computes the Adjusted Rand Index (ARI).
 *
 * The ARI measures the similarity between two clusterings, adjusted for chance.
 * It ranges from -1 to 1, where 1 indicates perfect agreement, 0 indicates
 * random labeling, and negative values indicate worse than random.
 *
 * **Time Complexity**: O(n + k₁ * k₂) where k₁, k₂ are cluster counts
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth cluster labels
 * @param labelsPred - Predicted cluster labels
 * @returns Adjusted Rand Index in range [-1, 1]
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
 * @throws {DataValidationError} If labels contain non-finite or non-integer values
 *
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Clustering Metrics}
 */
export function adjustedRandScore(labelsTrue: Tensor, labelsPred: Tensor): number {
  assertSameSize(labelsTrue, labelsPred, "labelsTrue", "labelsPred");

  const n = labelsTrue.size;
  if (n <= 1) return 1;

  const stats = buildContingencyStats(labelsTrue, labelsPred);
  const { contingencyDense, contingencySparse, trueCount, predCount } = stats;

  let sumComb = 0;

  if (contingencyDense) {
    for (let idx = 0; idx < contingencyDense.length; idx++) {
      const nij = readIndex(contingencyDense, idx, "contingencyDense");
      if (nij > 0) sumComb += comb2(nij);
    }
  } else if (contingencySparse) {
    for (const nij of contingencySparse.values()) {
      sumComb += comb2(nij);
    }
  }

  let sumCombTrue = 0;
  for (let i = 0; i < trueCount.length; i++) {
    sumCombTrue += comb2(readIndex(trueCount, i, "trueCount"));
  }

  let sumCombPred = 0;
  for (let j = 0; j < predCount.length; j++) {
    sumCombPred += comb2(readIndex(predCount, j, "predCount"));
  }

  const totalPairs = comb2(n);
  if (totalPairs === 0) return 1;

  const expectedIndex = (sumCombTrue * sumCombPred) / totalPairs;
  const maxIndex = (sumCombTrue + sumCombPred) / 2;

  const denom = maxIndex - expectedIndex;
  if (denom === 0) return 1;

  return (sumComb - expectedIndex) / denom;
}

/**
 * Computes the Adjusted Mutual Information (AMI) between two clusterings.
 *
 * AMI adjusts the Mutual Information score to account for chance, providing
 * a normalized measure of agreement between two clusterings.
 *
 * **Time Complexity**: O(n + k₁ * k₂ * min(k₁, k₂))
 * **Space Complexity**: O(k₁ * k₂ + n)
 *
 * @param labelsTrue - Ground truth cluster labels
 * @param labelsPred - Predicted cluster labels
 * @param averageMethod - Method to compute the normalizer: 'min', 'geometric', 'arithmetic' (default), or 'max'
 * @returns Adjusted Mutual Information score, typically in range [0, 1]
 *
 * @throws {ShapeError} If labelsTrue and labelsPred have different sizes
 * @throws {DTypeError} If labels are string
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
  const { n, trueCount, predCount } = stats;
  if (n <= 1) return 1;

  const mi = mutualInformationFromContingency(stats);
  const hTrue = entropyFromCountArray(trueCount, n);
  const hPred = entropyFromCountArray(predCount, n);
  const emi = expectedMutualInformation(stats);

  const normalizer = averageEntropy(hTrue, hPred, averageMethod);
  if (Math.abs(normalizer) < 1e-15) return 1;

  const denom = normalizer - emi;
  if (Math.abs(denom) < 1e-15) return 0;

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
 * where 1 indicates perfect correlation between clusterings.
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

  const mi = mutualInformationFromContingency(stats);
  const ht = entropyFromCountArray(trueCount, n);
  const hp = entropyFromCountArray(predCount, n);

  if (ht === 0 || hp === 0) {
    return ht === 0 && hp === 0 ? 1.0 : 0.0;
  }

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
  const { contingencyDense, contingencySparse, trueCount, predCount, n } = stats;

  if (n === 0) return 1.0;

  let tk = 0;
  if (contingencyDense) {
    for (let idx = 0; idx < contingencyDense.length; idx++) {
      const nij = readIndex(contingencyDense, idx, "contingencyDense");
      if (nij > 0) tk += comb2(nij);
    }
  } else if (contingencySparse) {
    for (const nij of contingencySparse.values()) {
      tk += comb2(nij);
    }
  }

  let pk = 0;
  for (let i = 0; i < trueCount.length; i++) pk += comb2(readIndex(trueCount, i, "trueCount"));

  let qk = 0;
  for (let j = 0; j < predCount.length; j++) qk += comb2(readIndex(predCount, j, "predCount"));

  if (pk === 0 || qk === 0) return 0.0;
  return tk / Math.sqrt(pk * qk);
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
  const stats = buildContingencyStats(labelsTrue, labelsPred);
  const { contingencyDense, contingencySparse, predCount, trueCount, nPred, n } = stats;

  if (n === 0) return 1.0;

  let hck = 0;

  if (contingencyDense) {
    for (let idx = 0; idx < contingencyDense.length; idx++) {
      const nij = readIndex(contingencyDense, idx, "contingencyDense");
      if (nij <= 0) continue;

      const p = idx - Math.floor(idx / nPred) * nPred;
      const nj = readIndex(predCount, p, "predCount");
      if (nj > 0) hck -= (nij / n) * Math.log(nij / nj);
    }
  } else if (contingencySparse) {
    for (const [key, nij] of contingencySparse) {
      if (nij <= 0) continue;

      const p = key - Math.floor(key / nPred) * nPred;
      const nj = readIndex(predCount, p, "predCount");
      if (nj > 0) hck -= (nij / n) * Math.log(nij / nj);
    }
  }

  const hc = entropyFromCountArray(trueCount, n);
  return hc === 0 ? 1.0 : 1.0 - hck / hc;
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
  const stats = buildContingencyStats(labelsTrue, labelsPred);
  const { contingencyDense, contingencySparse, trueCount, predCount, nPred, n } = stats;

  if (n === 0) return 1.0;

  let hkc = 0;

  if (contingencyDense) {
    for (let idx = 0; idx < contingencyDense.length; idx++) {
      const nij = readIndex(contingencyDense, idx, "contingencyDense");
      if (nij <= 0) continue;

      const t = Math.floor(idx / nPred);
      const ni = readIndex(trueCount, t, "trueCount");
      if (ni > 0) hkc -= (nij / n) * Math.log(nij / ni);
    }
  } else if (contingencySparse) {
    for (const [key, nij] of contingencySparse) {
      if (nij <= 0) continue;

      const t = Math.floor(key / nPred);
      const ni = readIndex(trueCount, t, "trueCount");
      if (ni > 0) hkc -= (nij / n) * Math.log(nij / ni);
    }
  }

  const hk = entropyFromCountArray(predCount, n);
  return hk === 0 ? 1.0 : 1.0 - hkc / hk;
}

/**
 * Computes the V-measure score of a clustering.
 *
 * V-measure is the harmonic mean of homogeneity and completeness, controlled
 * by the beta parameter. When beta > 1, completeness is weighted more; when
 * beta < 1, homogeneity is weighted more.
 *
 * **Formula**: v = (1 + beta) * h * c / (beta * h + c)
 *
 * **Time Complexity**: O(n + k₁ * k₂)
 * **Space Complexity**: O(k₁ * k₂)
 *
 * @param labelsTrue - Ground truth class labels
 * @param labelsPred - Predicted cluster labels
 * @param beta - Weight of homogeneity vs completeness (default: 1.0)
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

  const h = homogeneityScore(labelsTrue, labelsPred);
  const c = completenessScore(labelsTrue, labelsPred);

  if (h + c === 0) return 0;
  return ((1 + beta) * h * c) / (beta * h + c);
}
