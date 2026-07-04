import { DeepboxError, InvalidParameterError } from "../core/errors";
import { type Tensor, Tensor as TensorClass, tensor } from "../ndarray";
import {
  assertBoolean,
  assertPositiveInt,
  createRng,
  type DatasetRng,
  normal01,
  normalizeOptionalSeed,
  shuffleRowsPairedInPlace,
} from "./utils";

function readAt<T>(arr: ArrayLike<T>, index: number, label: string): T {
  if (!Number.isInteger(index) || index < 0 || index >= arr.length) {
    throw new DeepboxError(`Internal error: ${label}[${index}] is out of bounds`);
  }
  const v = arr[index];
  if (v === undefined) {
    throw new DeepboxError(`Internal error: ${label}[${index}] is undefined`);
  }
  return v;
}

function readFiniteNumber(arr: ArrayLike<number>, index: number, label: string): number {
  const v = readAt(arr, index, label);
  if (!Number.isFinite(v)) {
    throw new InvalidParameterError("Center coordinates must be finite", "centers", v);
  }
  return v;
}

function getRequired<T>(arr: readonly T[], index: number, label: string): T {
  const v = arr[index];
  if (v === undefined) {
    throw new DeepboxError(`Internal error: ${label}[${index}] is undefined`);
  }
  return v;
}

/**
 * Add `std`-scaled standard-normal noise to `out[0..count)` in place, using a
 * single bulk Ziggurat draw. For generators whose geometry occupies the output
 * buffer already (so the noise can't be written there directly first).
 */
function addNoiseInPlace(out: Float64Array, count: number, std: number, rng: DatasetRng): void {
  const noise = new Float64Array(count);
  rng.fillNormal(noise, 0, count);
  for (let i = 0; i < count; i++) out[i] = (out[i] as number) + (noise[i] as number) * std;
}

/**
 * Generate a random n-class classification dataset.
 *
 * Produces informative features drawn from class-conditional Gaussians,
 * redundant features as random linear combinations of the informative ones,
 * and noise features sampled from N(0, 1).
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.nFeatures - Total number of features (default: 20).
 * @param options.nInformative - Number of informative features (default: 2).
 * @param options.nRedundant - Number of redundant features (default: 2).
 * @param options.nClasses - Number of classes (default: 2).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, nFeatures]` and y has shape `[nSamples]` with dtype `int32`.
 *
 * @see {@link https://deepbox.dev/docs/datasets-synthetic | Deepbox Synthetic Datasets}
 */
export function makeClassification(
  options: {
    nSamples?: number;
    nFeatures?: number;
    nInformative?: number;
    nRedundant?: number;
    nClasses?: number;
    flipY?: number;
    randomState?: number;
  } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 20;
  const nInformative = options.nInformative ?? 2;
  const nRedundant = options.nRedundant ?? 2;
  const nClasses = options.nClasses ?? 2;
  const flipY = options.flipY ?? 0.01;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);
  assertPositiveInt("nInformative", nInformative);
  assertPositiveInt("nClasses", nClasses);

  if (!Number.isInteger(nRedundant) || nRedundant < 0 || !Number.isSafeInteger(nRedundant)) {
    throw new InvalidParameterError(
      `nRedundant must be a non-negative safe integer; received ${nRedundant}`,
      "nRedundant",
      nRedundant
    );
  }
  if (nInformative > nFeatures) {
    throw new InvalidParameterError(
      `nInformative (${nInformative}) cannot exceed nFeatures (${nFeatures})`,
      "nInformative",
      nInformative
    );
  }
  if (nInformative + nRedundant > nFeatures) {
    throw new InvalidParameterError(
      `nInformative + nRedundant (${
        nInformative + nRedundant
      }) cannot exceed nFeatures (${nFeatures})`,
      "nRedundant",
      nRedundant
    );
  }

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  // classMeans[c][k]
  const classMeans = Array.from({ length: nClasses }, () => new Float64Array(nInformative));
  for (let c = 0; c < nClasses; c++) {
    const meanVec = getRequired(classMeans, c, "classMeans");
    for (let k = 0; k < nInformative; k++) {
      meanVec[k] = (rng() - 0.5) * 2 * nClasses;
    }
  }

  // weights[j][k] for redundant features
  const weights = Array.from({ length: nRedundant }, () => new Float64Array(nInformative));
  for (let j = 0; j < nRedundant; j++) {
    const w = getRequired(weights, j, "weights");
    for (let k = 0; k < nInformative; k++) {
      w[k] = rng() - 0.5;
    }
  }

  // Row-major flat buffer written directly (same rng/normal01 draw order as
  // the nested-array version, so seeded output is bit-identical) — skips the
  // array-of-arrays allocation and the nested tensor() flatten/validation,
  // which dominates at large nSamples×nFeatures.
  const XData = new Float64Array(nSamples * nFeatures);
  const yData = new Int32Array(nSamples);

  const nNoise = nFeatures - nInformative - nRedundant;
  const informative = new Float64Array(nInformative);

  for (let i = 0; i < nSamples; i++) {
    const label = Math.floor(rng() * nClasses);
    yData[i] = label;

    const meanVec = getRequired(classMeans, label, "classMeans");
    const rowOff = i * nFeatures;
    let col = 0;

    // informative
    for (let k = 0; k < nInformative; k++) {
      const mean = readAt(meanVec, k, "meanVec");
      const v = mean + normal01(rng);
      informative[k] = v;
      XData[rowOff + col++] = v;
    }

    // redundant
    for (let j = 0; j < nRedundant; j++) {
      const w = getRequired(weights, j, "weights");
      let val = 0;
      for (let k = 0; k < nInformative; k++) {
        val += (informative[k] as number) * readAt(w, k, "weights");
      }
      XData[rowOff + col++] = val + normal01(rng) * 0.01;
    }

    // noise — bulk Ziggurat fill (this block dominates wide feature counts)
    if (nNoise > 0) {
      rng.fillNormal(XData, rowOff + col, nNoise);
      col += nNoise;
    }
  }

  // Flip a fraction of labels
  if (flipY > 0) {
    for (let i = 0; i < nSamples; i++) {
      if (rng() < flipY) {
        yData[i] = Math.floor(rng() * nClasses);
      }
    }
  }

  return [
    TensorClass.fromTypedArray({
      data: XData,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({ data: yData, shape: [nSamples], dtype: "int32", device: "cpu" }),
  ];
}

/**
 * Generate a random regression dataset.
 *
 * Features are drawn from N(0, 1) and the target is a linear combination
 * of the features with optional Gaussian noise.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.nFeatures - Number of features (default: 100).
 * @param options.noise - Standard deviation of Gaussian noise on the target (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, nFeatures]` and y has shape `[nSamples]`.
 *
 * @see {@link https://deepbox.dev/docs/datasets-synthetic | Deepbox Synthetic Datasets}
 */
export function makeRegression(
  options: { nSamples?: number; nFeatures?: number; noise?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 100;
  const noiseStd = options.noise ?? 0.0;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);

  if (!Number.isFinite(noiseStd) || noiseStd < 0) {
    throw new InvalidParameterError(
      `noise must be a non-negative finite number; received ${noiseStd}`,
      "noise",
      noiseStd
    );
  }

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const weights = new Float64Array(nFeatures);
  for (let j = 0; j < nFeatures; j++) {
    weights[j] = normal01(rng);
  }

  const XData = new Float64Array(nSamples * nFeatures);
  const yData = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const rowOff = i * nFeatures;
    // Draw the row's features as one contiguous normal block (twin-caching
    // bulk sampler), then form the linear target from them.
    rng.fillNormal(XData, rowOff, nFeatures);
    let val = 0;
    for (let j = 0; j < nFeatures; j++) {
      val += (XData[rowOff + j] as number) * readAt(weights, j, "weights");
    }
    if (noiseStd > 0) val += normal01(rng) * noiseStd;
    yData[i] = val;
  }

  return [
    TensorClass.fromTypedArray({
      data: XData,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({ data: yData, shape: [nSamples], dtype: "float64", device: "cpu" }),
  ];
}

/**
 * Generate isotropic Gaussian blobs for clustering.
 *
 * Samples are drawn from Gaussian distributions centered at randomly generated
 * or user-specified locations. Useful for testing clustering algorithms.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Total number of samples (default: 100).
 * @param options.nFeatures - Number of features per sample (default: 2). Ignored when `centers` is an array.
 * @param options.centers - Number of cluster centers or explicit center coordinates (default: 3).
 * @param options.clusterStd - Standard deviation of each cluster (default: 1.0).
 * @param options.shuffle - Whether to shuffle the samples (default: true).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, nFeatures]` and y has shape `[nSamples]` with dtype `int32`.
 *
 * @see {@link https://deepbox.dev/docs/datasets-synthetic | Deepbox Synthetic Datasets}
 */
export function makeBlobs(
  options: {
    nSamples?: number;
    nFeatures?: number;
    centers?: number | number[][];
    clusterStd?: number;
    randomState?: number;
    shuffle?: boolean;
  } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const clusterStd = options.clusterStd ?? 1.0;
  const shuffle = options.shuffle ?? true;

  assertPositiveInt("nSamples", nSamples);
  if (!Number.isFinite(clusterStd) || clusterStd <= 0) {
    throw new InvalidParameterError(
      `clusterStd must be positive; received ${clusterStd}`,
      "clusterStd",
      clusterStd
    );
  }
  assertBoolean("shuffle", shuffle);

  const centersInput = options.centers === undefined ? 3 : options.centers;
  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  let nFeatures: number;
  let centerLocations: Float64Array[];

  if (typeof centersInput === "number") {
    assertPositiveInt("centers", centersInput);

    const nFeat = options.nFeatures ?? 2;
    assertPositiveInt("nFeatures", nFeat);
    nFeatures = nFeat;

    centerLocations = Array.from({ length: centersInput }, () => {
      const c = new Float64Array(nFeat);
      for (let j = 0; j < nFeat; j++) c[j] = (rng() - 0.5) * 20;
      return c;
    });
  } else if (Array.isArray(centersInput)) {
    if (centersInput.length === 0) {
      throw new InvalidParameterError("centers cannot be empty", "centers");
    }
    const firstCenter = centersInput[0];
    if (!Array.isArray(firstCenter) || firstCenter.length === 0) {
      throw new InvalidParameterError("centers must be a non-empty array of arrays", "centers");
    }

    nFeatures = options.nFeatures ?? firstCenter.length;
    assertPositiveInt("nFeatures", nFeatures);

    centerLocations = centersInput.map((c) => {
      if (!Array.isArray(c) || c.length !== nFeatures) {
        throw new InvalidParameterError(
          `Center dimension mismatch. Expected ${nFeatures}; received ${
            Array.isArray(c) ? c.length : 0
          }`,
          "centers",
          c
        );
      }
      const out = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        out[j] = readFiniteNumber(c, j, "centers");
      }
      return out;
    });
  } else {
    throw new InvalidParameterError("centers must be an int or an array of arrays", "centers");
  }

  const nCenters = centerLocations.length;
  const base = Math.floor(nSamples / nCenters);
  const remainder = nSamples % nCenters;

  // Write samples straight into row-major flat buffers (same normal01 draw
  // order as the nested-array version, so seeded output is bit-identical),
  // avoiding the array-of-arrays allocation and the nested-array tensor()
  // re-validation/flatten pass.
  const XData = new Float64Array(nSamples * nFeatures);
  const yData = new Int32Array(nSamples);

  // Bulk-fill all cluster noise (contiguous draw block), then add each
  // sample's center offset in a second pass. `normal01` is drawn per-cluster
  // block so the twin-caching bulk sampler stays deterministic per seed.
  let row = 0;
  for (let c = 0; c < nCenters; c++) {
    const nC = base + (c < remainder ? 1 : 0);
    const center = getRequired(centerLocations, c, "centerLocations");
    rng.fillNormal(XData, row * nFeatures, nC * nFeatures);
    for (let i = 0; i < nC; i++) {
      const off = row * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        XData[off + j] = readAt(center, j, "center") + (XData[off + j] as number) * clusterStd;
      }
      yData[row] = c;
      row++;
    }
  }

  if (shuffle) shuffleRowsPairedInPlace(XData, yData, nSamples, nFeatures, rng);

  return [
    TensorClass.fromTypedArray({
      data: XData,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({ data: yData, shape: [nSamples], dtype: "int32", device: "cpu" }),
  ];
}

/**
 * Generate two interleaving half-circle (moons) dataset.
 *
 * Useful for testing algorithms that handle non-linearly separable data.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Total number of samples, split evenly between the two moons (default: 100).
 * @param options.noise - Standard deviation of Gaussian noise (default: 0).
 * @param options.shuffle - Whether to shuffle the samples (default: true).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, 2]` and y has shape `[nSamples]` with dtype `int32`.
 *
 * @see {@link https://deepbox.dev/docs/datasets-synthetic | Deepbox Synthetic Datasets}
 */
export function makeMoons(
  options: { nSamples?: number; noise?: number; randomState?: number; shuffle?: boolean } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const noiseStd = options.noise ?? 0.0;
  const shuffle = options.shuffle ?? true;

  assertPositiveInt("nSamples", nSamples);
  if (!Number.isFinite(noiseStd) || noiseStd < 0) {
    throw new InvalidParameterError(
      `noise must be a non-negative finite number; received ${noiseStd}`,
      "noise",
      noiseStd
    );
  }
  assertBoolean("shuffle", shuffle);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const samplesFirst = Math.ceil(nSamples / 2);
  const samplesSecond = nSamples - samplesFirst;

  const XData = new Float64Array(nSamples * 2);
  const yData = new Int32Array(nSamples);

  // Bulk-fill the per-sample Gaussian noise (2 values per sample, in the same
  // sequential draw order the interleaved version used), then add it to the
  // manifold positions — the twin-caching sampler halves the transcendentals.
  if (noiseStd > 0) rng.fillNormal(XData, 0, nSamples * 2);
  for (let i = 0; i < samplesFirst; i++) {
    const angle = Math.PI * (i / samplesFirst);
    XData[i * 2] = Math.cos(angle) + (XData[i * 2] as number) * noiseStd;
    XData[i * 2 + 1] = Math.sin(angle) + (XData[i * 2 + 1] as number) * noiseStd;
    yData[i] = 0;
  }

  for (let i = 0; i < samplesSecond; i++) {
    const angle = Math.PI * (i / samplesSecond);
    const row = samplesFirst + i;
    XData[row * 2] = 1 - Math.cos(angle) + (XData[row * 2] as number) * noiseStd;
    XData[row * 2 + 1] = 0.5 - Math.sin(angle) + (XData[row * 2 + 1] as number) * noiseStd;
    yData[row] = 1;
  }

  if (shuffle) shuffleRowsPairedInPlace(XData, yData, nSamples, 2, rng);

  return [
    TensorClass.fromTypedArray({
      data: XData,
      shape: [nSamples, 2],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({ data: yData, shape: [nSamples], dtype: "int32", device: "cpu" }),
  ];
}

/**
 * Generate a large circle containing a smaller circle in 2D.
 *
 * Useful for testing algorithms that handle non-linearly separable data.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Total number of samples, split evenly between inner and outer circles (default: 100).
 * @param options.noise - Standard deviation of Gaussian noise (default: 0).
 * @param options.factor - Scale factor between inner and outer circle, must be in (0, 1) (default: 0.8).
 * @param options.shuffle - Whether to shuffle the samples (default: true).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, 2]` and y has shape `[nSamples]` with dtype `int32`.
 *
 * @see {@link https://deepbox.dev/docs/datasets-synthetic | Deepbox Synthetic Datasets}
 */
export function makeCircles(
  options: {
    nSamples?: number;
    noise?: number;
    factor?: number;
    randomState?: number;
    shuffle?: boolean;
  } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const noiseStd = options.noise ?? 0.0;
  const factor = options.factor ?? 0.8;
  const shuffle = options.shuffle ?? true;

  assertPositiveInt("nSamples", nSamples);
  if (!Number.isFinite(noiseStd) || noiseStd < 0) {
    throw new InvalidParameterError(
      `noise must be a non-negative finite number; received ${noiseStd}`,
      "noise",
      noiseStd
    );
  }
  if (!Number.isFinite(factor) || factor <= 0 || factor >= 1) {
    throw new InvalidParameterError(
      `factor must be in (0, 1); received ${factor}`,
      "factor",
      factor
    );
  }
  assertBoolean("shuffle", shuffle);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const samplesOuter = Math.ceil(nSamples / 2);
  const samplesInner = nSamples - samplesOuter;

  const XData = new Float64Array(nSamples * 2);
  const yData = new Int32Array(nSamples);

  // Bulk-fill per-sample noise (same sequential draw order), then add.
  if (noiseStd > 0) rng.fillNormal(XData, 0, nSamples * 2);
  for (let i = 0; i < samplesOuter; i++) {
    const angle = 2 * Math.PI * (i / samplesOuter);
    XData[i * 2] = Math.cos(angle) + (XData[i * 2] as number) * noiseStd;
    XData[i * 2 + 1] = Math.sin(angle) + (XData[i * 2 + 1] as number) * noiseStd;
    yData[i] = 0;
  }

  for (let i = 0; i < samplesInner; i++) {
    const angle = 2 * Math.PI * (i / samplesInner);
    const row = samplesOuter + i;
    XData[row * 2] = factor * Math.cos(angle) + (XData[row * 2] as number) * noiseStd;
    XData[row * 2 + 1] = factor * Math.sin(angle) + (XData[row * 2 + 1] as number) * noiseStd;
    yData[row] = 1;
  }

  if (shuffle) shuffleRowsPairedInPlace(XData, yData, nSamples, 2, rng);

  return [
    TensorClass.fromTypedArray({
      data: XData,
      shape: [nSamples, 2],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({ data: yData, shape: [nSamples], dtype: "int32", device: "cpu" }),
  ];
}

/**
 * Generate a dataset with classes separated by concentric Gaussian quantile shells.
 *
 * Samples are drawn from an isotropic Gaussian and assigned to classes based on
 * quantile boundaries of their Euclidean distance from the origin.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.nFeatures - Number of features (default: 2).
 * @param options.nClasses - Number of classes (default: 3).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, nFeatures]` and y has shape `[nSamples]` with dtype `int32`.
 *
 * @see {@link https://deepbox.dev/docs/datasets-synthetic | Deepbox Synthetic Datasets}
 */
export function makeGaussianQuantiles(
  options: { nSamples?: number; nFeatures?: number; nClasses?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 2;
  const nClasses = options.nClasses ?? 3;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);
  assertPositiveInt("nClasses", nClasses);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XData = new Float64Array(nSamples * nFeatures);
  const distances = new Float64Array(nSamples);

  // Bulk-fill all Gaussian coordinates (contiguous block), then row-reduce.
  rng.fillNormal(XData, 0, nSamples * nFeatures);
  for (let i = 0; i < nSamples; i++) {
    const base = i * nFeatures;
    let distSq = 0;
    for (let j = 0; j < nFeatures; j++) {
      const v = XData[base + j] as number;
      distSq += v * v;
    }
    distances[i] = Math.sqrt(distSq);
  }

  const sortedDistances = new Float64Array(distances);
  sortedDistances.sort();
  const quantileBoundaries: number[] = [];
  for (let c = 1; c < nClasses; c++) {
    const idx = Math.floor((c * nSamples) / nClasses);
    quantileBoundaries.push(sortedDistances[idx] as number);
  }

  const yData = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const dist = distances[i] as number;
    let label = 0;
    for (const boundary of quantileBoundaries) {
      if (dist > boundary) label++;
      else break;
    }
    yData[i] = label;
  }

  return [
    TensorClass.fromTypedArray({
      data: XData,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({ data: yData, shape: [nSamples], dtype: "int32", device: "cpu" }),
  ];
}

/**
 * Generate the Friedman #1 regression dataset.
 *
 * y = 10 * sin(pi * x0 * x1) + 20 * (x2 - 0.5)^2 + 10 * x3 + 5 * x4 + noise
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.nFeatures - Number of features, must be >= 5 (default: 10).
 * @param options.noise - Standard deviation of Gaussian noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]`.
 */
export function makeFriedman1(
  options: { nSamples?: number; nFeatures?: number; noise?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 10;
  const noiseStd = options.noise ?? 0.0;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);
  if (nFeatures < 5) {
    throw new InvalidParameterError(
      `nFeatures must be >= 5 for Friedman #1; received ${nFeatures}`,
      "nFeatures",
      nFeatures
    );
  }

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XFlat = new Float64Array(nSamples * nFeatures);
  const yFlat = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const base = i * nFeatures;
    for (let j = 0; j < nFeatures; j++) XFlat[base + j] = rng();
    const x0 = XFlat[base] as number;
    const x1 = XFlat[base + 1] as number;
    const x2 = XFlat[base + 2] as number;
    const x3 = XFlat[base + 3] as number;
    const x4 = XFlat[base + 4] as number;
    yFlat[i] =
      10 * Math.sin(Math.PI * x0 * x1) +
      20 * (x2 - 0.5) ** 2 +
      10 * x3 +
      5 * x4 +
      (noiseStd > 0 ? normal01(rng) * noiseStd : 0);
  }

  return [
    TensorClass.fromTypedArray({
      data: XFlat,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({ data: yFlat, shape: [nSamples], dtype: "float64", device: "cpu" }),
  ];
}

/**
 * Generate the Friedman #2 regression dataset.
 *
 * y = sqrt(x0^2 + (x1 * x2 - 1/(x1 * x3))^2) + noise
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.noise - Standard deviation of Gaussian noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]`.
 */
export function makeFriedman2(
  options: { nSamples?: number; noise?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const noiseStd = options.noise ?? 0.0;

  assertPositiveInt("nSamples", nSamples);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XFlat2 = new Float64Array(nSamples * 4);
  const yFlat2 = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const x0 = rng() * 100;
    const x1 = rng() * (560 * Math.PI) + 40 * Math.PI;
    const x2 = rng();
    const x3 = rng() * 10 + 1;
    const base = i * 4;
    XFlat2[base] = x0;
    XFlat2[base + 1] = x1;
    XFlat2[base + 2] = x2;
    XFlat2[base + 3] = x3;
    yFlat2[i] =
      Math.sqrt(x0 ** 2 + (x1 * x2 - 1 / (x1 * x3)) ** 2) +
      (noiseStd > 0 ? normal01(rng) * noiseStd : 0);
  }

  return [
    TensorClass.fromTypedArray({
      data: XFlat2,
      shape: [nSamples, 4],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({
      data: yFlat2,
      shape: [nSamples],
      dtype: "float64",
      device: "cpu",
    }),
  ];
}

/**
 * Generate the Friedman #3 regression dataset.
 *
 * y = atan((x1 * x2 - 1/(x1 * x3)) / x0) + noise
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.noise - Standard deviation of Gaussian noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]`.
 */
export function makeFriedman3(
  options: { nSamples?: number; noise?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const noiseStd = options.noise ?? 0.0;

  assertPositiveInt("nSamples", nSamples);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XFlat3 = new Float64Array(nSamples * 4);
  const yFlat3 = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const x0 = rng() * 100;
    const x1 = rng() * (560 * Math.PI) + 40 * Math.PI;
    const x2 = rng();
    const x3 = rng() * 10 + 1;
    const base = i * 4;
    XFlat3[base] = x0;
    XFlat3[base + 1] = x1;
    XFlat3[base + 2] = x2;
    XFlat3[base + 3] = x3;
    yFlat3[i] =
      Math.atan((x1 * x2 - 1 / (x1 * x3)) / x0) + (noiseStd > 0 ? normal01(rng) * noiseStd : 0);
  }

  return [
    TensorClass.fromTypedArray({
      data: XFlat3,
      shape: [nSamples, 4],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({
      data: yFlat3,
      shape: [nSamples],
      dtype: "float64",
      device: "cpu",
    }),
  ];
}

/**
 * Generate a Swiss Roll dataset (3D manifold).
 *
 * Useful for testing manifold learning and dimensionality reduction algorithms.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.noise - Standard deviation of Gaussian noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, t]` where X is 3D coordinates and t is the manifold position.
 */
export function makeSwissRoll(
  options: { nSamples?: number; noise?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const noiseStd = options.noise ?? 0.0;

  assertPositiveInt("nSamples", nSamples);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XSwiss = new Float64Array(nSamples * 3);
  const tSwiss = new Float64Array(nSamples);

  // Pass 1: geometry (uniform draws) into X; pass 2: bulk Ziggurat noise added
  // in place — one contiguous normal block instead of 3 Box-Muller draws/sample.
  for (let i = 0; i < nSamples; i++) {
    const t = 1.5 * Math.PI * (1 + 2 * rng());
    const height = rng() * 21;
    const base = i * 3;
    XSwiss[base] = t * Math.cos(t);
    XSwiss[base + 1] = height;
    XSwiss[base + 2] = t * Math.sin(t);
    tSwiss[i] = t;
  }
  if (noiseStd > 0) addNoiseInPlace(XSwiss, nSamples * 3, noiseStd, rng);

  return [
    TensorClass.fromTypedArray({
      data: XSwiss,
      shape: [nSamples, 3],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({
      data: tSwiss,
      shape: [nSamples],
      dtype: "float64",
      device: "cpu",
    }),
  ];
}

/**
 * Generate an S-curve dataset (3D manifold).
 *
 * Useful for testing manifold learning algorithms.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.noise - Standard deviation of Gaussian noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, t]` where X is 3D coordinates and t is the manifold position.
 */
export function makeSCurve(
  options: { nSamples?: number; noise?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const noiseStd = options.noise ?? 0.0;

  assertPositiveInt("nSamples", nSamples);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XSCurve = new Float64Array(nSamples * 3);
  const tSCurve = new Float64Array(nSamples);

  // Pass 1: geometry (uniform draws); pass 2: bulk Ziggurat noise in place.
  for (let i = 0; i < nSamples; i++) {
    const t = 3 * Math.PI * (rng() - 0.5);
    const base = i * 3;
    XSCurve[base] = Math.sin(t);
    XSCurve[base + 1] = 2 * rng();
    XSCurve[base + 2] = Math.sign(t) * (Math.cos(t) - 1);
    tSCurve[i] = t;
  }
  if (noiseStd > 0) addNoiseInPlace(XSCurve, nSamples * 3, noiseStd, rng);

  return [
    TensorClass.fromTypedArray({
      data: XSCurve,
      shape: [nSamples, 3],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({
      data: tSCurve,
      shape: [nSamples],
      dtype: "float64",
      device: "cpu",
    }),
  ];
}

/**
 * Generate a sparse uncorrelated regression dataset.
 *
 * Only the first 5 features are informative; the rest are noise.
 * y = x0 + 2*x1 + ... (first 5 features with fixed coefficients)
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.nFeatures - Number of features (default: 10).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]`.
 */
export function makeSparseUncorrelated(
  options: { nSamples?: number; nFeatures?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 10;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const coefs = [1, 2, 0, 0, 3];
  const XData = new Float64Array(nSamples * nFeatures);
  const yData = new Float64Array(nSamples);
  const nCoef = Math.min(coefs.length, nFeatures);

  for (let i = 0; i < nSamples; i++) {
    const base = i * nFeatures;
    rng.fillNormal(XData, base, nFeatures);
    let y = 0;
    for (let j = 0; j < nCoef; j++) y += (coefs[j] ?? 0) * (XData[base + j] as number);
    yData[i] = y + normal01(rng);
  }

  return [
    TensorClass.fromTypedArray({
      data: XData,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: "cpu",
    }),
    TensorClass.fromTypedArray({ data: yData, shape: [nSamples], dtype: "float64", device: "cpu" }),
  ];
}

/**
 * Generate a low-rank matrix with optional noise.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of rows (default: 100).
 * @param options.nFeatures - Number of columns (default: 10).
 * @param options.effectiveRank - Approximate rank of the matrix (default: 10).
 * @param options.randomState - Seed for reproducibility.
 * @returns A 2D Tensor of shape [nSamples, nFeatures].
 */
export function makeLowRankMatrix(
  options: {
    nSamples?: number;
    nFeatures?: number;
    effectiveRank?: number;
    randomState?: number;
  } = {}
): Tensor {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 10;
  const effectiveRank = options.effectiveRank ?? 10;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);
  assertPositiveInt("effectiveRank", effectiveRank);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const k = Math.min(nSamples, nFeatures, effectiveRank);

  // Generate U (nSamples x k) and V (k x nFeatures) with random values
  const U: number[][] = [];
  for (let i = 0; i < nSamples; i++) {
    const row: number[] = [];
    for (let j = 0; j < k; j++) row.push(normal01(rng));
    U.push(row);
  }

  const V: number[][] = [];
  for (let i = 0; i < k; i++) {
    const row: number[] = [];
    for (let j = 0; j < nFeatures; j++) row.push(normal01(rng));
    V.push(row);
  }

  // Singular values with exponential decay
  const s: number[] = [];
  for (let i = 0; i < k; i++) {
    s.push(Math.exp(-i / effectiveRank));
  }

  // Compute result = U * diag(s) * V
  const result: number[][] = [];
  for (let i = 0; i < nSamples; i++) {
    const row: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      let val = 0;
      for (let l = 0; l < k; l++) {
        val += U[i]![l]! * s[l]! * V[l]![j]!;
      }
      row.push(val);
    }
    result.push(row);
  }

  return tensor(result);
}

/**
 * Generate a random symmetric positive-definite matrix.
 *
 * @param options - Configuration options.
 * @param options.nDim - Dimension of the matrix (default: 5).
 * @param options.randomState - Seed for reproducibility.
 * @returns A 2D Tensor of shape [nDim, nDim].
 */
export function makeSPDMatrix(options: { nDim?: number; randomState?: number } = {}): Tensor {
  const nDim = options.nDim ?? 5;
  assertPositiveInt("nDim", nDim);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  // Generate random matrix A (row-major flat), then result = AᵀA (SPD).
  const A = new Float64Array(nDim * nDim);
  rng.fillNormal(A, 0, nDim * nDim);

  const result = new Float64Array(nDim * nDim);
  for (let i = 0; i < nDim; i++) {
    for (let j = 0; j < nDim; j++) {
      let val = 0;
      for (let k = 0; k < nDim; k++) {
        val += (A[k * nDim + i] as number) * (A[k * nDim + j] as number);
      }
      result[i * nDim + j] = val;
    }
  }

  return TensorClass.fromTypedArray({
    data: result,
    shape: [nDim, nDim],
    dtype: "float64",
    device: "cpu",
  });
}

/**
 * Generate a bicluster dataset.
 *
 * Creates a matrix with block-diagonal structure plus noise.
 *
 * @param options - Configuration options.
 * @param options.shape - Shape [nRows, nCols] (default: [300, 300]).
 * @param options.nClusters - Number of biclusters (default: 5).
 * @param options.noise - Standard deviation of noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, rows, cols]` where rows/cols indicate cluster membership.
 */
export function makeBiclusters(
  options: {
    shape?: [number, number];
    nClusters?: number;
    noise?: number;
    randomState?: number;
  } = {}
): [Tensor, Tensor, Tensor] {
  const shape = options.shape ?? [300, 300];
  const nClusters = options.nClusters ?? 5;
  const noiseStd = options.noise ?? 0;

  const [nRows, nCols] = shape;
  assertPositiveInt("nRows", nRows);
  assertPositiveInt("nCols", nCols);
  assertPositiveInt("nClusters", nClusters);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  // Assign rows and cols to clusters
  const rowLabels: number[] = new Array(nRows);
  const colLabels: number[] = new Array(nCols);
  for (let i = 0; i < nRows; i++) rowLabels[i] = Math.floor(rng() * nClusters);
  for (let i = 0; i < nCols; i++) colLabels[i] = Math.floor(rng() * nClusters);

  // Build matrix: value is 1 if row and col belong to same cluster, else 0.
  // Bulk-fill the whole noise field with the Ziggurat sampler (row-major, same
  // draw order as per-cell normal01), then add the block pattern.
  const X = new Float64Array(nRows * nCols);
  if (noiseStd > 0) rng.fillNormal(X, 0, nRows * nCols);
  for (let i = 0; i < nRows; i++) {
    const rl = rowLabels[i];
    const base = i * nCols;
    for (let j = 0; j < nCols; j++) {
      const v = rl === colLabels[j] ? 1 : 0;
      X[base + j] = v + (X[base + j] as number) * noiseStd;
    }
  }

  return [
    TensorClass.fromTypedArray({ data: X, shape: [nRows, nCols], dtype: "float64", device: "cpu" }),
    tensor(rowLabels, { dtype: "int32" }),
    tensor(colLabels, { dtype: "int32" }),
  ];
}

/**
 * Generate a checkerboard dataset.
 *
 * Creates a matrix with checkerboard block structure plus noise.
 *
 * @param options - Configuration options.
 * @param options.shape - Shape [nRows, nCols] (default: [300, 300]).
 * @param options.nClusters - Number of clusters per dimension (default: [5, 5]).
 * @param options.noise - Standard deviation of noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, rows, cols]` where rows/cols indicate cluster membership.
 */
export function makeCheckerboard(
  options: {
    shape?: [number, number];
    nClusters?: [number, number];
    noise?: number;
    randomState?: number;
  } = {}
): [Tensor, Tensor, Tensor] {
  const shape = options.shape ?? [300, 300];
  const nClusters = options.nClusters ?? [5, 5];
  const noiseStd = options.noise ?? 0;

  const [nRows, nCols] = shape;
  const [nRowClusters, nColClusters] = nClusters;
  assertPositiveInt("nRows", nRows);
  assertPositiveInt("nCols", nCols);
  assertPositiveInt("nRowClusters", nRowClusters);
  assertPositiveInt("nColClusters", nColClusters);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const rowSize = Math.ceil(nRows / nRowClusters);
  const colSize = Math.ceil(nCols / nColClusters);

  const rowLabels: number[] = new Array(nRows);
  const colLabels: number[] = new Array(nCols);
  for (let i = 0; i < nRows; i++)
    rowLabels[i] = Math.min(Math.floor(i / rowSize), nRowClusters - 1);
  for (let i = 0; i < nCols; i++)
    colLabels[i] = Math.min(Math.floor(i / colSize), nColClusters - 1);

  const X = new Float64Array(nRows * nCols);
  if (noiseStd > 0) rng.fillNormal(X, 0, nRows * nCols);
  for (let i = 0; i < nRows; i++) {
    const rl = rowLabels[i] as number;
    const base = i * nCols;
    for (let j = 0; j < nCols; j++) {
      const v = (rl + (colLabels[j] as number)) % 2 === 0 ? 1 : 0;
      X[base + j] = v + (X[base + j] as number) * noiseStd;
    }
  }

  return [
    TensorClass.fromTypedArray({ data: X, shape: [nRows, nCols], dtype: "float64", device: "cpu" }),
    tensor(rowLabels, { dtype: "int32" }),
    tensor(colLabels, { dtype: "int32" }),
  ];
}
