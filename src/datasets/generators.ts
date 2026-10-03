import { DeepboxError, InvalidParameterError } from "../core/errors";
import { type Tensor, Tensor as TensorClass, tensor } from "../ndarray";
import { __randomBelow } from "../random/random";
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

/** Validate a standard-deviation style option: finite and >= 0. */
function assertNonNegativeFinite(name: string, value: number): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError(
      `${name} must be a non-negative finite number; received ${value}`,
      name,
      value
    );
  }
}

/**
 * Build a `rows x cols` matrix (`cols <= rows`) with orthonormal columns from
 * Gaussian draws, using Gram-Schmidt with one re-orthogonalization pass.
 * The result is column-major: column `j` occupies `[j * rows, (j + 1) * rows)`.
 */
function randomOrthonormalColumns(rows: number, cols: number, rng: DatasetRng): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let j = 0; j < cols; j++) {
    const col = out.subarray(j * rows, (j + 1) * rows);
    for (let attempt = 0; ; attempt++) {
      rng.fillNormal(col, 0, rows);
      for (let pass = 0; pass < 2; pass++) {
        for (let k = 0; k < j; k++) {
          const prev = out.subarray(k * rows, (k + 1) * rows);
          let dot = 0;
          for (let i = 0; i < rows; i++) dot += (prev[i] as number) * (col[i] as number);
          for (let i = 0; i < rows; i++) col[i] = (col[i] as number) - dot * (prev[i] as number);
        }
      }
      let norm = 0;
      for (let i = 0; i < rows; i++) norm += (col[i] as number) * (col[i] as number);
      norm = Math.sqrt(norm);
      if (norm > 1e-8) {
        for (let i = 0; i < rows; i++) col[i] = (col[i] as number) / norm;
        break;
      }
      if (attempt >= 8) {
        throw new DeepboxError("Internal error: failed to draw an orthonormal basis");
      }
    }
  }
  return out;
}

/** Validate a `[rows, cols]` pair of positive integers and return it. */
function parseShapePair(name: string, value: unknown): [number, number] {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new InvalidParameterError(
      `${name} must be an array of two positive integers; received ${JSON.stringify(value)}`,
      name,
      value
    );
  }
  return [value[0] as number, value[1] as number];
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
 * @param options.flipY - Fraction of samples whose label is replaced by a uniformly random class, in `[0, 1]` (default: 0.01).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, nFeatures]` and y has shape `[nSamples]` with dtype `int32`.
 * @throws {@link InvalidParameterError} If a count is not a positive integer, if `nInformative + nRedundant > nFeatures`, or if `flipY` is outside `[0, 1]`.
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
  if (typeof flipY !== "number" || !(flipY >= 0 && flipY <= 1)) {
    throw new InvalidParameterError(
      `flipY must be a number in [0, 1]; received ${flipY}`,
      "flipY",
      flipY
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
  // the nested-array version, so seeded output is bit-identical), skips the
  // array-of-arrays allocation and the nested tensor() flatten/validation,
  // which dominates at large nSamples×nFeatures.
  const XData = new Float64Array(nSamples * nFeatures);
  const yData = new Int32Array(nSamples);

  const nNoise = nFeatures - nInformative - nRedundant;
  const informative = new Float64Array(nInformative);

  for (let i = 0; i < nSamples; i++) {
    const label = __randomBelow(rng, nClasses);
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

    // noise, bulk Ziggurat fill (this block dominates wide feature counts)
    if (nNoise > 0) {
      rng.fillNormal(XData, rowOff + col, nNoise);
      col += nNoise;
    }
  }

  // Flip a fraction of labels
  if (flipY > 0) {
    for (let i = 0; i < nSamples; i++) {
      if (rng() < flipY) {
        yData[i] = __randomBelow(rng, nClasses);
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
 * of the first `nInformative` features (coefficients drawn from N(0, 1)) plus
 * an optional bias and Gaussian noise.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.nFeatures - Number of features (default: 100).
 * @param options.nInformative - Number of features with a non-zero coefficient (default: `nFeatures`).
 * @param options.bias - Constant added to every target value (default: 0).
 * @param options.noise - Standard deviation of Gaussian noise on the target (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, nFeatures]` and y has shape `[nSamples]`.
 * @throws {@link InvalidParameterError} If a count is invalid, `nInformative > nFeatures`, or `bias`/`noise` is not finite.
 *
 * @see {@link https://deepbox.dev/docs/datasets-synthetic | Deepbox Synthetic Datasets}
 */
export function makeRegression(
  options: {
    nSamples?: number;
    nFeatures?: number;
    nInformative?: number;
    bias?: number;
    noise?: number;
    randomState?: number;
  } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 100;
  const nInformative = options.nInformative ?? nFeatures;
  const bias = options.bias ?? 0;
  const noiseStd = options.noise ?? 0.0;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);
  assertPositiveInt("nInformative", nInformative);
  if (nInformative > nFeatures) {
    throw new InvalidParameterError(
      `nInformative (${nInformative}) cannot exceed nFeatures (${nFeatures})`,
      "nInformative",
      nInformative
    );
  }
  if (typeof bias !== "number" || !Number.isFinite(bias)) {
    throw new InvalidParameterError(`bias must be a finite number; received ${bias}`, "bias", bias);
  }
  assertNonNegativeFinite("noise", noiseStd);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  // All nFeatures coefficients are drawn so the RNG stream does not depend on
  // nInformative; the uninformative ones are then zeroed.
  const weights = new Float64Array(nFeatures);
  for (let j = 0; j < nFeatures; j++) {
    weights[j] = normal01(rng);
  }
  for (let j = nInformative; j < nFeatures; j++) weights[j] = 0;

  const XData = new Float64Array(nSamples * nFeatures);
  const yData = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const rowOff = i * nFeatures;
    // Draw the row's features as one contiguous normal block (twin-caching
    // bulk sampler), then form the linear target from them.
    rng.fillNormal(XData, rowOff, nFeatures);
    let val = bias;
    for (let j = 0; j < nInformative; j++) {
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
 * @param options.nFeatures - Number of features per sample (default: 2). When `centers` is an array, defaults to the length of its first row and must match every row.
 * @param options.centers - Number of cluster centers (positions drawn uniformly from [-10, 10) per axis) or explicit center coordinates (default: 3).
 * @param options.clusterStd - Standard deviation of the clusters, either one positive value for all clusters or one value per center (default: 1.0).
 * @param options.shuffle - Whether to shuffle the samples (default: true).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]` where X has shape `[nSamples, nFeatures]` and y has shape `[nSamples]` with dtype `int32`. The first `nSamples % nCenters` clusters get one extra sample.
 * @throws {@link InvalidParameterError} If an option is invalid or the center dimensions disagree.
 *
 * @see {@link https://deepbox.dev/docs/datasets-synthetic | Deepbox Synthetic Datasets}
 */
export function makeBlobs(
  options: {
    nSamples?: number;
    nFeatures?: number;
    centers?: number | number[][];
    clusterStd?: number | readonly number[];
    randomState?: number;
    shuffle?: boolean;
  } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const clusterStd = options.clusterStd ?? 1.0;
  const shuffle = options.shuffle ?? true;

  assertPositiveInt("nSamples", nSamples);
  const assertStd = (v: unknown): void => {
    if (typeof v !== "number" || !Number.isFinite(v) || v <= 0) {
      throw new InvalidParameterError(
        `clusterStd must be positive and finite; received ${String(v)}`,
        "clusterStd",
        v
      );
    }
  };
  if (Array.isArray(clusterStd)) {
    for (const v of clusterStd) assertStd(v);
  } else {
    assertStd(clusterStd);
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
  if (Array.isArray(clusterStd) && clusterStd.length !== nCenters) {
    throw new InvalidParameterError(
      `clusterStd has ${clusterStd.length} entries but there are ${nCenters} centers`,
      "clusterStd",
      clusterStd
    );
  }
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
    const std = typeof clusterStd === "number" ? clusterStd : (clusterStd[c] as number);
    rng.fillNormal(XData, row * nFeatures, nC * nFeatures);
    for (let i = 0; i < nC; i++) {
      const off = row * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        XData[off + j] = readAt(center, j, "center") + (XData[off + j] as number) * std;
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
 * Class 0 is the upper half of the unit circle, class 1 the lower half of a
 * unit circle shifted by (1, 0.5). Both arcs include their endpoints, as in
 * scikit-learn. Useful for testing algorithms that handle non-linearly
 * separable data.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Total number of samples, split between the two moons (default: 100). For odd values class 1 gets the extra sample, as in scikit-learn.
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
  assertNonNegativeFinite("noise", noiseStd);
  assertBoolean("shuffle", shuffle);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const samplesFirst = Math.floor(nSamples / 2);
  const samplesSecond = nSamples - samplesFirst;

  const XData = new Float64Array(nSamples * 2);
  const yData = new Int32Array(nSamples);

  // Bulk-fill the per-sample Gaussian noise (2 values per sample, in the same
  // sequential draw order the interleaved version used), then add it to the
  // manifold positions, the twin-caching sampler halves the transcendentals.
  if (noiseStd > 0) rng.fillNormal(XData, 0, nSamples * 2);
  // Arc positions are evenly spaced including both endpoints (np.linspace(0, pi, n)).
  for (let i = 0; i < samplesFirst; i++) {
    const angle = samplesFirst > 1 ? Math.PI * (i / (samplesFirst - 1)) : 0;
    XData[i * 2] = Math.cos(angle) + (XData[i * 2] as number) * noiseStd;
    XData[i * 2 + 1] = Math.sin(angle) + (XData[i * 2 + 1] as number) * noiseStd;
    yData[i] = 0;
  }

  for (let i = 0; i < samplesSecond; i++) {
    const angle = samplesSecond > 1 ? Math.PI * (i / (samplesSecond - 1)) : 0;
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
 * @param options.nSamples - Total number of samples, split between the outer (class 0) and inner (class 1) circle (default: 100). For odd values class 1 gets the extra sample, as in scikit-learn.
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
  assertNonNegativeFinite("noise", noiseStd);
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

  const samplesOuter = Math.floor(nSamples / 2);
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
 * Samples are drawn from an isotropic Gaussian and ranked by their Euclidean
 * distance from the origin. Class `c` receives the samples in rank block `c`, so
 * class sizes differ by at most one (the last class takes the remainder, as in
 * scikit-learn). Samples are returned in draw order, not sorted by class.
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

  // Rank-based assignment: ties in distance cannot unbalance the classes.
  const order = new Uint32Array(nSamples);
  for (let i = 0; i < nSamples; i++) order[i] = i;
  order.sort((a, b) => (distances[a] as number) - (distances[b] as number) || a - b);

  const yData = new Int32Array(nSamples);
  const step = Math.floor(nSamples / nClasses);
  for (let rank = 0; rank < nSamples; rank++) {
    const label =
      step > 0
        ? Math.min(Math.floor(rank / step), nClasses - 1)
        : Math.floor((rank * nClasses) / nSamples);
    yData[order[rank] as number] = label;
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
 * Features are uniform on [0, 1); only the first five influence the target.
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
  assertNonNegativeFinite("noise", noiseStd);
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
 * Features: x0 ~ U[0, 100), x1 ~ U[40 pi, 560 pi), x2 ~ U[0, 1), x3 ~ U[1, 11).
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
  assertNonNegativeFinite("noise", noiseStd);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XFlat2 = new Float64Array(nSamples * 4);
  const yFlat2 = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const x0 = rng() * 100;
    const x1 = rng() * (520 * Math.PI) + 40 * Math.PI;
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
 * Features: x0 ~ U[0, 100), x1 ~ U[40 pi, 560 pi), x2 ~ U[0, 1), x3 ~ U[1, 11).
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
  assertNonNegativeFinite("noise", noiseStd);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XFlat3 = new Float64Array(nSamples * 4);
  const yFlat3 = new Float64Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const x0 = rng() * 100;
    const x1 = rng() * (520 * Math.PI) + 40 * Math.PI;
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
  assertNonNegativeFinite("noise", noiseStd);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const XSwiss = new Float64Array(nSamples * 3);
  const tSwiss = new Float64Array(nSamples);

  // Pass 1: geometry (uniform draws) into X; pass 2: bulk Ziggurat noise added
  // in place, one contiguous normal block instead of 3 Box-Muller draws/sample.
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
  assertNonNegativeFinite("noise", noiseStd);

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
 * Features are drawn from N(0, 1). Only the first four influence the target:
 * `y ~ N(x0 + 2*x1 - 2*x2 - 1.5*x3, 1)`, the same model as scikit-learn's
 * `make_sparse_uncorrelated`. The remaining features are pure noise.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of samples (default: 100).
 * @param options.nFeatures - Number of features, must be at least 4 (default: 10).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, y]`.
 * @throws {@link InvalidParameterError} If `nFeatures < 4` or a count is not a positive integer.
 */
export function makeSparseUncorrelated(
  options: { nSamples?: number; nFeatures?: number; randomState?: number } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 10;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);
  if (nFeatures < 4) {
    throw new InvalidParameterError(
      `nFeatures must be >= 4 for makeSparseUncorrelated; received ${nFeatures}`,
      "nFeatures",
      nFeatures
    );
  }

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const coefs = [1, 2, -2, -1.5];
  const XData = new Float64Array(nSamples * nFeatures);
  const yData = new Float64Array(nSamples);
  const nCoef = coefs.length;

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
 * Generate a mostly low-rank matrix with a bell-shaped singular value profile.
 *
 * The matrix is `U diag(s) V^T` where `U` and `V` have random orthonormal
 * columns, so its singular values are exactly
 * `s_i = (1 - tailStrength) * exp(-(i / effectiveRank)^2) + tailStrength * exp(-0.1 * i / effectiveRank)`
 * for `i = 0 .. min(nSamples, nFeatures) - 1` (the model used by scikit-learn's
 * `make_low_rank_matrix`). Most of the spectrum is concentrated in the first
 * `effectiveRank` values; the tail decays slowly.
 *
 * @param options - Configuration options.
 * @param options.nSamples - Number of rows (default: 100).
 * @param options.nFeatures - Number of columns (default: 10).
 * @param options.effectiveRank - Approximate rank of the matrix (default: 10).
 * @param options.tailStrength - Relative weight of the slowly decaying tail, in `[0, 1]` (default: 0.5).
 * @param options.randomState - Seed for reproducibility.
 * @returns A 2D Tensor of shape [nSamples, nFeatures].
 * @throws {@link InvalidParameterError} If a count is not a positive integer or `tailStrength` is outside `[0, 1]`.
 */
export function makeLowRankMatrix(
  options: {
    nSamples?: number;
    nFeatures?: number;
    effectiveRank?: number;
    tailStrength?: number;
    randomState?: number;
  } = {}
): Tensor {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 10;
  const effectiveRank = options.effectiveRank ?? 10;
  const tailStrength = options.tailStrength ?? 0.5;

  assertPositiveInt("nSamples", nSamples);
  assertPositiveInt("nFeatures", nFeatures);
  assertPositiveInt("effectiveRank", effectiveRank);
  if (typeof tailStrength !== "number" || !(tailStrength >= 0 && tailStrength <= 1)) {
    throw new InvalidParameterError(
      `tailStrength must be a number in [0, 1]; received ${tailStrength}`,
      "tailStrength",
      tailStrength
    );
  }

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  const k = Math.min(nSamples, nFeatures);
  const U = randomOrthonormalColumns(nSamples, k, rng); // column-major, nSamples x k
  const V = randomOrthonormalColumns(nFeatures, k, rng); // column-major, nFeatures x k

  const s = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    const r = i / effectiveRank;
    s[i] = (1 - tailStrength) * Math.exp(-(r * r)) + tailStrength * Math.exp(-0.1 * r);
  }

  // result = U * diag(s) * V^T, accumulated one rank-1 term at a time so the
  // inner loop runs over contiguous memory.
  const result = new Float64Array(nSamples * nFeatures);
  for (let i = 0; i < nSamples; i++) {
    const rowOff = i * nFeatures;
    for (let l = 0; l < k; l++) {
      const coeff = (U[l * nSamples + i] as number) * (s[l] as number);
      const vOff = l * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        result[rowOff + j] = (result[rowOff + j] as number) + coeff * (V[vOff + j] as number);
      }
    }
  }

  return TensorClass.fromTypedArray({
    data: result,
    shape: [nSamples, nFeatures],
    dtype: "float64",
    device: "cpu",
  });
}

/**
 * Generate a random symmetric positive-definite matrix.
 *
 * The matrix is `Q diag(lambda) Q^T` with `Q` a random orthogonal matrix and
 * eigenvalues `lambda_i` uniform on `[1, 2)`, so it is exactly symmetric,
 * positive definite and well conditioned (condition number below 2), as in
 * scikit-learn's `make_spd_matrix`.
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

  const Q = randomOrthonormalColumns(nDim, nDim, rng);
  const lambda = new Float64Array(nDim);
  for (let i = 0; i < nDim; i++) lambda[i] = 1 + rng();

  // Fill the upper triangle and mirror it so the result is exactly symmetric.
  const result = new Float64Array(nDim * nDim);
  for (let i = 0; i < nDim; i++) {
    for (let j = i; j < nDim; j++) {
      let val = 0;
      for (let k = 0; k < nDim; k++) {
        val += (Q[k * nDim + i] as number) * (lambda[k] as number) * (Q[k * nDim + j] as number);
      }
      result[i * nDim + j] = val;
      result[j * nDim + i] = val;
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
 * Every row and every column is assigned independently and uniformly to one of
 * `nClusters` clusters (so for small shapes a cluster can end up empty). Entry
 * `(i, j)` is 1 when row `i` and column `j` share a cluster and 0 otherwise,
 * plus Gaussian noise.
 *
 * @param options - Configuration options.
 * @param options.shape - Shape [nRows, nCols] (default: [300, 300]).
 * @param options.nClusters - Number of biclusters (default: 5).
 * @param options.noise - Standard deviation of noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, rows, cols]` where `rows` (length nRows) and `cols` (length nCols) are int32 cluster labels.
 * @throws {@link InvalidParameterError} If `shape` is not two positive integers, `nClusters` is not a positive integer, or `noise` is negative or not finite.
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

  const [nRows, nCols] = parseShapePair("shape", shape);
  assertPositiveInt("nRows", nRows);
  assertPositiveInt("nCols", nCols);
  assertPositiveInt("nClusters", nClusters);
  assertNonNegativeFinite("noise", noiseStd);

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  // Assign rows and cols to clusters
  const rowLabels: number[] = new Array(nRows);
  const colLabels: number[] = new Array(nCols);
  for (let i = 0; i < nRows; i++) rowLabels[i] = __randomBelow(rng, nClusters);
  for (let i = 0; i < nCols; i++) colLabels[i] = __randomBelow(rng, nClusters);

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
 * Rows and columns are split into contiguous, near-equal blocks. Entry
 * `(i, j)` is 1 when the parity of its row block index plus its column block
 * index is even and 0 otherwise, plus Gaussian noise.
 *
 * @param options - Configuration options.
 * @param options.shape - Shape [nRows, nCols] (default: [300, 300]).
 * @param options.nClusters - Number of row and column clusters, `[nRowClusters, nColClusters]` (default: [5, 5]).
 * @param options.noise - Standard deviation of noise (default: 0).
 * @param options.randomState - Seed for reproducibility.
 * @returns A tuple `[X, rows, cols]` where `rows` (length nRows) and `cols` (length nCols) are int32 block labels.
 * @throws {@link InvalidParameterError} If `shape` or `nClusters` is not two positive integers, if a cluster count exceeds the matching dimension, or if `noise` is negative or not finite.
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

  const [nRows, nCols] = parseShapePair("shape", shape);
  const [nRowClusters, nColClusters] = parseShapePair("nClusters", nClusters);
  assertPositiveInt("nRows", nRows);
  assertPositiveInt("nCols", nCols);
  assertPositiveInt("nRowClusters", nRowClusters);
  assertPositiveInt("nColClusters", nColClusters);
  assertNonNegativeFinite("noise", noiseStd);
  if (nRowClusters > nRows || nColClusters > nCols) {
    throw new InvalidParameterError(
      `nClusters [${nRowClusters}, ${nColClusters}] cannot exceed shape [${nRows}, ${nCols}]`,
      "nClusters",
      nClusters
    );
  }

  const seed = normalizeOptionalSeed("randomState", options.randomState);
  const rng = createRng(seed);

  // Near-equal contiguous blocks; every cluster is non-empty because the
  // cluster count never exceeds the dimension.
  const rowLabels: number[] = new Array(nRows);
  const colLabels: number[] = new Array(nCols);
  for (let i = 0; i < nRows; i++) rowLabels[i] = Math.floor((i * nRowClusters) / nRows);
  for (let i = 0; i < nCols; i++) colLabels[i] = Math.floor((i * nColClusters) / nCols);

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
