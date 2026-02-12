import { DeepboxError, InvalidParameterError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";
import {
  assertBoolean,
  assertPositiveInt,
  createRng,
  normal01,
  normalizeOptionalSeed,
  shufflePairedInPlace,
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
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html | sklearn.datasets.make_classification}
 */
export function makeClassification(
  options: {
    nSamples?: number;
    nFeatures?: number;
    nInformative?: number;
    nRedundant?: number;
    nClasses?: number;
    randomState?: number;
  } = {}
): [Tensor, Tensor] {
  const nSamples = options.nSamples ?? 100;
  const nFeatures = options.nFeatures ?? 20;
  const nInformative = options.nInformative ?? 2;
  const nRedundant = options.nRedundant ?? 2;
  const nClasses = options.nClasses ?? 2;

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

  const XData: number[][] = new Array(nSamples);
  const yData: number[] = new Array(nSamples);

  const nNoise = nFeatures - nInformative - nRedundant;

  for (let i = 0; i < nSamples; i++) {
    const label = Math.floor(rng() * nClasses);
    yData[i] = label;

    const meanVec = getRequired(classMeans, label, "classMeans");
    const row: number[] = [];

    // informative
    const informative = new Float64Array(nInformative);
    for (let k = 0; k < nInformative; k++) {
      const mean = readAt(meanVec, k, "meanVec");
      const v = mean + normal01(rng);
      informative[k] = v;
      row.push(v);
    }

    // redundant
    for (let j = 0; j < nRedundant; j++) {
      const w = getRequired(weights, j, "weights");
      let val = 0;
      for (let k = 0; k < nInformative; k++) {
        val += readAt(informative, k, "informative") * readAt(w, k, "weights");
      }
      row.push(val + normal01(rng) * 0.01);
    }

    // noise
    for (let j = 0; j < nNoise; j++) {
      row.push(normal01(rng));
    }

    XData[i] = row;
  }

  return [tensor(XData), tensor(yData, { dtype: "int32" })];
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
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html | sklearn.datasets.make_regression}
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

  const XData: number[][] = new Array(nSamples);
  const yData: number[] = new Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const row: number[] = new Array(nFeatures);
    let val = 0;

    for (let j = 0; j < nFeatures; j++) {
      const x = normal01(rng);
      row[j] = x;
      val += x * readAt(weights, j, "weights");
    }

    if (noiseStd > 0) val += normal01(rng) * noiseStd;

    XData[i] = row;
    yData[i] = val;
  }

  return [tensor(XData), tensor(yData)];
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
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html | sklearn.datasets.make_blobs}
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

  const XData: number[][] = [];
  const yData: number[] = [];

  for (let c = 0; c < nCenters; c++) {
    const nC = base + (c < remainder ? 1 : 0);
    const center = getRequired(centerLocations, c, "centerLocations");

    for (let i = 0; i < nC; i++) {
      const row: number[] = new Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        row[j] = readAt(center, j, "center") + normal01(rng) * clusterStd;
      }
      XData.push(row);
      yData.push(c);
    }
  }

  if (shuffle) shufflePairedInPlace(XData, yData, rng);

  return [tensor(XData), tensor(yData, { dtype: "int32" })];
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
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html | sklearn.datasets.make_moons}
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

  const XData: number[][] = [];
  const yData: number[] = [];

  for (let i = 0; i < samplesFirst; i++) {
    const angle = Math.PI * (i / samplesFirst);
    const x = Math.cos(angle) + normal01(rng) * noiseStd;
    const y = Math.sin(angle) + normal01(rng) * noiseStd;
    XData.push([x, y]);
    yData.push(0);
  }

  for (let i = 0; i < samplesSecond; i++) {
    const angle = Math.PI * (i / samplesSecond);
    const x = 1 - Math.cos(angle) + normal01(rng) * noiseStd;
    const y = 0.5 - Math.sin(angle) + normal01(rng) * noiseStd;
    XData.push([x, y]);
    yData.push(1);
  }

  if (shuffle) shufflePairedInPlace(XData, yData, rng);

  return [tensor(XData), tensor(yData, { dtype: "int32" })];
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
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html | sklearn.datasets.make_circles}
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

  const XData: number[][] = [];
  const yData: number[] = [];

  for (let i = 0; i < samplesOuter; i++) {
    const angle = 2 * Math.PI * (i / samplesOuter);
    const x = Math.cos(angle) + normal01(rng) * noiseStd;
    const y = Math.sin(angle) + normal01(rng) * noiseStd;
    XData.push([x, y]);
    yData.push(0);
  }

  for (let i = 0; i < samplesInner; i++) {
    const angle = 2 * Math.PI * (i / samplesInner);
    const x = factor * Math.cos(angle) + normal01(rng) * noiseStd;
    const y = factor * Math.sin(angle) + normal01(rng) * noiseStd;
    XData.push([x, y]);
    yData.push(1);
  }

  if (shuffle) shufflePairedInPlace(XData, yData, rng);

  return [tensor(XData), tensor(yData, { dtype: "int32" })];
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
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html | sklearn.datasets.make_gaussian_quantiles}
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

  const XData: number[][] = new Array(nSamples);
  const distances: number[] = new Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const row: number[] = new Array(nFeatures);
    let distSq = 0;

    for (let j = 0; j < nFeatures; j++) {
      const v = normal01(rng);
      row[j] = v;
      distSq += v * v;
    }

    XData[i] = row;
    distances[i] = Math.sqrt(distSq);
  }

  const sortedDistances = [...distances].sort((a, b) => a - b);
  const quantileBoundaries: number[] = [];

  for (let c = 1; c < nClasses; c++) {
    const idx = Math.floor((c * nSamples) / nClasses);
    quantileBoundaries.push(getRequired(sortedDistances, idx, "sortedDistances"));
  }

  const yData: number[] = new Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const dist = getRequired(distances, i, "distances");
    let label = 0;
    for (const boundary of quantileBoundaries) {
      if (dist > boundary) label++;
      else break;
    }
    yData[i] = label;
  }

  return [tensor(XData), tensor(yData, { dtype: "int32" })];
}
