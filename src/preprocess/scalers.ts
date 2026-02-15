import {
  DataValidationError,
  DeepboxError,
  DTypeError,
  InvalidParameterError,
  NotFittedError,
} from "../core/errors";
import { type Tensor, tensor, zeros } from "../ndarray";
import {
  assert2D,
  assertNumericTensor,
  createSeededRandom,
  getShape2D,
  getStride1D,
  getStrides2D,
  shuffleIndicesInPlace,
} from "./_internal";

function getNumericData(X: Tensor, name: string): ArrayLike<number | bigint> {
  if (X.dtype === "string") {
    throw new DTypeError(`${name} must be numeric`);
  }
  if (Array.isArray(X.data)) {
    throw new DeepboxError("Internal error: invalid numeric tensor storage");
  }
  return X.data;
}

function parseBooleanOption(value: unknown, name: string, defaultValue: boolean): boolean {
  if (value === undefined) {
    return defaultValue;
  }
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(`${name} must be a boolean`, name, value);
  }
  return value;
}

function validateFiniteData(X: Tensor, name: string): void {
  const [nSamples, nFeatures] = getShape2D(X);
  const data = getNumericData(X, name);
  const [stride0, stride1] = getStrides2D(X);
  let flatIndex = 0;

  for (let i = 0; i < nSamples; i++) {
    const rowBase = X.offset + i * stride0;
    for (let j = 0; j < nFeatures; j++) {
      const raw = data[rowBase + j * stride1];
      if (raw === undefined) {
        throw new DeepboxError("Internal error: numeric tensor access out of bounds");
      }
      const val = Number(raw);
      if (!Number.isFinite(val)) {
        throw new DataValidationError(`${name} contains NaN or Infinity at index ${flatIndex}`);
      }
      flatIndex += 1;
    }
  }
}

function snapInverseValue(value: number): number {
  if (!Number.isFinite(value)) return value;
  const rounded = Math.round(value);
  if (Math.abs(value - rounded) < 1e-12) return rounded;
  const scaled = Math.round(value * 1e12) / 1e12;
  if (Math.abs(value - scaled) < 1e-12) return scaled;
  return value;
}

function normalQuantile(p: number): number {
  if (!Number.isFinite(p) || p <= 0 || p >= 1) {
    throw new InvalidParameterError(
      "normalQuantile requires p in the open interval (0, 1)",
      "p",
      p
    );
  }
  // Acklam's inverse normal CDF approximation.
  const a1 = -3.969683028665376e1;
  const a2 = 2.209460984245205e2;
  const a3 = -2.759285104469687e2;
  const a4 = 1.38357751867269e2;
  const a5 = -3.066479806614716e1;
  const a6 = 2.506628277459239;

  const b1 = -5.447609879822406e1;
  const b2 = 1.615858368580409e2;
  const b3 = -1.556989798598866e2;
  const b4 = 6.680131188771972e1;
  const b5 = -1.328068155288572e1;

  const c1 = -7.784894002430293e-3;
  const c2 = -3.223964580411365e-1;
  const c3 = -2.400758277161838;
  const c4 = -2.549732539343734;
  const c5 = 4.374664141464968;
  const c6 = 2.938163982698783;

  const d1 = 7.784695709041462e-3;
  const d2 = 3.224671290700398e-1;
  const d3 = 2.445134137142996;
  const d4 = 3.754408661907416;

  const plow = 0.02425;
  const phigh = 1 - plow;

  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    );
  }
  if (p > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    );
  }

  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q) /
    (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
  );
}

/**
 * Standardize features by removing mean and scaling to unit variance.
 *
 * **Formula**: z = (x - μ) / σ
 *
 * **Attributes** (after fitting):
 * - `mean_`: Mean of each feature
 * - `scale_`: Standard deviation of each feature
 *
 * @example
 * ```js
 * import { StandardScaler } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6]]);
 * const scaler = new StandardScaler();
 * scaler.fit(X);
 * const XScaled = scaler.transform(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class StandardScaler {
  private fitted = false;
  private mean_: Tensor | undefined;
  private scale_: Tensor | undefined;
  private withMean: boolean;
  private withStd: boolean;

  /**
   * Creates a new StandardScaler.
   *
   * @param options - Configuration options
   * @param options.withMean - Center data before scaling (default: true)
   * @param options.withStd - Scale data to unit variance (default: true)
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   */
  constructor(options: { withMean?: boolean; withStd?: boolean; copy?: boolean } = {}) {
    this.withMean = parseBooleanOption(options.withMean, "withMean", true);
    this.withStd = parseBooleanOption(options.withStd, "withStd", true);
    parseBooleanOption(options.copy, "copy", true);
  }

  fit(X: Tensor): this {
    if (X.size === 0) {
      throw new InvalidParameterError("X must contain at least one sample", "X");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    let means: number[] | undefined;
    if (this.withMean || this.withStd) {
      means = new Array<number>(nFeatures).fill(0);
      for (let j = 0; j < nFeatures; j++) {
        let sum = 0;
        for (let i = 0; i < nSamples; i++) {
          const raw = data[X.offset + i * stride0 + j * stride1];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: numeric tensor access out of bounds");
          }
          sum += Number(raw);
        }
        if (means) {
          means[j] = sum / nSamples;
        }
      }
    }

    if (this.withStd) {
      const stds = new Array<number>(nFeatures).fill(0);
      for (let j = 0; j < nFeatures; j++) {
        const mean = means ? (means[j] ?? 0) : 0;
        let sumSq = 0;
        for (let i = 0; i < nSamples; i++) {
          const raw = data[X.offset + i * stride0 + j * stride1];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: numeric tensor access out of bounds");
          }
          const val = Number(raw) - mean;
          sumSq += val * val;
        }
        stds[j] = Math.sqrt(sumSq / nSamples);
      }
      this.scale_ = tensor(stds, { dtype: "float64" });
    } else {
      this.scale_ = undefined;
    }

    this.mean_ = this.withMean && means ? tensor(means, { dtype: "float64" }) : undefined;

    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StandardScaler must be fitted before transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);
    const mean = this.mean_;
    const scale = this.scale_;
    const meanData = mean ? getNumericData(mean, "mean_") : undefined;
    const scaleData = scale ? getNumericData(scale, "scale_") : undefined;
    const meanStride = mean ? getStride1D(mean) : 0;
    const scaleStride = scale ? getStride1D(scale) : 0;

    if (this.withMean && !mean) {
      throw new DeepboxError("StandardScaler internal error: missing mean_");
    }
    if (this.withStd && !scale) {
      throw new DeepboxError("StandardScaler internal error: missing scale_");
    }

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        let val = Number(raw);

        if (this.withMean && mean && meanData) {
          const meanValue = meanData[mean.offset + j * meanStride];
          if (meanValue === undefined) {
            throw new DeepboxError("Internal error: mean tensor access out of bounds");
          }
          val -= Number(meanValue);
        }

        if (this.withStd && scale && scaleData) {
          const rawScale = scaleData[scale.offset + j * scaleStride];
          if (rawScale === undefined) {
            throw new DeepboxError("Internal error: scale tensor access out of bounds");
          }
          const std = Number(rawScale);
          const safeStd = std === 0 ? 1 : std;
          val /= safeStd;
        }

        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[j] = val;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StandardScaler must be fitted before inverse_transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);
    const mean = this.mean_;
    const scale = this.scale_;
    const meanData = mean ? getNumericData(mean, "mean_") : undefined;
    const scaleData = scale ? getNumericData(scale, "scale_") : undefined;
    const meanStride = mean ? getStride1D(mean) : 0;
    const scaleStride = scale ? getStride1D(scale) : 0;

    if (this.withMean && !mean) {
      throw new DeepboxError("StandardScaler internal error: missing mean_");
    }
    if (this.withStd && !scale) {
      throw new DeepboxError("StandardScaler internal error: missing scale_");
    }

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        let val = Number(raw);

        if (this.withStd && scale && scaleData) {
          const rawScale = scaleData[scale.offset + j * scaleStride];
          if (rawScale === undefined) {
            throw new DeepboxError("Internal error: scale tensor access out of bounds");
          }
          const std = Number(rawScale);
          const safeStd = std === 0 ? 1 : std;
          val *= safeStd;
        }

        if (this.withMean && mean && meanData) {
          const meanValue = meanData[mean.offset + j * meanStride];
          if (meanValue === undefined) {
            throw new DeepboxError("Internal error: mean tensor access out of bounds");
          }
          val += Number(meanValue);
        }

        const resultRow = result[i];
        if (resultRow === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        resultRow[j] = snapInverseValue(val);
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }
}

/**
 * Scale features to a range [min, max].
 *
 * **Formula**: X_scaled = (X - X.min) / (X.max - X.min) * (max - min) + min
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class MinMaxScaler {
  private fitted = false;
  private dataMin_?: Tensor;
  private dataMax_?: Tensor;
  private featureRange: [number, number];
  private clip: boolean;

  /**
   * Creates a new MinMaxScaler.
   *
   * @param options - Configuration options
   * @param options.featureRange - Desired feature range [min, max] (default: [0, 1])
   * @param options.clip - Clip transformed values to featureRange (default: false)
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   */
  constructor(
    options: {
      featureRange?: [number, number];
      clip?: boolean;
      copy?: boolean;
    } = {}
  ) {
    this.featureRange = options.featureRange ?? [0, 1];
    this.clip = parseBooleanOption(options.clip, "clip", false);
    parseBooleanOption(options.copy, "copy", true);
    const [minRange, maxRange] = this.featureRange;
    if (!Number.isFinite(minRange) || !Number.isFinite(maxRange) || minRange >= maxRange) {
      throw new InvalidParameterError(
        "featureRange must be [min, max] with min < max",
        "featureRange",
        this.featureRange
      );
    }
  }

  fit(X: Tensor): this {
    if (X.size === 0) {
      throw new InvalidParameterError("X must contain at least one sample", "X");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    const mins = new Array<number>(nFeatures).fill(Number.POSITIVE_INFINITY);
    const maxs = new Array<number>(nFeatures).fill(Number.NEGATIVE_INFINITY);

    for (let j = 0; j < nFeatures; j++) {
      for (let i = 0; i < nSamples; i++) {
        const raw = data[X.offset + i * stride0 + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        const currentMin = mins[j];
        const currentMax = maxs[j];
        if (currentMin === undefined || currentMax === undefined) {
          throw new DeepboxError("Internal error: min/max array access failed");
        }
        mins[j] = Math.min(currentMin, val);
        maxs[j] = Math.max(currentMax, val);
      }
    }

    this.dataMin_ = tensor(mins, { dtype: "float64" });
    this.dataMax_ = tensor(maxs, { dtype: "float64" });
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MinMaxScaler must be fitted before transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const [minRange, maxRange] = this.featureRange;
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);
    const dataMin = this.dataMin_;
    const dataMax = this.dataMax_;

    if (!dataMin || !dataMax) {
      throw new DeepboxError("MinMaxScaler internal error: missing fitted min/max");
    }
    const minData = getNumericData(dataMin, "dataMin_");
    const maxData = getNumericData(dataMax, "dataMax_");
    const minStride = getStride1D(dataMin);
    const maxStride = getStride1D(dataMax);

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        const rawMin = minData[dataMin.offset + j * minStride];
        const rawMax = maxData[dataMax.offset + j * maxStride];
        if (rawMin === undefined || rawMax === undefined) {
          throw new DeepboxError("Internal error: min/max tensor access out of bounds");
        }
        const min = Number(rawMin);
        const max = Number(rawMax);
        const range = max - min;

        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        let scaled =
          range !== 0 ? ((val - min) / range) * (maxRange - minRange) + minRange : minRange;
        if (this.clip) {
          scaled = Math.max(minRange, Math.min(maxRange, scaled));
        }
        row[j] = scaled;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MinMaxScaler must be fitted before inverse_transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const [minRange, maxRange] = this.featureRange;
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);
    const dataMin = this.dataMin_;
    const dataMax = this.dataMax_;

    if (!dataMin || !dataMax) {
      throw new DeepboxError("MinMaxScaler internal error: missing fitted min/max");
    }
    const minData = getNumericData(dataMin, "dataMin_");
    const maxData = getNumericData(dataMax, "dataMax_");
    const minStride = getStride1D(dataMin);
    const maxStride = getStride1D(dataMax);

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        const rawMin = minData[dataMin.offset + j * minStride];
        const rawMax = maxData[dataMax.offset + j * maxStride];
        if (rawMin === undefined || rawMax === undefined) {
          throw new DeepboxError("Internal error: min/max tensor access out of bounds");
        }
        const min = Number(rawMin);
        const max = Number(rawMax);
        const range = max - min;

        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[j] = ((val - minRange) / (maxRange - minRange)) * range + min;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }
}

/**
 * Scale features by maximum absolute value.
 *
 * Scales to range [-1, 1]. Suitable for data that is already centered at zero.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class MaxAbsScaler {
  private fitted = false;
  private maxAbs_?: Tensor;

  /**
   * Creates a new MaxAbsScaler.
   *
   * @param options - Configuration options
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   */
  constructor(options: { copy?: boolean } = {}) {
    parseBooleanOption(options.copy, "copy", true);
  }

  fit(X: Tensor): this {
    if (X.size === 0) {
      throw new InvalidParameterError("X must contain at least one sample", "X");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    const maxAbs = new Array<number>(nFeatures).fill(0);

    for (let j = 0; j < nFeatures; j++) {
      for (let i = 0; i < nSamples; i++) {
        const raw = data[X.offset + i * stride0 + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const currentMax = maxAbs[j];
        if (currentMax === undefined) {
          throw new DeepboxError("Internal error: maxAbs array access failed");
        }
        maxAbs[j] = Math.max(currentMax, Math.abs(Number(raw)));
      }
    }

    this.maxAbs_ = tensor(maxAbs, { dtype: "float64" });
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MaxAbsScaler must be fitted before transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);
    const maxAbs = this.maxAbs_;
    if (!maxAbs) {
      throw new DeepboxError("MaxAbsScaler internal error: missing fitted maxAbs");
    }
    const maxData = getNumericData(maxAbs, "maxAbs_");
    const maxStride = getStride1D(maxAbs);

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        const rawScale = maxData[maxAbs.offset + j * maxStride];
        if (rawScale === undefined) {
          throw new DeepboxError("Internal error: maxAbs tensor access out of bounds");
        }
        const scale = Number(rawScale);
        const safeScale = scale === 0 ? 1 : scale;
        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[j] = val / safeScale;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MaxAbsScaler must be fitted before inverse_transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);
    const maxAbs = this.maxAbs_;
    if (!maxAbs) {
      throw new DeepboxError("MaxAbsScaler internal error: missing fitted maxAbs");
    }
    const maxData = getNumericData(maxAbs, "maxAbs_");
    const maxStride = getStride1D(maxAbs);

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        const rawScale = maxData[maxAbs.offset + j * maxStride];
        if (rawScale === undefined) {
          throw new DeepboxError("Internal error: maxAbs tensor access out of bounds");
        }
        const scale = Number(rawScale);
        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[j] = val * scale;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }
}

/**
 * Robust scaler using median and IQR.
 *
 * Robust to outliers.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class RobustScaler {
  private fitted = false;
  private center_: Tensor | undefined;
  private scale_: Tensor | undefined;
  private withCentering: boolean;
  private withScaling: boolean;
  private quantileRange: [number, number];
  private unitVariance: boolean;

  /**
   * Creates a new RobustScaler.
   *
   * @param options - Configuration options
   * @param options.withCentering - Center data using median (default: true)
   * @param options.withScaling - Scale data using IQR (default: true)
   * @param options.quantileRange - Quantile range for IQR as percentiles (default: [25, 75])
   * @param options.unitVariance - Scale so that features have unit variance under normality (default: false)
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   */
  constructor(
    options: {
      withCentering?: boolean;
      withScaling?: boolean;
      quantileRange?: [number, number];
      unitVariance?: boolean;
      copy?: boolean;
    } = {}
  ) {
    this.withCentering = parseBooleanOption(options.withCentering, "withCentering", true);
    this.withScaling = parseBooleanOption(options.withScaling, "withScaling", true);
    this.quantileRange = options.quantileRange ?? [25, 75];
    this.unitVariance = parseBooleanOption(options.unitVariance, "unitVariance", false);
    parseBooleanOption(options.copy, "copy", true);
    const [lower, upper] = this.quantileRange;
    if (
      !Number.isFinite(lower) ||
      !Number.isFinite(upper) ||
      lower < 0 ||
      upper > 100 ||
      lower >= upper
    ) {
      throw new InvalidParameterError(
        "quantileRange must be a valid ascending percentile range",
        "quantileRange",
        this.quantileRange
      );
    }
  }

  fit(X: Tensor): this {
    if (X.size === 0) {
      throw new InvalidParameterError("X must contain at least one sample", "X");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    const centers = new Array<number>(nFeatures).fill(0);
    const scales = new Array<number>(nFeatures).fill(0);

    // Convert quantile range from percentiles to fractions
    const [lowerPercentile, upperPercentile] = this.quantileRange;
    const lowerFraction = lowerPercentile / 100;
    const upperFraction = upperPercentile / 100;
    const normalizer = this.unitVariance
      ? normalQuantile(upperFraction) - normalQuantile(lowerFraction)
      : 1;
    if (this.unitVariance && (!Number.isFinite(normalizer) || normalizer <= 0)) {
      throw new DeepboxError("RobustScaler internal error: invalid unit variance normalizer");
    }

    for (let j = 0; j < nFeatures; j++) {
      const values: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const raw = data[X.offset + i * stride0 + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        values.push(Number(raw));
      }
      values.sort((a, b) => a - b);

      const interpolate = (q: number): number => {
        if (values.length === 0) {
          throw new DeepboxError("Internal error: cannot interpolate empty values");
        }
        if (values.length === 1) {
          const only = values[0];
          if (only === undefined) {
            throw new DeepboxError("Internal error: missing sorted value");
          }
          return only;
        }

        const position = q * (values.length - 1);
        const lower = Math.floor(position);
        const upper = Math.ceil(position);
        const lowerValue = values[lower];
        const upperValue = values[upper];
        if (lowerValue === undefined || upperValue === undefined) {
          throw new DeepboxError("Internal error: quantile interpolation index out of bounds");
        }
        if (upper === lower) {
          return lowerValue;
        }
        const weight = position - lower;
        return lowerValue * (1 - weight) + upperValue * weight;
      };

      // Median for centering and IQR for scaling
      centers[j] = interpolate(0.5);
      const qLower = interpolate(lowerFraction);
      const qUpper = interpolate(upperFraction);
      const iqr = qUpper - qLower;
      scales[j] = this.unitVariance ? iqr / normalizer : iqr;
    }

    this.center_ = this.withCentering ? tensor(centers, { dtype: "float64" }) : undefined;
    this.scale_ = this.withScaling ? tensor(scales, { dtype: "float64" }) : undefined;
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("RobustScaler must be fitted before transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);
    const center = this.center_;
    const scale = this.scale_;
    const centerData = center ? getNumericData(center, "center_") : undefined;
    const scaleData = scale ? getNumericData(scale, "scale_") : undefined;
    const centerStride = center ? getStride1D(center) : 0;
    const scaleStride = scale ? getStride1D(scale) : 0;

    if (this.withCentering && !center) {
      throw new DeepboxError("RobustScaler internal error: missing center_");
    }
    if (this.withScaling && !scale) {
      throw new DeepboxError("RobustScaler internal error: missing scale_");
    }

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        let val = Number(raw);

        if (this.withCentering && center && centerData) {
          const rawCenter = centerData[center.offset + j * centerStride];
          if (rawCenter === undefined) {
            throw new DeepboxError("Internal error: center tensor access out of bounds");
          }
          val -= Number(rawCenter);
        }

        if (this.withScaling && scale && scaleData) {
          const rawScale = scaleData[scale.offset + j * scaleStride];
          if (rawScale === undefined) {
            throw new DeepboxError("Internal error: scale tensor access out of bounds");
          }
          const scaleValue = Number(rawScale);
          const safeScale = scaleValue === 0 ? 1 : scaleValue;
          val /= safeScale;
        }

        const resultRow = result[i];
        if (resultRow === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        resultRow[j] = val;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("RobustScaler must be fitted before inverse_transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);
    const center = this.center_;
    const scale = this.scale_;
    const centerData = center ? getNumericData(center, "center_") : undefined;
    const scaleData = scale ? getNumericData(scale, "scale_") : undefined;
    const centerStride = center ? getStride1D(center) : 0;
    const scaleStride = scale ? getStride1D(scale) : 0;

    if (this.withCentering && !center) {
      throw new DeepboxError("RobustScaler internal error: missing center_");
    }
    if (this.withScaling && !scale) {
      throw new DeepboxError("RobustScaler internal error: missing scale_");
    }

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        let val = Number(raw);

        if (this.withScaling && scale && scaleData) {
          const rawScale = scaleData[scale.offset + j * scaleStride];
          if (rawScale === undefined) {
            throw new DeepboxError("Internal error: scale tensor access out of bounds");
          }
          const scaleValue = Number(rawScale);
          const safeScale = scaleValue === 0 ? 1 : scaleValue;
          val *= safeScale;
        }

        if (this.withCentering && center && centerData) {
          const rawCenter = centerData[center.offset + j * centerStride];
          if (rawCenter === undefined) {
            throw new DeepboxError("Internal error: center tensor access out of bounds");
          }
          val += Number(rawCenter);
        }

        const resultRow = result[i];
        if (resultRow === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        resultRow[j] = val;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }
}

/**
 * Normalize samples to unit norm.
 *
 * Scales each sample (row) to have unit norm.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class Normalizer {
  private norm: "l1" | "l2" | "max";

  /**
   * Creates a new Normalizer.
   *
   * @param options - Configuration options
   * @param options.norm - Norm to use (default: "l2")
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   */
  constructor(options: { norm?: "l1" | "l2" | "max"; copy?: boolean } = {}) {
    this.norm = options.norm ?? "l2";
    if (this.norm !== "l1" && this.norm !== "l2" && this.norm !== "max") {
      throw new InvalidParameterError("norm must be one of: l1, l2, max", "norm", this.norm);
    }
    parseBooleanOption(options.copy, "copy", true);
  }

  fit(_X: Tensor): this {
    return this;
  }

  transform(X: Tensor): Tensor {
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      let norm = 0;
      const rowBase = X.offset + i * stride0;

      if (this.norm === "l2") {
        for (let j = 0; j < nFeatures; j++) {
          const raw = data[rowBase + j * stride1];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: numeric tensor access out of bounds");
          }
          const val = Number(raw);
          norm += val * val;
        }
        norm = Math.sqrt(norm);
      } else if (this.norm === "l1") {
        for (let j = 0; j < nFeatures; j++) {
          const raw = data[rowBase + j * stride1];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: numeric tensor access out of bounds");
          }
          norm += Math.abs(Number(raw));
        }
      } else if (this.norm === "max") {
        for (let j = 0; j < nFeatures; j++) {
          const raw = data[rowBase + j * stride1];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: numeric tensor access out of bounds");
          }
          norm = Math.max(norm, Math.abs(Number(raw)));
        }
      }

      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[j] = norm === 0 ? val : val / norm;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  fitTransform(X: Tensor): Tensor {
    return this.transform(X);
  }
}

/**
 * Transform features using quantiles.
 *
 * Maps to uniform or normal distribution.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class QuantileTransformer {
  private fitted = false;
  private nQuantiles: number;
  private outputDistribution: "uniform" | "normal";
  private quantiles_?: Map<number, { quantiles: number[]; references: number[] }>;
  private subsample: number | undefined;
  private randomState: number | undefined;

  /**
   * Creates a new QuantileTransformer.
   *
   * @param options - Configuration options
   * @param options.nQuantiles - Number of quantiles to use (default: 1000)
   * @param options.outputDistribution - "uniform" or "normal" (default: "uniform")
   * @param options.subsample - Subsample size for quantile estimation (default: use all samples)
   * @param options.randomState - Seed for subsampling reproducibility
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   */
  constructor(
    options: {
      nQuantiles?: number;
      outputDistribution?: "uniform" | "normal";
      subsample?: number;
      randomState?: number;
      copy?: boolean;
    } = {}
  ) {
    this.nQuantiles = options.nQuantiles ?? 1000;
    this.outputDistribution = options.outputDistribution ?? "uniform";
    this.subsample = options.subsample;
    this.randomState = options.randomState;
    parseBooleanOption(options.copy, "copy", true);
    if (
      !Number.isFinite(this.nQuantiles) ||
      !Number.isInteger(this.nQuantiles) ||
      this.nQuantiles < 2
    ) {
      throw new InvalidParameterError(
        "nQuantiles must be at least 2",
        "nQuantiles",
        this.nQuantiles
      );
    }
    if (this.outputDistribution !== "uniform" && this.outputDistribution !== "normal") {
      throw new InvalidParameterError(
        "outputDistribution must be 'uniform' or 'normal'",
        "outputDistribution",
        this.outputDistribution
      );
    }
    if (this.subsample !== undefined) {
      if (
        !Number.isFinite(this.subsample) ||
        !Number.isInteger(this.subsample) ||
        this.subsample < 2
      ) {
        throw new InvalidParameterError(
          "subsample must be an integer >= 2",
          "subsample",
          this.subsample
        );
      }
    }
  }

  fit(X: Tensor): this {
    if (X.size === 0) {
      throw new InvalidParameterError("X must contain at least one sample", "X");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    this.quantiles_ = new Map();
    const sampleCount =
      this.subsample !== undefined ? Math.min(this.subsample, nSamples) : nSamples;
    const nQuantilesEffective = Math.min(this.nQuantiles, sampleCount);
    const references =
      nQuantilesEffective <= 1
        ? [0.5]
        : Array.from({ length: nQuantilesEffective }, (_, i) => i / (nQuantilesEffective - 1));

    let sampleIndices: number[] | undefined;
    if (sampleCount < nSamples) {
      sampleIndices = Array.from({ length: nSamples }, (_, i) => i);
      const random =
        this.randomState !== undefined ? createSeededRandom(this.randomState) : Math.random;
      shuffleIndicesInPlace(sampleIndices, random);
      sampleIndices = sampleIndices.slice(0, sampleCount);
    }

    for (let j = 0; j < nFeatures; j++) {
      const values: number[] = [];
      if (sampleIndices) {
        for (const idx of sampleIndices) {
          const raw = data[X.offset + idx * stride0 + j * stride1];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: numeric tensor access out of bounds");
          }
          values.push(Number(raw));
        }
      } else {
        for (let i = 0; i < nSamples; i++) {
          const raw = data[X.offset + i * stride0 + j * stride1];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: numeric tensor access out of bounds");
          }
          values.push(Number(raw));
        }
      }
      const sorted = [...values].sort((a, b) => a - b);
      const quantiles = references.map((q) => this.interpolateFromSorted(sorted, q));
      this.quantiles_.set(j, { quantiles, references });
    }

    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.quantiles_) {
      throw new NotFittedError("QuantileTransformer must be fitted before transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    if (nSamples === 0) {
      return zeros([0, nFeatures], { dtype: "float64" });
    }

    const result = new Array<number[]>(nSamples);
    for (let i = 0; i < nSamples; i++) {
      result[i] = new Array<number>(nFeatures);
    }

    for (let j = 0; j < nFeatures; j++) {
      const feature = this.quantiles_.get(j);
      if (!feature) {
        throw new DeepboxError(`Internal error: missing fitted quantiles for feature ${j}`);
      }

      for (let i = 0; i < nSamples; i++) {
        const raw = data[X.offset + i * stride0 + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        const quantile = this.mapValueToQuantile(val, feature.quantiles, feature.references);

        const row = result[i];
        if (!row) {
          throw new DeepboxError("Internal error: result row access failed");
        }

        if (this.outputDistribution === "uniform") {
          row[j] = quantile;
        } else {
          // Transform to normal distribution using inverse error function
          // Clamp quantile to avoid numerical issues at boundaries
          const clampedQuantile = Math.max(1e-7, Math.min(1 - 1e-7, quantile));
          const z = Math.sqrt(2) * this.erfInv(2 * clampedQuantile - 1);
          row[j] = z;
        }
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  /**
   * Inverse transform data back to the original feature space.
   *
   * If `outputDistribution="normal"`, values are first mapped back to uniform
   * quantiles before being projected into the original data distribution.
   *
   * @param X - Transformed data (2D tensor)
   * @returns Data in the original feature space
   * @throws {NotFittedError} If transformer is not fitted
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.quantiles_) {
      throw new NotFittedError("QuantileTransformer must be fitted before inverse_transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    if (nSamples === 0) {
      return zeros([0, nFeatures], { dtype: "float64" });
    }

    const result = new Array<number[]>(nSamples);
    for (let i = 0; i < nSamples; i++) {
      result[i] = new Array<number>(nFeatures);
    }

    for (let j = 0; j < nFeatures; j++) {
      const feature = this.quantiles_.get(j);
      if (!feature) {
        throw new DeepboxError(`Internal error: missing fitted quantiles for feature ${j}`);
      }

      for (let i = 0; i < nSamples; i++) {
        const raw = data[X.offset + i * stride0 + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const value = Number(raw);
        let quantile = this.outputDistribution === "normal" ? this.normalCdf(value) : value;

        quantile = Math.max(0, Math.min(1, quantile));
        const row = result[i];
        if (!row) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[j] = this.mapQuantileToValue(quantile, feature.quantiles, feature.references);
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  private erf(x: number): number {
    // Abramowitz and Stegun approximation
    const sign = x < 0 ? -1 : 1;
    const absX = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * absX);
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    return sign * (1 - poly * Math.exp(-absX * absX));
  }

  private normalCdf(z: number): number {
    return 0.5 * (1 + this.erf(z / Math.sqrt(2)));
  }

  private erfInv(x: number): number {
    const a = 0.147;
    const b = 2 / (Math.PI * a) + Math.log(1 - x * x) / 2;
    const sign = x < 0 ? -1 : 1;
    return sign * Math.sqrt(Math.sqrt(b * b - Math.log(1 - x * x) / a) - b);
  }

  private interpolateFromSorted(sorted: number[], q: number): number {
    if (sorted.length === 0) {
      throw new DeepboxError("Internal error: cannot interpolate empty sorted values");
    }
    if (sorted.length === 1) {
      const only = sorted[0];
      if (only === undefined) {
        throw new DeepboxError("Internal error: missing sorted value");
      }
      return only;
    }

    const position = q * (sorted.length - 1);
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    const lowerValue = sorted[lower];
    const upperValue = sorted[upper];
    if (lowerValue === undefined || upperValue === undefined) {
      throw new DeepboxError("Internal error: quantile interpolation index out of bounds");
    }

    if (upper === lower) {
      return lowerValue;
    }

    const weight = position - lower;
    return lowerValue * (1 - weight) + upperValue * weight;
  }

  private mapValueToQuantile(value: number, quantiles: number[], references: number[]): number {
    const n = quantiles.length;
    if (n === 0) {
      return 0;
    }
    if (n === 1) {
      const onlyReference = references[0];
      if (onlyReference === undefined) {
        throw new DeepboxError("Internal error: missing quantile reference");
      }
      return onlyReference;
    }

    const firstQuantile = quantiles[0];
    const lastQuantile = quantiles[n - 1];
    if (firstQuantile === undefined || lastQuantile === undefined) {
      throw new DeepboxError("Internal error: missing quantile endpoints");
    }
    if (value <= firstQuantile) {
      return 0;
    }
    if (value >= lastQuantile) {
      return 1;
    }

    let left = 0;
    let right = n - 1;
    while (left + 1 < right) {
      const mid = Math.floor((left + right) / 2);
      const midValue = quantiles[mid];
      if (midValue === undefined) {
        throw new DeepboxError("Internal error: missing quantile midpoint");
      }
      if (midValue <= value) {
        left = mid;
      } else {
        right = mid;
      }
    }

    const qLeft = quantiles[left];
    const qRight = quantiles[right];
    const rLeft = references[left];
    const rRight = references[right];
    if (
      qLeft === undefined ||
      qRight === undefined ||
      rLeft === undefined ||
      rRight === undefined
    ) {
      throw new DeepboxError("Internal error: missing quantile interpolation points");
    }
    if (qRight <= qLeft) {
      return (rLeft + rRight) / 2;
    }

    const ratio = (value - qLeft) / (qRight - qLeft);
    return rLeft + ratio * (rRight - rLeft);
  }

  private mapQuantileToValue(quantile: number, quantiles: number[], references: number[]): number {
    const n = references.length;
    if (n === 0) {
      return 0;
    }
    if (n === 1) {
      const onlyQuantile = quantiles[0];
      if (onlyQuantile === undefined) {
        throw new DeepboxError("Internal error: missing quantile value");
      }
      return onlyQuantile;
    }

    const firstRef = references[0];
    const lastRef = references[n - 1];
    if (firstRef === undefined || lastRef === undefined) {
      throw new DeepboxError("Internal error: missing reference endpoints");
    }
    if (quantile <= firstRef) {
      const firstQuantile = quantiles[0];
      if (firstQuantile === undefined) {
        throw new DeepboxError("Internal error: missing quantile endpoints");
      }
      return firstQuantile;
    }
    if (quantile >= lastRef) {
      const lastQuantile = quantiles[n - 1];
      if (lastQuantile === undefined) {
        throw new DeepboxError("Internal error: missing quantile endpoints");
      }
      return lastQuantile;
    }

    let left = 0;
    let right = n - 1;
    while (left + 1 < right) {
      const mid = Math.floor((left + right) / 2);
      const midRef = references[mid];
      if (midRef === undefined) {
        throw new DeepboxError("Internal error: missing quantile reference");
      }
      if (midRef <= quantile) {
        left = mid;
      } else {
        right = mid;
      }
    }

    const rLeft = references[left];
    const rRight = references[right];
    const qLeft = quantiles[left];
    const qRight = quantiles[right];
    if (
      rLeft === undefined ||
      rRight === undefined ||
      qLeft === undefined ||
      qRight === undefined
    ) {
      throw new DeepboxError("Internal error: missing quantile interpolation points");
    }
    if (rRight <= rLeft) {
      return (qLeft + qRight) / 2;
    }
    const ratio = (quantile - rLeft) / (rRight - rLeft);
    return qLeft + ratio * (qRight - qLeft);
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }
}

/**
 * Apply power transform to make data more Gaussian-like.
 *
 * Supports Box-Cox and Yeo-Johnson transforms, with optional standardization.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class PowerTransformer {
  private fitted = false;
  private method: "box-cox" | "yeo-johnson";
  private lambdas_: number[] | undefined;
  private standardize: boolean;
  private mean_: number[] | undefined;
  private scale_: number[] | undefined;

  /**
   * Creates a new PowerTransformer.
   *
   * @param options - Configuration options
   * @param options.method - "box-cox" or "yeo-johnson" (default: "yeo-johnson")
   * @param options.standardize - Whether to standardize transformed features (default: false)
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   */
  constructor(
    options: {
      method?: "box-cox" | "yeo-johnson";
      standardize?: boolean;
      copy?: boolean;
    } = {}
  ) {
    this.method = options.method ?? "yeo-johnson";
    if (this.method !== "box-cox" && this.method !== "yeo-johnson") {
      throw new InvalidParameterError(
        "method must be 'box-cox' or 'yeo-johnson'",
        "method",
        this.method
      );
    }
    this.standardize = parseBooleanOption(options.standardize, "standardize", false);
    parseBooleanOption(options.copy, "copy", true);
  }

  fit(X: Tensor): this {
    if (X.size === 0) {
      throw new InvalidParameterError("X must contain at least one sample", "X");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    const lambdas = new Array<number>(nFeatures);
    const means = this.standardize ? new Array<number>(nFeatures).fill(0) : undefined;
    const scales = this.standardize ? new Array<number>(nFeatures).fill(0) : undefined;

    for (let j = 0; j < nFeatures; j++) {
      const featureValues = new Array<number>(nSamples);
      for (let i = 0; i < nSamples; i++) {
        const raw = data[X.offset + i * stride0 + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const value = Number(raw);
        if (this.method === "box-cox" && value <= 0) {
          throw new InvalidParameterError(
            `Box-Cox requires strictly positive values in fit data (feature ${j})`,
            "X",
            value
          );
        }
        featureValues[i] = value;
      }
      const lambda = this.optimizeLambda(featureValues);
      lambdas[j] = lambda;

      if (this.standardize && means && scales) {
        let sum = 0;
        // Pass 1: Mean
        for (const value of featureValues) {
          const transformed =
            this.method === "box-cox"
              ? this.boxCoxTransformValue(value, lambda)
              : this.yeoJohnsonTransformValue(value, lambda);
          sum += transformed;
        }
        const mean = sum / nSamples;
        means[j] = mean;

        // Pass 2: Variance (stable)
        let sumSqDiff = 0;
        for (const value of featureValues) {
          const transformed =
            this.method === "box-cox"
              ? this.boxCoxTransformValue(value, lambda)
              : this.yeoJohnsonTransformValue(value, lambda);
          const diff = transformed - mean;
          sumSqDiff += diff * diff;
        }
        const variance = sumSqDiff / nSamples;
        const std = Math.sqrt(Math.max(variance, 0));
        scales[j] = std === 0 ? 1 : std;
      }
    }

    this.lambdas_ = lambdas;
    this.mean_ = this.standardize ? means : undefined;
    this.scale_ = this.standardize ? scales : undefined;
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.lambdas_) {
      throw new NotFittedError("PowerTransformer must be fitted before transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    if (this.standardize && (!this.mean_ || !this.scale_)) {
      throw new DeepboxError("PowerTransformer internal error: missing standardization stats");
    }

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        const lambda = this.lambdas_[j];
        if (lambda === undefined) {
          throw new DeepboxError(`Internal error: missing fitted lambda for feature ${j}`);
        }

        let transformed: number;
        if (this.method === "box-cox") {
          if (val <= 0) {
            throw new InvalidParameterError("Box-Cox requires strictly positive values", "X", val);
          }
          transformed = this.boxCoxTransformValue(val, lambda);
        } else {
          transformed = this.yeoJohnsonTransformValue(val, lambda);
        }

        if (this.standardize && this.mean_ && this.scale_) {
          const mean = this.mean_[j] ?? 0;
          const scale = this.scale_[j] ?? 1;
          transformed = (transformed - mean) / scale;
        }

        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[j] = transformed;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  /**
   * Inverse transform data back to the original feature space.
   * If `standardize=true`, de-standardizes before applying the inverse power transform.
   *
   * @param X - Transformed data (2D tensor)
   * @returns Data in the original feature space
   * @throws {NotFittedError} If transformer is not fitted
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.lambdas_) {
      throw new NotFittedError("PowerTransformer must be fitted before inverse_transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    validateFiniteData(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const data = getNumericData(X, "X");
    const [stride0, stride1] = getStrides2D(X);

    if (this.standardize && (!this.mean_ || !this.scale_)) {
      throw new DeepboxError("PowerTransformer internal error: missing standardization stats");
    }

    const result = Array.from({ length: nSamples }, () => new Array<number>(nFeatures).fill(0));

    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * stride0;
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        let val = Number(raw);

        if (this.standardize && this.mean_ && this.scale_) {
          const mean = this.mean_[j] ?? 0;
          const scale = this.scale_[j] ?? 1;
          val = val * scale + mean;
        }

        const lambda = this.lambdas_[j];
        if (lambda === undefined) {
          throw new DeepboxError(`Internal error: missing fitted lambda for feature ${j}`);
        }

        let inverted: number;
        if (this.method === "box-cox") {
          inverted = this.boxCoxInverseValue(val, lambda);
        } else {
          inverted = this.yeoJohnsonInverseValue(val, lambda);
        }

        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[j] = inverted;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  private boxCoxTransformValue(value: number, lambda: number): number {
    return Math.abs(lambda) < 1e-12 ? Math.log(value) : (value ** lambda - 1) / lambda;
  }

  private yeoJohnsonTransformValue(value: number, lambda: number): number {
    if (value >= 0) {
      return Math.abs(lambda) < 1e-12 ? Math.log(value + 1) : ((value + 1) ** lambda - 1) / lambda;
    }
    const twoMinusLambda = 2 - lambda;
    return Math.abs(twoMinusLambda) < 1e-12
      ? -Math.log(1 - value)
      : -((1 - value) ** twoMinusLambda - 1) / twoMinusLambda;
  }

  private boxCoxInverseValue(value: number, lambda: number): number {
    if (Math.abs(lambda) < 1e-12) {
      return Math.exp(value);
    }
    const base = value * lambda + 1;
    if (base <= 0) {
      throw new InvalidParameterError("Box-Cox inverse encountered invalid value", "X", value);
    }
    return base ** (1 / lambda);
  }

  private yeoJohnsonInverseValue(value: number, lambda: number): number {
    if (value >= 0) {
      if (Math.abs(lambda) < 1e-12) {
        return Math.exp(value) - 1;
      }
      const base = value * lambda + 1;
      if (base <= 0) {
        throw new InvalidParameterError(
          "Yeo-Johnson inverse encountered invalid value",
          "X",
          value
        );
      }
      return base ** (1 / lambda) - 1;
    }

    const twoMinusLambda = 2 - lambda;
    if (Math.abs(twoMinusLambda) < 1e-12) {
      return 1 - Math.exp(-value);
    }
    const base = 1 - value * twoMinusLambda;
    if (base <= 0) {
      throw new InvalidParameterError("Yeo-Johnson inverse encountered invalid value", "X", value);
    }
    return 1 - base ** (1 / twoMinusLambda);
  }

  private logLikelihood(values: readonly number[], lambda: number): number {
    const transformed = new Array<number>(values.length);
    let jacobian = 0;

    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (value === undefined) {
        throw new DeepboxError("Internal error: missing feature value during optimization");
      }

      let transformedValue: number;
      if (this.method === "box-cox") {
        if (value <= 0) {
          return Number.NEGATIVE_INFINITY;
        }
        transformedValue = this.boxCoxTransformValue(value, lambda);
        jacobian += (lambda - 1) * Math.log(value);
      } else {
        transformedValue = this.yeoJohnsonTransformValue(value, lambda);
        jacobian +=
          value >= 0 ? (lambda - 1) * Math.log(value + 1) : (1 - lambda) * Math.log(1 - value);
      }

      if (!Number.isFinite(transformedValue)) {
        return Number.NEGATIVE_INFINITY;
      }
      transformed[i] = transformedValue;
    }

    let sum = 0;
    for (const value of transformed) {
      sum += value;
    }
    const mean = sum / transformed.length;

    let varianceSum = 0;
    for (const value of transformed) {
      const delta = value - mean;
      varianceSum += delta * delta;
    }
    const variance = varianceSum / transformed.length;
    if (!Number.isFinite(variance) || variance <= 1e-15) {
      return Number.NEGATIVE_INFINITY;
    }

    return -0.5 * transformed.length * Math.log(variance) + jacobian;
  }

  private optimizeLambda(values: readonly number[]): number {
    if (values.length < 2) {
      return 1;
    }

    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    for (const value of values) {
      if (value < minValue) minValue = value;
      if (value > maxValue) maxValue = value;
    }

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || maxValue - minValue <= 1e-15) {
      return 1;
    }

    let left = -5;
    let right = 5;
    const phi = (Math.sqrt(5) - 1) / 2;
    let c = right - phi * (right - left);
    let d = left + phi * (right - left);
    let fc = this.logLikelihood(values, c);
    let fd = this.logLikelihood(values, d);

    for (let iter = 0; iter < 80; iter++) {
      if (Math.abs(right - left) < 1e-6) break;
      if (fc > fd) {
        right = d;
        d = c;
        fd = fc;
        c = right - phi * (right - left);
        fc = this.logLikelihood(values, c);
      } else {
        left = c;
        c = d;
        fc = fd;
        d = left + phi * (right - left);
        fd = this.logLikelihood(values, d);
      }
    }

    const candidates = [left, right, (left + right) / 2, 0, 1, 2, -2];
    let bestLambda = 1;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const lambda of candidates) {
      const score = this.logLikelihood(values, lambda);
      if (score > bestScore) {
        bestScore = score;
        bestLambda = lambda;
      }
    }

    return Number.isFinite(bestLambda) ? bestLambda : 1;
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }
}
