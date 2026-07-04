import {
  DTypeError,
  dtypeToTypedArrayCtor,
  ensureNumericDType,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  ShapeError,
} from "../../core";
import {
  type AnyTensor,
  GradTensor,
  noGrad,
  ones,
  parameter,
  varianceGrad,
  zeros,
} from "../../ndarray";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides, Tensor as TensorClass } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";

function toContiguousTensor(t: TensorClass): TensorClass {
  if (isContiguous(t.shape, t.strides)) {
    return t;
  }
  if (t.dtype === "string") {
    throw new DTypeError("Normalization does not support string dtype");
  }
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const data = t.data;

  if (Array.isArray(data)) {
    throw new DTypeError("Normalization does not support string dtype");
  }

  if (data instanceof BigInt64Array) {
    if (!(out instanceof BigInt64Array)) {
      throw new DTypeError("Expected int64 output buffer for int64 tensor");
    }
    for (let i = 0; i < t.size; i++) {
      const offset = offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = getBigIntElement(data, offset);
    }
  } else {
    if (out instanceof BigInt64Array) {
      throw new DTypeError("Unexpected int64 output buffer for numeric tensor");
    }
    for (let i = 0; i < t.size; i++) {
      const offset = offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      out[i] = getNumericElement(data, offset);
    }
  }

  return TensorClass.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Internal shared BatchNorm implementation.
 *
 * BatchNorm1d, BatchNorm2d, and BatchNorm3d differ only in the expected
 * input dimensionality, spatial permute order, and broadcast shapes.
 * This base class parameterises those differences.
 */
abstract class _BatchNorm extends Module {
  protected readonly numFeatures: number;
  protected readonly eps: number;
  protected readonly momentum: number;
  protected readonly affine: boolean;
  protected readonly trackRunningStats: boolean;

  protected gamma?: GradTensor;
  protected beta?: GradTensor;
  protected runningMean: GradTensor;
  protected runningVar: GradTensor;

  private readonly name: string;

  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
    } = {},
    name: string
  ) {
    super();
    if (
      !Number.isFinite(numFeatures) ||
      numFeatures <= 0 ||
      Math.trunc(numFeatures) !== numFeatures
    ) {
      throw new InvalidParameterError(
        "numFeatures must be a positive integer",
        "numFeatures",
        numFeatures
      );
    }
    this.name = name;
    this.numFeatures = numFeatures;
    this.eps = options.eps ?? 1e-5;
    if (!Number.isFinite(this.eps) || this.eps <= 0) {
      throw new InvalidParameterError("eps must be a positive number", "eps", this.eps);
    }
    this.momentum = options.momentum ?? 0.1;
    if (!Number.isFinite(this.momentum) || this.momentum < 0 || this.momentum > 1) {
      throw new InvalidParameterError(
        "momentum must be in range [0, 1]",
        "momentum",
        this.momentum
      );
    }
    this.affine = options.affine ?? true;
    this.trackRunningStats = options.trackRunningStats ?? true;

    if (this.affine) {
      this.gamma = parameter(ones([numFeatures]));
      this.beta = parameter(zeros([numFeatures]));
      this.registerParameter("weight", this.gamma);
      this.registerParameter("bias", this.beta);
    }

    this.runningMean = GradTensor.fromTensor(zeros([numFeatures]), {
      requiresGrad: false,
    });
    this.runningVar = GradTensor.fromTensor(ones([numFeatures]), {
      requiresGrad: false,
    });

    if (this.trackRunningStats) {
      this.registerBuffer("running_mean", this.runningMean.tensor);
      this.registerBuffer("running_var", this.runningVar.tensor);
    }
  }

  /**
   * Validate the input ndim is valid for this BatchNorm variant.
   * BatchNorm1d accepts 2D or 3D, BatchNorm2d only 4D, BatchNorm3d only 5D.
   */
  protected abstract validateInputNdims(input: GradTensor): void;

  /**
   * Permute and reshape the input to (spatial*batch, numFeatures) for statistics.
   */
  protected abstract flattenForStats(input: GradTensor): GradTensor;

  /**
   * Build the broadcast shape for mean/var/gamma/beta to match the input.
   */
  protected abstract broadcastShape(nFeatures: number, input: GradTensor): readonly number[];

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError(`${this.name} does not support string dtype`);
    }

    this.validateInputNdims(input);

    const nFeatures = input.shape[1] ?? 0;
    if (nFeatures !== this.numFeatures) {
      throw new ShapeError(`Expected ${this.numFeatures} channels, got ${nFeatures}`);
    }

    const flat = this.flattenForStats(input);

    const useBatchStats = this.training || !this.trackRunningStats;

    let mean: GradTensor;
    let varVal: GradTensor;

    if (useBatchStats) {
      if (flat.shape[0] === 0) {
        throw new InvalidParameterError(
          "BatchNorm requires at least one element",
          "input",
          input.shape
        );
      }

      mean = flat.mean(0);
      varVal = varianceGrad(flat, 0, 0);

      if (this.trackRunningStats) {
        noGrad(() => {
          const n = flat.shape[0] ?? 0;
          const unbiasedVar = n > 1 ? varianceGrad(flat, 0, 1) : varianceGrad(flat, 0, 0);
          const m = this.momentum;
          const statsDtype = this.runningMean.dtype;
          if (statsDtype === "string") {
            throw new DTypeError(`${this.name} running statistics must be numeric`);
          }
          const oneMinusM = GradTensor.scalar(1 - m, { dtype: statsDtype });
          const mScalar = GradTensor.scalar(m, { dtype: statsDtype });
          const newMean = this.runningMean.mul(oneMinusM).add(mean.mul(mScalar));
          const newVar = this.runningVar.mul(oneMinusM).add(unbiasedVar.mul(mScalar));
          this.runningMean = GradTensor.fromTensor(newMean.tensor, {
            requiresGrad: false,
          });
          this.runningVar = GradTensor.fromTensor(newVar.tensor, {
            requiresGrad: false,
          });
          this.registerBuffer("running_mean", this.runningMean.tensor);
          this.registerBuffer("running_var", this.runningVar.tensor);
        });
      }
    } else {
      mean = this.runningMean;
      varVal = this.runningVar;
    }

    const bcastShape = this.broadcastShape(nFeatures, input);
    const meanB = mean.reshape(bcastShape);
    const varB = varVal.reshape(bcastShape);

    const epsTensor = GradTensor.scalar(this.eps, { dtype: input.dtype });
    const denom = varB.add(epsTensor).sqrt();
    let out = input.sub(meanB).div(denom);

    if (this.affine && this.gamma && this.beta) {
      const gammaB = this.gamma.reshape(bcastShape);
      const betaB = this.beta.reshape(bcastShape);
      out = out.mul(gammaB).add(betaB);
    }

    return out;
  }

  override toString(): string {
    return `${this.name}(${this.numFeatures}, eps=${this.eps}, momentum=${this.momentum}, affine=${this.affine})`;
  }
}

/**
 * Batch Normalization layer.
 *
 * Normalizes the input over the batch dimension for faster and more stable training.
 *
 * **Formula**: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * During training, uses batch statistics. During evaluation, uses running statistics
 * unless `trackRunningStats=false`, in which case batch statistics are always used.
 *
 * @example
 * ```ts
 * import { BatchNorm1d } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const bn = new BatchNorm1d(10);
 * const x = tensor([[1, 2, 3], [4, 5, 6]]);
 * const y = bn.forward(x);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-normalization | Deepbox Normalization & Dropout}
 */
export class BatchNorm1d extends _BatchNorm {
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
    } = {}
  ) {
    super(numFeatures, options, "BatchNorm1d");
  }

  protected validateInputNdims(input: GradTensor): void {
    if (input.ndim !== 2 && input.ndim !== 3) {
      throw new ShapeError(`BatchNorm1d expects 2D or 3D input; got ndim=${input.ndim}`);
    }
  }

  protected flattenForStats(input: GradTensor): GradTensor {
    const nFeatures = input.shape[1] ?? 0;
    if (input.ndim === 3) {
      const batch = input.shape[0] ?? 0;
      const length = input.shape[2] ?? 0;
      const flat = batch * length;
      const inputDtype = ensureNumericDType(input.dtype, "BatchNorm1d");
      return input
        .transpose([0, 2, 1])
        .mul(GradTensor.scalar(1, { dtype: inputDtype }))
        .reshape([flat, nFeatures]);
    }
    return input;
  }

  protected broadcastShape(nFeatures: number, input: GradTensor): readonly number[] {
    if (input.ndim === 3) {
      return [1, nFeatures, 1];
    }
    return [1, nFeatures];
  }
}

/**
 * Layer Normalization.
 *
 * Normalizes across the feature dimensions (trailing dimensions specified by `normalizedShape`)
 * for each sample independently. Unlike BatchNorm, LayerNorm works the same way during training
 * and evaluation.
 *
 * **Formula**: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * @example
 * ```ts
 * import { LayerNorm } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const ln = new LayerNorm([10]);
 * const x = tensor([[1, 2, 3]]);
 * const y = ln.forward(x);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-normalization | Deepbox Normalization & Dropout}
 */
export class LayerNorm extends Module {
  private readonly normalizedShape: readonly number[];
  private readonly eps: number;
  private readonly elementwiseAffine: boolean;

  private gamma?: GradTensor;
  private beta?: GradTensor;

  constructor(
    normalizedShape: number | readonly number[],
    options: {
      readonly eps?: number;
      readonly elementwiseAffine?: boolean;
    } = {}
  ) {
    super();
    this.normalizedShape =
      typeof normalizedShape === "number" ? [normalizedShape] : Array.from(normalizedShape);

    if (this.normalizedShape.length === 0) {
      throw new InvalidParameterError(
        "normalizedShape must contain at least one dimension",
        "normalizedShape",
        normalizedShape
      );
    }

    for (const dim of this.normalizedShape) {
      if (!Number.isFinite(dim) || dim <= 0 || Math.trunc(dim) !== dim) {
        throw new InvalidParameterError(
          "All dimensions in normalizedShape must be positive integers",
          "normalizedShape",
          normalizedShape
        );
      }
    }

    this.eps = options.eps ?? 1e-5;
    if (!Number.isFinite(this.eps) || this.eps <= 0) {
      throw new InvalidParameterError("eps must be a positive number", "eps", this.eps);
    }

    this.elementwiseAffine = options.elementwiseAffine ?? true;

    if (this.elementwiseAffine) {
      this.gamma = parameter(ones(this.normalizedShape));
      this.beta = parameter(zeros(this.normalizedShape));
      this.registerParameter("weight", this.gamma);
      this.registerParameter("bias", this.beta);
    }
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    const inputDtype = input.dtype;
    if (inputDtype === "string") {
      throw new DTypeError("LayerNorm does not support string dtype");
    }

    let workingInput = input;
    if (!isContiguous(input.tensor.shape, input.tensor.strides)) {
      // Materialize a contiguous copy while preserving the autograd graph.
      // Multiplying by 1 produces a fresh contiguous tensor and a proper
      // backward node, unlike a raw `fromTensor` copy which would detach.
      const numericDtype = ensureNumericDType(inputDtype, "LayerNorm");
      workingInput = input.mul(GradTensor.scalar(1, { dtype: numericDtype }));
    }

    // Check if input shape ends with normalizedShape
    const inputShape = workingInput.shape;
    const normShape = this.normalizedShape;
    if (normShape.length > inputShape.length) {
      throw new ShapeError(`Input shape ${inputShape} too small for normalizedShape ${normShape}`);
    }

    // Check suffix
    const suffixStart = inputShape.length - normShape.length;
    for (let i = 0; i < normShape.length; i++) {
      if (inputShape[suffixStart + i] !== normShape[i]) {
        throw new ShapeError(
          `Input shape ${inputShape} does not end with normalizedShape ${normShape}`
        );
      }
    }

    // We need to flatten the normalized dimensions to calculate mean/var over them.
    // Dimensions to reduce: [suffixStart, ..., inputShape.length - 1]
    // We can reshape input to (..., Product(normShape)).
    // Then reduce over last dimension.

    const outerDims = inputShape.slice(0, suffixStart);
    const normSize = normShape.reduce((a, b) => a * b, 1);

    const flattenedShape = [...outerDims, normSize];
    const inputReshaped = workingInput.reshape(flattenedShape);

    // Mean and Var over last dim (-1)
    const mean = inputReshaped.mean(-1, true); // Keep dims to facilitate broadcasting (..., 1)
    // Biased variance (population, ddof=0) is used for normalization, matching PyTorch.
    // varianceGrad reduces the axis without keepdims, so reshape it back to the
    // kept-dims shape of `mean` for broadcasting.
    const varVal = varianceGrad(inputReshaped, -1, 0);
    const varReshaped = varVal.reshape(mean.shape);

    // Normalize
    const epsTensor = GradTensor.scalar(this.eps, { dtype: inputDtype });
    const denom = varReshaped.add(epsTensor).sqrt();
    const normalizedReshaped = inputReshaped.sub(mean).div(denom);

    // Reshape back to original shape
    let out = normalizedReshaped.reshape(inputShape);

    // Apply affine
    if (this.elementwiseAffine && this.gamma && this.beta) {
      // gamma/beta shape: normShape
      // input shape: (..., normShape)
      // Broadcasting works automatically since trailing dims match.
      out = out.mul(this.gamma).add(this.beta);
    }

    return out;
  }

  override toString(): string {
    return `LayerNorm(${this.normalizedShape}, eps=${this.eps}, elementwise_affine=${this.elementwiseAffine})`;
  }
}

/**
 * Group Normalization.
 *
 * Divides channels into groups and normalizes within each group.
 * Works well with small batch sizes where BatchNorm struggles.
 *
 * **Formula**: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * @example
 * ```ts
 * const gn = new GroupNorm(32, 256); // 32 groups, 256 channels
 * ```
 */
export class GroupNorm extends Module {
  private readonly numGroups: number;
  private readonly numChannels: number;
  private readonly eps: number;
  private readonly affine: boolean;

  private gamma?: GradTensor;
  private beta?: GradTensor;

  constructor(
    numGroups: number,
    numChannels: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
    } = {}
  ) {
    super();

    if (!Number.isInteger(numGroups) || numGroups <= 0) {
      throw new InvalidParameterError(
        "numGroups must be a positive integer",
        "numGroups",
        numGroups
      );
    }
    if (!Number.isInteger(numChannels) || numChannels <= 0) {
      throw new InvalidParameterError(
        "numChannels must be a positive integer",
        "numChannels",
        numChannels
      );
    }
    if (numChannels % numGroups !== 0) {
      throw new InvalidParameterError(
        `numChannels (${numChannels}) must be divisible by numGroups (${numGroups})`,
        "numGroups",
        numGroups
      );
    }

    this.numGroups = numGroups;
    this.numChannels = numChannels;
    this.eps = options.eps ?? 1e-5;
    this.affine = options.affine ?? true;

    if (this.affine) {
      this.gamma = parameter(ones([numChannels]));
      this.beta = parameter(zeros([numChannels]));
      this.registerParameter("weight", this.gamma);
      this.registerParameter("bias", this.beta);
    }
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("GroupNorm does not support string dtype");
    }

    // Input: (N, C, *) where * is any number of spatial dims
    if (input.ndim < 2) {
      throw new ShapeError(`GroupNorm expects at least 2D input; got ${input.ndim}D`);
    }

    const nChannels = input.shape[1] ?? 0;
    if (nChannels !== this.numChannels) {
      throw new ShapeError(`Expected ${this.numChannels} channels, got ${nChannels}`);
    }

    const batch = input.shape[0] ?? 0;
    const channelsPerGroup = this.numChannels / this.numGroups;

    // Reshape to (N, G, C/G, *) then normalize over (C/G, *)
    const spatialDims = input.shape.slice(2);
    const spatialSize = spatialDims.reduce((a, b) => a * b, 1);
    const groupShape = [batch, this.numGroups, channelsPerGroup * spatialSize];

    const reshaped = input.reshape(groupShape);
    const mean = reshaped.mean(-1, true);
    const varVal = varianceGrad(reshaped, -1, 0);
    const varReshaped = varVal.reshape(mean.shape);

    const epsTensor = GradTensor.scalar(this.eps, { dtype: input.dtype });
    const denom = varReshaped.add(epsTensor).sqrt();
    const normalized = reshaped.sub(mean).div(denom);

    // Reshape back to original
    let out = normalized.reshape(input.shape);

    if (this.affine && this.gamma && this.beta) {
      // gamma/beta shape: (C,) -> broadcast to (1, C, 1, 1, ...)
      const broadcastShape = [1, this.numChannels, ...spatialDims.map(() => 1)];
      const gammaB = this.gamma.reshape(broadcastShape);
      const betaB = this.beta.reshape(broadcastShape);
      out = out.mul(gammaB).add(betaB);
    }

    return out;
  }

  override toString(): string {
    return `GroupNorm(${this.numGroups}, ${this.numChannels}, eps=${this.eps}, affine=${this.affine})`;
  }
}

/**
 * Instance Normalization.
 *
 * Normalizes each channel of each sample independently.
 * Equivalent to GroupNorm with numGroups = numChannels.
 * Used in style transfer and image generation.
 *
 * @example
 * ```ts
 * const inorm = new InstanceNorm(64); // 64 channels
 * ```
 */
export class InstanceNorm extends Module {
  private readonly groupNorm: GroupNorm;

  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
    } = {}
  ) {
    super();
    // InstanceNorm = GroupNorm where numGroups == numChannels
    this.groupNorm = new GroupNorm(numFeatures, numFeatures, options);
    this.registerModule("group_norm", this.groupNorm);
  }

  forward(x: AnyTensor): GradTensor {
    return this.groupNorm.forward(x);
  }

  override toString(): string {
    return `InstanceNorm(${this.groupNorm.toString()})`;
  }
}

/**
 * Root Mean Square Layer Normalization (RMSNorm).
 *
 * Normalizes the input by the RMS of the feature values, without centering.
 * Used in LLaMA, Gemma, and other modern LLM architectures.
 *
 * **Formula**: y = x / RMS(x) * gamma
 * where RMS(x) = sqrt(mean(x^2) + eps)
 *
 * @example
 * ```ts
 * const rms = new RMSNorm(512);
 * ```
 */
export class RMSNorm extends Module {
  private readonly normalizedShape: readonly number[];
  private readonly eps: number;
  private gamma: GradTensor;

  constructor(
    normalizedShape: number | readonly number[],
    options: {
      readonly eps?: number;
    } = {}
  ) {
    super();
    this.normalizedShape =
      typeof normalizedShape === "number" ? [normalizedShape] : Array.from(normalizedShape);

    if (this.normalizedShape.length === 0) {
      throw new InvalidParameterError(
        "normalizedShape must contain at least one dimension",
        "normalizedShape",
        normalizedShape
      );
    }

    this.eps = options.eps ?? 1e-5;
    this.gamma = parameter(ones(this.normalizedShape));
    this.registerParameter("weight", this.gamma);
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("RMSNorm does not support string dtype");
    }

    let workingInput = input;
    if (!isContiguous(input.tensor.shape, input.tensor.strides)) {
      // Materialize a contiguous copy while preserving the autograd graph
      // (see LayerNorm.forward for rationale).
      const numericDtype = ensureNumericDType(input.dtype, "RMSNorm");
      workingInput = input.mul(GradTensor.scalar(1, { dtype: numericDtype }));
    }

    const inputShape = workingInput.shape;
    const normShape = this.normalizedShape;
    const suffixStart = inputShape.length - normShape.length;

    if (suffixStart < 0) {
      throw new ShapeError(`Input shape ${inputShape} too small for normalizedShape ${normShape}`);
    }

    for (let i = 0; i < normShape.length; i++) {
      if (inputShape[suffixStart + i] !== normShape[i]) {
        throw new ShapeError(
          `Input shape ${inputShape} does not end with normalizedShape ${normShape}`
        );
      }
    }

    const outerDims = inputShape.slice(0, suffixStart);
    const normSize = normShape.reduce((a, b) => a * b, 1);
    const flattenedShape = [...outerDims, normSize];
    const inputReshaped = workingInput.reshape(flattenedShape);

    // RMS = sqrt(mean(x^2) + eps)
    const squared = inputReshaped.mul(inputReshaped);
    const meanSquared = squared.mean(-1, true);
    const epsTensor = GradTensor.scalar(this.eps, { dtype: input.dtype });
    const rms = meanSquared.add(epsTensor).sqrt();

    const normalized = inputReshaped.div(rms);
    let out = normalized.reshape(inputShape);

    // Apply scale
    out = out.mul(this.gamma);

    return out;
  }

  override toString(): string {
    return `RMSNorm(${this.normalizedShape}, eps=${this.eps})`;
  }
}

/**
 * 2D Batch Normalization.
 *
 * Normalizes over (N, H, W) dimensions for each channel, designed for
 * 4D input tensors (batch, channels, height, width).
 *
 * @example
 * ```ts
 * const bn2d = new BatchNorm2d(64);
 * // input: (N, 64, H, W) -> normalized output
 * ```
 */
export class BatchNorm2d extends _BatchNorm {
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
    } = {}
  ) {
    super(numFeatures, options, "BatchNorm2d");
  }

  protected validateInputNdims(input: GradTensor): void {
    if (input.ndim !== 4) {
      throw new ShapeError(`BatchNorm2d expects 4D input (N, C, H, W); got ${input.ndim}D`);
    }
  }

  protected flattenForStats(input: GradTensor): GradTensor {
    const batch = input.shape[0] ?? 0;
    const nFeatures = input.shape[1] ?? 0;
    const height = input.shape[2] ?? 0;
    const width = input.shape[3] ?? 0;
    const inputDtype = ensureNumericDType(input.dtype, "BatchNorm2d");
    return input
      .transpose([0, 2, 3, 1])
      .mul(GradTensor.scalar(1, { dtype: inputDtype }))
      .reshape([batch * height * width, nFeatures]);
  }

  protected broadcastShape(nFeatures: number, _input: GradTensor): readonly number[] {
    return [1, nFeatures, 1, 1];
  }
}

/**
 * 3D Batch Normalization.
 *
 * Normalizes over (N, D1, D2, D3) dimensions for each channel, designed for
 * 5D input tensors (batch, channels, depth, height, width).
 *
 * @example
 * ```ts
 * const bn3d = new BatchNorm3d(64);
 * // input: (N, 64, D, H, W) -> normalized output
 * ```
 */
export class BatchNorm3d extends _BatchNorm {
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
    } = {}
  ) {
    super(numFeatures, options, "BatchNorm3d");
  }

  protected validateInputNdims(input: GradTensor): void {
    if (input.ndim !== 5) {
      throw new ShapeError(`BatchNorm3d expects 5D input (N, C, D, H, W); got ${input.ndim}D`);
    }
  }

  protected flattenForStats(input: GradTensor): GradTensor {
    const batch = input.shape[0] ?? 0;
    const nFeatures = input.shape[1] ?? 0;
    const depth = input.shape[2] ?? 0;
    const height = input.shape[3] ?? 0;
    const width = input.shape[4] ?? 0;
    const inputDtype = ensureNumericDType(input.dtype, "BatchNorm3d");
    return input
      .transpose([0, 2, 3, 4, 1])
      .mul(GradTensor.scalar(1, { dtype: inputDtype }))
      .reshape([batch * depth * height * width, nFeatures]);
  }

  protected broadcastShape(nFeatures: number, _input: GradTensor): readonly number[] {
    return [1, nFeatures, 1, 1, 1];
  }
}

/**
 * 1D Instance Normalization.
 *
 * Normalizes each channel independently per sample for 3D inputs (N, C, L).
 * Equivalent to GroupNorm where numGroups == numChannels.
 *
 * @example
 * ```ts
 * const inorm = new InstanceNorm1d(64);
 * // input: (N, 64, L) -> normalized output
 * ```
 */
export class InstanceNorm1d extends Module {
  private readonly groupNorm: GroupNorm;

  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
    } = {}
  ) {
    super();
    this.groupNorm = new GroupNorm(numFeatures, numFeatures, options);
    this.registerModule("group_norm", this.groupNorm);
  }

  forward(x: AnyTensor): GradTensor {
    return this.groupNorm.forward(x);
  }

  override toString(): string {
    return `InstanceNorm1d(${this.groupNorm.toString()})`;
  }
}

/**
 * 2D Instance Normalization.
 *
 * Normalizes each channel independently per sample for 4D inputs (N, C, H, W).
 * Equivalent to GroupNorm where numGroups == numChannels.
 *
 * @example
 * ```ts
 * const inorm = new InstanceNorm2d(64);
 * // input: (N, 64, H, W) -> normalized output
 * ```
 */
export class InstanceNorm2d extends Module {
  private readonly groupNorm: GroupNorm;

  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
    } = {}
  ) {
    super();
    this.groupNorm = new GroupNorm(numFeatures, numFeatures, options);
    this.registerModule("group_norm", this.groupNorm);
  }

  forward(x: AnyTensor): GradTensor {
    return this.groupNorm.forward(x);
  }

  override toString(): string {
    return `InstanceNorm2d(${this.groupNorm.toString()})`;
  }
}

/**
 * 3D Instance Normalization.
 *
 * Normalizes each channel independently per sample for 5D inputs (N, C, D, H, W).
 * Equivalent to GroupNorm where numGroups == numChannels.
 *
 * @example
 * ```ts
 * const inorm = new InstanceNorm3d(64);
 * // input: (N, 64, D, H, W) -> normalized output
 * ```
 */
export class InstanceNorm3d extends Module {
  private readonly groupNorm: GroupNorm;

  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
    } = {}
  ) {
    super();
    this.groupNorm = new GroupNorm(numFeatures, numFeatures, options);
    this.registerModule("group_norm", this.groupNorm);
  }

  forward(x: AnyTensor): GradTensor {
    return this.groupNorm.forward(x);
  }

  override toString(): string {
    return `InstanceNorm3d(${this.groupNorm.toString()})`;
  }
}

/**
 * Local Response Normalization.
 *
 * Applies local response normalization over an input signal, as described in
 * the AlexNet paper. For each element, the normalization is computed across
 * nearby channels.
 *
 * **Formula**: out_i = x_i / (k + alpha/size * sum(x_j^2))^beta
 * where the sum is over the `size` nearest channels.
 *
 * @example
 * ```ts
 * const lrn = new LocalResponseNorm(5);
 * // input: (N, C, ...) -> normalized output
 * ```
 */
export class LocalResponseNorm extends Module {
  private readonly size: number;
  private readonly alpha: number;
  private readonly beta: number;
  private readonly k: number;

  /**
   * @param size - Number of channels to normalize across (must be odd)
   * @param options.alpha - Multiplicative factor (default: 1e-4)
   * @param options.beta - Exponent (default: 0.75)
   * @param options.k - Additive factor (default: 1)
   */
  constructor(
    size: number,
    options: {
      readonly alpha?: number;
      readonly beta?: number;
      readonly k?: number;
    } = {}
  ) {
    super();
    if (!Number.isInteger(size) || size < 1) {
      throw new InvalidParameterError("size must be a positive integer", "size", size);
    }
    this.size = size;
    this.alpha = options.alpha ?? 1e-4;
    this.beta = options.beta ?? 0.75;
    this.k = options.k ?? 1;
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("LocalResponseNorm does not support string dtype");
    }

    if (input.ndim < 3) {
      throw new ShapeError(
        `LocalResponseNorm expects at least 3D input (N, C, ...); got ${input.ndim}D`
      );
    }

    const t = toContiguousTensor(input.tensor);
    const shape = t.shape;
    const C = shape[1] ?? 0;
    const batchSize = shape[0] ?? 0;

    // Compute spatial size (product of dims after channel)
    let spatialSize = 1;
    for (let d = 2; d < shape.length; d++) {
      spatialSize *= shape[d] ?? 1;
    }

    const inputDtype = ensureNumericDType(t.dtype, "LocalResponseNorm");
    const Ctor = dtypeToTypedArrayCtor(inputDtype);
    const outData = new Ctor(t.size);
    const data = t.data;

    if (Array.isArray(data) || data instanceof BigInt64Array) {
      throw new DTypeError("LocalResponseNorm requires numeric non-bigint dtype");
    }

    if (outData instanceof BigInt64Array) {
      throw new DTypeError("Unexpected BigInt64Array");
    }

    const halfSize = Math.floor(this.size / 2);
    const alphaOverN = this.alpha / this.size;
    const beta = this.beta;

    // Cache the un-exponentiated normalizer s_i = k + (alpha/n) * sum(x_j^2) for
    // every element so the backward pass can reuse it without recomputation.
    const sBase = new Float64Array(t.size);

    for (let n = 0; n < batchSize; n++) {
      for (let c = 0; c < C; c++) {
        const cStart = Math.max(0, c - halfSize);
        const cEnd = Math.min(C - 1, c + halfSize);

        for (let s = 0; s < spatialSize; s++) {
          // Sum of squares over nearby channels
          let sqSum = 0;
          for (let j = cStart; j <= cEnd; j++) {
            const idx = n * C * spatialSize + j * spatialSize + s;
            const val = getNumericElement(data, idx);
            sqSum += val * val;
          }

          const sVal = this.k + alphaOverN * sqSum;
          const inIdx = n * C * spatialSize + c * spatialSize + s;
          sBase[inIdx] = sVal;
          const scale = sVal ** beta;

          const val = getNumericElement(data, inIdx);
          outData[inIdx] = val / scale;
        }
      }
    }

    const outTensor = TensorClass.fromTypedArray({
      data: outData,
      shape: shape.slice(),
      dtype: inputDtype,
      device: t.device,
    });

    const requiresGrad = input.requiresGrad;
    if (!requiresGrad) {
      return GradTensor.fromTensor(outTensor, { requiresGrad: false });
    }

    // Custom backward: out_i = x_i * s_i^(-beta), with
    //   s_i = k + (alpha/n) * sum_{j in window(i)} x_j^2.
    // For an input element m:
    //   dL/dx_m = g_m * s_m^(-beta)
    //           + sum_{i: m in window(i)} g_i * x_i * (-beta) * s_i^(-beta-1) * (2*alpha/n) * x_m
    // where g = dL/dout. The window relationship is symmetric (m in window(i)
    // iff i in window(m)), so we iterate i over the window of m.
    const inputRef = input;
    const out = GradTensor.create({
      tensor: outTensor,
      requiresGrad: true,
      prev: [inputRef],
      backward: () => {
        const go = out.grad;
        if (go === null) {
          return;
        }
        const goData = go.data;
        if (Array.isArray(goData) || goData instanceof BigInt64Array) {
          throw new DTypeError("LocalResponseNorm requires numeric non-bigint gradient");
        }
        const goStrides = computeStrides(go.shape);
        const gradOut = new Float64Array(t.size);
        const twoAlphaOverN = 2 * alphaOverN;

        for (let n = 0; n < batchSize; n++) {
          for (let m = 0; m < C; m++) {
            const cStart = Math.max(0, m - halfSize);
            const cEnd = Math.min(C - 1, m + halfSize);

            for (let s = 0; s < spatialSize; s++) {
              const mIdx = n * C * spatialSize + m * spatialSize + s;
              const xm = getNumericElement(data, mIdx);

              // Direct path: contribution from out_m itself.
              const gm = getNumericElement(
                goData,
                offsetFromFlatIndex(mIdx, goStrides, go.strides, go.offset)
              );
              const sm = sBase[mIdx] ?? 1;
              let acc = gm * sm ** -beta;

              // Cross-channel path: every out_i whose window includes m.
              for (let i = cStart; i <= cEnd; i++) {
                const iIdx = n * C * spatialSize + i * spatialSize + s;
                const gi = getNumericElement(
                  goData,
                  offsetFromFlatIndex(iIdx, goStrides, go.strides, go.offset)
                );
                const xi = getNumericElement(data, iIdx);
                const si = sBase[iIdx] ?? 1;
                acc += gi * xi * -beta * si ** (-beta - 1) * twoAlphaOverN * xm;
              }

              gradOut[mIdx] = acc;
            }
          }
        }

        const gradCtor = dtypeToTypedArrayCtor(inputDtype);
        const gradData = new gradCtor(t.size);
        if (gradData instanceof BigInt64Array) {
          throw new DTypeError("Unexpected BigInt64Array for LocalResponseNorm gradient");
        }
        for (let i = 0; i < t.size; i++) {
          gradData[i] = gradOut[i] ?? 0;
        }
        inputRef.accumulateGrad(
          TensorClass.fromTypedArray({
            data: gradData,
            shape: shape.slice(),
            dtype: inputDtype,
            device: t.device,
          })
        );
      },
    });

    return out;
  }

  override toString(): string {
    return `LocalResponseNorm(size=${this.size}, alpha=${this.alpha}, beta=${this.beta}, k=${this.k})`;
  }
}
