/**
 * Normalization layers: BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm,
 * InstanceNorm1d/2d/3d, RMSNorm and LocalResponseNorm.
 *
 * Layers with parameters or running statistics compute in the layer dtype (the `dtype`
 * option, or the global default dtype, `float32` unless changed) and cast the input to
 * it, so a `float64` or integer input works with the default `float32` parameters.
 * `LocalResponseNorm` has no parameters: it keeps a `float32` or `float64` input dtype and
 * converts other inputs to its `dtype` option.
 *
 * A `GradTensor` input gives a `GradTensor`. A plain `Tensor` input gives a `GradTensor`
 * that tracks the parameters while they require grad and gradient tracking is on, and a
 * plain `Tensor` otherwise (inside `noGrad()` or with frozen or absent parameters).
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-normalization | Deepbox Normalization & Dropout}
 */

import {
  type DType,
  DTypeError,
  ensureNumericDType,
  getConfig,
  InvalidParameterError,
  ShapeError,
} from "../../core";
import {
  type AnyTensor,
  customOp,
  GradTensor,
  noGrad,
  ones,
  parameter,
  varianceGrad,
  zeros,
} from "../../ndarray";
import { readNumbers } from "../../ndarray/ops/_internal";
import { isContiguous } from "../../ndarray/tensor/strides";
import { Tensor as TensorClass } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";
import { allPlain, settle } from "./_shared";

type FloatDType = "float32" | "float64";

function asGrad(x: AnyTensor): GradTensor {
  return GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);
}

function resolveLayerDtype(dtype: FloatDType | undefined): FloatDType {
  if (dtype === undefined) {
    const configured = getConfig().defaultDtype;
    return configured === "float64" ? "float64" : "float32";
  }
  const resolved: string = dtype;
  if (resolved !== "float32" && resolved !== "float64") {
    throw new InvalidParameterError("dtype must be 'float32' or 'float64'", "dtype", dtype);
  }
  return resolved;
}

function validateEps(eps: number): number {
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new InvalidParameterError("eps must be a positive number", "eps", eps);
  }
  return eps;
}

/**
 * Dtype the layer computes in: the input dtype when it is float32 or float64,
 * otherwise the layer's own dtype.
 */
function computeDtypeFor(inputDtype: DType, layerDtype: FloatDType): FloatDType {
  return inputDtype === "float64" || inputDtype === "float32" ? inputDtype : layerDtype;
}

function normalizeShapeOption(
  normalizedShape: number | readonly number[],
  requirePositive: boolean
): readonly number[] {
  const shape =
    typeof normalizedShape === "number" ? [normalizedShape] : Array.from(normalizedShape);
  if (shape.length === 0) {
    throw new InvalidParameterError(
      "normalizedShape must contain at least one dimension",
      "normalizedShape",
      normalizedShape
    );
  }
  if (requirePositive) {
    for (const dim of shape) {
      if (!Number.isFinite(dim) || dim <= 0 || Math.trunc(dim) !== dim) {
        throw new InvalidParameterError(
          "All dimensions in normalizedShape must be positive integers",
          "normalizedShape",
          normalizedShape
        );
      }
    }
  }
  return shape;
}

/**
 * Check that the trailing dimensions of `inputShape` equal `normShape` and
 * return the index where the normalized dimensions start.
 */
function matchTrailingShape(inputShape: readonly number[], normShape: readonly number[]): number {
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
  return suffixStart;
}

/**
 * Return `input` itself when it is contiguous. Otherwise materialize a
 * contiguous copy while preserving the autograd graph: multiplying by 1
 * produces a fresh contiguous tensor with a proper backward node, unlike a
 * raw `fromTensor` copy which would detach.
 */
function materialize(input: GradTensor, dtype: FloatDType): GradTensor {
  if (isContiguous(input.tensor.shape, input.tensor.strides)) {
    return input;
  }
  return input.mul(GradTensor.scalar(1, { dtype }));
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
  protected readonly layerDtype: FloatDType;

  protected gamma?: GradTensor;
  protected beta?: GradTensor;

  private readonly name: string;

  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
      readonly dtype?: FloatDType;
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
    this.eps = validateEps(options.eps ?? 1e-5);
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
    this.layerDtype = resolveLayerDtype(options.dtype);
    const dtypeOpt = { dtype: this.layerDtype };

    if (this.affine) {
      this.gamma = parameter(ones([numFeatures], dtypeOpt));
      this.beta = parameter(zeros([numFeatures], dtypeOpt));
      this.registerParameter("weight", this.gamma);
      this.registerParameter("bias", this.beta);
    }

    if (this.trackRunningStats) {
      this.registerBuffer("running_mean", zeros([numFeatures], dtypeOpt));
      this.registerBuffer("running_var", ones([numFeatures], dtypeOpt));
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

  /**
   * Current tensor registered under a buffer name. Reading through the module
   * keeps the layer in sync when `loadStateDict` or `to(device)` touches the
   * buffers.
   */
  private bufferTensor(name: "running_mean" | "running_var"): TensorClass {
    for (const [bufferName, buffer] of this.namedBuffers("", false)) {
      if (bufferName === name) return buffer;
    }
    throw new InvalidParameterError(
      `${this.name} has no ${name} buffer`,
      "trackRunningStats",
      false
    );
  }

  forward(x: GradTensor): GradTensor;
  forward(x: TensorClass): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const raw = asGrad(x);

    if (raw.dtype === "string") {
      throw new DTypeError(`${this.name} does not support string dtype`);
    }

    this.validateInputNdims(raw);

    const nFeatures = raw.shape[1] ?? 0;
    if (nFeatures !== this.numFeatures) {
      throw new ShapeError(`Expected ${this.numFeatures} channels, got ${nFeatures}`);
    }

    const dtype = this.layerDtype;
    const input = raw.astype(dtype);
    const flat = this.flattenForStats(input);

    const useBatchStats = this.training || !this.trackRunningStats;

    let mean: GradTensor;
    let varVal: GradTensor;

    if (useBatchStats) {
      const n = flat.shape[0] ?? 0;
      if (n === 0) {
        throw new InvalidParameterError(
          "BatchNorm requires at least one element",
          "input",
          input.shape
        );
      }
      if (n === 1) {
        throw new InvalidParameterError(
          `${this.name} needs more than one value per channel to compute batch statistics; ` +
            `got input shape [${input.shape}]. Use a larger batch, or call eval() with ` +
            "trackRunningStats enabled to use the running statistics.",
          "input",
          input.shape
        );
      }

      mean = flat.mean(0);
      varVal = varianceGrad(flat, 0, 0);

      if (this.trackRunningStats) {
        noGrad(() => {
          const unbiasedVar = varianceGrad(flat, 0, 1);
          const m = this.momentum;
          const oneMinusM = GradTensor.scalar(1 - m, { dtype });
          const mScalar = GradTensor.scalar(m, { dtype });

          const prevMean = this.bufferTensor("running_mean");
          const prevVar = this.bufferTensor("running_var");
          const meanBase = GradTensor.fromTensor(prevMean.astype(dtype));
          const varBase = GradTensor.fromTensor(prevVar.astype(dtype));

          const newMean = meanBase.mul(oneMinusM).add(mean.mul(mScalar));
          const newVar = varBase.mul(oneMinusM).add(unbiasedVar.mul(mScalar));
          this.registerBuffer("running_mean", newMean.tensor.astype(prevMean.dtype));
          this.registerBuffer("running_var", newVar.tensor.astype(prevVar.dtype));
        });
      }
    } else {
      mean = GradTensor.fromTensor(this.bufferTensor("running_mean").astype(dtype));
      varVal = GradTensor.fromTensor(this.bufferTensor("running_var").astype(dtype));
    }

    const bcastShape = this.broadcastShape(nFeatures, input);
    const meanB = mean.reshape(bcastShape);
    const varB = varVal.reshape(bcastShape);

    const epsTensor = GradTensor.scalar(this.eps, { dtype });
    const denom = varB.add(epsTensor).sqrt();
    let out = input.sub(meanB).div(denom);

    if (this.affine && this.gamma && this.beta) {
      const gammaB = this.gamma.astype(dtype).reshape(bcastShape);
      const betaB = this.beta.astype(dtype).reshape(bcastShape);
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
 * During training, uses batch statistics (biased variance for the
 * normalization, unbiased variance for the running estimate). During
 * evaluation, uses running statistics unless `trackRunningStats=false`, in
 * which case batch statistics are always used. Batch statistics need more than
 * one value per channel, otherwise an {@link InvalidParameterError} is thrown.
 *
 * Accepts `(N, C)` or `(N, C, L)` input.
 *
 * @example
 * ```ts
 * import { BatchNorm1d } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const bn = new BatchNorm1d(3);
 * const x = tensor([[1, 2, 3], [4, 5, 6]]);
 * const y = bn.forward(x);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-normalization | Deepbox Normalization & Dropout}
 */
export class BatchNorm1d extends _BatchNorm {
  /**
   * @param numFeatures - Number of channels `C`
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.momentum - Running statistics update factor in [0, 1] (default: 0.1)
   * @param options.affine - Learn a per-channel scale and shift (default: true)
   * @param options.trackRunningStats - Keep running mean and variance (default: true)
   * @param options.dtype - Dtype of the parameters and running statistics (default: the global default dtype)
   */
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
      readonly dtype?: FloatDType;
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
 * const ln = new LayerNorm(3);
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
  private readonly layerDtype: FloatDType;

  private gamma?: GradTensor;
  private beta?: GradTensor;

  /**
   * @param normalizedShape - Size of the trailing dimension(s) to normalize over
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.elementwiseAffine - Learn a scale (`weight`) and shift (`bias`) (default: true)
   * @param options.bias - With `elementwiseAffine`, also learn the shift (default: true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    normalizedShape: number | readonly number[],
    options: {
      readonly eps?: number;
      readonly elementwiseAffine?: boolean;
      readonly bias?: boolean;
      readonly dtype?: FloatDType;
    } = {}
  ) {
    super();
    this.normalizedShape = normalizeShapeOption(normalizedShape, true);
    this.eps = validateEps(options.eps ?? 1e-5);
    this.elementwiseAffine = options.elementwiseAffine ?? true;
    this.layerDtype = resolveLayerDtype(options.dtype);
    const dtypeOpt = { dtype: this.layerDtype };

    if (this.elementwiseAffine) {
      this.gamma = parameter(ones(this.normalizedShape, dtypeOpt));
      this.registerParameter("weight", this.gamma);
      if (options.bias ?? true) {
        this.beta = parameter(zeros(this.normalizedShape, dtypeOpt));
        this.registerParameter("bias", this.beta);
      }
    }
  }

  forward(x: GradTensor): GradTensor;
  forward(x: TensorClass): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const raw = asGrad(x);

    if (raw.dtype === "string") {
      throw new DTypeError("LayerNorm does not support string dtype");
    }

    // Without affine parameters there is nothing to match, so keep the input float dtype.
    const dtype = this.gamma ? this.layerDtype : computeDtypeFor(raw.dtype, this.layerDtype);
    const workingInput = materialize(raw.astype(dtype), dtype);

    // The input shape must end with normalizedShape.
    const inputShape = workingInput.shape;
    const normShape = this.normalizedShape;
    const suffixStart = matchTrailingShape(inputShape, normShape);

    // Reshape input to (..., Product(normShape)) and reduce over the last dimension.
    const outerDims = inputShape.slice(0, suffixStart);
    const normSize = normShape.reduce((a, b) => a * b, 1);
    const inputReshaped = workingInput.reshape([...outerDims, normSize]);

    // Keep dims on the mean to facilitate broadcasting (..., 1)
    const mean = inputReshaped.mean(-1, true);
    // Biased variance (population, ddof=0) is used for normalization, matching PyTorch.
    // varianceGrad reduces the axis without keepdims, so reshape it back to the
    // kept-dims shape of `mean` for broadcasting.
    const varVal = varianceGrad(inputReshaped, -1, 0);
    const varReshaped = varVal.reshape(mean.shape);

    const epsTensor = GradTensor.scalar(this.eps, { dtype });
    const denom = varReshaped.add(epsTensor).sqrt();
    const normalizedReshaped = inputReshaped.sub(mean).div(denom);

    let out = normalizedReshaped.reshape(inputShape);

    // gamma/beta have shape normShape, which matches the trailing dims of the input.
    if (this.gamma) {
      out = out.mul(this.gamma.astype(dtype));
    }
    if (this.beta) {
      out = out.add(this.beta.astype(dtype));
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
 * Accepts `(N, C, *)` input with any number of trailing spatial dimensions.
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
  private readonly layerDtype: FloatDType;

  private gamma?: GradTensor;
  private beta?: GradTensor;

  /**
   * @param numGroups - Number of groups the channels are split into
   * @param numChannels - Number of channels `C` (must be divisible by `numGroups`)
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.affine - Learn a per-channel scale and shift (default: true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    numGroups: number,
    numChannels: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
      readonly dtype?: FloatDType;
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
    this.eps = validateEps(options.eps ?? 1e-5);
    this.affine = options.affine ?? true;
    this.layerDtype = resolveLayerDtype(options.dtype);
    const dtypeOpt = { dtype: this.layerDtype };

    if (this.affine) {
      this.gamma = parameter(ones([numChannels], dtypeOpt));
      this.beta = parameter(zeros([numChannels], dtypeOpt));
      this.registerParameter("weight", this.gamma);
      this.registerParameter("bias", this.beta);
    }
  }

  forward(x: GradTensor): GradTensor;
  forward(x: TensorClass): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const raw = asGrad(x);

    if (raw.dtype === "string") {
      throw new DTypeError("GroupNorm does not support string dtype");
    }

    // Input: (N, C, *) where * is any number of spatial dims
    if (raw.ndim < 2) {
      throw new ShapeError(`GroupNorm expects at least 2D input; got ${raw.ndim}D`);
    }

    const nChannels = raw.shape[1] ?? 0;
    if (nChannels !== this.numChannels) {
      throw new ShapeError(`Expected ${this.numChannels} channels, got ${nChannels}`);
    }

    // Without affine parameters there is nothing to match, so keep the input float dtype.
    const dtype = this.gamma ? this.layerDtype : computeDtypeFor(raw.dtype, this.layerDtype);
    const input = materialize(raw.astype(dtype), dtype);

    const batch = input.shape[0] ?? 0;
    const channelsPerGroup = this.numChannels / this.numGroups;

    // Reshape to (N, G, C/G * spatial) then normalize over the last axis
    const spatialDims = input.shape.slice(2);
    const spatialSize = spatialDims.reduce((a, b) => a * b, 1);
    const groupShape = [batch, this.numGroups, channelsPerGroup * spatialSize];

    const reshaped = input.reshape(groupShape);
    const mean = reshaped.mean(-1, true);
    const varVal = varianceGrad(reshaped, -1, 0);
    const varReshaped = varVal.reshape(mean.shape);

    const epsTensor = GradTensor.scalar(this.eps, { dtype });
    const denom = varReshaped.add(epsTensor).sqrt();
    const normalized = reshaped.sub(mean).div(denom);

    let out = normalized.reshape(input.shape);

    if (this.affine && this.gamma && this.beta) {
      // gamma/beta shape: (C,) -> broadcast to (1, C, 1, 1, ...)
      const broadcastShape = [1, this.numChannels, ...spatialDims.map(() => 1)];
      const gammaB = this.gamma.astype(dtype).reshape(broadcastShape);
      const betaB = this.beta.astype(dtype).reshape(broadcastShape);
      out = out.mul(gammaB).add(betaB);
    }

    return out;
  }

  /** Per-channel scale (`gamma`), or `undefined` when `affine` is false. */
  get weight(): GradTensor | undefined {
    return this.gamma;
  }

  /** Per-channel shift (`beta`), or `undefined` when `affine` is false. */
  get bias(): GradTensor | undefined {
    return this.beta;
  }

  override toString(): string {
    return `GroupNorm(${this.numGroups}, ${this.numChannels}, eps=${this.eps}, affine=${this.affine})`;
  }
}

/**
 * Shared implementation of the instance normalization layers.
 *
 * Instance normalization is group normalization with one group per channel.
 * Subclasses fix which input ranks are accepted; the lowest accepted rank is
 * the unbatched `(C, *)` layout.
 */
abstract class _InstanceNorm extends Module {
  private readonly groupNorm: GroupNorm;
  private readonly name: string;
  private readonly batchedNdim: number;
  private readonly allowUnbatched: boolean;

  constructor(
    name: string,
    batchedNdim: number,
    allowUnbatched: boolean,
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
      readonly dtype?: FloatDType;
    }
  ) {
    super();
    this.name = name;
    this.batchedNdim = batchedNdim;
    this.allowUnbatched = allowUnbatched;
    // InstanceNorm = GroupNorm where numGroups == numChannels
    this.groupNorm = new GroupNorm(numFeatures, numFeatures, options);
    this.registerModule("group_norm", this.groupNorm);
  }

  forward(x: GradTensor): GradTensor;
  forward(x: TensorClass): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = asGrad(x);
    const ndim = input.ndim;
    const unbatched = this.allowUnbatched && ndim === this.batchedNdim - 1;
    // The generic InstanceNorm accepts any (N, C, *) input with at least one spatial dim.
    const valid = this.batchedNdim === 0 ? ndim >= 3 : ndim === this.batchedNdim || unbatched;
    if (!valid) {
      const expected =
        this.batchedNdim === 0
          ? "at least 3D input (N, C, *)"
          : this.allowUnbatched
            ? `${this.batchedNdim}D input (N, C, ...) or ${this.batchedNdim - 1}D input (C, ...)`
            : `${this.batchedNdim}D input (N, C, ...)`;
      throw new ShapeError(`${this.name} expects ${expected}; got ${ndim}D`);
    }

    const batched = unbatched ? input.reshape([1, ...input.shape]) : input;
    const spatialSize = batched.shape.slice(2).reduce((a, b) => a * b, 1);
    if (spatialSize === 1 && batched.size > 0) {
      throw new InvalidParameterError(
        `${this.name} needs more than one spatial element per channel; got input shape [${input.shape}]`,
        "input",
        input.shape
      );
    }
    const out = this.groupNorm.forward(batched);
    return unbatched ? out.reshape(input.shape) : out;
  }

  /**
   * Per-channel scale, or `undefined` when `affine` is false. In a state dict the
   * parameter is stored as `group_norm.weight` (the 1.0.0 layout, kept so existing
   * checkpoints load); this accessor gives the PyTorch-style `layer.weight` view.
   */
  get weight(): GradTensor | undefined {
    return this.groupNorm.weight;
  }

  /** Per-channel shift, or `undefined` when `affine` is false. See {@link _InstanceNorm.weight}. */
  get bias(): GradTensor | undefined {
    return this.groupNorm.bias;
  }

  override toString(): string {
    return `${this.name}(${this.groupNorm.toString()})`;
  }
}

/**
 * Instance Normalization.
 *
 * Normalizes each channel of each sample independently.
 * Equivalent to GroupNorm with numGroups = numChannels.
 * Used in style transfer and image generation.
 *
 * Accepts `(N, C, *)` input with at least one spatial dimension that holds more
 * than one element. Unlike PyTorch, `affine` defaults to `true`; pass
 * `{ affine: false }` to match `torch.nn.InstanceNorm`.
 *
 * @example
 * ```ts
 * const inorm = new InstanceNorm(64); // 64 channels
 * ```
 */
export class InstanceNorm extends _InstanceNorm {
  /**
   * @param numFeatures - Number of channels `C`
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.affine - Learn a per-channel scale and shift (default: true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
      readonly dtype?: FloatDType;
    } = {}
  ) {
    super("InstanceNorm", 0, false, numFeatures, options);
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
 * The default `eps` is 1e-5; PyTorch's `RMSNorm` defaults to the machine
 * epsilon of the input dtype, so pass `eps` explicitly when comparing outputs.
 *
 * @example
 * ```ts
 * const rms = new RMSNorm(512);
 * ```
 */
export class RMSNorm extends Module {
  private readonly normalizedShape: readonly number[];
  private readonly eps: number;
  private readonly layerDtype: FloatDType;
  private gamma?: GradTensor;

  /**
   * @param normalizedShape - Size of the trailing dimension(s) to normalize over
   * @param options.eps - Value added to the mean square for numerical stability (default: 1e-5)
   * @param options.elementwiseAffine - Learn a per-element scale (`weight`) (default: true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    normalizedShape: number | readonly number[],
    options: {
      readonly eps?: number;
      readonly elementwiseAffine?: boolean;
      readonly dtype?: FloatDType;
    } = {}
  ) {
    super();
    this.normalizedShape = normalizeShapeOption(normalizedShape, true);
    this.eps = validateEps(options.eps ?? 1e-5);
    this.layerDtype = resolveLayerDtype(options.dtype);

    if (options.elementwiseAffine ?? true) {
      this.gamma = parameter(ones(this.normalizedShape, { dtype: this.layerDtype }));
      this.registerParameter("weight", this.gamma);
    }
  }

  forward(x: GradTensor): GradTensor;
  forward(x: TensorClass): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const raw = asGrad(x);

    if (raw.dtype === "string") {
      throw new DTypeError("RMSNorm does not support string dtype");
    }

    // Without affine parameters there is nothing to match, so keep the input float dtype.
    const dtype = this.gamma ? this.layerDtype : computeDtypeFor(raw.dtype, this.layerDtype);
    const workingInput = materialize(raw.astype(dtype), dtype);

    const inputShape = workingInput.shape;
    const normShape = this.normalizedShape;
    const suffixStart = matchTrailingShape(inputShape, normShape);

    const outerDims = inputShape.slice(0, suffixStart);
    const normSize = normShape.reduce((a, b) => a * b, 1);
    const inputReshaped = workingInput.reshape([...outerDims, normSize]);

    // RMS = sqrt(mean(x^2) + eps)
    const squared = inputReshaped.mul(inputReshaped);
    const meanSquared = squared.mean(-1, true);
    const epsTensor = GradTensor.scalar(this.eps, { dtype });
    const rms = meanSquared.add(epsTensor).sqrt();

    const normalized = inputReshaped.div(rms);
    let out = normalized.reshape(inputShape);

    if (this.gamma) {
      out = out.mul(this.gamma.astype(dtype));
    }

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
  /**
   * @param numFeatures - Number of channels `C`
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.momentum - Running statistics update factor in [0, 1] (default: 0.1)
   * @param options.affine - Learn a per-channel scale and shift (default: true)
   * @param options.trackRunningStats - Keep running mean and variance (default: true)
   * @param options.dtype - Dtype of the parameters and running statistics (default: the global default dtype)
   */
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
      readonly dtype?: FloatDType;
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
  /**
   * @param numFeatures - Number of channels `C`
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.momentum - Running statistics update factor in [0, 1] (default: 0.1)
   * @param options.affine - Learn a per-channel scale and shift (default: true)
   * @param options.trackRunningStats - Keep running mean and variance (default: true)
   * @param options.dtype - Dtype of the parameters and running statistics (default: the global default dtype)
   */
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
      readonly dtype?: FloatDType;
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
 * An unbatched 2D input (C, L) is also accepted.
 * Equivalent to GroupNorm where numGroups == numChannels.
 * Unlike PyTorch, `affine` defaults to `true`; pass `{ affine: false }` to match `torch.nn.InstanceNorm1d`.
 *
 * @example
 * ```ts
 * const inorm = new InstanceNorm1d(64);
 * // input: (N, 64, L) -> normalized output
 * ```
 */
export class InstanceNorm1d extends _InstanceNorm {
  /**
   * @param numFeatures - Number of channels `C`
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.affine - Learn a per-channel scale and shift (default: true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
      readonly dtype?: FloatDType;
    } = {}
  ) {
    super("InstanceNorm1d", 3, true, numFeatures, options);
  }
}

/**
 * 2D Instance Normalization.
 *
 * Normalizes each channel independently per sample for 4D inputs (N, C, H, W).
 * An unbatched 3D input (C, H, W) is also accepted.
 * Equivalent to GroupNorm where numGroups == numChannels.
 * Unlike PyTorch, `affine` defaults to `true`; pass `{ affine: false }` to match `torch.nn.InstanceNorm2d`.
 *
 * @example
 * ```ts
 * const inorm = new InstanceNorm2d(64);
 * // input: (N, 64, H, W) -> normalized output
 * ```
 */
export class InstanceNorm2d extends _InstanceNorm {
  /**
   * @param numFeatures - Number of channels `C`
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.affine - Learn a per-channel scale and shift (default: true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
      readonly dtype?: FloatDType;
    } = {}
  ) {
    super("InstanceNorm2d", 4, true, numFeatures, options);
  }
}

/**
 * 3D Instance Normalization.
 *
 * Normalizes each channel independently per sample for 5D inputs (N, C, D, H, W).
 * An unbatched 4D input (C, D, H, W) is also accepted.
 * Equivalent to GroupNorm where numGroups == numChannels.
 * Unlike PyTorch, `affine` defaults to `true`; pass `{ affine: false }` to match `torch.nn.InstanceNorm3d`.
 *
 * @example
 * ```ts
 * const inorm = new InstanceNorm3d(64);
 * // input: (N, 64, D, H, W) -> normalized output
 * ```
 */
export class InstanceNorm3d extends _InstanceNorm {
  /**
   * @param numFeatures - Number of channels `C`
   * @param options.eps - Value added to the variance for numerical stability (default: 1e-5)
   * @param options.affine - Learn a per-channel scale and shift (default: true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   */
  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly affine?: boolean;
      readonly dtype?: FloatDType;
    } = {}
  ) {
    super("InstanceNorm3d", 5, true, numFeatures, options);
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
 * where the sum runs over `size` neighbouring channels: channel `i` itself,
 * `floor(size / 2)` channels before it and `floor((size - 1) / 2)` after it.
 * Channels outside the input count as zero, and the divisor is always `size`.
 * This matches `torch.nn.LocalResponseNorm`, including for even `size`.
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
  private readonly layerDtype: FloatDType;

  /**
   * @param size - Number of neighbouring channels used for normalization
   * @param options.alpha - Multiplicative factor (default: 1e-4)
   * @param options.beta - Exponent (default: 0.75)
   * @param options.k - Additive factor (default: 1)
   * @param options.dtype - Compute dtype for non-float32/float64 input (default: the global default dtype)
   */
  constructor(
    size: number,
    options: {
      readonly alpha?: number;
      readonly beta?: number;
      readonly k?: number;
      readonly dtype?: FloatDType;
    } = {}
  ) {
    super();
    this.layerDtype = resolveLayerDtype(options.dtype);
    if (!Number.isInteger(size) || size < 1) {
      throw new InvalidParameterError("size must be a positive integer", "size", size);
    }
    this.size = size;
    this.alpha = options.alpha ?? 1e-4;
    this.beta = options.beta ?? 0.75;
    this.k = options.k ?? 1;
    for (const [name, value] of [
      ["alpha", this.alpha],
      ["beta", this.beta],
      ["k", this.k],
    ] as const) {
      if (!Number.isFinite(value)) {
        throw new InvalidParameterError(`${name} must be a finite number`, name, value);
      }
    }
  }

  forward(x: GradTensor): GradTensor;
  forward(x: TensorClass): TensorClass;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const raw = asGrad(x);

    if (raw.dtype === "string") {
      throw new DTypeError("LocalResponseNorm does not support string dtype");
    }

    if (raw.ndim < 3) {
      throw new ShapeError(
        `LocalResponseNorm expects at least 3D input (N, C, ...); got ${raw.ndim}D`
      );
    }

    const dtype = computeDtypeFor(raw.dtype, this.layerDtype);
    const input = raw.astype(dtype);
    const t = input.tensor;
    const shape = t.shape;
    const C = shape[1] ?? 0;
    const batchSize = shape[0] ?? 0;

    // Compute spatial size (product of dims after channel)
    let spatialSize = 1;
    for (let d = 2; d < shape.length; d++) {
      spatialSize *= shape[d] ?? 1;
    }

    // Zero-based row-major view of the values; honours strides and offsets.
    const data = readNumbers(t, "LocalResponseNorm");

    const front = Math.floor(this.size / 2);
    const back = Math.floor((this.size - 1) / 2);
    const alphaOverN = this.alpha / this.size;
    const beta = this.beta;
    const channelStride = spatialSize;
    const sampleStride = C * spatialSize;

    // Cache the un-exponentiated normalizer s_i = k + (alpha/n) * sum(x_j^2) for
    // every element so the backward pass can reuse it without recomputation.
    const sBase = new Float64Array(t.size);
    const outData = dtype === "float64" ? new Float64Array(t.size) : new Float32Array(t.size);

    for (let n = 0; n < batchSize; n++) {
      for (let c = 0; c < C; c++) {
        const cStart = Math.max(0, c - front);
        const cEnd = Math.min(C - 1, c + back);

        for (let s = 0; s < spatialSize; s++) {
          let sqSum = 0;
          for (let j = cStart; j <= cEnd; j++) {
            const val = data[n * sampleStride + j * channelStride + s] as number;
            sqSum += val * val;
          }

          const sVal = this.k + alphaOverN * sqSum;
          const idx = n * sampleStride + c * channelStride + s;
          sBase[idx] = sVal;
          outData[idx] = (data[idx] as number) / sVal ** beta;
        }
      }
    }

    const outTensor = TensorClass.fromTypedArray({
      data: outData,
      shape: shape.slice(),
      dtype,
      device: t.device,
    });

    // Backward: out_i = x_i * s_i^(-beta), with
    //   s_i = k + (alpha/n) * sum_{j in window(i)} x_j^2.
    // For an input element m:
    //   dL/dx_m = g_m * s_m^(-beta)
    //           + sum_{i: m in window(i)} g_i * x_i * (-beta) * s_i^(-beta-1) * (2*alpha/n) * x_m
    // where g = dL/dout. The window of i is [i - front, i + back], so m is in
    // window(i) exactly when i lies in [m - back, m + front].
    const twoAlphaOverN = 2 * alphaOverN;
    return customOp(outTensor, [
      [
        input,
        (go: TensorClass): TensorClass => {
          const g = readNumbers(go, "LocalResponseNorm");
          const gradOut = dtype === "float64" ? new Float64Array(t.size) : new Float32Array(t.size);

          for (let n = 0; n < batchSize; n++) {
            for (let m = 0; m < C; m++) {
              const iStart = Math.max(0, m - back);
              const iEnd = Math.min(C - 1, m + front);

              for (let s = 0; s < spatialSize; s++) {
                const mIdx = n * sampleStride + m * channelStride + s;
                const xm = data[mIdx] as number;

                // Direct path: contribution from out_m itself.
                let acc = (g[mIdx] as number) * (sBase[mIdx] as number) ** -beta;

                // Cross-channel path: every out_i whose window includes m.
                for (let i = iStart; i <= iEnd; i++) {
                  const iIdx = n * sampleStride + i * channelStride + s;
                  acc +=
                    (g[iIdx] as number) *
                    (data[iIdx] as number) *
                    -beta *
                    (sBase[iIdx] as number) ** (-beta - 1) *
                    twoAlphaOverN *
                    xm;
                }

                gradOut[mIdx] = acc;
              }
            }
          }

          return TensorClass.fromTypedArray({
            data: gradOut,
            shape: shape.slice(),
            dtype,
            device: t.device,
          });
        },
      ],
    ]);
  }

  override toString(): string {
    return `LocalResponseNorm(size=${this.size}, alpha=${this.alpha}, beta=${this.beta}, k=${this.k})`;
  }
}
