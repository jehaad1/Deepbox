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
export class BatchNorm1d extends Module {
  private readonly numFeatures: number;
  private readonly eps: number;
  private readonly momentum: number;
  private readonly affine: boolean;
  private readonly trackRunningStats: boolean;

  private gamma?: GradTensor;
  private beta?: GradTensor;
  private runningMean: GradTensor;
  private runningVar: GradTensor;

  constructor(
    numFeatures: number,
    options: {
      readonly eps?: number;
      readonly momentum?: number;
      readonly affine?: boolean;
      readonly trackRunningStats?: boolean;
    } = {}
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
      const gamma = ones([numFeatures]);
      const beta = zeros([numFeatures]);
      this.gamma = parameter(gamma);
      this.beta = parameter(beta);
      this.registerParameter("weight", this.gamma);
      this.registerParameter("bias", this.beta);
    }

    // buffers
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

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    const inputDtype = input.dtype;
    if (inputDtype === "string") {
      throw new DTypeError("BatchNorm1d does not support string dtype");
    }

    if (input.ndim !== 2 && input.ndim !== 3) {
      throw new ShapeError(`BatchNorm1d expects 2D or 3D input; got ndim=${input.ndim}`);
    }

    const nFeatures = input.shape[1] ?? 0;
    if (nFeatures !== this.numFeatures) {
      throw new ShapeError(`Expected ${this.numFeatures} features, got ${nFeatures}`);
    }

    const useBatchStats = this.training || !this.trackRunningStats;

    let mean: GradTensor;
    let varVal: GradTensor;

    // We need to reduce over all dims except channel (dim 1).
    // For 2D (N, C): reduce over 0.
    // For 3D (N, C, L): reduce over 0 and 2.
    // To use single variance op, we reshape to (N*L, C) or (C, N*L).
    // Better to reshape to (BatchTotal, C) and reduce over 0.
    // For 2D: already (N, C).
    // For 3D: permute to (N, L, C) -> reshape (N*L, C).

    let inputReshaped = input;
    if (input.ndim === 3) {
      // (N, C, L) -> (N, L, C) -> (N*L, C)
      const batch = input.shape[0] ?? 0;
      const length = input.shape[2] ?? 0;
      const flat = batch * length;
      const numericInputDtype = ensureNumericDType(inputDtype, "BatchNorm1d");
      inputReshaped = input
        .transpose([0, 2, 1])
        .mul(GradTensor.scalar(1, { dtype: numericInputDtype }))
        .reshape([flat, nFeatures]);
    }

    if (useBatchStats) {
      if (inputReshaped.shape[0] === 0) {
        throw new InvalidParameterError(
          "BatchNorm requires at least one element",
          "input",
          input.shape
        );
      }

      // Compute batch statistics (biased variance for normalization)
      mean = inputReshaped.mean(0);
      varVal = varianceGrad(inputReshaped, 0, 0);

      if (this.trackRunningStats) {
        noGrad(() => {
          // Use unbiased variance (Bessel's correction) for running stats update
          const n = inputReshaped.shape[0] ?? 0;
          const unbiasedVar =
            n > 1 ? varianceGrad(inputReshaped, 0, 1) : varianceGrad(inputReshaped, 0, 0);

          const m = this.momentum;
          const statsDtype = this.runningMean.dtype;
          if (statsDtype === "string") {
            throw new DTypeError("BatchNorm running statistics must be numeric");
          }
          const oneMinusM = GradTensor.scalar(1 - m, { dtype: statsDtype });
          const mScalar = GradTensor.scalar(m, { dtype: statsDtype });

          // Exponential moving average: running = (1 - momentum) * running + momentum * batch
          const newMean = this.runningMean.mul(oneMinusM).add(mean.mul(mScalar));
          const newVar = this.runningVar.mul(oneMinusM).add(unbiasedVar.mul(mScalar));

          // Detach to prevent graph history in running stats
          this.runningMean = GradTensor.fromTensor(newMean.tensor, {
            requiresGrad: false,
          });
          this.runningVar = GradTensor.fromTensor(newVar.tensor, {
            requiresGrad: false,
          });

          // Re-register buffers to keep Module._buffers map in sync
          this.registerBuffer("running_mean", this.runningMean.tensor);
          this.registerBuffer("running_var", this.runningVar.tensor);
        });
      }
    } else {
      mean = this.runningMean;
      varVal = this.runningVar;
    }

    // Normalize: (x - mean) / sqrt(var + eps)
    // We need to broadcast mean/var to input shape.
    // Input: (N, C) or (N, C, L). Mean/Var: (C).
    // For (N, C), broadcasting (C) works (last dim matches).
    // For (N, C, L), we need (1, C, 1).

    let meanBroadcast = mean;
    let varBroadcast = varVal;

    if (input.ndim === 3) {
      meanBroadcast = mean.reshape([1, nFeatures, 1]);
      varBroadcast = varVal.reshape([1, nFeatures, 1]);
    } else {
      meanBroadcast = mean.reshape([1, nFeatures]);
      varBroadcast = varVal.reshape([1, nFeatures]);
    }

    const epsTensor = GradTensor.scalar(this.eps, { dtype: inputDtype });
    const denom = varBroadcast.add(epsTensor).sqrt();
    let out = input.sub(meanBroadcast).div(denom);

    if (this.affine && this.gamma && this.beta) {
      let gammaB = this.gamma;
      let betaB = this.beta;
      if (input.ndim === 3) {
        gammaB = this.gamma.reshape([1, nFeatures, 1]);
        betaB = this.beta.reshape([1, nFeatures, 1]);
      } else {
        gammaB = this.gamma.reshape([1, nFeatures]);
        betaB = this.beta.reshape([1, nFeatures]);
      }
      out = out.mul(gammaB).add(betaB);
    }

    return out;
  }

  override toString(): string {
    return `BatchNorm1d(${this.numFeatures}, eps=${this.eps}, momentum=${this.momentum}, affine=${this.affine})`;
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
      const contiguous = toContiguousTensor(input.tensor);
      workingInput = GradTensor.fromTensor(contiguous, {
        requiresGrad: input.requiresGrad,
      });
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
    const varVal = varianceGrad(inputReshaped, -1, 0); // Biased variance for normalization
    // varianceGrad returns tensor with reduced dim removed?
    // Wait, varianceGrad(..., axis) removes the axis unless keepdims?
    // My implementation of `varianceGrad` calls `mean` and `sum`.
    // `mean(axis)` removes dim unless `keepdims=true`.
    // `varianceGrad` implementation in `autograd`:
    // It doesn't take `keepdims` param. It reduces.
    // So `varVal` has shape (...,).
    // We need to reshape it to (..., 1) for broadcasting.

    const varReshaped = varVal.reshape(mean.shape); // mean has kept dims

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
