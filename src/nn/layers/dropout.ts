/**
 * Dropout layers: Dropout, Dropout2d and AlphaDropout.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-normalization | Deepbox Normalization & Dropout}
 */

import { DeviceError, type DType, DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { type AnyTensor, customOp, dropoutGrad, type GradTensor, Tensor } from "../../ndarray";
import { readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides } from "../../ndarray/tensor/Tensor";
import { __fillUniform } from "../../random/random";
import { Module } from "../module/Module";
import { allPlain, settle, toGradInput } from "./_shared";

/** Materialize a numeric tensor into a contiguous logical-order Float64Array. */
function denseFloat64Drop(t: Tensor): Float64Array {
  if (t.isDeviceTensor) {
    throw new DeviceError(
      `This layer runs on the host and cannot read a tensor stored on device "${t.device}"; ` +
        "move it back with `await tensor.cpu()` first"
    );
  }
  const out = new Float64Array(t.size);
  const data = requireNumericData(t.data, "Dropout");
  const contig = isContiguous(t.shape, t.strides);
  const logical = computeStrides(t.shape);
  for (let i = 0; i < t.size; i++) {
    const off = contig ? t.offset + i : offsetFromFlatIndex(i, logical, t.strides, t.offset);
    out[i] = readAsNumber(data, off);
  }
  return out;
}

function isFloatDType(dtype: DType): boolean {
  return dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
}

/**
 * Dropout rescales survivors by `1 / (1 - p)`, which an integer or bool tensor cannot
 * hold, so those inputs are converted to float32 first. Floating inputs are kept as they are.
 */
function toFloatInput(input: GradTensor): GradTensor {
  return isFloatDType(input.dtype) ? input : input.astype("float32");
}

/** Wrap a float64 result in a tensor of the dtype the layer should return. */
function resultTensor(
  data: Float64Array,
  shape: readonly number[],
  device: Tensor["device"],
  dtype: DType
): Tensor {
  const out = Tensor.fromTypedArray({ data, shape: [...shape], dtype: "float64", device });
  return dtype === "float64" ? out : out.astype(dtype);
}

function validateRate(layer: string, p: number): void {
  if (!Number.isFinite(p) || p < 0 || p >= 1) {
    throw new InvalidParameterError(`${layer} probability must be in [0, 1), got ${p}`, "p", p);
  }
}

/**
 * Applies Dropout regularization during training.
 *
 * **Mathematical Formulation:**
 * During training:
 * ```
 * y = x * mask / (1 - p)
 * ```
 * where mask is a binary tensor with probability (1-p) of being 1.
 *
 * During evaluation:
 * ```
 * y = x
 * ```
 *
 * **Purpose:**
 * - Prevents overfitting by randomly zeroing elements during training
 * - Forces network to learn redundant representations
 * - Improves generalization performance
 *
 * **Scaling:**
 * The output is scaled by 1/(1-p) during training to maintain expected value.
 * This is called "inverted dropout" and eliminates the need for scaling during inference.
 *
 * @example
 * ```ts
 * import { Dropout } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const dropout = new Dropout(0.5); // Drop 50% of neurons
 * const input = tensor([[1, 2, 3, 4]]);
 *
 * // Training mode: randomly zeros ~50% of elements
 * dropout.train();
 * const output = dropout.forward(input);
 *
 * // Evaluation mode: passes input unchanged
 * dropout.eval();
 * const output2 = dropout.forward(input); // Same as input
 * ```
 *
 * References:
 * - Dropout paper: https://jmlr.org/papers/v15/srivastava14a.html
 * - Deepbox Dropout: https://deepbox.dev/docs/nn-normalization
 *
 * @category Neural Network Layers
 */
export class Dropout extends Module {
  /** Probability of an element being zeroed (dropout rate) */
  private readonly p: number;

  /**
   * Create a new Dropout layer.
   *
   * @param p - Probability of an element being zeroed (0 <= p < 1)
   * @throws {InvalidParameterError} If p is not in valid range [0, 1)
   */
  constructor(p = 0.5) {
    super();

    validateRate("Dropout", p);
    this.p = p;
  }

  /**
   * Forward pass: apply dropout during training, identity during evaluation.
   *
   * @param input - Input tensor of any shape (Tensor or GradTensor)
   * @returns Output tensor with same shape as input. Integer and bool inputs become float32
   *   when dropout is active, because the survivors are scaled by `1 / (1 - p)`.
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    // Convert to GradTensor if needed
    const inputTensor = toGradInput(input);

    if (inputTensor.dtype === "string") {
      throw new DTypeError("Dropout does not support string dtype");
    }

    // Use vectorized dropout implementation from autograd
    // This handles training/eval mode and mask generation
    const active = this.training && this.p > 0;
    return dropoutGrad(active ? toFloatInput(inputTensor) : inputTensor, this.p, this.training);
  }

  /**
   * Get string representation of the layer.
   *
   * @returns String representation with dropout probability
   */
  override toString(): string {
    return `Dropout(p=${this.p})`;
  }

  /**
   * Get the dropout probability.
   */
  get dropoutRate(): number {
    return this.p;
  }
}

/**
 * Applies 2D channel-wise Dropout during training.
 *
 * Randomly zeros entire channels (feature maps) of the input.
 * Input is expected to be 4-D: (N, C, H, W), or 3-D (C, H, W) for a single sample.
 * Each channel is either entirely kept or entirely zeroed, and kept channels are
 * scaled by `1 / (1 - p)`. In evaluation mode (or with `p = 0`) the input is returned as is.
 *
 * @example
 * ```ts
 * const dropout2d = new Dropout2d(0.5);
 * const input = tensor([[[[1,2],[3,4]], [[5,6],[7,8]]]]);
 * dropout2d.train();
 * const output = dropout2d.forward(input);
 * ```
 *
 * @category Neural Network Layers
 */
export class Dropout2d extends Module {
  private readonly p: number;

  /**
   * @param p - Probability of a channel being zeroed (0 <= p < 1)
   * @throws {InvalidParameterError} If p is not in [0, 1)
   */
  constructor(p = 0.5) {
    super();
    validateRate("Dropout2d", p);
    this.p = p;
  }

  /**
   * Forward pass: zero whole channels during training, identity during evaluation.
   *
   * @param input - Tensor of shape `(N, C, H, W)`, or `(C, H, W)` for a single sample
   * @returns Tensor with the same shape as `input`. Integer and bool inputs become float32
   *   when dropout is active.
   * @throws {ShapeError} If dropout is active and the input is not 3-D or 4-D
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const inputTensor = toGradInput(input);

    if (inputTensor.dtype === "string") {
      throw new DTypeError("Dropout2d does not support string dtype");
    }

    if (!this.training || this.p === 0) {
      return inputTensor;
    }

    // Expect 4-D input (N, C, H, W); a 3-D input is one sample (C, H, W).
    if (inputTensor.ndim !== 4 && inputTensor.ndim !== 3) {
      throw new ShapeError(
        `Dropout2d expects 4D input (N,C,H,W) or 3D input (C,H,W), got ${inputTensor.ndim}D`
      );
    }

    const shape = inputTensor.shape;
    const lead = inputTensor.ndim === 4 ? (shape[0] ?? 0) : 1;
    const C = shape[inputTensor.ndim - 3] ?? 0;
    const planeSize = (shape[inputTensor.ndim - 2] ?? 0) * (shape[inputTensor.ndim - 1] ?? 0);
    const dtype = isFloatDType(inputTensor.dtype) ? inputTensor.dtype : "float32";

    const dense = denseFloat64Drop(inputTensor.tensor);
    const outData = new Float64Array(dense.length);
    // Per-element multiplier (0 for dropped channels, 1/(1-p) for kept). The
    // op is an elementwise product with this fixed mask, so the backward
    // multiplies the upstream gradient by the same mask.
    const mult = new Float64Array(dense.length);
    const scale = 1 / (1 - this.p);

    // One draw per (sample, channel), in row-major order.
    const draws = new Float64Array(lead * C);
    __fillUniform(draws, draws.length);

    for (let plane = 0; plane < draws.length; plane++) {
      const m = (draws[plane] as number) >= this.p ? scale : 0;
      const base = plane * planeSize;
      for (let i = 0; i < planeSize; i++) {
        mult[base + i] = m;
        outData[base + i] = (dense[base + i] as number) * m;
      }
    }

    const device = inputTensor.device;
    const outTensor = resultTensor(outData, shape, device, dtype);

    return customOp(outTensor, [
      [
        inputTensor,
        (g: Tensor): Tensor => {
          const gd = denseFloat64Drop(g);
          const gi = new Float64Array(gd.length);
          for (let i = 0; i < gi.length; i++) gi[i] = (gd[i] ?? 0) * (mult[i] ?? 0);
          return resultTensor(gi, shape, device, dtype);
        },
      ],
    ]);
  }

  override toString(): string {
    return `Dropout2d(p=${this.p})`;
  }

  get dropoutRate(): number {
    return this.p;
  }
}

/**
 * Applies Alpha Dropout during training, designed for SELU-activated networks.
 *
 * Unlike standard Dropout which zeros elements, AlphaDropout replaces dropped
 * elements with the negative saturation value of SELU, then applies an affine
 * transformation to maintain self-normalizing properties.
 *
 * **Mathematical Formulation:**
 * During training:
 * 1. Generate binary mask with keep probability (1 - p)
 * 2. Replace dropped values with α' = -λα ≈ -1.7580993408
 * 3. Apply affine transform: y = a * (x * mask + α' * (1 - mask)) + b
 *    where a = 1 / sqrt((1 - p) * (1 + p * α'²)) and b = -a * α' * p, which keep
 *    mean 0 and variance 1 for standard-normal inputs
 *
 * During evaluation:
 * ```
 * y = x
 * ```
 *
 * **Purpose:**
 * - Maintains the self-normalizing property of SELU networks
 * - Keeps mean ≈ 0 and variance ≈ 1 after dropout
 *
 * @example
 * ```ts
 * import { AlphaDropout, SELU, Linear, Sequential } from 'deepbox/nn';
 *
 * const model = new Sequential([
 *   new Linear(784, 256),
 *   new SELU(),
 *   new AlphaDropout(0.1),
 *   new Linear(256, 10),
 * ]);
 * ```
 *
 * References:
 * - Self-Normalizing Neural Networks (Klambauer et al., 2017)
 *
 * @category Neural Network Layers
 */
export class AlphaDropout extends Module {
  private readonly p: number;

  // SELU fixed-point parameters
  private static readonly ALPHA = 1.6732632423543772;
  private static readonly SCALE = 1.0507009873554805;
  private static readonly ALPHA_PRIME = -AlphaDropout.SCALE * AlphaDropout.ALPHA;

  /**
   * Create a new AlphaDropout layer.
   *
   * @param p - Probability of an element being dropped (0 <= p < 1)
   * @throws {InvalidParameterError} If p is not in valid range [0, 1)
   */
  constructor(p = 0.5) {
    super();

    validateRate("AlphaDropout", p);
    this.p = p;
  }

  /**
   * Forward pass: apply alpha dropout during training, identity during evaluation.
   *
   * @param input - Input tensor of any shape (Tensor or GradTensor)
   * @returns Output tensor with same shape as input. Integer and bool inputs become float32
   *   when dropout is active.
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const inputTensor = toGradInput(input);

    if (inputTensor.dtype === "string") {
      throw new DTypeError("AlphaDropout does not support string dtype");
    }

    if (!this.training || this.p === 0) {
      return inputTensor;
    }

    const alphaPrime = AlphaDropout.ALPHA_PRIME;

    // Compute affine parameters to maintain mean=0, var=1
    const q = 1 - this.p;
    const a = 1 / Math.sqrt(q + alphaPrime * alphaPrime * this.p * q);
    const b = -a * (this.p * alphaPrime);

    const size = inputTensor.size;
    const dtype = isFloatDType(inputTensor.dtype) ? inputTensor.dtype : "float32";
    const dense = denseFloat64Drop(inputTensor.tensor);
    const outData = new Float64Array(size);
    // Per-element gradient factor: kept elements are affine in the input
    // (d/dval = a); dropped elements are the constant alphaPrime (d/dval = 0).
    const factor = new Float64Array(size);

    // One draw per element, in row-major order.
    const draws = new Float64Array(size);
    __fillUniform(draws, size);
    for (let i = 0; i < size; i++) {
      const keep = (draws[i] as number) >= this.p;
      outData[i] = a * (keep ? (dense[i] as number) : alphaPrime) + b;
      factor[i] = keep ? a : 0;
    }

    const shape = inputTensor.shape;
    const device = inputTensor.device;
    const outTensor = resultTensor(outData, shape, device, dtype);

    return customOp(outTensor, [
      [
        inputTensor,
        (g: Tensor): Tensor => {
          const gd = denseFloat64Drop(g);
          const gi = new Float64Array(size);
          for (let i = 0; i < size; i++) gi[i] = (gd[i] ?? 0) * (factor[i] ?? 0);
          return resultTensor(gi, shape, device, dtype);
        },
      ],
    ]);
  }

  override toString(): string {
    return `AlphaDropout(p=${this.p})`;
  }

  get dropoutRate(): number {
    return this.p;
  }
}
