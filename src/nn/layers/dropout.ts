import { DTypeError, InvalidParameterError } from "../../core";
import { type AnyTensor, customOp, dropoutGrad, GradTensor, Tensor } from "../../ndarray";
import { readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides } from "../../ndarray/tensor/Tensor";
import { __random } from "../../random/random";
import { Module } from "../module/Module";

/** Materialize a numeric tensor into a contiguous logical-order Float64Array. */
function denseFloat64Drop(t: Tensor): Float64Array {
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

    // Validate dropout probability is in valid range
    if (!Number.isFinite(p) || p < 0 || p >= 1) {
      throw new InvalidParameterError(`Dropout probability must be in [0, 1), got ${p}`, "p", p);
    }

    this.p = p;
  }

  /**
   * Forward pass: apply dropout during training, identity during evaluation.
   *
   * @param input - Input tensor of any shape (Tensor or GradTensor)
   * @returns Output tensor with same shape as input
   */
  forward(input: AnyTensor): GradTensor {
    // Convert to GradTensor if needed
    const inputTensor = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);

    if (inputTensor.dtype === "string") {
      throw new DTypeError("Dropout does not support string dtype");
    }

    // Use vectorized dropout implementation from autograd
    // This handles training/eval mode and mask generation
    return dropoutGrad(inputTensor, this.p, this.training);
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
 * Input is expected to be 4-D: (N, C, H, W).
 * Each channel is either entirely kept or entirely zeroed.
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

  constructor(p = 0.5) {
    super();
    if (!Number.isFinite(p) || p < 0 || p >= 1) {
      throw new InvalidParameterError(`Dropout2d probability must be in [0, 1), got ${p}`, "p", p);
    }
    this.p = p;
  }

  forward(input: AnyTensor): GradTensor {
    const inputTensor = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);

    if (inputTensor.dtype === "string") {
      throw new DTypeError("Dropout2d does not support string dtype");
    }

    if (!this.training || this.p === 0) {
      return inputTensor;
    }

    // Expect 4-D input: (N, C, H, W)
    if (inputTensor.ndim !== 4) {
      throw new InvalidParameterError(
        `Dropout2d expects 4D input (N,C,H,W), got ${inputTensor.ndim}D`,
        "input"
      );
    }

    const [N, C, H, W] = inputTensor.shape as [number, number, number, number];
    const data = inputTensor.data as Float64Array | Float32Array;
    const offset = inputTensor.offset;
    const strides = inputTensor.strides;
    const s0 = strides[0] ?? 0;
    const s1 = strides[1] ?? 0;
    const s2 = strides[2] ?? 0;
    const s3 = strides[3] ?? 0;

    const outData = new Float64Array(N * C * H * W);
    // Per-element multiplier (0 for dropped channels, 1/(1-p) for kept). The
    // op is an elementwise product with this fixed mask, so the backward
    // multiplies the upstream gradient by the same mask.
    const mult = new Float64Array(N * C * H * W);
    const scale = 1 / (1 - this.p);

    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        const m = __random() >= this.p ? scale : 0;
        for (let h = 0; h < H; h++) {
          for (let w = 0; w < W; w++) {
            const inIdx = offset + n * s0 + c * s1 + h * s2 + w * s3;
            const outIdx = n * C * H * W + c * H * W + h * W + w;
            mult[outIdx] = m;
            outData[outIdx] = Number(data[inIdx]) * m;
          }
        }
      }
    }

    const outTensor = Tensor.fromTypedArray({
      data: outData,
      shape: [N, C, H, W],
      dtype: "float64",
      device: inputTensor.device,
    });

    return customOp(outTensor, [
      [
        inputTensor,
        (g: Tensor): Tensor => {
          const gd = denseFloat64Drop(g);
          const gi = new Float64Array(gd.length);
          for (let i = 0; i < gi.length; i++) gi[i] = (gd[i] ?? 0) * (mult[i] ?? 0);
          return Tensor.fromTypedArray({
            data: gi,
            shape: [N, C, H, W],
            dtype: "float64",
            device: inputTensor.device,
          });
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
 *    where a and b are chosen to preserve mean 0 and variance 1
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

    if (!Number.isFinite(p) || p < 0 || p >= 1) {
      throw new InvalidParameterError(
        `AlphaDropout probability must be in [0, 1), got ${p}`,
        "p",
        p
      );
    }

    this.p = p;
  }

  /**
   * Forward pass: apply alpha dropout during training, identity during evaluation.
   *
   * @param input - Input tensor of any shape (Tensor or GradTensor)
   * @returns Output tensor with same shape as input
   */
  forward(input: AnyTensor): GradTensor {
    const inputTensor = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);

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

    const data = inputTensor.data;
    const size = inputTensor.size;
    const outData = new Float64Array(size);
    // Per-element gradient factor: kept elements are affine in the input
    // (d/dval = a); dropped elements are the constant alphaPrime (d/dval = 0).
    const factor = new Float64Array(size);

    const strides = inputTensor.strides;
    const shape = inputTensor.shape;
    const ndim = inputTensor.ndim;
    const offset = inputTensor.offset;

    if (ndim === 0) {
      const val = Number(data[offset]);
      const keep = __random() >= this.p;
      outData[0] = a * (keep ? val : alphaPrime) + b;
      factor[0] = keep ? a : 0;
    } else {
      const idx = new Array<number>(ndim).fill(0);
      let srcOffset = offset;

      for (let count = 0; count < size; count++) {
        const val = Number(data[srcOffset]);
        const keep = __random() >= this.p;
        outData[count] = a * (keep ? val : alphaPrime) + b;
        factor[count] = keep ? a : 0;

        for (let d = ndim - 1; d >= 0; d--) {
          const dim = shape[d] ?? 1;
          const stride = strides[d] ?? 0;
          const nextIdx = (idx[d] ?? 0) + 1;
          idx[d] = nextIdx;
          srcOffset += stride;
          if (nextIdx < dim) break;
          srcOffset -= nextIdx * stride;
          idx[d] = 0;
        }
      }
    }

    const outTensor = Tensor.fromTypedArray({
      data: outData,
      shape: [...inputTensor.shape],
      dtype: "float64",
      device: inputTensor.device,
    });

    return customOp(outTensor, [
      [
        inputTensor,
        (g: Tensor): Tensor => {
          const gd = denseFloat64Drop(g);
          const gi = new Float64Array(size);
          for (let i = 0; i < size; i++) gi[i] = (gd[i] ?? 0) * (factor[i] ?? 0);
          return Tensor.fromTypedArray({
            data: gi,
            shape: [...inputTensor.shape],
            dtype: "float64",
            device: inputTensor.device,
          });
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
