/**
 * Fully connected (dense) layer.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox documentation}
 */

import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import type { AnyTensor, Tensor } from "../../ndarray";
import { add, dot, GradTensor, parameter, reshape, transpose } from "../../ndarray";
import { Module } from "../module/Module";
import { isGradEnabled, resolveLayerDtype, toGradInput, uniformTensor } from "./_shared";

/**
 * Applies a linear transformation to the incoming data: y = xA^T + b
 *
 * This is also known as a fully connected layer or dense layer.
 *
 * **Mathematical Formulation:**
 * ```
 * y = x * W^T + b
 * ```
 *
 * Where:
 * - x is the input tensor of shape (*, in_features)
 * - W is the weight matrix of shape (out_features, in_features)
 * - b is the bias vector of shape (out_features,)
 * - y is the output tensor of shape (*, out_features)
 *
 * **Shape Conventions:**
 * - Input: `(*, in_features)` where `*` means any number of leading dimensions
 *   - 1D: `(in_features)` → Output: `(out_features)`
 *   - 2D: `(batch, in_features)` → Output: `(batch, out_features)`
 *   - 3D: `(batch, seq_len, in_features)` → Output: `(batch, seq_len, out_features)`
 * - The last dimension must equal `in_features`
 * - All leading dimensions are preserved in the output
 *
 * **Parameters:**
 * - `inFeatures`: Size of each input sample
 * - `outFeatures`: Size of each output sample
 * - `bias`: If true, adds a learnable bias to the output
 *
 * **Attributes:**
 * - `weight`: Learnable weights of shape (out_features, in_features)
 * - `bias`: Learnable bias of shape (out_features,) if bias=true
 *
 * **Initialization:**
 * As in PyTorch, weights and biases are drawn from the uniform distribution
 * `U(-1/sqrt(in_features), 1/sqrt(in_features))` (Kaiming uniform with `a = sqrt(5)`
 * for the weights). Seed the global generator with `manualSeed` for reproducible weights.
 *
 * **Input dtype:**
 * The layer computes in its parameter dtype (`float32` unless `dtype` is given, or the
 * global default dtype), so integer, boolean and `float64` inputs all work. For a
 * `GradTensor` input the conversion is differentiable.
 *
 * **Gradient tracking:**
 * With a `GradTensor` input the result is a `GradTensor`. With a plain `Tensor` input the
 * result is a `GradTensor` that tracks the weights while gradient tracking is on and a
 * weight requires grad (the input itself is not tracked), so a training loop needs no
 * wrapping of the data. Inside `noGrad()`, or when every weight is frozen, a plain `Tensor`
 * is returned. `eval()` alone does not switch tracking off.
 *
 * @example
 * ```ts
 * import { Linear } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Create a linear layer with 3 input features and 2 output features
 * const layer = new Linear(3, 2);
 *
 * // Forward pass
 * const input = tensor([[1, 2, 3]]);   // shape: (1, 3)
 * const output = layer.forward(input); // shape: (1, 2)
 *
 * // Without bias
 * const layerNoBias = new Linear(10, 5, { bias: false });
 * ```
 *
 * References:
 * - Deepbox Linear: https://deepbox.dev/docs/nn-layers
 * - He et al., "Delving Deep into Rectifiers" (2015): https://arxiv.org/abs/1502.01852
 *
 * @category Neural Network Layers
 */
export class Linear extends Module {
  /** Weight parameter of shape (out_features, in_features) */
  private weightParam: GradTensor;

  /** Bias parameter of shape (out_features,) */
  private biasParam?: GradTensor;

  /** Number of input features */
  private readonly inFeatures: number;

  /** Number of output features */
  private readonly outFeatures: number;

  /** Whether this layer has a bias */
  private readonly useBias: boolean;

  /**
   * Create a new Linear layer.
   *
   * @param inFeatures - Size of each input sample
   * @param outFeatures - Size of each output sample
   * @param options - Configuration options
   * @param options.bias - If true, add learnable bias (default: true)
   * @param options.dtype - Data type for weights (default: the global default dtype, 'float32')
   * @param options.device - Device to place tensors on (default: 'cpu')
   */
  constructor(
    inFeatures: number,
    outFeatures: number,
    options: {
      readonly bias?: boolean;
      readonly dtype?: "float32" | "float64";
      readonly device?: "cpu" | "webgpu" | "wasm";
    } = {}
  ) {
    // Call parent Module constructor to initialize base class
    super();

    // Validate dimensions
    if (inFeatures <= 0 || !Number.isInteger(inFeatures)) {
      throw new InvalidParameterError(
        "inFeatures must be a positive integer",
        "inFeatures",
        inFeatures
      );
    }
    if (outFeatures <= 0 || !Number.isInteger(outFeatures)) {
      throw new InvalidParameterError(
        "outFeatures must be a positive integer",
        "outFeatures",
        outFeatures
      );
    }

    // Store layer dimensions for validation and access
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    // Default to using bias unless explicitly disabled
    this.useBias = options.bias ?? true;

    // PyTorch default: weight and bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    // (kaiming_uniform with a = sqrt(5) gives the same bound for the weight.)
    const dtype = resolveLayerDtype(options.dtype);
    const device = options.device ?? "cpu";
    const bound = 1 / Math.sqrt(inFeatures);

    // Weight has shape (out_features, in_features); forward computes y = x * W^T.
    this.weightParam = parameter(
      uniformTensor([outFeatures, inFeatures], bound, { dtype, device })
    );

    // Register weight as a trainable parameter for optimizer access
    this.registerParameter("weight", this.weightParam);

    if (this.useBias) {
      // Bias has shape (out_features,) - one value per output neuron
      this.biasParam = parameter(uniformTensor([outFeatures], bound, { dtype, device }));
      this.registerParameter("bias", this.biasParam);
    }
  }

  /**
   * Forward pass: compute y = x * W^T + b
   *
   * Inputs whose dtype differs from the layer's dtype are converted first
   * (a differentiable cast for `GradTensor` inputs).
   *
   * A `GradTensor` input gives a `GradTensor`. A plain `Tensor` input gives a `GradTensor`
   * that tracks the weights when gradient tracking is on and a weight requires grad, and a
   * plain `Tensor` otherwise (inside `noGrad()` or with frozen weights).
   *
   * @param input - Input tensor of shape (*, in_features)
   * @returns Output tensor of shape (*, out_features)
   * @throws {ShapeError} If input shape is invalid
   * @throws {DTypeError} If input dtype is unsupported
   *
   * @example
   * ```ts
   * const layer = new Linear(3, 2);
   * const out = layer.forward(tensor([[1, 2, 3]])); // GradTensor that tracks the weights
   * ```
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    if (input.dtype === "string") {
      throw new DTypeError("Linear layer does not support string dtype");
    }

    // Validate input dimensionality - must be at least 1D
    // 0D (scalar) inputs are not valid for linear transformations
    if (input.ndim < 1) {
      throw new ShapeError(`Linear layer expects at least 1D input; got ndim=${input.ndim}`);
    }

    // Extract the last dimension size (number of features)
    // For input shape (batch, seq_len, features), this gets 'features'
    const inputFeatures = input.shape[input.shape.length - 1] ?? 0;

    // Validate that input features match the layer's expected input size
    if (inputFeatures !== this.inFeatures) {
      throw new ShapeError(
        `Linear layer expects ${this.inFeatures} input features; got ${inputFeatures}`
      );
    }

    // Convert to the layer dtype. `astype` honours strides and offsets, and the
    // GradTensor variant keeps the autograd link to the caller's graph.
    const layerDtype = this.weightParam.tensor.dtype;
    if (input.dtype !== layerDtype) {
      if (layerDtype === "string") {
        throw new DTypeError("Linear layer weight must be numeric");
      }
      input = input.astype(layerDtype);
    }

    // Compute the linear transformation: y = x * W^T + b
    // Weight is stored as (out_features, in_features), so we transpose it
    // This allows efficient computation: (batch, in_features) @ (in_features, out_features)

    // Check if input is a 1D vector (no batch dimension)
    const isVectorInput = input.ndim === 1;

    // Calculate total batch size (handles multi-dimensional batches)
    // For shape (batch, seq, features): batchSize = batch * seq
    const batchSize = input.size / this.inFeatures;

    // Output keeps all leading dimensions and replaces the last with outFeatures
    const outputShape = isVectorInput
      ? [this.outFeatures]
      : [...input.shape.slice(0, -1), this.outFeatures];

    // A plain input is tracked (wrapped as a leaf that does not require grad) only when the
    // weights can receive gradients; otherwise it takes the faster tensor path below.
    const tracked =
      GradTensor.isGradTensor(input) ||
      (isGradEnabled() && (this.weightParam.requiresGrad || !!this.biasParam?.requiresGrad));

    if (tracked) {
      const input2d = toGradInput(input).reshape([batchSize, this.inFeatures]);
      const output2d = input2d.matmul(this.weightParam.transpose());
      let output = output2d.reshape(outputShape);
      if (this.biasParam) {
        output = output.add(this.biasParam);
      }
      return output;
    }

    // Reshape input to 2D: (batchSize, inFeatures) for matrix multiplication
    const plain = GradTensor.isGradTensor(input) ? input.tensor : input;
    const input2d = reshape(plain, [batchSize, this.inFeatures]);

    // Perform matrix multiplication: (batchSize, inFeatures) @ (inFeatures, outFeatures)
    // Result shape: (batchSize, outFeatures)
    const output2d = dot(input2d, transpose(this.weightParam.tensor));

    // Restore original batch shape, replacing last dimension with outFeatures
    const output = reshape(output2d, outputShape);

    // Add bias term if enabled
    if (this.biasParam) {
      // Bias shape (outFeatures,) broadcasts to (..., outFeatures)
      return add(output, this.biasParam.tensor);
    }

    return output;
  }

  /**
   * Get extra representation string for this layer.
   *
   * @returns String representation of layer parameters
   */
  override toString(): string {
    const biasStr = this.useBias ? "bias=true" : "bias=false";
    return `Linear(in_features=${this.inFeatures}, out_features=${this.outFeatures}, ${biasStr})`;
  }

  /**
   * Get the weight matrix.
   *
   * @returns Weight tensor of shape (out_features, in_features)
   */
  getWeight(): Tensor {
    return this.weightParam.tensor;
  }

  /**
   * Get the bias vector.
   *
   * @returns Bias tensor of shape (out_features,) or undefined if no bias
   */
  getBias(): Tensor | undefined {
    return this.biasParam?.tensor;
  }

  /**
   * Get the number of input features.
   */
  get inputSize(): number {
    return this.inFeatures;
  }

  /**
   * Get the number of output features.
   */
  get outputSize(): number {
    return this.outFeatures;
  }
}
