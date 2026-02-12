import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import {
  add,
  dot,
  GradTensor,
  mulScalar,
  parameter,
  randn,
  reshape,
  transpose,
  zeros,
} from "../../ndarray";
import { Module } from "../module/Module";

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
 * Uses Kaiming/He initialization: weights ~ N(0, sqrt(2/in_features))
 * Biases are initialized to zeros
 *
 * @example
 * ```ts
 * import { Linear } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Create a linear layer with 20 input features and 30 output features
 * const layer = new Linear(20, 30);
 *
 * // Forward pass
 * const input = tensor([[1, 2, ..., 20]]); // shape: (1, 20)
 * const output = layer.forward(input);     // shape: (1, 30)
 *
 * // Without bias
 * const layerNoBias = new Linear(10, 5, { bias: false });
 * ```
 *
 * References:
 * - PyTorch Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
 * - Xavier/Glorot initialization: http://proceedings.mlr.press/v9/glorot10a.html
 *
 * @category Neural Network Layers
 */
export class Linear extends Module {
  /** Weight matrix of shape (out_features, in_features) */
  private weight: Tensor;
  private weightParam: GradTensor;

  /** Bias vector of shape (out_features,) */
  private bias?: Tensor;
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
   * @param options.dtype - Data type for weights (default: 'float32')
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

    // Initialize weights using Kaiming initialization (He initialization)
    // This initialization is optimal for ReLU activations
    // Standard deviation: sqrt(2 / fan_in) where fan_in = inFeatures
    const stdDev = Math.sqrt(2.0 / inFeatures);

    // Create weight matrix with shape (out_features, in_features)
    // Transposed storage allows efficient matrix multiplication: y = x * W^T
    const weightTensor = randn([outFeatures, inFeatures], {
      dtype: options.dtype ?? "float32",
      device: options.device ?? "cpu",
    });

    // Scale the randomly initialized weights by the computed standard deviation
    // This ensures proper gradient flow during backpropagation
    const scaledWeight = mulScalar(weightTensor, stdDev);
    this.weightParam = parameter(scaledWeight);
    this.weight = this.weightParam.tensor;

    // Register weight as a trainable parameter for optimizer access
    this.registerParameter("weight", this.weightParam);

    // Initialize bias to zeros if enabled (common practice)
    // Bias allows the layer to shift the output independently of input
    if (this.useBias) {
      // Bias has shape (out_features,) - one value per output neuron
      const biasTensor = zeros([outFeatures], {
        dtype: options.dtype ?? "float32",
        device: options.device ?? "cpu",
      });
      this.biasParam = parameter(biasTensor);
      this.bias = this.biasParam.tensor;
      // Register bias as a trainable parameter
      this.registerParameter("bias", this.biasParam);
    }
  }

  /**
   * Forward pass: compute y = x * W^T + b
   *
   * @param input - Input tensor of shape (*, in_features)
   * @returns Output tensor of shape (*, out_features)
   * @throws {ShapeError} If input shape is invalid
   * @throws {DTypeError} If input dtype is unsupported
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    const inputTensor = input instanceof GradTensor ? input.tensor : input;

    if (inputTensor.dtype === "string") {
      throw new DTypeError("Linear layer does not support string dtype");
    }

    // Validate input dimensionality - must be at least 1D
    // 0D (scalar) inputs are not valid for linear transformations
    if (inputTensor.ndim < 1) {
      throw new ShapeError(`Linear layer expects at least 1D input; got ndim=${inputTensor.ndim}`);
    }

    // Extract the last dimension size (number of features)
    // For input shape (batch, seq_len, features), this gets 'features'
    const inputFeatures = inputTensor.shape[inputTensor.shape.length - 1] ?? 0;

    // Validate that input features match the layer's expected input size
    if (inputFeatures !== this.inFeatures) {
      throw new ShapeError(
        `Linear layer expects ${this.inFeatures} input features; got ${inputFeatures}`
      );
    }

    // Compute the linear transformation: y = x * W^T + b
    // Weight is stored as (out_features, in_features), so we transpose it
    // This allows efficient computation: (batch, in_features) @ (in_features, out_features)

    // Check if input is a 1D vector (no batch dimension)
    const isVectorInput = inputTensor.ndim === 1;

    // Calculate total batch size (handles multi-dimensional batches)
    // For shape (batch, seq, features): batchSize = batch * seq
    const batchSize = inputTensor.size / this.inFeatures;

    // Reshape input to 2D: (batchSize, inFeatures) for matrix multiplication
    const outputShape = isVectorInput
      ? [this.outFeatures]
      : [...inputTensor.shape.slice(0, -1), this.outFeatures];

    if (input instanceof GradTensor) {
      const input2d = input.reshape([batchSize, this.inFeatures]);
      const output2d = input2d.matmul(this.weightParam.transpose());
      let output = output2d.reshape(outputShape);
      if (this.useBias && this.biasParam) {
        output = output.add(this.biasParam);
      }
      return output;
    }

    const input2d = reshape(inputTensor, [batchSize, this.inFeatures]);

    // Perform matrix multiplication: (batchSize, inFeatures) @ (inFeatures, outFeatures)
    // Result shape: (batchSize, outFeatures)
    const output2d = dot(input2d, transpose(this.weight));

    // Restore original batch shape, replacing last dimension with outFeatures
    // If input was 1D, output should be 1D; if (batch, features), output is (batch, outFeatures)
    const output = reshape(output2d, outputShape);

    // Add bias term if enabled
    if (this.useBias && this.bias) {
      // Broadcasting automatically handles all batch dimensions
      // Bias shape (outFeatures,) broadcasts to (..., outFeatures)
      return add(output, this.bias);
    }

    // Return output without bias
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
    return this.weight;
  }

  /**
   * Get the bias vector.
   *
   * @returns Bias tensor of shape (out_features,) or undefined if no bias
   */
  getBias(): Tensor | undefined {
    return this.bias;
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
