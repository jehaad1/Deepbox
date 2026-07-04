/**
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
import { GradTensor, type Tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { Module } from "../module/Module";

/**
 * Applies Spectral Normalization to a weight parameter of a module.
 *
 * Spectral normalization constrains the spectral norm (largest singular value)
 * of a weight matrix to 1, stabilizing the training of GANs and other networks.
 *
 * **Mathematical Formulation:**
 * ```
 * W_SN = W / σ(W)
 * ```
 * where σ(W) is the largest singular value of W, estimated via power iteration.
 *
 * **Purpose:**
 * - Stabilizes GAN training by controlling the Lipschitz constant
 * - Prevents mode collapse in generative models
 * - More efficient than full SVD: uses power iteration (O(mn) per step)
 *
 * @example
 * ```ts
 * import { SpectralNorm, Linear } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const linear = new Linear(10, 5);
 * const snLinear = new SpectralNorm(linear, 'weight');
 * const input = tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
 * const output = snLinear.forward(input);
 * ```
 *
 * References:
 * - Spectral Normalization for GANs (Miyato et al., 2018)
 *
 * @category Neural Network Layers
 */
export class SpectralNorm extends Module {
  private readonly wrapped: Module;
  private readonly weightName: string;
  private readonly nPowerIterations: number;
  private readonly eps: number;

  // Power iteration vectors
  private u: Float64Array;
  private v: Float64Array;

  // Cached dimensions
  private readonly rows: number;
  private readonly cols: number;

  /**
   * Create a SpectralNorm wrapper around a module.
   *
   * @param module - The module whose weight to normalize
   * @param weightName - Name of the weight parameter (default 'weight')
   * @param nPowerIterations - Number of power iterations per forward pass (default 1)
   * @param eps - Small constant for numerical stability (default 1e-12)
   */
  constructor(module: Module, weightName = "weight", nPowerIterations = 1, eps = 1e-12) {
    super();

    if (nPowerIterations < 1 || !Number.isInteger(nPowerIterations)) {
      throw new InvalidParameterError(
        `nPowerIterations must be a positive integer, got ${nPowerIterations}`,
        "nPowerIterations",
        nPowerIterations
      );
    }

    if (eps <= 0) {
      throw new InvalidParameterError(`eps must be positive, got ${eps}`, "eps", eps);
    }

    this.wrapped = module;
    this.weightName = weightName;
    this.nPowerIterations = nPowerIterations;
    this.eps = eps;

    // Get the weight tensor to determine dimensions
    const weight = this.getWeightTensor();
    if (weight.ndim < 2) {
      throw new ShapeError(
        `SpectralNorm requires weight with at least 2 dimensions, got ${weight.ndim}D`
      );
    }

    // Reshape to 2D: (outFeatures, inFeatures*)
    this.rows = weight.shape[0] ?? 1;
    this.cols = weight.size / this.rows;

    // Initialize u and v with random unit vectors
    this.u = randomUnitVector(this.rows);
    this.v = randomUnitVector(this.cols);
  }

  /**
   * Forward pass: normalize the weight by its spectral norm, then delegate.
   *
   * @param input - Input tensor
   * @returns Output from the wrapped module with spectrally normalized weight
   */
  forward(input: GradTensor | Tensor): GradTensor {
    const weight = this.getWeightTensor();
    const weightData = this.extractWeightData(weight);

    // Power iteration to estimate largest singular value
    for (let i = 0; i < this.nPowerIterations; i++) {
      // v = W^T u / ||W^T u||
      this.v = this.matVecTranspose(weightData, this.u);
      normalizeInPlace(this.v, this.eps);

      // u = W v / ||W v||
      this.u = this.matVec(weightData, this.v);
      normalizeInPlace(this.u, this.eps);
    }

    // σ = u^T W v
    const Wv = this.matVec(weightData, this.v);
    let sigma = 0;
    for (let i = 0; i < this.rows; i++) {
      sigma += (this.u[i] ?? 0) * (Wv[i] ?? 0);
    }
    sigma = Math.max(sigma, this.eps);

    // Normalize weight: W_SN = W / σ
    const normalizedData = new Float64Array(weight.size);
    for (let i = 0; i < weight.size; i++) {
      normalizedData[i] = (weightData[i] ?? 0) / sigma;
    }

    // Set the normalized weight on the wrapped module
    this.setWeightData(weight, normalizedData);

    // Delegate forward to wrapped module
    const inputTensor = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const result = this.wrapped.forward(inputTensor);

    // Restore original weight after forward pass
    this.setWeightData(weight, weightData);

    return GradTensor.isGradTensor(result) ? result : GradTensor.fromTensor(result);
  }

  /**
   * Get the current estimated spectral norm (largest singular value).
   */
  get spectralNormValue(): number {
    const weight = this.getWeightTensor();
    const weightData = this.extractWeightData(weight);
    const Wv = this.matVec(weightData, this.v);
    let sigma = 0;
    for (let i = 0; i < this.rows; i++) {
      sigma += (this.u[i] ?? 0) * (Wv[i] ?? 0);
    }
    return Math.max(sigma, this.eps);
  }

  /**
   * Get the wrapped module.
   */
  get module(): Module {
    return this.wrapped;
  }

  override toString(): string {
    return `SpectralNorm(${this.wrapped.toString()}, name=${this.weightName})`;
  }

  // ─── Internal helpers ───────────────────────────────────────────────

  private getWeightTensor(): Tensor {
    for (const [name, param] of this.wrapped.namedParameters()) {
      if (name === this.weightName) {
        return param.tensor;
      }
    }
    throw new InvalidParameterError(
      `Module does not have a parameter named '${this.weightName}'`,
      "weightName",
      this.weightName
    );
  }

  private extractWeightData(weight: Tensor): Float64Array {
    const data = weight.data;
    const size = weight.size;
    const out = new Float64Array(size);
    if (weight.offset === 0 && !Array.isArray(data) && !(data instanceof BigInt64Array)) {
      for (let i = 0; i < size; i++) {
        out[i] = Number(data[i]);
      }
    } else {
      for (let i = 0; i < size; i++) {
        out[i] = Number(data[weight.offset + i]);
      }
    }
    return out;
  }

  private setWeightData(weight: Tensor, newData: Float64Array): void {
    const data = weight.data;
    if (!Array.isArray(data) && !(data instanceof BigInt64Array)) {
      for (let i = 0; i < newData.length; i++) {
        (data as Float64Array | Float32Array)[weight.offset + i] = newData[i] ?? 0;
      }
    }
  }

  // Matrix-vector product: y = W * x, where W is (rows x cols)
  private matVec(W: Float64Array, x: Float64Array): Float64Array {
    const y = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      let sum = 0;
      for (let j = 0; j < this.cols; j++) {
        sum += (W[i * this.cols + j] ?? 0) * (x[j] ?? 0);
      }
      y[i] = sum;
    }
    return y;
  }

  // Matrix-transpose-vector product: y = W^T * x, where W is (rows x cols)
  private matVecTranspose(W: Float64Array, x: Float64Array): Float64Array {
    const y = new Float64Array(this.cols);
    for (let i = 0; i < this.rows; i++) {
      const xi = x[i] ?? 0;
      for (let j = 0; j < this.cols; j++) {
        y[j] = (y[j] ?? 0) + (W[i * this.cols + j] ?? 0) * xi;
      }
    }
    return y;
  }
}

function randomUnitVector(n: number): Float64Array {
  const v = new Float64Array(n);
  let norm = 0;
  for (let i = 0; i < n; i++) {
    v[i] = __random() - 0.5;
    norm += (v[i] ?? 0) * (v[i] ?? 0);
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < n; i++) {
      v[i] = (v[i] ?? 0) / norm;
    }
  }
  return v;
}

function normalizeInPlace(v: Float64Array, eps: number): void {
  let norm = 0;
  for (let i = 0; i < v.length; i++) {
    norm += (v[i] ?? 0) * (v[i] ?? 0);
  }
  norm = Math.sqrt(norm);
  const invNorm = 1 / Math.max(norm, eps);
  for (let i = 0; i < v.length; i++) {
    v[i] = (v[i] ?? 0) * invNorm;
  }
}
