/**
 * Spectral normalization wrapper.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { type AnyTensor, customOp, GradTensor, type Tensor } from "../../ndarray";
import { readNumbers } from "../../ndarray/ops/_internal";
import { Tensor as TensorClass } from "../../ndarray/tensor/Tensor";
import { __random } from "../../random/random";
import { Module } from "../module/Module";
import { allPlain, settle } from "./_shared";

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
 * **Behavior:**
 * - The wrapped module is registered as the child `module`, so `parameters()` and
 *   `stateDict()` include its weights (for example `module.weight`).
 * - The raw weight stays in the wrapped module and is never modified. During
 *   `forward` the module sees the normalized weight, and gradients flow back to
 *   the raw weight through the normalization, treating the power-iteration
 *   vectors as constants (as PyTorch does).
 * - The power iteration vectors `u` and `v` are stored as buffers named
 *   `<weightName>_u` and `<weightName>_v`. They are refined by
 *   `nPowerIterations` steps on each training-mode forward pass and are left
 *   unchanged in evaluation mode. Fifteen warm-up iterations run at construction.
 * - Weights with more than two dimensions are reshaped to `(shape[0], rest)`.
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
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  private readonly wrapped: Module;
  private readonly weightName: string;
  private readonly nPowerIterations: number;
  private readonly eps: number;

  // Power iteration vectors. They are updated in place so the registered
  // buffers (which share their memory) always reflect the current state.
  private readonly u: Float64Array;
  private readonly v: Float64Array;

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
   * @throws {InvalidParameterError} If `nPowerIterations` or `eps` is invalid, or the
   *   module has no parameter called `weightName`
   * @throws {ShapeError} If the weight has fewer than two dimensions or no elements
   * @throws {DTypeError} If the weight is not float32 or float64
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

    if (!Number.isFinite(eps) || eps <= 0) {
      throw new InvalidParameterError(`eps must be positive, got ${eps}`, "eps", eps);
    }

    this.wrapped = module;
    this.weightName = weightName;
    this.nPowerIterations = nPowerIterations;
    this.eps = eps;
    this.registerModule("module", module);

    // Get the weight tensor to determine dimensions
    const weight = this.getWeightParam().tensor;
    if (weight.dtype !== "float32" && weight.dtype !== "float64") {
      throw new DTypeError(
        `SpectralNorm requires a float32 or float64 weight, got ${weight.dtype}`
      );
    }
    if (weight.ndim < 2) {
      throw new ShapeError(
        `SpectralNorm requires weight with at least 2 dimensions, got ${weight.ndim}D`
      );
    }
    if (weight.size === 0) {
      throw new ShapeError("SpectralNorm requires a non-empty weight");
    }

    // Reshape to 2D: (outFeatures, inFeatures*)
    this.rows = weight.shape[0] ?? 1;
    this.cols = weight.size / this.rows;

    // Initialize u and v with random unit vectors
    this.u = randomUnitVector(this.rows);
    this.v = randomUnitVector(this.cols);

    const bufferBase = weightName.replaceAll(".", "_");
    this.registerBuffer(`${bufferBase}_u`, vectorTensor(this.u));
    this.registerBuffer(`${bufferBase}_v`, vectorTensor(this.v));

    // Warm up the estimate so the first forward pass already uses a good sigma.
    const W = readNumbers(weight, "SpectralNorm");
    for (let i = 0; i < 15; i++) {
      this.powerStep(W);
    }
  }

  /**
   * Forward pass: normalize the weight by its spectral norm, then delegate.
   *
   * In training mode the power iteration vectors are refined first. The wrapped
   * module's weight parameter is swapped for the normalized weight only while
   * the wrapped `forward` runs, and is restored afterwards, also when it throws.
   *
   * A `GradTensor` input gives a `GradTensor`. A plain `Tensor` input gives a `GradTensor`
   * while the wrapped weight requires grad and gradient tracking is on, and a plain `Tensor`
   * otherwise.
   *
   * @param input - Input tensor
   * @returns Output from the wrapped module with spectrally normalized weight
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const param = this.getWeightParam();
    const weight = param.tensor;
    const W = readNumbers(weight, "SpectralNorm");

    if (this.training) {
      for (let i = 0; i < this.nPowerIterations; i++) {
        this.powerStep(W);
      }
    }

    const sigma = this.sigmaFor(W);

    // Normalized weight W / sigma, stored in the weight's dtype.
    const normalized =
      weight.dtype === "float64" ? new Float64Array(W.length) : new Float32Array(W.length);
    for (let i = 0; i < W.length; i++) {
      normalized[i] = (W[i] as number) / sigma;
    }
    const normalizedTensor = TensorClass.fromTypedArray({
      data: normalized,
      shape: weight.shape.slice(),
      dtype: weight.dtype === "float64" ? "float64" : "float32",
      device: "cpu",
    });

    // Gradient of W_SN = W / sigma(W) with sigma = u^T W v and u, v held constant:
    //   dL/dW = (G - <G, W_SN> u v^T) / sigma
    const uSnap = this.u.slice();
    const vSnap = this.v.slice();
    const cols = this.cols;
    const dtype = weight.dtype === "float64" ? "float64" : "float32";
    const shape = weight.shape.slice();
    const normalizedWeight = customOp(normalizedTensor, [
      [
        param,
        (g: TensorClass): TensorClass => {
          const G = readNumbers(g, "SpectralNorm");
          let inner = 0;
          for (let i = 0; i < G.length; i++) {
            inner += (G[i] as number) * (normalized[i] as number);
          }
          const out = dtype === "float64" ? new Float64Array(G.length) : new Float32Array(G.length);
          for (let i = 0; i < G.length; i++) {
            const row = Math.floor(i / cols);
            const col = i - row * cols;
            out[i] =
              ((G[i] as number) - inner * (uSnap[row] as number) * (vSnap[col] as number)) / sigma;
          }
          return TensorClass.fromTypedArray({ data: out, shape, dtype, device: "cpu" });
        },
      ],
    ]);

    const inputGrad = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const restore = this.substituteParameter(param, normalizedWeight);
    let result: GradTensor | Tensor;
    try {
      result = this.wrapped.forward(inputGrad);
    } finally {
      restore();
    }

    return GradTensor.isGradTensor(result) ? result : GradTensor.fromTensor(result);
  }

  /**
   * Get the current estimated spectral norm (largest singular value).
   */
  get spectralNormValue(): number {
    const W = readNumbers(this.getWeightParam().tensor, "SpectralNorm");
    return this.sigmaFor(W);
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

  private getWeightParam(): GradTensor {
    for (const [name, param] of this.wrapped.namedParameters()) {
      if (name === this.weightName) {
        return param;
      }
    }
    throw new InvalidParameterError(
      `Module does not have a parameter named '${this.weightName}'`,
      "weightName",
      this.weightName
    );
  }

  /**
   * Point every reference to `target` inside the wrapped module tree (fields,
   * parameter registries) at `replacement`. Returns a function that undoes it.
   */
  private substituteParameter(target: GradTensor, replacement: GradTensor): () => void {
    const undo: Array<() => void> = [];
    for (const module of this.wrapped.modules()) {
      for (const [key, value] of Object.entries(module)) {
        if (value === target) {
          Reflect.set(module, key, replacement);
          undo.push(() => Reflect.set(module, key, target));
        } else if (value instanceof Map) {
          for (const [mapKey, mapValue] of value) {
            if (mapValue === target) {
              value.set(mapKey, replacement);
              undo.push(() => value.set(mapKey, target));
            }
          }
        } else if (Array.isArray(value)) {
          for (let i = 0; i < value.length; i++) {
            if (value[i] === target) {
              value[i] = replacement;
              undo.push(() => {
                value[i] = target;
              });
            }
          }
        }
      }
    }
    if (undo.length === 0) {
      throw new InvalidParameterError(
        `Could not locate parameter '${this.weightName}' inside the wrapped module`,
        "weightName",
        this.weightName
      );
    }
    return () => {
      for (let i = undo.length - 1; i >= 0; i--) (undo[i] as () => void)();
    };
  }

  /** One power iteration step: v = normalize(W^T u), u = normalize(W v), in place. */
  private powerStep(W: ArrayLike<number>): void {
    this.v.set(this.matVecTranspose(W, this.u));
    normalizeInPlace(this.v, this.eps);
    this.u.set(this.matVec(W, this.v));
    normalizeInPlace(this.u, this.eps);
  }

  /** sigma = u^T W v, floored at eps. */
  private sigmaFor(W: ArrayLike<number>): number {
    const Wv = this.matVec(W, this.v);
    let sigma = 0;
    for (let i = 0; i < this.rows; i++) {
      sigma += (this.u[i] as number) * (Wv[i] as number);
    }
    return Math.max(sigma, this.eps);
  }

  // Matrix-vector product: y = W * x, where W is (rows x cols)
  private matVec(W: ArrayLike<number>, x: Float64Array): Float64Array {
    const y = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      let sum = 0;
      const base = i * this.cols;
      for (let j = 0; j < this.cols; j++) {
        sum += (W[base + j] as number) * (x[j] as number);
      }
      y[i] = sum;
    }
    return y;
  }

  // Matrix-transpose-vector product: y = W^T * x, where W is (rows x cols)
  private matVecTranspose(W: ArrayLike<number>, x: Float64Array): Float64Array {
    const y = new Float64Array(this.cols);
    for (let i = 0; i < this.rows; i++) {
      const xi = x[i] as number;
      const base = i * this.cols;
      for (let j = 0; j < this.cols; j++) {
        y[j] = (y[j] as number) + (W[base + j] as number) * xi;
      }
    }
    return y;
  }
}

function vectorTensor(data: Float64Array): TensorClass {
  return TensorClass.fromTypedArray({
    data,
    shape: [data.length],
    dtype: "float64",
    device: "cpu",
  });
}

function randomUnitVector(n: number): Float64Array {
  const v = new Float64Array(n);
  let norm = 0;
  for (let i = 0; i < n; i++) {
    v[i] = __random() - 0.5;
    norm += (v[i] as number) * (v[i] as number);
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < n; i++) {
      v[i] = (v[i] as number) / norm;
    }
  }
  return v;
}

function normalizeInPlace(v: Float64Array, eps: number): void {
  let norm = 0;
  for (let i = 0; i < v.length; i++) {
    norm += (v[i] as number) * (v[i] as number);
  }
  norm = Math.sqrt(norm);
  const invNorm = 1 / Math.max(norm, eps);
  for (let i = 0; i < v.length; i++) {
    v[i] = (v[i] as number) * invNorm;
  }
}
