/**
 * Weight initialization functions for neural network layers.
 *
 * Provides standard initialization strategies used in deep learning:
 * - Xavier/Glorot (for sigmoid/tanh activations)
 * - Kaiming/He (for ReLU activations)
 * - Uniform, Normal, Constant, Zeros, Ones
 * - Orthogonal, Sparse
 *
 * All functions modify the tensor in-place and return it for chaining.
 *
 * @example
 * ```ts
 * import { Linear } from 'deepbox/nn';
 * import { xavier_uniform_, kaiming_normal_ } from 'deepbox/nn';
 *
 * const layer = new Linear(128, 64);
 * xavier_uniform_(layer.weight);
 * zeros_(layer.bias);
 * ```
 *
 * @module nn/init
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Weight Initialization}
 */

import { InvalidParameterError } from "../core";
import type { Tensor } from "../ndarray";
import { __random } from "../random/random";

// Simple seeded RNG for reproducibility
function makeRng(seed?: number): () => number {
  if (seed !== undefined) {
    let s = seed;
    return () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
  }
  return __random;
}

// Box-Muller transform for normal distribution
function boxMuller(rng: () => number): number {
  let u1: number;
  do {
    u1 = rng();
  } while (u1 === 0);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function calculateFanInOut(tensor: Tensor): { fanIn: number; fanOut: number } {
  const ndim = tensor.ndim;
  if (ndim < 1) {
    throw new InvalidParameterError(
      "Fan in/out cannot be computed for scalar tensors",
      "tensor",
      tensor.shape
    );
  }
  if (ndim === 1) {
    return { fanIn: tensor.shape[0] ?? 1, fanOut: tensor.shape[0] ?? 1 };
  }
  if (ndim === 2) {
    return { fanIn: tensor.shape[1] ?? 1, fanOut: tensor.shape[0] ?? 1 };
  }
  // Conv: [outChannels, inChannels, ...kernelSize]
  const outChannels = tensor.shape[0] ?? 1;
  const inChannels = tensor.shape[1] ?? 1;
  let receptiveFieldSize = 1;
  for (let i = 2; i < ndim; i++) {
    receptiveFieldSize *= tensor.shape[i] ?? 1;
  }
  return {
    fanIn: inChannels * receptiveFieldSize,
    fanOut: outChannels * receptiveFieldSize,
  };
}

function calculateGain(nonlinearity: string, param?: number): number {
  switch (nonlinearity) {
    case "linear":
    case "conv1d":
    case "conv2d":
    case "conv3d":
    case "sigmoid":
      return 1;
    case "tanh":
      return 5.0 / 3;
    case "relu":
      return Math.SQRT2;
    case "leaky_relu": {
      const slope = param ?? 0.01;
      return Math.sqrt(2.0 / (1 + slope * slope));
    }
    case "selu":
      return 3.0 / 4;
    default:
      throw new InvalidParameterError(
        `Unsupported nonlinearity: ${nonlinearity}`,
        "nonlinearity",
        nonlinearity
      );
  }
}

/**
 * Fill tensor with values drawn from a uniform distribution U(low, high).
 *
 * @param tensor - Tensor to fill in-place
 * @param low - Lower bound (default: 0)
 * @param high - Upper bound (default: 1)
 * @returns The modified tensor
 */
export function uniform_(tensor: Tensor, low = 0, high = 1): Tensor {
  const rng = __random;
  for (let i = 0; i < tensor.size; i++) {
    tensor.data[tensor.offset + i] = low + (high - low) * rng();
  }
  return tensor;
}

/**
 * Fill tensor with values drawn from a normal distribution N(mean, std²).
 *
 * @param tensor - Tensor to fill in-place
 * @param mean - Mean of the distribution (default: 0)
 * @param std - Standard deviation (default: 1)
 * @returns The modified tensor
 */
export function normal_(tensor: Tensor, mean = 0, std = 1): Tensor {
  const rng = __random;
  for (let i = 0; i < tensor.size; i++) {
    tensor.data[tensor.offset + i] = mean + std * boxMuller(rng);
  }
  return tensor;
}

/**
 * Fill tensor with a constant value.
 *
 * @param tensor - Tensor to fill in-place
 * @param val - Value to fill with
 * @returns The modified tensor
 */
export function constant_(tensor: Tensor, val: number): Tensor {
  for (let i = 0; i < tensor.size; i++) {
    tensor.data[tensor.offset + i] = val;
  }
  return tensor;
}

/**
 * Fill tensor with zeros.
 *
 * @param tensor - Tensor to fill in-place
 * @returns The modified tensor
 */
export function zeros_(tensor: Tensor): Tensor {
  return constant_(tensor, 0);
}

/**
 * Fill tensor with ones.
 *
 * @param tensor - Tensor to fill in-place
 * @returns The modified tensor
 */
export function ones_(tensor: Tensor): Tensor {
  return constant_(tensor, 1);
}

/**
 * Fill tensor using Xavier uniform initialization.
 *
 * Draws from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
 * Recommended for sigmoid/tanh activations.
 *
 * @param tensor - Tensor to fill in-place
 * @param gain - Scaling factor (default: 1.0)
 * @returns The modified tensor
 * @deprecated Prefer {@link xavierUniform_}.
 */
export function xavier_uniform_(tensor: Tensor, gain = 1.0): Tensor {
  const { fanIn, fanOut } = calculateFanInOut(tensor);
  const a = gain * Math.sqrt(6.0 / (fanIn + fanOut));
  return uniform_(tensor, -a, a);
}

/**
 * Fill tensor using Xavier normal initialization.
 *
 * Draws from N(0, std²) where std = gain * sqrt(2 / (fan_in + fan_out)).
 * Recommended for sigmoid/tanh activations.
 *
 * @param tensor - Tensor to fill in-place
 * @param gain - Scaling factor (default: 1.0)
 * @returns The modified tensor
 * @deprecated Prefer {@link xavierNormal_}.
 */
export function xavier_normal_(tensor: Tensor, gain = 1.0): Tensor {
  const { fanIn, fanOut } = calculateFanInOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  return normal_(tensor, 0, std);
}

/**
 * Fill tensor using Kaiming uniform initialization (He initialization).
 *
 * Draws from U(-bound, bound) where bound = gain * sqrt(3 / fan).
 * Recommended for ReLU activations.
 *
 * @param tensor - Tensor to fill in-place
 * @param a - Negative slope of rectifier (for leaky_relu, default: 0)
 * @param mode - 'fan_in' (default) or 'fan_out'
 * @param nonlinearity - Activation function name (default: 'leaky_relu')
 * @returns The modified tensor
 * @deprecated Prefer {@link kaimingUniform_}.
 */
export function kaiming_uniform_(
  tensor: Tensor,
  a = 0,
  mode: "fan_in" | "fan_out" = "fan_in",
  nonlinearity = "leaky_relu"
): Tensor {
  const { fanIn, fanOut } = calculateFanInOut(tensor);
  const fan = mode === "fan_in" ? fanIn : fanOut;
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  const bound = Math.sqrt(3.0) * std;
  return uniform_(tensor, -bound, bound);
}

/**
 * Fill tensor using Kaiming normal initialization (He initialization).
 *
 * Draws from N(0, std²) where std = gain / sqrt(fan).
 * Recommended for ReLU activations.
 *
 * @param tensor - Tensor to fill in-place
 * @param a - Negative slope of rectifier (for leaky_relu, default: 0)
 * @param mode - 'fan_in' (default) or 'fan_out'
 * @param nonlinearity - Activation function name (default: 'leaky_relu')
 * @returns The modified tensor
 * @deprecated Prefer {@link kaimingNormal_}.
 */
export function kaiming_normal_(
  tensor: Tensor,
  a = 0,
  mode: "fan_in" | "fan_out" = "fan_in",
  nonlinearity = "leaky_relu"
): Tensor {
  const { fanIn, fanOut } = calculateFanInOut(tensor);
  const fan = mode === "fan_in" ? fanIn : fanOut;
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  return normal_(tensor, 0, std);
}

/**
 * Fill tensor with an orthogonal matrix.
 *
 * Uses QR decomposition of a random matrix. For non-square matrices,
 * fills with semi-orthogonal rows or columns.
 *
 * @param tensor - 2D tensor to fill in-place
 * @param gain - Scaling factor (default: 1.0)
 * @returns The modified tensor
 */
export function orthogonal_(tensor: Tensor, gain = 1.0): Tensor {
  if (tensor.ndim < 2) {
    throw new InvalidParameterError(
      "orthogonal_ requires at least 2D tensor",
      "tensor",
      tensor.shape
    );
  }
  const rows = tensor.shape[0] ?? 1;
  let cols = 1;
  for (let i = 1; i < tensor.ndim; i++) {
    cols *= tensor.shape[i] ?? 1;
  }

  // Generate random matrix
  const rng = __random;
  const flat: number[] = [];
  for (let i = 0; i < rows * cols; i++) {
    flat.push(boxMuller(rng));
  }

  // Simple Gram-Schmidt orthogonalization
  const n = Math.min(rows, cols);
  const vectors: number[][] = [];
  for (let i = 0; i < (rows <= cols ? rows : cols); i++) {
    const v =
      rows <= cols
        ? flat.slice(i * cols, (i + 1) * cols)
        : Array.from({ length: rows }, (_, r) => flat[r * cols + i] ?? 0);
    // Subtract projections of previous vectors
    for (const u of vectors) {
      let dotProd = 0;
      for (let j = 0; j < v.length; j++) {
        dotProd += (v[j] ?? 0) * (u[j] ?? 0);
      }
      for (let j = 0; j < v.length; j++) {
        v[j] = (v[j] ?? 0) - dotProd * (u[j] ?? 0);
      }
    }
    // Normalize
    let norm = 0;
    for (let j = 0; j < v.length; j++) {
      norm += (v[j] ?? 0) * (v[j] ?? 0);
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let j = 0; j < v.length; j++) {
        v[j] = (v[j] ?? 0) / norm;
      }
    }
    vectors.push(v);
  }

  // Fill tensor
  if (rows <= cols) {
    for (let i = 0; i < n; i++) {
      const v = vectors[i]!;
      for (let j = 0; j < cols; j++) {
        tensor.data[tensor.offset + i * cols + j] = gain * (v[j] ?? 0);
      }
    }
    // Fill remaining rows with zeros
    for (let i = n; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        tensor.data[tensor.offset + i * cols + j] = 0;
      }
    }
  } else {
    // Fill column by column
    for (let j = 0; j < n; j++) {
      const v = vectors[j]!;
      for (let i = 0; i < rows; i++) {
        tensor.data[tensor.offset + i * cols + j] = gain * (v[i] ?? 0);
      }
    }
    for (let j = n; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        tensor.data[tensor.offset + i * cols + j] = 0;
      }
    }
  }

  return tensor;
}

/**
 * Fill tensor as a sparse matrix with normally distributed non-zero entries.
 *
 * @param tensor - 2D tensor to fill in-place
 * @param sparsity - Fraction of elements to be zero (default: 0.1)
 * @param std - Standard deviation of the normal distribution (default: 0.01)
 * @returns The modified tensor
 */
export function sparse_(tensor: Tensor, sparsity = 0.1, std = 0.01): Tensor {
  if (tensor.ndim !== 2) {
    throw new InvalidParameterError("sparse_ requires 2D tensor", "tensor", tensor.shape);
  }
  const rows = tensor.shape[0] ?? 1;
  const cols = tensor.shape[1] ?? 1;
  const rng = __random;

  // Zero out everything
  zeros_(tensor);

  // For each column, select non-zero entries
  const nNonZero = Math.round(rows * (1 - sparsity));
  for (let j = 0; j < cols; j++) {
    // Randomly select rows for non-zero entries
    const indices: number[] = [];
    for (let i = 0; i < rows; i++) indices.push(i);
    // Fisher-Yates shuffle
    for (let i = rows - 1; i > 0; i--) {
      const idx = Math.floor(rng() * (i + 1));
      const tmp = indices[i]!;
      indices[i] = indices[idx]!;
      indices[idx] = tmp;
    }
    for (let k = 0; k < nNonZero && k < rows; k++) {
      const i = indices[k]!;
      tensor.data[tensor.offset + i * cols + j] = std * boxMuller(rng);
    }
  }

  return tensor;
}

export { calculateFanInOut, calculateGain, makeRng };

// ---------------------------------------------------------------------------
// Canonical camelCase aliases
//
// The snake_case spellings above mirror PyTorch and remain exported for
// backward compatibility. These camelCase aliases are the recommended names on
// Deepbox's public surface; each refers to the exact same in-place function.
// The trailing-underscore forms preserve PyTorch's in-place convention; the
// no-underscore forms are provided as a convenience.
// ---------------------------------------------------------------------------

/** Canonical camelCase alias of {@link kaiming_normal_} (in-place). */
export const kaimingNormal_ = kaiming_normal_;
/** Convenience alias of {@link kaiming_normal_} without the trailing underscore. */
export const kaimingNormal = kaiming_normal_;
/** Canonical camelCase alias of {@link kaiming_uniform_} (in-place). */
export const kaimingUniform_ = kaiming_uniform_;
/** Convenience alias of {@link kaiming_uniform_} without the trailing underscore. */
export const kaimingUniform = kaiming_uniform_;
/** Canonical camelCase alias of {@link xavier_normal_} (in-place). */
export const xavierNormal_ = xavier_normal_;
/** Convenience alias of {@link xavier_normal_} without the trailing underscore. */
export const xavierNormal = xavier_normal_;
/** Canonical camelCase alias of {@link xavier_uniform_} (in-place). */
export const xavierUniform_ = xavier_uniform_;
/** Convenience alias of {@link xavier_uniform_} without the trailing underscore. */
export const xavierUniform = xavier_uniform_;
/** Convenience alias of {@link orthogonal_} without the trailing underscore. */
export const orthogonal = orthogonal_;
/** Convenience alias of {@link zeros_} without the trailing underscore. */
export const zeros = zeros_;
/** Convenience alias of {@link ones_} without the trailing underscore. */
export const ones = ones_;
/** Convenience alias of {@link constant_} without the trailing underscore. */
export const constant = constant_;
