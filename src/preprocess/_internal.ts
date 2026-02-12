/**
 * Shared internal helpers for the preprocess module.
 *
 * These functions are used across scalers and splitting utilities
 * to handle tensor shape validation, seeded RNG, and index shuffling.
 *
 * @internal
 */

import { DeepboxError, DTypeError, InvalidParameterError, ShapeError } from "../core/errors";
import type { Tensor } from "../ndarray";

/**
 * Assert that a tensor has numeric dtype (not string).
 *
 * @internal
 */
export function assertNumericTensor(X: Tensor, name: string): void {
  if (X.dtype === "string") {
    throw new DTypeError(`${name} must be numeric`);
  }
}

/**
 * Assert that a tensor is 2-dimensional.
 *
 * @internal
 */
export function assert2D(X: Tensor, name: string): void {
  if (X.ndim !== 2) {
    throw new ShapeError(`${name} must be a 2D tensor, got ${X.ndim}D`);
  }
}

/**
 * Extract and validate the shape of a 2D tensor.
 *
 * @internal
 */
export function getShape2D(X: Tensor): [number, number] {
  if (X.ndim !== 2 || X.shape[0] === undefined || X.shape[1] === undefined) {
    throw new ShapeError(`Expected 2D tensor with valid shape, got shape [${X.shape.join(", ")}]`);
  }
  return [X.shape[0], X.shape[1]];
}

/**
 * Extract the stride of a 1D tensor.
 *
 * @internal
 */
export function getStride1D(X: Tensor): number {
  const stride = X.strides[0];
  if (stride === undefined) {
    throw new DeepboxError("Internal error: missing stride for 1D tensor");
  }
  return stride;
}

/**
 * Extract the strides of a 2D tensor.
 *
 * @internal
 */
export function getStrides2D(X: Tensor): [number, number] {
  const stride0 = X.strides[0];
  const stride1 = X.strides[1];
  if (stride0 === undefined || stride1 === undefined) {
    throw new DeepboxError("Internal error: missing strides for 2D tensor");
  }
  return [stride0, stride1];
}

/**
 * Simple seeded random number generator using Linear Congruential Generator (LCG).
 * Provides reproducible pseudo-random sequences when given a seed.
 *
 * @param seed - Non-negative safe integer seed value
 * @returns Function that generates random numbers in [0, 1)
 *
 * @internal
 */
export function createSeededRandom(seed: number): () => number {
  const a = 1103515245;
  const c = 12345;
  const m = 2 ** 31;

  if (
    !Number.isFinite(seed) ||
    !Number.isInteger(seed) ||
    !Number.isSafeInteger(seed) ||
    seed < 0
  ) {
    throw new InvalidParameterError(
      "randomState must be a non-negative safe integer",
      "randomState",
      seed
    );
  }

  let state = seed % m;

  return () => {
    state = (a * state + c) % m;
    return state / m;
  };
}

/**
 * Shuffle an array of indices in-place using Fisher-Yates algorithm.
 *
 * @param indices - Array of integer indices to shuffle
 * @param random - Random number generator returning values in [0, 1)
 *
 * @internal
 */
export function shuffleIndicesInPlace(indices: number[], random: () => number): void {
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const temp = indices[i];
    if (temp === undefined) {
      throw new DeepboxError("Internal error: shuffle source index missing");
    }
    const swap = indices[j];
    if (swap === undefined) {
      throw new DeepboxError("Internal error: shuffle target index missing");
    }
    indices[i] = swap;
    indices[j] = temp;
  }
}
