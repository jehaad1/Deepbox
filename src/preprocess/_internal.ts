/**
 * Shared internal helpers for the preprocess module.
 *
 * These functions are used across scalers and splitting utilities
 * to handle tensor shape validation, seeded RNG, and index shuffling.
 *
 * @internal
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import { DeepboxError, DTypeError, InvalidParameterError, ShapeError } from "../core/errors";
import type { Tensor } from "../ndarray";
import { __randomBelow, __SeededRandom } from "../random/random";

/**
 * Assert that a tensor has a real numeric dtype (not string or complex).
 *
 * @internal
 */
export function assertNumericTensor(X: Tensor, name: string): void {
  if (X.dtype === "string") {
    throw new DTypeError(`${name} must be numeric`);
  }
  if (X.dtype === "complex64" || X.dtype === "complex128") {
    throw new DTypeError(`${name} must be real-valued; received dtype ${X.dtype}`);
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
 * Seeded random number generator using a 31-bit Linear Congruential Generator
 * (the ANSI C `rand` constants). Provides reproducible pseudo-random sequences
 * when given a seed.
 *
 * The multiply-add is done in exact 32-bit integer arithmetic with
 * `Math.imul`. A plain `a * state` exceeds 2^53 and silently drops low bits,
 * which collapses the sequence into a cycle of a few thousand values.
 *
 * Neighbouring seeds give related streams. New code should use {@link createRandomStream}.
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

  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new InvalidParameterError(
      "randomState must be a non-negative safe integer",
      "randomState",
      seed
    );
  }

  let state = seed % m;

  return () => {
    // Math.imul keeps the exact low 32 bits of the product; masking to 31 bits
    // is the same as reducing modulo 2^31.
    state = (Math.imul(a, state) + c) & 0x7fffffff;
    return state / m;
  };
}

const UINT64_MASK = (1n << 64n) - 1n;

/**
 * Derive an independent seed for stream number `index` from a base seed.
 *
 * Splitters that run several iterations (repeated K-fold, shuffle splits) must not
 * seed iteration `i` with `seed + i`: iteration `i + 1` of seed `s` would then equal
 * iteration `i` of seed `s + 1`. The pair is hashed with the SplitMix64 finalizer
 * instead, so every (seed, index) pair gives its own stream.
 *
 * @param seed - Non-negative safe integer base seed
 * @param index - Non-negative integer stream number
 * @returns Non-negative safe integer seed
 *
 * @internal
 */
export function deriveSeed(seed: number, index: number): number {
  assertSeed(seed);
  if (!Number.isSafeInteger(index) || index < 0) {
    throw new InvalidParameterError(
      "stream index must be a non-negative safe integer",
      "index",
      index
    );
  }
  let z =
    (BigInt(seed) * 0xd1b54a32d192ed03n + (BigInt(index) + 1n) * 0x9e3779b97f4a7c15n) & UINT64_MASK;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & UINT64_MASK;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & UINT64_MASK;
  z ^= z >> 31n;
  return Number(z & BigInt(Number.MAX_SAFE_INTEGER));
}

function assertSeed(seed: number): void {
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new InvalidParameterError(
      "randomState must be a non-negative safe integer",
      "randomState",
      seed
    );
  }
}

/**
 * Seeded random number generator for shuffles and samples: xoshiro128++ from the random
 * module, with the state expanded from the hashed pair (`seed`, `stream`).
 *
 * Unlike {@link createSeededRandom} (a 31-bit LCG whose streams for neighbouring seeds are
 * related), neighbouring seeds and neighbouring stream numbers give unrelated sequences.
 * Splitters that need several streams pass the iteration number as `stream`.
 *
 * @param seed - Non-negative safe integer seed value
 * @param stream - Non-negative integer stream number (default 0)
 * @returns Function that generates random numbers in [0, 1)
 *
 * @internal
 */
export function createRandomStream(seed: number, stream = 0): () => number {
  assertSeed(seed);
  const rng = new __SeededRandom(BigInt(deriveSeed(seed, stream)));
  return () => rng.next();
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
    const j = __randomBelow(random, i + 1);
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
