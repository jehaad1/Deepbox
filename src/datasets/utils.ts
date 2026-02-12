import { DeepboxError, InvalidParameterError } from "../core/errors";

/**
 * Assert that an input is a positive integer.
 *
 * @internal
 */
export function assertPositiveInt(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0 || !Number.isSafeInteger(value)) {
    throw new InvalidParameterError(
      `${name} must be a positive safe integer; received ${value}`,
      name,
      value
    );
  }
}

/**
 * Assert that a value is a boolean.
 *
 * @internal
 */
export function assertBoolean(name: string, value: unknown): void {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(
      `${name} must be a boolean; received ${String(value)}`,
      name,
      value
    );
  }
}

/**
 * Normalize and validate an optional seed value.
 *
 * @internal
 */
export function normalizeOptionalSeed(name: string, value: number | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Number.isFinite(value) || !Number.isInteger(value) || !Number.isSafeInteger(value)) {
    throw new InvalidParameterError(
      `${name} must be a finite safe integer; received ${value}`,
      name,
      value
    );
  }
  return value;
}

/**
 * Create a pseudo-random number generator.
 *
 * Uses a Linear Congruential Generator (LCG) with parameters from Numerical Recipes.
 * If seed is undefined, falls back to Math.random().
 *
 * @internal
 */
export function createRng(seed?: number): () => number {
  if (seed === undefined) return () => Math.random();

  let state = (seed ^ 0x9e3779b9) >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/**
 * Sample from standard normal distribution N(0, 1) using Box-Muller transform.
 *
 * @internal
 */
export function normal01(rng: () => number): number {
  const u1 = Math.max(rng(), Number.EPSILON);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Shuffle an array in-place using Fisher-Yates.
 *
 * @internal
 */
export function shuffleInPlace<T>(array: T[], rng: () => number): void {
  if (array.length <= 1) return;

  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    // Safety: i and j are both in [0, array.length - 1]
    const tmp = array[i];
    const swap = array[j];
    if (tmp === undefined || swap === undefined) {
      throw new DeepboxError(
        `Internal error: shuffle index out of bounds (i=${i}, j=${j}, len=${array.length})`
      );
    }
    array[i] = swap;
    array[j] = tmp;
  }
}

/**
 * Shuffle two aligned arrays in-place using Fisher-Yates.
 *
 * @internal
 */
export function shufflePairedInPlace<T, U>(left: T[], right: U[], rng: () => number): void {
  if (left.length !== right.length) {
    throw new DeepboxError(
      `Internal error: array length mismatch during shuffle (${left.length} vs ${right.length})`
    );
  }
  if (left.length <= 1) return;

  for (let i = left.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    // Safety: i and j are both in [0, left.length - 1]
    const leftTmp = left[i];
    const leftSwap = left[j];
    if (leftTmp === undefined || leftSwap === undefined) {
      throw new DeepboxError(
        `Internal error: shuffle index out of bounds (i=${i}, j=${j}, len=${left.length})`
      );
    }
    left[i] = leftSwap;
    left[j] = leftTmp;

    const rightTmp = right[i];
    const rightSwap = right[j];
    if (rightTmp === undefined || rightSwap === undefined) {
      throw new DeepboxError(
        `Internal error: shuffle index out of bounds (i=${i}, j=${j}, len=${right.length})`
      );
    }
    right[i] = rightSwap;
    right[j] = rightTmp;
  }
}
