/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";

/**
 * Validates that a number is a positive safe integer.
 * @throws {InvalidParameterError} If `v` is not a number, not an integer, not positive, or
 *   beyond `Number.MAX_SAFE_INTEGER`.
 * @internal
 */
export function assertPositiveInt(name: string, v: number): void {
  if (typeof v !== "number" || !Number.isSafeInteger(v) || v <= 0) {
    throw new InvalidParameterError(`${name} must be a positive integer; received ${v}`, name, v);
  }
}

/**
 * Checks if a number is finite.
 * @internal
 */
export function isFiniteNumber(x: number): boolean {
  return Number.isFinite(x);
}

/**
 * Truncates `x` toward zero and clamps the result to `[lo, hi]`. Non-finite input
 * (NaN and the infinities) returns `lo`.
 * @internal
 */
export function clampInt(x: number, lo: number, hi: number): number {
  if (!Number.isFinite(x)) return lo;
  if (x < lo) return lo;
  if (x > hi) return hi;
  // `+ 0` turns the -0 produced by truncating values in (-1, 0) into 0.
  return Math.trunc(x) + 0;
}
