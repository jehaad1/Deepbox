import { InvalidParameterError } from "../../core";

/**
 * Validates that a number is a positive integer.
 * @internal
 */
export function assertPositiveInt(name: string, v: number): void {
  if (!Number.isFinite(v)) {
    throw new InvalidParameterError(`${name} must be a positive integer; received ${v}`, name, v);
  }
  if (v <= 0) {
    throw new InvalidParameterError(`${name} must be a positive integer; received ${v}`, name, v);
  }
  if (Math.trunc(v) !== v) {
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
 * Clamps an integer value to a range.
 * @internal
 */
export function clampInt(x: number, lo: number, hi: number): number {
  if (!Number.isFinite(x)) return lo;
  if (x < lo) return lo;
  if (x > hi) return hi;
  return Math.trunc(x);
}
