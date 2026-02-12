/**
 * Shared internal helpers for the metrics module.
 *
 * These functions are used across classification, regression, and clustering metrics
 * to handle strided tensor access and input validation.
 *
 * @internal
 */

import { DataValidationError, ShapeError } from "../core/errors";
import type { Tensor } from "../ndarray";

/**
 * Converts a flat (logical) index to a physical buffer offset.
 *
 * @internal
 */
export type FlatOffsetter = (flatIndex: number) => number;

/**
 * Compute row-major logical strides from a tensor shape.
 *
 * @internal
 */
export function computeLogicalStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    const dim = shape[i];
    if (dim === undefined) {
      throw new ShapeError("Tensor shape must be fully defined");
    }
    strides[i] = stride;
    stride *= dim;
  }
  return strides;
}

/**
 * Build a function that maps flat (logical) indices to physical buffer offsets.
 *
 * Handles arbitrary strides (views, slices, transposes).
 *
 * @internal
 */
export function createFlatOffsetter(t: Tensor): FlatOffsetter {
  const base = t.offset;

  if (t.ndim <= 1) {
    const stride0 = t.strides[0] ?? 1;
    return (flatIndex: number) => base + flatIndex * stride0;
  }

  const logicalStrides = computeLogicalStrides(t.shape);
  const strides = t.strides;

  return (flatIndex: number) => {
    let rem = flatIndex;
    let offset = base;

    for (let axis = 0; axis < logicalStrides.length; axis++) {
      const axisLogicalStride = logicalStrides[axis] ?? 1;
      const coord = Math.floor(rem / axisLogicalStride);
      rem -= coord * axisLogicalStride;
      offset += coord * (strides[axis] ?? 0);
    }

    return offset;
  };
}

/**
 * Assert that a numeric value is finite.
 *
 * @param value - The value to check
 * @param name - Name of the tensor for error messages
 * @param detail - Additional detail for the error message (e.g., "index 5")
 *
 * @internal
 */
export function assertFiniteNumber(value: number, name: string, detail: string): void {
  if (!Number.isFinite(value)) {
    throw new DataValidationError(
      `${name} must contain only finite numbers; found ${String(value)} at ${detail}`
    );
  }
}

/**
 * Assert that a tensor is 1D or a column vector.
 *
 * @internal
 */
export function assertVectorLike(t: Tensor, name: string): void {
  if (t.ndim <= 1) return;
  if (t.ndim === 2 && (t.shape[1] ?? 0) === 1) return;
  throw new ShapeError(`${name} must be 1D or a column vector`);
}

/**
 * Assert that two tensors have the same size and are vector-like.
 *
 * @internal
 */
export function assertSameSizeVectors(a: Tensor, b: Tensor, nameA: string, nameB: string): void {
  if (a.size !== b.size) {
    throw new ShapeError(
      `${nameA} (size ${a.size}) and ${nameB} (size ${b.size}) must have same size`
    );
  }
  assertVectorLike(a, nameA);
  assertVectorLike(b, nameB);
}

/**
 * Assert that two tensors have the same size (without vector-like check).
 *
 * @internal
 */
export function assertSameSize(a: Tensor, b: Tensor, nameA: string, nameB: string): void {
  if (a.size !== b.size) {
    throw new ShapeError(
      `${nameA} (size ${a.size}) and ${nameB} (size ${b.size}) must have same size`
    );
  }
}
