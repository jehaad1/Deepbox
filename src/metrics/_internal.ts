/**
 * Shared internal helpers for the metrics module.
 *
 * These functions are used across classification, regression, and clustering metrics
 * to handle strided tensor access and input validation.
 *
 * @internal
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox documentation}
 */

import { DataValidationError, ShapeError } from "../core/errors";
import type { Tensor } from "../ndarray";

/**
 * Read a numeric tensor's logical elements into a dense Float64Array in
 * row-major order, honouring strides/offset. Optionally validates that every
 * value is finite in the same pass. Metric hot loops then index `[i]`
 * directly instead of paying a per-element offsetter closure + bounds-checked
 * `getNumericElement` + finite-check call.
 *
 * Throws for string/int64 tensors (callers guard dtype upstream).
 */
export function denseFloat64(t: Tensor, name: string, checkFinite = true): Float64Array {
  const data = t.data;
  if (Array.isArray(data) || data instanceof BigInt64Array) {
    throw new DataValidationError(`${name} must be a numeric (non-int64) tensor`);
  }
  const size = t.size;
  const out = new Float64Array(size);
  if (size === 0) return out;

  const shape = t.shape;
  const ndim = shape.length;
  const strides = t.strides;
  const offset = t.offset;

  if (ndim <= 1) {
    const s0 = strides[0] ?? 1;
    if (s0 === 1 && offset === 0 && data.length === size) {
      out.set(data as unknown as ArrayLike<number>);
    } else {
      for (let i = 0; i < size; i++) out[i] = data[offset + i * s0] as number;
    }
  } else {
    // Row-major odometer over the strided view.
    const inner = shape[ndim - 1] ?? 1;
    const innerStride = strides[ndim - 1] ?? 0;
    const outer = inner === 0 ? 0 : size / inner;
    const coords = new Array<number>(Math.max(0, ndim - 1)).fill(0);
    let base = offset;
    let pos = 0;
    for (let b = 0; b < outer; b++) {
      let idx = base;
      for (let j = 0; j < inner; j++) {
        out[pos++] = data[idx] as number;
        idx += innerStride;
      }
      for (let d = ndim - 2; d >= 0; d--) {
        coords[d] = (coords[d] ?? 0) + 1;
        base += strides[d] ?? 0;
        if ((coords[d] ?? 0) < (shape[d] ?? 0)) break;
        base -= (strides[d] ?? 0) * (shape[d] ?? 0);
        coords[d] = 0;
      }
    }
  }

  if (checkFinite) {
    for (let i = 0; i < size; i++) {
      const v = out[i] as number;
      if (!Number.isFinite(v)) {
        throw new DataValidationError(`${name} contains NaN or infinite value at index ${i}`);
      }
    }
  }
  return out;
}

/**
 * Converts a flat (logical) index to a physical buffer offset.
 *
 * @internal
 */
export type FlatOffsetter = (flatIndex: number) => number;

/**
 * Dense-read a tensor's logical elements into a Float64Array when it is a
 * plain numeric typed array (not string/int64), else return null. No
 * finiteness check — intended for label comparison where any finite integer
 * is valid. Lets label metrics take a monomorphic fast path for the common
 * numeric case and fall back to the generic per-element reader otherwise.
 *
 * @internal
 */
export function tryDenseNumeric(t: Tensor): Float64Array | null {
  const data = t.data;
  if (Array.isArray(data) || data instanceof BigInt64Array) return null;
  const size = t.size;
  const out = new Float64Array(size);
  if (size === 0) return out;
  const shape = t.shape;
  const ndim = shape.length;
  const strides = t.strides;
  const offset = t.offset;
  if (ndim <= 1) {
    const s0 = strides[0] ?? 1;
    if (s0 === 1 && offset === 0 && data.length === size) {
      out.set(data as unknown as ArrayLike<number>);
    } else {
      for (let i = 0; i < size; i++) out[i] = data[offset + i * s0] as number;
    }
    return out;
  }
  const inner = shape[ndim - 1] ?? 1;
  const innerStride = strides[ndim - 1] ?? 0;
  const outer = inner === 0 ? 0 : size / inner;
  const coords = new Array<number>(Math.max(0, ndim - 1)).fill(0);
  let base = offset;
  let pos = 0;
  for (let b = 0; b < outer; b++) {
    let idx = base;
    for (let j = 0; j < inner; j++) {
      out[pos++] = data[idx] as number;
      idx += innerStride;
    }
    for (let d = ndim - 2; d >= 0; d--) {
      coords[d] = (coords[d] ?? 0) + 1;
      base += strides[d] ?? 0;
      if ((coords[d] ?? 0) < (shape[d] ?? 0)) break;
      base -= (strides[d] ?? 0) * (shape[d] ?? 0);
      coords[d] = 0;
    }
  }
  return out;
}

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
