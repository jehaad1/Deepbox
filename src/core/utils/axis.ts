/**
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

import { InvalidParameterError } from "../errors";
import type { Axis } from "../types/common";

/**
 * Normalize an axis argument to a valid dimension index.
 *
 * Supports negative indexing (Python-style) and string aliases:
 * - "index" or "rows" -> 0
 * - "columns" -> 1
 *
 * @param axis - Axis identifier (index or alias)
 * @param ndim - Number of dimensions in the tensor
 * @returns Non-negative integer dimension index
 * @throws {InvalidParameterError} If `ndim` is not a non-negative integer
 * @throws {InvalidParameterError} If axis is not an integer or is out of bounds
 *   (valid range is `-ndim` to `ndim - 1`)
 * @throws {InvalidParameterError} If axis alias is invalid
 *
 * @example
 * ```ts
 * normalizeAxis(-1, 3);        // 2
 * normalizeAxis("columns", 2); // 1
 * ```
 */
export function normalizeAxis(axis: Axis, ndim: number): number {
  if (!Number.isInteger(ndim) || ndim < 0) {
    throw new InvalidParameterError(
      `ndim must be a non-negative integer; received ${ndim}`,
      "ndim",
      ndim
    );
  }

  let ax: number;
  if (typeof axis === "string") {
    if (axis === "index" || axis === "rows") {
      ax = 0;
    } else if (axis === "columns") {
      ax = 1;
    } else {
      throw new InvalidParameterError(
        `Invalid axis: '${axis}'. Must be 'index', 'rows', or 'columns'`,
        "axis",
        axis
      );
    }
  } else {
    if (!Number.isInteger(axis)) {
      throw new InvalidParameterError(`axis must be an integer; received ${axis}`, "axis", axis);
    }
    ax = axis;
  }

  // Handle negative indexing
  const normalized = ax < 0 ? ndim + ax : ax;

  if (normalized < 0 || normalized >= ndim) {
    throw new InvalidParameterError(`axis ${axis} is out of bounds for ndim=${ndim}`, "axis", axis);
  }

  // `-0` is a valid integer axis; return a plain 0 so callers never see negative zero.
  return normalized === 0 ? 0 : normalized;
}

/**
 * Normalize a list of axes to valid dimension indices.
 *
 * Checks for duplicates (including a negative axis that resolves to an
 * already listed one, e.g. `[0, -2]` for ndim=2) and validity of each axis.
 *
 * @param axis - Single axis or array of axes
 * @param ndim - Number of dimensions
 * @returns Array of unique non-negative integer dimension indices
 * @throws {InvalidParameterError} If any axis is invalid, out of bounds, or duplicated
 */
export function normalizeAxes(axis: Axis | readonly Axis[], ndim: number): number[] {
  if (!Number.isInteger(ndim) || ndim < 0) {
    throw new InvalidParameterError(
      `ndim must be a non-negative integer; received ${ndim}`,
      "ndim",
      ndim
    );
  }

  const axes: readonly Axis[] = Array.isArray(axis) ? (axis as readonly Axis[]) : [axis as Axis];
  const seen = new Set<number>();
  const normalized: number[] = [];
  for (const ax of axes) {
    const norm = normalizeAxis(ax, ndim);
    if (seen.has(norm)) {
      throw new InvalidParameterError(
        `duplicate axis: axis ${String(ax)} resolves to dimension ${norm}, which is already listed`,
        "axis",
        axis
      );
    }
    seen.add(norm);
    normalized.push(norm);
  }
  return normalized;
}
