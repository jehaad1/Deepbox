/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
import type { AnyTensor } from "../../ndarray";
import { tensorToFloat64Matrix2D, tensorToFloat64Vector1D } from "./tensor";
import { isFiniteNumber } from "./validation";

/** Structured grid of scalar values for contour computation. */
export type ContourGrid = {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;
  readonly xCoords: Float64Array;
  readonly yCoords: Float64Array;
  readonly dataMin: number;
  readonly dataMax: number;
};

/** Bounding rectangle for contour plot coordinates. */
export type ContourExtent = {
  readonly xmin: number;
  readonly xmax: number;
  readonly ymin: number;
  readonly ymax: number;
};

function tensorSize(t: AnyTensor): number {
  const tensor = "tensor" in t ? t.tensor : t;
  if (tensor.ndim === 0) return 1;
  let size = 1;
  for (const dim of tensor.shape) {
    const d = dim ?? 0;
    if (d === 0) return 0;
    size *= d;
  }
  return size;
}

function assertFiniteCoords(values: Float64Array, name: string): void {
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? 0;
    if (!isFiniteNumber(v)) {
      throw new InvalidParameterError(`${name} coordinates must be finite`, name, v);
    }
  }
}

function assertStrictlyMonotonic(values: Float64Array, name: string): void {
  if (values.length < 2) return;
  const direction = Math.sign((values[1] ?? 0) - (values[0] ?? 0));
  for (let i = 1; i < values.length; i++) {
    const diff = (values[i] ?? 0) - (values[i - 1] ?? 0);
    if (direction === 0 || Math.sign(diff) !== direction) {
      throw new InvalidParameterError(
        `${name} coordinates must be strictly increasing or strictly decreasing; ` +
          `found ${values[i - 1]} followed by ${values[i]} at index ${i}`,
        name,
        values[i]
      );
    }
  }
}

/** Smallest and largest finite value of `data`; NaN and infinite values are ignored. */
function computeDataRange(data: Float64Array): {
  readonly min: number;
  readonly max: number;
} {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] ?? 0;
    if (!isFiniteNumber(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    throw new InvalidParameterError("Contour requires at least one finite Z value", "Z", data);
  }
  return { min, max };
}

function meshgridToCoords(
  X: AnyTensor,
  Y: AnyTensor,
  rows: number,
  cols: number
): { readonly xCoords: Float64Array; readonly yCoords: Float64Array } {
  const xMat = tensorToFloat64Matrix2D(X);
  const yMat = tensorToFloat64Matrix2D(Y);
  if (xMat.rows !== rows || xMat.cols !== cols) {
    throw new ShapeError("X must match Z shape for meshgrid input");
  }
  if (yMat.rows !== rows || yMat.cols !== cols) {
    throw new ShapeError("Y must match Z shape for meshgrid input");
  }

  const xCoords = new Float64Array(cols);
  const yCoords = new Float64Array(rows);
  for (let j = 0; j < cols; j++) {
    xCoords[j] = xMat.data[j] ?? 0;
  }
  for (let i = 0; i < rows; i++) {
    yCoords[i] = xMat.cols > 0 ? (yMat.data[i * cols] ?? 0) : 0;
  }

  // Relative tolerance: absolute 1e-12 would reject grids with large coordinates.
  const tolFor = (expected: number): number => 1e-12 * Math.max(1, Math.abs(expected));
  for (let i = 0; i < rows; i++) {
    const rowOffset = i * cols;
    for (let j = 0; j < cols; j++) {
      const expectedX = xCoords[j] ?? 0;
      const expectedY = yCoords[i] ?? 0;
      const xv = xMat.data[rowOffset + j] ?? 0;
      const yv = yMat.data[rowOffset + j] ?? 0;
      if (!isFiniteNumber(xv) || !isFiniteNumber(yv)) {
        throw new InvalidParameterError("X/Y meshgrid values must be finite", "X/Y", {
          x: xv,
          y: yv,
        });
      }
      if (
        Math.abs(xv - expectedX) > tolFor(expectedX) ||
        Math.abs(yv - expectedY) > tolFor(expectedY)
      ) {
        throw new InvalidParameterError(
          "Contour supports only rectilinear grids (use 1D X/Y or meshgrid)",
          "X/Y",
          { x: xv, y: yv }
        );
      }
    }
  }

  assertFiniteCoords(xCoords, "X");
  assertFiniteCoords(yCoords, "Y");
  assertStrictlyMonotonic(xCoords, "X");
  assertStrictlyMonotonic(yCoords, "Y");

  return { xCoords, yCoords };
}

/**
 * Builds a rectilinear contour grid from X/Y/Z tensors.
 *
 * `Z` is a `rows x cols` matrix. `X` and `Y` are either 1D coordinate vectors (length `cols`
 * and `rows`), 2D meshgrids matching `Z` (`indexing="xy"`, so X varies along columns), or both
 * empty, in which case the coordinates come from `extent` (or the integer indices). Coordinates
 * must be finite and strictly monotonic.
 * @throws {InvalidParameterError} For non-finite or non-monotonic coordinates, a non-rectilinear
 *   meshgrid, an invalid `extent`, or a `Z` without any finite value.
 * @throws {ShapeError} If the shapes of `X`, `Y` and `Z` disagree.
 * @internal
 */
export function buildContourGrid(
  X: AnyTensor,
  Y: AnyTensor,
  Z: AnyTensor,
  extent?: ContourExtent
): ContourGrid {
  const zMat = tensorToFloat64Matrix2D(Z);
  const { rows, cols, data } = zMat;
  const { min, max } = computeDataRange(data);

  const xSize = tensorSize(X);
  const ySize = tensorSize(Y);

  let xCoords: Float64Array;
  let yCoords: Float64Array;

  if (xSize === 0 && ySize === 0) {
    xCoords = new Float64Array(cols);
    yCoords = new Float64Array(rows);
    if (extent) {
      const { xmin, xmax, ymin, ymax } = extent;
      if (
        !Number.isFinite(xmin) ||
        !Number.isFinite(xmax) ||
        !Number.isFinite(ymin) ||
        !Number.isFinite(ymax)
      ) {
        throw new InvalidParameterError("extent values must be finite", "extent", extent);
      }
      if (xmax <= xmin || ymax <= ymin) {
        throw new InvalidParameterError("extent ranges must be positive", "extent", extent);
      }
      const xDenom = Math.max(1, cols - 1);
      const yDenom = Math.max(1, rows - 1);
      for (let j = 0; j < cols; j++) {
        xCoords[j] = xmin + ((xmax - xmin) * j) / xDenom;
      }
      for (let i = 0; i < rows; i++) {
        yCoords[i] = ymin + ((ymax - ymin) * i) / yDenom;
      }
    } else {
      for (let j = 0; j < cols; j++) xCoords[j] = j;
      for (let i = 0; i < rows; i++) yCoords[i] = i;
    }
  } else if (xSize > 0 && ySize > 0) {
    if (extent) {
      throw new InvalidParameterError(
        "extent cannot be combined with explicit X/Y coordinates",
        "extent",
        extent
      );
    }
    const rawX = "tensor" in X ? X.tensor : X;
    const rawY = "tensor" in Y ? Y.tensor : Y;
    if (rawX.ndim === 1 && rawY.ndim === 1) {
      xCoords = tensorToFloat64Vector1D(X);
      yCoords = tensorToFloat64Vector1D(Y);
      if (xCoords.length !== cols) {
        throw new ShapeError(`X must have length ${cols}; received ${xCoords.length}`);
      }
      if (yCoords.length !== rows) {
        throw new ShapeError(`Y must have length ${rows}; received ${yCoords.length}`);
      }
      assertFiniteCoords(xCoords, "X");
      assertFiniteCoords(yCoords, "Y");
      assertStrictlyMonotonic(xCoords, "X");
      assertStrictlyMonotonic(yCoords, "Y");
    } else if (rawX.ndim === 2 && rawY.ndim === 2) {
      ({ xCoords, yCoords } = meshgridToCoords(X, Y, rows, cols));
    } else {
      throw new ShapeError("X and Y must be 1D vectors or 2D meshgrids matching Z");
    }
  } else {
    throw new InvalidParameterError("Both X and Y must be provided or both empty", "X/Y", {
      xSize,
      ySize,
    });
  }

  return {
    rows,
    cols,
    data,
    xCoords,
    yCoords,
    dataMin: min,
    dataMax: max,
  };
}
