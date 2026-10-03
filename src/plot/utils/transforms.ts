/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import type { DataRange, DataTransform, Drawable, Viewport } from "../types";

/**
 * Computes the combined data range of all drawables, padded by 5% on every side.
 *
 * Non-finite bounds reported by a drawable are ignored so a single NaN cannot discard the
 * extent of every other drawable. An axis with no finite bounds falls back to `[0, 1]`, and a
 * zero-width axis is widened around its value.
 * @internal
 */
export function computeAutoRange(drawables: readonly Drawable[]): DataRange {
  let xmin = Number.POSITIVE_INFINITY;
  let xmax = Number.NEGATIVE_INFINITY;
  let ymin = Number.POSITIVE_INFINITY;
  let ymax = Number.NEGATIVE_INFINITY;

  for (const d of drawables) {
    const r = d.getDataRange();
    if (!r) continue;
    if (Number.isFinite(r.xmin)) xmin = Math.min(xmin, r.xmin);
    if (Number.isFinite(r.xmax)) xmax = Math.max(xmax, r.xmax);
    if (Number.isFinite(r.ymin)) ymin = Math.min(ymin, r.ymin);
    if (Number.isFinite(r.ymax)) ymax = Math.max(ymax, r.ymax);
  }

  if (!Number.isFinite(xmin) || !Number.isFinite(xmax)) {
    xmin = 0;
    xmax = 1;
  } else if (xmin === xmax) {
    const span = Math.max(1, Math.abs(xmin) * 0.05);
    xmin -= span;
    xmax += span;
  }
  if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) {
    ymin = 0;
    ymax = 1;
  } else if (ymin === ymax) {
    const span = Math.max(1, Math.abs(ymin) * 0.05);
    ymin -= span;
    ymax += span;
  }

  const xPad = (xmax - xmin) * 0.05;
  const yPad = (ymax - ymin) * 0.05;
  return {
    xmin: xmin - xPad,
    xmax: xmax + xPad,
    ymin: ymin - yPad,
    ymax: ymax + yPad,
  };
}

/**
 * Creates the data-to-pixel transform for a range drawn into a viewport. The y axis is
 * flipped (larger data values map to smaller pixel rows). A zero-width or non-finite range
 * collapses that axis onto the viewport's origin edge.
 * @internal
 */
export function makeTransform(range: DataRange, viewport: Viewport): DataTransform {
  const dx = range.xmax - range.xmin;
  const dy = range.ymax - range.ymin;
  const sx = Number.isFinite(dx) && dx !== 0 ? viewport.width / dx : 0;
  const sy = Number.isFinite(dy) && dy !== 0 ? viewport.height / dy : 0;

  return {
    xToPx: (x) => viewport.x + (x - range.xmin) * sx,
    yToPx: (y) => viewport.y + viewport.height - (y - range.ymin) * sy,
  };
}

/**
 * Smallest strictly positive x and y coordinate among the points whose x and y are both finite
 * (`Infinity` for a coordinate without a positive value). Used by log axes to start at the
 * smallest value that can be shown. The two arrays may differ in length; the shorter one wins.
 * @internal
 */
export function positiveMinOfPoints(
  x: ArrayLike<number>,
  y: ArrayLike<number>
): { readonly x: number; readonly y: number } {
  let px = Number.POSITIVE_INFINITY;
  let py = Number.POSITIVE_INFINITY;
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i++) {
    const xi = x[i] ?? Number.NaN;
    const yi = y[i] ?? Number.NaN;
    if (!Number.isFinite(xi) || !Number.isFinite(yi)) continue;
    if (xi > 0 && xi < px) px = xi;
    if (yi > 0 && yi < py) py = yi;
  }
  return { x: px, y: py };
}
