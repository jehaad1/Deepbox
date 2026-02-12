import type { DataRange, DataTransform, Drawable, Viewport } from "../types";

/**
 * Computes data range from drawables.
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
    xmin = Math.min(xmin, r.xmin);
    xmax = Math.max(xmax, r.xmax);
    ymin = Math.min(ymin, r.ymin);
    ymax = Math.max(ymax, r.ymax);
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
 * Creates coordinate transformation.
 * @internal
 */
export function makeTransform(range: DataRange, viewport: Viewport): DataTransform {
  const dx = range.xmax - range.xmin;
  const dy = range.ymax - range.ymin;
  const sx = dx !== 0 ? viewport.width / dx : 0;
  const sy = dy !== 0 ? viewport.height / dy : 0;

  return {
    xToPx: (x) => viewport.x + (x - range.xmin) * sx,
    yToPx: (y) => viewport.y + viewport.height - (y - range.ymin) * sy,
  };
}
