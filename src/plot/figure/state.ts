import { InvalidParameterError } from "../../core";
import type { Color } from "../types";
import type { Axes } from "./Axes";
import { Figure } from "./Figure";

let _currentFigure: Figure | null = null;
let _currentAxes: Axes | null = null;

function gcf(): Figure {
  if (!_currentFigure) {
    _currentFigure = new Figure({ width: 320, height: 240 });
  }
  return _currentFigure;
}

/**
 * Get the current axes, creating one if needed.
 */
export function gca(): Axes {
  const fig = gcf();
  if (_currentAxes && fig.axesList.includes(_currentAxes)) return _currentAxes;
  if (fig.axesList.length === 0) {
    _currentAxes = fig.addAxes();
    return _currentAxes;
  }
  const firstAxes = fig.axesList[0];
  if (!firstAxes) {
    _currentAxes = fig.addAxes();
    return _currentAxes;
  }
  _currentAxes = firstAxes;
  return _currentAxes;
}

/**
 * Create a new figure and set it as current.
 */
export function figure(
  options: { readonly width?: number; readonly height?: number; readonly background?: Color } = {}
): Figure {
  _currentFigure = new Figure({
    width: options.width ?? 320,
    height: options.height ?? 240,
    ...(options.background !== undefined && { background: options.background }),
  });
  _currentAxes = _currentFigure.addAxes();
  return _currentFigure;
}

/**
 * Create a subplot and set it as current axes.
 */
export function subplot(
  rows: number,
  cols: number,
  index: number,
  options: { readonly padding?: number; readonly facecolor?: Color } = {}
): Axes {
  if (!Number.isFinite(rows) || Math.trunc(rows) !== rows || rows <= 0) {
    throw new InvalidParameterError(
      `rows must be a positive integer; received ${rows}`,
      "rows",
      rows
    );
  }
  if (!Number.isFinite(cols) || Math.trunc(cols) !== cols || cols <= 0) {
    throw new InvalidParameterError(
      `cols must be a positive integer; received ${cols}`,
      "cols",
      cols
    );
  }
  const total = rows * cols;
  if (total > 10000) {
    throw new InvalidParameterError(
      `Subplot grid too large (${rows}×${cols}=${total}). Maximum is 10,000 subplots.`,
      "rows*cols",
      total
    );
  }
  if (!Number.isFinite(index) || Math.trunc(index) !== index || index < 1 || index > total) {
    throw new InvalidParameterError(
      `index must be in [1, ${total}]; received ${index}`,
      "index",
      index
    );
  }

  const fig = gcf();
  const idx0 = index - 1;
  const row = Math.floor(idx0 / cols);
  const col = idx0 % cols;
  const cellW = fig.width / cols;
  const cellH = fig.height / rows;
  const viewport = {
    x: col * cellW,
    y: row * cellH,
    width: cellW,
    height: cellH,
  };
  const ax = fig.addAxes({ ...options, viewport });
  _currentAxes = ax;
  return ax;
}
