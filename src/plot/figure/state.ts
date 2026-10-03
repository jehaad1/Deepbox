/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type { Color } from "../types";
import type { Axes } from "./Axes";
import { Figure } from "./Figure";

let _currentFigure: Figure | null = null;
let _currentAxes: Axes | null = null;
// Axes created implicitly by figure() or gca(). One that is still untouched when
// a subplot grid is requested is dropped, so it does not draw a stray full-size frame.
const _implicitAxes = new WeakSet<Axes>();
// Subplot position ("rows x cols : index") of axes created by subplot().
const _subplotSlots = new WeakMap<Axes, string>();

/**
 * Get the current figure, creating a 320 x 240 one if needed.
 */
export function gcf(): Figure {
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
    _implicitAxes.add(_currentAxes);
    return _currentAxes;
  }
  const firstAxes = fig.axesList[0];
  if (!firstAxes) {
    _currentAxes = fig.addAxes();
    _implicitAxes.add(_currentAxes);
    return _currentAxes;
  }
  _currentAxes = firstAxes;
  return _currentAxes;
}

/**
 * Make `ax` the current axes (like `plt.sca`). Later calls of the global helpers
 * such as `plot()` and `title()` draw on it. When the axes belongs to another
 * figure, that figure becomes the current figure too.
 * @throws {InvalidParameterError} If `ax` is not part of any figure's axes list
 *   (it was not created with `Figure.addAxes()` or `subplot()`).
 */
export function sca(ax: Axes): Axes {
  const owner = ax.fig;
  if (!owner.axesList.includes(ax)) {
    throw new InvalidParameterError(
      "sca: the axes does not belong to its figure (create it with Figure.addAxes())",
      "ax",
      ax
    );
  }
  _currentFigure = owner;
  _currentAxes = ax;
  return ax;
}

/**
 * Create a new figure and set it as current. The figure starts with one axes
 * that fills it (default size 320 x 240; `new Figure()` defaults to 640 x 480).
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
  _implicitAxes.add(_currentAxes);
  return _currentFigure;
}

/**
 * Create a subplot in a `rows` x `cols` grid on the current figure and make it
 * the current axes. Positions are numbered from 1, row by row, starting at the
 * top left. Asking again for a position that already has a subplot returns that
 * axes (and ignores `options`). The empty axes that `figure()` creates is
 * dropped when the first subplot is added.
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
  const slot = `${rows}x${cols}:${index}`;
  const existing = fig.axesList.find((a) => _subplotSlots.get(a) === slot);
  if (existing) {
    _currentAxes = existing;
    return existing;
  }
  for (let i = fig.axesList.length - 1; i >= 0; i--) {
    const candidate = fig.axesList[i];
    if (candidate && _implicitAxes.has(candidate) && candidate.isBlank()) {
      fig.axesList.splice(i, 1);
    }
  }
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
  _subplotSlots.set(ax, slot);
  _currentAxes = ax;
  return ax;
}
