/**
 * Interactive plot support for Deepbox.
 *
 * Generates standalone HTML files with embedded JavaScript for
 * client-side interactivity. Zero external dependencies — all
 * interaction logic is inlined in the HTML output.
 *
 * Supported interactions:
 * - Pan (click + drag)
 * - Zoom (scroll wheel)
 * - Tooltips (hover over data points)
 * - Crosshair cursor
 * - Reset view (double-click)
 *
 * @module plot/interactive
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core/errors/invalid_parameter";
import type { Figure } from "../figure/Figure";

/**
 * Options for interactive plot rendering.
 */
export type InteractiveOptions = {
  /** Enable pan interaction. Default: true. */
  readonly pan?: boolean;
  /** Enable zoom interaction. Default: true. */
  readonly zoom?: boolean;
  /** Enable tooltips on hover. Default: true. */
  readonly tooltips?: boolean;
  /** Enable crosshair cursor. Default: false. */
  readonly crosshair?: boolean;
  /** Enable reset on double-click. Default: true. */
  readonly resetOnDoubleClick?: boolean;
  /** Minimum zoom level. Default: 0.1. */
  readonly minZoom?: number;
  /** Maximum zoom level. Default: 10. */
  readonly maxZoom?: number;
  /** Custom CSS to inject. */
  readonly customCSS?: string;
  /** HTML page title. Default: "Deepbox Plot". */
  readonly title?: string;
};

/**
 * Data point for tooltip display.
 */
export type TooltipDataPoint = {
  readonly x: number;
  readonly y: number;
  readonly label?: string;
  readonly series?: string;
};

/**
 * Result of rendering an interactive plot.
 */
export type InteractiveResult = {
  /** The complete HTML document string. */
  readonly html: string;
  /** The SVG content embedded in the HTML. */
  readonly svg: string;
  /** Options used for rendering. */
  readonly options: Required<Omit<InteractiveOptions, "customCSS" | "title">>;
};

/**
 * Interactive plot wrapper for Deepbox figures.
 *
 * Wraps a static SVG plot in an HTML document with embedded JavaScript
 * that provides pan, zoom, tooltip, and crosshair interactions.
 *
 * @example
 * ```ts
 * import { InteractivePlot, figure } from 'deepbox/plot';
 *
 * const fig = figure({ width: 800, height: 600 });
 * const ax = fig.addAxes();
 * ax.plot([1, 2, 3, 4], [10, 20, 15, 25]);
 *
 * const interactive = new InteractivePlot(fig, {
 *   tooltips: true,
 *   crosshair: true,
 *   zoom: true,
 *   pan: true,
 * });
 *
 * // Add tooltip data points
 * interactive.addDataPoints([
 *   { x: 1, y: 10, label: "Point A" },
 *   { x: 2, y: 20, label: "Point B" },
 * ]);
 *
 * const result = interactive.render();
 * // Write result.html to a file and open in a browser
 * ```
 */
export class InteractivePlot {
  private readonly fig: Figure;
  private readonly options: Required<Omit<InteractiveOptions, "customCSS" | "title">>;
  private readonly customCSS: string;
  private readonly title: string;
  private dataPoints: TooltipDataPoint[] = [];

  constructor(fig: Figure, options: InteractiveOptions = {}) {
    this.fig = fig;
    this.options = {
      pan: options.pan ?? true,
      zoom: options.zoom ?? true,
      tooltips: options.tooltips ?? true,
      crosshair: options.crosshair ?? false,
      resetOnDoubleClick: options.resetOnDoubleClick ?? true,
      minZoom: options.minZoom ?? 0.1,
      maxZoom: options.maxZoom ?? 10,
    };
    this.customCSS = options.customCSS ?? "";
    this.title = options.title ?? "Deepbox Plot";

    if (this.options.minZoom <= 0) {
      throw new InvalidParameterError("minZoom must be positive", "minZoom", this.options.minZoom);
    }
    if (this.options.maxZoom < this.options.minZoom) {
      throw new InvalidParameterError(
        "maxZoom must be >= minZoom",
        "maxZoom",
        this.options.maxZoom
      );
    }
  }

  /**
   * Add data points for tooltip display.
   *
   * @param points - Array of data points with x, y coordinates and optional labels
   */
  addDataPoints(points: readonly TooltipDataPoint[]): void {
    this.dataPoints.push(...points);
  }

  /**
   * Clear all registered data points.
   */
  clearDataPoints(): void {
    this.dataPoints = [];
  }

  /**
   * Render the interactive plot as a standalone HTML document.
   *
   * @returns Interactive rendering result
   */
  render(): InteractiveResult {
    const svgResult = this.fig.renderSVG();
    const svg = svgResult.svg;

    const html = this.buildHTML(svg);

    return {
      html,
      svg,
      options: this.options,
    };
  }

  private buildHTML(svg: string): string {
    const { pan, zoom, tooltips, crosshair, resetOnDoubleClick, minZoom, maxZoom } = this.options;
    // Escape characters that could break out of the inline <script> element
    // or the JS string context. JSON.stringify alone does NOT escape "</script>"
    // or "<", so a data-point label containing markup would execute (XSS).
    const dataPointsJSON = JSON.stringify(this.dataPoints)
      .replace(/</g, "\\u003c")
      .replace(/>/g, "\\u003e")
      .replace(/&/g, "\\u0026")
      .replace(/\u2028/g, "\\u2028")
      .replace(/\u2029/g, "\\u2029");

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${escapeHtml(this.title)}</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #f5f5f5; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
#plot-container { position: relative; display: inline-block; background: white; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); overflow: hidden; cursor: ${pan ? "grab" : "default"}; }
#plot-container.dragging { cursor: grabbing; }
#plot-container svg { display: block; }
#tooltip { position: absolute; pointer-events: none; background: rgba(0,0,0,0.85); color: white; padding: 6px 10px; border-radius: 4px; font-size: 12px; line-height: 1.4; white-space: nowrap; display: none; z-index: 10; }
#crosshair-h, #crosshair-v { position: absolute; pointer-events: none; background: rgba(100,100,100,0.3); display: none; z-index: 5; }
#crosshair-h { left: 0; right: 0; height: 1px; }
#crosshair-v { top: 0; bottom: 0; width: 1px; }
#controls { position: absolute; top: 8px; right: 8px; display: flex; gap: 4px; z-index: 20; }
#controls button { background: rgba(255,255,255,0.9); border: 1px solid #ddd; border-radius: 4px; width: 28px; height: 28px; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; }
#controls button:hover { background: #eee; }
${this.customCSS}
</style>
</head>
<body>
<div id="plot-container">
${svg}
<div id="tooltip"></div>
<div id="crosshair-h"></div>
<div id="crosshair-v"></div>
<div id="controls">
${zoom ? '<button id="zoom-in" title="Zoom In">+</button><button id="zoom-out" title="Zoom Out">−</button>' : ""}
<button id="reset" title="Reset View">⟲</button>
</div>
</div>
<script>
(function() {
  const container = document.getElementById('plot-container');
  const svgEl = container.querySelector('svg');
  const tooltip = document.getElementById('tooltip');
  const crosshairH = document.getElementById('crosshair-h');
  const crosshairV = document.getElementById('crosshair-v');
  const dataPoints = ${dataPointsJSON};

  let scale = 1;
  let translateX = 0;
  let translateY = 0;
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let startTX = 0;
  let startTY = 0;

  const minZoom = ${minZoom};
  const maxZoom = ${maxZoom};

  function updateTransform() {
    svgEl.style.transform = 'translate(' + translateX + 'px, ' + translateY + 'px) scale(' + scale + ')';
    svgEl.style.transformOrigin = '0 0';
  }

  function resetView() {
    scale = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();
  }

  ${
    pan
      ? `
  container.addEventListener('mousedown', function(e) {
    if (e.button !== 0) return;
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    startTX = translateX;
    startTY = translateY;
    container.classList.add('dragging');
    e.preventDefault();
  });

  window.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    translateX = startTX + (e.clientX - startX);
    translateY = startTY + (e.clientY - startY);
    updateTransform();
  });

  window.addEventListener('mouseup', function() {
    isDragging = false;
    container.classList.remove('dragging');
  });
  `
      : ""
  }

  ${
    zoom
      ? `
  container.addEventListener('wheel', function(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.min(maxZoom, Math.max(minZoom, scale * delta));
    const rect = container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    translateX = mouseX - (mouseX - translateX) * (newScale / scale);
    translateY = mouseY - (mouseY - translateY) * (newScale / scale);
    scale = newScale;
    updateTransform();
  }, { passive: false });

  var zoomInBtn = document.getElementById('zoom-in');
  var zoomOutBtn = document.getElementById('zoom-out');
  if (zoomInBtn) zoomInBtn.addEventListener('click', function() {
    scale = Math.min(maxZoom, scale * 1.2);
    updateTransform();
  });
  if (zoomOutBtn) zoomOutBtn.addEventListener('click', function() {
    scale = Math.max(minZoom, scale / 1.2);
    updateTransform();
  });
  `
      : ""
  }

  ${
    tooltips
      ? `
  container.addEventListener('mousemove', function(e) {
    if (isDragging || dataPoints.length === 0) {
      tooltip.style.display = 'none';
      return;
    }
    var rect = svgEl.getBoundingClientRect();
    var mx = (e.clientX - rect.left) / scale;
    var my = (e.clientY - rect.top) / scale;
    var closest = null;
    var minDist = Infinity;
    for (var i = 0; i < dataPoints.length; i++) {
      var dx = dataPoints[i].x - mx;
      var dy = dataPoints[i].y - my;
      var d = Math.sqrt(dx*dx + dy*dy);
      if (d < minDist) { minDist = d; closest = dataPoints[i]; }
    }
    if (closest && minDist < 50) {
      var label = closest.label || ('(' + closest.x.toFixed(2) + ', ' + closest.y.toFixed(2) + ')');
      if (closest.series) label = closest.series + ': ' + label;
      tooltip.textContent = label;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX - container.getBoundingClientRect().left + 12) + 'px';
      tooltip.style.top = (e.clientY - container.getBoundingClientRect().top - 28) + 'px';
    } else {
      tooltip.style.display = 'none';
    }
  });

  container.addEventListener('mouseleave', function() {
    tooltip.style.display = 'none';
  });
  `
      : ""
  }

  ${
    crosshair
      ? `
  container.addEventListener('mousemove', function(e) {
    var rect = container.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    crosshairH.style.display = 'block';
    crosshairV.style.display = 'block';
    crosshairH.style.top = y + 'px';
    crosshairV.style.left = x + 'px';
  });
  container.addEventListener('mouseleave', function() {
    crosshairH.style.display = 'none';
    crosshairV.style.display = 'none';
  });
  `
      : ""
  }

  ${
    resetOnDoubleClick
      ? `
  container.addEventListener('dblclick', resetView);
  `
      : ""
  }

  document.getElementById('reset').addEventListener('click', resetView);
})();
</script>
</body>
</html>`;
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Create an interactive plot from a Figure.
 *
 * @param fig - The figure to make interactive
 * @param options - Interactive options
 * @returns A new InteractivePlot instance
 */
export function createInteractivePlot(fig: Figure, options?: InteractiveOptions): InteractivePlot {
  return new InteractivePlot(fig, options);
}
