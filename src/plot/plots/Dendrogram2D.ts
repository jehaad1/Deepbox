/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import type {
  Color,
  DataRange,
  Drawable,
  LegendEntry,
  PlotOptions,
  RasterDrawContext,
  SvgDrawContext,
} from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { escapeXml } from "../utils/xml";

/**
 * Linkage row: [clusterA, clusterB, distance, count].
 */
export type LinkageRow = readonly [number, number, number, number];

/**
 * @internal
 * Draws a dendrogram from a linkage matrix.
 *
 * Each merge in the linkage matrix is rendered as a U-shaped connection:
 * a horizontal bar at the merge height connecting two vertical lines
 * down to the child nodes.
 */
export class Dendrogram2D implements Drawable {
  readonly kind = "dendrogram";
  private readonly color: Color;
  private readonly linewidth: number;

  // Precomputed drawing segments: [x0, y0, x1, y1]
  private readonly segments: Array<readonly [number, number, number, number]>;
  private readonly xRange: [number, number];
  private readonly yRange: [number, number];

  constructor(linkage: readonly LinkageRow[], nLeaves: number, options: PlotOptions) {
    this.color = normalizeColor(options.color, "#1f77b4");
    this.linewidth = 1.5;

    // Compute x-positions of each node (leaves are evenly spaced)
    const nodeX = new Map<number, number>();
    const nodeY = new Map<number, number>();

    // Leaves at y=0, evenly spaced from 0..nLeaves-1
    for (let i = 0; i < nLeaves; i++) {
      nodeX.set(i, i);
      nodeY.set(i, 0);
    }

    const segs: Array<readonly [number, number, number, number]> = [];

    for (let i = 0; i < linkage.length; i++) {
      const row = linkage[i]!;
      const [a, b, dist] = row;
      const newNode = nLeaves + i;

      const xA = nodeX.get(a) ?? 0;
      const yA = nodeY.get(a) ?? 0;
      const xB = nodeX.get(b) ?? 0;
      const yB = nodeY.get(b) ?? 0;

      // The merged node's x is the midpoint
      const xMid = (xA + xB) / 2;

      nodeX.set(newNode, xMid);
      nodeY.set(newNode, dist);

      // Draw U-shape: vertical from A up to dist, horizontal across, vertical down to B
      // Left vertical: (xA, yA) -> (xA, dist)
      segs.push([xA, yA, xA, dist]);
      // Horizontal: (xA, dist) -> (xB, dist)
      segs.push([xA, dist, xB, dist]);
      // Right vertical: (xB, yB) -> (xB, dist)
      segs.push([xB, yB, xB, dist]);
    }

    this.segments = segs;

    // Compute ranges
    let xmin = 0;
    let xmax = Math.max(nLeaves - 1, 1);
    let ymin = 0;
    let ymax = 1;
    for (const [x0, y0, x1, y1] of segs) {
      xmin = Math.min(xmin, x0, x1);
      xmax = Math.max(xmax, x0, x1);
      ymin = Math.min(ymin, y0, y1);
      ymax = Math.max(ymax, y0, y1);
    }
    this.xRange = [xmin - 0.5, xmax + 0.5];
    this.yRange = [ymin, ymax * 1.05];
  }

  getDataRange(): DataRange | null {
    return {
      xmin: this.xRange[0],
      xmax: this.xRange[1],
      ymin: this.yRange[0],
      ymax: this.yRange[1],
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    for (const [x0, y0, x1, y1] of this.segments) {
      const px0 = ctx.transform.xToPx(x0);
      const py0 = ctx.transform.yToPx(y0);
      const px1 = ctx.transform.xToPx(x1);
      const py1 = ctx.transform.yToPx(y1);
      ctx.push(
        `<line x1="${px0.toFixed(2)}" y1="${py0.toFixed(2)}" x2="${px1.toFixed(2)}" y2="${py1.toFixed(2)}" stroke="${escapeXml(this.color)}" stroke-width="${this.linewidth}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const rgba = parseHexColorToRGBA(this.color);
    for (const [x0, y0, x1, y1] of this.segments) {
      const px0 = Math.round(ctx.transform.xToPx(x0));
      const py0 = Math.round(ctx.transform.yToPx(y0));
      const px1 = Math.round(ctx.transform.xToPx(x1));
      const py1 = Math.round(ctx.transform.yToPx(y1));
      ctx.canvas.drawLineRGBA(px0, py0, px1, py1, rgba.r, rgba.g, rgba.b, rgba.a);
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    return null;
  }
}
