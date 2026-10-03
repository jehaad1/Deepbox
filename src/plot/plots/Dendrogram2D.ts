/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
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
 * down to the child nodes. Leaves are placed left to right in the order
 * given by a depth-first walk from the root (first child before second
 * child), the same order as `scipy.cluster.hierarchy.dendrogram`, so the
 * U shapes never cross. The walk order is exposed as {@link Dendrogram2D.leaves}.
 */
export class Dendrogram2D implements Drawable {
  readonly kind = "dendrogram";
  /**
   * Original observation indices in the left-to-right order they are drawn.
   * Leaf `leaves[k]` sits at x = k.
   */
  readonly leaves: readonly number[];
  private readonly color: Color;
  private readonly linewidth: number;

  // Precomputed drawing segments: [x0, y0, x1, y1]
  private readonly segments: Array<readonly [number, number, number, number]>;
  private readonly xRange: [number, number];
  private readonly yRange: [number, number];

  constructor(linkage: readonly LinkageRow[], nLeaves: number, options: PlotOptions) {
    if (!Number.isInteger(nLeaves) || nLeaves <= 0) {
      throw new InvalidParameterError(
        `nLeaves must be a positive integer; received ${nLeaves}`,
        "nLeaves",
        nLeaves
      );
    }
    this.color = normalizeColor(options.color, "#1f77b4");
    this.linewidth = 1.5;

    const merges = linkage.length;
    const total = nLeaves + merges;
    const left = new Int32Array(merges);
    const right = new Int32Array(merges);
    const height = new Float64Array(total); // leaves stay at 0
    const hasParent = new Uint8Array(total);

    for (let i = 0; i < merges; i++) {
      const row = linkage[i];
      const a = row?.[0];
      const b = row?.[1];
      const dist = row?.[2];
      if (a === undefined || b === undefined || dist === undefined) {
        throw new InvalidParameterError(
          `linkage row ${i} must have 4 entries [a, b, distance, count]`,
          "linkage",
          row
        );
      }
      const limit = nLeaves + i;
      for (const id of [a, b]) {
        if (!Number.isInteger(id) || id < 0 || id >= limit) {
          throw new InvalidParameterError(
            `linkage row ${i} references cluster ${id}; ids must be integers in [0, ${limit})`,
            "linkage",
            row
          );
        }
        if (hasParent[id] === 1) {
          throw new InvalidParameterError(
            `linkage row ${i} merges cluster ${id} more than once`,
            "linkage",
            row
          );
        }
      }
      if (a === b) {
        throw new InvalidParameterError(
          `linkage row ${i} merges cluster ${a} with itself`,
          "linkage",
          row
        );
      }
      if (!Number.isFinite(dist)) {
        throw new InvalidParameterError(
          `linkage row ${i} has a non-finite distance (${dist})`,
          "linkage",
          row
        );
      }
      hasParent[a] = 1;
      hasParent[b] = 1;
      left[i] = a;
      right[i] = b;
      height[nLeaves + i] = dist;
    }

    // Depth-first leaf order (iterative, trees can be as deep as the leaf count).
    // A complete linkage has one root; an incomplete one is a forest, whose trees are
    // laid out in order of their smallest leaf index.
    const nodeX = new Float64Array(total);
    const leafOrder: number[] = [];
    const minLeaf = new Int32Array(total);
    for (let i = 0; i < nLeaves; i++) minLeaf[i] = i;
    for (let i = 0; i < merges; i++) {
      minLeaf[nLeaves + i] = Math.min(minLeaf[left[i] ?? 0] ?? 0, minLeaf[right[i] ?? 0] ?? 0);
    }
    const roots: number[] = [];
    for (let node = 0; node < total; node++) {
      if (hasParent[node] === 0) roots.push(node);
    }
    roots.sort((p, q) => (minLeaf[q] ?? 0) - (minLeaf[p] ?? 0));
    // Reverse order on a LIFO stack: the root with the smallest leaf pops first.
    const stack: number[] = roots;
    while (stack.length > 0) {
      const node = stack.pop();
      if (node === undefined) break;
      if (node < nLeaves) {
        nodeX[node] = leafOrder.length;
        leafOrder.push(node);
      } else {
        const m = node - nLeaves;
        stack.push(right[m] ?? 0);
        stack.push(left[m] ?? 0);
      }
    }
    this.leaves = leafOrder;

    const segs: Array<readonly [number, number, number, number]> = [];
    for (let i = 0; i < merges; i++) {
      const a = left[i] ?? 0;
      const b = right[i] ?? 0;
      const node = nLeaves + i;
      const xA = nodeX[a] ?? 0;
      const xB = nodeX[b] ?? 0;
      const yA = height[a] ?? 0;
      const yB = height[b] ?? 0;
      const dist = height[node] ?? 0;
      nodeX[node] = (xA + xB) / 2;

      // U shape: left vertical, horizontal bar, right vertical.
      segs.push([xA, yA, xA, dist]);
      segs.push([xA, dist, xB, dist]);
      segs.push([xB, yB, xB, dist]);
    }
    this.segments = segs;

    // Axis ranges follow the data; leaves occupy x = 0 .. nLeaves - 1.
    let ymin = 0;
    let ymax = 0;
    for (const [, y0, , y1] of segs) {
      ymin = Math.min(ymin, y0, y1);
      ymax = Math.max(ymax, y0, y1);
    }
    if (ymax <= 0) ymax = 1;
    this.xRange = [-0.5, Math.max(nLeaves - 1, 1) + 0.5];
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
