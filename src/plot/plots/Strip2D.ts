/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type {
  Color,
  DataRange,
  Drawable,
  LegendEntry,
  RasterDrawContext,
  SvgDrawContext,
} from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { buildLegendEntry, normalizeLegendLabel } from "../utils/legend";
import { isFiniteNumber } from "../utils/validation";
import { escapeXml } from "../utils/xml";

/**
 * Deterministic pseudo-random sequence in [0, 1) for jitter (32-bit LCG with the
 * glibc constants, so SVG and raster output place every point identically).
 */
function createJitterSource(): () => number {
  let seed = 12345;
  return () => {
    seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
}

/**
 * Strip plot: jittered categorical scatter plot.
 * Group `g` is plotted at x = g with horizontal jitter of up to `jitter` data
 * units on either side. The jitter is deterministic, so a figure renders the
 * same every time.
 * @internal
 */
export class Strip2D implements Drawable {
  readonly kind = "strip";
  readonly groups: Float64Array[];
  readonly colors: Color[];
  readonly labels: (string | null)[];
  readonly markerSize: number;
  readonly jitter: number;

  constructor(
    groups: Float64Array[],
    options: {
      colors?: readonly Color[];
      labels?: readonly string[];
      size?: number;
      jitter?: number;
    } = {}
  ) {
    if (groups.length === 0)
      throw new InvalidParameterError("At least one group is required", "groups", 0);
    this.groups = groups;
    const size = options.size ?? 3;
    if (!Number.isFinite(size) || size <= 0) {
      throw new InvalidParameterError(
        `size must be a positive number; received ${size}`,
        "size",
        size
      );
    }
    this.markerSize = size;
    const jitter = options.jitter ?? 0.2;
    if (!Number.isFinite(jitter) || jitter < 0) {
      throw new InvalidParameterError(
        `jitter must be a non-negative number; received ${jitter}`,
        "jitter",
        jitter
      );
    }
    this.jitter = jitter;
    this.colors = groups.map((_, i) =>
      normalizeColor(
        options.colors?.[i],
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][i % 6] ?? "#1f77b4"
      )
    );
    this.labels = groups.map((_, i) => normalizeLegendLabel(options.labels?.[i]));
  }

  getDataRange(): DataRange | null {
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const group of this.groups) {
      for (let i = 0; i < group.length; i++) {
        const v = group[i] ?? 0;
        if (!isFiniteNumber(v)) continue;
        ymin = Math.min(ymin, v);
        ymax = Math.max(ymax, v);
      }
    }
    if (!isFiniteNumber(ymin) || !isFiniteNumber(ymax)) return null;
    // Cover the category slots, widened if the jitter spills past half a slot.
    const pad = Math.max(0.5, this.jitter);
    return {
      xmin: -pad,
      xmax: this.groups.length - 1 + pad,
      ymin,
      ymax,
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const nextRand = createJitterSource();

    for (let g = 0; g < this.groups.length; g++) {
      const group = this.groups[g];
      if (!group) continue;
      const ec = escapeXml(this.colors[g] ?? "#1f77b4");
      for (let i = 0; i < group.length; i++) {
        const v = group[i] ?? 0;
        if (!isFiniteNumber(v)) continue;
        const jx = g + (nextRand() - 0.5) * 2 * this.jitter;
        const px = ctx.transform.xToPx(jx);
        const py = ctx.transform.yToPx(v);
        ctx.push(
          `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${this.markerSize}" fill="${ec}" opacity="0.7" />`
        );
      }
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const nextRand = createJitterSource();

    for (let g = 0; g < this.groups.length; g++) {
      const group = this.groups[g];
      if (!group) continue;
      const rgba = parseHexColorToRGBA(this.colors[g] ?? "#1f77b4");
      for (let i = 0; i < group.length; i++) {
        const v = group[i] ?? 0;
        if (!isFiniteNumber(v)) continue;
        const jx = g + (nextRand() - 0.5) * 2 * this.jitter;
        const px = Math.round(ctx.transform.xToPx(jx));
        const py = Math.round(ctx.transform.yToPx(v));
        ctx.canvas.drawCircleRGBA(
          px,
          py,
          this.markerSize,
          rgba.r,
          rgba.g,
          rgba.b,
          Math.round(rgba.a * 0.7)
        );
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entries: LegendEntry[] = [];
    for (let i = 0; i < this.groups.length; i++) {
      const entry = buildLegendEntry(this.labels[i] ?? null, {
        color: this.colors[i] ?? "#1f77b4",
        shape: "marker",
        markerSize: this.markerSize,
      });
      if (entry) entries.push(entry);
    }
    return entries.length > 0 ? entries : null;
  }
}
