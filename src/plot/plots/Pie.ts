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
import { normalizeLegendLabel } from "../utils/legend";
import { isFiniteNumber } from "../utils/validation";
import { escapeXml } from "../utils/xml";

/**
 * @internal
 */
export class Pie implements Drawable {
  readonly kind = "pie";
  readonly centerX: number;
  readonly centerY: number;
  readonly radius: number;
  readonly values: readonly number[];
  readonly angles: readonly number[];
  readonly colors: readonly Color[];
  readonly labels: readonly string[];
  readonly rangeOverride: DataRange | undefined;
  readonly label: string | null;

  constructor(
    centerX: number,
    centerY: number,
    radius: number,
    values: Float64Array,
    labels?: readonly string[],
    options: PlotOptions = {},
    rangeOverride?: DataRange
  ) {
    if (!Number.isFinite(centerX) || !Number.isFinite(centerY)) {
      throw new InvalidParameterError("pie center must be finite", "center", { centerX, centerY });
    }
    if (!Number.isFinite(radius) || radius <= 0) {
      throw new InvalidParameterError(
        `pie radius must be positive; received ${radius}`,
        "radius",
        radius
      );
    }
    this.centerX = centerX;
    this.centerY = centerY;
    this.radius = radius;
    if (values.length === 0) {
      throw new InvalidParameterError("pie requires at least one value", "values", values.length);
    }
    if (labels && labels.length !== values.length) {
      throw new InvalidParameterError(
        `labels length (${labels.length}) must match values length (${values.length})`,
        "labels",
        labels
      );
    }
    this.labels = labels ?? [];
    this.label = normalizeLegendLabel(options.label);
    this.rangeOverride = rangeOverride;

    // Filter and validate values

    const validValues: number[] = [];
    for (let i = 0; i < values.length; i++) {
      const v = values[i] ?? 0;
      if (!isFiniteNumber(v)) {
        throw new InvalidParameterError("pie values must be finite", "values", v);
      }
      if (v < 0) {
        throw new InvalidParameterError("pie values must be non-negative", "values", v);
      }
      validValues.push(v);
    }
    const total = validValues.reduce((sum, val) => sum + val, 0);
    if (!Number.isFinite(total) || total <= 0) {
      throw new InvalidParameterError(
        "pie values must sum to a positive number",
        "values",
        validValues
      );
    }

    this.values = validValues;

    // Calculate angles for each slice
    // angles[0] = start of first slice (0)
    // angles[i+1] = end of slice i (cumulative angle after slice i)
    const angles: number[] = [0];
    let cumulative = 0;
    for (const value of validValues) {
      cumulative += (value / total) * 2 * Math.PI;
      angles.push(cumulative);
    }
    this.angles = angles;

    // Generate colors
    const defaultColors = [
      "#1f77b4",
      "#ff7f0e",
      "#2ca02c",
      "#d62728",
      "#9467bd",
      "#8c564b",
      "#e377c2",
      "#7f7f7f",
      "#bcbd22",
      "#17becf",
    ];

    const colors: Color[] = [];
    if (options.colors && options.colors.length > 0) {
      for (let i = 0; i < validValues.length; i++) {
        const colorOption = options.colors[i % options.colors.length];
        const colorToUse: Color | undefined = colorOption;
        const defaultColor = defaultColors[i % defaultColors.length] ?? "#1f77b4";
        colors.push(normalizeColor(colorToUse, defaultColor));
      }
    } else {
      for (let i = 0; i < validValues.length; i++) {
        const defaultColor = defaultColors[i % defaultColors.length] ?? "#1f77b4";
        colors.push(normalizeColor(undefined, defaultColor));
      }
    }
    this.colors = colors;
  }

  getDataRange(): DataRange | null {
    if (this.rangeOverride) return this.rangeOverride;
    return {
      xmin: this.centerX - this.radius,
      xmax: this.centerX + this.radius,
      ymin: this.centerY - this.radius,
      ymax: this.centerY + this.radius,
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const cx = ctx.transform.xToPx(this.centerX);
    const cy = ctx.transform.yToPx(this.centerY);
    const rx = Math.abs(ctx.transform.xToPx(this.centerX + this.radius) - cx);
    const ry = Math.abs(ctx.transform.yToPx(this.centerY + this.radius) - cy);
    const r = Math.min(rx, ry);

    for (let i = 0; i < this.values.length; i++) {
      const startAngle = this.angles[i] ?? 0;
      const endAngle = this.angles[i + 1] ?? 2 * Math.PI;

      // Convert to SVG arc parameters
      const x1 = cx + r * Math.cos(startAngle);
      const y1 = cy - r * Math.sin(startAngle); // SVG Y is flipped
      const x2 = cx + r * Math.cos(endAngle);
      const y2 = cy - r * Math.sin(endAngle);

      const largeArcFlag = endAngle - startAngle > Math.PI ? 1 : 0;

      const pathData = [
        `M ${cx.toFixed(2)} ${cy.toFixed(2)}`,
        `L ${x1.toFixed(2)} ${y1.toFixed(2)}`,
        `A ${r.toFixed(2)} ${r.toFixed(2)} 0 ${largeArcFlag} 0 ${x2.toFixed(2)} ${y2.toFixed(2)}`,
        "Z",
      ].join(" ");

      const fillColor = this.colors[i] ?? "#1f77b4";
      ctx.push(
        `<path d="${pathData}" fill="${escapeXml(fillColor)}" stroke="#ffffff" stroke-width="1" />`
      );

      // Add labels if provided
      const label = this.labels[i];
      if (i < this.labels.length && label) {
        const midAngle = (startAngle + endAngle) / 2;
        const labelRadius = r * 0.7;
        const labelX = cx + labelRadius * Math.cos(midAngle);
        const labelY = cy - labelRadius * Math.sin(midAngle);

        ctx.push(
          `<text x="${labelX.toFixed(2)}" y="${labelY.toFixed(2)}" text-anchor="middle" font-size="12" fill="#ffffff">${escapeXml(label)}</text>`
        );
      }
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const cx = Math.round(ctx.transform.xToPx(this.centerX));
    const cy = Math.round(ctx.transform.yToPx(this.centerY));
    const rx = Math.abs(ctx.transform.xToPx(this.centerX + this.radius) - cx);
    const ry = Math.abs(ctx.transform.yToPx(this.centerY + this.radius) - cy);
    const r = Math.round(Math.min(rx, ry));
    const edge = parseHexColorToRGBA("#ffffff");

    for (let i = 0; i < this.values.length; i++) {
      const startAngle = this.angles[i] ?? 0;
      const endAngle = this.angles[i + 1] ?? 2 * Math.PI;

      const fillColor = this.colors[i] ?? "#1f77b4";
      const rgba = parseHexColorToRGBA(fillColor);

      // Draw pie slice using triangle approximation
      // Adaptive resolution: aim for ~5px segments along the arc to ensure smoothness
      const arcLength = (endAngle - startAngle) * r;
      const numSegments = Math.max(5, Math.ceil(arcLength / 5));

      for (let j = 0; j < numSegments; j++) {
        const angle1 = startAngle + (j / numSegments) * (endAngle - startAngle);
        const angle2 = startAngle + ((j + 1) / numSegments) * (endAngle - startAngle);

        const x1 = cx + r * Math.cos(angle1);
        const y1 = cy - r * Math.sin(angle1);
        const x2 = cx + r * Math.cos(angle2);
        const y2 = cy - r * Math.sin(angle2);

        // Fill triangle using rasterization
        ctx.canvas.fillTriangleRGBA(cx, cy, x1, y1, x2, y2, rgba.r, rgba.g, rgba.b, rgba.a);
      }

      const edgeX = Math.round(cx + r * Math.cos(startAngle));
      const edgeY = Math.round(cy - r * Math.sin(startAngle));
      ctx.canvas.drawLineRGBA(cx, cy, edgeX, edgeY, edge.r, edge.g, edge.b, edge.a);

      const label = this.labels[i];
      if (i < this.labels.length && label) {
        const midAngle = (startAngle + endAngle) / 2;
        const labelRadius = r * 0.7;
        const labelX = cx + labelRadius * Math.cos(midAngle);
        const labelY = cy - labelRadius * Math.sin(midAngle);
        const textColor = parseHexColorToRGBA("#ffffff");
        ctx.canvas.drawTextRGBA(
          label,
          Math.round(labelX),
          Math.round(labelY),
          textColor.r,
          textColor.g,
          textColor.b,
          textColor.a,
          { fontSize: 12, align: "middle", baseline: "middle" }
        );
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entries: LegendEntry[] = [];
    if (this.labels.length > 0) {
      for (let i = 0; i < this.labels.length; i++) {
        const label = normalizeLegendLabel(this.labels[i]);
        if (!label) continue;
        const color = this.colors[i] ?? "#1f77b4";
        entries.push({ label, color, shape: "box" });
      }
      return entries.length > 0 ? entries : null;
    }
    if (!this.label) {
      return null;
    }
    const entry: LegendEntry = {
      label: this.label,
      color: this.colors[0] ?? "#1f77b4",
      shape: "box",
    };
    return [entry];
  }
}
