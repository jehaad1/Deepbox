/**
 * Color specification as a CSS color string (e.g., "#ff0000", "rgb(255,0,0)").
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */
export type Color = string;

/**
 * Colormaps available for heatmaps, images, contour lines and filled contours. Values are mapped
 * from the normalized range [0, 1]; all but "grayscale" are the matplotlib maps of the same name.
 */
export type ColormapName = "viridis" | "plasma" | "inferno" | "magma" | "cividis" | "grayscale";

/**
 * Options for customizing plot appearance and behavior.
 * Different plot types use different subsets of these options.
 *
 * **Supported Color Formats:**
 * - Hex: "#RGB", "#RGBA", "#RRGGBB" or "#RRGGBBAA" (e.g., "#f00", "#ff0000", "#ff000080")
 * - RGB: "rgb(255, 0, 0)", "rgba(255, 0, 0, 0.5)" or "rgb(100% 0% 0% / 50%)"
 * - HSL: "hsl(0, 100%, 50%)" or "hsla(0, 100%, 50%, 0.5)"
 * - Named: "red", "blue", "forestgreen", etc. (the 148 CSS color names) and "transparent"
 * - Anything else (including misspelled names) is drawn as black
 */
export type PlotOptions = {
  /** Optional label used by legends */
  readonly label?: string;
  /** Line or marker color */
  readonly color?: Color;
  /** Line width in pixels */
  readonly linewidth?: number;
  /** Marker size in pixels */
  readonly size?: number;
  /** Edge color for bars and shapes */
  readonly edgecolor?: Color;
  /** Number of histogram bins (overrides the bins argument if provided) */
  readonly bins?: number;
  /** Minimum value for color mapping */
  readonly vmin?: number;
  /** Maximum value for color mapping */
  readonly vmax?: number;
  /** Explicit data extent for heatmaps, images, and contour plots */
  readonly extent?: {
    readonly xmin: number;
    readonly xmax: number;
    readonly ymin: number;
    readonly ymax: number;
  };
  /** Array of colors for multi-series plots */
  readonly colors?: readonly Color[];
  /** Contour levels (number or explicit values) */
  readonly levels?: number | readonly number[];
  /** Background color for axes */
  readonly facecolor?: Color;
  /** Colormap for heatmaps, images and contours (viridis, plasma, inferno, magma, cividis, grayscale) */
  readonly colormap?: ColormapName;
  /**
   * Which edge of the y range shows row 0 of a heatmap or image: "lower" (default, row 0 at the
   * bottom) or "upper" (row 0 at the top, like matplotlib's `imshow` default).
   */
  readonly origin?: "lower" | "upper";
  /** Bar width in data units for `bar` (default 0.8). Must be positive and finite. */
  readonly barWidth?: number;
  /** Bar thickness in data units for `barh` (default 0.8). Must be positive and finite. */
  readonly barHeight?: number;
};

/**
 * Options of `Axes.annotate` and `Axes.text`.
 */
export type TextOptions = {
  /** Text color (default: the axes text color, black in the default theme) */
  readonly color?: Color;
  /** Font size in pixels (default 10) */
  readonly fontSize?: number;
  /** Horizontal alignment relative to x: "left" (default, the text starts at x), "center" or "right" */
  readonly ha?: "left" | "center" | "right";
  /** Vertical alignment relative to y: "bottom" (default, the text sits on y), "center" or "top" */
  readonly va?: "bottom" | "center" | "top";
};

/**
 * Legend display options.
 */
export type LegendOptions = {
  /** Whether the legend should be visible */
  readonly visible?: boolean;
  /** Legend placement */
  readonly location?: "upper-right" | "upper-left" | "lower-right" | "lower-left";
  /** Legend font size in pixels */
  readonly fontSize?: number;
  /** Legend padding in pixels */
  readonly padding?: number;
  /** Legend background color */
  readonly background?: Color;
  /** Legend border color */
  readonly borderColor?: Color;
};

/**
 * Legend entry definition.
 */
export type LegendEntry = {
  readonly label: string;
  readonly color: Color;
  /** Optional symbol shape for legend marker. */
  readonly shape?: "line" | "marker" | "box";
  /** Line width for line legend entries. */
  readonly lineWidth?: number;
  /** Marker size for marker legend entries. */
  readonly markerSize?: number;
};

/**
 * Result of SVG rendering containing the complete SVG document as a string.
 */
export type RenderedSVG = {
  /** Discriminator for the rendered output type */
  readonly kind: "svg";
  /** Complete SVG document as XML string */
  readonly svg: string;
};

/**
 * Result of PNG rendering containing the image dimensions and the encoded PNG file bytes.
 * PNG encoding is only available in Node.js environments.
 */
export type RenderedPNG = {
  /** Discriminator for the rendered output type */
  readonly kind: "png";
  /** Image width in pixels */
  readonly width: number;
  /** Image height in pixels */
  readonly height: number;
  /** PNG file data as byte array */
  readonly bytes: Uint8Array;
};

/**
 * Result of PDF rendering containing the PDF file data.
 * PDF rendering is only available in Node.js environments.
 */
export type RenderedPDF = {
  /** Discriminator for the rendered output type */
  readonly kind: "pdf";
  /** PDF file data as byte array */
  readonly bytes: Uint8Array;
  /** Page width in points */
  readonly width: number;
  /** Page height in points */
  readonly height: number;
};

/**
 * Axis-aligned bounds of a drawable in data coordinates.
 * @internal
 */
export type DataRange = {
  readonly xmin: number;
  readonly xmax: number;
  readonly ymin: number;
  readonly ymax: number;
};

/**
 * Pixel rectangle of the plotting area; `x`/`y` is the top-left corner.
 * @internal
 */
export type Viewport = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

/**
 * Maps data coordinates to pixels. `yToPx` is flipped: larger data values give smaller pixel rows.
 * @internal
 */
export type DataTransform = {
  readonly xToPx: (x: number) => number;
  readonly yToPx: (y: number) => number;
};

/**
 * What a drawable receives when rendering to SVG: the transform and a sink for SVG elements.
 * @internal
 */
export type SvgDrawContext = {
  readonly transform: DataTransform;
  push(element: string): void;
};

/**
 * What a drawable receives when rendering to pixels: the transform and the canvas to draw on.
 * @internal
 */
export type RasterDrawContext = {
  readonly transform: DataTransform;
  readonly canvas: import("./canvas/RasterCanvas").RasterCanvas;
};

/**
 * Anything an axes can draw. `getDataRange` returns null when the drawable has no finite data.
 * `getPositiveMin` is optional: it gives the smallest strictly positive x and y coordinate the
 * drawable occupies (`Infinity` for an axis without one), which lets a log axis start at the
 * smallest positive value instead of at a non-positive minimum.
 * @internal
 */
export type Drawable = {
  readonly kind: string;
  getDataRange(): DataRange | null;
  getPositiveMin?(): { readonly x: number; readonly y: number };
  drawSVG(ctx: SvgDrawContext): void;
  drawRaster(ctx: RasterDrawContext): void;
  getLegendEntries?(): readonly LegendEntry[] | null;
};
