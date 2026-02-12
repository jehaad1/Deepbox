/**
 * Color specification as a CSS color string (e.g., "#ff0000", "rgb(255,0,0)").
 */
export type Color = string;

/**
 * Options for customizing plot appearance and behavior.
 * Different plot types use different subsets of these options.
 *
 * **Supported Color Formats:**
 * - Hex: "#RRGGBB" or "#RRGGBBAA" (e.g., "#ff0000", "#ff0000ff")
 * - RGB: "rgb(255, 0, 0)" or "rgba(255, 0, 0, 0.5)"
 * - HSL: "hsl(0, 100%, 50%)" or "hsla(0, 100%, 50%, 0.5)"
 * - Named: "red", "blue", "forestgreen", etc. (140 CSS color names)
 * - Invalid colors default to black
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
  /** Colormap for heatmaps and images (viridis, plasma, inferno, magma, grayscale) */
  readonly colormap?: "viridis" | "plasma" | "inferno" | "magma" | "grayscale";
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
 * Result of PNG rendering containing the image dimensions and raw byte data.
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
 * @internal
 */
export type DataRange = {
  readonly xmin: number;
  readonly xmax: number;
  readonly ymin: number;
  readonly ymax: number;
};

/**
 * @internal
 */
export type Viewport = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

/**
 * @internal
 */
export type DataTransform = {
  readonly xToPx: (x: number) => number;
  readonly yToPx: (y: number) => number;
};

/**
 * @internal
 */
export type SvgDrawContext = {
  readonly transform: DataTransform;
  push(element: string): void;
};

/**
 * @internal
 */
export type RasterDrawContext = {
  readonly transform: DataTransform;
  readonly canvas: import("./canvas/RasterCanvas").RasterCanvas;
};

/**
 * @internal
 */
export type Drawable = {
  readonly kind: string;
  getDataRange(): DataRange | null;
  drawSVG(ctx: SvgDrawContext): void;
  drawRaster(ctx: RasterDrawContext): void;
  getLegendEntries?(): readonly LegendEntry[] | null;
};
