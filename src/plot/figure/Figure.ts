import { InvalidParameterError, NotImplementedError } from "../../core";
import { RasterCanvas } from "../canvas/RasterCanvas";
import { isNodeEnvironment_export, pngEncodeRGBA } from "../renderers/png";
import type { Color, RenderedPNG, RenderedSVG, Viewport } from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { assertPositiveInt } from "../utils/validation";
import { escapeXml } from "../utils/xml";
import { Axes } from "./Axes";

/**
 * A Figure represents the entire plotting canvas.
 */
export class Figure {
  readonly width: number;
  readonly height: number;
  readonly background: Color;
  public readonly axesList: Axes[];

  constructor(
    options: {
      readonly width?: number;
      readonly height?: number;
      readonly background?: Color;
    } = {}
  ) {
    this.width = options.width ?? 640;
    this.height = options.height ?? 480;
    assertPositiveInt("Figure.width", this.width);
    assertPositiveInt("Figure.height", this.height);

    if (this.width > 32768 || this.height > 32768) {
      throw new InvalidParameterError(
        "Figure dimensions too large. Maximum is 32,768 pixels.",
        "width/height",
        { width: this.width, height: this.height }
      );
    }

    this.background = normalizeColor(options.background, "#ffffff");
    this.axesList = [];
  }

  /**
   * Add a new axes to the figure.
   * @param options - Axes layout options
   */
  addAxes(
    options: {
      readonly padding?: number;
      readonly facecolor?: Color;
      readonly viewport?: Viewport;
    } = {}
  ): Axes {
    const ax = new Axes(this, options);
    this.axesList.push(ax);
    return ax;
  }

  /**
   * Render this figure to SVG.
   */
  renderSVG(): RenderedSVG {
    const elements: string[] = [];
    elements.push(
      `<rect x="0" y="0" width="${this.width}" height="${this.height}" fill="${escapeXml(this.background)}" />`
    );
    for (const ax of this.axesList) ax.renderSVGInto(elements);
    const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${this.width}" height="${this.height}" viewBox="0 0 ${this.width} ${this.height}">\n${elements.join("\n")}\n</svg>`;
    return { kind: "svg", svg };
  }

  /**
   * Render this figure to PNG (Node.js only).
   *
   * Note: PNG text rendering uses a built-in bitmap font for basic ASCII.
   * Unsupported characters are rendered as "?".
   */
  async renderPNG(): Promise<RenderedPNG> {
    if (!isNodeEnvironment_export()) {
      throw new NotImplementedError(
        "PNG rendering is only available in Node.js environments. " +
          "Use renderSVG() for browser compatibility, or run in Node.js to generate PNG files."
      );
    }

    const canvas = new RasterCanvas(this.width, this.height);
    const bg = parseHexColorToRGBA(this.background);
    canvas.clearRGBA(bg.r, bg.g, bg.b, bg.a);
    for (const ax of this.axesList) ax.renderRasterInto(canvas);
    const bytes = await pngEncodeRGBA(this.width, this.height, canvas.data);
    return { kind: "png", width: this.width, height: this.height, bytes };
  }
}
