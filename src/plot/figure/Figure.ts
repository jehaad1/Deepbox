/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError, NotImplementedError } from "../../core";
import { RasterCanvas } from "../canvas/RasterCanvas";
import { svgToPdf } from "../renderers/pdf";
import { isNodeEnvironment_export, pngEncodeRGBA } from "../renderers/png";
import type { Color, RenderedPDF, RenderedPNG, RenderedSVG, Viewport } from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { assertPositiveInt } from "../utils/validation";
import { escapeXml } from "../utils/xml";
import { Axes } from "./Axes";
import { getTheme } from "./theme";

/**
 * A Figure represents the entire plotting canvas.
 *
 * A new Figure has no axes; add them with {@link Figure.addAxes}. It renders to
 * SVG, PDF, or (in Node.js) PNG.
 *
 * @example
 * ```ts
 * const fig = new Figure({ width: 400, height: 300 });
 * const ax = fig.addAxes();
 * ax.plot(tensor([0, 1, 2]), tensor([0, 1, 4]));
 * const { svg } = fig.renderSVG();
 * ```
 */
export class Figure {
  /** Width in pixels. */
  readonly width: number;
  /** Height in pixels. */
  readonly height: number;
  /** Background color. */
  readonly background: Color;
  /** Axes in drawing order (later axes are drawn on top). */
  public readonly axesList: Axes[];

  /**
   * @param options - `width` and `height` in pixels (positive integers up to 32,768; default
   *   640 x 480) and the `background` color (default: the figure color of the active theme,
   *   white in the default theme)
   */
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

    this.background = normalizeColor(options.background ?? getTheme().figureFacecolor, "#ffffff");
    this.axesList = [];
  }

  /**
   * Add a new axes to the figure and return it.
   * @param options - `padding` (pixels kept free on each side for ticks and labels; default a
   *   quarter of the smaller viewport side, at most 50), `facecolor` (default: the axes color of
   *   the active theme, white in the default theme) and `viewport` (the figure region the axes
   *   occupies; default the whole figure)
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
   * @returns The complete SVG document
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
   * Lowercase letters are drawn as uppercase and unsupported characters are
   * rendered as "?". Throws a MemoryError if the pixel buffer would exceed 2 GiB.
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

  /**
   * Render this figure to PDF.
   *
   * Converts the SVG output to a minimal PDF document with vector
   * drawing commands. The resulting PDF preserves vector quality and
   * is suitable for publication or print.
   *
   * @returns PDF rendering result with byte data
   */
  renderPDF(): RenderedPDF {
    const svg = this.renderSVG();
    const bytes = svgToPdf(svg.svg, this.width, this.height);
    return { kind: "pdf", bytes, width: this.width, height: this.height };
  }
}
