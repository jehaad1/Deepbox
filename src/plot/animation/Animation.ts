/**
 * Animation support for Deepbox plots.
 *
 * Provides frame-based animation that generates a sequence of SVG frames
 * with timing metadata. Supports exporting as:
 * - Individual SVG frames
 * - Animated SVG using SMIL `<animate>` elements
 * - Frame sequence for external video encoding
 *
 * @module plot/animation
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core/errors/invalid_parameter";
import type { Figure } from "../figure/Figure";
import type { RenderedSVG } from "../types";

/** Names accepted by {@link AnimationOptions.easing}. */
export type AnimationEasing = "linear" | "ease-in" | "ease-out" | "ease-in-out";

/**
 * Options for creating an animation.
 */
export type AnimationOptions = {
  /** Frames per second, between 1 and 120. Default: 30. */
  readonly fps?: number;
  /** Total duration in milliseconds (at least 1). Default: 1000. */
  readonly duration?: number;
  /** Whether the animation should loop. Default: true. */
  readonly loop?: boolean;
  /**
   * Easing applied to the `progress` argument passed to the frame generator.
   * Default: "linear".
   */
  readonly easing?: AnimationEasing;
};

/**
 * A single animation frame.
 */
export type AnimationFrame = {
  /** Frame index (0-based). */
  readonly index: number;
  /** Timestamp in milliseconds from the start. */
  readonly timeMs: number;
  /** The rendered SVG for this frame. */
  readonly svg: RenderedSVG;
};

/**
 * Result of rendering an animation.
 */
export type AnimationResult = {
  /** Total number of frames. */
  readonly frameCount: number;
  /** Frames per second. */
  readonly fps: number;
  /** Requested duration in milliseconds. */
  readonly durationMs: number;
  /** Whether the animation loops. */
  readonly loop: boolean;
  /** Individual frames. */
  readonly frames: readonly AnimationFrame[];
};

/**
 * Function that builds the figure for one frame.
 *
 * - `frameIndex` is the 0-based frame number.
 * - `totalFrames` is the number of frames in the animation.
 * - `progress` is the animation position in [0, 1] after easing. For a looping
 *   animation it is `frameIndex / totalFrames` (the end state is the start
 *   state of the next cycle); for a non-looping animation the last frame
 *   reaches exactly 1.
 */
export type AnimationFrameGenerator = (
  frameIndex: number,
  totalFrames: number,
  progress: number
) => Figure;

const EASINGS: Readonly<Record<AnimationEasing, (t: number) => number>> = {
  linear: (t) => t,
  "ease-in": (t) => t * t,
  "ease-out": (t) => t * (2 - t),
  "ease-in-out": (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
};

/** Upper bound on rendered frames, to catch accidental huge durations. */
const MAX_FRAMES = 50_000;

function formatNumber(v: number): string {
  return String(Number(v.toFixed(8)));
}

/**
 * Prefix every `id` in an SVG fragment (and the references to it) so the
 * fragments of several frames can live in one document without clashing.
 */
function scopeIds(fragment: string, prefix: string): string {
  const ids = new Set<string>();
  for (const m of fragment.matchAll(/\sid="([^"]+)"/g)) ids.add(m[1] ?? "");
  if (ids.size === 0) return fragment;
  return fragment
    .replace(/(\s)id="([^"]+)"/g, (_m, ws: string, id: string) => `${ws}id="${prefix}${id}"`)
    .replace(/url\(#([^)]+)\)/g, (m, id: string) => (ids.has(id) ? `url(#${prefix}${id})` : m))
    .replace(/((?:xlink:)?href)="#([^"]+)"/g, (m, attr: string, id: string) =>
      ids.has(id) ? `${attr}="#${prefix}${id}"` : m
    );
}

/**
 * Frame-based animation controller for Deepbox plots.
 *
 * Create an animation by registering a frame generator function that
 * builds a figure for each frame. The animation then renders all
 * frames and can export them in various formats.
 *
 * @example
 * ```ts
 * import { Animation, Figure } from 'deepbox/plot';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const anim = new Animation({ fps: 24, duration: 2000 });
 *
 * anim.animate((frameIndex, totalFrames, progress) => {
 *   const fig = new Figure({ width: 400, height: 300 });
 *   const ax = fig.addAxes();
 *   ax.plot(tensor([0, 0.5, 1]), tensor([0, Math.sin(progress * Math.PI * 2), 0]));
 *   return fig;
 * });
 *
 * const result = anim.render();
 * console.log(result.frameCount); // 48 frames (24fps × 2s)
 *
 * // Export as animated SVG
 * const svg = anim.toAnimatedSVG();
 * ```
 */
export class Animation {
  private readonly fps: number;
  private readonly durationMs: number;
  private readonly loop: boolean;
  private readonly easing: AnimationEasing;
  private frameGenerator: AnimationFrameGenerator | null = null;
  private renderedFrames: AnimationFrame[] = [];

  constructor(options: AnimationOptions = {}) {
    this.fps = options.fps ?? 30;
    this.durationMs = options.duration ?? 1000;
    this.loop = options.loop ?? true;
    this.easing = options.easing ?? "linear";

    if (!Number.isFinite(this.fps) || this.fps < 1 || this.fps > 120) {
      throw new InvalidParameterError(
        `FPS must be between 1 and 120; received ${this.fps}`,
        "fps",
        this.fps
      );
    }
    if (!Number.isFinite(this.durationMs) || this.durationMs < 1) {
      throw new InvalidParameterError(
        `Duration must be a finite number of at least 1 millisecond; received ${this.durationMs}`,
        "duration",
        this.durationMs
      );
    }
    if (!Object.keys(EASINGS).includes(this.easing)) {
      throw new InvalidParameterError(
        `easing must be one of ${Object.keys(EASINGS).join(", ")}; received ${String(this.easing)}`,
        "easing",
        this.easing
      );
    }
    const frames = this.computeFrameCount();
    if (frames > MAX_FRAMES) {
      throw new InvalidParameterError(
        `Animation would have ${frames} frames; the maximum is ${MAX_FRAMES}. Lower fps or duration.`,
        "duration",
        this.durationMs
      );
    }
  }

  private computeFrameCount(): number {
    // The small tolerance keeps products like 29.97 * 1000 / 1000 from gaining a frame.
    return Math.max(1, Math.ceil((this.fps * this.durationMs) / 1000 - 1e-9));
  }

  /**
   * Register a frame generator function.
   *
   * The function receives the frame index, the total frame count and the eased
   * progress in [0, 1], and must return a Figure to render for that frame.
   * Registering a generator discards frames rendered earlier.
   *
   * @param generator - Function that creates a figure for each frame
   */
  animate(generator: AnimationFrameGenerator): void {
    if (typeof generator !== "function") {
      throw new InvalidParameterError("generator must be a function", "generator", generator);
    }
    this.frameGenerator = generator;
    this.renderedFrames = [];
  }

  /**
   * Render all frames of the animation.
   *
   * @returns Animation result with all rendered frames
   */
  render(): AnimationResult {
    const generator = this.frameGenerator;
    if (!generator) {
      throw new InvalidParameterError(
        "No frame generator registered. Call animate() first.",
        "frameGenerator",
        null
      );
    }

    const totalFrames = this.computeFrameCount();
    const frameDurationMs = 1000 / this.fps;
    const ease = EASINGS[this.easing];
    // Render into a local list so a throwing generator leaves no partial state behind.
    const frames: AnimationFrame[] = [];

    for (let i = 0; i < totalFrames; i++) {
      const raw = this.loop ? i / totalFrames : totalFrames > 1 ? i / (totalFrames - 1) : 0;
      const fig = generator(i, totalFrames, ease(raw));
      frames.push({ index: i, timeMs: i * frameDurationMs, svg: fig.renderSVG() });
    }
    this.renderedFrames = frames;

    return {
      frameCount: totalFrames,
      fps: this.fps,
      durationMs: this.durationMs,
      loop: this.loop,
      frames: [...frames],
    };
  }

  /**
   * Get a specific rendered frame.
   *
   * @param index - Frame index (0-based)
   * @returns The animation frame, or undefined if not rendered
   */
  getFrame(index: number): AnimationFrame | undefined {
    return this.renderedFrames[index];
  }

  /**
   * Export the animation as a single animated SVG.
   *
   * Every frame becomes a group whose opacity is switched on for exactly its
   * time slot with a SMIL `<animate>` element (discrete steps, so frames never
   * blend). Playback runs at the configured fps: one cycle lasts
   * `frameCount / fps` seconds. The first frame is the static fallback for
   * viewers that ignore SMIL. The canvas size is taken from the first frame.
   * Renders the frames first if that has not happened yet.
   *
   * @returns SVG string with embedded SMIL animation
   */
  toAnimatedSVG(): string {
    if (this.renderedFrames.length === 0) {
      this.render();
    }
    const frames = this.renderedFrames;
    const first = frames[0];
    if (!first) {
      return '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>';
    }

    // Canvas size from the root <svg> tag of the first frame.
    const rootTag = /<svg\b[^>]*>/.exec(first.svg.svg)?.[0] ?? "";
    const widthMatch = /\swidth="([\d.]+)"/.exec(rootTag);
    const heightMatch = /\sheight="([\d.]+)"/.exec(rootTag);
    const width = widthMatch ? Number.parseFloat(widthMatch[1] ?? "") : 640;
    const height = heightMatch ? Number.parseFloat(heightMatch[1] ?? "") : 480;

    const totalFrames = frames.length;
    const cycle = `${formatNumber(totalFrames / this.fps)}s`;
    const repeatCount = this.loop ? "indefinite" : "1";

    const frameGroups: string[] = [];
    for (let i = 0; i < totalFrames; i++) {
      const frame = frames[i];
      if (!frame) continue;
      const svg = frame.svg.svg;
      const open = /<svg\b[^>]*>/.exec(svg);
      const closeAt = svg.lastIndexOf("</svg>");
      const inner =
        open && closeAt >= open.index + open[0].length
          ? svg.slice(open.index + open[0].length, closeAt)
          : "";
      const body = scopeIds(inner, `f${i}-`);

      if (totalFrames === 1) {
        frameGroups.push(`<g class="frame-${i}" opacity="1">${body}</g>`);
        continue;
      }

      // Opacity timeline over one cycle: hidden until this frame's slot starts,
      // visible during it, hidden afterwards (the last frame stays visible).
      const times: number[] = [0];
      const values: number[] = [i === 0 ? 1 : 0];
      if (i > 0) {
        times.push(i / totalFrames);
        values.push(1);
      }
      if (i < totalFrames - 1) {
        times.push((i + 1) / totalFrames);
        values.push(0);
      }
      frameGroups.push(
        `<g class="frame-${i}" opacity="${i === 0 ? 1 : 0}">` +
          `<animate attributeName="opacity" calcMode="discrete" ` +
          `values="${values.join(";")}" keyTimes="${times.map(formatNumber).join(";")}" ` +
          `dur="${cycle}" begin="0s" repeatCount="${repeatCount}" fill="freeze"/>` +
          body +
          `</g>`
      );
    }

    return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
${frameGroups.join("\n")}
</svg>`;
  }

  /**
   * Export individual frame SVGs.
   * Renders the frames first if that has not happened yet.
   *
   * @returns Array of SVG strings, one per frame
   */
  toFrames(): string[] {
    if (this.renderedFrames.length === 0) {
      this.render();
    }
    return this.renderedFrames.map((f) => f.svg.svg);
  }

  /**
   * Get animation metadata without frame data.
   */
  info(): {
    fps: number;
    durationMs: number;
    totalFrames: number;
    loop: boolean;
    easing: AnimationEasing;
  } {
    return {
      fps: this.fps,
      durationMs: this.durationMs,
      totalFrames: this.computeFrameCount(),
      loop: this.loop,
      easing: this.easing,
    };
  }
}

/**
 * Create a new Animation instance.
 *
 * @param options - Animation options
 * @returns A new Animation
 */
export function createAnimation(options?: AnimationOptions): Animation {
  return new Animation(options);
}
