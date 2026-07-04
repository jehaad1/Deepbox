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

/**
 * Options for creating an animation.
 */
export type AnimationOptions = {
  /** Frames per second. Default: 30. */
  readonly fps?: number;
  /** Total duration in milliseconds. Default: inferred from frames. */
  readonly duration?: number;
  /** Whether the animation should loop. Default: true. */
  readonly loop?: boolean;
  /** Easing function name. Default: "linear". */
  readonly easing?: "linear" | "ease-in" | "ease-out" | "ease-in-out";
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
  /** Total duration in milliseconds. */
  readonly durationMs: number;
  /** Whether the animation loops. */
  readonly loop: boolean;
  /** Individual frames. */
  readonly frames: readonly AnimationFrame[];
};

/**
 * Frame-based animation controller for Deepbox plots.
 *
 * Create an animation by registering a frame generator function that
 * updates the figure for each frame. The animation then renders all
 * frames and can export them in various formats.
 *
 * @example
 * ```ts
 * import { Animation, figure } from 'deepbox/plot';
 *
 * const anim = new Animation({ fps: 24, duration: 2000 });
 *
 * anim.animate((frameIndex, totalFrames) => {
 *   const fig = figure({ width: 400, height: 300 });
 *   const ax = fig.addAxes();
 *   const t = frameIndex / totalFrames;
 *   ax.plot([0, t, 1], [0, Math.sin(t * Math.PI * 2), 0]);
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
  private readonly easing: string;
  private frameGenerator: ((frameIndex: number, totalFrames: number) => Figure) | null = null;
  private renderedFrames: AnimationFrame[] = [];

  constructor(options: AnimationOptions = {}) {
    this.fps = options.fps ?? 30;
    this.durationMs = options.duration ?? 1000;
    this.loop = options.loop ?? true;
    this.easing = options.easing ?? "linear";

    if (this.fps < 1 || this.fps > 120) {
      throw new InvalidParameterError("FPS must be between 1 and 120", "fps", this.fps);
    }
    if (this.durationMs < 1) {
      throw new InvalidParameterError("Duration must be positive", "duration", this.durationMs);
    }
  }

  /**
   * Register a frame generator function.
   *
   * The function receives the current frame index and total frame count,
   * and must return a Figure to render for that frame.
   *
   * @param generator - Function that creates a figure for each frame
   */
  animate(generator: (frameIndex: number, totalFrames: number) => Figure): void {
    this.frameGenerator = generator;
    this.renderedFrames = [];
  }

  /**
   * Render all frames of the animation.
   *
   * @returns Animation result with all rendered frames
   */
  render(): AnimationResult {
    if (!this.frameGenerator) {
      throw new InvalidParameterError(
        "No frame generator registered. Call animate() first.",
        "frameGenerator",
        null
      );
    }

    const totalFrames = Math.ceil((this.fps * this.durationMs) / 1000);
    const frameDurationMs = 1000 / this.fps;
    this.renderedFrames = [];

    for (let i = 0; i < totalFrames; i++) {
      const fig = this.frameGenerator(i, totalFrames);
      const svg = fig.renderSVG();
      this.renderedFrames.push({
        index: i,
        timeMs: i * frameDurationMs,
        svg,
      });
    }

    return {
      frameCount: totalFrames,
      fps: this.fps,
      durationMs: this.durationMs,
      loop: this.loop,
      frames: this.renderedFrames,
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
   * Export the animation as an animated SVG using CSS animations.
   *
   * Generates a single SVG file that cycles through all frames
   * using CSS keyframe animations. This is universally supported
   * in modern browsers.
   *
   * @returns SVG string with embedded CSS animation
   */
  toAnimatedSVG(): string {
    if (this.renderedFrames.length === 0) {
      this.render();
    }

    if (this.renderedFrames.length === 0) {
      return '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>';
    }

    const first = this.renderedFrames[0]!;
    // Extract width/height from first frame's SVG
    const widthMatch = /width="(\d+)"/.exec(first.svg.svg);
    const heightMatch = /height="(\d+)"/.exec(first.svg.svg);
    const width = widthMatch ? parseInt(widthMatch[1]!, 10) : 640;
    const height = heightMatch ? parseInt(heightMatch[1]!, 10) : 480;

    const totalFrames = this.renderedFrames.length;
    const durationSec = this.durationMs / 1000;
    const repeatCount = this.loop ? "indefinite" : "1";

    // Build CSS keyframe animation
    const keyframes: string[] = [];
    for (let i = 0; i < totalFrames; i++) {
      const pctStart = ((i / totalFrames) * 100).toFixed(2);
      const pctEnd = (((i + 1) / totalFrames) * 100).toFixed(2);
      keyframes.push(`  ${pctStart}%, ${pctEnd}% { opacity: ${i === 0 ? 1 : 0}; }`);
    }

    // Build frame groups
    const frameGroups: string[] = [];
    for (let i = 0; i < totalFrames; i++) {
      const frame = this.renderedFrames[i]!;
      // Extract inner SVG content (everything between the <svg> tags)
      const innerMatch = /<svg[^>]*>([\s\S]*)<\/svg>/.exec(frame.svg.svg);
      const inner = innerMatch ? innerMatch[1]! : "";

      const delay = (i / totalFrames) * durationSec;
      const frameDur = durationSec;

      frameGroups.push(
        `<g class="frame-${i}" opacity="0">` +
          `<animate attributeName="opacity" values="0;1;1;0" ` +
          `keyTimes="0;${(0.001).toFixed(3)};${(1 / totalFrames - 0.001).toFixed(3)};${(1 / totalFrames).toFixed(3)}" ` +
          `dur="${frameDur}s" begin="${delay.toFixed(3)}s" ` +
          `repeatCount="${repeatCount}" fill="freeze"/>` +
          inner +
          `</g>`
      );
    }

    return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="${width}" height="${height}" fill="#ffffff"/>
${frameGroups.join("\n")}
</svg>`;
  }

  /**
   * Export individual frame SVGs.
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
  info(): { fps: number; durationMs: number; totalFrames: number; loop: boolean; easing: string } {
    return {
      fps: this.fps,
      durationMs: this.durationMs,
      totalFrames: Math.ceil((this.fps * this.durationMs) / 1000),
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
