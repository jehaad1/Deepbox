/**
 * v1.5.0 regression tests for src/plot/animation, src/plot/canvas and src/plot/figure.
 *
 * Reference values for step() come from NumPy (np.repeat construction used by
 * matplotlib's pts_to_prestep/poststep/midstep); alpha compositing is checked
 * against the source-over formula evaluated in Python.
 */

import { describe, expect, it } from "vitest";
import { InvalidParameterError, MemoryError, ShapeError } from "../../src/core";
import { tensor } from "../../src/ndarray";
import { Animation } from "../../src/plot/animation/Animation";
import { RasterCanvas } from "../../src/plot/canvas/RasterCanvas";
import type { Axes } from "../../src/plot/figure/Axes";
import { Figure } from "../../src/plot/figure/Figure";
import { figure, gca, subplot } from "../../src/plot/figure/state";
import type { Drawable } from "../../src/plot/types";

function pixel(c: RasterCanvas, x: number, y: number): [number, number, number, number] {
  const i = (y * c.width + x) * 4;
  return [c.data[i] ?? 0, c.data[i + 1] ?? 0, c.data[i + 2] ?? 0, c.data[i + 3] ?? 0];
}

function countPixels(c: RasterCanvas, pred: (p: [number, number, number, number]) => boolean) {
  let n = 0;
  for (let y = 0; y < c.height; y++) {
    for (let x = 0; x < c.width; x++) if (pred(pixel(c, x, y))) n++;
  }
  return n;
}

function drawablesOf(ax: Axes): Drawable[] {
  return (ax as unknown as { drawables: Drawable[] }).drawables;
}

function polylinePoints(svg: string): Array<Array<[number, number]>> {
  const out: Array<Array<[number, number]>> = [];
  for (const m of svg.matchAll(/<polyline[^>]*points="([^"]*)"/g)) {
    const pts = (m[1] ?? "")
      .split(" ")
      .filter((s) => s.length > 0)
      .map((s) => {
        const [a, b] = s.split(",");
        return [Number(a), Number(b)] as [number, number];
      });
    out.push(pts);
  }
  return out;
}

function tickX(svg: string, label: string): number {
  const re = new RegExp(`class="tick-label tick-label-x" x="([-\\d.]+)"[^>]*>${label}</text>`);
  const m = re.exec(svg);
  if (!m) throw new Error(`no x tick labeled ${label}`);
  return Number(m[1]);
}

describe("RasterCanvas", () => {
  it("composites translucent colors over the background (source-over)", () => {
    const c = new RasterCanvas(4, 4);
    c.clearRGBA(255, 255, 255, 255);
    c.setPixelRGBA(1, 1, 255, 0, 0, 128);
    // 255*128/255 + 255*127/255 = 255; 0*128/255 + 255*127/255 = 127
    expect(pixel(c, 1, 1)).toEqual([255, 127, 127, 255]);
    expect(pixel(c, 0, 0)).toEqual([255, 255, 255, 255]);
  });

  it("keeps the source alpha when drawing onto a transparent pixel", () => {
    const c = new RasterCanvas(2, 2);
    c.setPixelRGBA(0, 0, 10, 20, 30, 100);
    expect(pixel(c, 0, 0)).toEqual([10, 20, 30, 100]);
    // fully transparent source leaves the canvas untouched
    c.setPixelRGBA(1, 1, 9, 9, 9, 0);
    expect(pixel(c, 1, 1)).toEqual([0, 0, 0, 0]);
  });

  it("blends fills and spans, and fills opaque spans exactly", () => {
    const c = new RasterCanvas(6, 3);
    c.clearRGBA(0, 0, 255, 255);
    c.fillRectRGBA(1, 1, 3, 1, 255, 0, 0, 255);
    expect(pixel(c, 1, 1)).toEqual([255, 0, 0, 255]);
    expect(pixel(c, 3, 1)).toEqual([255, 0, 0, 255]);
    expect(pixel(c, 4, 1)).toEqual([0, 0, 255, 255]);
    c.fillRectRGBA(0, 0, 6, 1, 255, 255, 255, 51);
    // 0 * 0.8 + 255 * 0.2 = 51 for red, 255 stays at 255 for blue
    expect(pixel(c, 2, 0)).toEqual([51, 51, 255, 255]);
  });

  it("floors fractional pixel coordinates and ignores NaN", () => {
    const c = new RasterCanvas(4, 4);
    c.setPixelRGBA(1.7, 2.2, 1, 2, 3, 255);
    expect(pixel(c, 1, 2)).toEqual([1, 2, 3, 255]);
    c.setPixelRGBA(Number.NaN, 0, 9, 9, 9, 255);
    expect(countPixels(c, (p) => p[3] > 0)).toBe(1);
  });

  it("keeps the slope of lines whose end point is far outside the canvas", () => {
    const c = new RasterCanvas(100, 100);
    c.drawLineRGBA(0, 0, 1e8, 5e7, 0, 0, 0, 255);
    // y = x / 2: (50, 25) is on the line, the diagonal (50, 50) is not.
    expect(pixel(c, 50, 25)[3]).toBe(255);
    expect(pixel(c, 50, 50)[3]).toBe(0);
    expect(pixel(c, 98, 49)[3]).toBe(255);
  });

  it("draws lines entering from outside and skips lines that miss the canvas", () => {
    const c = new RasterCanvas(20, 20);
    c.drawLineRGBA(-50, 10, 70, 10, 0, 0, 0, 255);
    expect(pixel(c, 0, 10)[3]).toBe(255);
    expect(pixel(c, 19, 10)[3]).toBe(255);
    const d = new RasterCanvas(20, 20);
    d.drawLineRGBA(-1e7, -1e7, -1e7, 1e7, 0, 0, 0, 255);
    d.drawLineRGBA(Number.NaN, 0, 5, 5, 0, 0, 0, 255);
    d.drawLineRGBA(0, 0, Number.POSITIVE_INFINITY, 5, 0, 0, 0, 255);
    expect(countPixels(d, (p) => p[3] > 0)).toBe(0);
  });

  it("rasterizes ordinary lines as before (Bresenham)", () => {
    const c = new RasterCanvas(10, 10);
    c.drawLineRGBA(1, 1, 8, 4, 0, 0, 0, 255);
    const set: string[] = [];
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 10; x++) if (pixel(c, x, y)[3] > 0) set.push(`${x},${y}`);
    }
    // Reference from the integer Bresenham algorithm (evaluated in Python).
    expect(set.sort()).toEqual(["1,1", "2,1", "3,2", "4,2", "5,3", "6,3", "7,4", "8,4"]);
  });

  it("terminates on triangles with huge or non-finite vertices", () => {
    const c = new RasterCanvas(30, 30);
    c.fillTriangleRGBA(-1e9, -1e9, 1e9, -1e9, 0, 1e9, 1, 2, 3, 255);
    expect(pixel(c, 15, 15)).toEqual([1, 2, 3, 255]);
    const d = new RasterCanvas(30, 30);
    d.fillTriangleRGBA(0, 0, 10, 0, 5, Number.POSITIVE_INFINITY, 1, 2, 3, 255);
    d.fillTriangleRGBA(0, 0, 10, Number.NaN, 5, 5, 1, 2, 3, 255);
    expect(countPixels(d, (p) => p[3] > 0)).toBe(0);
  });

  it("clips triangles to the clip rectangle", () => {
    const c = new RasterCanvas(30, 30);
    c.setClipRect(10, 10, 20, 20);
    c.fillTriangleRGBA(0, 0, 30, 0, 0, 30, 9, 9, 9, 255);
    c.clearClip();
    expect(pixel(c, 5, 5)[3]).toBe(0);
    expect(pixel(c, 12, 12)[3]).toBe(255);
    expect(countPixels(c, (p) => p[3] > 0)).toBeLessThanOrEqual(100);
  });

  it("handles infinite rectangles and circles", () => {
    const c = new RasterCanvas(8, 8);
    c.fillRectRGBA(2, 2, Number.POSITIVE_INFINITY, 2, 5, 5, 5, 255);
    expect(pixel(c, 7, 3)).toEqual([5, 5, 5, 255]);
    expect(pixel(c, 7, 5)[3]).toBe(0);
    const d = new RasterCanvas(8, 8);
    d.drawCircleRGBA(4, 4, Number.POSITIVE_INFINITY, 1, 1, 1, 255);
    expect(countPixels(d, (p) => p[3] === 255)).toBe(64);
    const e = new RasterCanvas(8, 8);
    e.drawCircleRGBA(3.5, 3.5, Number.NaN, 1, 1, 1, 255);
    expect(countPixels(e, (p) => p[3] > 0)).toBe(0);
  });

  it("fills polygons with the even-odd rule at pixel centers", () => {
    const c = new RasterCanvas(10, 10);
    c.fillPolygonRGBA([2, 8, 8, 2], [2, 2, 6, 6], 7, 7, 7, 255);
    // Pixels 2..7 in x and 2..5 in y have their centers inside the rectangle.
    expect(countPixels(c, (p) => p[3] === 255)).toBe(6 * 4);
    expect(pixel(c, 2, 2)[3]).toBe(255);
    expect(pixel(c, 7, 5)[3]).toBe(255);
    expect(pixel(c, 8, 5)[3]).toBe(0);
    expect(pixel(c, 7, 6)[3]).toBe(0);
    const d = new RasterCanvas(10, 10);
    d.fillPolygonRGBA([1, 5], [1, 5], 1, 1, 1, 255);
    expect(countPixels(d, (p) => p[3] > 0)).toBe(0);
  });

  it("measures and draws text per code point", () => {
    const c = new RasterCanvas(40, 20);
    const one = c.measureText("?", 7);
    expect(c.measureText("\u{1F600}", 7)).toEqual(one);
    expect(c.measureText("a\u{1F600}b", 7).width).toBe(3 * 5 + 2);
    c.drawTextRGBA("\u{1F600}", 0, 0, 0, 0, 0, 255, { fontSize: 7 });
    const ref = new RasterCanvas(40, 20);
    ref.drawTextRGBA("?", 0, 0, 0, 0, 0, 255, { fontSize: 7 });
    expect(Array.from(c.data)).toEqual(Array.from(ref.data));
  });

  it("validates fontSize and rotation, and caps huge font sizes", () => {
    const c = new RasterCanvas(10, 10);
    expect(() => c.measureText("a", Number.NaN)).toThrow(InvalidParameterError);
    expect(() => c.drawTextRGBA("a", 0, 0, 0, 0, 0, 255, { fontSize: -3 })).toThrow(
      InvalidParameterError
    );
    expect(() =>
      c.drawTextRGBA("a", 0, 0, 0, 0, 0, 255, { rotation: 45 as unknown as 90 })
    ).toThrow(InvalidParameterError);
    const t0 = Date.now();
    c.drawTextRGBA("abc", 0, 0, 0, 0, 0, 255, { fontSize: 1e9 });
    expect(Date.now() - t0).toBeLessThan(2000);
  });

  it("has glyphs for #, $, &, @, |, ~ and a full-height slash", () => {
    const question = new RasterCanvas(8, 8);
    question.drawTextRGBA("?", 0, 0, 0, 0, 0, 255, { fontSize: 7 });
    for (const ch of ["#", "$", "&", "@", "\\", "^", "`", "{", "|", "}", "~"]) {
      const c = new RasterCanvas(8, 8);
      c.drawTextRGBA(ch, 0, 0, 0, 0, 0, 255, { fontSize: 7 });
      expect(countPixels(c, (p) => p[3] > 0)).toBeGreaterThan(0);
      expect(Array.from(c.data)).not.toEqual(Array.from(question.data));
    }
    const slash = new RasterCanvas(8, 8);
    slash.drawTextRGBA("/", 0, 0, 0, 0, 0, 255, { fontSize: 7 });
    // bottom row (y = 6) of the glyph is inked, the old glyph left it empty
    let bottom = 0;
    for (let x = 0; x < 8; x++) if (pixel(slash, x, 6)[3] > 0) bottom++;
    expect(bottom).toBeGreaterThan(0);
  });

  it("rejects canvases that cannot be allocated and non-finite clip rectangles", () => {
    expect(() => new RasterCanvas(40000, 40000)).toThrow(MemoryError);
    const c = new RasterCanvas(4, 4);
    expect(() => c.setClipRect(0, 0, Number.NaN, 4)).toThrow(InvalidParameterError);
  });

  it("clearRGBA overwrites without blending", () => {
    const c = new RasterCanvas(3, 3);
    c.clearRGBA(1, 2, 3, 4);
    expect(pixel(c, 2, 2)).toEqual([1, 2, 3, 4]);
    c.clearRGBA(255, 255, 255, 255);
    c.clearRGBA(10, 20, 30, 40);
    expect(pixel(c, 0, 0)).toEqual([10, 20, 30, 40]);
  });
});

describe("Animation", () => {
  const makeFigure = (): Figure => {
    const fig = new Figure({ width: 120, height: 80 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    return fig;
  };

  it("rejects NaN/Infinity options, unknown easing and absurd frame counts", () => {
    expect(() => new Animation({ fps: Number.NaN })).toThrow(/FPS/);
    expect(() => new Animation({ fps: Number.POSITIVE_INFINITY })).toThrow(/FPS/);
    expect(() => new Animation({ duration: Number.NaN })).toThrow(/Duration/);
    expect(() => new Animation({ duration: Number.POSITIVE_INFINITY })).toThrow(/Duration/);
    expect(() => new Animation({ easing: "bounce" as unknown as "linear" })).toThrow(
      InvalidParameterError
    );
    expect(() => new Animation({ fps: 120, duration: 1e9 })).toThrow(/frames/);
  });

  it("does not add a frame because of floating-point noise", () => {
    expect(new Animation({ fps: 7.000000000000001, duration: 1000 }).info().totalFrames).toBe(7);
    expect(new Animation({ fps: 24, duration: 1010 }).info().totalFrames).toBe(25);
    expect(new Animation({ fps: 1, duration: 1 }).info().totalFrames).toBe(1);
  });

  it("passes eased progress to the generator", () => {
    const seen: number[] = [];
    const anim = new Animation({ fps: 4, duration: 1000, loop: false, easing: "ease-in" });
    anim.animate((_i, _n, progress) => {
      seen.push(progress);
      return makeFigure();
    });
    anim.render();
    // t = 0, 1/3, 2/3, 1 and ease-in(t) = t^2
    expect(seen[0]).toBeCloseTo(0, 12);
    expect(seen[1]).toBeCloseTo(1 / 9, 12);
    expect(seen[2]).toBeCloseTo(4 / 9, 12);
    expect(seen[3]).toBeCloseTo(1, 12);

    const looped: number[] = [];
    const loop = new Animation({ fps: 4, duration: 1000 });
    loop.animate((_i, _n, p) => {
      looped.push(p);
      return makeFigure();
    });
    loop.render();
    expect(looped).toEqual([0, 0.25, 0.5, 0.75]);

    const out: number[] = [];
    const ease = new Animation({ fps: 4, duration: 1000, loop: false, easing: "ease-in-out" });
    ease.animate((_i, _n, p) => {
      out.push(p);
      return makeFigure();
    });
    ease.render();
    // 2t^2 for t < 0.5, -1 + (4 - 2t) t otherwise
    expect(out[1]).toBeCloseTo(2 / 9, 12);
    expect(out[2]).toBeCloseTo(-1 + (4 - 4 / 3) * (2 / 3), 12);
  });

  it("keeps no partial state when the generator throws", () => {
    const anim = new Animation({ fps: 4, duration: 1000 });
    anim.animate((i) => {
      if (i === 2) throw new Error("boom");
      return makeFigure();
    });
    expect(() => anim.render()).toThrow("boom");
    expect(anim.getFrame(0)).toBeUndefined();
    anim.animate(() => makeFigure());
    expect(anim.toFrames()).toHaveLength(4);
  });

  it("requires a function generator", () => {
    const anim = new Animation();
    expect(() => anim.animate(null as unknown as () => Figure)).toThrow(InvalidParameterError);
  });

  it("emits a valid SMIL timeline with one discrete step per frame", () => {
    const anim = new Animation({ fps: 4, duration: 1000 });
    anim.animate(() => makeFigure());
    const svg = anim.toAnimatedSVG();
    const animates = [...svg.matchAll(/<animate [^>]*\/>/g)].map((m) => m[0]);
    expect(animates).toHaveLength(4);
    for (const [i, tag] of animates.entries()) {
      const times = /keyTimes="([^"]*)"/.exec(tag)?.[1]?.split(";").map(Number) ?? [];
      const values = /values="([^"]*)"/.exec(tag)?.[1]?.split(";").map(Number) ?? [];
      expect(times.length).toBe(values.length);
      expect(times[0]).toBe(0);
      for (let k = 1; k < times.length; k++) {
        expect(times[k] ?? 0).toBeGreaterThanOrEqual(times[k - 1] ?? 0);
        expect(times[k] ?? 0).toBeLessThanOrEqual(1);
      }
      expect(tag).toContain('calcMode="discrete"');
      expect(tag).toContain('dur="1s"');
      expect(tag).toContain('begin="0s"');
      expect(tag).toContain('repeatCount="indefinite"');
      if (i === 0) {
        expect(values).toEqual([1, 0]);
        expect(times).toEqual([0, 0.25]);
      } else if (i === 3) {
        expect(values).toEqual([0, 1]);
        expect(times).toEqual([0, 0.75]);
      } else {
        expect(values).toEqual([0, 1, 0]);
        expect(times).toEqual([0, i / 4, (i + 1) / 4]);
      }
    }
    // frame 0 is the static fallback
    expect(svg).toContain('<g class="frame-0" opacity="1">');
    expect(svg).toContain('<g class="frame-1" opacity="0">');
    expect(svg).toContain('width="120" height="80" viewBox="0 0 120 80"');
  });

  it("plays at the configured fps and freezes on the last frame when not looping", () => {
    const anim = new Animation({ fps: 24, duration: 1010, loop: false });
    anim.animate(() => makeFigure());
    const svg = anim.toAnimatedSVG();
    // 25 frames at 24 fps
    expect(svg).toContain('dur="1.04166667s"');
    expect(svg).toContain('repeatCount="1"');
    const last = [...svg.matchAll(/<animate [^>]*\/>/g)].map((m) => m[0]).pop() ?? "";
    expect(last).toContain('values="0;1"');
    expect(last).toContain('fill="freeze"');
  });

  it("renders a single frame as a static group", () => {
    const anim = new Animation({ fps: 1, duration: 1 });
    anim.animate(() => makeFigure());
    const svg = anim.toAnimatedSVG();
    expect(svg).not.toContain("<animate");
    expect(svg).toContain('<g class="frame-0" opacity="1">');
  });

  it("namespaces ids per frame so frames cannot clash", () => {
    const anim = new Animation({ fps: 3, duration: 1000 });
    anim.animate(() => makeFigure());
    const svg = anim.toAnimatedSVG();
    const ids = [...svg.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]);
    expect(ids.length).toBe(3);
    expect(new Set(ids).size).toBe(3);
    for (const id of ids) expect(svg).toContain(`url(#${id})`);
    expect(svg).not.toMatch(/url\(#dbclip/);
  });

  it("returns a copy of the frame list from render()", () => {
    const anim = new Animation({ fps: 2, duration: 1000 });
    anim.animate(() => makeFigure());
    const res = anim.render();
    expect(res.frames).toHaveLength(2);
    expect(res.frames[1]?.timeMs).toBe(500);
    (res.frames as unknown[]).length = 0;
    expect(anim.toFrames()).toHaveLength(2);
  });
});

describe("Axes data fills", () => {
  it("fill_between paints the band (SVG path and raster pixels)", () => {
    const fig = new Figure({ width: 200, height: 160 });
    const ax = fig.addAxes({ padding: 20 });
    ax.fill_between(tensor([0, 1, 2]), tensor([0, 0, 0]), tensor([1, 1, 1]), {
      color: "#ff0000",
      label: "band",
    });
    ax.legend();
    const svg = fig.renderSVG().svg;
    expect(svg).toMatch(/<path d="M[^"]* Z" fill="#ff0000" fill-rule="evenodd" stroke="none"/);
    expect(svg).toContain('class="legend-swatch"');

    const canvas = new RasterCanvas(200, 160);
    canvas.clearRGBA(255, 255, 255, 255);
    ax.renderRasterInto(canvas);
    // Middle of the band: y = 0.5 at the horizontal center of the axes.
    expect(pixel(canvas, 100, 80)).toEqual([255, 0, 0, 255]);
  });

  it("area fills between the curve and zero", () => {
    const fig = new Figure({ width: 200, height: 160 });
    const ax = fig.addAxes({ padding: 20 });
    ax.area(tensor([0, 1, 2]), tensor([2, 2, 2]), { color: "#00ff00" });
    const canvas = new RasterCanvas(200, 160);
    canvas.clearRGBA(255, 255, 255, 255);
    ax.renderRasterInto(canvas);
    expect(pixel(canvas, 100, 100)).toEqual([0, 255, 0, 255]);
    // Above the curve there is only the white axes background.
    expect(pixel(canvas, 100, 22)).toEqual([255, 255, 255, 255]);
  });

  it("fill_between and area validate lengths and drop non-finite samples", () => {
    const ax = new Figure().addAxes();
    expect(() => ax.fill_between(tensor([0, 1]), tensor([0, 1, 2]), tensor([1, 2]))).toThrow(
      ShapeError
    );
    expect(() => ax.fill_between(tensor([0, 1]), tensor([0, 1]), tensor([1]))).toThrow(ShapeError);
    expect(() => ax.area(tensor([0, 1, 2]), tensor([0, 1]))).toThrow(ShapeError);
    const d = ax.fill_between(
      tensor([0, 1, 2, 3]),
      tensor([0, 0, Number.NaN, 0]),
      tensor([1, 1, 1, 1])
    );
    // 3 usable samples -> 6 polygon vertices plus the closing vertex
    expect(d.x.length).toBe(7);
  });

  it("fill_between with no finite samples adds no stray (0, 0) point", () => {
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    const d = ax.fill_between(tensor([Number.NaN, Number.NaN]), tensor([1, 2]), tensor([3, 4]));
    expect(d.x.length).toBe(0);
    expect(d.getDataRange()).toBeNull();
    // Nothing to fit: the axes falls back to the default [0, 1] range, not data around (0, 0).
    expect(() => fig.renderSVG()).not.toThrow();
  });

  it("stackedBar fills the bars and validates series lengths", () => {
    const fig = new Figure({ width: 200, height: 160 });
    const ax = fig.addAxes({ padding: 20 });
    ax.stackedBar(tensor([1, 2]), [tensor([1, 1]), tensor([1, 1])], {
      colors: ["#ff0000", "#0000ff"],
    });
    const canvas = new RasterCanvas(200, 160);
    canvas.clearRGBA(255, 255, 255, 255);
    ax.renderRasterInto(canvas);
    expect(countPixels(canvas, (p) => p[0] === 255 && p[1] === 0 && p[2] === 0)).toBeGreaterThan(
      200
    );
    expect(countPixels(canvas, (p) => p[0] === 0 && p[1] === 0 && p[2] === 255)).toBeGreaterThan(
      200
    );
    expect(() => ax.stackedBar(tensor([1, 2]), [tensor([1, 2, 3])])).toThrow(ShapeError);
  });

  it("stackedBar treats non-finite heights as zero for the stack", () => {
    const ax = new Figure().addAxes();
    ax.stackedBar(tensor([1, 2]), [tensor([Number.NaN, 2]), tensor([3, 3])]);
    const ranges = drawablesOf(ax).map((d) => d.getDataRange());
    // first series: only the second bar; second series: bars start at 0 and 2
    expect(ranges).toHaveLength(3);
    expect(ranges[1]?.ymax).toBe(3);
    expect(ranges[2]?.ymax).toBe(5);
    expect(ranges[1]?.ymin).toBe(0);
  });

  it("groupedBar packs each group into a total width of 0.8", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.groupedBar(tensor([1, 2]), [tensor([1, 2]), tensor([2, 1]), tensor([3, 3])]);
    const rects = [
      ...fig
        .renderSVG()
        .svg.matchAll(
          /<rect x="([-\d.]+)" y="[-\d.]+" width="([-\d.]+)" height="[-\d.]+" fill="#[0-9a-f]+" stroke="#000000"/g
        ),
    ].map((m) => [Number(m[1]), Number(m[2])] as const);
    expect(rects).toHaveLength(6);
    // bars of the same group are adjacent and equally wide, never overlapping
    for (const g of [0, 1]) {
      const [a, b, c] = [rects[g], rects[g + 2], rects[g + 4]];
      // series order in the SVG is series-major: s0 g0, s0 g1, s1 g0, ...
      expect(a?.[1]).toBeCloseTo(b?.[1] ?? 0, 6);
      expect((a?.[0] ?? 0) + (a?.[1] ?? 0)).toBeCloseTo(b?.[0] ?? 0, 1);
      expect((b?.[0] ?? 0) + (b?.[1] ?? 0)).toBeCloseTo(c?.[0] ?? 0, 1);
    }
    // total group width is 0.8 of the 1-unit category spacing
    const spacing = (rects[1]?.[0] ?? 0) - (rects[0]?.[0] ?? 0);
    const groupWidth = (rects[4]?.[0] ?? 0) + (rects[4]?.[1] ?? 0) - (rects[0]?.[0] ?? 0);
    expect(groupWidth / spacing).toBeCloseTo(0.8, 1);
    expect(() => ax.groupedBar(tensor([1, 2]), [tensor([1])])).toThrow(ShapeError);
  });
});

describe("Axes.step", () => {
  const x = tensor([0, 1, 3, 4]);
  const y = tensor([2, 5, 1, 3]);

  it("matches the NumPy reference for pre, post and mid", () => {
    const ax = new Figure().addAxes();
    const post = ax.step(x, y);
    expect(Array.from(post.x)).toEqual([0, 1, 1, 3, 3, 4, 4]);
    expect(Array.from(post.y)).toEqual([2, 2, 5, 5, 1, 1, 3]);
    const pre = ax.step(x, y, { where: "pre" });
    expect(Array.from(pre.x)).toEqual([0, 0, 1, 1, 3, 3, 4]);
    expect(Array.from(pre.y)).toEqual([2, 5, 5, 1, 1, 3, 3]);
    const mid = ax.step(x, y, { where: "mid" });
    expect(Array.from(mid.x)).toEqual([0, 0.5, 0.5, 2, 2, 3.5, 3.5, 4]);
    expect(Array.from(mid.y)).toEqual([2, 2, 5, 5, 1, 1, 3, 3]);
  });

  it("validates lengths, where and handles tiny inputs", () => {
    const ax = new Figure().addAxes();
    expect(() => ax.step(tensor([0, 1]), tensor([0, 1, 2]))).toThrow(ShapeError);
    expect(() => ax.step(x, y, { where: "left" as unknown as "pre" })).toThrow(
      InvalidParameterError
    );
    const one = ax.step(tensor([1]), tensor([2]), { where: "mid" });
    expect(Array.from(one.x)).toEqual([1]);
    const none = ax.step(tensor([]), tensor([]));
    expect(none.x.length).toBe(0);
  });
});

describe("Axes.errorbar", () => {
  it("supports asymmetric [2, n] errors", () => {
    const ax = new Figure().addAxes();
    ax.errorbar(
      tensor([0, 1]),
      tensor([10, 20]),
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    const ranges = drawablesOf(ax).map((d) => d.getDataRange());
    expect(ranges).toHaveLength(3);
    expect(ranges[1]).toMatchObject({ ymin: 9, ymax: 13 });
    expect(ranges[2]).toMatchObject({ ymin: 18, ymax: 24 });
  });

  it("uses symmetric 1D errors and rejects bad shapes and negative errors", () => {
    const ax = new Figure().addAxes();
    ax.errorbar(tensor([0, 1]), tensor([10, 20]), tensor([1, 2]));
    expect(drawablesOf(ax)[2]?.getDataRange()).toMatchObject({ ymin: 18, ymax: 22 });
    expect(() => ax.errorbar(tensor([0, 1]), tensor([10, 20]), tensor([1]))).toThrow(ShapeError);
    expect(() => ax.errorbar(tensor([0, 1]), tensor([10, 20]), tensor([1, -2]))).toThrow(
      InvalidParameterError
    );
    expect(() => ax.errorbar(tensor([0, 1]), tensor([10, 20]), tensor([[1, 2, 3]]))).toThrow(
      ShapeError
    );
    expect(() => ax.errorbar(tensor([0, 1, 2]), tensor([10, 20]), tensor([1, 2, 3]))).toThrow(
      ShapeError
    );
  });
});

describe("Axes limits and scales", () => {
  it("validates xlim/ylim", () => {
    const ax = new Figure().addAxes();
    expect(() => ax.xlim(Number.NaN, 1)).toThrow(InvalidParameterError);
    expect(() => ax.ylim(0, Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
    expect(() => ax.xlim(2, 2)).toThrow(InvalidParameterError);
  });

  it("supports reversed limits (ticks and data are mirrored)", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 10]), tensor([0, 1]));
    ax.xlim(10, 0);
    const svg = fig.renderSVG().svg;
    expect(tickX(svg, "0")).toBeGreaterThan(tickX(svg, "10"));
    const [line] = polylinePoints(svg);
    expect(line?.[0]?.[0]).toBeGreaterThan(line?.[1]?.[0] ?? Number.POSITIVE_INFINITY);
  });

  it("keeps ticks that sit on a limit despite rounding error", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.xlim(0, 0.30000000000000004);
    ax.setXTicks([0.1 + 0.2], ["target"]);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain(">target</text>");
    ax.setXTicks([]);
    expect(fig.renderSVG().svg).not.toContain(">target</text>");
  });

  it("pads automatic log ranges in log space", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 10, 100]), tensor([1, 2, 3]));
    ax.setXScale("log");
    const svg = fig.renderSVG().svg;
    // Axes viewport is x = 50 .. 350; data 1..100 spans 2 decades, padded by 5%.
    // Python: 10 ** (0 - 0.1) = 0.794..., 10 ** (2 + 0.1) = 125.89...
    const lo = Math.log10(0.7943282347242815);
    const hi = Math.log10(125.89254117941675);
    const px = (v: number) => 50 + ((Math.log10(v) - lo) / (hi - lo)) * 300;
    expect(tickX(svg, "1")).toBeCloseTo(px(1), 1);
    expect(tickX(svg, "100")).toBeCloseTo(px(100), 1);
    expect(tickX(svg, "10")).toBeCloseTo(px(10), 1);
  });

  it("falls back sensibly when log data includes non-positive values", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.bar(tensor([1, 2]), tensor([100, 1000]));
    ax.setYScale("log");
    const svg = fig.renderSVG().svg;
    expect(svg).toContain(">1000</text>");
    expect(svg).toContain(">0.1</text>");
    expect(svg).not.toContain("NaN");
  });

  it("rejects non-positive limits on log axes and unknown scales", () => {
    const ax = new Figure().addAxes();
    ax.setYScale("log");
    expect(() => ax.ylim(0, 10)).toThrow(InvalidParameterError);
    const bx = new Figure().addAxes();
    bx.xlim(-1, 5);
    expect(() => bx.setXScale("log")).toThrow(InvalidParameterError);
    expect(() => bx.setYScale("sqrt" as unknown as "log")).toThrow(InvalidParameterError);
    expect(() => bx.setXScale("sqrt" as unknown as "log")).toThrow(InvalidParameterError);
  });
});

describe("Axes.twinx", () => {
  it("shares one x range between the axes", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax1 = fig.addAxes();
    ax1.plot(tensor([0, 1]), tensor([0, 1]));
    const ax2 = ax1.twinx();
    ax2.plot(tensor([0, 10]), tensor([0, 100]));
    const [parent, twin] = polylinePoints(fig.renderSVG().svg);
    // x = 0 maps to the same pixel for both axes
    expect(parent?.[0]?.[0]).toBeCloseTo(twin?.[0]?.[0] ?? Number.NaN, 6);
    // the parent's x = 1 is a tenth of the way across the shared 0..10 (plus margin) range
    const lo = -0.5;
    const hi = 10.5;
    expect(parent?.[1]?.[0]).toBeCloseTo(50 + ((1 - lo) / (hi - lo)) * 300, 1);
    expect(twin?.[1]?.[0]).toBeCloseTo(50 + ((10 - lo) / (hi - lo)) * 300, 1);
  });

  it("applies xlim, scale and tick overrides of a twin to the shared axis", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax1 = fig.addAxes();
    ax1.plot(tensor([0, 1]), tensor([0, 1]));
    const ax2 = ax1.twinx();
    ax2.plot(tensor([0, 1]), tensor([5, 6]));
    ax2.xlim(0, 2);
    ax2.setXTicks([1], ["one"]);
    const svg = fig.renderSVG().svg;
    const [parent] = polylinePoints(svg);
    expect(parent?.[1]?.[0]).toBeCloseTo(50 + 0.5 * 300, 6);
    expect(tickX(svg, "one")).toBeCloseTo(200, 6);
  });

  it("a twin's log x scale applies to the parent", () => {
    const fig = new Figure();
    const ax1 = fig.addAxes();
    const ax2 = ax1.twinx();
    ax2.setXScale("log");
    expect(() => ax1.xlim(-1, 1)).toThrow(InvalidParameterError);
    expect(() => ax2.xlim(0, 1)).toThrow(InvalidParameterError);
  });
});

describe("Reference lines, annotations and grid", () => {
  it("includes axhline/axvline values in the automatic range", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.axhline(10);
    ax.axvline(5);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('class="axhline"');
    expect(svg).toContain('class="axvline"');
    // y range 0..10 plus 5% margin: the line at 10 is inside the viewport
    const m = /class="axhline" x1="[\d.]+" y1="([\d.]+)"/.exec(svg);
    expect(Number(m?.[1])).toBeGreaterThan(50);
    expect(Number(m?.[1])).toBeLessThan(250);
  });

  it("ignores non-positive reference lines on a log axis", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.setYScale("log");
    ax.setXScale("log");
    ax.plot(tensor([1, 10, 100]), tensor([1, 10, 100]));
    const before = fig.renderSVG().svg;
    ax.axhline(-5);
    ax.axhline(0);
    ax.axvline(-1);
    const after = fig.renderSVG().svg;
    expect(after).not.toContain('class="axhline"');
    expect(after).not.toContain('class="axvline"');
    // The range is unchanged: the tick labels are the same as without the lines.
    const labels = (svg: string) =>
      [...svg.matchAll(/class="tick-label[^>]*>([^<]*)</g)].map((m) => m[1]);
    expect(labels(after)).toEqual(labels(before));
    const canvas = new RasterCanvas(400, 300);
    canvas.clearRGBA(255, 255, 255, 255);
    expect(() => ax.renderRasterInto(canvas)).not.toThrow();
  });

  it("does not draw reference lines outside explicit limits", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.ylim(0, 1);
    ax.xlim(0, 1);
    ax.axhline(100);
    ax.axvline(-100);
    const svg = fig.renderSVG().svg;
    expect(svg).not.toContain('class="axhline"');
    expect(svg).not.toContain('class="axvline"');
    const canvas = new RasterCanvas(400, 300);
    canvas.clearRGBA(255, 255, 255, 255);
    ax.renderRasterInto(canvas);
    expect(
      countPixels(canvas, (p) => p[0] === 255 && p[1] === 255 && p[2] === 255)
    ).toBeGreaterThan(0);
  });

  it("lists labeled reference lines in the legend", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]), { label: "series" });
    ax.axhline(0.5, { label: "threshold", color: "#ff0000", linewidth: 3 });
    ax.axvline(0.5, { label: "cutoff" });
    ax.legend();
    const svg = fig.renderSVG().svg;
    expect(svg).toContain(">threshold</text>");
    expect(svg).toContain(">cutoff</text>");
    expect(svg).toMatch(/class="legend-line"[^>]*stroke="#ff0000" stroke-width="3"/);
  });

  it("validates reference lines and annotations", () => {
    const ax = new Figure().addAxes();
    expect(() => ax.axhline(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => ax.axvline(Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
    expect(() => ax.axhline(1, { linewidth: 0 })).toThrow(InvalidParameterError);
    expect(() => ax.axvline(1, { linewidth: -1 })).toThrow(InvalidParameterError);
    expect(() => ax.annotate("a", 0, 0, { fontSize: 0 })).toThrow(InvalidParameterError);
    expect(() => ax.annotate("a", 0, 0, { fontSize: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("draws dashed reference lines in PNG output", () => {
    const fig = new Figure({ width: 100, height: 100 });
    const ax = fig.addAxes({ padding: 10 });
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.axhline(0.5, { color: "#ff0000", linewidth: 1 });
    const canvas = new RasterCanvas(100, 100);
    canvas.clearRGBA(255, 255, 255, 255);
    ax.renderRasterInto(canvas);
    // Find the dashed row by looking for the red pixels.
    const rows = new Map<number, number[]>();
    for (let y = 0; y < 100; y++) {
      for (let x = 0; x < 100; x++) {
        const p = pixel(canvas, x, y);
        if (p[0] === 255 && p[1] === 0 && p[2] === 0) rows.set(y, [...(rows.get(y) ?? []), x]);
      }
    }
    expect(rows.size).toBe(1);
    const xs = [...rows.values()][0] ?? [];
    // 4 pixels on, 3 off
    expect(xs.slice(0, 5)).toEqual([
      xs[0],
      (xs[0] ?? 0) + 1,
      (xs[0] ?? 0) + 2,
      (xs[0] ?? 0) + 3,
      (xs[0] ?? 0) + 7,
    ]);
  });

  it("draws annotations and the grid in PNG output", () => {
    const fig = new Figure({ width: 160, height: 120 });
    const ax = fig.addAxes({ padding: 20 });
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.grid(true, { color: "#ff0000" });
    ax.annotate("Hi", 0.3, 0.6, { color: "#0000ff", fontSize: 10 });
    const canvas = new RasterCanvas(160, 120);
    canvas.clearRGBA(255, 255, 255, 255);
    ax.renderRasterInto(canvas);
    expect(countPixels(canvas, (p) => p[0] === 255 && p[1] === 0 && p[2] === 0)).toBeGreaterThan(
      100
    );
    expect(countPixels(canvas, (p) => p[0] === 0 && p[1] === 0 && p[2] === 255)).toBeGreaterThan(5);
  });

  it("normalizes the grid color", () => {
    const fig = new Figure({ width: 160, height: 120 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.grid(true, { color: "RED" });
    expect(fig.renderSVG().svg).toContain('stroke="#ff0000" stroke-width="0.5"');
  });
});

describe("Raster legend and rendering edge cases", () => {
  it("sizes the PNG legend box from the bitmap font width", () => {
    const fig = new Figure({ width: 400, height: 200 });
    const ax = fig.addAxes({ padding: 20 });
    ax.plot(tensor([0, 1]), tensor([0, 1]), { label: "abcdefghij", color: "#00ff00" });
    ax.legend({ borderColor: "#ff0000", background: "#ffffff" });
    const canvas = new RasterCanvas(400, 200);
    canvas.clearRGBA(255, 255, 255, 255);
    ax.renderRasterInto(canvas);
    let borderMax = -1;
    let textMax = -1;
    for (let y = 0; y < 200; y++) {
      for (let x = 0; x < 400; x++) {
        const p = pixel(canvas, x, y);
        if (p[0] === 255 && p[1] === 0 && p[2] === 0) borderMax = Math.max(borderMax, x);
      }
    }
    // text pixels are black and lie inside the legend (upper right quadrant, below the top)
    for (let y = 26; y < 50; y++) {
      for (let x = 200; x < 378; x++) {
        const p = pixel(canvas, x, y);
        if (p[0] === 0 && p[1] === 0 && p[2] === 0) textMax = Math.max(textMax, x);
      }
    }
    expect(textMax).toBeGreaterThan(0);
    expect(textMax).toBeLessThan(borderMax);
  });

  it("clears the clip rectangle even when a drawable throws", () => {
    const fig = new Figure({ width: 60, height: 60 });
    const ax = fig.addAxes({ padding: 10 });
    drawablesOf(ax).push({
      kind: "boom",
      getDataRange: () => null,
      drawSVG: () => {},
      drawRaster: () => {
        throw new Error("boom");
      },
    });
    const canvas = new RasterCanvas(60, 60);
    expect(() => ax.renderRasterInto(canvas)).toThrow("boom");
    canvas.setPixelRGBA(0, 0, 1, 2, 3, 255);
    expect(pixel(canvas, 0, 0)).toEqual([1, 2, 3, 255]);
  });

  it("reports an unallocatable PNG as MemoryError", async () => {
    const fig = new Figure({ width: 32768, height: 32768 });
    await expect(fig.renderPNG()).rejects.toThrow(MemoryError);
  });

  it("renders PDF for filled areas without throwing", () => {
    const fig = new Figure({ width: 120, height: 90 });
    const ax = fig.addAxes();
    ax.fill_between(tensor([0, 1]), tensor([0, 0]), tensor([1, 2]));
    expect(fig.renderPDF().bytes.length).toBeGreaterThan(100);
  });
});

describe("state helpers (figure / gca / subplot)", () => {
  it("drops the untouched default axes when subplots are requested", () => {
    const fig = figure({ width: 300, height: 200 });
    expect(fig.axesList).toHaveLength(1);
    const a = subplot(1, 2, 1);
    const b = subplot(1, 2, 2);
    expect(fig.axesList).toEqual([a, b]);
    const rects = fig.renderSVG().svg.match(/<rect [^>]*stroke="#000"/g) ?? [];
    expect(rects).toHaveLength(2);
  });

  it("keeps a default axes that already holds content", () => {
    const fig = figure({ width: 300, height: 200 });
    gca().setTitle("keep me");
    subplot(1, 2, 1);
    expect(fig.axesList).toHaveLength(2);
  });

  it("returns the same axes for the same subplot position", () => {
    figure({ width: 300, height: 200 });
    const a = subplot(2, 2, 3);
    const again = subplot(2, 2, 3);
    expect(again).toBe(a);
    expect(gca()).toBe(a);
    const other = subplot(2, 2, 4);
    expect(other).not.toBe(a);
    expect(subplot(2, 2, 3)).toBe(a);
    expect(gca()).toBe(a);
  });

  it("works from an implicit figure created by gca()", () => {
    figure();
    const fig = gca().fig;
    expect(fig.axesList).toHaveLength(1);
    subplot(1, 1, 1);
    expect(fig.axesList).toHaveLength(1);
    expect(fig.axesList[0]).toBe(gca());
  });
});
