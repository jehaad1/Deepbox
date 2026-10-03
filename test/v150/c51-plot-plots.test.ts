import { inflateSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import { InvalidParameterError, ShapeError } from "../../src/core";
import { Tensor, tensor, transpose } from "../../src/ndarray";
import { RasterCanvas } from "../../src/plot/canvas/RasterCanvas";
import { Scatter3D, Surface3D, Wireframe3D } from "../../src/plot/plots/Surface3D";
import { Violinplot } from "../../src/plot/plots/Violinplot";
import { Waterfall2D } from "../../src/plot/plots/Waterfall2D";
import { svgToPdf } from "../../src/plot/renderers/pdf";
import { pngEncodeRGBA } from "../../src/plot/renderers/png";
import { applyColormap } from "../../src/plot/utils/colormaps";
import {
  getPalette,
  getPaletteColor,
  normalizeColor,
  parseHexColorToRGBA,
} from "../../src/plot/utils/colors";
import { buildContourGrid } from "../../src/plot/utils/contours";
import {
  calculateQuartiles,
  calculateWhiskers,
  kernelDensityEstimation,
  silvermanBandwidth,
} from "../../src/plot/utils/statistics";
import { tensorToFloat64Matrix2D, tensorToFloat64Vector1D } from "../../src/plot/utils/tensor";
import { estimateTextWidth } from "../../src/plot/utils/text";
import { generateLogTicks, generateTicks } from "../../src/plot/utils/ticks";
import { computeAutoRange, makeTransform } from "../../src/plot/utils/transforms";
import { assertPositiveInt, clampInt } from "../../src/plot/utils/validation";
import { escapeXml } from "../../src/plot/utils/xml";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const decodeLatin1 = (bytes: Uint8Array): string => new TextDecoder("latin1").decode(bytes);

describe("escapeXml", () => {
  it("escapes the five predefined entities", () => {
    expect(escapeXml(`a&b<c>"d'`)).toBe("a&amp;b&lt;c&gt;&quot;d&apos;");
  });

  it("removes characters XML 1.0 cannot represent and repairs lone surrogates", () => {
    expect(escapeXml("a\u0000b\u0008c\u000bd￾e")).toBe("abcde");
    expect(escapeXml("x\ud800y")).toBe("x�y");
    expect(escapeXml("x\udc00y")).toBe("x�y");
  });

  it("keeps tab, newline, carriage return and valid surrogate pairs", () => {
    expect(escapeXml("a\tb\nc\rd")).toBe("a\tb\nc\rd");
    expect(escapeXml("smile \u{1f600}")).toBe("smile \u{1f600}");
  });

  it("returns the input unchanged when nothing needs escaping", () => {
    const s = "plain text 123";
    expect(escapeXml(s)).toBe(s);
  });
});

describe("validation helpers", () => {
  it("assertPositiveInt rejects values beyond the safe integer range and non-numbers", () => {
    expect(() => assertPositiveInt("n", 3)).not.toThrow();
    expect(() => assertPositiveInt("n", 2 ** 60)).toThrow(InvalidParameterError);
    expect(() => assertPositiveInt("n", 0)).toThrow(InvalidParameterError);
    expect(() => assertPositiveInt("n", 1.5)).toThrow(InvalidParameterError);
    expect(() => assertPositiveInt("n", Number.NaN)).toThrow(InvalidParameterError);
    expect(() => assertPositiveInt("n", "3" as unknown as number)).toThrow(InvalidParameterError);
  });

  it("clampInt never returns negative zero", () => {
    expect(Object.is(clampInt(-0.5, -5, 5), 0)).toBe(true);
    expect(clampInt(2.9, 0, 10)).toBe(2);
    expect(clampInt(-3.9, -5, 5)).toBe(-3);
    expect(clampInt(99, 0, 10)).toBe(10);
    expect(clampInt(Number.NaN, 4, 10)).toBe(4);
  });

  it("estimateTextWidth guards against invalid font sizes", () => {
    expect(estimateTextWidth("abc", 10)).toBeCloseTo(18, 12);
    expect(estimateTextWidth("abc", -3)).toBe(0);
    expect(estimateTextWidth("abc", Number.NaN)).toBe(0);
  });
});

describe("generateTicks", () => {
  it("labels fractional steps without rounding them away", () => {
    // step 0.25: the old formatter used one decimal and printed 0.25 as "0.3".
    const quarter = generateTicks(0, 1, 4);
    expect(quarter.map((t) => t.value)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(quarter.map((t) => t.label)).toEqual(["0", "0.25", "0.5", "0.75", "1"]);
    // step 2.5: the old formatter used zero decimals and printed 2.5 as "3".
    const twoHalf = generateTicks(0, 10, 4);
    expect(twoHalf.map((t) => t.label)).toEqual(["0", "2.5", "5", "7.5", "10"]);
  });

  it("produces exact multiples of the step with no accumulated error", () => {
    const ticks = generateTicks(0, 1, 11);
    expect(ticks.map((t) => t.value)).toEqual([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]);
  });

  it("terminates when the step is below the floating point resolution", () => {
    const ticks = generateTicks(1e16, 1e16 + 2);
    expect(ticks.length).toBeGreaterThan(0);
    expect(ticks.length).toBeLessThanOrEqual(2);
  });

  it("caps absurd tick requests and still rejects invalid ones", () => {
    expect(generateTicks(0, 1, 1e9).length).toBeLessThanOrEqual(1001);
    expect(generateTicks(0, 1, 0)).toEqual([]);
    expect(generateTicks(0, 1, Number.NaN)).toEqual([]);
    expect(generateTicks(0, Number.POSITIVE_INFINITY)).toEqual([]);
  });

  it("keeps neighbouring ticks distinct on axes with a large offset", () => {
    const ticks = generateTicks(1e12, 1e12 + 10);
    const values = ticks.map((t) => t.value);
    expect(values).toEqual([1e12, 1e12 + 2, 1e12 + 4, 1e12 + 6, 1e12 + 8, 1e12 + 10]);
    expect(new Set(ticks.map((t) => t.label)).size).toBe(ticks.length);
  });

  it("returns +0 rather than -0 for the zero tick", () => {
    const zero = generateTicks(-1, 1).find((t) => t.label === "0");
    expect(Object.is(zero?.value, 0)).toBe(true);
  });
});

describe("generateLogTicks", () => {
  it("falls back to 1-2-5 ticks when less than two decades are visible", () => {
    expect(generateLogTicks(2, 8).map((t) => t.value)).toEqual([2, 5]);
    expect(generateLogTicks(1, 6).map((t) => t.value)).toEqual([1, 2, 5]);
  });

  it("thins decades when there are more than maxTicks", () => {
    const ticks = generateLogTicks(1e-30, 1e30);
    expect(ticks.length).toBeLessThanOrEqual(10);
    expect(ticks.length).toBeGreaterThan(2);
    const all = generateLogTicks(1e-30, 1e30, 100);
    expect(all.length).toBe(61);
  });

  it("keeps one tick per decade for a normal range", () => {
    expect(generateLogTicks(0.5, 2000).map((t) => t.value)).toEqual([1, 10, 100, 1000]);
    expect(generateLogTicks(-1, 10)).toEqual([]);
  });
});

describe("transforms", () => {
  const drawable = (range: { xmin: number; xmax: number; ymin: number; ymax: number } | null) => ({
    kind: "t",
    getDataRange: () => range,
    drawSVG: () => {},
    drawRaster: () => {},
  });

  it("computeAutoRange ignores non-finite bounds instead of discarding all data", () => {
    const r = computeAutoRange([
      drawable({ xmin: 0, xmax: 10, ymin: 0, ymax: 10 }),
      drawable({ xmin: Number.NaN, xmax: 20, ymin: 0, ymax: Number.POSITIVE_INFINITY }),
    ]);
    // data extent x:[0,20] y:[0,10], padded by 5%
    expect(r.xmin).toBeCloseTo(-1, 12);
    expect(r.xmax).toBeCloseTo(21, 12);
    expect(r.ymin).toBeCloseTo(-0.5, 12);
    expect(r.ymax).toBeCloseTo(10.5, 12);
  });

  it("makeTransform collapses a non-finite range instead of returning NaN", () => {
    const t = makeTransform(
      { xmin: 0, xmax: Number.NaN, ymin: 0, ymax: 1 },
      { x: 10, y: 20, width: 100, height: 50 }
    );
    expect(t.xToPx(5)).toBe(10);
    expect(t.yToPx(0)).toBe(70);
    expect(t.yToPx(1)).toBe(20);
  });
});

describe("tensor conversion", () => {
  it("reads strided and offset views correctly", () => {
    const view = Tensor.fromTypedArray({
      data: new Float64Array([0, 1, 2, 3, 4, 5]),
      shape: [3],
      dtype: "float64",
      device: "cpu",
      offset: 1,
      strides: [2],
    });
    expect(Array.from(tensorToFloat64Vector1D(view))).toEqual([1, 3, 5]);
    const ints = Tensor.fromTypedArray({
      data: new Int32Array([9, 8, 7, 6, 5, 4]),
      shape: [2],
      dtype: "int32",
      device: "cpu",
      offset: 2,
      strides: [3],
    });
    expect(Array.from(tensorToFloat64Vector1D(ints))).toEqual([7, 4]);
  });

  it("returns a copy that does not alias the tensor storage", () => {
    const t = tensor([1, 2, 3], { dtype: "float64" });
    const out = tensorToFloat64Vector1D(t);
    out[0] = 99;
    expect(t.data[0]).toBe(1);
  });

  it("converts int64 (bigint) tensors and 2D non-contiguous views", () => {
    const big = tensor([1, 2, 3], { dtype: "int64" });
    expect(Array.from(tensorToFloat64Vector1D(big))).toEqual([1, 2, 3]);
    const m = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const mt = transpose(m);
    const out = tensorToFloat64Matrix2D(mt);
    expect(out.rows).toBe(3);
    expect(out.cols).toBe(2);
    expect(Array.from(out.data)).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("reports the received rank in dimension errors", () => {
    expect(() => tensorToFloat64Vector1D(tensor([[1, 2]]))).toThrow(/ndim=2/);
    expect(() => tensorToFloat64Matrix2D(tensor([1, 2]))).toThrow(/ndim=1/);
  });
});

describe("color parsing", () => {
  it("supports #rgb and #rgba shorthand (previously parsed as black)", () => {
    expect(parseHexColorToRGBA("#f80")).toEqual({ r: 255, g: 136, b: 0, a: 255 });
    expect(parseHexColorToRGBA("#f808")).toEqual({ r: 255, g: 136, b: 0, a: 136 });
    expect(normalizeColor("#abc", "#000000")).toBe("#aabbcc");
  });

  it("supports percentage, decimal and space separated rgb()", () => {
    expect(parseHexColorToRGBA("rgb(100%, 0%, 50%)")).toEqual({ r: 255, g: 0, b: 128, a: 255 });
    expect(parseHexColorToRGBA("rgb(255 0 0 / 50%)")).toEqual({ r: 255, g: 0, b: 0, a: 128 });
    expect(parseHexColorToRGBA("rgb(12.4, 99.6, 3)")).toEqual({ r: 12, g: 100, b: 3, a: 255 });
  });

  it("wraps negative hsl hues and handles transparent", () => {
    expect(parseHexColorToRGBA("hsl(-120, 100%, 50%)")).toEqual({ r: 0, g: 0, b: 255, a: 255 });
    expect(parseHexColorToRGBA("hsla(120, 100%, 25%, 0.5)")).toEqual({
      r: 0,
      g: 128,
      b: 0,
      a: 128,
    });
    expect(parseHexColorToRGBA("transparent").a).toBe(0);
    expect(normalizeColor("transparent", "#123456")).toBe("#00000000");
  });

  it("treats Object.prototype names and unknown text as black", () => {
    expect(parseHexColorToRGBA("constructor")).toEqual({ r: 0, g: 0, b: 0, a: 255 });
    expect(parseHexColorToRGBA("__proto__")).toEqual({ r: 0, g: 0, b: 0, a: 255 });
    expect(parseHexColorToRGBA("rgb(1,2)")).toEqual({ r: 0, g: 0, b: 0, a: 255 });
    expect(parseHexColorToRGBA(undefined as unknown as string)).toEqual({
      r: 0,
      g: 0,
      b: 0,
      a: 255,
    });
  });

  it("keeps parsing correctly after the cache is flushed", () => {
    for (let i = 0; i < 2000; i++) parseHexColorToRGBA(`rgb(${i % 256}, 0, ${i % 7})`);
    expect(parseHexColorToRGBA("#102030")).toEqual({ r: 16, g: 32, b: 48, a: 255 });
  });
});

describe("palettes", () => {
  it("getPaletteColor wraps negative indices and rejects fractions", () => {
    expect(getPaletteColor("tab10", -1)).toBe("#17becf");
    expect(getPaletteColor("tab10", 11)).toBe("#ff7f0e");
    expect(() => getPaletteColor("tab10", 1.5)).toThrow(InvalidParameterError);
  });

  it("getPalette does not resolve inherited object properties", () => {
    expect(() => getPalette("constructor")).toThrow(InvalidParameterError);
    expect(() => getPalette("toString")).toThrow(InvalidParameterError);
  });

  it("cividis palette matches matplotlib's cividis sampled at i/9", () => {
    // matplotlib.colormaps["cividis"](i / 9) rounded to #rrggbb
    expect(getPalette("cividis")).toEqual([
      "#00224e",
      "#123570",
      "#3b496c",
      "#575d6d",
      "#707173",
      "#8a8678",
      "#a59c74",
      "#c3b369",
      "#e1cc55",
      "#fee838",
    ]);
  });

  it("getPalette returns a copy, so callers cannot corrupt the shared palette", () => {
    const p = getPalette("tab10") as string[];
    p[0] = "#000000";
    p.push("#ffffff");
    expect(getPalette("tab10")).toHaveLength(10);
    expect(getPalette("tab10")[0]).toBe("#1f77b4");
    expect(getPaletteColor("tab10", 0)).toBe("#1f77b4");
  });
});

describe("applyColormap", () => {
  // matplotlib.colormaps[name](t) * 255, rounded
  const reference: Record<string, Record<number, [number, number, number]>> = {
    viridis: { 0: [68, 1, 84], 0.37: [45, 113, 142], 0.9: [189, 223, 38], 1: [253, 231, 37] },
    plasma: { 0: [13, 8, 135], 0.37: [167, 33, 151], 0.95: [247, 228, 37], 1: [240, 249, 33] },
    inferno: { 0: [0, 0, 4], 0.37: [135, 33, 107], 0.95: [241, 239, 117], 1: [252, 255, 164] },
    magma: { 0: [0, 0, 4], 0.37: [128, 37, 130], 0.95: [253, 231, 169], 1: [252, 253, 191] },
  };

  it("tracks matplotlib within a few levels over the whole range, including the top end", () => {
    for (const [name, points] of Object.entries(reference)) {
      for (const [t, rgb] of Object.entries(points)) {
        const got = applyColormap(Number(t), name as "viridis");
        for (let k = 0; k < 3; k++) {
          expect(Math.abs((got[k] ?? 0) - (rgb[k] ?? 0))).toBeLessThanOrEqual(4);
        }
      }
    }
  });

  it("no longer saturates before t = 1 (plasma, inferno and magma repeated their last color)", () => {
    for (const name of ["plasma", "inferno", "magma"] as const) {
      expect(applyColormap(0.95, name)).not.toEqual(applyColormap(1, name));
      expect(applyColormap(0.9, name)).not.toEqual(applyColormap(0.95, name));
    }
  });

  it("grayscale is exactly linear", () => {
    expect(applyColormap(0.5, "grayscale")).toEqual([128, 128, 128]);
    expect(applyColormap(0.2, "grayscale")).toEqual([51, 51, 51]);
  });

  it("clamps out-of-range values, maps NaN to black, rejects unknown names", () => {
    expect(applyColormap(-5, "viridis")).toEqual(applyColormap(0, "viridis"));
    expect(applyColormap(7, "viridis")).toEqual(applyColormap(1, "viridis"));
    expect(applyColormap(Number.NaN, "viridis")).toEqual([0, 0, 0]);
    expect(() => applyColormap(0.5, "nope" as "viridis")).toThrow(InvalidParameterError);
    expect(() => applyColormap(0.5, "constructor" as "viridis")).toThrow(InvalidParameterError);
  });
});

describe("statistics helpers", () => {
  it("quartiles match numpy.percentile (linear)", () => {
    // numpy.percentile([1,2,2,3,7,9], [25,50,75]) -> [2.0, 2.5, 6.0]
    expect(calculateQuartiles([1, 2, 2, 3, 7, 9])).toEqual({ q1: 2, median: 2.5, q3: 6 });
    // numpy.percentile([0.1,0.35,2.2,2.2,3.9,4.4,100.0], [25,50,75])
    const q = calculateQuartiles(new Float64Array([0.1, 0.35, 2.2, 2.2, 3.9, 4.4, 100]));
    expect(q.q1).toBeCloseTo(1.275, 12);
    expect(q.median).toBeCloseTo(2.2, 12);
    expect(q.q3).toBeCloseTo(4.15, 12);
  });

  it("Silverman bandwidth and KDE match scipy.stats.gaussian_kde(bw_method='silverman')", () => {
    const data = [1, 2, 2, 3, 7, 9];
    // kde.factor * data.std(ddof=1)
    expect(silvermanBandwidth(data)).toBeCloseTo(2.387119535268336, 12);
    const got = kernelDensityEstimation(data, [0, 1, 2.5, 5, 9, 12], 0);
    const expected = [0.077778, 0.09977525, 0.11000773, 0.0781918, 0.04950362, 0.0157832];
    got.forEach((v, i) => {
      expect(v).toBeCloseTo(expected[i] ?? 0, 7);
    });
  });

  it("KDE integrates to one and accepts typed arrays and an explicit bandwidth", () => {
    const grid: number[] = [];
    for (let i = 0; i <= 2000; i++) grid.push(-10 + (i * 30) / 2000);
    const dens = kernelDensityEstimation(new Float64Array([0, 1, 2, 5]), grid, 0.7);
    const step = 30 / 2000;
    const area = dens.reduce((s, v) => s + v * step, 0);
    expect(area).toBeCloseTo(1, 4);
  });

  it("KDE ignores non-finite samples and returns zeros without data", () => {
    const clean = kernelDensityEstimation([1, 2, 3], [2], 0.5);
    const dirty = kernelDensityEstimation(
      [1, Number.NaN, 2, Number.POSITIVE_INFINITY, 3],
      [2],
      0.5
    );
    expect(dirty).toEqual(clean);
    expect(kernelDensityEstimation([], [0, 1], 1)).toEqual([0, 0]);
    expect(kernelDensityEstimation([Number.NaN], [0, 1], 1)).toEqual([0, 0]);
  });

  it("silvermanBandwidth is finite and positive for degenerate samples", () => {
    expect(silvermanBandwidth([5])).toBe(1);
    expect(silvermanBandwidth([5, 5, 5])).toBe(1);
    expect(silvermanBandwidth([])).toBe(1);
    expect(silvermanBandwidth([1e8, 1e8 + 1, 1e8 + 2])).toBeGreaterThan(0);
  });

  it("calculateWhiskers skips NaN instead of using it as a whisker", () => {
    const w = calculateWhiskers([1, 2, 3, 4, 100, Number.NaN], 1.75, 3.25);
    expect(w.lowerWhisker).toBe(1);
    expect(w.upperWhisker).toBe(4);
    expect(w.outliers).toEqual([100]);
  });
});

describe("buildContourGrid", () => {
  const z = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  it("rejects coordinates that are not strictly monotonic", () => {
    expect(() => buildContourGrid(tensor([0, 2, 1]), tensor([0, 1]), z)).toThrow(
      InvalidParameterError
    );
    expect(() => buildContourGrid(tensor([0, 1, 1]), tensor([0, 1]), z)).toThrow(/strictly/);
  });

  it("accepts strictly decreasing coordinates", () => {
    const g = buildContourGrid(tensor([2, 1, 0]), tensor([1, 0]), z);
    expect(Array.from(g.xCoords)).toEqual([2, 1, 0]);
  });

  it("uses a relative tolerance when checking meshgrids at large coordinates", () => {
    const base = 1e9;
    const X = tensor(
      [
        [base, base + 1, base + 2],
        [base + 1e-7, base + 1 + 1e-7, base + 2 + 1e-7],
      ],
      { dtype: "float64" }
    );
    const Y = tensor([
      [0, 0, 0],
      [1, 1, 1],
    ]);
    const g = buildContourGrid(X, Y, z);
    expect(g.xCoords[0]).toBe(base);
    expect(g.cols).toBe(3);
  });

  it("still rejects genuinely curvilinear grids", () => {
    const X = tensor([
      [0, 1, 2],
      [0.5, 1.5, 2.5],
    ]);
    const Y = tensor([
      [0, 0, 0],
      [1, 1, 1],
    ]);
    expect(() => buildContourGrid(X, Y, z)).toThrow(InvalidParameterError);
  });
});

describe("Violinplot", () => {
  it("gives a constant sample a visible shape instead of a zero-height violin", () => {
    const v = new Violinplot(0, new Float64Array([5, 5, 5]), {});
    const range = v.getDataRange();
    expect(range).not.toBeNull();
    expect(range!.ymax).toBeGreaterThan(range!.ymin);
    expect(range!.ymin).toBeLessThan(5);
    expect(range!.ymax).toBeGreaterThan(5);
    expect(v.q1).toBe(5);
    expect(v.median).toBe(5);
    // density peaks at the sample value
    const peak = v.kdeValues.indexOf(Math.max(...v.kdeValues));
    expect(v.kdePoints[peak]).toBeCloseTo(5, 1);
  });

  it("uses Silverman's density (matches scipy) on the padded grid", () => {
    const data = new Float64Array([1, 2, 2, 3, 7, 9]);
    const v = new Violinplot(0, data, {});
    const scipyAt = (x: number): number => {
      const h = 2.387119535268336;
      let s = 0;
      for (const d of data) s += Math.exp(-0.5 * ((x - d) / h) ** 2);
      return s / (data.length * h * Math.sqrt(2 * Math.PI));
    };
    for (const i of [0, 30, 60, 99]) {
      expect(v.kdeValues[i]).toBeCloseTo(scipyAt(v.kdePoints[i] ?? 0), 12);
    }
    // grid: data range [1, 9] padded by 10% of 8
    expect(v.kdePoints[0]).toBeCloseTo(0.2, 12);
    expect(v.kdePoints[99]).toBeCloseTo(9.8, 12);
  });

  it("ignores NaN values, does not mutate the input, and rejects all-NaN data", () => {
    const data = new Float64Array([3, Number.NaN, 1, 2]);
    const v = new Violinplot(0, data, {});
    expect(v.median).toBe(2);
    expect(Array.from(data).map((x) => (Number.isNaN(x) ? "nan" : x))).toEqual([3, "nan", 1, 2]);
    expect(() => new Violinplot(0, new Float64Array([Number.NaN]), {})).toThrow(
      InvalidParameterError
    );
  });

  it("raster rendering fills the body and draws an edge-colored outline", () => {
    const v = new Violinplot(5, new Float64Array([1, 2, 3, 4, 5]), {
      color: "#ff0000",
      edgecolor: "#0000ff",
    });
    const canvas = new RasterCanvas(120, 120);
    canvas.clearRGBA(255, 255, 255, 255);
    const fill = vi.spyOn(canvas, "fillPolygonRGBA");
    const line = vi.spyOn(canvas, "drawLineRGBA");
    v.drawRaster({
      canvas,
      transform: { xToPx: (x) => 10 + x * 20, yToPx: (y) => 110 - y * 20 },
    });
    expect(fill).toHaveBeenCalledTimes(1);
    expect(line.mock.calls.some((c) => c[4] === 0 && c[5] === 0 && c[6] === 255)).toBe(true);
    // a pixel inside the body, away from the quartile lines, is the fill color
    const px = Math.round(10 + 5 * 20);
    const py = Math.round(110 - 3.6 * 20);
    const i = (py * 120 + px) * 4;
    expect([canvas.data[i], canvas.data[i + 1], canvas.data[i + 2]]).toEqual([255, 0, 0]);
  });

  it("rejects data whose span overflows", () => {
    expect(() => new Violinplot(0, new Float64Array([-1.7e308, 1.7e308]), {})).toThrow(
      InvalidParameterError
    );
  });
});

describe("Waterfall2D", () => {
  it("rejects non-finite values and invalid bar widths", () => {
    expect(() => new Waterfall2D(["a", "b"], new Float64Array([1, Number.NaN]))).toThrow(
      InvalidParameterError
    );
    expect(() => new Waterfall2D(["a"], new Float64Array([1]), { barWidth: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => new Waterfall2D(["a"], new Float64Array([1]), { barWidth: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new Waterfall2D(["a"], new Float64Array([1, 2]))).toThrow(ShapeError);
  });

  it("copies its inputs", () => {
    const values = new Float64Array([1, 2]);
    const cats = ["a", "b"];
    const w = new Waterfall2D(cats, values);
    values[0] = 100;
    cats[0] = "z";
    expect(w.values[0]).toBe(1);
    expect(w.categories[0]).toBe("a");
  });

  it("data range covers the running total", () => {
    const w = new Waterfall2D(["a", "b", "c"], new Float64Array([10, -14, 3]));
    expect(w.getDataRange()).toEqual({ xmin: -0.5, xmax: 2.5, ymin: -4, ymax: 10 });
    expect(new Waterfall2D([], new Float64Array(0)).getDataRange()).toBeNull();
  });

  it("raster output uses the bar colors and draws connectors", () => {
    const w = new Waterfall2D(["a", "b", "c"], new Float64Array([10, -4, 3]));
    const canvas = new RasterCanvas(60, 50);
    canvas.clearRGBA(255, 255, 255, 255);
    const line = vi.spyOn(canvas, "drawLineRGBA");
    w.drawRaster({
      canvas,
      transform: { xToPx: (x) => 10 * x + 15, yToPx: (y) => 40 - 3 * y },
    });
    const at = (x: number, y: number) => {
      const i = (y * 60 + x) * 4;
      return [canvas.data[i], canvas.data[i + 1], canvas.data[i + 2]];
    };
    expect(at(15, 25)).toEqual([44, 160, 44]); // positive step
    expect(at(25, 15)).toEqual([214, 39, 40]); // negative step (10 -> 6)
    expect(at(35, 17)).toEqual([31, 119, 180]); // last bar uses totalColor
    expect(line).toHaveBeenCalled();
  });
});

describe("3D plots", () => {
  const scale = {
    xToPx: (v: number) => v * 1000,
    yToPx: (v: number) => v * 1000,
  };

  /** Screen displacement (right, up) of a unit move along each axis, read from a 2-point wireframe. */
  const axisVectors = (options: { elevation: number; azimuth: number }) => {
    const move = (dx: number, dy: number, dz: number) => {
      const w = new Wireframe3D(
        [new Float64Array([0, dx])],
        [new Float64Array([0, dy])],
        [new Float64Array([0, dz])],
        options
      );
      const pushed: string[] = [];
      w.drawSVG({ transform: scale, push: (e) => pushed.push(e) });
      const pts = /points="([^"]+)"/.exec(pushed[0] ?? "")?.[1]?.split(" ") ?? [];
      const [p0 = [0, 0], p1 = [0, 0]] = pts.map((p) => p.split(",").map(Number));
      return [(p1[0] ?? 0) - (p0[0] ?? 0), (p1[1] ?? 0) - (p0[1] ?? 0)];
    };
    return { x: move(1, 0, 0), y: move(0, 1, 0), z: move(0, 0, 1) };
  };

  const expectVector = (got: number[] | undefined, want: [number, number]) => {
    expect(Math.abs((got?.[0] ?? 1e9) - want[0])).toBeLessThan(0.11);
    expect(Math.abs((got?.[1] ?? 1e9) - want[1])).toBeLessThan(0.11);
  };

  it("projects like Matplotlib's orthographic camera (elevation 30, azimuth -60)", () => {
    // matplotlib mplot3d, view_init(30, -60), proj_type='ortho': screen displacement of a unit
    // move along x/y/z, with "up" positive, scaled by 1000 here.
    const v = axisVectors({ elevation: 30, azimuth: -60 });
    expectVector(v.x, [866.025, -250]);
    expectVector(v.y, [500, 433.013]);
    expectVector(v.z, [0, 866.025]);
  });

  it("matches Matplotlib for other view angles", () => {
    const a = axisVectors({ elevation: 20, azimuth: 45 });
    expectVector(a.x, [-707.107, -241.845]);
    expectVector(a.y, [707.107, -241.845]);
    expectVector(a.z, [0, 939.693]);
    const b = axisVectors({ elevation: 60, azimuth: 120 });
    expectVector(b.x, [-866.025, 433.013]);
    expectVector(b.y, [-500, -750]);
    expectVector(b.z, [0, 500]);
  });

  it("the default camera is Matplotlib's default (30, -60)", () => {
    expect(axisVectors({ elevation: 30, azimuth: -60 })).toEqual(
      (() => {
        const w = (dx: number, dy: number, dz: number) => {
          const g = new Wireframe3D(
            [new Float64Array([0, dx])],
            [new Float64Array([0, dy])],
            [new Float64Array([0, dz])]
          );
          const pushed: string[] = [];
          g.drawSVG({ transform: scale, push: (e) => pushed.push(e) });
          const pts = /points="([^"]+)"/.exec(pushed[0] ?? "")?.[1]?.split(" ") ?? [];
          const [p0 = [0, 0], p1 = [0, 0]] = pts.map((p) => p.split(",").map(Number));
          return [(p1[0] ?? 0) - (p0[0] ?? 0), (p1[1] ?? 0) - (p0[1] ?? 0)];
        };
        return { x: w(1, 0, 0), y: w(0, 1, 0), z: w(0, 0, 1) };
      })()
    );
  });

  it("draws higher z higher on screen (the old projection drew surfaces upside down)", () => {
    const s = new Scatter3D(
      new Float64Array([0, 0]),
      new Float64Array([0, 0]),
      new Float64Array([0, 1])
    );
    const pushed: string[] = [];
    s.drawSVG({
      transform: { xToPx: (v) => v * 100, yToPx: (v) => 100 - v * 100 },
      push: (e) => pushed.push(e),
    });
    const cy = pushed.map((e) => Number(/cy="(-?[\d.]+)"/.exec(e)?.[1]));
    // far-to-near order does not matter here: the point with z=1 has the smaller pixel row
    expect(Math.min(...cy)).toBeLessThan(Math.max(...cy));
    const range = s.getDataRange()!;
    expect(range.ymax).toBeGreaterThan(range.ymin);
  });

  it("scatter draws far points first and skips non-finite points", () => {
    const s = new Scatter3D(
      new Float64Array([0, 1, Number.NaN, 0.5]),
      new Float64Array([0, 1, 0, 0.5]),
      new Float64Array([0, 0, 0, 0]),
      { azimuth: 0, elevation: 0 }
    );
    // azimuth 0, elevation 0: the viewer is on +x, so depth grows with x
    const pushed: string[] = [];
    s.drawSVG({ transform: scale, push: (e) => pushed.push(e) });
    expect(pushed).toHaveLength(3);
    expect(pushed.join("")).not.toContain("NaN");
    // right = y: depth order is x ascending => y = 0, 0.5, 1 (normalized screen x -0.5, 0, 0.5)
    const cx = pushed.map((e) => Number(/cx="(-?[\d.]+)"/.exec(e)?.[1]));
    expect(cx).toEqual([-500, 0, 500]);
  });

  it("surface paints cells far to near and skips cells with non-finite corners", () => {
    const n = 3;
    const row = (f: (c: number) => number) => {
      const a = new Float64Array(n);
      for (let c = 0; c < n; c++) a[c] = f(c);
      return a;
    };
    const xg = [0, 1, 2].map(() => row((c) => c));
    const yg = [0, 1, 2].map((r) => row(() => r));
    const zg = [0, 1, 2].map(() => row(() => 0));
    const s = new Surface3D(xg, yg, zg, { alpha: 1 });
    const pushed: string[] = [];
    s.drawSVG({ transform: scale, push: (e) => pushed.push(e) });
    expect(pushed).toHaveLength(4);

    // Viewer at azimuth -60 sits towards +x, -y, so the nearest cell is (row 0, col 1)
    // and the farthest is (row 1, col 0). Cell corner 0 is at normalized (x, y) =
    // (-0.5 + c/2, -0.5 + r/2); screen x = -x sin(az) + y cos(az).
    const az = (-60 * Math.PI) / 180;
    const screenX = (r: number, c: number) =>
      (-(-0.5 + c / 2) * Math.sin(az) + (-0.5 + r / 2) * Math.cos(az)) * 1000;
    const firstX = Number(/M(-?[\d.]+),/.exec(pushed[0] ?? "")?.[1]);
    const lastX = Number(/M(-?[\d.]+),/.exec(pushed[3] ?? "")?.[1]);
    expect(firstX).toBeCloseTo(screenX(1, 0), 0);
    expect(lastX).toBeCloseTo(screenX(0, 1), 0);

    const holed = zg.map((r) => Float64Array.from(r));
    (holed[0] as Float64Array)[0] = Number.NaN;
    const s2 = new Surface3D(xg, yg, holed);
    const p2: string[] = [];
    s2.drawSVG({ transform: scale, push: (e) => p2.push(e) });
    expect(p2).toHaveLength(3);
    expect(p2.join("")).not.toContain("NaN");
  });

  it("surface raster fills polygons before outlining them", () => {
    const g = [new Float64Array([0, 1]), new Float64Array([0, 1])];
    const gy = [new Float64Array([0, 0]), new Float64Array([1, 1])];
    const gz = [new Float64Array([0, 0.5]), new Float64Array([0.5, 1])];
    const s = new Surface3D(g, gy, gz);
    const canvas = new RasterCanvas(100, 100);
    const fill = vi.spyOn(canvas, "fillPolygonRGBA");
    const line = vi.spyOn(canvas, "drawLineRGBA");
    s.drawRaster({
      canvas,
      transform: { xToPx: (v) => 50 + v * 60, yToPx: (v) => 50 - v * 60 },
    });
    expect(fill).toHaveBeenCalledTimes(1);
    expect(line).toHaveBeenCalledTimes(4);
  });

  it("validates grid shapes, angles, alpha, linewidth and size", () => {
    const g = [new Float64Array([0, 1]), new Float64Array([0, 1])];
    const ragged = [new Float64Array([0, 1]), new Float64Array([0])];
    expect(() => new Surface3D(g, g, ragged)).toThrow(ShapeError);
    expect(() => new Wireframe3D(g, ragged, g)).toThrow(ShapeError);
    expect(() => new Wireframe3D([], [], [])).toThrow(ShapeError);
    expect(() => new Surface3D(g, g, g, { elevation: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new Surface3D(g, g, g, { azimuth: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    expect(() => new Surface3D(g, g, g, { alpha: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new Wireframe3D(g, g, g, { linewidth: 0 })).toThrow(InvalidParameterError);
    expect(
      () => new Scatter3D(new Float64Array(1), new Float64Array(1), new Float64Array(2))
    ).toThrow(ShapeError);
    expect(
      () =>
        new Scatter3D(new Float64Array(1), new Float64Array(1), new Float64Array(1), { size: -1 })
    ).toThrow(InvalidParameterError);
  });

  it("copies grids on construction", () => {
    const x = [new Float64Array([0, 1]), new Float64Array([0, 1])];
    const y = [new Float64Array([0, 0]), new Float64Array([1, 1])];
    const z = [new Float64Array([0, 0]), new Float64Array([0, 1])];
    const s = new Surface3D(x, y, z);
    const before = s.getDataRange();
    z[1]![1] = 1000;
    expect(s.getDataRange()).toEqual(before);
  });

  it("returns a null data range when no point is finite", () => {
    const nan = [new Float64Array([Number.NaN, Number.NaN])];
    expect(new Wireframe3D(nan, nan, nan).getDataRange()).toBeNull();
    const e = new Float64Array(0);
    expect(new Scatter3D(e, e, e).getDataRange()).toBeNull();
  });

  it("wireframe breaks lines at non-finite points", () => {
    const x = [new Float64Array([0, 1, 2]), new Float64Array([0, 1, 2])];
    const y = [new Float64Array([0, 0, 0]), new Float64Array([1, 1, 1])];
    const z = [new Float64Array([0, Number.NaN, 0]), new Float64Array([0, 1, 0])];
    const w = new Wireframe3D(x, y, z);
    const pushed: string[] = [];
    w.drawSVG({ transform: scale, push: (e) => pushed.push(e) });
    // row 0 is cut into two single points (nothing), row 1 is one line, columns 0 and 2 lines
    expect(pushed.filter((p) => p.startsWith("<polyline"))).toHaveLength(3);
    expect(pushed.join("")).not.toContain("NaN");
  });

  it("applies alpha to scatter and wireframe only when requested", () => {
    const sc = new Scatter3D(
      new Float64Array([0, 1]),
      new Float64Array([0, 1]),
      new Float64Array([0, 1]),
      {
        alpha: 0.5,
      }
    );
    const out: string[] = [];
    sc.drawSVG({ transform: scale, push: (e) => out.push(e) });
    expect(out.every((e) => e.includes('fill-opacity="0.5"'))).toBe(true);
    const opaque = new Scatter3D(
      new Float64Array([0]),
      new Float64Array([0]),
      new Float64Array([0])
    );
    const out2: string[] = [];
    opaque.drawSVG({ transform: scale, push: (e) => out2.push(e) });
    expect(out2[0]).not.toContain("opacity");
  });

  it("legend entries carry line width and marker size", () => {
    const g = [new Float64Array([0, 1]), new Float64Array([0, 1])];
    const w = new Wireframe3D(g, g, g, { label: " mesh ", linewidth: 3 });
    expect(w.getLegendEntries()).toEqual([
      { label: "mesh", color: "#333333", shape: "line", lineWidth: 3 },
    ]);
    const s = new Scatter3D(new Float64Array([0]), new Float64Array([0]), new Float64Array([0]), {
      label: "pts",
      size: 7,
    });
    expect(s.getLegendEntries()?.[0]?.markerSize).toBe(7);
    expect(new Surface3D(g, g, g).getLegendEntries()).toBeNull();
  });
});

describe("pngEncodeRGBA", () => {
  const rgba = (w: number, h: number) => {
    const px = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        px[i] = (x * 3) & 255;
        px[i + 1] = (y * 4) & 255;
        px[i + 2] = (x * y) & 255;
        px[i + 3] = 255 - x;
      }
    }
    return px;
  };

  const chunks = (png: Uint8Array) => {
    const out: { type: string; data: Uint8Array; crc: number }[] = [];
    const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
    let pos = 8;
    while (pos < png.length) {
      const len = view.getUint32(pos);
      const type = String.fromCharCode(...png.subarray(pos + 4, pos + 8));
      out.push({
        type,
        data: png.subarray(pos + 8, pos + 8 + len),
        crc: view.getUint32(pos + 8 + len),
      });
      pos += 12 + len;
    }
    return out;
  };

  const crcOf = (type: string, data: Uint8Array): number => {
    let crc = 0xffffffff;
    const feed = (b: number) => {
      crc ^= b;
      for (let k = 0; k < 8; k++) crc = (crc & 1) !== 0 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
    };
    for (const ch of type) feed(ch.charCodeAt(0));
    for (const b of data) feed(b);
    return (crc ^ 0xffffffff) >>> 0;
  };

  const decode = (png: Uint8Array, w: number, h: number) => {
    const idat = chunks(png).find((c) => c.type === "IDAT");
    const raw = inflateSync(idat!.data);
    const stride = w * 4;
    expect(raw.length).toBe(h * (stride + 1));
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
      expect(raw[y * (stride + 1)]).toBe(0);
      out.set(raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1)), y * stride);
    }
    return out;
  };

  it("writes valid chunk CRCs and lossless pixel data", async () => {
    const w = 70;
    const h = 60;
    const px = rgba(w, h);
    const png = await pngEncodeRGBA(w, h, px);
    expect(Array.from(png.subarray(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    const cs = chunks(png);
    expect(cs.map((c) => c.type)).toEqual(["IHDR", "IDAT", "IEND"]);
    for (const c of cs) expect(c.crc).toBe(crcOf(c.type, c.data));
    expect(Array.from(decode(png, w, h))).toEqual(Array.from(px));
  });

  it("the stored-block fallback (no Node zlib) produces a valid, lossless stream", async () => {
    const w = 300;
    const h = 100; // > 65535 bytes of raw data, so several stored blocks
    const px = rgba(w, h);
    vi.stubGlobal("process", undefined);
    const png = await pngEncodeRGBA(w, h, px);
    vi.unstubAllGlobals();
    for (const c of chunks(png)) expect(c.crc).toBe(crcOf(c.type, c.data));
    expect(Array.from(decode(png, w, h))).toEqual(Array.from(px));
  });

  it("rejects dimensions above the PNG limit and mismatched buffers", async () => {
    await expect(pngEncodeRGBA(2 ** 31, 1, new Uint8ClampedArray(4))).rejects.toThrow(
      InvalidParameterError
    );
    await expect(pngEncodeRGBA(2, 2, new Uint8ClampedArray(4))).rejects.toThrow(/RGBA buffer/);
    await expect(pngEncodeRGBA(1.5, 1, new Uint8ClampedArray(4))).rejects.toThrow(
      InvalidParameterError
    );
  });
});

describe("svgToPdf", () => {
  const content = (pdf: Uint8Array): string => {
    const text = decodeLatin1(pdf);
    const start = text.indexOf("stream\n") + "stream\n".length;
    return text.slice(start, text.indexOf("\nendstream"));
  };
  const svgDoc = (body: string, attrs = 'width="200" height="200" viewBox="0 0 200 200"') =>
    `<svg xmlns="http://www.w3.org/2000/svg" ${attrs}>${body}</svg>`;

  it("keeps rect fill colors (previously every rect was filled white)", () => {
    const pdf = svgToPdf(
      svgDoc('<rect x="10" y="20" width="30" height="40" fill="#112233" />'),
      200,
      200
    );
    const c = content(pdf);
    expect(c).toContain("0.0667 0.1333 0.2 rg");
    expect(c).toContain("10 20 30 40 re");
    expect(c).toContain("1 0 0 -1 0 200 cm");
  });

  it("honors fill, stroke, stroke-width, in any attribute order", () => {
    const c = content(
      svgToPdf(
        svgDoc(
          '<rect stroke-width="3" stroke="#ff0000" fill="#00ff00" height="5" width="6" y="2" x="1"/>'
        ),
        200,
        200
      )
    );
    expect(c).toContain("0 1 0 rg");
    expect(c).toContain("1 0 0 RG");
    expect(c).toContain("3 w");
    expect(c).toContain("1 2 6 5 re");
    expect(c).toMatch(/\nB\n/);
  });

  it("applies SVG defaults: black fill, no stroke", () => {
    const c = content(svgToPdf(svgDoc('<rect x="0" y="0" width="5" height="5"/>'), 200, 200));
    expect(c).toContain("0 0 0 rg");
    expect(c).toMatch(/\nf\n/);
    const line = content(svgToPdf(svgDoc('<line x1="0" y1="0" x2="5" y2="5"/>'), 200, 200));
    expect(line).not.toContain(" l\nS");
  });

  it("draws circles with correct Bezier control points and the circle's fill", () => {
    const c = content(svgToPdf(svgDoc('<circle cx="40" cy="40" r="5" fill="#ff0000"/>'), 200, 200));
    expect(c).toContain("1 0 0 rg");
    const k = 0.5522847498307936 * 5;
    // third quadrant arc: from (35, 40) to (40, 35) with controls (35, 40 - k) and (40 - k, 35)
    const arc = `${(40 - 5).toFixed(0)} ${(40 - k).toFixed(4).replace(/0+$/, "")} ${(40 - k)
      .toFixed(4)
      .replace(/0+$/, "")} ${(35).toFixed(0)} 40 35 c`;
    expect(c).toContain(arc);
  });

  it("renders polygons, polylines and paths in document order", () => {
    const c = content(
      svgToPdf(
        svgDoc(
          '<polygon points="0,0 10,0 10,10" fill="#ff0000"/>' +
            '<polyline points="0,0 5,5 9,1" fill="none" stroke="#0000ff"/>' +
            '<path d="M0 0 L10 0 L10 10 Z" fill="#00ff00"/>'
        ),
        200,
        200
      )
    );
    const iPoly = c.indexOf("1 0 0 rg");
    const iLine = c.indexOf("0 0 1 RG");
    const iPath = c.indexOf("0 1 0 rg");
    expect(iPoly).toBeGreaterThan(-1);
    expect(iLine).toBeGreaterThan(iPoly);
    expect(iPath).toBeGreaterThan(iLine);
  });

  it("parses relative, horizontal/vertical, smooth and arc path commands", () => {
    const rel = content(svgToPdf(svgDoc('<path d="m10 10 l5 0 h5 v5 z" fill="red"/>'), 200, 200));
    expect(rel).toContain("10 10 m");
    expect(rel).toContain("15 10 l");
    expect(rel).toContain("20 10 l");
    expect(rel).toContain("20 15 l");
    expect(rel).toContain("\nh\n");

    const curve = content(
      svgToPdf(
        svgDoc('<path d="M0 0 C 1 1 2 1 3 0 S 5 -1 6 0 Q 7 2 8 0" stroke="black" fill="none"/>'),
        200,
        200
      )
    );
    expect(curve).toContain("1 1 2 1 3 0 c");
    expect(curve).toContain("4 -1 5 -1 6 0 c"); // reflected first control point of S
    expect(curve.match(/ c\n/g)?.length).toBe(3);

    // A half circle of radius 10 from (0,0) to (20,0): ends exactly at the end point
    const arc = content(
      svgToPdf(svgDoc('<path d="M0 0 A10 10 0 0 1 20 0" stroke="black" fill="none"/>'), 200, 200)
    );
    const curves = arc.split("\n").filter((l) => l.endsWith(" c"));
    expect(curves).toHaveLength(2);
    expect(curves[1]?.endsWith("20 0 c")).toBe(true);
    expect(curves[0]).toContain(" 10 -10 c");
  });

  it("parses compact path data with packed arc flags and exponents", () => {
    const c = content(
      svgToPdf(svgDoc('<path d="M0,0a5,5 0 1110,0l1e1-5" stroke="black" fill="none"/>'), 200, 200)
    );
    expect(c).toContain("10 0 c");
    expect(c).toContain("20 -5 l");
  });

  it("decodes entities, escapes string delimiters and encodes WinAnsi characters", () => {
    const c = content(
      svgToPdf(
        svgDoc('<text x="5" y="15" font-size="12" fill="#000000">A &amp; (b) \\ é € ☃</text>'),
        200,
        200
      )
    );
    expect(c).toContain("(A & \\(b\\) \\\\ \\351 \\200 ?) Tj");
    expect(c).toContain("/F1 12 Tf");
    expect(c).toContain("1 0 0 -1 5 15 Tm");
  });

  it("positions text by anchor and baseline using Helvetica metrics", () => {
    // "Hi" = 722 + 222 = 944 units -> 11.328 pt at 12 pt
    const mid = content(
      svgToPdf(
        svgDoc('<text x="100" y="50" font-size="12" text-anchor="middle">Hi</text>'),
        200,
        200
      )
    );
    expect(mid).toContain("1 0 0 -1 94.336 50 Tm");
    const end = content(
      svgToPdf(svgDoc('<text x="100" y="50" font-size="12" text-anchor="end">Hi</text>'), 200, 200)
    );
    expect(end).toContain("1 0 0 -1 88.672 50 Tm");
    const hanging = content(
      svgToPdf(
        svgDoc('<text x="0" y="10" font-size="10" dominant-baseline="hanging">H</text>'),
        200,
        200
      )
    );
    expect(hanging).toContain("1 0 0 -1 0 17.2 Tm");
    const bold = content(
      svgToPdf(
        svgDoc(
          '<text x="100" y="50" font-size="10" font-weight="bold" text-anchor="end">Hi</text>'
        ),
        200,
        200
      )
    );
    expect(bold).toContain("/F2 10 Tf");
    // Helvetica-Bold: H 722 + i 278 = 1000 units -> 10 pt
    expect(bold).toContain("1 0 0 -1 90 50 Tm");
  });

  it("applies transform attributes such as the rotated y label", () => {
    const c = content(
      svgToPdf(
        svgDoc(
          '<text x="10" y="100" font-size="12" text-anchor="middle" transform="rotate(-90 10 100)">y</text>'
        ),
        200,
        200
      )
    );
    // rotate(-90 10 100): cos = 0, sin = -1 -> matrix 0 -1 1 0 e f with e = 10 - 0 + (-1)(100)... computed
    // as cx - cos*cx + sin*cy = 10 - 0 - 100 = -90 and cy - sin*cx - cos*cy = 100 + 10 - 0 = 110
    expect(c).toContain("0 -1 1 0 -90 110 cm");
  });

  it("ignores whitespace-only text and leaves no stray operators", () => {
    const c = content(svgToPdf(svgDoc('<text x="1" y="2">   </text>'), 100, 100));
    expect(c).not.toContain("BT");
  });

  it("supports dashes, caps, joins, opacity and fill-opacity through graphics states", () => {
    const pdf = svgToPdf(
      svgDoc(
        '<line x1="0" y1="0" x2="9" y2="9" stroke="#000" stroke-dasharray="4,3" stroke-linecap="round" stroke-linejoin="bevel"/>' +
          '<rect x="0" y="0" width="5" height="5" fill="#f00" fill-opacity="0.25" opacity="0.5"/>'
      ),
      200,
      200
    );
    const c = content(pdf);
    expect(c).toContain("[4 3] 0 d");
    expect(c).toContain("1 J");
    expect(c).toContain("2 j");
    expect(c).toContain("/GS1 gs");
    expect(decodeLatin1(pdf)).toContain("/GS1 << /Type /ExtGState /ca 0.125 /CA 1 >>");
  });

  it("uses inherited group styles, group opacity and the style attribute", () => {
    const pdf = svgToPdf(
      svgDoc(
        '<g fill="#00f" opacity="0.5"><rect x="0" y="0" width="4" height="4"/>' +
          '<rect x="5" y="0" width="4" height="4" style="fill:#0f0"/></g>'
      ),
      200,
      200
    );
    const c = content(pdf);
    expect(c).toContain("0 0 1 rg");
    expect(c).toContain("0 1 0 rg");
    expect(decodeLatin1(pdf)).toContain("/ca 0.5 /CA 1");
  });

  it("applies clip paths from groups and keeps q/Q balanced", () => {
    const svg = svgDoc(
      '<clipPath id="c1"><path d="M0 0 L50 0 L50 50 L0 50 Z"/></clipPath>' +
        '<g clip-path="url(#c1)"><rect x="0" y="0" width="100" height="100" fill="#f00"/></g>' +
        '<rect x="1" y="1" width="2" height="2" fill="#0f0"/>'
    );
    const c = content(svgToPdf(svg, 200, 200));
    expect(c).toContain("W n");
    const lines = c.split("\n");
    expect(lines.filter((l) => l === "q")).toHaveLength(lines.filter((l) => l === "Q").length);
    // the clipped rect is drawn after the clip operator, the unclipped one after the group closed
    expect(c.indexOf("W n")).toBeLessThan(c.indexOf("1 0 0 rg"));
  });

  it("skips display:none content, gradients and unknown elements", () => {
    const c = content(
      svgToPdf(
        svgDoc(
          '<g display="none"><rect x="0" y="0" width="9" height="9" fill="#f00"/></g>' +
            '<rect x="0" y="0" width="9" height="9" fill="url(#grad)"/>' +
            '<image href="x.png" x="0" y="0" width="5" height="5"/>'
        ),
        200,
        200
      )
    );
    expect(c).not.toContain("re\nf");
    expect(c).not.toContain("1 0 0 rg");
  });

  it("does not draw the contents of <defs> but still resolves clip paths defined there", () => {
    const c = content(
      svgToPdf(
        svgDoc(
          '<defs><rect x="0" y="0" width="9" height="9" fill="#f00"/>' +
            '<clipPath id="k"><path d="M0 0 L5 0 L5 5 Z"/></clipPath></defs>' +
            '<g clip-path="url(#k)"><rect x="1" y="1" width="2" height="2" fill="#0f0"/></g>'
        ),
        200,
        200
      )
    );
    expect(c).not.toContain("1 0 0 rg");
    expect(c).toContain("0 1 0 rg");
    expect(c).toContain("W n");
  });

  it("scales the viewBox to the page while keeping the aspect ratio", () => {
    const c = content(
      svgToPdf(svgDoc('<rect x="0" y="0" width="1" height="1"/>', 'viewBox="0 0 100 50"'), 200, 100)
    );
    expect(c).toContain("2 0 0 -2 0 100 cm");
    // 100x50 into 200x200 keeps the 2x scale and centers vertically (offset 50)
    const centered = content(
      svgToPdf(svgDoc('<rect x="0" y="0" width="1" height="1"/>', 'viewBox="0 0 100 50"'), 200, 200)
    );
    expect(centered).toContain("2 0 0 -2 0 150 cm");
  });

  it("produces a structurally valid PDF with a correct xref table and stream length", () => {
    const pdf = svgToPdf(
      svgDoc('<rect x="1" y="2" width="3" height="4" fill="#123456"/>'),
      300,
      150
    );
    const text = decodeLatin1(pdf);
    expect(text.startsWith("%PDF-1.4\n")).toBe(true);
    expect(text.trimEnd().endsWith("%%EOF")).toBe(true);
    expect(text).toContain("/MediaBox [0 0 300 150]");

    const startxref = Number(/startxref\n(\d+)\n%%EOF/.exec(text)?.[1]);
    expect(text.slice(startxref, startxref + 4)).toBe("xref");
    const entries = text
      .slice(startxref)
      .split("\n")
      .filter((l) => /^\d{10} \d{5} [nf] $/.test(l));
    expect(entries).toHaveLength(7);
    entries.slice(1).forEach((entry, i) => {
      const offset = Number(entry.slice(0, 10));
      expect(text.slice(offset, offset + `${i + 1} 0 obj`.length)).toBe(`${i + 1} 0 obj`);
    });

    const declared = Number(/\/Length (\d+)/.exec(text)?.[1]);
    expect(declared).toBe(content(pdf).length);
    // fonts referenced by the page exist
    expect(text).toContain("/F1 5 0 R");
    expect(text).toMatch(/5 0 obj\n<< \/Type \/Font \/Subtype \/Type1 \/BaseFont \/Helvetica /);
    expect(text).toContain("/BaseFont /Helvetica-Bold");
  });

  it("only emits ASCII in the content stream and never NaN or exponent numbers", () => {
    const c = content(
      svgToPdf(
        svgDoc('<rect x="1e30" y="NaN" width="4" height="4"/><circle cx="1e-9" cy="2" r="3"/>'),
        200,
        200
      )
    );
    const printable = Array.from(c).every((ch) => {
      const code = ch.charCodeAt(0);
      return code === 9 || code === 10 || code === 13 || (code >= 0x20 && code <= 0x7e);
    });
    expect(printable).toBe(true);
    expect(c).not.toMatch(/NaN|Infinity|e[+-]\d/);
  });

  it("rejects invalid page sizes", () => {
    expect(() => svgToPdf("<svg/>", 0, 100)).toThrow(InvalidParameterError);
    expect(() => svgToPdf("<svg/>", 100, -1)).toThrow(InvalidParameterError);
    expect(() => svgToPdf("<svg/>", Number.NaN, 100)).toThrow(InvalidParameterError);
  });

  it("tolerates malformed markup without throwing", () => {
    expect(() => svgToPdf("<svg><g><rect x='1'", 100, 100)).not.toThrow();
    expect(() => svgToPdf("", 100, 100)).not.toThrow();
    const unbalanced = content(
      svgToPdf(svgDoc('<g transform="translate(5 5)"><rect width="1" height="1"/>'), 50, 50)
    );
    const lines = unbalanced.split("\n");
    expect(lines.filter((l) => l === "q")).toHaveLength(lines.filter((l) => l === "Q").length);
  });
});
