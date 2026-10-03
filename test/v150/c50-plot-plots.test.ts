/**
 * Regression tests for the plot drawables in src/plot/plots (v1.5.0 review).
 *
 * Reference values come from NumPy 2.4 (`numpy.histogram`), SciPy 1.17
 * (`scipy.cluster.hierarchy.dendrogram`) and matplotlib's `cbook.boxplot_stats`.
 */
import { describe, expect, it, vi } from "vitest";
import { InvalidParameterError, ShapeError } from "../../src/core";
import { RasterCanvas } from "../../src/plot/canvas/RasterCanvas";
import { Boxplot } from "../../src/plot/plots/Boxplot";
import { Contour2D } from "../../src/plot/plots/Contour2D";
import { ContourF2D } from "../../src/plot/plots/ContourF2D";
import { Dendrogram2D, type LinkageRow } from "../../src/plot/plots/Dendrogram2D";
import { Heatmap2D } from "../../src/plot/plots/Heatmap2D";
import { Histogram } from "../../src/plot/plots/Histogram";
import { Line2D } from "../../src/plot/plots/Line2D";
import { Pie } from "../../src/plot/plots/Pie";
import { Polar2D } from "../../src/plot/plots/Polar2D";
import { Quiver2D } from "../../src/plot/plots/Quiver2D";
import { Radar2D } from "../../src/plot/plots/Radar2D";
import { Stem2D } from "../../src/plot/plots/Stem2D";
import { Strip2D } from "../../src/plot/plots/Strip2D";
import type { DataTransform, Drawable } from "../../src/plot/types";
import type { ContourGrid } from "../../src/plot/utils/contours";

const transform: DataTransform = {
  xToPx: (x) => 100 + x * 40,
  yToPx: (y) => 100 - y * 40,
};

function svgOf(d: Drawable): string[] {
  const out: string[] = [];
  d.drawSVG({ transform, push: (e) => out.push(e) });
  return out;
}

function rasterOf(d: Drawable): RasterCanvas {
  const canvas = new RasterCanvas(200, 200);
  d.drawRaster({ transform, canvas });
  return canvas;
}

function grid(rows: number, cols: number, values: number[]): ContourGrid {
  const data = new Float64Array(values);
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return {
    rows,
    cols,
    data,
    xCoords: Float64Array.from({ length: cols }, (_, j) => j),
    yCoords: Float64Array.from({ length: rows }, (_, i) => i),
    dataMin: min,
    dataMax: max,
  };
}

describe("Histogram", () => {
  it("assigns values to the same bins as numpy.histogram despite rounding", () => {
    // d = round(0.1 * k, 10) for k in 0..m-1; numpy.histogram(d, bins=n)[0]
    const cases: Array<[number, number, number[]]> = [
      [9, 22, [3, 2, 2, 3, 2, 2, 3, 2, 3]],
      [14, 23, [2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2]],
      [18, 19, [1, 1, 2, 0, 1, 2, 1, 0, 1, 1, 1, 2, 0, 2, 0, 1, 2, 1]],
    ];
    for (const [n, m, expected] of cases) {
      const data = Float64Array.from({ length: m }, (_, k) => Math.round(0.1 * k * 1e10) / 1e10);
      const h = new Histogram(data, n, {});
      expect(Array.from(h.counts)).toEqual(expected);
    }
  });

  it("bins a constant sample over [v - 0.5, v + 0.5] like numpy", () => {
    // numpy.histogram([5, 5, 5], bins=4) -> counts [0, 0, 3, 0], edges 4.5 .. 5.5
    const h = new Histogram(new Float64Array([5, 5, 5]), 4, {});
    expect(Array.from(h.counts)).toEqual([0, 0, 3, 0]);
    expect(Array.from(h.bins)).toEqual([4.5, 4.75, 5, 5.25]);
    expect(h.binWidth).toBeCloseTo(0.25, 15);
    expect(h.getDataRange()).toEqual({ xmin: 4.5, xmax: 5.5, ymin: 0, ymax: 3 });

    // bins=3 -> counts [0, 3, 0]
    const h3 = new Histogram(new Float64Array([5, 5, 5]), 3, {});
    expect(Array.from(h3.counts)).toEqual([0, 3, 0]);
    expect(h3.binWidth).toBeCloseTo(1 / 3, 15);
  });

  it("rejects a data range too small to split into bins", () => {
    // span = 5e-324, so span / 4 underflows to 0
    expect(() => new Histogram(new Float64Array([0, 5e-324]), 4, {})).toThrow(
      InvalidParameterError
    );
  });

  it("rejects a data range that overflows", () => {
    expect(() => new Histogram(new Float64Array([-1.7e308, 1.7e308]), 4, {})).toThrow(
      InvalidParameterError
    );
  });
});

describe("Boxplot", () => {
  it("ends a whisker at the quartile when no in-fence value lies beyond it", () => {
    // matplotlib.cbook.boxplot_stats([0, 0, 100 x6]) -> q1 75, med 100, q3 100,
    // whislo 75, whishi 100, fliers [0, 0]
    const b = new Boxplot(1, new Float64Array([0, 0, 100, 100, 100, 100, 100, 100]), {});
    expect(b.q1).toBe(75);
    expect(b.median).toBe(100);
    expect(b.q3).toBe(100);
    expect(b.whiskerLow).toBe(75);
    expect(b.whiskerHigh).toBe(100);
    expect(Array.from(b.outliers)).toEqual([0, 0]);
  });

  it("sorts numerically and ignores non-finite values", () => {
    const b = new Boxplot(2, new Float64Array([10, NaN, 2, Infinity, 1, 100, 3]), {});
    // finite sample [1, 2, 3, 10, 100]: q1 = 2, median = 3, q3 = 10
    expect([b.q1, b.median, b.q3]).toEqual([2, 3, 10]);
    expect(b.whiskerLow).toBe(1);
    expect(b.whiskerHigh).toBe(10);
    expect(Array.from(b.outliers)).toEqual([100]);
  });

  it("rejects a non-finite position", () => {
    expect(() => new Boxplot(Number.NaN, new Float64Array([1, 2, 3]), {})).toThrow(
      InvalidParameterError
    );
  });
});

describe("Dendrogram2D", () => {
  const linkage: LinkageRow[] = [
    [0, 2, 1, 2],
    [1, 3, 1.5, 2],
    [4, 5, 3, 4],
  ];

  it("orders leaves like scipy.cluster.hierarchy.dendrogram so U shapes do not cross", () => {
    // scipy: leaves == [0, 2, 1, 3]
    const d = new Dendrogram2D(linkage, 4, {});
    expect(d.leaves).toEqual([0, 2, 1, 3]);
    // Leaf 2 sits right of leaf 0: the first merge joins x = 0 and x = 1 at height 1.
    const lines = svgOf(d);
    expect(lines).toHaveLength(9);
    expect(lines[1]).toContain('x1="100.00" y1="60.00" x2="140.00" y2="60.00"');
    // Root bar joins the midpoints 0.5 and 2.5 at height 3.
    expect(lines[7]).toContain('x1="120.00" y1="-20.00" x2="200.00" y2="-20.00"');
  });

  it("walks the left child first, as scipy does", () => {
    const swapped: LinkageRow[] = [
      [1, 3, 0.2, 2],
      [0, 2, 0.3, 2],
      [5, 4, 0.5, 4],
    ];
    expect(new Dendrogram2D(swapped, 4, {}).leaves).toEqual([0, 2, 1, 3]);
  });

  it("scales the y range to the merge distances", () => {
    const d = new Dendrogram2D(
      [
        [0, 1, 0.2, 2],
        [2, 3, 0.5, 2],
      ],
      3,
      {}
    );
    const range = d.getDataRange();
    expect(range?.ymin).toBe(0);
    expect(range?.ymax).toBeCloseTo(0.5 * 1.05, 12);
    expect(range?.xmin).toBe(-0.5);
    expect(range?.xmax).toBe(2.5);
  });

  it("rejects malformed linkage matrices", () => {
    expect(() => new Dendrogram2D(linkage, 0, {})).toThrow(InvalidParameterError);
    expect(() => new Dendrogram2D(linkage, 2.5, {})).toThrow(InvalidParameterError);
    // cluster 4 does not exist yet in row 0
    expect(() => new Dendrogram2D([[0, 4, 1, 2]], 4, {})).toThrow(/ids must be integers/);
    // cluster 0 merged twice
    expect(
      () =>
        new Dendrogram2D(
          [
            [0, 1, 1, 2],
            [0, 2, 2, 3],
          ],
          3,
          {}
        )
    ).toThrow(/more than once/);
    expect(() => new Dendrogram2D([[1, 1, 1, 2]], 2, {})).toThrow(/with itself/);
    expect(() => new Dendrogram2D([[0, 1, Number.NaN, 2]], 2, {})).toThrow(/non-finite distance/);
  });

  it("accepts a single leaf and an incomplete forest", () => {
    expect(new Dendrogram2D([], 1, {}).leaves).toEqual([0]);
    const forest = new Dendrogram2D([[0, 1, 1, 2]], 3, {});
    expect(forest.leaves).toEqual([0, 1, 2]);
    expect(svgOf(forest)).toHaveLength(3);
  });
});

describe("Pie", () => {
  it("draws a single full slice as a circle (a zero-length SVG arc draws nothing)", () => {
    const pie = new Pie(0.5, 0.5, 0.35, new Float64Array([3]), ["only"], {});
    expect(pie.angles).toEqual([0, 2 * Math.PI]);
    const out = svgOf(pie);
    expect(out.some((e) => e.startsWith("<circle"))).toBe(true);
    expect(out.some((e) => e.startsWith("<path"))).toBe(false);
  });

  it("skips zero-valued slices and ends exactly at 2*pi", () => {
    const pie = new Pie(0, 0, 1, new Float64Array([0, 1, 2, 0, 4]), undefined, {});
    expect(pie.angles).toHaveLength(6);
    expect(pie.angles[5]).toBe(2 * Math.PI);
    expect(pie.angles[1]).toBe(0);
    expect(pie.angles[2]).toBeCloseTo((1 / 7) * 2 * Math.PI, 14);
    const paths = svgOf(pie).filter((e) => e.startsWith("<path"));
    expect(paths).toHaveLength(3);
  });

  it("fills the whole disc for a one-slice raster pie", () => {
    const pie = new Pie(0, 0, 1, new Float64Array([1, 0]), undefined, { colors: ["#ff0000"] });
    const canvas = rasterOf(pie);
    const idx = (100 * 200 + 120) * 4; // 20 px right of the center, inside the radius of 40 px
    expect(canvas.data[idx]).toBe(255);
    expect(canvas.data[idx + 3]).toBe(255);
  });

  it("does not alias the caller's label array", () => {
    const labels = ["a", "b"];
    const pie = new Pie(0, 0, 1, new Float64Array([1, 1]), labels, {});
    labels[0] = "changed";
    expect(pie.labels[0]).toBe("a");
  });
});

describe("Line2D", () => {
  it("breaks the SVG polyline at non-finite samples, like the raster output", () => {
    const line = new Line2D(
      new Float64Array([0, 1, 2, 3, 4]),
      new Float64Array([0, 1, Number.NaN, 3, 4]),
      {}
    );
    const out = svgOf(line);
    expect(out).toHaveLength(2);
    expect(out[0]).toContain('points="100.00,100.00 140.00,60.00"');
    expect(out[1]).toContain('points="220.00,-20.00 260.00,-60.00"');
  });

  it("keeps a single polyline for finite data", () => {
    const line = new Line2D(new Float64Array([0, 1, 2]), new Float64Array([0, 1, 2]), {});
    expect(svgOf(line)).toHaveLength(1);
  });

  it("emits nothing when no sample is finite", () => {
    const line = new Line2D(new Float64Array([Number.NaN]), new Float64Array([1]), {});
    expect(svgOf(line)).toHaveLength(0);
  });
});

describe("Polar2D", () => {
  it("validates linewidth", () => {
    expect(
      () => new Polar2D(new Float64Array([0]), new Float64Array([1]), { linewidth: 0 })
    ).toThrow(InvalidParameterError);
    expect(
      () => new Polar2D(new Float64Array([0]), new Float64Array([1]), { linewidth: Number.NaN })
    ).toThrow(InvalidParameterError);
  });

  it("draws the polar grid and the filled area in raster output", () => {
    const theta = Float64Array.from({ length: 4 }, (_, i) => (i * Math.PI) / 2);
    const p = new Polar2D(theta, new Float64Array([1, 1, 1, 1]), { fill: true, color: "#ff0000" });
    const canvas = new RasterCanvas(200, 200);
    const polygon = vi.spyOn(canvas, "fillPolygonRGBA");
    const line = vi.spyOn(canvas, "drawLineRGBA");
    p.drawRaster({ transform, canvas });
    expect(polygon).toHaveBeenCalledTimes(1);
    // 4 circles x 60 chords + 8 radial lines + 3 outline edges + 1 closing edge
    expect(line).toHaveBeenCalledTimes(4 * 60 + 8 + 4);
    // Inside the diamond, away from every line, the translucent fill is blended.
    const idx = (100 * 200 + 110) * 4;
    expect(canvas.data[idx + 3]).toBeGreaterThan(0);
    expect(canvas.data[idx]).toBeGreaterThan(canvas.data[idx + 1] ?? 0);
  });

  it("splits SVG output at non-finite samples", () => {
    const p = new Polar2D(
      new Float64Array([0, 1, Number.NaN, 2, 3]),
      new Float64Array([1, 1, 1, 1, 1]),
      {}
    );
    const lines = svgOf(p).filter((e) => e.includes('stroke-width="2"'));
    expect(lines).toHaveLength(2);
  });
});

describe("Quiver2D", () => {
  it("ignores arrows with a non-finite component in range and output", () => {
    const q = new Quiver2D(
      new Float64Array([0, 1, 2]),
      new Float64Array([0, 0, 0]),
      new Float64Array([1, Number.NaN, 1]),
      new Float64Array([0, 1, 0]),
      {}
    );
    expect(q.getDataRange()).toEqual({ xmin: 0, xmax: 3, ymin: 0, ymax: 0 });
    const out = svgOf(q).join("\n");
    expect(out).not.toContain("NaN");
    expect(svgOf(q).filter((e) => e.startsWith("<line"))).toHaveLength(2);
  });

  it("draws nothing for a zero-length arrow", () => {
    const q = new Quiver2D(
      new Float64Array([0]),
      new Float64Array([0]),
      new Float64Array([0]),
      new Float64Array([0]),
      {}
    );
    expect(svgOf(q)).toHaveLength(0);
  });

  it("shrinks the head of a short arrow to half its length", () => {
    const q = new Quiver2D(
      new Float64Array([0]),
      new Float64Array([0]),
      new Float64Array([0.1]),
      new Float64Array([0]),
      {}
    );
    const poly = svgOf(q).find((e) => e.startsWith("<polygon")) ?? "";
    // The shaft is 4 px; the head is 2 px long, so its corners sit at x = 102.27 / 102.27.
    const xs = Array.from(poly.matchAll(/points="([^"]+)"/g))
      .flatMap((m) => (m[1] ?? "").split(" "))
      .map((pt) => Number(pt.split(",")[0]));
    expect(Math.max(...xs)).toBeCloseTo(104, 5);
    expect(Math.min(...xs)).toBeGreaterThan(101.9);
  });

  it("draws arrowheads in raster output and validates options", () => {
    const q = new Quiver2D(
      new Float64Array([0]),
      new Float64Array([0]),
      new Float64Array([2]),
      new Float64Array([0]),
      {}
    );
    const canvas = new RasterCanvas(200, 200);
    const tri = vi.spyOn(canvas, "fillTriangleRGBA");
    q.drawRaster({ transform, canvas });
    expect(tri).toHaveBeenCalledTimes(1);

    const a = new Float64Array([0]);
    expect(() => new Quiver2D(a, a, a, a, { scale: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new Quiver2D(a, a, a, a, { linewidth: -1 })).toThrow(InvalidParameterError);
  });
});

describe("Radar2D", () => {
  const series = [new Float64Array([1, 1, 1, 1])];

  it("rejects non-finite values and a bad linewidth", () => {
    expect(() => new Radar2D([new Float64Array([1, Number.NaN, 2])])).toThrow(
      InvalidParameterError
    );
    expect(() => new Radar2D([new Float64Array([1, Infinity, 2])])).toThrow(InvalidParameterError);
    expect(() => new Radar2D(series, { linewidth: 0 })).toThrow(InvalidParameterError);
  });

  it("puts the first axis at the top and runs clockwise", () => {
    const radar = new Radar2D(series);
    const lines = svgOf(radar).filter((e) => e.includes('stroke="#999"'));
    expect(lines).toHaveLength(4);
    const coords = (el: string) =>
      ["x2", "y2"].map((a) => Number(new RegExp(`${a}="([^"]+)"`).exec(el)?.[1]));
    // Axis 0 ends above the center (smaller pixel y), axis 1 to the right.
    const [x0, y0] = coords(lines[0] ?? "");
    const [x1, y1] = coords(lines[1] ?? "");
    expect(x0).toBeCloseTo(100, 6);
    expect(y0).toBeCloseTo(60, 6);
    expect(x1).toBeCloseTo(140, 6);
    expect(y1).toBeCloseTo(100, 6);
  });

  it("draws grid, axes and a translucent fill in raster output", () => {
    const radar = new Radar2D(series, { colors: ["#ff0000"] });
    const canvas = new RasterCanvas(200, 200);
    const polygon = vi.spyOn(canvas, "fillPolygonRGBA");
    const line = vi.spyOn(canvas, "drawLineRGBA");
    radar.drawRaster({ transform, canvas });
    expect(polygon).toHaveBeenCalledTimes(1);
    // 4 rings x 4 edges + 4 axes + 4 outline edges
    expect(line).toHaveBeenCalledTimes(16 + 4 + 4);
  });
});

describe("Stem2D", () => {
  it("draws the baseline across the finite x extent, even for unsorted data", () => {
    const s = new Stem2D(
      new Float64Array([3, Number.NaN, 1, 2]),
      new Float64Array([1, 5, 2, 3]),
      {}
    );
    const base = svgOf(s)[0] ?? "";
    expect(base).toContain('x1="140.00"');
    expect(base).toContain('x2="220.00"');
    expect(base).toContain("stroke-dasharray");
  });

  it("draws no baseline when no point is finite", () => {
    const s = new Stem2D(new Float64Array([Number.NaN]), new Float64Array([1]), {});
    expect(svgOf(s)).toHaveLength(0);
  });

  it("draws the baseline in raster output", () => {
    const s = new Stem2D(new Float64Array([0, 2]), new Float64Array([1, 1]), { color: "#000000" });
    const canvas = rasterOf(s);
    // baseline at y = 0 -> pixel row 100, dashes start at x = 100 (4 on, 2 off)
    const px = (x: number) => canvas.data[(100 * 200 + x) * 4 + 3];
    expect(px(101)).toBe(255);
    expect(px(105)).toBe(0);
  });

  it("validates baseline, size and linewidth", () => {
    const a = new Float64Array([1]);
    expect(() => new Stem2D(a, a, { baseline: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new Stem2D(a, a, { size: 0 })).toThrow(InvalidParameterError);
    expect(() => new Stem2D(a, a, { linewidth: -1 })).toThrow(InvalidParameterError);
  });
});

describe("Strip2D", () => {
  it("places points identically in SVG and raster output, within the jitter", () => {
    const strip = new Strip2D([new Float64Array([1, 2, 3, 4]), new Float64Array([2, 3])], {
      jitter: 0.1,
    });
    const first = svgOf(strip);
    const again = svgOf(strip);
    expect(first).toEqual(again);
    const g0 = first.slice(0, 4).map((e) => Number(/cx="([^"]+)"/.exec(e)?.[1]));
    for (const cx of g0) expect(Math.abs((cx - 100) / 40)).toBeLessThanOrEqual(0.1 + 1e-9);
    const g1 = first.slice(4).map((e) => Number(/cx="([^"]+)"/.exec(e)?.[1]));
    for (const cx of g1) expect(Math.abs((cx - 140) / 40)).toBeLessThanOrEqual(0.1 + 1e-9);
  });

  it("spreads the jitter over the whole [-jitter, jitter] interval", () => {
    const strip = new Strip2D([new Float64Array(400).fill(1)], { jitter: 0.5 });
    const offsets = svgOf(strip).map((e) => (Number(/cx="([^"]+)"/.exec(e)?.[1]) - 100) / 40);
    const quarters = [0, 0, 0, 0];
    for (const o of offsets) {
      const q = Math.min(3, Math.floor((o + 0.5) * 4));
      quarters[q] = (quarters[q] ?? 0) + 1;
    }
    for (const count of quarters) expect(count).toBeGreaterThan(60);
    expect(new Set(offsets.map((o) => o.toFixed(4))).size).toBeGreaterThan(300);
  });

  it("widens the x range when the jitter exceeds half a slot", () => {
    const narrow = new Strip2D([new Float64Array([1]), new Float64Array([2])], {});
    expect(narrow.getDataRange()).toMatchObject({ xmin: -0.5, xmax: 1.5 });
    const wide = new Strip2D([new Float64Array([1]), new Float64Array([2])], { jitter: 0.8 });
    expect(wide.getDataRange()).toMatchObject({ xmin: -0.8, xmax: 1.8 });
  });

  it("validates size and jitter", () => {
    const g = [new Float64Array([1])];
    expect(() => new Strip2D(g, { size: 0 })).toThrow(InvalidParameterError);
    expect(() => new Strip2D(g, { jitter: -0.1 })).toThrow(InvalidParameterError);
    expect(() => new Strip2D(g, { jitter: Number.NaN })).toThrow(InvalidParameterError);
  });
});

describe("Contour2D", () => {
  const Z = grid(2, 2, [1, 2, 3, 4]);

  it("keeps explicit colors tied to their level when other levels miss the data", () => {
    // Level 0.5 lies below the data, so only 2.5 and 3.5 produce lines.
    const c = new Contour2D(Z, {
      levels: [0.5, 2.5, 3.5],
      colors: ["#ff0000", "#00ff00", "#0000ff"],
    });
    const colorsUsed = new Set(c.segments.map((s) => c.levelColors[s.levelIndex]));
    expect(colorsUsed).toEqual(new Set(["#00ff00", "#0000ff"]));
  });

  it("maps a colormap by level value", () => {
    const c = new Contour2D(Z, { levels: [0.5, 2.5, 3.5], colormap: "grayscale" });
    // t = (level - 0.5) / 3 -> 0, 2/3, 1 over the grayscale ramp (0..255).
    expect(c.levelColors[0]).toBe("#000000");
    expect(c.levelColors[2]).toBe("#ffffff");
    const mid = Number.parseInt((c.levelColors[1] ?? "#000000").slice(1, 3), 16);
    expect(mid).toBeGreaterThan(160);
    expect(mid).toBeLessThan(180);
  });

  it("starts the automatic palette at the first drawn level", () => {
    const c = new Contour2D(Z, {});
    expect(c.levelColors[0]).toBe("#1f77b4");
    expect(c.segments[0]?.levelIndex).toBe(0);
  });

  it("does not emit zero-length segments for a level that only touches the maximum", () => {
    const peak = grid(3, 3, [0, 0, 0, 0, 5, 0, 0, 0, 0]);
    const c = new Contour2D(peak, { levels: [5] });
    expect(c.segments).toHaveLength(0);
    for (const seg of new Contour2D(peak, { levels: 6 }).segments) {
      expect(seg.x1 !== seg.x2 || seg.y1 !== seg.y2).toBe(true);
    }
  });

  it("uses the lowest drawn level for the legend color", () => {
    const c = new Contour2D(Z, {
      levels: [0.5, 2.5],
      colors: ["#ff0000", "#00ff00"],
      label: "z",
    });
    expect(c.getLegendEntries()?.[0]?.color).toBe("#00ff00");
  });
});

describe("ContourF2D", () => {
  it("makes the last boundary exactly the maximum", () => {
    // min + 7 * ((max - min) / 7) evaluates to 0.6299999999999999 < 0.63
    const Z = grid(3, 3, [-1.048, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.63]);
    const f = new ContourF2D(Z, { levels: 7 });
    expect(f.levels).toHaveLength(8);
    expect(f.levels[0]).toBe(-1.048);
    expect(f.levels[7]).toBe(0.63);
    // The top band must reach the maximum corner.
    const top = f.triangles.filter((t) => t.band === 6);
    expect(top.some((t) => [t.x1, t.x2, t.x3].includes(2) && [t.y1, t.y2, t.y3].includes(2))).toBe(
      true
    );
  });
});

describe("Heatmap2D", () => {
  it("rejects data whose length does not match rows * cols", () => {
    expect(() => new Heatmap2D(new Float64Array(5), 2, 3, {})).toThrow(ShapeError);
    expect(() => new Heatmap2D(new Float64Array(6), 2.5, 2, {})).toThrow(InvalidParameterError);
    expect(() => new Heatmap2D(new Float64Array(6), 2, 3, {})).not.toThrow();
  });
});
