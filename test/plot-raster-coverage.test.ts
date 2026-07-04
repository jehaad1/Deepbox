import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { figure, legend, polar, quiver, radar, stem, strip, waterfall } from "../src/plot";

describe("Plot raster rendering coverage", () => {
  describe("Stem2D raster path", () => {
    it("renders PNG with stems and markers", async () => {
      const fig = figure();
      const x = tensor([1, 2, 3, 4, 5]);
      const y = tensor([1, 3, 2, 5, 4]);
      stem(x, y, { color: "#ff0000", baseline: 2 });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("renders PNG with default baseline", async () => {
      const fig = figure();
      const x = tensor([0, 1, 2]);
      const y = tensor([3, 1, 4]);
      stem(x, y);
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
    });

    it("getLegendEntries returns entries when label set", () => {
      const fig = figure();
      const x = tensor([1, 2, 3]);
      const y = tensor([4, 5, 6]);
      stem(x, y, { label: "my stem" });
      legend();
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("my stem");
    });

    it("handles NaN values in data", async () => {
      const fig = figure();
      const x = tensor([1, 2, NaN, 4]);
      const y = tensor([2, NaN, 3, 5]);
      stem(x, y);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
      const png = await fig.renderPNG();
      expect(png.bytes.length).toBeGreaterThan(0);
    });
  });

  describe("Strip2D raster path", () => {
    it("renders PNG with jittered points", async () => {
      const fig = figure();
      const g1 = tensor([1, 2, 3, 4, 5]);
      const g2 = tensor([2, 3, 4, 5, 6]);
      strip([g1, g2], { labels: ["A", "B"], colors: ["#ff0000", "#0000ff"] });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("renders with custom jitter and size", async () => {
      const fig = figure();
      const g1 = tensor([10, 20, 30]);
      strip([g1], { jitter: 0.4, size: 5 });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
    });

    it("getLegendEntries returns entries when labels set", () => {
      const fig = figure();
      const g1 = tensor([1, 2, 3]);
      strip([g1], { labels: ["Group A"] });
      legend();
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("Group A");
    });

    it("handles NaN values in groups", async () => {
      const fig = figure();
      const g1 = tensor([1, NaN, 3]);
      const g2 = tensor([NaN, 2, 4]);
      strip([g1, g2]);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
      const png = await fig.renderPNG();
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("getDataRange returns null for all-NaN data", () => {
      const fig = figure();
      const g1 = tensor([NaN, NaN]);
      strip([g1]);
      // Rendering should not crash even with bad data
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });
  });

  describe("Radar2D raster path", () => {
    it("renders PNG with radar data", async () => {
      const fig = figure();
      const s1 = tensor([4, 3, 5, 2, 4]);
      const s2 = tensor([3, 5, 2, 4, 3]);
      radar([s1, s2], { labels: ["Series 1", "Series 2"] });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("renders PNG with single series", async () => {
      const fig = figure();
      const s1 = tensor([1, 2, 3, 4, 5]);
      radar([s1], { colors: ["#00ff00"] });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
    });

    it("getLegendEntries returns entries for labeled series", () => {
      const fig = figure();
      const s1 = tensor([1, 2, 3]);
      radar([s1], { labels: ["Radar A"] });
      legend();
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("Radar A");
    });

    it("handles all-zero series (maxVal fallback)", async () => {
      const fig = figure();
      const s1 = tensor([0, 0, 0, 0, 0]);
      radar([s1]);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<polygon");
      const png = await fig.renderPNG();
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("renders multiple series with custom linewidth", async () => {
      const fig = figure();
      const s1 = tensor([1, 5, 3, 2, 4]);
      const s2 = tensor([4, 2, 5, 3, 1]);
      const s3 = tensor([2, 3, 4, 5, 1]);
      radar([s1, s2, s3], { linewidth: 3 });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
    });

    it("throws for mismatched series lengths", () => {
      figure();
      const s1 = tensor([1, 2, 3, 4, 5]);
      const s2 = tensor([1, 2, 3]);
      expect(() => radar([s1, s2])).toThrow();
    });

    it("throws for empty series", () => {
      figure();
      expect(() => radar([])).toThrow();
    });
  });

  describe("Waterfall2D raster path", () => {
    it("renders PNG with positive and negative values", async () => {
      const fig = figure();
      const cats = ["Start", "+Revenue", "-Cost", "-Tax", "Total"];
      const vals = tensor([100, 50, -30, -10, 110]);
      waterfall(cats, vals);
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("renders with custom colors and barWidth", async () => {
      const fig = figure();
      const cats = ["A", "B", "C"];
      const vals = tensor([10, -5, 5]);
      waterfall(cats, vals, {
        positiveColor: "#00ff00",
        negativeColor: "#ff0000",
        totalColor: "#0000ff",
        barWidth: 0.8,
      });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
    });

    it("throws for mismatched categories/values length", () => {
      figure();
      expect(() => waterfall(["A", "B"], tensor([1, 2, 3]))).toThrow();
    });

    it("getDataRange returns null for empty data", () => {
      const fig = figure();
      waterfall([], tensor([]));
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("renders single bar as total", async () => {
      const fig = figure();
      waterfall(["Total"], tensor([100]));
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
    });
  });

  describe("Quiver2D raster path", () => {
    it("renders PNG with vector field arrows", async () => {
      const fig = figure();
      const x = tensor([0, 1, 0, 1]);
      const y = tensor([0, 0, 1, 1]);
      const u = tensor([1, 0, -1, 0]);
      const v = tensor([0, 1, 0, -1]);
      quiver(x, y, u, v, { color: "#333333", scale: 0.5 });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("renders with custom linewidth", async () => {
      const fig = figure();
      const x = tensor([0, 1]);
      const y = tensor([0, 0]);
      const u = tensor([1, -1]);
      const v = tensor([1, -1]);
      quiver(x, y, u, v, { linewidth: 3 });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
    });

    it("getLegendEntries returns entries when label set", () => {
      const fig = figure();
      const x = tensor([0]);
      const y = tensor([0]);
      const u = tensor([1]);
      const v = tensor([1]);
      quiver(x, y, u, v, { label: "vectors" });
      legend();
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("vectors");
    });

    it("handles NaN coordinates", async () => {
      const fig = figure();
      const x = tensor([0, NaN, 2]);
      const y = tensor([0, 1, NaN]);
      const u = tensor([1, 1, 1]);
      const v = tensor([0, 0, 0]);
      quiver(x, y, u, v);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
      const png = await fig.renderPNG();
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("getDataRange returns null for all-NaN data", () => {
      const fig = figure();
      const x = tensor([NaN]);
      const y = tensor([NaN]);
      const u = tensor([1]);
      const v = tensor([1]);
      quiver(x, y, u, v);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("throws for mismatched lengths", () => {
      figure();
      expect(() => quiver(tensor([0, 1]), tensor([0]), tensor([1, 0]), tensor([0, 1]))).toThrow();
    });
  });

  describe("Polar2D raster path", () => {
    it("renders PNG with polar data", async () => {
      const fig = figure();
      const n = 20;
      const thetaArr: number[] = [];
      const rArr: number[] = [];
      for (let i = 0; i < n; i++) {
        thetaArr.push((2 * Math.PI * i) / n);
        rArr.push(1 + 0.5 * Math.sin((3 * (2 * Math.PI * i)) / n));
      }
      polar(tensor(thetaArr), tensor(rArr), { color: "#ff00ff" });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("renders filled polar plot in raster", async () => {
      const fig = figure();
      const theta = tensor([0, 1, 2, 3, 4, 5]);
      const r = tensor([1, 2, 1.5, 2, 1, 1.5]);
      polar(theta, r, { fill: true });
      const png = await fig.renderPNG();
      expect(png.kind).toBe("png");
    });

    it("handles all-zero r values (maxR fallback)", async () => {
      const fig = figure();
      const theta = tensor([0, 1, 2, 3]);
      const r = tensor([0, 0, 0, 0]);
      polar(theta, r);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
      const png = await fig.renderPNG();
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("handles empty data", () => {
      const fig = figure();
      polar(tensor([]), tensor([]), { color: "#000" });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("handles NaN values in polar data", async () => {
      const fig = figure();
      const theta = tensor([0, NaN, 2, 3]);
      const r = tensor([1, 2, NaN, 4]);
      polar(theta, r);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
      const png = await fig.renderPNG();
      expect(png.bytes.length).toBeGreaterThan(0);
    });

    it("getLegendEntries returns entry with label", () => {
      const fig = figure();
      polar(tensor([0, 1, 2]), tensor([1, 2, 3]), { label: "polar data" });
      legend();
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("polar data");
    });

    it("getLegendEntries returns null without label", () => {
      const fig = figure();
      polar(tensor([0, 1, 2]), tensor([1, 2, 3]));
      // No crash, just renders without legend
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("throws for mismatched theta/r lengths", () => {
      figure();
      expect(() => polar(tensor([0, 1]), tensor([1, 2, 3]))).toThrow();
    });
  });
});
