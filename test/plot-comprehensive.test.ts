import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Figure } from "../src/plot";

const renderSvg = (fig: Figure): string => fig.renderSVG().svg;

describe("Plot - Comprehensive Test Suite", () => {
  describe("Figure Creation", () => {
    it("should create figure with minimum dimensions", () => {
      const fig = new Figure({ width: 1, height: 1 });
      expect(fig.width).toBe(1);
      expect(fig.height).toBe(1);
    });

    it("should create figure with large dimensions", () => {
      const fig = new Figure({ width: 10000, height: 10000 });
      expect(fig.width).toBe(10000);
      expect(fig.height).toBe(10000);
    });

    it("should throw on zero width", () => {
      expect(() => new Figure({ width: 0, height: 100 })).toThrow(/positive integer/);
    });

    it("should throw on zero height", () => {
      expect(() => new Figure({ width: 100, height: 0 })).toThrow(/positive integer/);
    });

    it("should throw on negative width", () => {
      expect(() => new Figure({ width: -100, height: 100 })).toThrow(/positive integer/);
    });

    it("should throw on negative height", () => {
      expect(() => new Figure({ width: 100, height: -100 })).toThrow(/positive integer/);
    });

    it("should throw on fractional width", () => {
      expect(() => new Figure({ width: 100.5, height: 100 })).toThrow(/positive integer/);
    });

    it("should throw on fractional height", () => {
      expect(() => new Figure({ width: 100, height: 100.5 })).toThrow(/positive integer/);
    });

    it("should throw on NaN width", () => {
      expect(() => new Figure({ width: NaN, height: 100 })).toThrow(/positive integer/);
    });

    it("should throw on NaN height", () => {
      expect(() => new Figure({ width: 100, height: NaN })).toThrow(/positive integer/);
    });

    it("should throw on Infinity width", () => {
      expect(() => new Figure({ width: Infinity, height: 100 })).toThrow(/positive integer/);
    });

    it("should throw on Infinity height", () => {
      expect(() => new Figure({ width: 100, height: Infinity })).toThrow(/positive integer/);
    });

    it("should accept custom background color", () => {
      const fig = new Figure({ background: "#ff0000" });
      expect(fig.background).toBe("#ff0000");
    });

    it("should handle empty background color", () => {
      const fig = new Figure({ background: "" });
      expect(fig.background).toBe("#ffffff");
    });

    it("should handle whitespace background color", () => {
      const fig = new Figure({ background: "   " });
      expect(fig.background).toBe("#ffffff");
    });

    it("should trim background color", () => {
      const fig = new Figure({ background: "  #ff0000  " });
      expect(fig.background).toBe("#ff0000");
    });
  });

  describe("Axes Management", () => {
    it("should start with empty axes list", () => {
      const fig = new Figure();
      expect(fig.axesList.length).toBe(0);
    });

    it("should add single axes", () => {
      const fig = new Figure();
      fig.addAxes();
      expect(fig.axesList.length).toBe(1);
    });

    it("should add multiple axes", () => {
      const fig = new Figure();
      fig.addAxes();
      fig.addAxes();
      fig.addAxes();
      expect(fig.axesList.length).toBe(3);
    });

    it("should return axes instance", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      expect(ax).toBeDefined();
      expect(typeof ax.plot).toBe("function");
    });

    it("should accept custom padding", () => {
      const fig = new Figure();
      const ax = fig.addAxes({ padding: 100 });
      expect(ax).toBeDefined();
    });

    it("should accept custom facecolor", () => {
      const fig = new Figure();
      const ax = fig.addAxes({ facecolor: "#f0f0f0" });
      expect(ax).toBeDefined();
    });
  });

  describe("Line Plot - Edge Cases", () => {
    it("should handle single point", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1]), tensor([1]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should handle two points", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should handle large dataset", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const n = 10000;
      const x = Array.from({ length: n }, (_, i) => i);
      const y = Array.from({ length: n }, (_, i) => i * i);
      ax.plot(tensor(x), tensor(y));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should handle all zeros", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([0, 0, 0]), tensor([0, 0, 0]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should handle negative values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([-1, -2, -3]), tensor([-1, -4, -9]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should handle mixed positive and negative", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([-1, 0, 1]), tensor([-1, 0, 1]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should handle very large values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1e10, 2e10]), tensor([1e10, 2e10]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should handle very small values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1e-10, 2e-10]), tensor([1e-10, 2e-10]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should skip NaN in x", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, NaN, 3]), tensor([1, 2, 3]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should skip NaN in y", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, NaN, 3]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should skip Infinity in x", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, Infinity, 3]), tensor([1, 2, 3]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should skip Infinity in y", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, Infinity, 3]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should skip -Infinity in x", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, -Infinity, 3]), tensor([1, 2, 3]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should skip -Infinity in y", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, -Infinity, 3]));
      const svg = renderSvg(fig);
      expect(svg).toContain("polyline");
    });

    it("should handle all NaN", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([NaN, NaN]), tensor([NaN, NaN]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<svg");
    });

    it("should handle all Infinity", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([Infinity, Infinity]), tensor([Infinity, Infinity]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<svg");
    });

    it("should apply custom color", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const line = ax.plot(tensor([1, 2]), tensor([1, 2]), {
        color: "#ff0000",
      });
      expect(line.color).toBe("#ff0000");
    });

    it("should apply custom linewidth", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const line = ax.plot(tensor([1, 2]), tensor([1, 2]), { linewidth: 5 });
      expect(line.linewidth).toBe(5);
    });

    it("should use default color when not specified", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const line = ax.plot(tensor([1, 2]), tensor([1, 2]));
      expect(line.color).toBe("#1f77b4");
    });

    it("should use default linewidth when not specified", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const line = ax.plot(tensor([1, 2]), tensor([1, 2]));
      expect(line.linewidth).toBe(2);
    });
  });

  describe("Scatter Plot - Edge Cases", () => {
    it("should handle single point", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.scatter(tensor([1]), tensor([1]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<circle");
    });

    it("should handle large dataset", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const n = 10000;
      const x = Array.from({ length: n }, (_, i) => i);
      const y = Array.from({ length: n }, (_, _i) => Math.random());
      ax.scatter(tensor(x), tensor(y));
      const svg = renderSvg(fig);
      expect(svg).toContain("<circle");
    });

    it("should apply custom size", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const scatter = ax.scatter(tensor([1, 2]), tensor([1, 2]), { size: 10 });
      expect(scatter.size).toBe(10);
    });

    it("should use default size when not specified", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const scatter = ax.scatter(tensor([1, 2]), tensor([1, 2]));
      expect(scatter.size).toBe(5);
    });

    it("should handle zero size", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      expect(() => ax.scatter(tensor([1, 2]), tensor([1, 2]), { size: 0 })).toThrow(/positive/);
    });

    it("should skip invalid points", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.scatter(tensor([1, NaN, 3]), tensor([1, 2, Infinity]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<circle");
    });
  });

  describe("Bar Chart - Edge Cases", () => {
    it("should handle single bar", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.bar(tensor([1]), tensor([5]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle zero height", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.bar(tensor([1, 2, 3]), tensor([0, 0, 0]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle negative heights", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.bar(tensor([1, 2, 3]), tensor([-1, -2, -3]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle mixed positive and negative heights", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.bar(tensor([1, 2, 3]), tensor([-1, 0, 1]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should apply custom edgecolor", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const bar = ax.bar(tensor([1, 2]), tensor([3, 4]), {
        edgecolor: "#ff0000",
      });
      expect(bar.edgecolor).toBe("#ff0000");
    });

    it("should use default edgecolor", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const bar = ax.bar(tensor([1, 2]), tensor([3, 4]));
      expect(bar.edgecolor).toBe("#000000");
    });
  });

  describe("Histogram - Edge Cases", () => {
    it("should handle single value", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.hist(tensor([5]), 5);
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle all same values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.hist(tensor([5, 5, 5, 5]), 5);
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle 1 bin", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const hist = ax.hist(tensor([1, 2, 3, 4, 5]), 1);
      expect(hist.bins.length).toBe(1);
    });

    it("should handle many bins", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const hist = ax.hist(tensor([1, 2, 3, 4, 5]), 100);
      expect(hist.bins.length).toBe(100);
    });

    it("should handle large dataset", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const data = Array.from({ length: 100000 }, () => Math.random());
      const hist = ax.hist(tensor(data), 50);
      expect(hist.bins.length).toBe(50);
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should skip NaN values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const hist = ax.hist(tensor([1, NaN, 3, NaN, 5]), 5);
      expect(hist.bins.length).toBe(5);
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should skip Infinity values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const hist = ax.hist(tensor([1, Infinity, 3, -Infinity, 5]), 5);
      expect(hist.bins.length).toBe(5);
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle all NaN", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const hist = ax.hist(tensor([NaN, NaN, NaN]), 5);
      expect(hist.bins.length).toBe(0);
    });

    it("should count correctly", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const hist = ax.hist(tensor([1, 1, 2, 2, 2, 3]), 3);
      let total = 0;
      for (let i = 0; i < hist.counts.length; i++) {
        total += hist.counts[i] ?? 0;
      }
      expect(total).toBe(6);
    });
  });

  describe("Heatmap - Edge Cases", () => {
    it("should handle 1x1 matrix", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.heatmap(tensor([[5]]));
      const svg = renderSvg(fig);
      expect(svg).toContain("rgb(");
    });

    it("should handle large matrix", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const data = Array.from({ length: 100 }, () =>
        Array.from({ length: 100 }, () => Math.random())
      );
      ax.heatmap(tensor(data));
      const svg = renderSvg(fig);
      expect(svg).toContain("rgb(");
    });

    it("should handle all zeros", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.heatmap(
        tensor([
          [0, 0],
          [0, 0],
        ])
      );
      const svg = renderSvg(fig);
      expect(svg).toContain("rgb(");
    });

    it("should handle negative values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.heatmap(
        tensor([
          [-1, -2],
          [-3, -4],
        ])
      );
      const svg = renderSvg(fig);
      expect(svg).toContain("rgb(");
    });

    it("should apply custom vmin", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const heatmap = ax.heatmap(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        { vmin: 0 }
      );
      expect(heatmap.vmin).toBe(0);
    });

    it("should apply custom vmax", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const heatmap = ax.heatmap(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        { vmax: 10 }
      );
      expect(heatmap.vmax).toBe(10);
    });

    it("should handle NaN values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.heatmap(
        tensor([
          [1, NaN],
          [3, 4],
        ])
      );
      const svg = renderSvg(fig);
      expect(svg).toContain("rgb(");
    });
  });

  describe("SVG Rendering - Validation", () => {
    it("should produce valid XML", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      const result = fig.renderSVG();
      expect(result.svg).toContain('<?xml version="1.0"');
      expect(result.svg).toContain("<svg");
      expect(result.svg).toContain("</svg>");
    });

    it("should include xmlns attribute", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = fig.renderSVG();
      expect(result.svg).toContain('xmlns="http://www.w3.org/2000/svg"');
    });

    it("should set correct viewBox", () => {
      const fig = new Figure({ width: 800, height: 600 });
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = fig.renderSVG();
      expect(result.svg).toContain('viewBox="0 0 800 600"');
    });

    it("should escape special characters in title", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setTitle("Test <>&\"'");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("&lt;");
      expect(result.svg).toContain("&gt;");
      expect(result.svg).toContain("&amp;");
    });

    it("should include title when set", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setTitle("My Title");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("My Title");
    });

    it("should include xlabel when set", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setXLabel("X Axis");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("X Axis");
    });

    it("should include ylabel when set", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setYLabel("Y Axis");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("Y Axis");
    });

    it("should render empty plot", () => {
      const fig = new Figure();
      fig.addAxes();
      const result = fig.renderSVG();
      expect(result.kind).toBe("svg");
      expect(result.svg).toContain("<svg");
    });

    it("should render multiple axes", () => {
      const fig = new Figure();
      fig.addAxes();
      fig.addAxes();
      const result = fig.renderSVG();
      expect(result.kind).toBe("svg");
    });
  });

  describe("PNG Rendering - Validation", () => {
    it("should produce valid PNG signature", async () => {
      const fig = new Figure({ width: 10, height: 10 });
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = await fig.renderPNG();
      expect(result.bytes[0]).toBe(137);
      expect(result.bytes[1]).toBe(80);
      expect(result.bytes[2]).toBe(78);
      expect(result.bytes[3]).toBe(71);
    });

    it("should set correct dimensions", async () => {
      const fig = new Figure({ width: 123, height: 456 });
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = await fig.renderPNG();
      expect(result.width).toBe(123);
      expect(result.height).toBe(456);
    });

    it("should produce non-empty bytes", async () => {
      const fig = new Figure({ width: 10, height: 10 });
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const result = await fig.renderPNG();
      expect(result.bytes.length).toBeGreaterThan(64);
    });

    it("should render empty plot", async () => {
      const fig = new Figure({ width: 10, height: 10 });
      fig.addAxes();
      const result = await fig.renderPNG();
      expect(result.kind).toBe("png");
      expect(result.bytes.length).toBeGreaterThan(0);
    });
  });

  describe("Title and Labels", () => {
    it("should set title", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setTitle("Test Title");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const svg = renderSvg(fig);
      expect(svg).toContain("Test Title");
    });

    it("should set xlabel", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setXLabel("X Label");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const svg = renderSvg(fig);
      expect(svg).toContain("X Label");
    });

    it("should set ylabel", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setYLabel("Y Label");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const svg = renderSvg(fig);
      expect(svg).toContain("Y Label");
    });

    it("should handle empty title", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setTitle("Filled");
      ax.setTitle("");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const svg = renderSvg(fig);
      expect(svg).not.toContain("Filled");
    });

    it("should handle long title", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const longTitle = "A".repeat(1000);
      ax.setTitle(longTitle);
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const svg = renderSvg(fig);
      expect(svg).toContain(longTitle.slice(0, 50));
    });

    it("should handle unicode in title", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.setTitle("测试 🎨 αβγ");
      ax.plot(tensor([1, 2]), tensor([1, 2]));
      const svg = renderSvg(fig);
      expect(svg).toContain("测试 🎨 αβγ");
    });
  });

  describe("Color Handling", () => {
    it("should accept hex colors", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "#ff0000" });
      const svg = renderSvg(fig);
      expect(svg).toContain('stroke="#ff0000"');
    });

    it("should accept 8-digit hex colors with alpha", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "#ff0000ff" });
      const svg = renderSvg(fig);
      expect(svg).toContain('stroke="#ff0000ff"');
    });

    it("should handle uppercase hex", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "#FF0000" });
      const svg = renderSvg(fig);
      expect(svg).toContain('stroke="#ff0000"');
    });

    it("should handle mixed case hex", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "#Ff00Aa" });
      const svg = renderSvg(fig);
      expect(svg).toContain('stroke="#ff00aa"');
    });

    it("should trim color whitespace", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "  #ff0000  " });
      const svg = renderSvg(fig);
      expect(svg).toContain('stroke="#ff0000"');
    });
  });

  describe("Data Validation", () => {
    it("should throw on mismatched x and y lengths", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      expect(() => ax.plot(tensor([1, 2, 3]), tensor([1, 2]))).toThrow(/same length/);
    });

    it("should throw on 2D tensor for plot x", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      expect(() => ax.plot(tensor([[1, 2]]), tensor([1, 2]))).toThrow(/1D tensor/);
    });

    it("should throw on 2D tensor for plot y", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      expect(() => ax.plot(tensor([1, 2]), tensor([[1, 2]]))).toThrow(/1D tensor/);
    });

    it("should throw on 1D tensor for heatmap", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      expect(() => ax.heatmap(tensor([1, 2, 3]))).toThrow(/2D tensor/);
    });

    it("should throw on 3D tensor for heatmap", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      expect(() => ax.heatmap(tensor([[[1]]]))).toThrow(/2D tensor/);
    });
  });

  describe("Performance Tests", () => {
    it("should handle 10k points efficiently", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const n = 10000;
      const x = Array.from({ length: n }, (_, i) => i);
      const y = Array.from({ length: n }, (_, i) => Math.sin(i / 100));
      const start = Date.now();
      ax.plot(tensor(x), tensor(y));
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(1000); // Should complete in under 1 second
    });

    it("should render 10k points to SVG efficiently", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const n = 10000;
      const x = Array.from({ length: n }, (_, i) => i);
      const y = Array.from({ length: n }, (_, i) => Math.sin(i / 100));
      ax.plot(tensor(x), tensor(y));
      const start = Date.now();
      fig.renderSVG();
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(2000); // Should complete in under 2 seconds
    });

    it("should handle large histogram efficiently", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const n = 100000;
      const data = Array.from({ length: n }, () => Math.random());
      const start = Date.now();
      ax.hist(tensor(data), 100);
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(1000);
    });
  });

  describe("Multiple Plots", () => {
    it("should render multiple line plots with different colors", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, 2, 3]), { color: "#ff0000" });
      ax.plot(tensor([1, 2, 3]), tensor([3, 2, 1]), { color: "#00ff00" });
      ax.plot(tensor([1, 2, 3]), tensor([2, 2, 2]), { color: "#0000ff" });
      const result = fig.renderSVG();
      expect(result.svg).toContain("#ff0000");
      expect(result.svg).toContain("#00ff00");
      expect(result.svg).toContain("#0000ff");
    });

    it("should render line and scatter together", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, 2, 3]));
      ax.scatter(tensor([1, 2, 3]), tensor([3, 2, 1]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
      expect(result.svg).toContain("circle");
    });

    it("should render 10 plots on same axes", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      for (let i = 0; i < 10; i++) {
        ax.plot(tensor([1, 2, 3]), tensor([i, i + 1, i + 2]));
      }
      const result = fig.renderSVG();
      expect(result.kind).toBe("svg");
    });
  });

  describe("Horizontal Bar Chart", () => {
    it("should create horizontal bars", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.barh(tensor([1, 2, 3]), tensor([5, 10, 7]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle single bar", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.barh(tensor([1]), tensor([5]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle zero width", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.barh(tensor([1, 2]), tensor([0, 0]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should handle negative widths", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.barh(tensor([1, 2]), tensor([-5, -10]));
      const svg = renderSvg(fig);
      expect(svg).toContain("<rect");
    });

    it("should throw on mismatched lengths", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      expect(() => ax.barh(tensor([1, 2, 3]), tensor([5, 10]))).toThrow(/same length/);
    });
  });

  describe("Imshow", () => {
    it("should display image", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.imshow(
        tensor([
          [1, 2],
          [3, 4],
        ])
      );
      const svg = renderSvg(fig);
      expect(svg).toContain("rgb(");
    });

    it("should handle 1x1 image", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.imshow(tensor([[255]]));
      const svg = renderSvg(fig);
      expect(svg).toContain("rgb(");
    });

    it("should handle large image", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const img = Array.from({ length: 100 }, () =>
        Array.from({ length: 100 }, () => Math.random() * 255)
      );
      ax.imshow(tensor(img));
      const svg = renderSvg(fig);
      expect(svg).toContain("rgb(");
    });
  });
});
