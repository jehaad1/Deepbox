import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { figure, jointplot, polar, quiver, radar, stem, strip, waterfall } from "../src/plot";

describe("Plot stem, strip & specialty charts", () => {
  describe("stem()", () => {
    it("renders SVG with stems and markers", () => {
      const fig = figure();
      const x = tensor([1, 2, 3, 4, 5]);
      const y = tensor([1, 3, 2, 5, 4]);
      stem(x, y, { color: "#ff0000" });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<line");
      expect(svg).toContain("<circle");
    });

    it("supports custom baseline", () => {
      const fig = figure();
      const x = tensor([1, 2, 3]);
      const y = tensor([5, 6, 7]);
      stem(x, y, { baseline: 4 });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });
  });

  describe("strip()", () => {
    it("renders jittered categorical scatter", () => {
      const fig = figure();
      const g1 = tensor([1, 2, 3, 4, 5]);
      const g2 = tensor([2, 3, 4, 5, 6]);
      strip([g1, g2], { labels: ["A", "B"], colors: ["#ff0000", "#0000ff"] });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<circle");
    });
  });

  describe("radar()", () => {
    it("renders radar chart with grid and data polygon", () => {
      const fig = figure();
      const s1 = tensor([4, 3, 5, 2, 4]);
      const s2 = tensor([3, 5, 2, 4, 3]);
      radar([s1, s2], { labels: ["Series 1", "Series 2"] });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<polygon");
    });

    it("throws for fewer than 3 axes", () => {
      figure();
      const s1 = tensor([1, 2]);
      expect(() => radar([s1])).toThrow();
    });
  });

  describe("waterfall()", () => {
    it("renders waterfall chart with bars", () => {
      const fig = figure();
      const cats = ["Start", "+Rev", "-Cost", "Total"];
      const vals = tensor([100, 50, -30, 120]);
      waterfall(cats, vals);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<rect");
    });
  });

  describe("quiver()", () => {
    it("renders vector field arrows", () => {
      const fig = figure();
      const x = tensor([0, 1, 0, 1]);
      const y = tensor([0, 0, 1, 1]);
      const u = tensor([1, 0, -1, 0]);
      const v = tensor([0, 1, 0, -1]);
      quiver(x, y, u, v, { color: "#333333" });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<line");
      expect(svg).toContain("<polygon"); // arrowheads
    });
  });

  describe("polar()", () => {
    it("renders polar coordinate data", () => {
      const fig = figure();
      const n = 50;
      const thetaArr: number[] = [];
      const rArr: number[] = [];
      for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        thetaArr.push(t);
        rArr.push(1 + 0.5 * Math.sin(3 * t));
      }
      const theta = tensor(thetaArr);
      const r = tensor(rArr);
      polar(theta, r, { color: "#ff00ff" });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<polyline");
    });

    it("supports fill option", () => {
      const fig = figure();
      const theta = tensor([0, 1, 2, 3, 4, 5]);
      const r = tensor([1, 2, 1.5, 2, 1, 1.5]);
      polar(theta, r, { fill: true });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("fill-opacity");
    });
  });

  describe("jointplot()", () => {
    it("returns a Figure with scatter and marginal histograms", () => {
      const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const y = tensor([2, 4, 3, 5, 7, 6, 8, 9, 10, 11]);
      const fig = jointplot(x, y, {
        color: "#1f77b4",
        bins: 5,
        xlabel: "X",
        ylabel: "Y",
        title: "Joint",
      });
      const svg = fig.renderSVG().svg;
      // Should contain scatter circles and histogram bars
      expect(svg).toContain("<circle");
      expect(svg).toContain("<rect");
    });
  });
});
