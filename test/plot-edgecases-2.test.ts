import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { boxplot, Figure, violinplot } from "../src/plot";

const countOccurrences = (haystack: string, needle: string): number => {
  let count = 0;
  let pos = 0;
  for (;;) {
    const next = haystack.indexOf(needle, pos);
    if (next === -1) return count;
    count += 1;
    pos = next + needle.length;
  }
};

describe("Plot - Additional edge cases", () => {
  it("handles barh with NaN/Inf values", () => {
    const fig = new Figure({ width: 240, height: 160 });
    const ax = fig.addAxes();
    ax.barh(tensor([0, 1, 2]), tensor([NaN, Infinity, -Infinity]));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });

  it("handles boxplot with NaN/Inf values", () => {
    const data = tensor([1, NaN, 2, Infinity, -Infinity, 3]);
    const fig = new Figure({ width: 240, height: 160 });
    const ax = fig.addAxes();
    ax.boxplot(data);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });

  it("handles violinplot with NaN/Inf values", () => {
    const data = tensor([1, NaN, 2, Infinity, -Infinity, 3]);
    const fig = new Figure({ width: 240, height: 160 });
    const ax = fig.addAxes();
    ax.violinplot(data);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<path");
  });

  it("skips non-finite values in boxplot and violinplot", () => {
    const fig = new Figure({ width: 240, height: 160 });
    fig.addAxes({ padding: 10 });
    expect(() => boxplot(tensor([NaN, Infinity, -Infinity]))).toThrow(
      /boxplot data must contain at least one finite value/i
    );
    expect(() => violinplot(tensor([NaN, Infinity, -Infinity]))).toThrow(
      /violinplot data must contain at least one finite value/i
    );
  });

  it("counts expected primitives in SVG for mixed plots", () => {
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1, 2]), tensor([1, 2, 3]));
    ax.scatter(tensor([0, 1]), tensor([1, 1]));
    ax.bar(tensor([0, 1]), tensor([2, 3]));
    const svg = fig.renderSVG().svg;
    expect(countOccurrences(svg, "<polyline")).toBe(1);
    expect(countOccurrences(svg, "<circle")).toBe(2);
    expect(countOccurrences(svg, "<rect")).toBeGreaterThan(1);
  });

  it("handles contour/contourf with single-cell grids and levels", () => {
    const Z = tensor([[5]]);
    const fig1 = new Figure({ width: 200, height: 120 });
    const ax1 = fig1.addAxes();
    ax1.contour(tensor([]), tensor([]), Z, { levels: 3 });
    const svg1 = fig1.renderSVG().svg;
    expect(svg1).not.toContain('stroke="#1f77b4"');

    const fig2 = new Figure({ width: 200, height: 120 });
    const ax2 = fig2.addAxes();
    ax2.contourf(tensor([]), tensor([]), Z, { levels: [1, 3, 5] });
    const svg2 = fig2.renderSVG().svg;
    expect(countOccurrences(svg2, "<rect")).toBe(2); // background + axes only
  });
});
