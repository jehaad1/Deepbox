import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { boxplot, contour, contourf, Figure, figure, violinplot } from "../src/plot";

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

describe("Plot - additional edge cases and canvas assertions", () => {
  it("skips non-finite values in barh and renders expected rects", () => {
    const fig = new Figure({ width: 200, height: 120 });
    const ax = fig.addAxes({ padding: 10 });
    ax.barh(tensor([0, 1, 2]), tensor([1, NaN, Infinity]));
    const svg = fig.renderSVG().svg;
    const rects = countOccurrences(svg, "<rect");
    // background + axes + 1 bar
    expect(rects).toBe(3);
  });

  it("renders boxplot and violinplot with NaN/Inf filtered", () => {
    const data = tensor([NaN, 1, 2, Infinity, 3]);

    const fig1 = figure({ width: 180, height: 120 });
    boxplot(data, { color: "#8c564b" });
    const svg1 = fig1.renderSVG().svg;
    expect(svg1).toContain('fill="#8c564b"'); // Box body
    expect(svg1).toContain('stroke-width="2"'); // Median line

    const fig2 = figure({ width: 180, height: 120 });
    violinplot(data, { color: "#17becf" });
    const svg2 = fig2.renderSVG().svg;
    expect(svg2).toContain("<path"); // Violin shape (KDE)
    expect(svg2).toContain('fill="#17becf"');
  });

  it("contour/contourf render expected primitives", () => {
    const z = tensor([
      [NaN, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const fig1 = figure({ width: 200, height: 100 });
    contour(tensor([]), tensor([]), z);
    const svg1 = fig1.renderSVG().svg;
    expect(svg1).toContain('stroke="#1f77b4"');

    const fig2 = figure({ width: 200, height: 100 });
    contourf(
      tensor([]),
      tensor([]),
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    const svg2 = fig2.renderSVG().svg;
    expect(svg2).toContain("<path");
  });
});
