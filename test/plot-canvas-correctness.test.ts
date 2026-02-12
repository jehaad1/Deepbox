import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Figure } from "../src/plot";

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

describe("Plot - Canvas correctness and edge cases", () => {
  it("renders exact rect count for bar chart", () => {
    const fig = new Figure({ width: 200, height: 100 });
    const ax = fig.addAxes({ padding: 10 });
    ax.bar(tensor([0, 1, 2]), tensor([1, 2, 3]));
    const svg = fig.renderSVG().svg;
    const rects = countOccurrences(svg, "<rect");
    expect(rects).toBe(5);
    expect(svg).toContain('<rect x="10" y="10" width="180" height="80"');
  });

  it("renders exact rect count for heatmap", () => {
    const fig = new Figure({ width: 200, height: 100 });
    const ax = fig.addAxes({ padding: 10 });
    ax.heatmap(
      tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
    );
    const svg = fig.renderSVG().svg;
    const rects = countOccurrences(svg, "<rect");
    expect(rects).toBe(8);
    expect(svg).toContain('<rect x="10" y="10" width="180" height="80"');
  });

  it("renders exact rect count for horizontal bar chart", () => {
    const fig = new Figure({ width: 200, height: 100 });
    const ax = fig.addAxes({ padding: 10 });
    ax.barh(tensor([0, 1, 2]), tensor([1, 2, 3]));
    const svg = fig.renderSVG().svg;
    const rects = countOccurrences(svg, "<rect");
    expect(rects).toBe(5);
    expect(svg).toContain('<rect x="10" y="10" width="180" height="80"');
  });

  it("skips non-finite heatmap cells in SVG", () => {
    const fig = new Figure({ width: 200, height: 100 });
    const ax = fig.addAxes({ padding: 10 });
    ax.heatmap(
      tensor([
        [1, NaN],
        [2, 3],
      ])
    );
    const svg = fig.renderSVG().svg;
    const rects = countOccurrences(svg, "<rect");
    // 1 figure background + 1 axes + 3 finite cells
    expect(rects).toBe(5);
  });

  it("handles histogram bin edges with small and large bins", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    const h1 = ax.hist(tensor([1, 2, 3, 4, 5]), 1);
    const h2 = ax.hist(tensor([1, 2, 3, 4, 5]), 50);
    expect(h1.bins.length).toBe(1);
    expect(h2.bins.length).toBe(50);
  });

  it("rejects histogram with no finite values", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    expect(() => ax.hist(tensor([NaN, Infinity, -Infinity]), 5)).toThrow(
      /histogram data must contain at least one finite value/i
    );
  });

  it("handles extreme padding and still renders", () => {
    const fig = new Figure({ width: 120, height: 80 });
    fig.addAxes({ padding: 0 });
    expect(() => fig.addAxes({ padding: 1000 })).toThrow(/padding is too large/i);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<svg");
  });

  it("renders PNG for mixed plot types", async () => {
    const fig = new Figure({ width: 160, height: 120 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1, 2]), tensor([1, 2, 3]));
    ax.scatter(tensor([0, 1, 2]), tensor([1, 1, 1]));
    ax.bar(tensor([0, 1, 2]), tensor([2, 1, 3]));
    ax.hist(tensor([1, 2, 2, 3, 3, 3]), 3);
    ax.heatmap(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    const png = await fig.renderPNG();
    expect(png.bytes[0]).toBe(137);
    expect(png.bytes.length).toBeGreaterThan(100);
  });
});
