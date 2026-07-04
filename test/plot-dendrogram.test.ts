import { describe, expect, it } from "vitest";
import { plotDendrogram } from "../src/plot";
import { Figure } from "../src/plot/figure/Figure";

describe("plotDendrogram", () => {
  it("renders a dendrogram from a linkage matrix without error", () => {
    // 4 leaves, 3 merges
    // [clusterA, clusterB, distance, count]
    const linkage: [number, number, number, number][] = [
      [0, 1, 1.0, 2], // merge 0 and 1 at distance 1.0 -> node 4
      [2, 3, 1.5, 2], // merge 2 and 3 at distance 1.5 -> node 5
      [4, 5, 3.0, 4], // merge node4 and node5 at distance 3.0 -> node 6
    ];

    const fig = new Figure();
    const ax = fig.addAxes();
    ax.dendrogram(linkage, 4);
    const svg = fig.renderSVG();
    expect(svg.svg).toContain("<svg");
    expect(svg.svg).toContain("<line");
  });

  it("renders via global plotDendrogram function", () => {
    const linkage: [number, number, number, number][] = [
      [0, 1, 0.5, 2],
      [2, 3, 1.0, 2],
      [4, 5, 2.0, 4],
    ];

    // Should not throw
    expect(() => plotDendrogram(linkage, 4)).not.toThrow();
  });

  it("handles single merge (2 leaves)", () => {
    const linkage: [number, number, number, number][] = [[0, 1, 1.0, 2]];

    const fig = new Figure();
    const ax = fig.addAxes();
    ax.dendrogram(linkage, 2);
    const svg = fig.renderSVG();
    expect(svg.svg).toContain("<line");
  });

  it("handles color option", () => {
    const linkage: [number, number, number, number][] = [[0, 1, 1.0, 2]];

    const fig = new Figure();
    const ax = fig.addAxes();
    ax.dendrogram(linkage, 2, { color: "#ff0000" });
    const svg = fig.renderSVG();
    expect(svg.svg).toContain("#ff0000");
  });
});
