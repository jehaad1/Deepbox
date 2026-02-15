import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { bar, Figure, heatmap, hist, plot, scatter } from "../src/plot";

describe("consumer API: plot", () => {
  it("Figure + Axes API renders SVG", () => {
    const fig = new Figure({ width: 800, height: 600 });
    const ax = fig.addAxes();
    ax.scatter(tensor([1, 2, 3]), tensor([4, 5, 6]), { color: "#1f77b4" });
    ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]), { color: "#ff7f0e" });
    ax.setTitle("Test Plot");
    ax.setXLabel("X");
    ax.setYLabel("Y");
    const svg = fig.renderSVG();
    expect(svg.svg.length).toBeGreaterThan(0);
    expect(svg.svg).toContain("<svg");
  });

  it("global plotting functions work without error", () => {
    scatter(tensor([1, 2]), tensor([3, 4]));
    plot(tensor([1, 2]), tensor([3, 4]));
    hist(tensor([1, 2, 2, 3, 3, 3, 4, 4, 5]), 5);
    bar(tensor([1, 2, 3]), tensor([10, 20, 30]));
    heatmap(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
  });

  it("hist accepts { bins: N } object form", () => {
    expect(() => hist(tensor([1, 2, 2, 3, 3, 3]), { bins: 3 })).not.toThrow();
  });
});
