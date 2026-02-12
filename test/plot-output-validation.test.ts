import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Figure, figure, plot, show } from "../src/plot";

describe("Plot Output Validation", () => {
  it("validates SVG structure for line plot", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
    const svg = fig.renderSVG().svg;

    // Verify SVG structure
    expect(svg).toContain('<?xml version="1.0"');
    expect(svg).toContain('<svg xmlns="http://www.w3.org/2000/svg"');
    expect(svg).toContain("</svg>");

    // Verify polyline element exists
    expect(svg).toContain("<polyline");
    expect(svg).toContain("points=");

    // Count polyline elements (should be 1)
    const polylineCount = (svg.match(/<polyline/g) || []).length;
    expect(polylineCount).toBe(1);
  });

  it("validates data points in scatter plot", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    const x = tensor([1, 2, 3]);
    const y = tensor([4, 5, 6]);
    ax.scatter(x, y);
    const svg = fig.renderSVG().svg;

    // Verify circle elements for each point
    const circleCount = (svg.match(/<circle/g) || []).length;
    expect(circleCount).toBe(3);

    // Verify circles have cx, cy, and r attributes
    expect(svg).toContain("cx=");
    expect(svg).toContain("cy=");
    expect(svg).toContain("r=");
  });

  it("validates bar chart rectangles", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.bar(tensor([1, 2, 3]), tensor([10, 20, 15]));
    const svg = fig.renderSVG().svg;

    // Count rect elements (background + axes + 3 bars)
    const rectCount = (svg.match(/<rect/g) || []).length;
    expect(rectCount).toBeGreaterThanOrEqual(5); // bg + axes + 3 bars

    // Verify rects have required attributes
    expect(svg).toContain("width=");
    expect(svg).toContain("height=");
    expect(svg).toContain("fill=");
  });

  it("validates histogram bin count", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    const bins = 5;
    ax.hist(tensor([1, 2, 2, 3, 3, 3, 4, 4, 5]), bins);
    const svg = fig.renderSVG().svg;

    // Count rect elements for bins (plus background and axes)
    const rectCount = (svg.match(/<rect/g) || []).length;
    expect(rectCount).toBeGreaterThanOrEqual(bins + 2);
  });

  it("validates heatmap cell count", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    const data = tensor([
      [1, 2],
      [3, 4],
    ]);
    ax.heatmap(data);
    const svg = fig.renderSVG().svg;

    // Should have 4 cells (2x2) plus background and axes
    const rectCount = (svg.match(/<rect/g) || []).length;
    expect(rectCount).toBeGreaterThanOrEqual(6);
  });

  it("validates color application", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "#ff0000" });
    const svg = fig.renderSVG().svg;

    // Verify red color is applied
    expect(svg).toContain('stroke="#ff0000"');
  });

  it("validates linewidth application", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2]), tensor([1, 2]), { linewidth: 5 });
    const svg = fig.renderSVG().svg;

    // Verify linewidth is applied
    expect(svg).toContain('stroke-width="5"');
  });

  it("validates figure dimensions", () => {
    const fig = new Figure({ width: 800, height: 600 });
    const svg = fig.renderSVG().svg;

    // Verify dimensions in SVG
    expect(svg).toContain('width="800"');
    expect(svg).toContain('height="600"');
    expect(svg).toContain('viewBox="0 0 800 600"');
  });

  it("validates title rendering", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.setTitle("Test Title");
    ax.plot(tensor([1, 2]), tensor([1, 2]));
    const svg = fig.renderSVG().svg;

    // Verify title is rendered
    expect(svg).toContain("Test Title");
    expect(svg).toContain("<text");
    expect(svg).toContain('font-weight="bold"');
  });

  it("validates axis labels", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.setXLabel("X Axis");
    ax.setYLabel("Y Axis");
    ax.plot(tensor([1, 2]), tensor([1, 2]));
    const svg = fig.renderSVG().svg;

    // Verify labels are rendered
    expect(svg).toContain("X Axis");
    expect(svg).toContain("Y Axis");
  });

  it("validates global plot function output", () => {
    figure();
    plot(tensor([1, 2, 3]), tensor([1, 4, 9]), { color: "red" });
    const rendered = show({ format: "svg" });
    expect(rendered).not.toBeInstanceOf(Promise);
    if (rendered instanceof Promise) return;
    expect(rendered.kind).toBe("svg");
    const svg = rendered.svg;

    // Verify plot was created
    expect(svg).toContain("<polyline");
    expect(svg).toContain('stroke="#ff0000"');
  });

  it("validates PNG dimensions", async () => {
    const fig = new Figure({ width: 320, height: 240 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2]), tensor([1, 2]));
    const png = await fig.renderPNG();

    // Verify PNG metadata
    expect(png.kind).toBe("png");
    expect(png.width).toBe(320);
    expect(png.height).toBe(240);
    expect(png.bytes).toBeInstanceOf(Uint8Array);
    expect(png.bytes.length).toBeGreaterThan(0);

    // Verify PNG signature
    expect(png.bytes[0]).toBe(137);
    expect(png.bytes[1]).toBe(80);
    expect(png.bytes[2]).toBe(78);
    expect(png.bytes[3]).toBe(71);
  });

  it("validates multiple plots on same axes", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "#ff0000" });
    ax.plot(tensor([1, 2]), tensor([2, 1]), { color: "#00ff00" });
    const svg = fig.renderSVG().svg;

    // Should have 2 polylines
    const polylineCount = (svg.match(/<polyline/g) || []).length;
    expect(polylineCount).toBe(2);

    // Both colors should be present
    expect(svg).toContain('stroke="#ff0000"');
    expect(svg).toContain('stroke="#00ff00"');
  });

  it("validates named color rendering", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "forestgreen" });
    const svg = fig.renderSVG().svg;

    // Named color should be converted to hex
    expect(svg).toContain('stroke="#228b22"');
  });

  it("renders ticks and tick labels", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    const svg = fig.renderSVG().svg;

    expect(svg).toContain('class="x-tick"');
    expect(svg).toContain('class="y-tick"');
    expect(svg).toContain('class="tick-label');
  });

  it("renders legend when configured", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([1, 2]), { label: "Series A" });
    ax.scatter(tensor([0.2, 0.8]), tensor([0.5, 1.5]), { label: "Series B" });
    ax.legend();
    const svg = fig.renderSVG().svg;

    expect(svg).toContain('class="legend"');
    expect(svg).toContain("Series A");
    expect(svg).toContain("Series B");
  });

  it("validates RGB color rendering", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.scatter(tensor([1]), tensor([1]), { color: "rgb(255, 0, 0)" });
    const svg = fig.renderSVG().svg;

    // RGB should be converted to hex
    expect(svg).toContain('fill="#ff0000"');
  });

  it("validates colormap application", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.heatmap(
      tensor([
        [0, 0.5],
        [0.5, 1],
      ]),
      { colormap: "viridis" }
    );
    const svg = fig.renderSVG().svg;

    // Should contain RGB color values from viridis colormap
    expect(svg).toContain("rgb(");
    expect(svg).toContain('fill="rgb(');
  });
});
