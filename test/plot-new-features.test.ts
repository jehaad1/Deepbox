import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { axhline, axvline, Figure, figure, groupedBar, plot, show, stackedBar } from "../src/plot";

function renderCurrentSvg(): string {
  const rendered = show({ format: "svg" });
  expect(rendered).not.toBeInstanceOf(Promise);
  if (rendered instanceof Promise) return "";
  return rendered.svg;
}

describe("axhline", () => {
  it("renders a horizontal dashed line in SVG", () => {
    figure();
    plot(tensor([0, 1, 2]), tensor([0, 1, 2]));
    axhline(1, { color: "#ff0000" });
    const svg = renderCurrentSvg();
    expect(svg).toContain('class="axhline"');
    expect(svg).toContain("stroke-dasharray");
    expect(svg).toContain("#ff0000");
  });

  it("works with default options", () => {
    figure();
    plot(tensor([0, 1]), tensor([0, 1]));
    axhline(0.5);
    const svg = renderCurrentSvg();
    expect(svg).toContain('class="axhline"');
  });

  it("renders multiple hlines", () => {
    figure();
    plot(tensor([0, 1, 2]), tensor([0, 1, 2]));
    axhline(0.5, { color: "#ff0000" });
    axhline(1.5, { color: "#00ff00" });
    const svg = renderCurrentSvg();
    const matches = svg.match(/class="axhline"/g);
    expect(matches).toHaveLength(2);
  });
});

describe("axvline", () => {
  it("renders a vertical dashed line in SVG", () => {
    figure();
    plot(tensor([0, 1, 2]), tensor([0, 1, 2]));
    axvline(1, { color: "#0000ff" });
    const svg = renderCurrentSvg();
    expect(svg).toContain('class="axvline"');
    expect(svg).toContain("stroke-dasharray");
    expect(svg).toContain("#0000ff");
  });

  it("works with default options", () => {
    figure();
    plot(tensor([0, 1]), tensor([0, 1]));
    axvline(0.5);
    const svg = renderCurrentSvg();
    expect(svg).toContain('class="axvline"');
  });
});

describe("setXScale / setYScale (log axes)", () => {
  it("log y-scale renders without error", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([10, 100, 1000]));
    ax.setYScale("log");
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<polyline");
  });

  it("log x-scale renders without error", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 10, 100]), tensor([1, 2, 3]));
    ax.setXScale("log");
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<polyline");
  });

  it("log both axes renders without error", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 10, 100]), tensor([10, 100, 1000]));
    ax.setXScale("log");
    ax.setYScale("log");
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<polyline");
  });

  it("linear scale is the default", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([1, 2, 3]));
    const svg1 = fig.renderSVG().svg;

    const fig2 = new Figure();
    const ax2 = fig2.addAxes();
    ax2.plot(tensor([1, 2, 3]), tensor([1, 2, 3]));
    ax2.setXScale("linear");
    ax2.setYScale("linear");
    const svg2 = fig2.renderSVG().svg;
    expect(svg1).toEqual(svg2);
  });
});

describe("stackedBar", () => {
  it("renders stacked bars for two series", () => {
    figure();
    const x = tensor([1, 2, 3]);
    const h1 = tensor([3, 5, 2]);
    const h2 = tensor([2, 3, 4]);
    stackedBar(x, [h1, h2], {
      colors: ["#ff0000", "#00ff00"],
      labels: ["A", "B"],
    });
    const svg = renderCurrentSvg();
    // Stacked bars render as filled polygons (Line2D polylines)
    expect(svg).toContain("<polyline");
  });

  it("works with default colors", () => {
    figure();
    stackedBar(tensor([1, 2]), [tensor([1, 2]), tensor([3, 4])]);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<polyline");
  });
});

describe("groupedBar", () => {
  it("renders grouped bars for two series", () => {
    figure();
    const x = tensor([1, 2, 3]);
    const h1 = tensor([3, 5, 2]);
    const h2 = tensor([2, 3, 4]);
    groupedBar(x, [h1, h2], {
      colors: ["#ff0000", "#00ff00"],
      labels: ["A", "B"],
    });
    const svg = renderCurrentSvg();
    // Grouped bars render as rect elements
    expect(svg).toContain("<rect");
  });

  it("renders correct number of bar groups", () => {
    figure();
    const x = tensor([1, 2, 3]);
    groupedBar(x, [tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9])]);
    const svg = renderCurrentSvg();
    // 3 categories × 3 series = 9 bars
    const rects = svg.match(/<rect /g) ?? [];
    // At least 9 bar rects (plus background rect)
    expect(rects.length).toBeGreaterThanOrEqual(9);
  });

  it("does nothing for empty heights", () => {
    figure();
    plot(tensor([0, 1]), tensor([0, 1]));
    groupedBar(tensor([1, 2]), []);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<polyline");
  });
});

describe("Axes methods directly", () => {
  it("axhline on axes instance", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.axhline(0.5, { color: "#ff0000", linewidth: 2 });
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('class="axhline"');
    expect(svg).toContain('stroke-width="2"');
  });

  it("axvline on axes instance", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.axvline(0.5, { color: "#0000ff", linewidth: 3 });
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('class="axvline"');
    expect(svg).toContain('stroke-width="3"');
  });

  it("stackedBar on axes instance", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.stackedBar(tensor([1, 2]), [tensor([3, 4]), tensor([1, 2])]);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<polyline");
  });

  it("groupedBar on axes instance", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.groupedBar(tensor([1, 2]), [tensor([3, 4]), tensor([1, 2])]);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });
});
