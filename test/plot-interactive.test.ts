import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../src/core/errors/invalid_parameter";
import { tensor } from "../src/ndarray";
import { createInteractivePlot, figure, InteractivePlot } from "../src/plot";

describe("plot/interactive — InteractivePlot", () => {
  it("renders HTML with embedded SVG and default options", () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
    const ip = new InteractivePlot(fig);
    const { html, svg, options } = ip.render();

    expect(svg).toContain("<svg");
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).toContain(svg.trim().slice(0, 20));
    expect(html).toContain("<title>Deepbox Plot</title>");
    expect(options.pan).toBe(true);
    expect(options.zoom).toBe(true);
    expect(options.tooltips).toBe(true);
    expect(options.crosshair).toBe(false);
    expect(html).not.toContain("crosshairH.style.display");
    expect(options.resetOnDoubleClick).toBe(true);
    expect(options.minZoom).toBe(0.1);
    expect(options.maxZoom).toBe(10);
  });

  it("escapes HTML in the page title", () => {
    const fig = figure({ width: 100, height: 80 });
    fig.addAxes();
    const ip = new InteractivePlot(fig, { title: "Test <script>" });
    const { html } = ip.render();
    expect(html).toContain("<title>Test &lt;script&gt;</title>");
    expect(html).not.toContain("<title>Test <script></title>");
  });

  it("throws when minZoom is not positive", () => {
    const fig = figure({ width: 100, height: 80 });
    fig.addAxes();
    expect(() => new InteractivePlot(fig, { minZoom: 0 })).toThrow(InvalidParameterError);
  });

  it("throws when maxZoom is less than minZoom", () => {
    const fig = figure({ width: 100, height: 80 });
    fig.addAxes();
    expect(() => new InteractivePlot(fig, { minZoom: 2, maxZoom: 1 })).toThrow(
      InvalidParameterError
    );
  });

  it("addDataPoints and clearDataPoints affect tooltip payload in HTML", () => {
    const fig = figure({ width: 120, height: 90 });
    fig.addAxes();
    const ip = new InteractivePlot(fig);
    ip.addDataPoints([{ x: 1, y: 2, label: "A" }]);
    let html = ip.render().html;
    expect(html).toContain('"label":"A"');

    ip.clearDataPoints();
    html = ip.render().html;
    expect(html).toContain("const dataPoints = [];");
  });

  it("omits zoom controls and wheel handler when zoom is disabled", () => {
    const fig = figure({ width: 100, height: 80 });
    fig.addAxes();
    const ip = new InteractivePlot(fig, { zoom: false });
    const { html } = ip.render();
    expect(html).not.toContain("zoom-in");
    expect(html).not.toContain("addEventListener('wheel'");
    expect(html).toContain('id="reset"');
  });

  it("omits pan handlers when pan is disabled", () => {
    const fig = figure({ width: 100, height: 80 });
    fig.addAxes();
    const ip = new InteractivePlot(fig, { pan: false });
    const { html } = ip.render();
    expect(html).not.toContain("mousedown");
    expect(html).toContain("cursor: default");
  });

  it("createInteractivePlot returns an InteractivePlot instance", () => {
    const fig = figure({ width: 100, height: 80 });
    fig.addAxes();
    const ip = createInteractivePlot(fig, { crosshair: true });
    expect(ip).toBeInstanceOf(InteractivePlot);
    expect(ip.render().html).toContain("crosshairH.style.display");
  });

  it("injects custom CSS into the document", () => {
    const fig = figure({ width: 100, height: 80 });
    fig.addAxes();
    const ip = new InteractivePlot(fig, { customCSS: "body { color: red; }" });
    expect(ip.render().html).toContain("body { color: red; }");
  });
});
