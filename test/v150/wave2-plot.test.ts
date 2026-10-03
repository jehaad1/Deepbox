import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { tensor } from "../../src/ndarray";
import type {
  AnimationEasing,
  AnimationFrameGenerator,
  Axes,
  ColormapName,
  PlotTheme,
  TextOptions,
  ViolinplotOptions,
} from "../../src/plot";
import {
  annotate,
  area,
  errorbar,
  Figure,
  figure,
  fillBetween,
  gca,
  gcf,
  getTheme,
  grid,
  groupedBar,
  kdeplot,
  plot,
  resetTheme,
  sca,
  setTheme,
  show,
  step,
  subplot,
  text,
  title,
  twinx,
  xlabel,
  xlim,
  ylabel,
  ylim,
} from "../../src/plot";
import { svgToPdf } from "../../src/plot/renderers/pdf";
import { applyColormap, assertColormapName, COLORMAP_NAMES } from "../../src/plot/utils/colormaps";
import { buildContourGrid } from "../../src/plot/utils/contours";
import { kernelDensityEstimation, scottBandwidth } from "../../src/plot/utils/statistics";
import { generateTicks } from "../../src/plot/utils/ticks";

type AnyDrawable = { readonly kind: string } & Record<string, unknown>;

function drawablesOf(ax: Axes): AnyDrawable[] {
  return (ax as unknown as { drawables: AnyDrawable[] }).drawables;
}

function pdfText(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("latin1");
}

beforeEach(() => {
  resetTheme();
  figure();
});

afterEach(() => {
  resetTheme();
});

describe("index exports and pyplot-style helpers", () => {
  it("exposes the new types (compile-time) and gcf/sca", () => {
    const easing: AnimationEasing = "ease-in";
    const generator: AnimationFrameGenerator = () => ({ svg: "<svg/>" }) as never;
    const cmap: ColormapName = "cividis";
    const textOptions: TextOptions = { ha: "center", va: "top" };
    const violin: ViolinplotOptions = { bandwidth: "scott" };
    const theme: PlotTheme = getTheme();
    expect(easing).toBe("ease-in");
    expect(typeof generator).toBe("function");
    expect(cmap).toBe("cividis");
    expect(textOptions.ha).toBe("center");
    expect(violin.bandwidth).toBe("scott");
    expect(theme.textColor).toBe("#000");
    const fig = figure({ width: 100, height: 80 });
    expect(gcf()).toBe(fig);
    const ax = fig.addAxes();
    expect(sca(ax)).toBe(ax);
    expect(gca()).toBe(ax);
  });

  it("sca rejects an axes that is not part of its figure", () => {
    const orphan = new Figure().addAxes();
    orphan.fig.axesList.length = 0;
    expect(() => sca(orphan)).toThrow(InvalidParameterError);
  });

  it("draws with step, errorbar, fillBetween and area on the current axes", () => {
    figure({ width: 200, height: 150 });
    const x = tensor([0, 1, 2, 3]);
    step(x, tensor([1, 3, 2, 4]), { where: "pre" });
    errorbar(x, tensor([1, 3, 2, 4]), tensor([0.5, 0.5, 0.5, 0.5]));
    fillBetween(x, tensor([0, 1, 1, 2]), tensor([2, 3, 3, 4]), { color: "#ff0000" });
    area(x, tensor([1, 2, 1, 2]), { color: "#00ff00" });
    const kinds = drawablesOf(gca()).map((d) => d.kind);
    // step (1) + errorbar (1 line + 4 bars) + fillBetween (1) + area (1)
    expect(kinds).toHaveLength(8);
    const svg = show().svg;
    expect(svg).toContain('fill="#ff0000"');
    expect(svg).toContain('fill="#00ff00"');
  });

  it("keeps the snake_case fill_between as a deprecated alias of fillBetween", () => {
    const ax = new Figure().addAxes();
    const a = ax.fillBetween(tensor([0, 1]), tensor([0, 0]), tensor([1, 1]));
    const b = ax.fill_between(tensor([0, 1]), tensor([0, 0]), tensor([1, 1]));
    expect(b.x).toEqual(a.x);
    expect(b.y).toEqual(a.y);
  });

  it("sets limits, grid, title and labels through the global helpers", () => {
    figure({ width: 300, height: 200 });
    plot(tensor([0, 1, 2]), tensor([0, 1, 4]));
    xlim(0, 10);
    ylim(-1, 5);
    grid(true, { color: "#123456" });
    title("My title");
    xlabel("the x");
    ylabel("the y");
    text(1, 1, "hello");
    annotate("world", 2, 2, { ha: "right" });
    const svg = show().svg;
    expect(svg).toContain(">My title</text>");
    expect(svg).toContain(">the x</text>");
    expect(svg).toContain(">the y</text>");
    expect(svg).toContain('stroke="#123456"');
    expect(svg).toContain(">hello</text>");
    expect(svg).toContain('text-anchor="end"');
    // xlim(0, 10): tick 10 is drawn, so the limit took effect.
    expect(svg).toContain(">10</text>");
  });

  it("twinx() returns the twin and makes it the current axes", () => {
    figure({ width: 300, height: 200 });
    plot(tensor([0, 1, 2]), tensor([0, 1, 2]));
    const base = gca();
    const twin = twinx();
    expect(twin).not.toBe(base);
    expect(twin.parent).toBe(base);
    expect(gca()).toBe(twin);
    plot(tensor([0, 1, 2]), tensor([0, 100, 200]), { color: "#ff0000" });
    expect(drawablesOf(twin)).toHaveLength(1);
    expect(drawablesOf(base)).toHaveLength(1);
    sca(base);
    expect(gca()).toBe(base);
  });

  it("subplot still replaces the untouched implicit axes", () => {
    figure({ width: 200, height: 200 });
    subplot(1, 2, 1);
    expect(gcf().axesList).toHaveLength(1);
  });
});

describe("colormaps", () => {
  it("has one shared list of names, including cividis", () => {
    expect(COLORMAP_NAMES).toContain("cividis");
    for (const name of COLORMAP_NAMES) expect(() => assertColormapName(name)).not.toThrow();
    expect(() => assertColormapName("jet")).toThrow(InvalidParameterError);
    expect(() => assertColormapName(undefined)).toThrow(InvalidParameterError);
  });

  it("cividis matches the matplotlib samples", () => {
    // matplotlib 3.10: cividis(0), cividis(0.5), cividis(1) as 8-bit RGB.
    expect(applyColormap(0, "cividis")).toEqual([0, 34, 78]);
    expect(applyColormap(0.5, "cividis")).toEqual([125, 124, 120]);
    expect(applyColormap(1, "cividis")).toEqual([254, 232, 56]);
  });

  it("is accepted by heatmap, contour and contourf and rejected when unknown", () => {
    const ax = new Figure().addAxes();
    const z = tensor([
      [0, 1, 2],
      [1, 2, 3],
      [2, 3, 4],
    ]);
    const empty = tensor([]);
    expect(() => ax.heatmap(z, { colormap: "cividis" })).not.toThrow();
    expect(() => ax.contour(empty, empty, z, { colormap: "cividis" })).not.toThrow();
    expect(() => ax.contourf(empty, empty, z, { colormap: "cividis" })).not.toThrow();
    const bad = { colormap: "jet" as unknown as ColormapName };
    expect(() => ax.heatmap(z, bad)).toThrow(InvalidParameterError);
    expect(() => ax.contour(empty, empty, z, bad)).toThrow(InvalidParameterError);
    expect(() => ax.contourf(empty, empty, z, bad)).toThrow(InvalidParameterError);
    // The name is checked even when explicit colors make the colormap unused.
    expect(() => ax.contour(empty, empty, z, { ...bad, color: "#ff0000" })).toThrow(
      InvalidParameterError
    );
    expect(() => ax.contourf(empty, empty, z, { ...bad, colors: ["#ff0000"] })).toThrow(
      InvalidParameterError
    );
  });
});

describe("auto ticks", () => {
  it("never leaves fewer than two ticks inside a narrow range", () => {
    expect(generateTicks(0.4, 3.6, 2).map((t) => t.label)).toEqual(["1", "2", "3"]);
    expect(generateTicks(0.55, 0.65, 2).length).toBeGreaterThanOrEqual(2);
    expect(generateTicks(0, 7, 2).length).toBeGreaterThanOrEqual(2);
    // A request that already yields enough ticks is unchanged.
    expect(generateTicks(0, 100, 5).map((t) => t.value)).toEqual([0, 20, 40, 60, 80, 100]);
  });

  it("shows every category position of a grouped bar chart", () => {
    figure({ width: 320, height: 240 });
    groupedBar(tensor([1, 2, 3]), [tensor([1, 2, 3]), tensor([3, 2, 1])]);
    const svg = show().svg;
    const labels = [...svg.matchAll(/class="tick-label tick-label-x"[^>]*>([^<]*)</g)].map(
      (m) => m[1]
    );
    expect(labels).toEqual(["1", "2", "3"]);
  });
});

describe("filled areas", () => {
  it("kdeplot with fill shades the area instead of drawing only an outline", () => {
    figure({ width: 200, height: 150 });
    kdeplot(tensor([1, 2, 2.5, 3, 4]), { fill: true, color: "#ff0000" });
    const svg = show().svg;
    expect(svg).toMatch(/<path d="M[^"]*Z" fill="#ff0000" fill-rule="evenodd"/);
  });
});

describe("PDF fills", () => {
  it("reads fill and stroke whatever the attribute order", () => {
    const svg =
      '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">' +
      '<path stroke="#0000ff" stroke-width="3" d="M10 10 L90 10 L90 90 Z" fill="#ff0000"/>' +
      '<polyline stroke-width="2" points="10,10 50,50 90,10" stroke="#00ff00" fill="none"/>' +
      "</svg>";
    const pdf = pdfText(svgToPdf(svg, 100, 100));
    expect(pdf).toContain("1 0 0 rg");
    expect(pdf).toContain("0 0 1 RG");
    expect(pdf).toContain("0 1 0 RG");
    // The polyline is not filled.
    expect(pdf.match(/ rg\n/g)?.length).toBe(2); // page background and the red path
  });

  it("fills an area plot in the PDF output", () => {
    const fig = new Figure({ width: 200, height: 150 });
    fig.addAxes().area(tensor([0, 1, 2]), tensor([1, 3, 2]), { color: "#ff0000" });
    const pdf = pdfText(fig.renderPDF().bytes);
    expect(pdf).toContain("1 0 0 rg");
    expect(pdf).toMatch(/f\*/);
  });
});

describe("bar width options", () => {
  it("Bar2D takes barWidth and groupedBar no longer needs a subclass", () => {
    const ax = new Figure().addAxes();
    const wide = ax.bar(tensor([1, 2]), tensor([1, 2]), { barWidth: 0.4 });
    expect(wide.barWidth).toBe(0.4);
    expect(wide.getDataRange()?.xmin).toBeCloseTo(0.8, 12);
    expect(ax.bar(tensor([1]), tensor([1])).barWidth).toBe(0.8);
    groupedBar(tensor([1, 2]), [tensor([1, 2]), tensor([2, 1]), tensor([3, 3])]);
    const grouped = drawablesOf(gca()).filter((d) => d.kind === "bar");
    expect(grouped).toHaveLength(3);
    for (const d of grouped) expect(d.barWidth).toBeCloseTo(0.8 / 3, 12);
    expect(grouped[0]?.constructor.name).toBe("Bar2D");
  });

  it("rejects a non-positive or non-finite bar width", () => {
    const ax = new Figure().addAxes();
    for (const barWidth of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => ax.bar(tensor([1]), tensor([1]), { barWidth })).toThrow(InvalidParameterError);
    }
  });

  it("HorizontalBar2D takes barHeight", () => {
    const ax = new Figure({ width: 200, height: 200 }).addAxes();
    const bars = ax.barh(tensor([0, 1]), tensor([3, 5]), { barHeight: 1 });
    expect(bars.barHeight).toBe(1);
    const range = bars.getDataRange();
    expect(range?.ymin).toBeCloseTo(-0.5, 12);
    expect(range?.ymax).toBeCloseTo(1.5, 12);
    expect(ax.barh(tensor([0]), tensor([1])).barHeight).toBe(0.8);
    for (const barHeight of [0, -2, Number.NaN]) {
      expect(() => ax.barh(tensor([0]), tensor([1]), { barHeight })).toThrow(InvalidParameterError);
    }
  });
});

describe("log axes", () => {
  it("clips non-positive values off screen instead of drawing them at 0.1", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([0, 10, 100]));
    ax.setYScale("log");
    const svg = fig.renderSVG().svg;
    const points = /<polyline[^>]*points="([^"]+)"/.exec(svg)?.[1]?.split(" ") ?? [];
    expect(points).toHaveLength(3);
    const ys = points.map((p) => Number(p.split(",")[1]));
    // The viewport spans y = 50 .. 250; the zero sample is far below it, not at 0.1.
    expect(ys[0]).toBeGreaterThan(1000);
    expect(ys[1]).toBeGreaterThan(50);
    expect(ys[1]).toBeLessThan(250);
    expect(svg).not.toContain("NaN");
    expect(svg).not.toContain("Infinity");
  });

  it("starts the auto range at the smallest positive value of lines and scatters", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([0, 10, 1000]));
    ax.setYScale("log");
    const svg = fig.renderSVG().svg;
    // Range is 10 .. 1000 padded by 5% of two decades, so no tick below 10 appears.
    expect(svg).toContain(">10</text>");
    expect(svg).toContain(">100</text>");
    expect(svg).toContain(">1000</text>");
    expect(svg).not.toContain('>1</text>\n<line class="y-tick"');
    const yTickLabels = [...svg.matchAll(/tick-label-y"[^>]*>([^<]*)</g)].map((m) => m[1]);
    expect(yTickLabels).toEqual(["10", "100", "1000"]);
  });

  it("keeps the fallback for drawables that only report their minimum", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.bar(tensor([1, 2]), tensor([100, 1000]));
    ax.setYScale("log");
    const labels = [...fig.renderSVG().svg.matchAll(/tick-label-y"[^>]*>([^<]*)</g)].map(
      (m) => m[1]
    );
    expect(labels).toContain("0.1");
    expect(labels).toContain("1000");
  });
});

describe("themes reach figures and axes", () => {
  it("leaves the default output unchanged", () => {
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.setTitle("t");
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('<rect x="0" y="0" width="200" height="150" fill="#ffffff" />');
    expect(svg).toContain('fill="#ffffff" stroke="#000" />');
    expect(svg).toContain('font-size="14" font-weight="bold" fill="#000">t</text>');
    expect(svg).toContain('font-size="10" fill="#000">');
    expect(svg).not.toContain('stroke="#cccccc"');
  });

  it("applies the dark theme to a new figure and axes", () => {
    setTheme("dark");
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]), { label: "a" });
    ax.setTitle("t");
    ax.legend();
    const svg = fig.renderSVG().svg;
    expect(fig.background).toBe("#1e1e1e");
    expect(svg).toContain('<rect x="0" y="0" width="200" height="150" fill="#1e1e1e" />');
    expect(svg).toContain('fill="#2d2d2d" stroke="#e6e6e6" />');
    expect(svg).toContain('stroke="#444444" stroke-width="0.5"'); // grid is on in this theme
    expect(svg).toContain('font-weight="bold" fill="#e6e6e6">t</text>');
    expect(svg).toContain('class="legend-box"');
    expect(svg).not.toContain('fill="#000"');
    expect(svg).not.toContain('stroke="#000"');
  });

  it("explicit options win over the theme", () => {
    setTheme("dark");
    const fig = new Figure({ width: 100, height: 80, background: "#112233" });
    const ax = fig.addAxes({ facecolor: "#445566" });
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.grid(false);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('fill="#112233"');
    expect(svg).toContain('fill="#445566"');
    expect(svg).not.toContain('stroke="#444444"');
  });

  it("does not restyle figures created before the theme was set", () => {
    const fig = new Figure({ width: 100, height: 80 });
    fig.addAxes().plot(tensor([0, 1]), tensor([0, 1]));
    setTheme("dark");
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('fill="#ffffff"');
    expect(svg).not.toContain("#1e1e1e");
  });

  it("scales the text sizes with the theme font size", () => {
    setTheme("presentation");
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    ax.setTitle("t");
    ax.setXLabel("x");
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('font-size="14" fill="#111111"'); // tick labels: fontSize - 2
    expect(svg).toContain('font-size="18" font-weight="bold"'); // title: fontSize + 2
    expect(svg).toContain('font-size="16" fill="#111111">x</text>'); // axis label
    setTheme("paper");
    const small = new Figure({ width: 300, height: 200 });
    small.addAxes().plot(tensor([0, 1]), tensor([0, 1]));
    expect(small.renderSVG().svg).toContain('font-size="8" fill="#000"');
  });

  it("uses the theme colors in PNG output too", async () => {
    setTheme("dark");
    const fig = new Figure({ width: 60, height: 40 });
    fig.addAxes().plot(tensor([0, 1]), tensor([0, 1]));
    const dark = await fig.renderPNG();
    resetTheme();
    const plain = new Figure({ width: 60, height: 40 });
    plain.addAxes().plot(tensor([0, 1]), tensor([0, 1]));
    const light = await plain.renderPNG();
    expect(Buffer.from(dark.bytes).equals(Buffer.from(light.bytes))).toBe(false);
  });

  it("does not let an untouched themed axes block subplot()", () => {
    setTheme("dark");
    figure({ width: 200, height: 200 });
    subplot(1, 2, 1);
    expect(gcf().axesList).toHaveLength(1);
  });

  it("gives a twin axes no grid of its own", () => {
    setTheme("dark");
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1, 2]), tensor([0, 1, 2]));
    const before = fig.renderSVG().svg.match(/stroke-width="0.5"/g)?.length ?? 0;
    const twin = ax.twinx();
    twin.plot(tensor([0, 1, 2]), tensor([0, 10, 20]));
    const after = fig.renderSVG().svg.match(/stroke-width="0.5"/g)?.length ?? 0;
    expect(after).toBe(before);
  });
});

describe("annotation alignment", () => {
  it("writes text-anchor and dominant-baseline for ha and va", () => {
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 2]), tensor([0, 2]));
    ax.text(1, 1, "mid", { ha: "center", va: "center" });
    ax.text(1, 1.5, "right", { ha: "right", va: "top" });
    ax.annotate("plain", 0.5, 0.5);
    const svg = fig.renderSVG().svg;
    expect(svg).toMatch(/<text[^>]*text-anchor="middle" dominant-baseline="middle"[^>]*>mid</);
    expect(svg).toMatch(/<text[^>]*text-anchor="end" dominant-baseline="hanging"[^>]*>right</);
    expect(svg).toMatch(/<text x="[\d.]+" y="[\d.]+" font-size="10" fill="#000000">plain</);
  });

  it("rejects an unknown alignment", () => {
    const ax = new Figure().addAxes();
    expect(() => ax.text(0, 0, "a", { ha: "middle" as unknown as "left" })).toThrow(
      InvalidParameterError
    );
    expect(() => ax.text(0, 0, "a", { va: "baseline" as unknown as "top" })).toThrow(
      InvalidParameterError
    );
  });

  it("draws annotations in PNG output and honours the alignment there", async () => {
    const make = async (options: TextOptions | null): Promise<Buffer> => {
      const fig = new Figure({ width: 120, height: 100 });
      const ax = fig.addAxes();
      ax.plot(tensor([0, 2]), tensor([0, 2]));
      if (options) ax.text(1, 1, "LABEL", options);
      return Buffer.from((await fig.renderPNG()).bytes);
    };
    const none = await make(null);
    const left = await make({});
    const centered = await make({ ha: "center", va: "center" });
    expect(left.equals(none)).toBe(false);
    expect(centered.equals(left)).toBe(false);
  });

  it("is also drawn in PDF output", () => {
    const fig = new Figure({ width: 120, height: 100 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 2]), tensor([0, 2]));
    ax.text(1, 1, "CELL", { ha: "center", va: "center" });
    expect(pdfText(fig.renderPDF().bytes)).toContain("(CELL) Tj");
  });
});

describe("heatmap origin", () => {
  const z = tensor([
    [0, 0],
    [10, 10],
  ]);

  function topRowColor(origin: "lower" | "upper" | undefined): string {
    const fig = new Figure({ width: 100, height: 100 });
    const ax = fig.addAxes({ padding: 10 });
    ax.heatmap(z, origin ? { origin, colormap: "grayscale" } : { colormap: "grayscale" });
    const svg = fig.renderSVG().svg;
    const rects = [...svg.matchAll(/<rect x="[\d.]+" y="([\d.]+)"[^>]*fill="(rgb\([^)]*\))"/g)];
    const top = rects.reduce((a, b) => (Number(b[1]) < Number(a[1]) ? b : a));
    return top[2] ?? "";
  }

  it("draws row 0 at the bottom by default and at the top for origin upper", () => {
    // Row 0 holds the minimum (black), row 1 the maximum (white).
    expect(topRowColor(undefined)).toBe("rgb(255,255,255)");
    expect(topRowColor("lower")).toBe("rgb(255,255,255)");
    expect(topRowColor("upper")).toBe("rgb(0,0,0)");
  });

  it("applies to imshow and rejects unknown values", () => {
    const ax = new Figure().addAxes();
    expect(ax.imshow(z, { origin: "upper" }).origin).toBe("upper");
    expect(() => ax.heatmap(z, { origin: "center" as unknown as "upper" })).toThrow(
      InvalidParameterError
    );
  });

  it("keeps plotConfusionMatrix drawing class 0 in the top row", async () => {
    const { plotConfusionMatrix } = await import("../../src/plot");
    figure({ width: 100, height: 100 });
    plotConfusionMatrix(
      tensor([
        [0, 0],
        [10, 10],
      ]),
      ["a", "b"],
      {
        colormap: "grayscale",
        origin: "lower",
      }
    );
    const heat = drawablesOf(gca())[0];
    expect(heat?.origin).toBe("upper");
    const svg = show().svg;
    const labelY = (label: string): number => {
      const m = new RegExp(`<text[^>]* y="([\\d.]+)"[^>]*text-anchor="end"[^>]*>${label}</text>`);
      return Number(m.exec(svg)?.[1]);
    };
    expect(labelY("a")).toBeLessThan(labelY("b"));
  });
});

describe("contours", () => {
  it("does not offset the data maximum when the grid has NaN", () => {
    const z = tensor([
      [0.25, 0.5],
      [Number.NaN, 0.25],
    ]);
    const nanGrid = buildContourGrid(tensor([]), tensor([]), z);
    // The old code returned 0.5 + Number.EPSILON here.
    expect(nanGrid.dataMax).toBe(0.5);
    expect(nanGrid.dataMin).toBe(0.25);
  });

  it("strokes filled contour triangles in their fill color to avoid seams", () => {
    const fig = new Figure({ width: 100, height: 100 });
    const ax = fig.addAxes();
    const empty = tensor([]);
    ax.contourf(
      empty,
      empty,
      tensor([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
      ])
    );
    const paths = fig.renderSVG().svg.match(/<path d="M [^"]*" fill="[^"]*"[^>]*>/g) ?? [];
    expect(paths.length).toBeGreaterThan(0);
    for (const p of paths) {
      const fill = /fill="([^"]*)"/.exec(p)?.[1];
      expect(p).toContain(`stroke="${fill}"`);
    }
  });

  it("marks heatmap cells with crisp edges", () => {
    const fig = new Figure({ width: 100, height: 100 });
    fig.addAxes().heatmap(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    const cells = fig.renderSVG().svg.match(/<rect [^>]*fill="rgb\([^>]*>/g) ?? [];
    expect(cells).toHaveLength(4);
    for (const c of cells) expect(c).toContain('shape-rendering="crispEdges"');
  });
});

describe("3D axes", () => {
  const g = [new Float64Array([0, 1]), new Float64Array([0, 1])];
  const gy = [new Float64Array([0, 0]), new Float64Array([1, 1])];
  const gz = [new Float64Array([0, 1]), new Float64Array([1, 2])];

  it("shows no tick labels or grid for projected 3D plots", () => {
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    ax.grid(true);
    ax.surface(g, gy, gz);
    ax.scatter3d(Float64Array.from([0, 1]), Float64Array.from([0, 1]), Float64Array.from([0, 1]));
    const svg = fig.renderSVG().svg;
    expect(svg).not.toContain("tick-label");
    expect(svg).not.toContain('class="x-tick"');
    expect(svg).not.toContain('stroke="#cccccc"');
  });

  it("still shows ticks once a 2D plot shares the axes", () => {
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    ax.surface(g, gy, gz);
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    expect(fig.renderSVG().svg).toContain("tick-label");
  });

  it("shows no tick labels in PNG output either", async () => {
    const make = async (flat: boolean): Promise<Buffer> => {
      const fig = new Figure({ width: 200, height: 150 });
      const ax = fig.addAxes();
      if (flat) ax.plot(tensor([0, 1]), tensor([0, 1]));
      else ax.wireframe(g, gy, gz);
      return Buffer.from((await fig.renderPNG()).bytes);
    };
    expect((await make(true)).length).toBeGreaterThan(0);
    expect((await make(false)).length).toBeGreaterThan(0);
  });
});

describe("violinplot options", () => {
  const data = tensor([1, 2, 4, 7, 11, 16]);

  it("places the violin at position with the given width", () => {
    const ax = new Figure().addAxes();
    const v = ax.violinplot(data, { position: 3, width: 0.5 });
    expect(v.position).toBe(3);
    expect(v.violinWidth).toBe(0.5);
    const range = v.getDataRange();
    expect(range?.xmin).toBeCloseTo(2.75, 12);
    expect(range?.xmax).toBeCloseTo(3.25, 12);
    const defaults = ax.violinplot(data);
    expect(defaults.position).toBe(1);
    expect(defaults.violinWidth).toBe(0.8);
  });

  it("supports Scott's rule and a numeric bandwidth, matching SciPy", () => {
    // scipy gaussian_kde(bw_method="scott").factor * std(ddof=1) = 4.036697087440496
    expect(scottBandwidth([1, 2, 4, 7, 11, 16])).toBeCloseTo(4.036697087440496, 10);
    const ax = new Figure().addAxes();
    const scott = ax.violinplot(data, { bandwidth: "scott" });
    const expected = kernelDensityEstimation(
      [1, 2, 4, 7, 11, 16],
      scott.kdePoints,
      4.036697087440496
    );
    for (let i = 0; i < expected.length; i++) {
      expect(scott.kdeValues[i]).toBeCloseTo(expected[i] ?? 0, 12);
    }
    const silverman = ax.violinplot(data);
    const wide = ax.violinplot(data, { bandwidth: 8 });
    // A larger bandwidth gives a flatter density.
    expect(Math.max(...wide.kdeValues)).toBeLessThan(Math.max(...silverman.kdeValues));
    // Scott's bandwidth (4.04) is narrower than Silverman's (4.28), so its peak is higher.
    expect(Math.max(...scott.kdeValues)).toBeGreaterThan(Math.max(...silverman.kdeValues) * 0.999);
  });

  it("validates the options", () => {
    const ax = new Figure().addAxes();
    expect(() => ax.violinplot(data, { width: 0 })).toThrow(InvalidParameterError);
    expect(() => ax.violinplot(data, { position: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => ax.violinplot(data, { bandwidth: -1 })).toThrow(InvalidParameterError);
    expect(() => ax.violinplot(data, { bandwidth: "bad" as unknown as "scott" })).toThrow(
      InvalidParameterError
    );
  });
});

describe("waterfall tick labels", () => {
  it("uses the category names as x tick labels", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.waterfall(["Start", "Sales", "Costs", "End"], tensor([100, 50, -30, 120]));
    const labels = [...fig.renderSVG().svg.matchAll(/tick-label-x"[^>]*>([^<]*)</g)].map(
      (m) => m[1]
    );
    expect(labels).toEqual(["Start", "Sales", "Costs", "End"]);
  });

  it("lets setXTicks override the category labels and ignores empty input", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.waterfall(["a", "b"], tensor([1, 2]));
    ax.setXTicks([0, 1], ["first", "second"]);
    expect(fig.renderSVG().svg).toContain(">second</text>");
    expect(() => new Figure().addAxes().waterfall([], tensor([]))).not.toThrow();
  });
});

describe("boxplot whiskers", () => {
  it("ends a whisker at the quartile when the in-fence value is inside the box", () => {
    const ax = new Figure().addAxes();
    // matplotlib boxplot_stats([1, 2, 3, 100]): q1 = 1.75, q3 = 27.25, whishi = 27.25.
    const box = ax.boxplot(tensor([1, 2, 3, 100]));
    expect(box.q3).toBeCloseTo(27.25, 12);
    expect(box.whiskerHigh).toBeCloseTo(27.25, 12);
    expect(box.whiskerLow).toBe(1);
    expect(box.outliers).toEqual([100]);
  });

  it("accepts typed input with non-finite values", () => {
    const ax = new Figure().addAxes();
    const box = ax.boxplot(tensor([3, Number.NaN, 1, 2, 4, 5]));
    expect(box.median).toBe(3);
    expect(box.q1).toBe(2);
    expect(box.q3).toBe(4);
  });
});
