import { stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { tensor, transpose } from "../src/ndarray";
import { Tensor } from "../src/ndarray/tensor";
import { Figure, figure, plotConfusionMatrix, saveFig, show, subplot } from "../src/plot";

describe("Plot - Edge Cases", () => {
  it("escapes special characters in titles and labels", () => {
    const fig = new Figure({ width: 240, height: 180 });
    const ax = fig.addAxes();
    ax.setTitle(`A&B <test> "plot" 'title'`);
    ax.setXLabel(`X & "axis"`);
    ax.setYLabel(`Y <axis>`);
    ax.plot(tensor([0, 1]), tensor([0, 1]));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("A&amp;B");
    expect(svg).toContain("&lt;test&gt;");
    expect(svg).toContain("&quot;plot&quot;");
    expect(svg).toContain("&apos;title&apos;");
    expect(svg).toContain("X &amp; &quot;axis&quot;");
    expect(svg).toContain("Y &lt;axis&gt;");
  });

  it("falls back to default colors when given empty or whitespace", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    const line = ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "   " });
    const bar = ax.bar(tensor([1, 2]), tensor([3, 4]), { edgecolor: "" });
    expect(line.color).toBe("#1f77b4");
    expect(bar.edgecolor).toBe("#000000");
  });

  it("handles string tensors by throwing a clear error", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    const x = Tensor.fromStringArray({ data: ["a", "b"], shape: [2] });
    const y = Tensor.fromStringArray({ data: ["c", "d"], shape: [2] });
    expect(() => ax.plot(x, y)).toThrow(/string tensors/);
  });

  it("covers computeAutoRange fallback when all data are invalid", () => {
    const fig = new Figure({ width: 200, height: 160 });
    const ax = fig.addAxes();
    ax.plot(tensor([NaN, NaN]), tensor([Infinity, -Infinity]));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<svg");
  });

  it("keeps constant-x data within the viewport", () => {
    const fig = new Figure({ width: 200, height: 120 });
    const ax = fig.addAxes();
    ax.plot(tensor([5, 5, 5]), tensor([0, 1, 2]));
    const svg = fig.renderSVG().svg;
    const match = svg.match(/points="([^"]+)"/);
    expect(match).toBeTruthy();
    const points = (match?.[1] ?? "").trim().split(/\s+/);
    for (const point of points) {
      const [xStr] = point.split(",");
      const x = Number.parseFloat(xStr ?? "");
      expect(Number.isFinite(x)).toBe(true);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(fig.width);
    }
  });

  it("rejects histogram with no finite values", () => {
    const fig = new Figure({ width: 220, height: 180 });
    const ax = fig.addAxes();
    expect(() => ax.hist(tensor([NaN, Infinity, -Infinity]), 4)).toThrow(
      /histogram data must contain at least one finite value/i
    );
  });

  it("uses non-zero bin width for constant histograms", () => {
    const fig = new Figure({ width: 220, height: 180 });
    const ax = fig.addAxes();
    const hist = ax.hist(tensor([5, 5, 5]), 1);
    expect(hist.binWidth).toBeGreaterThan(0);
    const range = hist.getDataRange();
    expect(range).toBeTruthy();
    if (range) {
      expect(range.xmax).toBeGreaterThan(range.xmin);
    }
  });

  it("validates histogram bin count", () => {
    const fig = new Figure({ width: 220, height: 180 });
    const ax = fig.addAxes();
    expect(() => ax.hist(tensor([1, 2, 3]), 0)).toThrow(/positive integer/);
    expect(() => ax.hist(tensor([1, 2, 3]), -2)).toThrow(/positive integer/);
  });

  it("allows histogram options.bins to override the bins argument", () => {
    const fig = new Figure({ width: 220, height: 180 });
    const ax = fig.addAxes();
    const hist = ax.hist(tensor([1, 2, 3, 4]), 2, { bins: 4 });
    expect(hist.bins.length).toBe(4);
  });

  it("includes negative values in bar ranges", () => {
    const fig = new Figure({ width: 220, height: 180 });
    const ax = fig.addAxes();
    const barPlot = ax.bar(tensor([0, 1]), tensor([-5, 10]));
    const barRange = barPlot.getDataRange();
    expect(barRange).toBeTruthy();
    if (barRange) {
      expect(barRange.ymin).toBeLessThan(0);
      expect(barRange.ymax).toBeGreaterThan(0);
    }
    const barhPlot = ax.barh(tensor([0, 1]), tensor([-2, 3]));
    const barhRange = barhPlot.getDataRange();
    expect(barhRange).toBeTruthy();
    if (barhRange) {
      expect(barhRange.xmin).toBeLessThan(0);
      expect(barhRange.xmax).toBeGreaterThan(0);
    }
  });

  it("renders heatmap with constant values (zero range)", () => {
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    ax.heatmap(
      tensor([
        [2, 2],
        [2, 2],
      ])
    );
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });

  it("validates heatmap vmin/vmax inputs", () => {
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    expect(() =>
      ax.heatmap(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        { vmin: NaN }
      )
    ).toThrow(/vmin must be finite/);
    expect(() =>
      ax.heatmap(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        { vmax: Infinity }
      )
    ).toThrow(/vmax must be finite/);
    expect(() =>
      ax.heatmap(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        { vmin: 5, vmax: 1 }
      )
    ).toThrow(/vmin must be <= vmax/);
  });

  it("validates marker size and linewidth", () => {
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    expect(() => ax.plot(tensor([0, 1]), tensor([0, 1]), { linewidth: 0 })).toThrow(
      /positive number/
    );
    expect(() => ax.scatter(tensor([0, 1]), tensor([0, 1]), { size: -1 })).toThrow(
      /positive number/
    );
  });

  it("validates padding and viewport sizing", () => {
    const fig = new Figure({ width: 200, height: 150 });
    expect(() => fig.addAxes({ padding: -1 })).toThrow(/non-negative/);
    expect(() =>
      fig.addAxes({
        viewport: { x: 0, y: 0, width: 20, height: 20 },
        padding: 15,
      })
    ).toThrow();
  });

  it("validates pie label length", () => {
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    expect(() => ax.pie(tensor([1, 2]), ["A"])).toThrow(/labels length/);
  });

  it("handles non-contiguous 2D tensors in heatmap", () => {
    const fig = new Figure({ width: 240, height: 180 });
    const ax = fig.addAxes();
    const base = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const transposed = transpose(base);
    ax.heatmap(transposed);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });

  it("renders PNG with invalid and 8-digit hex colors", async () => {
    const fig = new Figure({
      width: 80,
      height: 60,
      background: "not-a-color",
    });
    fig.addAxes().plot(tensor([0, 1]), tensor([0, 1]), { color: "#ff000080" });
    const png = await fig.renderPNG();
    expect(png.bytes[0]).toBe(137);
  });

  it("validates subplot inputs and ranges", () => {
    expect(() => subplot(0, 1, 1)).toThrow(/rows must be a positive integer/);
    expect(() => subplot(1, 0, 1)).toThrow(/cols must be a positive integer/);
    expect(() => subplot(1, 2, 0)).toThrow(/index must be in/);
    expect(() => subplot(1, 2, 3)).toThrow(/index must be in/);
  });

  it("validates confusion matrix label length", () => {
    const cm = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => plotConfusionMatrix(cm, ["A"])).toThrow(/labels length/);
  });

  it("respects heatmap extent bounds", () => {
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    const hm = ax.heatmap(
      tensor([
        [1, 2],
        [3, 4],
      ]),
      { extent: { xmin: -1, xmax: 1, ymin: -2, ymax: 2 } }
    );
    const range = hm.getDataRange();
    expect(range).toBeTruthy();
    if (!range) return;
    expect(range.xmin).toBe(-1);
    expect(range.xmax).toBe(1);
    expect(range.ymin).toBe(-2);
    expect(range.ymax).toBe(2);
  });

  it("respects contour extent bounds", () => {
    const fig = new Figure({ width: 200, height: 150 });
    const ax = fig.addAxes();
    const contour = ax.contour(
      tensor([]),
      tensor([]),
      tensor([
        [1, 2],
        [3, 4],
      ]),
      { extent: { xmin: 10, xmax: 20, ymin: 30, ymax: 40 }, levels: 3 }
    );
    const range = contour.getDataRange();
    expect(range).toBeTruthy();
    if (!range) return;
    expect(range.xmin).toBe(10);
    expect(range.xmax).toBe(20);
    expect(range.ymin).toBe(30);
    expect(range.ymax).toBe(40);
  });

  it("renders via show() in PNG mode", async () => {
    figure({ width: 120, height: 90 });
    const rendered = await show({ format: "png" });
    expect(rendered.kind).toBe("png");
    if (rendered.kind !== "png") return;
    expect(rendered.bytes[0]).toBe(137);
  });

  it("saveFig supports PNG and rejects unknown extensions", async () => {
    const fig = new Figure({ width: 160, height: 120 });
    fig.addAxes().plot(tensor([0, 1]), tensor([1, 0]));
    await expect(saveFig("/tmp/test.txt", { figure: fig })).rejects.toThrow(
      /does not match format/
    );
    const pngPath = join(tmpdir(), `deepbox-plot-${Date.now()}.png`);
    await saveFig(pngPath, { figure: fig });
    const info = await stat(pngPath);
    expect(info.size).toBeGreaterThan(0);
  });
});
