import { stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { figure, gca, plot, saveFig, scatter, show, subplot } from "../src/plot";

describe("deepbox/plot - public APIs", () => {
  it("should create a figure and return current axes", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    expect(fig.axesList.length).toBeGreaterThan(0);
    expect(ax).toBe(fig.axesList[0]);
  });

  it("should create subplots and set current axes", () => {
    figure({ width: 300, height: 200 });
    const ax1 = subplot(1, 2, 1);
    const ax2 = subplot(1, 2, 2);
    expect(ax1).not.toBe(ax2);
    expect(gca()).toBe(ax2);
  });

  it("should render to SVG via show()", () => {
    figure({ width: 200, height: 150 });
    const x = tensor([1, 2, 3]);
    const y = tensor([2, 4, 6]);
    plot(x, y);
    scatter(x, y);
    const rendered = show({ format: "svg" });
    if (!("kind" in rendered)) throw new Error("Expected synchronous SVG result");
    expect(rendered.kind).toBe("svg");
    expect(rendered.svg).toContain("<svg");
  });

  it("should save SVG to disk", async () => {
    figure({ width: 200, height: 150 });
    const x = tensor([1, 2, 3]);
    const y = tensor([1, 4, 9]);
    plot(x, y);
    const path = join(tmpdir(), `deepbox-plot-${Date.now()}.svg`);
    await saveFig(path);
    const info = await stat(path);
    expect(info.size).toBeGreaterThan(0);
  });
});
