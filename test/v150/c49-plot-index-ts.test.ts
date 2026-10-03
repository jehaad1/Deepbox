import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { InvalidParameterError, ShapeError } from "../../src/core";
import { tensor } from "../../src/ndarray";
import type { Axes } from "../../src/plot";
import {
  createInteractivePlot,
  figure,
  gca,
  getTheme,
  hist,
  InteractivePlot,
  jointplot,
  kdeplot,
  pairplot,
  plotConfusionMatrix,
  plotDecisionBoundary,
  plotFeatureImportance,
  plotLearningCurve,
  plotResiduals,
  plotRocCurve,
  plotValidationCurve,
  resetTheme,
  saveFig,
  scatter3d,
  setTheme,
  show,
  surface,
  wireframe,
} from "../../src/plot";

type AnyDrawable = {
  readonly kind: string;
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly color: string;
};

function drawablesOf(ax: Axes): AnyDrawable[] {
  return (ax as unknown as { drawables: AnyDrawable[] }).drawables;
}

/** Y pixel coordinate of the y-axis tick label (end-anchored text) whose content is `label`. */
function textY(svg: string, label: string): number {
  const re = new RegExp(`<text[^>]* y="([0-9.]+)"[^>]*text-anchor="end"[^>]*>${label}</text>`);
  const m = re.exec(svg);
  if (!m) throw new Error(`no <text> element with content ${label}`);
  return Number(m[1]);
}

beforeEach(() => {
  figure();
  resetTheme();
});

afterEach(() => {
  resetTheme();
});

describe("saveFig / show validation", () => {
  it("only looks for the extension in the file name, not in directory names", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deepbox-c49-"));
    try {
      const sub = join(dir, "out.v2");
      const { mkdir } = await import("node:fs/promises");
      await mkdir(sub);
      figure({ width: 100, height: 80 });
      gca().plot(tensor([0, 1]), tensor([0, 1]));
      // Before the fix "v2/plot" was parsed as the extension and the call threw.
      const target = join(sub, "plot");
      await saveFig(target);
      const text = await readFile(target, "utf-8");
      expect(text).toContain("<svg");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("rejects unknown formats", async () => {
    figure({ width: 100, height: 80 });
    await expect(
      saveFig(join(tmpdir(), "x.svg"), { format: "jpg" as unknown as "svg" })
    ).rejects.toThrow(InvalidParameterError);
    expect(() => show({ format: "pdf" as unknown as "svg" })).toThrow(InvalidParameterError);
  });
});

describe("hist wrapper", () => {
  it("merges the options object passed as the second argument with the third", () => {
    const fig = figure({ width: 200, height: 150 });
    hist(tensor([1, 2, 3, 4, 5, 6]), { bins: 3 }, { color: "#ff0000" });
    expect(fig.renderSVG().svg).toContain("#ff0000");
    const bars = drawablesOf(gca())[0] as unknown as { counts: Float64Array };
    expect(Array.from(bars.counts)).toEqual([2, 2, 2]);
  });
});

describe("plotConfusionMatrix", () => {
  it("draws class 0 in the top row and labels the ticks", () => {
    const fig = figure({ width: 320, height: 240 });
    plotConfusionMatrix(
      tensor([
        [5, 1],
        [2, 7],
      ]),
      ["Alpha", "Beta"]
    );
    const svg = fig.renderSVG().svg;
    expect(textY(svg, "Alpha")).toBeLessThan(textY(svg, "Beta"));
  });

  it("uses class indices as tick labels when no labels are given", () => {
    const fig = figure({ width: 320, height: 240 });
    plotConfusionMatrix(
      tensor([
        [5, 1],
        [2, 7],
      ])
    );
    const svg = fig.renderSVG().svg;
    expect(textY(svg, "0")).toBeLessThan(textY(svg, "1"));
  });

  it("validates labels before drawing anything", () => {
    figure();
    const ax = gca();
    expect(() =>
      plotConfusionMatrix(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        ["A"]
      )
    ).toThrow(/labels length/);
    expect(drawablesOf(ax)).toHaveLength(0);
  });

  it("rejects non-2D input with a ShapeError", () => {
    expect(() => plotConfusionMatrix(tensor([1, 2, 3]))).toThrow(ShapeError);
  });
});

describe("plotDecisionBoundary labels", () => {
  const X = tensor([
    [0, 0],
    [0.2, 0.1],
    [1, 1],
    [0.9, 1.1],
  ]);
  const predictByX = (t: {
    at: (...i: number[]) => unknown;
    shape: readonly number[];
  }): number[] => {
    const out: number[] = [];
    for (let i = 0; i < (t.shape[0] ?? 0); i++) out.push((t.at(i, 0) as number) > 0.5 ? 1 : 0);
    return out;
  };

  it("treats int64 labels and numeric predictions as the same classes", () => {
    const fig = figure({ width: 200, height: 160 });
    const y = tensor([0, 0, 1, 1], { dtype: "int64" });
    plotDecisionBoundary(X, y, { predict: (g) => tensor(predictByX(g)) });
    const svg = fig.renderSVG().svg;
    const fills = new Set(svg.match(/fill="rgb\(\d+,\d+,\d+\)"/g));
    // Two classes -> only the two extreme grayscale shades in the background.
    expect(fills).toEqual(new Set(['fill="rgb(0,0,0)"', 'fill="rgb(255,255,255)"']));
  });

  it("accepts predictions returned as an [n, 1] column of labels", () => {
    const fig = figure({ width: 200, height: 160 });
    const y = tensor([0, 0, 1, 1]);
    plotDecisionBoundary(X, y, {
      predict: (g) => tensor(predictByX(g)).reshape([g.shape[0] ?? 0, 1]),
    });
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('fill="rgb(0,0,0)"');
    expect(svg).toContain('fill="rgb(255,255,255)"');
  });
});

describe("kdeplot", () => {
  it("matches scipy.stats.gaussian_kde and extends three bandwidths past the data", () => {
    figure();
    kdeplot(tensor([1, 2, 3, 4, 5]), { gridSize: 5 });
    const line = drawablesOf(gca())[0];
    if (!line) throw new Error("no line drawn");
    // scipy: bw = 5**-0.2 * std(ddof=1) = 1.1459772694961639
    const expectedX = [
      -2.4379318084884916, 0.281034095755754, 2.9999999999999996, 5.718965904244245,
      8.437931808488491,
    ];
    const expectedY = [
      0.000812932469762659, 0.08433670468105155, 0.19514935243764917, 0.08433670468105162,
      0.0008129324697626592,
    ];
    for (let i = 0; i < 5; i++) {
      expect(line.x[i]).toBeCloseTo(expectedX[i] ?? 0, 9);
      expect(line.y[i]).toBeCloseTo(expectedY[i] ?? 0, 9);
    }
  });

  it("accepts int64 data and rejects non-1D data", () => {
    figure();
    expect(() => kdeplot(tensor([1, 2, 3], { dtype: "int64" }))).not.toThrow();
    expect(() =>
      kdeplot(
        tensor([
          [1, 2],
          [3, 4],
        ])
      )
    ).toThrow(ShapeError);
  });

  it("validates gridSize", () => {
    figure();
    expect(() => kdeplot(tensor([1, 2, 3]), { gridSize: 1 })).toThrow(InvalidParameterError);
    expect(() => kdeplot(tensor([1, 2, 3]), { gridSize: 2.5 })).toThrow(InvalidParameterError);
  });
});

describe("plotResiduals / plotFeatureImportance", () => {
  it("handles int64 inputs instead of silently dropping them", () => {
    figure();
    plotResiduals(tensor([3, 5, 7], { dtype: "int64" }), tensor([2, 6, 7], { dtype: "int64" }));
    const scatterDrawable = drawablesOf(gca())[0];
    if (!scatterDrawable) throw new Error("no scatter drawn");
    expect(Array.from(scatterDrawable.x)).toEqual([2, 6, 7]);
    expect(Array.from(scatterDrawable.y)).toEqual([1, -1, 0]);
  });

  it("rejects non-1D residual inputs", () => {
    figure();
    expect(() =>
      plotResiduals(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([
          [1, 2],
          [3, 4],
        ])
      )
    ).toThrow(ShapeError);
  });

  it("puts the most important feature on top", () => {
    const fig = figure({ width: 320, height: 240 });
    plotFeatureImportance(tensor([0.1, 0.6, 0.3]), ["Low", "High", "Mid"]);
    const svg = fig.renderSVG().svg;
    expect(textY(svg, "High")).toBeLessThan(textY(svg, "Mid"));
    expect(textY(svg, "Mid")).toBeLessThan(textY(svg, "Low"));
  });

  it("reads int64 importances and validates names and values", () => {
    figure();
    expect(() => plotFeatureImportance(tensor([3, 1, 2], { dtype: "int64" }))).not.toThrow();
    expect(() => plotFeatureImportance(tensor([1, 2, 3]), ["a", "b"])).toThrow(
      InvalidParameterError
    );
    expect(() => plotFeatureImportance(tensor([1, Number.NaN, 3]))).toThrow(/finite/);
  });
});

describe("learning and validation curves", () => {
  it("averages 2D fold scores over the fold axis", () => {
    figure();
    const sizes = tensor([10, 20, 30]);
    const train = tensor([
      [1, 0.8],
      [0.9, 0.7],
      [0.8, 0.6],
    ]);
    const val = tensor([
      [0.5, 0.7],
      [0.6, 0.8],
      [0.7, 0.9],
    ]);
    plotLearningCurve(sizes, train, val);
    const [a, b] = drawablesOf(gca());
    expect(Array.from(a?.y ?? [])).toEqual([0.9, 0.8, 0.7].map((v) => expect.closeTo(v, 6)));
    expect(Array.from(b?.y ?? [])).toEqual([0.6, 0.7, 0.8].map((v) => expect.closeTo(v, 6)));
  });

  it("keeps the validation line a different color when only color is given", () => {
    figure();
    plotValidationCurve(tensor([1, 2]), tensor([0.9, 0.8]), tensor([0.7, 0.6]), {
      color: "#112233",
    });
    const [a, b] = drawablesOf(gca());
    expect(a?.color).toBe("#112233");
    expect(b?.color).not.toBe("#112233");
  });

  it("rejects scores with more than two dimensions", () => {
    figure();
    expect(() => plotLearningCurve(tensor([1]), tensor([[[1]]]), tensor([1]))).toThrow(ShapeError);
  });
});

describe("pairplot", () => {
  it("supports kde on the diagonal", () => {
    const fig = pairplot(
      tensor([
        [1, 2],
        [2, 3.5],
        [3, 3],
        [4, 6],
      ]),
      { diagKind: "kde" }
    );
    const diag = drawablesOf(fig.axesList[0] as Axes)[0];
    expect(diag?.kind).toBe("line");
    expect(diag?.x.length).toBe(100);
    const offDiag = drawablesOf(fig.axesList[1] as Axes)[0];
    expect(offDiag?.kind).toBe("scatter");
  });

  it("keeps int64 values instead of replacing them with zero", () => {
    const fig = pairplot(
      tensor(
        [
          [1, 10],
          [2, 20],
          [3, 30],
        ],
        { dtype: "int64" }
      )
    );
    const scatterDrawable = drawablesOf(fig.axesList[1] as Axes)[0];
    expect(Array.from(scatterDrawable?.x ?? [])).toEqual([10, 20, 30]);
    expect(Array.from(scatterDrawable?.y ?? [])).toEqual([1, 2, 3]);
  });

  it("validates featureNames length and diagKind", () => {
    const data = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => pairplot(data, { featureNames: ["only"] })).toThrow(InvalidParameterError);
    expect(() => pairplot(data, { diagKind: "bad" as unknown as "hist" })).toThrow(
      InvalidParameterError
    );
  });
});

describe("jointplot", () => {
  it("counts every finite y value in the right marginal and shares axis ranges", () => {
    const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = tensor([2, 4, 3, 5, 7, 6, 8, 9, 10, 11]);
    const fig = jointplot(x, y, { bins: 5 });
    expect(fig.axesList).toHaveLength(3);
    const right = drawablesOf(fig.axesList[2] as Axes);
    expect(right).toHaveLength(5);
    let total = 0;
    for (const bar of right) total += bar.x[1] ?? 0;
    expect(total).toBe(10);
    // Bins span [min, max] of y with equal width (11 - 2) / 5.
    expect(right[0]?.y[0]).toBeCloseTo(2, 12);
    expect(right[4]?.y[2]).toBeCloseTo(11, 12);
  });

  it("validates lengths and bins", () => {
    expect(() => jointplot(tensor([1, 2, 3]), tensor([1, 2]))).toThrow(ShapeError);
    expect(() => jointplot(tensor([1, 2]), tensor([1, 2]), { bins: 0 })).toThrow(
      InvalidParameterError
    );
  });
});

describe("themes", () => {
  it("does not resolve names through the prototype chain", () => {
    expect(() => setTheme("constructor")).toThrow(/Unknown theme/);
    expect(() => setTheme("__proto__")).toThrow(/Unknown theme/);
    expect(() => setTheme("toString")).toThrow(/Unknown theme/);
  });

  it("returns frozen themes", () => {
    const theme = getTheme();
    expect(Object.isFrozen(theme)).toBe(true);
    expect(Object.isFrozen(theme.colorCycle)).toBe(true);
  });

  it("uses the active theme for default helper colors", () => {
    setTheme("dark");
    figure();
    plotRocCurve(tensor([0, 1]), tensor([0, 1]), 0.5);
    expect(drawablesOf(gca())[0]?.color).toBe("#58a6ff");
    resetTheme();
    figure();
    plotRocCurve(tensor([0, 1]), tensor([0, 1]), 0.5);
    expect(drawablesOf(gca())[0]?.color).toBe("#1f77b4");
  });
});

describe("3D wrappers", () => {
  it("accepts typed arrays and rejects ragged or mismatched grids", () => {
    figure();
    const g = [new Float64Array([0, 1]), new Float64Array([0, 1])];
    expect(() => surface(g, g, g)).not.toThrow();
    expect(() => wireframe(g, g, g)).not.toThrow();
    expect(() => scatter3d(new Float64Array([1, 2]), [1, 2], [3, 4])).not.toThrow();
    expect(() =>
      surface(
        [[0, 1], [0]],
        [
          [0, 1],
          [0, 1],
        ],
        [
          [0, 1],
          [0, 1],
        ]
      )
    ).toThrow(ShapeError);
    expect(() => surface([[0, 1]], g, g)).toThrow(ShapeError);
    expect(() => wireframe([], [], [])).toThrow(ShapeError);
  });
});

describe("InteractivePlot validation", () => {
  it("rejects non-finite zoom limits", () => {
    const fig = figure({ width: 100, height: 80 });
    expect(() => new InteractivePlot(fig, { minZoom: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new InteractivePlot(fig, { maxZoom: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    expect(() => new InteractivePlot(fig, { maxZoom: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("rejects custom CSS that closes the style element", () => {
    const fig = figure({ width: 100, height: 80 });
    expect(
      () => new InteractivePlot(fig, { customCSS: "</STYLE><script>alert(1)</script>" })
    ).toThrow(InvalidParameterError);
  });

  it("rejects non-finite data points without adding any of them", () => {
    const fig = figure({ width: 100, height: 80 });
    const ip = createInteractivePlot(fig);
    expect(() =>
      ip.addDataPoints([
        { x: 1, y: 2 },
        { x: Number.NaN, y: 3 },
      ])
    ).toThrow(InvalidParameterError);
    expect(ip.getDataPoints()).toHaveLength(0);
  });

  it("copies data points and handles very large batches", () => {
    const fig = figure({ width: 100, height: 80 });
    const ip = createInteractivePlot(fig);
    const point = { x: 1, y: 2, label: "A" };
    ip.addDataPoints([point]);
    const stored = ip.getDataPoints();
    expect(stored).toEqual([point]);
    expect(stored[0]).not.toBe(point);

    // push(...points) overflows the call stack for arrays of this size.
    const many = Array.from({ length: 300_000 }, (_, i) => ({ x: i, y: i }));
    expect(() => ip.addDataPoints(many)).not.toThrow();
    expect(ip.getDataPoints()).toHaveLength(300_001);
  });

  it("returns a copy of the options in each render result", () => {
    const fig = figure({ width: 100, height: 80 });
    const ip = new InteractivePlot(fig);
    const a = ip.render().options;
    const b = ip.render().options;
    expect(a).toEqual(b);
    expect(a).not.toBe(b);
  });

  it("zooms about the center for the zoom buttons and ignores zero-delta wheel events", () => {
    const fig = figure({ width: 100, height: 80 });
    const html = new InteractivePlot(fig).render().html;
    expect(html).toContain("function zoomAt(");
    expect(html).toContain("if (e.deltaY === 0) return;");
    expect(html).toContain("container.clientWidth / 2");
  });
});
