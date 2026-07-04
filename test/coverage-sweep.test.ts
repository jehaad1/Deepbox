/**
 * Behavioral tests for previously under-covered surfaces:
 * silhouette precomputed/sampled paths, SelfTrainingClassifier,
 * LinearRegression normalize/multi-target, Hardtanh/Tanhshrink/Softmin/
 * Softmax2d activations, 3-D plots (Surface3D/Wireframe3D/Scatter3D),
 * tick generation edge cases, and the remaining complex/half-precision
 * array methods.
 */

import { describe, expect, it } from "vitest";
import { InvalidParameterError, NotFittedError } from "../src/core";
import { silhouetteSamples, silhouetteScore } from "../src/metrics";
import { LinearRegression, LogisticRegression, SelfTrainingClassifier } from "../src/ml";
import {
  BFloat16Array,
  Complex,
  Complex128Array,
  Float16Array as DbFloat16Array,
  GradTensor,
  tensor,
} from "../src/ndarray";
import { Hardtanh, Softmax2d, Softmin, Tanhshrink } from "../src/nn";
import { figure } from "../src/plot";
import { generateLogTicks, generateTicks } from "../src/plot/utils/ticks";

const flat = (t: unknown): number[] => [t].flat(Infinity) as number[];

// ─── metrics/clustering: silhouette variants ─────────────────────────────────

describe("silhouette precomputed and sampled paths", () => {
  // Two tight clusters far apart:
  // p0=(0,0), p1=(0,1) in cluster 0; p2=(10,0), p3=(10,1) in cluster 1.
  const X = tensor([
    [0, 0],
    [0, 1],
    [10, 0],
    [10, 1],
  ]);
  const labels = tensor([0, 0, 1, 1]);
  // Hand-computed: a = 1, b = (10 + sqrt(101)) / 2 for every point.
  const b = (10 + Math.sqrt(101)) / 2;
  const expected = (b - 1) / b;

  function distMatrix(): number[][] {
    const pts = [
      [0, 0],
      [0, 1],
      [10, 0],
      [10, 1],
    ];
    return pts.map((p) => pts.map((q) => Math.hypot(p[0]! - q[0]!, p[1]! - q[1]!)));
  }

  it("matches the hand-computed euclidean silhouette", () => {
    expect(silhouetteScore(X, labels)).toBeCloseTo(expected, 10);
  });

  it("metric='precomputed' gives identical results", () => {
    const D = tensor(distMatrix());
    // distance matrix is stored float32 by default -> ~1e-7 precision
    expect(silhouetteScore(D, labels, "precomputed")).toBeCloseTo(expected, 7);
    const samples = flat(silhouetteSamples(D, labels, "precomputed").toArray());
    for (const s of samples) expect(s).toBeCloseTo(expected, 7);
  });

  it("sampleSize approximates deterministically with randomState", () => {
    const s1 = silhouetteScore(X, labels, "euclidean", { sampleSize: 3, randomState: 42 });
    const s2 = silhouetteScore(X, labels, "euclidean", { sampleSize: 3, randomState: 42 });
    expect(s1).toBe(s2);
    expect(s1).toBeGreaterThan(0.5);
    expect(() => silhouetteScore(X, labels, "euclidean", { sampleSize: 1 })).toThrow(
      InvalidParameterError
    );
  });

  it("rejects malformed precomputed matrices", () => {
    expect(() => silhouetteScore(tensor([[1, 2, 3]]), labels, "precomputed")).toThrow();
  });
});

// ─── ml/semi_supervised: SelfTrainingClassifier ──────────────────────────────

describe("SelfTrainingClassifier", () => {
  function blobs(): { X: number[][]; y: number[] } {
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 20; i++) {
      const c = i % 2;
      X.push([c * 10 + (i % 5) * 0.1, c * 10 + ((i * 3) % 5) * 0.1]);
      // Only 3 labeled samples per class; the rest are unlabeled (-1).
      y.push(i < 6 ? c : -1);
    }
    return { X, y };
  }

  it("labels unlabeled samples and classifies separable data", () => {
    const { X, y } = blobs();
    const clf = new SelfTrainingClassifier({
      baseEstimator: new LogisticRegression({ maxIter: 200 }),
      threshold: 0.7,
    });
    clf.fit(tensor(X), tensor(y));
    const pred = flat(clf.predict(tensor(X)).toArray());
    for (let i = 0; i < X.length; i++) {
      expect(pred[i]).toBe(i % 2);
    }
    const proba = clf.predictProba(tensor([[0, 0]]));
    const row = flat(proba.toArray());
    expect(row.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 6);
  });

  it("validates constructor options and fitted state", () => {
    const base = new LogisticRegression();
    expect(() => new SelfTrainingClassifier({ baseEstimator: base, threshold: 2 })).toThrow(
      InvalidParameterError
    );
    expect(() => new SelfTrainingClassifier({ baseEstimator: base, maxIter: 0 })).toThrow(
      InvalidParameterError
    );
    const clf = new SelfTrainingClassifier({ baseEstimator: base });
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(NotFittedError);
    expect(clf.getParams()).toHaveProperty("threshold");
  });
});

// ─── ml/linear: LinearRegression normalize + multi-target ────────────────────

describe("LinearRegression normalize and multi-target", () => {
  it("normalize:true recovers the same line as unnormalized", () => {
    // y = 3x + 1 with wildly scaled feature
    const X = [[100], [200], [300], [400]];
    const y = [301, 601, 901, 1201];
    const plain = new LinearRegression().fit(tensor(X), tensor(y));
    const norm = new LinearRegression({ normalize: true }).fit(tensor(X), tensor(y));
    expect(flat(norm.coef.toArray())[0]).toBeCloseTo(flat(plain.coef.toArray())[0]!, 6);
    expect(flat(norm.predict(tensor([[500]])).toArray())[0]).toBeCloseTo(1501, 4);
  });

  it("rejects 2-D targets with a clear error", () => {
    const model = new LinearRegression();
    expect(() =>
      model.fit(
        tensor([[1], [2]]),
        tensor([
          [1, 2],
          [3, 4],
        ])
      )
    ).toThrow(/1-dimensional/);
  });

  it("setParams validates and score requires fit", () => {
    const model = new LinearRegression();
    model.setParams({ normalize: true, copyX: false });
    expect(model.getParams().normalize).toBe(true);
    expect(() => model.setParams({ copyX: "yes" as unknown as boolean })).toThrow(
      InvalidParameterError
    );
    expect(() => model.setParams({ bogus: 1 })).toThrow(InvalidParameterError);
    expect(() => model.score(tensor([[1]]), tensor([1]))).toThrow(NotFittedError);
  });
});

// ─── nn activations: Hardtanh / Tanhshrink / Softmin / Softmax2d ─────────────

describe("uncommon activations", () => {
  it("Hardtanh clamps to [min, max]", () => {
    const layer = new Hardtanh();
    const out = layer.forward(tensor([-2, -0.5, 0.5, 2]));
    expect(flat(out.toArray())).toEqual([-1, -0.5, 0.5, 1]);
    const custom = new Hardtanh(0, 2);
    expect(flat(custom.forward(tensor([-1, 1, 3])).toArray())).toEqual([0, 1, 2]);
  });

  it("Tanhshrink computes x - tanh(x)", () => {
    const layer = new Tanhshrink();
    const out = flat(layer.forward(tensor([0, 1, -2])).toArray());
    expect(out[0]).toBeCloseTo(0, 10);
    expect(out[1]).toBeCloseTo(1 - Math.tanh(1), 6);
    expect(out[2]).toBeCloseTo(-2 - Math.tanh(-2), 6);
  });

  it("Softmin reverses the softmax ordering and sums to 1", () => {
    const layer = new Softmin();
    const out = flat(layer.forward(tensor([[1, 2, 3]])).toArray());
    expect(out.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 6);
    expect(out[0]).toBeGreaterThan(out[1]!);
    expect(out[1]).toBeGreaterThan(out[2]!);
    // softmin(x) == softmax(-x)
    const ref = [1, 2, 3].map((v) => Math.exp(-v));
    const z = ref.reduce((a, v) => a + v, 0);
    expect(out[0]).toBeCloseTo(ref[0]! / z, 6);
  });

  it("Softmax2d normalizes over channels per spatial position", () => {
    const layer = new Softmax2d();
    // shape [1, 2, 1, 1]: two channels, one pixel
    const input = tensor([[[[1]], [[3]]]]);
    const out = layer.forward(GradTensor.fromTensor(input, { requiresGrad: false }));
    const vals = flat(out.tensor.toArray());
    expect(vals[0]! + vals[1]!).toBeCloseTo(1, 6);
    expect(vals[1]).toBeCloseTo(Math.exp(3) / (Math.exp(1) + Math.exp(3)), 6);
  });
});

// ─── plot: 3-D drawables ─────────────────────────────────────────────────────

describe("3-D plots", () => {
  function grids(n: number) {
    const xg: Float64Array[] = [];
    const yg: Float64Array[] = [];
    const zg: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
      const xr = new Float64Array(n);
      const yr = new Float64Array(n);
      const zr = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        xr[j] = j;
        yr[j] = i;
        zr[j] = Math.sin(i) * Math.cos(j);
      }
      xg.push(xr);
      yg.push(yr);
      zg.push(zr);
    }
    return { xg, yg, zg };
  }

  it("renders surface, wireframe and scatter3d to valid SVG", () => {
    const { xg, yg, zg } = grids(6);
    const fig = figure({ width: 320, height: 240 });
    const ax = fig.addAxes();
    ax.surface(xg, yg, zg, { alpha: 0.9 });
    ax.wireframe(xg, yg, zg);
    ax.scatter3d(
      new Float64Array([0, 1, 2]),
      new Float64Array([0, 1, 2]),
      new Float64Array([1, 4, 9])
    );
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<svg");
    expect(/polygon|path|line/.test(svg)).toBe(true);
    expect(svg).not.toContain("NaN");
  });

  it("renders 3-D plots through the raster (PNG) fallback", async () => {
    const { xg, yg, zg } = grids(5);
    const fig = figure({ width: 160, height: 120 });
    const ax = fig.addAxes();
    ax.surface(xg, yg, zg);
    ax.wireframe(xg, yg, zg);
    const png = await fig.renderPNG();
    // PNG magic bytes
    expect(png.bytes[0]).toBe(0x89);
    expect(png.bytes[1]).toBe(0x50);
  });
});

// ─── plot/utils/ticks ────────────────────────────────────────────────────────

describe("tick generation edges", () => {
  it("expands degenerate ranges and rejects non-finite input", () => {
    const same = generateTicks(5, 5);
    expect(same.length).toBeGreaterThan(0);
    expect(generateTicks(Number.NaN, 1)).toEqual([]);
    expect(generateTicks(0, 1, 0)).toEqual([]);
    // reversed bounds still work
    const rev = generateTicks(10, 0, 5);
    expect(rev[0]?.value).toBe(0);
  });

  it("log ticks cover decades and label extremes exponentially", () => {
    const ticks = generateLogTicks(0.5, 2000);
    expect(ticks.map((t) => t.value)).toEqual([1, 10, 100, 1000]);
    const tiny = generateLogTicks(1e-8, 1e-6);
    expect(tiny.map((t) => t.label)).toEqual(["1e-8", "1e-7", "1e-6"]);
    expect(generateLogTicks(-1, 10)).toEqual([]);
  });
});

// ─── complex / half-precision remaining methods ──────────────────────────────

describe("Complex128Array full surface", () => {
  it("fromInterleaved/fill/set/setRI/proxy access", () => {
    const arr = Complex128Array.fromInterleaved([1, 2, 3, 4, 5, 6]);
    expect(arr.length).toBe(3);
    expect(arr[2]).toBe(5); // proxy returns real part
    arr.setRI(0, 9, -9);
    expect(arr.getComplex(0).equals(new Complex(9, -9), 1e-15)).toBe(true);
    arr.fill(new Complex(1, 1), 1);
    expect(arr.getImag(2)).toBe(1);
    const target = new Complex128Array(2);
    target.set([7, 8, 9, 10]);
    expect(target.getComplex(1).equals(new Complex(9, 10), 1e-15)).toBe(true);
    expect(target.toComplexArray()).toHaveLength(2);
    expect(String(target)).toContain("Complex128Array");
  });
});

describe("half-precision array remaining surface", () => {
  it("Float16Array constructs over an existing buffer with offset", () => {
    const backing = new ArrayBuffer(16);
    const arr = new DbFloat16Array(backing, 4, 4);
    expect(arr.byteOffset).toBe(4);
    arr[0] = 1.5;
    expect(new Uint16Array(backing, 4, 1)[0]).not.toBe(0);
    expect(arr.at(0)).toBe(1.5);
  });

  it("BFloat16Array set/subarray/copyWithin", () => {
    const arr = BFloat16Array.from([1, 2, 3, 4]);
    arr.set([9, 8], 1);
    expect(arr.toArray()).toEqual([1, 9, 8, 4]);
    const sub = arr.subarray(1, 3);
    sub[0] = 5;
    expect(arr[1]).toBe(5);
    arr.copyWithin(2, 0, 2);
    expect(arr[2]).toBe(1);
    expect(arr[3]).toBe(5);
    const sl = arr.slice(0, 2);
    sl[0] = 100;
    expect(arr[0]).toBe(1); // slice copies
  });
});
