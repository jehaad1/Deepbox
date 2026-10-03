import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ConvergenceError,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  cross_val_score,
  cross_validate,
  type Estimator,
  GridSearchCV,
  Isomap,
  LinearRegression,
  LogisticRegression,
  MDS,
  MLPClassifier,
  MLPRegressor,
  Pipeline,
  RandomizedSearchCV,
  Ridge,
  SpectralEmbedding,
  TSNE,
} from "../../src/ml";
import { tsneJointProbabilities } from "../../src/ml/manifold/TSNE";
import { type Tensor, tensor, transpose } from "../../src/ndarray";
import { StandardScaler } from "../../src/preprocess";
import { clearSeed, setSeed } from "../../src/random";

const f64 = (rows: number | number[] | number[][]): Tensor => tensor(rows, { dtype: "float64" });
const rows = (t: Tensor): number[][] => t.toArray() as number[][];
const flat = (t: Tensor): number[] =>
  Array.from(t.toArray() as ArrayLike<number>).flat() as number[];

/** Compare two embeddings column by column, ignoring the arbitrary sign of each column. */
function expectEmbeddingClose(actual: number[][], expected: number[][], tol: number): void {
  expect(actual.length).toBe(expected.length);
  const nc = expected[0]?.length ?? 0;
  for (let c = 0; c < nc; c++) {
    let dot = 0;
    for (let i = 0; i < expected.length; i++)
      dot += (actual[i]?.[c] ?? 0) * (expected[i]?.[c] ?? 0);
    const sign = dot < 0 ? -1 : 1;
    for (let i = 0; i < expected.length; i++) {
      expect(Math.abs(sign * (actual[i]?.[c] ?? 0) - (expected[i]?.[c] ?? 0))).toBeLessThan(tol);
    }
  }
}

// Two blobs of eight points in R^3 (scikit-learn 1.8 reference values below).
const A = [
  [1.764, 0.4, 0.979],
  [2.241, 1.868, -0.977],
  [0.95, -0.151, -0.103],
  [0.411, 0.144, 1.454],
  [0.761, 0.122, 0.444],
  [0.334, 1.494, -0.205],
  [0.313, -0.854, -2.553],
  [0.654, 0.864, -0.742],
  [6.27, 2.546, 4.046],
  [3.813, 5.533, 5.469],
  [4.155, 4.378, 3.112],
  [2.019, 3.652, 4.156],
  [5.23, 5.202, 3.613],
  [3.698, 2.951, 2.58],
  [2.294, 5.951, 3.49],
  [3.562, 2.747, 4.777],
];

afterEach(() => {
  clearSeed();
  vi.restoreAllMocks();
});

describe("classical MDS eigen-solver", () => {
  it("embeds a non-Euclidean matrix with its largest positive eigenvalues", () => {
    // B has eigenvalues -21.46, 0, 12.12, 18.38, 19.96. Power iteration locks onto -21.46
    // (largest magnitude) and returned an all-zero first column.
    const D = f64([
      [0, 6, 1, 1, 6],
      [6, 0, 1, 2, 5],
      [1, 1, 0, 6, 2],
      [1, 2, 6, 0, 1],
      [6, 5, 2, 1, 0],
    ]);
    const emb = rows(new MDS({ nComponents: 2, dissimilarity: "precomputed" }).fitTransform(D));
    expectEmbeddingClose(
      emb,
      [
        [3.551231, -0.0],
        [-1.912328, -0.749097],
        [0.136713, -2.937703],
        [0.136713, 2.937703],
        [-1.912328, 0.749097],
      ],
      1e-5
    );
  });

  it("is deterministic and independent of the global random stream", () => {
    const a = rows(new MDS({ nComponents: 2 }).fitTransform(f64(A)));
    setSeed(7);
    const b = rows(new MDS({ nComponents: 2 }).fitTransform(f64(A)));
    expect(b).toEqual(a);
  });

  it("reproduces the pairwise distances of planar data exactly", () => {
    // Points on a plane inside R^3: a 2-D classical MDS embedding is isometric.
    const P = [
      [0, 0, 0],
      [1, 0, 1],
      [0, 2, 2],
      [3, 1, 4],
      [2, 2, 4],
    ];
    const emb = rows(new MDS({ nComponents: 2 }).fitTransform(f64(P)));
    const dist = (u: number[], v: number[]): number =>
      Math.sqrt(u.reduce((s, x, k) => s + (x - (v[k] ?? 0)) ** 2, 0));
    for (let i = 0; i < P.length; i++) {
      for (let j = 0; j < P.length; j++) {
        expect(dist(emb[i] as number[], emb[j] as number[])).toBeCloseTo(
          dist(P[i] as number[], P[j] as number[]),
          8
        );
      }
    }
  });

  it("matches numpy classical MDS on the blob data", () => {
    const emb = rows(new MDS({ nComponents: 2 }).fitTransform(f64(A)));
    expectEmbeddingClose(
      emb.slice(0, 3),
      [
        [1.9878321, 0.6638708],
        [2.2032984, 0.2038355],
        [3.4075274, 0.3767023],
      ],
      1e-5
    );
  });

  it("returns exact zeros for components beyond the rank of the data", () => {
    // Collinear points: only one eigenvalue is positive, the rest is rounding noise.
    const emb = rows(new MDS({ nComponents: 3 }).fitTransform(f64([[0], [1], [2], [4]])));
    for (const row of emb) {
      expect(row[1]).toBe(0);
      expect(row[2]).toBe(0);
    }
    expect(emb[0]?.[0]).toBeCloseTo(-1.75, 12);
  });

  it("rejects nComponents larger than n_samples", () => {
    expect(() => new MDS({ nComponents: 5 }).fit(f64([[0], [1], [2]]))).toThrow(
      InvalidParameterError
    );
  });

  it("validates precomputed matrices", () => {
    const mds = new MDS({ dissimilarity: "precomputed" });
    expect(() =>
      mds.fit(
        f64([
          [0, 1, 2],
          [1, 0, 1],
        ])
      )
    ).toThrow(ShapeError);
    expect(() =>
      mds.fit(
        f64([
          [0, 1],
          [3, 0],
        ])
      )
    ).toThrow(DataValidationError);
    expect(() =>
      mds.fit(
        f64([
          [0, -1],
          [-1, 0],
        ])
      )
    ).toThrow(DataValidationError);
    expect(() => new MDS({ dissimilarity: "bogus" as "euclidean" })).toThrow(InvalidParameterError);
  });

  it("precomputed Euclidean distances give the same embedding", () => {
    const n = A.length;
    const D = A.map((u) =>
      A.map((v) => Math.sqrt(u.reduce((s, x, k) => s + (x - (v[k] ?? 0)) ** 2, 0)))
    );
    const direct = rows(new MDS({ nComponents: 2 }).fitTransform(f64(A)));
    const viaD = rows(
      new MDS({ nComponents: 2, dissimilarity: "precomputed" }).fitTransform(f64(D))
    );
    expect(viaD.length).toBe(n);
    expectEmbeddingClose(viaD, direct, 1e-8);
  });

  it("setParams validates and applies parameters", () => {
    const mds = new MDS();
    mds.setParams({ nComponents: 3 });
    expect(mds.getParams()["nComponents"]).toBe(3);
    expect(() => mds.setParams({ nComponents: 0 })).toThrow(InvalidParameterError);
    expect(mds.getParams()["nComponents"]).toBe(3);
    expect(() => mds.setParams({ nope: 1 })).toThrow(InvalidParameterError);
  });

  it("exposes nFeaturesIn only after fitting", () => {
    const mds = new MDS();
    expect(() => mds.nFeaturesIn).toThrow(NotFittedError);
    mds.fit(f64(A));
    expect(mds.nFeaturesIn).toBe(3);
  });
});

describe("Isomap", () => {
  it("matches scikit-learn (connected components joined) and the out-of-sample transform", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const iso = new Isomap({ nComponents: 2, nNeighbors: 4 });
    const emb = rows(iso.fitTransform(f64(A)));
    expect(warn).toHaveBeenCalled();
    expectEmbeddingClose(
      emb,
      [
        [2.067105, 0.026679],
        [4.219366, -0.011533],
        [3.50072, -0.077874],
        [3.239805, 0.033407],
        [3.179116, -0.045929],
        [4.750077, -0.11662],
        [5.997977, -0.252997],
        [4.140869, -0.117173],
        [-3.88236, 3.250356],
        [-5.603602, -0.899661],
        [-3.17634, -0.134908],
        [-3.569711, -1.080813],
        [-4.291609, 0.497949],
        [-1.650927, 0.302025],
        [-5.401138, -2.101942],
        [-3.519349, 0.729032],
      ],
      1e-4
    );

    const projected = rows(
      iso.transform(
        f64([
          [1.0, 0.5, 0.2],
          [5.0, 4.0, 4.5],
        ])
      )
    );
    expectEmbeddingClose(
      projected,
      [
        [3.095331, -0.036672],
        [-4.558841, 1.852477],
      ],
      1e-3
    );
    // Projecting the training data reproduces the fitted embedding.
    expectEmbeddingClose(rows(iso.transform(f64(A))), emb, 1e-9);
  });

  it("joins disconnected components instead of failing or returning NaN", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const emb = flat(new Isomap({ nComponents: 2, nNeighbors: 2 }).fitTransform(f64(A)));
    expect(emb.every(Number.isFinite)).toBe(true);
  });

  it("geodesic distances equal scipy's shortest paths on a connected k-NN graph", () => {
    const B = [
      [0, 0],
      [1, 0.2],
      [2, 0.1],
      [3, 0.4],
      [4, 0.3],
      [5, 0.6],
      [6, 0.5],
      [7, 0.9],
    ];
    const iso = new Isomap({ nComponents: 1, nNeighbors: 2 });
    const emb = flat(iso.fitTransform(f64(B)));
    const G = rows(iso.distMatrix);
    const ref = [0.0, 1.019804, 2.002498, 3.046529, 4.051517, 5.095547, 6.100535, 7.117922];
    for (const [j, v] of ref.entries()) expect(G[0]?.[j] as number).toBeCloseTo(v, 5);
    expect(G.flat().reduce((a, b) => a + b, 0)).toBeCloseTo(171.413296, 4);
    const refEmb = [
      -3.55093, -2.5525, -1.548936, -0.504482, 0.500913, 1.545366, 2.545655, 3.564914,
    ];
    const sign = (emb[0] as number) < 0 ? 1 : -1;
    for (const [i, v] of refEmb.entries()) expect(sign * (emb[i] as number)).toBeCloseTo(v, 4);
  });

  it("is deterministic regardless of the global seed", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const a = rows(new Isomap({ nNeighbors: 4 }).fitTransform(f64(A)));
    setSeed(3);
    const b = rows(new Isomap({ nNeighbors: 4 }).fitTransform(f64(A)));
    expect(b).toEqual(a);
  });

  it("rejects nComponents > n_samples and transform before fit", () => {
    expect(() =>
      new Isomap({ nComponents: 9, nNeighbors: 1 }).fit(f64([[0], [1], [2], [3]]))
    ).toThrow(InvalidParameterError);
    expect(() => new Isomap().transform(f64([[1, 2, 3]]))).toThrow(NotFittedError);
  });

  it("transform validates the feature count", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const iso = new Isomap({ nNeighbors: 4 }).fit(f64(A));
    expect(() => iso.transform(f64([[1, 2]]))).toThrow(ShapeError);
  });

  it("setParams validates and applies parameters", () => {
    const iso = new Isomap();
    iso.setParams({ nNeighbors: 7, nComponents: 3 });
    expect(iso.getParams()).toEqual({ nComponents: 3, nNeighbors: 7 });
    expect(() => iso.setParams({ nNeighbors: 0 })).toThrow(InvalidParameterError);
    expect(iso.getParams()).toEqual({ nComponents: 3, nNeighbors: 7 });
    expect(() => iso.setParams({ other: 1 })).toThrow(InvalidParameterError);
  });

  it("keeps projecting with the fitted settings after setParams", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const iso = new Isomap({ nComponents: 2, nNeighbors: 4 }).fit(f64(A));
    const before = rows(iso.transform(f64(A)));
    iso.setParams({ nComponents: 3, nNeighbors: 6 });
    const after = rows(iso.transform(f64(A)));
    expect(after).toEqual(before);
    expect(after[0]?.length).toBe(2);
  });

  it("handles a few hundred samples quickly (Dijkstra instead of Floyd-Warshall)", () => {
    const pts: number[][] = [];
    for (let i = 0; i < 400; i++) {
      const t = (i / 400) * 4 * Math.PI;
      pts.push([Math.cos(t) * (1 + t / 6), Math.sin(t) * (1 + t / 6), t / 5]);
    }
    const start = performance.now();
    const emb = rows(new Isomap({ nComponents: 2, nNeighbors: 6 }).fitTransform(f64(pts)));
    expect(emb.length).toBe(400);
    expect(performance.now() - start).toBeLessThan(5000);
  });
});

describe("SpectralEmbedding", () => {
  it("matches scikit-learn's rbf spectral embedding", () => {
    const emb = rows(new SpectralEmbedding({ nComponents: 2, gamma: 0.2 }).fitTransform(f64(A)));
    expectEmbeddingClose(
      emb,
      [
        [-0.113251, 0.022904],
        [-0.115319, 0.004638],
        [-0.122515, -0.008335],
        [-0.119197, 0.00928],
        [-0.121562, -0.000787],
        [-0.121388, -0.009937],
        [-0.124804, -0.042946],
        [-0.122733, -0.016588],
        [0.226384, 0.503267],
        [0.232029, -0.424206],
        [0.225454, 0.025848],
        [0.219644, -0.050539],
        [0.230405, -0.089298],
        [0.199243, 0.281809],
        [0.229813, -0.482447],
        [0.222793, 0.234063],
      ],
      1e-5
    );
  });

  it("is deterministic regardless of the global seed", () => {
    const a = rows(new SpectralEmbedding({ gamma: 0.2 }).fitTransform(f64(A)));
    setSeed(11);
    const b = rows(new SpectralEmbedding({ gamma: 0.2 }).fitTransform(f64(A)));
    expect(b).toEqual(a);
  });

  it("reports isolated samples instead of returning zeros", () => {
    // gamma = 1 with a squared distance of 3600 underflows every kernel value to 0.
    expect(() => new SpectralEmbedding({ gamma: 1 }).fit(f64([[0], [60], [120]]))).toThrow(
      DataValidationError
    );
  });

  it("requires nComponents < n_samples and rejects NaN gamma", () => {
    expect(() => new SpectralEmbedding({ nComponents: 3 }).fit(f64([[0], [1], [2]]))).toThrow(
      InvalidParameterError
    );
    expect(() => new SpectralEmbedding({ gamma: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("setParams validates and applies parameters", () => {
    const se = new SpectralEmbedding();
    se.setParams({ gamma: 0.5 });
    expect(se.getParams()).toEqual({ nComponents: 2, gamma: 0.5 });
    expect(() => se.setParams({ gamma: -1 })).toThrow(InvalidParameterError);
    expect(() => se.setParams({ bad: 1 })).toThrow(InvalidParameterError);
  });
});

describe("TSNE", () => {
  const C = [
    [1.789, 0.873, 0.048],
    [-1.863, -0.555, -0.177],
    [-0.083, -1.254, -0.022],
    [-0.477, -2.628, 0.442],
    [0.881, 3.419, 0.025],
    [-0.405, -1.091, -0.773],
    [0.982, -2.202, -0.593],
    [-0.206, 2.972, 0.118],
    [-1.024, -1.426, 0.313],
    [-0.161, -1.538, -0.115],
    [0.745, 3.952, -0.622],
    [-0.626, -1.608, -1.21],
  ];
  const P0 = [
    0.0, 1.3e-7, 0.00863661, 1e-8, 0.03063767, 0.00120638, 0.00020753, 0.00837361, 3.34e-6,
    0.00077249, 0.00400587, 2.13e-6,
  ];
  const P5 = [
    0.00120638, 0.00762093, 0.01952792, 0.00022228, 0.00021275, 0.0, 0.00232753, 0.00031705,
    0.0021825, 0.0181571, 0.00030404, 0.04657872,
  ];

  it("joint probabilities match scikit-learn's perplexity search", () => {
    const P = tsneJointProbabilities(Float64Array.from(C.flat()), 12, 3, 3);
    for (let j = 0; j < 12; j++) {
      expect(Math.abs((P[j] as number) - (P0[j] as number))).toBeLessThan(2e-8);
      expect(Math.abs((P[5 * 12 + j] as number) - (P5[j] as number))).toBeLessThan(2e-8);
    }
    expect(P.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (let i = 0; i < 12; i++) {
      expect(P[i * 12 + i]).toBe(0);
      for (let j = 0; j < 12; j++) expect(P[i * 12 + j]).toBe(P[j * 12 + i]);
    }
  });

  it("calibrates the bandwidth at any data scale", () => {
    // Squared distances scale with the square of the data, but the calibrated
    // probabilities must not change; very small and very large scales must not underflow.
    const base = tsneJointProbabilities(Float64Array.from(C.flat()), 12, 3, 3);
    for (const scale of [1e-6, 1e6]) {
      const scaled = tsneJointProbabilities(
        Float64Array.from(C.flat().map((v) => v * scale)),
        12,
        3,
        3
      );
      for (let k = 0; k < base.length; k++) {
        expect(Math.abs((scaled[k] as number) - (base[k] as number))).toBeLessThan(1e-6);
      }
    }
  });

  // Three well separated clusters of 20 points in R^5.
  const blobs = (): { X: number[][]; labels: number[] } => {
    let s = 12345;
    const rnd = (): number => {
      s = (s * 16807) % 2147483647;
      return s / 2147483647;
    };
    const X: number[][] = [];
    const labels: number[] = [];
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 20; i++) {
        X.push(Array.from({ length: 5 }, () => (c - 1) * 8 + (rnd() - 0.5) * 3));
        labels.push(c);
      }
    }
    return { X, labels };
  };

  const nearestNeighborAccuracy = (emb: number[][], labels: number[]): number => {
    let correct = 0;
    for (let i = 0; i < emb.length; i++) {
      let best = -1;
      let bestD = Infinity;
      for (let j = 0; j < emb.length; j++) {
        if (j === i) continue;
        const d = Math.hypot(
          (emb[i]?.[0] ?? 0) - (emb[j]?.[0] ?? 0),
          (emb[i]?.[1] ?? 0) - (emb[j]?.[1] ?? 0)
        );
        if (d < bestD) {
          bestD = d;
          best = j;
        }
      }
      if (labels[best] === labels[i]) correct++;
    }
    return correct / emb.length;
  };

  it("exact mode separates clusters and reaches a low KL divergence", () => {
    const { X, labels } = blobs();
    const tsne = new TSNE({ perplexity: 10, nIter: 500, randomState: 0 });
    const emb = rows(tsne.fitTransform(f64(X)));
    expect(nearestNeighborAccuracy(emb, labels)).toBe(1);
    expect(tsne.klDivergence).toBeLessThan(0.5);
    expect(emb.flat().every(Number.isFinite)).toBe(true);
  });

  it("approximate mode uses the true nearest neighbors", () => {
    // The previous implementation drew *random* "neighbors", so the affinities carried
    // no information about the data and the clusters were not recovered.
    const { X, labels } = blobs();
    const tsne = new TSNE({
      perplexity: 8,
      nIter: 1000,
      randomState: 1,
      method: "approximate",
      approximateNeighbors: 24,
      negativeSamples: 20,
    });
    const emb = rows(tsne.fitTransform(f64(X)));
    expect(nearestNeighborAccuracy(emb, labels)).toBeGreaterThanOrEqual(0.95);
    expect(tsne.klDivergence).toBeUndefined();
  });

  it("approximate mode with every point as a neighbor follows the exact trajectory", () => {
    // With k = n - 1 the sparse affinities are the exact ones and no negatives are
    // sampled, so a few iterations must agree with exact t-SNE up to rounding.
    const { X } = blobs();
    const small = f64(X.slice(0, 20));
    const exact = rows(new TSNE({ perplexity: 4, nIter: 5, randomState: 3 }).fitTransform(small));
    const sparse = rows(
      new TSNE({
        perplexity: 4,
        nIter: 5,
        randomState: 3,
        method: "approximate",
        approximateNeighbors: 19,
        negativeSamples: 5,
      }).fitTransform(small)
    );
    for (let i = 0; i < 20; i++) {
      for (let k = 0; k < 2; k++) {
        expect(Math.abs((sparse[i]?.[k] as number) - (exact[i]?.[k] as number))).toBeLessThan(1e-8);
      }
    }
  });

  it("is reproducible for a fixed randomState and honors the global seed otherwise", () => {
    const { X } = blobs();
    const opts = { perplexity: 10, nIter: 60 } as const;
    const a = rows(new TSNE({ ...opts, randomState: 5 }).fitTransform(f64(X)));
    const b = rows(new TSNE({ ...opts, randomState: 5 }).fitTransform(f64(X)));
    const c = rows(new TSNE({ ...opts, randomState: 6 }).fitTransform(f64(X)));
    expect(b).toEqual(a);
    expect(c).not.toEqual(a);

    setSeed(99);
    const g1 = rows(new TSNE({ ...opts }).fitTransform(f64(X)));
    setSeed(99);
    const g2 = rows(new TSNE({ ...opts }).fitTransform(f64(X)));
    expect(g2).toEqual(g1);
  });

  it("uses learningRate 'auto' by default and accepts a fixed rate", () => {
    const { X, labels } = blobs();
    const tsne = new TSNE({ perplexity: 10, nIter: 400, randomState: 0 });
    expect(tsne.getParams()["learningRate"]).toBe("auto");
    expect(nearestNeighborAccuracy(rows(tsne.fitTransform(f64(X))), labels)).toBeGreaterThanOrEqual(
      0.95
    );
    const fixed = new TSNE({ perplexity: 10, nIter: 400, randomState: 0, learningRate: 50 });
    expect(fixed.getParams()["learningRate"]).toBe(50);
    expect(rows(fixed.fitTransform(f64(X)))).toEqual(rows(tsne.fitTransform(f64(X))));
    expect(() => new TSNE({ learningRate: "fast" as "auto" })).toThrow(InvalidParameterError);
  });

  it("transform refuses new samples instead of returning the training embedding", () => {
    const { X } = blobs();
    const tsne = new TSNE({ perplexity: 10, nIter: 30, randomState: 0 });
    expect(() => tsne.transform()).toThrow(NotFittedError);
    tsne.fit(f64(X));
    expect(tsne.transform().shape).toEqual([60, 2]);
    expect(tsne.transform(f64(X)).shape).toEqual([60, 2]);
    expect(() => tsne.transform(f64(X.slice(0, 10)))).toThrow(ShapeError);
  });

  it("exposes `embedding` like the other manifold estimators", () => {
    const { X } = blobs();
    const tsne = new TSNE({ perplexity: 10, nIter: 30, randomState: 0 });
    expect(() => tsne.embedding).toThrow(NotFittedError);
    tsne.fit(f64(X));
    expect(tsne.embedding.shape).toEqual([60, 2]);
    expect(rows(tsne.embeddingResult)).toEqual(rows(tsne.embedding));
    // The returned tensor is a copy: mutating it must not change the model.
    const first = tsne.embedding;
    (first.data as Float64Array)[0] = 1234;
    expect(rows(tsne.embedding)[0]?.[0]).not.toBe(1234);
  });

  it("returns a float64 embedding", () => {
    const { X } = blobs();
    expect(new TSNE({ perplexity: 10, nIter: 5, randomState: 0 }).fitTransform(f64(X)).dtype).toBe(
      "float64"
    );
  });

  it("embedding is centered", () => {
    const { X } = blobs();
    const emb = rows(new TSNE({ perplexity: 10, nIter: 50, randomState: 0 }).fitTransform(f64(X)));
    for (let k = 0; k < 2; k++) {
      const mean = emb.reduce((s, r) => s + (r[k] as number), 0) / emb.length;
      expect(Math.abs(mean)).toBeLessThan(1e-9);
    }
  });

  it("stops the exaggeration phase early when the gradient vanishes", () => {
    // Identical points have P uniform and Q uniform: the gradient is ~0 immediately.
    const X = f64([
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
    ]);
    const emb = rows(new TSNE({ perplexity: 2, nIter: 100, randomState: 0 }).fitTransform(X));
    expect(emb.flat().every(Number.isFinite)).toBe(true);
  });

  it("validates approximate-mode neighbor counts", () => {
    const { X } = blobs();
    expect(() =>
      new TSNE({ perplexity: 10, method: "approximate", approximateNeighbors: 5 }).fit(f64(X))
    ).toThrow(InvalidParameterError);
  });

  it("accepts non-numeric garbage only as typed errors", () => {
    expect(() =>
      new TSNE({ perplexity: 2 }).fit(
        f64([
          [Number.NaN, 1],
          [1, 2],
          [2, 3],
          [3, 4],
        ])
      )
    ).toThrow(DataValidationError);
  });
});

describe("MLPClassifier", () => {
  const XOR = f64([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ]);
  const yXor = f64([0, 1, 1, 0]);

  it("learns XOR with a hidden layer", () => {
    for (const activation of ["tanh", "relu"] as const) {
      const clf = new MLPClassifier({
        hiddenLayerSizes: [8],
        activation,
        learningRate: 0.1,
        maxIter: 2000,
        tol: 0,
        nIterNoChange: 2000,
        randomState: 1,
      });
      clf.fit(XOR, yXor);
      expect(clf.score(XOR, yXor)).toBe(1);
      expect(clf.loss).toBeLessThan(0.05);
    }
  });

  it("is reproducible with randomState and shuffles by default", () => {
    const params = { hiddenLayerSizes: [6], learningRate: 0.05, maxIter: 30, randomState: 4 };
    const a = new MLPClassifier(params).fit(XOR, yXor);
    const b = new MLPClassifier(params).fit(XOR, yXor);
    expect(flat(a.predictProba(XOR))).toEqual(flat(b.predictProba(XOR)));
    const c = new MLPClassifier({ ...params, randomState: 5 }).fit(XOR, yXor);
    expect(flat(c.predictProba(XOR))).not.toEqual(flat(a.predictProba(XOR)));
    expect(a.getParams()["shuffle"]).toBe(true);
  });

  it("trains on class-sorted data (samples are shuffled every epoch)", () => {
    // 60 class-0 points followed by 60 class-1 points; without shuffling, plain SGD
    // ends up forgetting the first class.
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 60; i++) {
      X.push([-2 + (i % 7) * 0.1, (i % 5) * 0.2]);
      y.push(0);
    }
    for (let i = 0; i < 60; i++) {
      X.push([2 + (i % 7) * 0.1, (i % 5) * 0.2]);
      y.push(1);
    }
    const clf = new MLPClassifier({
      hiddenLayerSizes: [8],
      learningRate: 0.05,
      maxIter: 100,
      randomState: 0,
    }).fit(f64(X), f64(y));
    expect(clf.score(f64(X), f64(y))).toBe(1);
  });

  it("returns float64 probabilities that sum to one", () => {
    const X = f64([
      [1, 0],
      [2, 0],
      [0, 1],
      [0, 2],
      [1, 1],
      [2, 2],
    ]);
    const y = f64([0, 0, 1, 1, 2, 2]);
    const clf = new MLPClassifier({
      hiddenLayerSizes: [8],
      learningRate: 0.05,
      maxIter: 200,
      randomState: 0,
    }).fit(X, y);
    const proba = clf.predictProba(X);
    expect(proba.dtype).toBe("float64");
    expect(proba.shape).toEqual([6, 3]);
    for (const row of rows(proba)) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    }
    expect(clf.score(X, y)).toBeGreaterThanOrEqual(5 / 6);
  });

  it("binary probabilities are not clamped away from 0 and 1", () => {
    const clf = new MLPClassifier({
      hiddenLayerSizes: [4],
      learningRate: 0.5,
      maxIter: 300,
      randomState: 2,
    });
    const X = f64([[-5], [-4], [4], [5]]);
    const y = f64([0, 0, 1, 1]);
    clf.fit(X, y);
    const p = rows(clf.predictProba(f64([[5], [-5]])));
    expect((p[0]?.[0] as number) + (p[0]?.[1] as number)).toBeCloseTo(1, 14);
    expect(p[0]?.[1] as number).toBeGreaterThan(0.9);
    expect(p[1]?.[0] as number).toBeGreaterThan(0.9);
  });

  it("stops after nIterNoChange epochs without improvement (scikit-learn rule)", () => {
    // tol so large that no epoch counts as an improvement: epoch 1 sets the best loss,
    // epochs 2..(nIterNoChange + 2) fail to improve. scikit-learn reports n_iter_ = 5.
    const clf = new MLPClassifier({
      hiddenLayerSizes: [4],
      tol: 1e9,
      nIterNoChange: 3,
      maxIter: 100,
      randomState: 0,
    }).fit(XOR, yXor);
    expect(clf.nIter).toBe(5);
    expect(clf.lossCurve.length).toBe(5);
  });

  it("requires at least two integer-labelled classes", () => {
    const clf = new MLPClassifier({ hiddenLayerSizes: [2] });
    expect(() => clf.fit(XOR, f64([1, 1, 1, 1]))).toThrow(DataValidationError);
    expect(() => clf.fit(XOR, f64([0, 0.5, 1, 1]))).toThrow(DataValidationError);
  });

  it("keeps the previous fit when a refit fails", () => {
    const clf = new MLPClassifier({
      hiddenLayerSizes: [4],
      learningRate: 0.1,
      maxIter: 50,
      randomState: 0,
    });
    clf.fit(XOR, yXor);
    const before = flat(clf.predictProba(XOR));
    expect(() => clf.fit(XOR, f64([1, 1, 1, 1]))).toThrow(DataValidationError);
    expect(flat(clf.predictProba(XOR))).toEqual(before);
  });

  it("setParams validates every parameter and supports tol", () => {
    const clf = new MLPClassifier();
    clf.setParams({ tol: 1e-3, shuffle: false, nIterNoChange: 4, randomState: 9 });
    expect(clf.getParams()).toMatchObject({
      tol: 1e-3,
      shuffle: false,
      nIterNoChange: 4,
      randomState: 9,
    });
    expect(() => clf.setParams({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ learningRate: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ alpha: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ hiddenLayerSizes: [3, 1.5] })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ activation: "swish" })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ shuffle: "yes" })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ nope: 1 })).toThrow(InvalidParameterError);
    // a rejected update leaves the configuration untouched
    expect(clf.getParams()["tol"]).toBe(1e-3);
  });

  it("validates constructor options", () => {
    expect(() => new MLPClassifier({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => new MLPClassifier({ alpha: -1 })).toThrow(InvalidParameterError);
    expect(() => new MLPClassifier({ activation: "gelu" as "relu" })).toThrow(
      InvalidParameterError
    );
    expect(() => new MLPClassifier({ learningRate: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new MLPClassifier({ nIterNoChange: 0 })).toThrow(InvalidParameterError);
  });

  it("does not alias the hiddenLayerSizes array passed in", () => {
    const sizes = [3];
    const clf = new MLPClassifier({ hiddenLayerSizes: sizes });
    sizes.push(0);
    expect(clf.getParams()["hiddenLayerSizes"]).toEqual([3]);
  });

  it("score checks that X and y agree", () => {
    const clf = new MLPClassifier({ hiddenLayerSizes: [2], maxIter: 5, randomState: 0 }).fit(
      XOR,
      yXor
    );
    expect(() => clf.score(XOR, f64([0, 1]))).toThrow(ShapeError);
  });

  it("exposes fitted attributes only after fit", () => {
    const clf = new MLPClassifier({ hiddenLayerSizes: [2], maxIter: 5, randomState: 0 });
    expect(() => clf.nIter).toThrow(NotFittedError);
    expect(() => clf.lossCurve).toThrow(NotFittedError);
    expect(() => clf.nFeaturesIn).toThrow(NotFittedError);
    clf.fit(XOR, yXor);
    expect(clf.nFeaturesIn).toBe(2);
    expect(clf.nIter).toBeLessThanOrEqual(5);
  });

  it("works through GridSearchCV and cross_val_score (getParams round-trips)", () => {
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 40; i++) {
      X.push([i < 20 ? -1 - (i % 5) * 0.1 : 1 + (i % 5) * 0.1, (i % 3) * 0.1]);
      y.push(i < 20 ? 0 : 1);
    }
    const clf = new MLPClassifier({
      hiddenLayerSizes: [4],
      maxIter: 60,
      learningRate: 0.05,
      randomState: 0,
    });
    const scores = cross_val_score(clf, f64(X), f64(y), 4);
    expect(scores.length).toBe(4);
    for (const s of scores) expect(s).toBeGreaterThanOrEqual(0.9);
    const gs = new GridSearchCV(clf, { activation: ["tanh", "relu"] }, { cv: 2 });
    gs.fit(f64(X), f64(y));
    expect(["tanh", "relu"]).toContain(gs.bestParams["activation"]);
  });
});

describe("MLPRegressor", () => {
  it("fits a smooth function and returns float64 predictions", () => {
    const X = f64(Array.from({ length: 50 }, (_, i) => [i / 10]));
    const y = f64(Array.from({ length: 50 }, (_, i) => Math.sin(i / 10) * 3 + 1));
    const reg = new MLPRegressor({
      hiddenLayerSizes: [20],
      activation: "tanh",
      learningRate: 0.01,
      maxIter: 500,
      randomState: 0,
    }).fit(X, y);
    expect(reg.score(X, y)).toBeGreaterThan(0.95);
    const pred = reg.predict(X);
    expect(pred.dtype).toBe("float64");
    expect(reg.nIter).toBeGreaterThan(0);
  });

  it("backpropagates through several layers", () => {
    // y = 2 x1 - x2 + 1 with linear units in two hidden layers.
    const rowsX: number[][] = [];
    const ys: number[] = [];
    for (let i = 0; i < 40; i++) {
      const a = ((i * 7) % 11) / 5 - 1;
      const b = ((i * 3) % 13) / 6 - 1;
      rowsX.push([a, b]);
      ys.push(2 * a - b + 1);
    }
    const reg = new MLPRegressor({
      hiddenLayerSizes: [6, 4],
      activation: "identity",
      learningRate: 0.01,
      maxIter: 400,
      tol: 0,
      nIterNoChange: 400,
      randomState: 3,
    }).fit(f64(rowsX), f64(ys));
    expect(reg.score(f64(rowsX), f64(ys))).toBeGreaterThan(0.999);
  });

  it("validates constructor options that were previously unchecked", () => {
    expect(() => new MLPRegressor({ learningRate: -1 })).toThrow(InvalidParameterError);
    expect(() => new MLPRegressor({ maxIter: 0 })).toThrow(InvalidParameterError);
    expect(() => new MLPRegressor({ alpha: -0.1 })).toThrow(InvalidParameterError);
    expect(() => new MLPRegressor({ tol: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new MLPRegressor({ activation: "nope" as "relu" })).toThrow(InvalidParameterError);
  });

  it("setParams supports tol and rejects invalid values", () => {
    const reg = new MLPRegressor();
    reg.setParams({ tol: 1e-2, hiddenLayerSizes: [3, 2] });
    expect(reg.getParams()).toMatchObject({ tol: 1e-2, hiddenLayerSizes: [3, 2] });
    expect(() => reg.setParams({ hiddenLayerSizes: [3, -1] })).toThrow(InvalidParameterError);
    expect(() => reg.setParams({ learningRate: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("reports divergence as a ConvergenceError", () => {
    const reg = new MLPRegressor({
      hiddenLayerSizes: [3],
      activation: "identity",
      learningRate: 1e3,
      maxIter: 50,
      randomState: 0,
    });
    expect(() => reg.fit(f64([[1e3], [-1e3], [2e3]]), f64([1, 2, 3]))).toThrow(ConvergenceError);
  });

  it("score checks that X and y agree", () => {
    const reg = new MLPRegressor({ hiddenLayerSizes: [2], maxIter: 5, randomState: 0 }).fit(
      f64([[1], [2], [3]]),
      f64([1, 2, 3])
    );
    expect(() => reg.score(f64([[1], [2], [3]]), f64([1, 2]))).toThrow(ShapeError);
  });
});

/** Estimator stub that records what it is fitted and scored on. */
class Recorder implements Estimator {
  static fits: { X: number[][]; y: number[] }[] = [];
  static tests: number[][] = [];
  constructor(private readonly opts: Record<string, unknown> = {}) {}
  fit(X: Tensor, y: Tensor): this {
    Recorder.fits.push({ X: rows(X), y: Array.from(y.toArray() as ArrayLike<number>) });
    return this;
  }
  score(_X: Tensor, y: Tensor): number {
    Recorder.tests.push(Array.from(y.toArray() as ArrayLike<number>));
    return 0;
  }
  getParams(): Record<string, unknown> {
    return { ...this.opts };
  }
  setParams(params: Record<string, unknown>): this {
    Object.assign(this.opts, params);
    return this;
  }
}

/** Classifier-looking stub (exposes `classes`) so folds are stratified. */
class ClassRecorder extends Recorder {
  readonly classes = undefined;
}

/** Stub whose score is a deterministic function of its parameters. */
class ParamScorer implements Estimator {
  constructor(private readonly opts: Record<string, unknown> = {}) {}
  fit(): this {
    return this;
  }
  predict(X: Tensor): Tensor {
    return f64(Array.from({ length: X.shape[0] ?? 0 }, () => 0));
  }
  score(): number {
    let s = 0;
    for (const v of Object.values(this.opts)) s += typeof v === "number" ? v : 0;
    return s;
  }
  getParams(): Record<string, unknown> {
    return { ...this.opts };
  }
  setParams(params: Record<string, unknown>): this {
    Object.assign(this.opts, params);
    return this;
  }
}

describe("cross-validation helpers", () => {
  const reset = (): void => {
    Recorder.fits = [];
    Recorder.tests = [];
  };

  it("keeps float64 precision in the folds (no float32 round trip)", () => {
    reset();
    const X = f64(Array.from({ length: 6 }, (_, i) => [16777217 + i, 0.1 * i]));
    const y = f64(Array.from({ length: 6 }, (_, i) => 16777217 + i));
    cross_val_score(new Recorder(), X, y, 3);
    const seen = new Set(Recorder.fits.flatMap((f) => f.y));
    expect(seen.has(16777217)).toBe(true);
    expect(seen.has(16777216)).toBe(false);
    expect(Recorder.fits.flatMap((f) => f.X.map((r) => r[0])).includes(16777217)).toBe(true);
  });

  it("reads strided (transposed) inputs correctly", () => {
    reset();
    const base = f64([
      [0, 1, 2, 3, 4, 5],
      [10, 11, 12, 13, 14, 15],
    ]);
    const X = transpose(base); // shape [6, 2], non-contiguous view
    const y = f64([0, 1, 2, 3, 4, 5]);
    cross_val_score(new Recorder(), X, y, 3);
    for (const fit of Recorder.fits) {
      fit.X.forEach((row, i) => {
        // row = [k, 10 + k] and the label of the same sample is k
        expect(row[1]).toBe((row[0] as number) + 10);
        expect(fit.y[i]).toBe(row[0]);
      });
    }
  });

  it("preserves int32 labels", () => {
    reset();
    const y = tensor([0, 1, 0, 1, 0, 1], { dtype: "int32" });
    let dtype = "";
    class Probe extends Recorder {
      override fit(X: Tensor, yy: Tensor): this {
        dtype = yy.dtype;
        return super.fit(X, yy);
      }
    }
    cross_val_score(new Probe(), f64([[0], [1], [2], [3], [4], [5]]), y, 2);
    expect(dtype).toBe("int32");
  });

  it("flattens an (n_samples, 1) target column like the 1-D case", () => {
    reset();
    const X = f64(Array.from({ length: 6 }, (_, i) => [i]));
    const y = f64(Array.from({ length: 6 }, (_, i) => [i * 2]));
    let ndim = -1;
    class Probe extends Recorder {
      override fit(Xv: Tensor, yy: Tensor): this {
        ndim = yy.ndim;
        return super.fit(Xv, yy);
      }
    }
    cross_val_score(new Probe(), X, y, 3);
    expect(ndim).toBe(1);
  });

  it("uses fold sizes that differ by at most one (4, 3, 3 for 10 samples)", () => {
    reset();
    const X = f64(Array.from({ length: 10 }, (_, i) => [i]));
    const y = f64(Array.from({ length: 10 }, (_, i) => i * 1.5));
    cross_val_score(new Recorder(), X, y, 3);
    expect(Recorder.tests.map((t) => t.length).sort()).toEqual([3, 3, 4]);
  });

  it("stratifies folds for classifiers", () => {
    reset();
    // 12 samples of class 0 followed by 6 of class 1, cv=3: every test fold gets 2 of class 1.
    const y = f64([...Array(12).fill(0), ...Array(6).fill(1)]);
    const X = f64(Array.from({ length: 18 }, (_, i) => [i]));
    cross_val_score(new ClassRecorder(), X, y, 3);
    expect(Recorder.tests.length).toBe(3);
    for (const test of Recorder.tests) {
      expect(test.filter((v) => v === 1).length).toBe(2);
      expect(test.filter((v) => v === 0).length).toBe(4);
    }
  });

  it("does not stratify on continuous targets", () => {
    reset();
    const X = f64(Array.from({ length: 9 }, (_, i) => [i]));
    const y = f64(Array.from({ length: 9 }, (_, i) => i + 0.5));
    cross_val_score(new ClassRecorder(), X, y, 3);
    expect(Recorder.tests.map((t) => t.length)).toEqual([3, 3, 3]);
  });

  it("validates shapes and sample counts", () => {
    const X = f64([[1], [2], [3], [4]]);
    expect(() => cross_val_score(new Recorder(), X, f64([1, 2, 3]), 2)).toThrow(ShapeError);
    expect(() => cross_val_score(new Recorder(), f64([1, 2, 3, 4]), f64([1, 2, 3, 4]), 2)).toThrow(
      ShapeError
    );
    expect(() => cross_val_score(new Recorder(), X, f64([1, 2, 3, 4]), 5)).toThrow(
      InvalidParameterError
    );
    expect(() => cross_val_score(new Recorder(), X, f64([1, 2, 3, 4]), 2.5)).toThrow(
      InvalidParameterError
    );
  });

  it("cross_val_score accepts a scoring function", () => {
    const X = f64(Array.from({ length: 6 }, (_, i) => [i]));
    const y = f64([0, 1, 2, 3, 4, 5]);
    const scores = cross_val_score(new ParamScorer({ a: 1 }), X, y, 3, () => 42);
    expect(scores).toEqual([42, 42, 42]);
    expect(() =>
      cross_val_score(new ParamScorer(), X, y, 3, "no" as unknown as () => number)
    ).toThrow(InvalidParameterError);
  });

  it("cross_validate checks the scoring mapping and reports sub-millisecond timings", () => {
    const X = f64(Array.from({ length: 12 }, (_, i) => [i, i * i]));
    const y = f64(Array.from({ length: 12 }, (_, i) => 3 * i + 1));
    expect(() => cross_validate(new LinearRegression(), X, y, { cv: 3, scoring: {} })).toThrow(
      InvalidParameterError
    );
    expect(() =>
      cross_validate(new LinearRegression(), X, y, {
        cv: 3,
        scoring: { bad: 1 as unknown as () => number },
      })
    ).toThrow(InvalidParameterError);
    const res = cross_validate(new LinearRegression(), X, y, { cv: 3 });
    expect(res.testScores["score"]?.length).toBe(3);
    for (const t of [...res.fitTime, ...res.scoreTime]) {
      expect(Number.isFinite(t)).toBe(true);
      expect(t).toBeGreaterThanOrEqual(0);
    }
    // performance.now() has sub-millisecond resolution; Date.now() would give integers
    expect(res.fitTime.some((t) => !Number.isInteger(t))).toBe(true);
  });
});

describe("GridSearchCV", () => {
  const X = f64(Array.from({ length: 12 }, (_, i) => [i]));
  const y = f64(Array.from({ length: 12 }, (_, i) => i * 2));

  it("selects the best combination, breaking ties toward the first", () => {
    const gs = new GridSearchCV(
      new ParamScorer({ a: 0, b: 0 }),
      { a: [1, 3, 2], b: [5, 5] },
      { cv: 3 }
    );
    gs.fit(X, y);
    expect(gs.bestParams).toEqual({ a: 3, b: 5 });
    expect(gs.bestIndex).toBe(2);
    expect(gs.bestScore).toBe(8);
    expect(gs.cvResults.length).toBe(6);
    expect(gs.cvResults[0]?.params).toEqual({ a: 1, b: 5 });
    expect(gs.score(X, y)).toBe(8);
  });

  it("validates cv, grid and scoring at construction", () => {
    expect(() => new GridSearchCV(new Ridge(), { alpha: [1] }, { cv: 1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new GridSearchCV(new Ridge(), { alpha: [] })).toThrow(InvalidParameterError);
    expect(() => new GridSearchCV(new Ridge(), { alpha: 1 as unknown as number[] })).toThrow(
      InvalidParameterError
    );
    expect(
      () => new GridSearchCV(new Ridge(), { alpha: [1] }, { scoring: 3 as unknown as () => number })
    ).toThrow(InvalidParameterError);
  });

  it("raises a clear error for a bad parameter value instead of a cloning error", () => {
    const gs = new GridSearchCV(new Ridge(), { alpha: [-1] }, { cv: 3 });
    expect(() => gs.fit(X, y)).toThrow(/alpha/i);
  });

  it("refuses to pick a winner when every score is NaN", () => {
    const gs = new GridSearchCV(
      new ParamScorer({ a: 0 }),
      { a: [1, 2] },
      { cv: 3, scoring: () => Number.NaN }
    );
    expect(() => gs.fit(X, y)).toThrow(DataValidationError);
  });

  it("supports a custom scoring function and clone()-based estimators", () => {
    const pipe = new Pipeline([
      ["scale", new StandardScaler()],
      ["reg", new LinearRegression()],
    ]);
    const gs = new GridSearchCV(
      pipe,
      {},
      { cv: 3, scoring: (est, Xv, yv) => -Math.abs((est as Pipeline).score(Xv, yv) - 1) }
    );
    gs.fit(X, y);
    expect(gs.bestEstimator).toBeDefined();
    expect(gs.predict(X).shape).toEqual([12]);
  });

  it("checks shapes and the estimator interface", () => {
    const gs = new GridSearchCV(new Ridge(), { alpha: [1] }, { cv: 3 });
    expect(() => gs.fit(X, f64([1, 2, 3]))).toThrow(ShapeError);
    class NoScore {
      fit(): this {
        return this;
      }
      getParams(): Record<string, unknown> {
        return {};
      }
      setParams(): this {
        return this;
      }
    }
    expect(() => new GridSearchCV(new NoScore(), { a: [1] }).fit(X, y)).toThrow(
      InvalidParameterError
    );
  });

  it("setParams updates the configuration and rejects unknown keys", () => {
    const gs = new GridSearchCV(new ParamScorer({ a: 0 }), { a: [1] }, { cv: 3 });
    gs.setParams({ paramGrid: { a: [1, 9] }, cv: 4 });
    gs.fit(X, y);
    expect(gs.bestParams).toEqual({ a: 9 });
    expect(gs.getParams()["cv"]).toBe(4);
    expect(() => gs.setParams({ cv: 1 })).toThrow(InvalidParameterError);
    expect(() => gs.setParams({ nonsense: 1 })).toThrow(InvalidParameterError);
  });

  it("predictProba requires a fitted classifier-like estimator", () => {
    const gs = new GridSearchCV(new ParamScorer({ a: 0 }), { a: [1] }, { cv: 3 });
    expect(() => gs.predictProba(X)).toThrow(NotFittedError);
    gs.fit(X, y);
    expect(() => gs.predictProba(X)).toThrow(InvalidParameterError);
  });

  it("works with a real classifier on class-sorted data (stratified folds)", () => {
    const Xc: number[][] = [];
    const yc: number[] = [];
    for (let i = 0; i < 30; i++) {
      Xc.push([i < 24 ? -2 + (i % 6) * 0.1 : 2 + (i % 6) * 0.1, (i % 4) * 0.1]);
      yc.push(i < 24 ? 0 : 1);
    }
    const gs = new GridSearchCV(
      new LogisticRegression({ maxIter: 100 }),
      { C: [0.1, 1] },
      { cv: 3 }
    );
    gs.fit(f64(Xc), f64(yc));
    expect(gs.bestScore).toBeGreaterThanOrEqual(0.9);
    for (const r of gs.cvResults) for (const s of r.scores) expect(s).toBeGreaterThanOrEqual(0.8);
  });
});

describe("RandomizedSearchCV", () => {
  const X = f64(Array.from({ length: 12 }, (_, i) => [i]));
  const y = f64(Array.from({ length: 12 }, (_, i) => i));

  it("samples nIter distinct combinations and is reproducible", () => {
    const space = { a: [1, 2, 3, 4, 5], b: [10, 20, 30, 40] };
    const make = (): RandomizedSearchCV =>
      new RandomizedSearchCV(new ParamScorer({ a: 0, b: 0 }), space, {
        nIter: 7,
        cv: 3,
        randomState: 3,
      });
    const rs = make().fit(X, y);
    expect(rs.cvResults.length).toBe(7);
    const keys = rs.cvResults.map((r) => JSON.stringify(r.params));
    expect(new Set(keys).size).toBe(7);
    expect(
      make()
        .fit(X, y)
        .cvResults.map((r) => JSON.stringify(r.params))
    ).toEqual(keys);
    const best = Math.max(...rs.cvResults.map((r) => r.meanScore));
    expect(rs.bestScore).toBe(best);
    expect(rs.cvResults[rs.bestIndex]?.params).toEqual(rs.bestParams);
  });

  it("evaluates the whole grid when it has at most nIter combinations", () => {
    const rs = new RandomizedSearchCV(
      new ParamScorer({ a: 0 }),
      { a: [1, 2, 3] },
      { nIter: 10, cv: 3, randomState: 0 }
    ).fit(X, y);
    expect(rs.cvResults.map((r) => r.params["a"])).toEqual([1, 2, 3]);
    expect(rs.bestParams).toEqual({ a: 3 });
  });

  it("samples from very large grids without enumerating them", () => {
    // 10^9 combinations: enumerating them (the previous behavior) exhausts memory.
    const space: Record<string, number[]> = {};
    for (let k = 0; k < 9; k++) space[`p${k}`] = Array.from({ length: 10 }, (_, i) => i);
    const rs = new RandomizedSearchCV(new ParamScorer({}), space, {
      nIter: 5,
      cv: 3,
      randomState: 1,
    }).fit(X, y);
    expect(rs.cvResults.length).toBe(5);
  });

  it("validates nIter, cv and randomState", () => {
    const base = new ParamScorer({ a: 0 });
    expect(() => new RandomizedSearchCV(base, { a: [1] }, { nIter: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => new RandomizedSearchCV(base, { a: [1] }, { nIter: 1.5 })).toThrow(
      InvalidParameterError
    );
    expect(() => new RandomizedSearchCV(base, { a: [1] }, { cv: 1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new RandomizedSearchCV(base, { a: [1] }, { randomState: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new RandomizedSearchCV(base, { a: [] })).toThrow(InvalidParameterError);
  });

  it("checks that X has at least cv samples and matching y", () => {
    const rs = new RandomizedSearchCV(new ParamScorer({ a: 0 }), { a: [1] }, { cv: 5 });
    expect(() => rs.fit(f64([[1], [2]]), f64([1, 2]))).toThrow(InvalidParameterError);
    expect(() => rs.fit(X, f64([1, 2]))).toThrow(ShapeError);
  });

  it("getParams reports randomState and setParams round-trips", () => {
    const rs = new RandomizedSearchCV(new ParamScorer({ a: 0 }), { a: [1, 2] }, { randomState: 4 });
    expect(rs.getParams()["randomState"]).toBe(4);
    rs.setParams({ nIter: 2, cv: 3, randomState: 5 });
    expect(rs.getParams()).toMatchObject({ nIter: 2, cv: 3, randomState: 5 });
    expect(() => rs.setParams({ nIter: 0 })).toThrow(InvalidParameterError);
    expect(rs.getParams()["nIter"]).toBe(2);
    expect(() => rs.setParams({ extra: 1 })).toThrow(InvalidParameterError);
  });

  it("predict and score require fitting", () => {
    const rs = new RandomizedSearchCV(new ParamScorer({ a: 0 }), { a: [1] }, { cv: 3 });
    expect(() => rs.predict(X)).toThrow(NotFittedError);
    expect(() => rs.score(X, y)).toThrow(NotFittedError);
    rs.fit(X, y);
    expect(rs.predict(X).shape).toEqual([12]);
    expect(rs.score(X, y)).toBe(1);
  });
});
