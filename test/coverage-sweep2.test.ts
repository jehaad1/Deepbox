/**
 * Second coverage sweep: pipeline composition, forests/extra-trees,
 * MiniBatchKMeans, in-place LinearRegression paths, dropout/upsample
 * gradient paths, mixed-dtype comparisons, pairwise/NDCG 2-D metrics,
 * sylvester complex-block solves, LBFGS option validation, Axes helper
 * surface, string-tensor reshapes, slice validation, registry/warnings
 * guards, and DataFrame key hashing of exotic values.
 */

import { describe, expect, it } from "vitest";
import {
  catchWarnings,
  DeviceError,
  InvalidParameterError,
  setWarningHandler,
  warn,
} from "../src/core";
import { getKernelBackend, requireKernelBackend } from "../src/core/backend/registry";
import { DataFrame } from "../src/dataframe";
import { listKaggleFiles } from "../src/datasets";
import { lyapunov, sylvester } from "../src/linalg";
import {
  ndcgScore,
  pairwiseCosine,
  pairwiseEuclidean,
  pairwiseManhattan,
  reciprocalRank,
  silhouetteScore,
} from "../src/metrics";
import {
  ColumnTransformer,
  ExtraTreesClassifier,
  ExtraTreesRegressor,
  FeatureUnion,
  LinearRegression,
  LogisticRegression,
  MiniBatchKMeans,
  makePipeline,
  Pipeline,
  RandomForestClassifier,
  RandomForestRegressor,
} from "../src/ml";
import {
  dot,
  equal,
  GradTensor,
  greater,
  isclose,
  less,
  parameter,
  reshape,
  Tensor,
  tensor,
  transpose,
} from "../src/ndarray";
import { AlphaDropout, Dropout, Dropout2d, Upsample } from "../src/nn";
import { LBFGS } from "../src/optim";
import { figure } from "../src/plot";
import { StandardScaler } from "../src/preprocess";

const flat = (t: unknown): number[] => [t].flat(Infinity) as number[];

// ─── ml/pipeline ─────────────────────────────────────────────────────────────

describe("Pipeline / FeatureUnion / ColumnTransformer", () => {
  const X = [
    [0, 10],
    [1, 20],
    [2, 30],
    [3, 40],
  ];
  const y = [0, 0, 1, 1];

  it("pipeline transform/fitTransform/predictProba/score/stepNames", () => {
    const pipe = new Pipeline([
      ["scale", new StandardScaler()],
      ["clf", new LogisticRegression({ maxIter: 200 })],
    ]);
    pipe.fit(tensor(X), tensor(y));
    expect(pipe.stepNames).toEqual(["scale", "clf"]);
    expect(pipe.score(tensor(X), tensor(y))).toBeGreaterThan(0.7);
    const proba = pipe.predictProba(tensor(X));
    const row = flat(proba.toArray()).slice(0, 2);
    expect(row[0]! + row[1]!).toBeCloseTo(1, 6);

    // transform()/fitTransform() require a transformer as the final step.
    const tf = new Pipeline([["scale", new StandardScaler()]]);
    const Xt = tf.fitTransform(tensor(X), tensor(y));
    expect(Xt.shape).toEqual([4, 2]);
    expect(tf.transform(tensor(X)).shape).toEqual([4, 2]);

    // ...and throw a clear error when it is not.
    expect(() => pipe.transform(tensor(X))).toThrow(/transform/);
    const noProba = new Pipeline([["scale", new StandardScaler()]]);
    noProba.fit(tensor(X), tensor(y));
    expect(() => noProba.predictProba(tensor(X))).toThrow(/predictProba/);
    expect(() => noProba.score(tensor(X), tensor(y))).toThrow(/score/);
  });

  it("makePipeline auto-names steps", () => {
    const pipe = makePipeline(new StandardScaler(), new LogisticRegression({ maxIter: 100 }));
    pipe.fit(tensor(X), tensor(y));
    expect(pipe.stepNames).toHaveLength(2);
    expect(flat(pipe.predict(tensor(X)).toArray())).toHaveLength(4);
  });

  it("FeatureUnion concatenates transformer outputs and validates names", () => {
    const union = new FeatureUnion([
      ["a", new StandardScaler()],
      ["b", new StandardScaler()],
    ]);
    const Xt = union.fitTransform(tensor(X), tensor(y));
    expect(Xt.shape).toEqual([4, 4]);
    expect(union.transform(tensor(X)).shape).toEqual([4, 4]);
    expect(Object.keys(union.getParams())).toEqual(["a", "b"]);
    expect(() => new FeatureUnion([])).toThrow(InvalidParameterError);
    expect(
      () =>
        new FeatureUnion([
          ["dup", new StandardScaler()],
          ["dup", new StandardScaler()],
        ])
    ).toThrow(/Duplicate/);
  });

  it("ColumnTransformer applies per-column transforms with remainder handling", () => {
    const ct = new ColumnTransformer(
      [
        ["scaled", new StandardScaler(), [0]],
        ["dropped", "drop", [1]],
      ],
      { remainder: "drop" }
    );
    const Xt = ct.fitTransform(tensor(X), tensor(y));
    expect(Xt.shape).toEqual([4, 1]);

    const passthrough = new ColumnTransformer([["keep", "passthrough", [1]]], {
      remainder: "passthrough",
    });
    const Xp = passthrough.fitTransform(tensor(X), tensor(y));
    expect(Xp.shape).toEqual([4, 2]);
    expect(flat(Xp.toArray()).slice(0, 2)).toEqual([10, 0]);
  });
});

// ─── forests ─────────────────────────────────────────────────────────────────

describe("random forests and extra trees", () => {
  function makeData(n: number) {
    const X: number[][] = [];
    const yc: number[] = [];
    const yr: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = (i % 10) / 10;
      const b = ((i * 7) % 10) / 10;
      X.push([a, b]);
      yc.push(a > 0.5 ? 1 : 0);
      yr.push(2 * a + 3 * b);
    }
    return { X, yc, yr };
  }

  it("RandomForestClassifier computes an OOB score (and validates bootstrap)", () => {
    const { X, yc } = makeData(60);
    const rf = new RandomForestClassifier({
      nEstimators: 20,
      oobScore: true,
      randomState: 0,
    });
    rf.fit(tensor(X), tensor(yc));
    expect(rf.score(tensor(X), tensor(yc))).toBeGreaterThan(0.9);
    const oob = rf.oobScore;
    expect(oob).toBeGreaterThan(0.7);
    expect(oob).toBeLessThanOrEqual(1);

    expect(() => new RandomForestClassifier({ oobScore: true, bootstrap: false })).toThrow(
      InvalidParameterError
    );
  });

  it("RandomForestRegressor fits and scores", () => {
    const { X, yr } = makeData(60);
    const rf = new RandomForestRegressor({ nEstimators: 20, randomState: 0 });
    rf.fit(tensor(X), tensor(yr));
    expect(rf.score(tensor(X), tensor(yr))).toBeGreaterThan(0.9);
    const imp = flat(rf.featureImportances.toArray());
    expect(imp.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 6);
  });

  it("ExtraTrees classifier and regressor learn separable data", () => {
    const { X, yc, yr } = makeData(60);
    const clf = new ExtraTreesClassifier({ nEstimators: 20, randomState: 1 });
    clf.fit(tensor(X), tensor(yc));
    expect(clf.score(tensor(X), tensor(yc))).toBeGreaterThan(0.9);
    const proba = flat(clf.predictProba(tensor([[0.9, 0.1]])).toArray());
    expect(proba[0]! + proba[1]!).toBeCloseTo(1, 6);

    const reg = new ExtraTreesRegressor({ nEstimators: 20, randomState: 1 });
    reg.fit(tensor(X), tensor(yr));
    expect(reg.score(tensor(X), tensor(yr))).toBeGreaterThan(0.85);
  });
});

// ─── MiniBatchKMeans ─────────────────────────────────────────────────────────

describe("MiniBatchKMeans", () => {
  it("clusters two blobs with mini-batches, reproducibly", () => {
    const X: number[][] = [];
    for (let i = 0; i < 100; i++) {
      const c = i % 2;
      X.push([c * 10 + (i % 7) * 0.05, c * 10 + ((i * 3) % 7) * 0.05]);
    }
    const km = new MiniBatchKMeans({ nClusters: 2, batchSize: 16, randomState: 7 });
    km.fit(tensor(X));
    const labels = flat(km.predict(tensor(X)).toArray());
    // All points of one blob share a label, the two blobs differ.
    expect(labels[0]).toBe(labels[2]);
    expect(labels[1]).toBe(labels[3]);
    expect(labels[0]).not.toBe(labels[1]);

    const km2 = new MiniBatchKMeans({ nClusters: 2, batchSize: 16, randomState: 7 });
    km2.fit(tensor(X));
    expect(flat(km2.predict(tensor(X)).toArray())).toEqual(labels);

    expect(() => new MiniBatchKMeans({ nClusters: 0 })).toThrow(InvalidParameterError);
    expect(() => new MiniBatchKMeans({ nClusters: 2, batchSize: 0 })).toThrow(
      InvalidParameterError
    );
  });
});

// ─── LinearRegression in-place paths ─────────────────────────────────────────

describe("LinearRegression copyX:false", () => {
  it("centers and scales in place, still recovering the model", () => {
    const X = tensor(
      [
        [100, 1],
        [200, 2],
        [300, 3],
        [400, 4],
      ],
      { dtype: "float64" }
    );
    const y = tensor([305, 610, 915, 1220], { dtype: "float64" });
    const model = new LinearRegression({ normalize: true, copyX: false });
    model.fit(X, y);
    const pred = flat(model.predict(tensor([[500, 5]], { dtype: "float64" })).toArray());
    expect(pred[0]).toBeCloseTo(1525, 4);
  });
});

// ─── nn dropout / upsample gradient paths ────────────────────────────────────

describe("dropout layers", () => {
  it("Dropout2d drops whole channels in training and backprops the mask", () => {
    const layer = new Dropout2d(0.5);
    layer.train();
    const x = parameter(
      tensor([
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
      ]) // [1, 2, 2, 2]
    );
    const out = layer.forward(x);
    const vals = flat(out.tensor.toArray());
    // Each channel is either fully zero or fully scaled by 1/(1-p) = 2.
    const ch0Zero = vals.slice(0, 4).every((v) => v === 0);
    const ch0Scaled = vals.slice(0, 4).every((v, i) => v === 2 * (i + 1));
    expect(ch0Zero || ch0Scaled).toBe(true);

    out.sum().backward();
    const grads = flat((x.grad as GradTensor | null) ? x.grad!.toArray() : []);
    // Gradient mirrors the channel mask (0 or 2 everywhere in a channel).
    for (const g of grads) expect([0, 2]).toContain(g);

    layer.eval();
    const evalOut = layer.forward(x);
    expect(flat(evalOut.tensor.toArray())).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("AlphaDropout keeps mean/variance roughly stable and backprops", () => {
    const layer = new AlphaDropout(0.3);
    layer.train();
    const x = parameter(tensor(Array.from({ length: 512 }, (_, i) => Math.sin(i))));
    const out = layer.forward(x);
    const vals = flat(out.tensor.toArray());
    const mean = vals.reduce((a, v) => a + v, 0) / vals.length;
    expect(Math.abs(mean)).toBeLessThan(0.35);
    out.sum().backward();
    expect(x.grad).not.toBeNull();

    layer.eval();
    expect(flat(layer.forward(x).tensor.toArray())[3]).toBeCloseTo(Math.sin(3), 6);
  });

  it("Dropout handles non-contiguous inputs via the dense path", () => {
    const layer = new Dropout(0.5);
    layer.train();
    const base = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const strided = transpose(base); // non-contiguous view
    const out = layer.forward(GradTensor.fromTensor(strided, { requiresGrad: true }));
    const vals = flat(out.tensor.toArray());
    for (const v of vals) {
      // Every surviving value is scaled by 2.
      expect(v === 0 || v % 2 === 0).toBe(true);
    }
  });
});

describe("Upsample", () => {
  it("bilinear mode with explicit output size", () => {
    const up = new Upsample({ size: [4, 4], mode: "bilinear" });
    const x = tensor([
      [
        [
          [0, 3],
          [6, 9],
        ],
      ],
    ]); // [1,1,2,2]
    const out = up.forward(GradTensor.fromTensor(x, { requiresGrad: false }));
    expect(out.shape).toEqual([1, 1, 4, 4]);
    const vals = flat(out.tensor.toArray());
    // Corners preserved, interior interpolated monotonically.
    expect(vals[0]).toBeCloseTo(0, 5);
    expect(vals[15]).toBeCloseTo(9, 5);
    expect(vals[5]).toBeGreaterThan(vals[0]!);
    expect(vals[5]).toBeLessThan(vals[15]!);
  });

  it("nearest mode with scaleFactor and gradient flow", () => {
    const up = new Upsample({ scaleFactor: 2, mode: "nearest" });
    const x = parameter(
      tensor([
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
      ])
    );
    const out = up.forward(x);
    expect(out.shape).toEqual([1, 1, 4, 4]);
    out.sum().backward();
    // Each input pixel contributes to 4 outputs -> grad = 4 everywhere.
    expect(flat(x.grad?.toArray())).toEqual([4, 4, 4, 4]);
  });
});

// ─── mixed-dtype comparisons ─────────────────────────────────────────────────

describe("int64 vs float comparisons (compareMixed)", () => {
  const big = Tensor.fromTypedArray({
    data: new BigInt64Array([1n, 2n, 3n, 9007199254740993n]),
    shape: [4],
    dtype: "int64",
    device: "cpu",
  });

  it("compares int64 against non-integer and special floats", () => {
    const f = tensor([1.5, 2, NaN, 9007199254740992], { dtype: "float64" });
    expect(flat(greater(big, f).toArray())).toEqual([0, 0, 0, 1]);
    expect(flat(less(big, f).toArray())).toEqual([1, 0, 0, 0]);
    expect(flat(equal(big, f).toArray())).toEqual([0, 1, 0, 0]);

    const inf = tensor([Infinity, -Infinity, Infinity, -Infinity], { dtype: "float64" });
    expect(flat(less(big, inf).toArray())).toEqual([1, 0, 1, 0]);
    expect(flat(greater(big, inf).toArray())).toEqual([0, 1, 0, 1]);
    expect(flat(equal(big, inf).toArray())).toEqual([0, 0, 0, 0]);
  });

  it("isclose handles scalar broadcasting", () => {
    const a = tensor([1, 1.000001, 2]);
    const s = tensor(1);
    expect(flat(isclose(a, s, 1e-3).toArray())).toEqual([1, 1, 0]);
    expect(flat(isclose(s, a, 1e-3).toArray())).toEqual([1, 1, 0]);
  });
});

// ─── metrics: pairwise on views + 2-D NDCG ───────────────────────────────────

describe("pairwise metrics and ranking", () => {
  it("pairwise distances respect strided views", () => {
    const base = tensor([
      [0, 3],
      [4, 0],
    ]);
    const tr = transpose(base); // [[0,4],[3,0]]
    const d = pairwiseEuclidean(tr);
    expect(flat(d.toArray())).toEqual([0, 5, 5, 0]);
    const m = pairwiseManhattan(tr);
    expect(flat(m.toArray())).toEqual([0, 7, 7, 0]);
    const c = pairwiseCosine(
      tensor([
        [1, 0],
        [0, 1],
        [1, 0],
      ])
    );
    const cv = flat(c.toArray());
    expect(cv[1]).toBeCloseTo(1, 6); // orthogonal -> distance 1
    expect(cv[2]).toBeCloseTo(0, 6); // identical -> distance 0
  });

  it("ndcgScore averages 2-D rows and validates shapes", () => {
    const yTrue = tensor([
      [3, 2, 1, 0],
      [1, 0, 0, 3],
    ]);
    const perfect = tensor([
      [4, 3, 2, 1],
      [2, 1, 0, 5],
    ]);
    expect(ndcgScore(yTrue, perfect)).toBeCloseTo(1, 10);
    const some = ndcgScore(
      yTrue,
      tensor([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
      ])
    );
    expect(some).toBeGreaterThan(0);
    expect(some).toBeLessThan(1);
    expect(() => ndcgScore(yTrue, tensor([1, 2, 3, 4]))).toThrow(/ndim/);
    expect(() =>
      ndcgScore(
        yTrue,
        tensor([
          [1, 2],
          [3, 4],
        ])
      )
    ).toThrow(/shape/);
    expect(reciprocalRank(tensor([0, 0, 1]), tensor([0.3, 0.2, 0.1]))).toBeCloseTo(1 / 3, 10);
  });
});

// ─── linalg: sylvester with complex-eigenvalue blocks ────────────────────────

describe("sylvester/lyapunov with complex spectra", () => {
  function residual(A: number[][], B: number[][], C: number[][]): number {
    const X = sylvester(tensor(A), tensor(B), tensor(C));
    const AX = dot(tensor(A), X);
    const XB = dot(X, tensor(B));
    let maxErr = 0;
    const ax = flat(AX.toArray());
    const xb = flat(XB.toArray());
    const c = flat(tensor(C).toArray());
    for (let i = 0; i < c.length; i++) {
      maxErr = Math.max(maxErr, Math.abs(ax[i]! + xb[i]! - c[i]!));
    }
    return maxErr;
  }

  it("solves AX + XB = C when A and B have complex eigenvalues", () => {
    // Rotation-like matrices: eigenvalues a ± bi.
    const A = [
      [1, -2],
      [2, 1],
    ];
    const B = [
      [3, 4],
      [-4, 3],
    ];
    const C = [
      [1, 2],
      [3, 4],
    ];
    expect(residual(A, B, C)).toBeLessThan(1e-8);
  });

  it("solves a mixed real/complex 3x3 system and lyapunov", () => {
    const A = [
      [2, -5, 0],
      [5, 2, 0],
      [0, 0, 3],
    ];
    const B = [
      [1, 1, 0],
      [-1, 1, 0],
      [0, 0, 4],
    ];
    const C = [
      [1, 0, 2],
      [0, 1, 0],
      [2, 0, 1],
    ];
    expect(residual(A, B, C)).toBeLessThan(1e-8);

    // Lyapunov: A X + X A^T = Q with complex-eigenvalue stable A.
    const As = [
      [-1, -3],
      [3, -1],
    ];
    const Q = [
      [-2, 0],
      [0, -2],
    ];
    const X = lyapunov(tensor(As), tensor(Q));
    const AX = dot(tensor(As), X);
    const XAt = dot(X, transpose(tensor(As)));
    const lhs = flat(AX.toArray()).map((v, i) => v + flat(XAt.toArray())[i]!);
    const q = flat(tensor(Q).toArray());
    for (let i = 0; i < q.length; i++) {
      expect(lhs[i]).toBeCloseTo(q[i]!, 8);
    }
  });
});

// ─── optim: LBFGS option validation ──────────────────────────────────────────

describe("LBFGS option validation", () => {
  const p = () => parameter(tensor([0], { dtype: "float64" }));
  it("rejects invalid hyperparameters", () => {
    expect(() => new LBFGS([p()], { lr: -1 })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { maxIter: 0 })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { historySize: 0 })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { toleranceGrad: -1 })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { toleranceChange: -1 })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { lineSearchFn: "weak" as unknown as "strong_wolfe" })).toThrow(
      InvalidParameterError
    );
  });
});

// ─── plot: Axes helper surface ───────────────────────────────────────────────

describe("Axes helpers render", () => {
  it("twinx/grid/annotate/text/step/errorbar/fill_between/log-scale/limits", () => {
    const fig = figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    const x = tensor([1, 2, 3, 4]);
    ax.plot(x, tensor([1, 4, 9, 16]));
    ax.step(x, tensor([1, 2, 2, 3]));
    ax.errorbar(x, tensor([2, 3, 4, 5]), tensor([0.5, 0.25, 0.5, 0.25]));
    ax.fill_between(x, tensor([0, 1, 0, 1]), tensor([1, 2, 1, 2]));
    ax.grid(true);
    ax.annotate("peak", 3, 9);
    expect(() => ax.annotate("bad", Number.NaN, 1)).toThrow(InvalidParameterError);
    ax.text(1.5, 8, "note");
    ax.xlim(0, 5);
    ax.ylim(0, 20);
    ax.setTitle("helpers");

    const twin = ax.twinx();
    twin.plot(x, tensor([100, 200, 300, 400]));

    const svg = fig.renderSVG().svg;
    expect(svg).toContain("peak");
    expect(svg).toContain("note");
    expect(svg).not.toContain("NaN");

    const logFig = figure({ width: 200, height: 150 });
    const lax = logFig.addAxes();
    lax.plot(tensor([1, 2, 3]), tensor([1, 100, 10000]));
    lax.setYScale("log");
    const logSvg = logFig.renderSVG().svg;
    expect(logSvg).toContain("100");
  });

  it("contourf renders filled levels through the figure API", () => {
    const n = 12;
    const grid = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => Math.sin(i / 2) + Math.cos(j / 2))
    );
    const xs = Array.from({ length: n }, (_, j) => j);
    const fig = figure({ width: 240, height: 200 });
    const ax = fig.addAxes();
    ax.contourf(
      tensor(Array.from({ length: n }, () => xs)),
      tensor(Array.from({ length: n }, (_, i) => xs.map(() => i))),
      tensor(grid),
      { colormap: "viridis" }
    );
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("path");
    expect(svg).not.toContain("NaN");
  });
});

// ─── ndarray: string reshape + slice validation ──────────────────────────────

describe("string tensor reshape and slice validation", () => {
  it("reshapes string tensors, including non-contiguous views", () => {
    const t = tensor(
      [
        ["a", "b", "c"],
        ["d", "e", "f"],
      ],
      { dtype: "string" }
    );
    expect(reshape(t, [3, 2]).toArray()).toEqual([
      ["a", "b"],
      ["c", "d"],
      ["e", "f"],
    ]);
    const tr = transpose(t);
    expect(reshape(tr, [6]).toArray()).toEqual(["a", "d", "b", "e", "c", "f"]);
    // inferred dimension
    expect(reshape(t, [-1]).shape).toEqual([6]);
    expect(() => reshape(t, [-1, -1])).toThrow();
    expect(() => reshape(t, [4])).toThrow(/reshape/i);
  });

  it("rejects invalid slice ranges", () => {
    const t = tensor([1, 2, 3, 4]);
    expect(() => t.slice({ start: 0, end: 4, step: 0 })).toThrow();
    expect(() => t.slice(10)).toThrow();
    expect(flat(t.slice({ start: -3, end: -1 }).toArray())).toEqual([2, 3]);
  });
});

// ─── core: registry + warnings guards ────────────────────────────────────────

describe("registry and warnings guards", () => {
  it("kernel-backend lookups are null/throwing for non-kernel devices", () => {
    expect(getKernelBackend("cpu")).toBeNull(); // CPU backend has no kernels
    expect(() => requireKernelBackend("cpu", "test-op")).toThrow(DeviceError);
    expect(() => requireKernelBackend("cpu", "test-op")).toThrow(/does not implement/);
  });

  it("setWarningHandler intercepts warnings", () => {
    const seen: string[] = [];
    setWarningHandler((w) => {
      seen.push(w.message);
    });
    try {
      warn("custom handler test", "UserWarning");
      expect(seen).toContain("custom handler test");
    } finally {
      setWarningHandler(undefined);
    }
    // catchWarnings still records after the handler is removed.
    const warnings = catchWarnings(() => {
      warn("caught", "UserWarning");
    });
    expect(warnings.map((w) => w.message)).toContain("caught");
  });
});

// ─── datasets: kaggle listKaggleFiles validation ─────────────────────────────

describe("listKaggleFiles validation", () => {
  it("rejects malformed ids before any network access", async () => {
    const credentials = { username: "u", key: "k" };
    await expect(listKaggleFiles("not-a-valid-id", { credentials })).rejects.toThrow(
      InvalidParameterError
    );
  });
});

// ─── dataframe: exotic groupBy/duplicate keys ────────────────────────────────

describe("DataFrame key hashing of exotic values", () => {
  it("distinguishes NaN/±Infinity/bigint/array/nested-object values", () => {
    const df = new DataFrame({
      k: [
        Number.NaN,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        Number.NaN,
        Number.POSITIVE_INFINITY,
      ],
    });
    const dupes = flat(df.duplicated().toArray()).map(Number);
    expect(dupes).toEqual([0, 0, 0, 1, 1]);

    // Arrays and nested objects hash structurally (stable key order).
    const objDf = new DataFrame({
      k: [[1, 2], [1, 2], [1, 3], { a: 1, b: 2 }, { b: 2, a: 1 }] as unknown as number[],
    });
    expect(flat(objDf.duplicated().toArray()).map(Number)).toEqual([0, 1, 0, 0, 1]);
  });
});

// ─── estimator setParams surfaces ────────────────────────────────────────────

describe("ensemble/cluster setParams", () => {
  const forestParams = {
    nEstimators: 5,
    maxDepth: 3,
    minSamplesSplit: 4,
    minSamplesLeaf: 2,
    maxFeatures: "log2",
    bootstrap: true,
    warmStart: false,
    randomState: 3,
    maxSamples: 0.5,
    oobScore: false,
  };

  it("random forests and extra trees accept every documented parameter", () => {
    for (const est of [
      new RandomForestClassifier(),
      new RandomForestRegressor(),
      new ExtraTreesClassifier(),
      new ExtraTreesRegressor(),
    ]) {
      // warmStart/maxSamples/oobScore/nEstimators are RandomForest-only options;
      // ExtraTrees accepts the shared tree parameters.
      const isRF = est instanceof RandomForestClassifier || est instanceof RandomForestRegressor;
      const { warmStart, maxSamples, oobScore, nEstimators, ...shared } = forestParams;
      est.setParams(isRF ? forestParams : shared);
      const got = est.getParams();
      if (isRF) expect(got.nEstimators).toBe(5);
      expect(got.maxDepth).toBe(3);
      expect(got.maxFeatures).toBe("log2");
      expect(got.randomState).toBe(3);
      expect(() => est.setParams({ minSamplesSplit: 0 })).toThrow(InvalidParameterError);
      expect(() => est.setParams({ bogus: 1 })).toThrow(InvalidParameterError);
    }
  });

  it("MiniBatchKMeans accepts and validates its parameters", () => {
    const km = new MiniBatchKMeans({ nClusters: 2 });
    km.setParams({
      nClusters: 3,
      maxIter: 5,
      batchSize: 8,
      tol: 1e-3,
      nInit: 2,
      init: "random",
      randomState: 1,
    });
    expect(km.getParams().nClusters).toBe(3);
    expect(km.getParams().init).toBe("random");
    expect(() => km.setParams({ nClusters: 0 })).toThrow(InvalidParameterError);
    expect(() => km.setParams({ init: "voodoo" })).toThrow(InvalidParameterError);
    expect(() => km.setParams({ nope: 1 })).toThrow(InvalidParameterError);
  });
});

// ─── metrics internals (strided access helpers) ──────────────────────────────

describe("metrics internal helpers", () => {
  it("createFlatOffsetter walks strided 2-D views correctly", async () => {
    const internals = await import("../src/metrics/_internal");
    const base = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const view = transpose(base); // [3, 2] strided view
    const at = internals.createFlatOffsetter(view);
    const data = base.data as Float32Array;
    // flat order of the transposed view: 1,4,2,5,3,6
    expect([0, 1, 2, 3, 4, 5].map((i) => data[at(i)])).toEqual([1, 4, 2, 5, 3, 6]);

    expect(() => internals.assertFiniteNumber(Number.NaN, "x", "row 0")).toThrow(/finite/);
    internals.assertFiniteNumber(1, "x", "row 0");
    expect(() => internals.assertVectorLike(base, "y")).toThrow(/column vector/);
    internals.assertVectorLike(tensor([[1], [2]]), "y");
    expect(() =>
      internals.assertSameSizeVectors(tensor([1, 2]), tensor([1, 2, 3]), "a", "b")
    ).toThrow();
    expect(() => internals.assertSameSize(tensor([1, 2]), tensor([1]), "a", "b")).toThrow();
  });
});

// ─── silhouette with large-n sampling requirement ────────────────────────────

describe("silhouette large-n sampling", () => {
  it("requires sampleSize above 2000 samples and works with one", () => {
    const n = 2100;
    const X: number[][] = [];
    const labels: number[] = [];
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      X.push([c * 10 + (i % 5) * 0.01]);
      labels.push(c);
    }
    expect(() => silhouetteScore(tensor(X), tensor(labels))).toThrow(/sampleSize/);
    const s = silhouetteScore(tensor(X), tensor(labels), "euclidean", {
      sampleSize: 200,
      randomState: 3,
    });
    expect(s).toBeGreaterThan(0.9);
  });
});
