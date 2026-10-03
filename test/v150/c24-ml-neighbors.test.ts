import { afterEach, describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  BallTree,
  type Classifier,
  ColumnTransformer,
  FeatureUnion,
  GaussianRandomProjection,
  GridSearchCV,
  getEstimatorTags,
  KDTree,
  KMeans,
  KNeighborsClassifier,
  KNeighborsRegressor,
  LabelPropagation,
  LabelSpreading,
  LinearRegression,
  LogisticRegression,
  makePipeline,
  NearestCentroid,
  NearestNeighbors,
  Pipeline,
  RadiusNeighborsClassifier,
  RadiusNeighborsRegressor,
  SelfTrainingClassifier,
} from "../../src/ml";
import { johnsonLindenstraussMinDim } from "../../src/ml/random_projection";
import { type Tensor, tensor } from "../../src/ndarray";
import { StandardScaler } from "../../src/preprocess";
import { clearSeed, setSeed } from "../../src/random";

const f64 = (rows: number | number[] | number[][]): Tensor => tensor(rows, { dtype: "float64" });
const flat = (t: Tensor): number[] => {
  const out: number[] = [];
  for (let i = 0; i < t.size; i++) out.push(Number(t.data[t.offset + i]));
  return out;
};
const rows = (t: Tensor): number[][] => t.toArray() as number[][];

function expectClose(actual: readonly number[], expected: readonly number[], digits = 10): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] as number;
    if (Number.isNaN(e)) expect(Number.isNaN(actual[i] as number)).toBe(true);
    else expect(actual[i] as number).toBeCloseTo(e, digits);
  }
}

// ---------------------------------------------------------------------------
// KDTree / BallTree
// ---------------------------------------------------------------------------
const trees = [
  {
    name: "KDTree",
    make: (data: Float64Array, n: number, d: number) => new KDTree(data, n, d),
  },
  {
    name: "BallTree",
    make: (data: Float64Array, n: number, d: number) => new BallTree(data, n, d, 3),
  },
] as const;

describe.each(trees)("$name (c24)", ({ make }) => {
  const data = new Float64Array([0, 0, 1, 0, 0, 1, 1, 1, 5, 5]);

  it("matches a brute-force search, with ties ordered by index", () => {
    let seed = 12345;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let trial = 0; trial < 60; trial++) {
      const n = 1 + Math.floor(rnd() * 90);
      const d = 1 + Math.floor(rnd() * 4);
      const grid = trial % 2 === 0; // integer grid: many duplicates and equal distances
      const pts = new Float64Array(n * d);
      for (let i = 0; i < pts.length; i++) pts[i] = grid ? Math.floor(rnd() * 4) : rnd() * 10 - 5;
      const q = new Float64Array(d);
      for (let j = 0; j < d; j++) q[j] = grid ? Math.floor(rnd() * 4) : rnd() * 10 - 5;
      const k = 1 + Math.floor(rnd() * 7);
      const r = rnd() * 4;
      const all = Array.from({ length: n }, (_, i) => {
        let s = 0;
        for (let j = 0; j < d; j++) s += ((pts[i * d + j] as number) - (q[j] as number)) ** 2;
        return { s, i };
      }).sort((a, b) => a.s - b.s || a.i - b.i);

      const tree = make(pts, n, d);
      expect(tree.query(q, k).indices).toEqual(all.slice(0, k).map((x) => x.i));
      expect(tree.queryRadius(q, r).indices).toEqual(
        all.filter((x) => x.s <= r * r).map((x) => x.i)
      );
    }
  });

  it("rejects k that is not a positive integer (k = 0 used to crash with a TypeError)", () => {
    const tree = make(data, 5, 2);
    expect(() => tree.query(new Float64Array([0, 0]), 0)).toThrow(InvalidParameterError);
    expect(() => tree.query(new Float64Array([0, 0]), 1.5)).toThrow(InvalidParameterError);
    expect(() => tree.queryBatch(new Float64Array([0, 0]), 1, 0)).toThrow(InvalidParameterError);
  });

  it("rejects a negative radius (it used to be squared into a positive one)", () => {
    const tree = make(data, 5, 2);
    expect(() => tree.queryRadius(new Float64Array([0, 0]), -1)).toThrow(InvalidParameterError);
    expect(() => tree.queryRadius(new Float64Array([0, 0]), Number.NaN)).toThrow(
      InvalidParameterError
    );
    expect(tree.queryRadius(new Float64Array([0, 0]), 0).indices).toEqual([0]);
    expect(
      tree.queryRadius(new Float64Array([0, 0]), Number.POSITIVE_INFINITY).indices
    ).toHaveLength(5);
  });

  it("validates the constructor arguments", () => {
    expect(() => make(new Float64Array(4), 5, 2)).toThrow(ShapeError);
    expect(() => make(new Float64Array(4), 2, 0)).toThrow(InvalidParameterError);
    expect(() => make(new Float64Array(4), 1.5, 2)).toThrow(InvalidParameterError);
    expect(() => make(new Float64Array([0, Number.NaN]), 1, 2)).toThrow(DataValidationError);
  });

  it("validates query points", () => {
    const tree = make(data, 5, 2);
    expect(() => tree.query(new Float64Array([0]), 1)).toThrow(ShapeError);
    expect(() => tree.query(new Float64Array([0, Number.NaN]), 1)).toThrow(DataValidationError);
    expect(() => tree.queryRadius(new Float64Array([Number.POSITIVE_INFINITY, 0]), 1)).toThrow(
      DataValidationError
    );
  });

  it("queryBatch marks missing neighbors with -1 / Infinity and checks the input length", () => {
    const tree = make(new Float64Array([0, 0, 3, 4]), 2, 2);
    const { distances, indices } = tree.queryBatch(new Float64Array([0, 0]), 1, 3);
    expect(Array.from(indices)).toEqual([0, 1, -1]);
    expect(distances[0]).toBe(0);
    expect(distances[1]).toBe(5);
    expect(distances[2]).toBe(Number.POSITIVE_INFINITY);
    expect(() => tree.queryBatch(new Float64Array([0, 0, 1]), 2, 1)).toThrow(ShapeError);
  });

  it("handles an empty tree", () => {
    const tree = make(new Float64Array(0), 0, 2);
    expect(tree.query(new Float64Array([0, 0]), 2)).toEqual({ distances: [], indices: [] });
    expect(tree.queryRadius(new Float64Array([0, 0]), 1)).toEqual({ distances: [], indices: [] });
    const batch = tree.queryBatch(new Float64Array(0), 0, 2);
    expect(batch.indices).toHaveLength(0);
  });

  it("exposes nSamples and nDims", () => {
    const tree = make(data, 5, 2);
    expect(tree.nSamples).toBe(5);
    expect(tree.nDims).toBe(2);
  });
});

describe("BallTree leafSize (c24)", () => {
  it("rejects leafSize < 1 (0 used to recurse forever)", () => {
    const data = new Float64Array([0, 0, 1, 1, 2, 2]);
    expect(() => new BallTree(data, 3, 2, 0)).toThrow(InvalidParameterError);
    expect(() => new BallTree(data, 3, 2, 2.5)).toThrow(InvalidParameterError);
    expect(new BallTree(data, 3, 2, 1).query(new Float64Array([1.9, 1.9]), 1).indices).toEqual([2]);
  });

  it("handles identical points that exceed the leaf size", () => {
    const pts = new Float64Array(60).fill(1);
    const tree = new BallTree(pts, 30, 2, 4);
    const res = tree.query(new Float64Array([1, 1]), 3);
    expect(res.indices).toEqual([0, 1, 2]);
    expect(res.distances).toEqual([0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// KNeighbors / NearestNeighbors
// ---------------------------------------------------------------------------
describe("KNeighbors (c24)", () => {
  const X = [
    [0.1, 0.2],
    [0.9, 0.4],
    [1.7, 1.1],
    [2.2, 0.3],
    [3.1, 2.6],
    [0.4, 1.9],
    [2.8, 1.5],
    [1.2, 2.2],
    [3.4, 0.8],
    [0.7, 0.7],
    [2.0, 2.4],
    [1.5, 0.1],
  ];
  const y = [0, 1, 1, 2, 2, 0, 1, 0, 2, 1, 2, 0];
  const yr = [1.5, -0.5, 2.0, 3.5, 0.25, 4.0, -1.0, 2.5, 0.0, 1.0, -2.0, 0.75];
  const Xt = [
    [0.5, 0.5],
    [2.5, 1.0],
    [1.0, 2.0],
    [3.0, 0.0],
  ];
  // Reference values from scikit-learn 1.8 (n_neighbors=4 for the classifier, 3 for the regressor).
  const ref = {
    "uniform-euclidean": {
      proba: [
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.25, 0.25],
        [0.25, 0.25, 0.5],
      ],
      pred: [0, 1, 0, 2],
      reg: [0.6666666666666666, 1.5, 1.5, 1.4166666666666667],
    },
    "distance-euclidean": {
      proba: [
        [0.32943591385407067, 0.6705640861459293, 0.0],
        [0.0, 0.5520840080792211, 0.4479159919207789],
        [0.7415147327700317, 0.12556201507342943, 0.13292325215653888],
        [0.18403611572132, 0.18282666483845025, 0.6331372194402297],
      ],
      pred: [1, 1, 0, 2],
      reg: [0.6686257034386237, 1.2560765387710264, 2.2196856306988284, 1.555822520479602],
    },
    "uniform-manhattan": {
      proba: [
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.25, 0.25],
        [0.25, 0.25, 0.5],
      ],
      pred: [0, 1, 0, 2],
      reg: [0.6666666666666666, 1.5, 1.5, 1.4166666666666667],
    },
    "distance-manhattan": {
      proba: [
        [0.3225806451612903, 0.6774193548387097, 0.0],
        [0.0, 0.5529272619751625, 0.4470727380248374],
        [0.7457627118644068, 0.1186440677966102, 0.13559322033898308],
        [0.2114587259705993, 0.1990199773840934, 0.5895212966453072],
      ],
      pred: [1, 1, 0, 2],
      reg: [0.6144578313253013, 1.330578512396694, 2.269230769230769, 1.542],
    },
  } as const;

  for (const weights of ["uniform", "distance"] as const) {
    for (const metric of ["euclidean", "manhattan"] as const) {
      const key = `${weights}-${metric}` as keyof typeof ref;
      it(`matches scikit-learn (${key}), including ties between classes`, () => {
        const clf = new KNeighborsClassifier({ nNeighbors: 4, weights, metric }).fit(
          f64(X),
          tensor(y)
        );
        const proba = clf.predictProba(f64(Xt));
        expect(proba.dtype).toBe("float64");
        expectClose(flat(proba), ref[key].proba.flat(), 12);
        // Query 0 ties between classes 0 and 1 (uniform): the smaller label must win.
        expect(flat(clf.predict(f64(Xt)))).toEqual([...ref[key].pred]);

        const reg = new KNeighborsRegressor({ nNeighbors: 3, weights, metric }).fit(
          f64(X),
          f64(yr)
        );
        const pred = reg.predict(f64(Xt));
        expect(pred.dtype).toBe("float64");
        expectClose(flat(pred), ref[key].reg, 12);
      });
    }
  }

  it("distance weights: an exact match decides the prediction without a 1e-10 fudge", () => {
    const Xs = f64([[0], [1], [2]]);
    const clf = new KNeighborsClassifier({ nNeighbors: 3, weights: "distance" }).fit(
      Xs,
      tensor([0, 1, 1])
    );
    // scikit-learn: [[1, 0], [0.42857143, 0.57142857]]
    const proba = rows(clf.predictProba(f64([[0], [0.5]])));
    expect(proba[0]).toEqual([1, 0]);
    expectClose(proba[1] as number[], [0.42857142857142855, 0.5714285714285714], 12);

    const reg = new KNeighborsRegressor({ nNeighbors: 3, weights: "distance" }).fit(
      Xs,
      f64([5, 1, 2])
    );
    // scikit-learn: [5, 2.85714286, 2.06282723]
    const out = flat(reg.predict(f64([[0], [0.5], [1.7]])));
    expect(out[0]).toBe(5);
    expectClose(out, [5, 2.857142857142857, 2.0628272251308903], 12);
  });

  it("keeps non-integer class labels instead of truncating them to int32", () => {
    const clf = new KNeighborsClassifier({ nNeighbors: 1 }).fit(
      f64([[0], [1], [5], [6]]),
      f64([0.5, 1.5, 1.5, 0.5])
    );
    const pred = clf.predict(f64([[0.1], [5.2]]));
    expect(pred.dtype).toBe("float64");
    expect(flat(pred)).toEqual([0.5, 1.5]);
    // Integer labels still come back as int32.
    const ints = new KNeighborsClassifier({ nNeighbors: 1 }).fit(f64([[0], [1]]), tensor([3, 4]));
    expect(ints.predict(f64([[0.2]])).dtype).toBe("int32");
  });

  it("exposes the sorted class labels", () => {
    const clf = new KNeighborsClassifier({ nNeighbors: 1 });
    expect(() => clf.classes).toThrow(NotFittedError);
    clf.fit(f64([[0], [1], [2]]), tensor([7, 3, 7]));
    expect(flat(clf.classes)).toEqual([3, 7]);
  });

  it("is not affected by later edits of the training arrays", () => {
    const Xtr = f64([[0], [1], [10], [11]]);
    const ytr = tensor([0, 0, 1, 1]);
    const clf = new KNeighborsClassifier({ nNeighbors: 1 }).fit(Xtr, ytr);
    (Xtr.data as Float64Array)[0] = 100;
    (ytr.data as Int32Array)[3] = 0;
    expect(flat(clf.predict(f64([[0.1], [10.9]])))).toEqual([0, 1]);
  });

  it("throws when nNeighbors exceeds the training size after setParams", () => {
    const clf = new KNeighborsClassifier({ nNeighbors: 2 }).fit(
      f64([[0], [1], [2]]),
      tensor([0, 1, 1])
    );
    clf.setParams({ nNeighbors: 5 });
    expect(() => clf.predict(f64([[0.5]]))).toThrow(InvalidParameterError);
    expect(() => clf.predictProba(f64([[0.5]]))).toThrow(InvalidParameterError);
  });

  it("validates score targets", () => {
    const clf = new KNeighborsClassifier({ nNeighbors: 1 }).fit(f64([[0], [1]]), tensor([0, 1]));
    expect(() => clf.score(f64([[0], [1]]), tensor([0]))).toThrow(ShapeError);
    expect(() => clf.score(f64([[0]]), tensor([]))).toThrow(DataValidationError);
    expect(clf.score(f64([[0], [1]]), tensor([0, 1]))).toBe(1);
  });

  it("clone() returns an unfitted copy with the same parameters", () => {
    const clf = new KNeighborsClassifier({
      nNeighbors: 3,
      weights: "distance",
      metric: "manhattan",
    });
    const copy = clf.clone();
    expect(copy.getParams()).toEqual(clf.getParams());
    expect(() => copy.predict(f64([[0]]))).toThrow(NotFittedError);
  });

  it("NearestNeighbors matches scikit-learn and honors per-call overrides", () => {
    const nn = new NearestNeighbors({ nNeighbors: 2 }).fit(
      f64([
        [0, 0],
        [1, 0],
        [0, 2],
        [5, 5],
      ])
    );
    const { distances, indices } = nn.kneighbors(f64([[0.2, 0]]));
    expect(distances.dtype).toBe("float64");
    expect(indices.dtype).toBe("int32");
    expectClose(flat(distances), [0.2, 0.8], 12);
    expect(flat(indices)).toEqual([0, 1]);

    const three = nn.kneighbors(f64([[0.2, 0]]), 3);
    expect(three.indices.shape).toEqual([1, 3]);
    expect(flat(three.indices)).toEqual([0, 1, 2]);

    const rad = nn.radiusNeighbors(f64([[0.2, 0]]), 1.5);
    expect(rad.indices).toEqual([[0, 1]]);
    expectClose(rad.distances[0] as number[], [0.2, 0.8], 12);
    expect(nn.radiusNeighbors(f64([[0.2, 0]])).indices).toEqual([[0, 1]]);
    expect(() => nn.radiusNeighbors(undefined, -1)).toThrow(InvalidParameterError);
    expect(() => nn.kneighbors(undefined, 4)).toThrow(InvalidParameterError);
    expect(() => nn.kneighbors(undefined, 0)).toThrow(InvalidParameterError);
  });

  it("NearestNeighbors excludes each sample from its own neighbors and handles empty queries", () => {
    const nn = new NearestNeighbors({ nNeighbors: 1 }).fit(f64([[0], [1], [3]]));
    expect(flat(nn.kneighbors().indices)).toEqual([1, 0, 1]);
    const empty = nn.kneighbors(tensor(new Float64Array(0)).reshape([0, 1]));
    expect(empty.distances.shape).toEqual([0, 1]);
    expect(empty.indices.shape).toEqual([0, 1]);
  });
});

// ---------------------------------------------------------------------------
// NearestCentroid
// ---------------------------------------------------------------------------
describe("NearestCentroid (c24)", () => {
  const X = f64([
    [0.1, 0.2],
    [0.9, 0.4],
    [1.7, 1.1],
    [2.2, 0.3],
    [3.1, 2.6],
    [0.4, 1.9],
    [2.8, 1.5],
    [1.2, 2.2],
    [3.4, 0.8],
    [0.7, 0.7],
    [2.0, 2.4],
    [1.5, 0.1],
  ]);
  const y = tensor([0, 1, 1, 2, 2, 0, 1, 0, 2, 1, 2, 0]);
  const Xt = f64([
    [0.5, 0.5],
    [2.5, 1.0],
    [1.0, 2.0],
    [3.0, 0.0],
  ]);

  // Reference values from scikit-learn 1.8.
  it("euclidean: centroids, probabilities and decision function", () => {
    const m = new NearestCentroid().fit(X, y);
    expectClose(flat(m.centroids), [0.8, 1.1, 1.525, 0.925, 2.675, 1.525], 12);
    expect(flat(m.predict(Xt))).toEqual([0, 2, 0, 2]);
    const proba = m.predictProba(Xt);
    expect(proba.dtype).toBe("float64");
    expectClose(
      flat(proba),
      [
        0.8023628606668423, 0.19746760690754184, 0.0001695324256158276, 0.008842920315302507,
        0.22311826341187846, 0.768038816272819, 0.6740385146311326, 0.3131451290093435,
        0.01281635635952391, 0.0011070249853338489, 0.13625034891600327, 0.8626426260986629,
      ],
      12
    );
    expectClose(
      flat(m.decisionFunction(Xt)),
      [
        -2.7374058098979703, -4.139392204932861, -11.19967782905312, -7.0113985866432955,
        -3.7833138010617553, -2.5471754872559234, -3.1425492481070836, -3.9091697467220423,
        -7.105114304986458, -11.55438496665061, -6.741567196041544, -4.896060691605903,
      ],
      12
    );
  });

  it("manhattan metric uses per-feature medians and refuses predictProba", () => {
    const m = new NearestCentroid({ metric: "manhattan" }).fit(X, y);
    expectClose(flat(m.centroids), [0.8, 1.05, 1.3, 0.9, 2.65, 1.6], 12);
    expect(flat(m.predict(Xt))).toEqual([0, 2, 0, 2]);
    expect(() => m.predictProba(Xt)).toThrow(InvalidParameterError);
    expect(() => m.decisionFunction(Xt)).toThrow(InvalidParameterError);
  });

  it("shrinkThreshold shrinks centroids like scikit-learn", () => {
    const m = new NearestCentroid({ shrinkThreshold: 0.5 }).fit(X, y);
    expectClose(
      flat(m.centroids),
      [
        1.135483133469207, 1.1833333333333333, 1.6666666666666667, 1.1833333333333333,
        2.339516866530793, 1.1833333333333333,
      ],
      12
    );
    expectClose(
      flat(m.predictProba(Xt)).slice(0, 3),
      [0.8259153523018369, 0.16825118281434, 0.00583346488382319],
      12
    );
  });

  it("priors change predict and the decision function", () => {
    const m = new NearestCentroid({ priors: [0.2, 0.3, 0.5] }).fit(X, y);
    expect(flat(m.predict(Xt))).toEqual([0, 2, 1, 2]);
    expectClose(
      flat(m.predictProba(Xt)).slice(0, 3),
      [0.6430608888627682, 0.3560899031475249, 0.0008492079897068739],
      12
    );
    expectClose(flat(m.classPrior), [0.2, 0.3, 0.5], 12);
    expect(() => new NearestCentroid({ priors: [0.5, 0.5] }).fit(X, y)).toThrow(
      InvalidParameterError
    );
    expect(() => new NearestCentroid({ priors: [-1, 1, 1] })).toThrow(InvalidParameterError);
  });

  it("shrinking with empirical priors matches scikit-learn on 4 classes and 4 features", () => {
    // Reference values from scikit-learn 1.8: NearestCentroid(shrink_threshold=0.4, priors="empirical").
    const X4 = f64([
      [0.5, 1.2, -0.3, 2.0],
      [0.7, 0.9, -0.1, 1.5],
      [0.4, 1.1, -0.6, 2.2],
      [2.5, 0.2, 1.0, 0.1],
      [2.9, 0.5, 0.7, 0.4],
      [2.2, 0.1, 1.3, -0.2],
      [1.0, 3.0, 0.0, 0.9],
      [1.3, 2.6, 0.4, 1.1],
      [0.8, 2.9, -0.2, 0.7],
      [3.5, 3.4, 1.8, 2.9],
      [3.1, 3.0, 2.2, 3.3],
      [3.8, 3.9, 1.5, 2.6],
      [1.1, 1.0, 0.5, 1.0],
    ]);
    const y4 = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]);
    const Xq = f64([
      [0.6, 1.0, 0.0, 1.8],
      [3.0, 3.2, 1.9, 3.0],
      [1.2, 2.0, 0.3, 1.0],
    ]);
    const m = new NearestCentroid({ shrinkThreshold: 0.4, priors: "empirical" }).fit(X4, y4);
    expectClose(
      flat(m.centroids),
      [
        0.7854062008610083, 1.15159187324765, -0.0051505604396906435, 1.5524114397364028,
        2.3989511851110663, 0.39032035697098544, 0.854123907663259, 0.24921004388545254,
        1.1677154815556008, 2.7096796430290144, 0.21254275900340763, 1.0492100438854526,
        3.3322845184443985, 3.3096796430290145, 1.6874572409965922, 2.784123289447881,
      ],
      12
    );
    expectClose(flat(m.classPrior), [4 / 13, 3 / 13, 3 / 13, 3 / 13], 12);
    expect(flat(m.predict(Xq))).toEqual([0, 3, 2]);
    const proba = flat(m.predictProba(Xq));
    expectClose(proba.slice(0, 4), [1, 7.383796494183051e-26, 8.886402693122991e-22, 0], 12);
    expectClose(proba.slice(8, 12), [0.0011128899036717869, 0, 0.9988871100963282, 0], 12);
  });

  it("binary decision function is the log-likelihood ratio (1-D)", () => {
    const yb = tensor([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]);
    const m = new NearestCentroid().fit(X, yb);
    const dec = m.decisionFunction(Xt);
    expect(dec.shape).toEqual([4]);
    expectClose(
      flat(dec),
      [-7.580075462302757, 2.389510881611427, -3.401067058822341, 3.5019523441793545],
      12
    );
  });

  it("handles a single class and requires more samples than classes for shrinking", () => {
    const one = new NearestCentroid().fit(
      f64([
        [1, 2],
        [3, 4],
      ]),
      tensor([5, 5])
    );
    expect(flat(one.predict(f64([[0, 0]])))).toEqual([5]);
    expect(flat(one.predictProba(f64([[0, 0]])))).toEqual([1]);
    const shrink = new NearestCentroid({ shrinkThreshold: 0.5 });
    expect(() =>
      shrink.fit(
        f64([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 1])
      )
    ).toThrow(DataValidationError);
  });

  it("validates parameters and score targets", () => {
    expect(() => new NearestCentroid({ metric: "cosine" as unknown as "euclidean" })).toThrow(
      InvalidParameterError
    );
    expect(() => new NearestCentroid({ shrinkThreshold: 0 })).toThrow(InvalidParameterError);
    expect(() => new NearestCentroid().setParams({ nope: 1 })).toThrow(InvalidParameterError);
    const m = new NearestCentroid().fit(X, y);
    expect(() => m.score(Xt, tensor([0, 1]))).toThrow(ShapeError);
    expect(m.score(Xt, tensor([0, 2, 0, 2]))).toBe(1);
    expect(m.clone().getParams()).toEqual(m.getParams());
  });
});

// ---------------------------------------------------------------------------
// RadiusNeighbors
// ---------------------------------------------------------------------------
describe("RadiusNeighbors (c24)", () => {
  it("breaks class ties towards the smaller label (scikit-learn)", () => {
    const X = f64([[0], [0.1], [0.2], [0.3]]);
    const y = tensor([1, 0, 1, 0]);
    const clf = new RadiusNeighborsClassifier({ radius: 1 }).fit(X, y);
    expect(flat(clf.predict(f64([[0.15]])))).toEqual([0]);
    expect(flat(clf.predictProba(f64([[0.15]])))).toEqual([0.5, 0.5]);
  });

  it("throws for samples without neighbors unless outlierLabel is set", () => {
    const X = f64([[0], [1], [2]]);
    const y = tensor([3, 7, 7]);
    const clf = new RadiusNeighborsClassifier({ radius: 0.5 }).fit(X, y);
    expect(() => clf.predict(f64([[50]]))).toThrow(DataValidationError);
    expect(() => clf.predictProba(f64([[50]]))).toThrow(DataValidationError);

    // scikit-learn: outlier_label="most_frequent" -> 7; outlier_label=9 -> 9 with all-zero probabilities.
    const mf = new RadiusNeighborsClassifier({ radius: 0.5, outlierLabel: "most_frequent" }).fit(
      X,
      y
    );
    expect(flat(mf.predict(f64([[50]])))).toEqual([7]);
    expect(rows(mf.predictProba(f64([[50], [1.1]])))).toEqual([
      [0, 1],
      [0, 1],
    ]);
    const nine = new RadiusNeighborsClassifier({ radius: 0.5, outlierLabel: 9 }).fit(X, y);
    expect(flat(nine.predict(f64([[50], [0.1]])))).toEqual([9, 3]);
    expect(rows(nine.predictProba(f64([[50]])))).toEqual([[0, 0]]);
  });

  it("supports distance weights like scikit-learn", () => {
    const X = f64([[0], [1], [2]]);
    const y = tensor([0, 1, 1]);
    const clf = new RadiusNeighborsClassifier({
      radius: 0.6,
      weights: "distance",
      outlierLabel: -1,
    }).fit(X, y);
    expect(flat(clf.predict(f64([[0], [1.4], [10]])))).toEqual([0, 1, -1]);

    const reg = new RadiusNeighborsRegressor({ radius: 1.5, weights: "distance" }).fit(
      f64([[0], [1], [3]]),
      f64([0, 10, 30])
    );
    const out = flat(reg.predict(f64([[0], [0.5], [2], [100]])));
    expectClose(out, [0, 5, 20, Number.NaN], 12);
  });

  it("regressor returns NaN (not 0) for samples without neighbors", () => {
    const reg = new RadiusNeighborsRegressor({ radius: 0.5 }).fit(f64([[0], [1]]), f64([4, 8]));
    const out = flat(reg.predict(f64([[0.2], [30]])));
    expect(out[0]).toBe(4);
    expect(Number.isNaN(out[1] as number)).toBe(true);
    expect(Number.isNaN(reg.score(f64([[0.2], [30]]), f64([4, 8])))).toBe(true);
  });

  it("score returns 1 for a perfect fit of a constant target", () => {
    const reg = new RadiusNeighborsRegressor({ radius: 1 }).fit(f64([[0], [1]]), f64([2, 2]));
    expect(reg.score(f64([[0], [1]]), f64([2, 2]))).toBe(1);
    expect(() => reg.score(f64([[0], [1]]), f64([2]))).toThrow(ShapeError);
  });

  it("validates parameters", () => {
    expect(() => new RadiusNeighborsClassifier({ radius: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new RadiusNeighborsRegressor({ radius: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new RadiusNeighborsClassifier({ weights: "x" as "uniform" })).toThrow(
      InvalidParameterError
    );
    expect(() => new RadiusNeighborsClassifier({ outlierLabel: Number.NaN })).toThrow(
      InvalidParameterError
    );
    const clf = new RadiusNeighborsClassifier({ radius: 2 });
    expect(() => clf.setParams({ k: 1 })).toThrow(InvalidParameterError);
    expect(clf.setParams({ radius: 3, weights: "distance" }).getParams()).toEqual({
      radius: 3,
      weights: "distance",
      outlierLabel: null,
    });
    expect(clf.clone().getParams()).toEqual(clf.getParams());
  });

  it("returns int32 labels for integer classes and keeps non-integer classes", () => {
    const X = f64([[0], [10]]);
    const ints = new RadiusNeighborsClassifier({ radius: 1 }).fit(X, tensor([2, 5]));
    expect(ints.predict(f64([[0.1]])).dtype).toBe("int32");
    const floats = new RadiusNeighborsClassifier({ radius: 1 }).fit(X, f64([0.5, 1.5]));
    expect(flat(floats.predict(f64([[10.2]])))).toEqual([1.5]);
  });
});

// ---------------------------------------------------------------------------
// Pipeline, FeatureUnion, ColumnTransformer
// ---------------------------------------------------------------------------
describe("Pipeline family (c24)", () => {
  const X = f64([
    [1, 10, 0.5],
    [2, 20, 1.5],
    [3, 30, 2.5],
    [4, 45, 2.0],
    [5, 50, 4.0],
    [6, 65, 3.5],
  ]);
  const y = f64([1, 2, 3, 4, 5, 6]);
  const yc = tensor([0, 0, 0, 1, 1, 1]);

  it("ColumnTransformer keeps float64 precision (it used to round through float32)", () => {
    const tricky = 0.1 + 1e-9;
    const data = f64([
      [tricky, 1],
      [0.2, 2],
      [0.3, 4],
    ]);
    const ct = new ColumnTransformer([
      ["p", "passthrough", [0]],
      ["s", new StandardScaler(), [1]],
    ]);
    const Xt = ct.fitTransform(data);
    expect(Xt.dtype).toBe("float64");
    expect(Number(Xt.data[0])).toBe(tricky);
    expect(Xt.shape).toEqual([3, 2]);
  });

  it("FeatureUnion output is float64 and concatenates in order", () => {
    const union = new FeatureUnion([
      ["a", new StandardScaler({ withStd: false })],
      ["b", new StandardScaler({ withMean: false, withStd: false })],
    ]);
    const data = f64([
      [0.1 + 1e-9, 2],
      [0.3, 4],
    ]);
    const out = union.fitTransform(data);
    expect(out.dtype).toBe("float64");
    expect(out.shape).toEqual([2, 4]);
    expect(Number(out.data[2])).toBe(0.1 + 1e-9); // second transformer is the identity
  });

  it("ColumnTransformer with no output columns gives an (n, 0) matrix (it used to throw)", () => {
    const ct = new ColumnTransformer([["d", "drop", [0]]]);
    const out = ct.fitTransform(
      f64([
        [1, 2],
        [3, 4],
      ])
    );
    expect(out.shape).toEqual([2, 0]);
  });

  it("ColumnTransformer skips specs with no columns and resolves negative indices", () => {
    const ct = new ColumnTransformer([["s", new StandardScaler(), []]], {
      remainder: "passthrough",
    });
    const out = ct.fitTransform(X);
    expect(rows(out)).toEqual(rows(X));

    const last = new ColumnTransformer([["p", "passthrough", [-1]]]).fitTransform(X);
    expect(flat(last)).toEqual([0.5, 1.5, 2.5, 2.0, 4.0, 3.5]);
  });

  it("ColumnTransformer validates columns and feature counts", () => {
    expect(() => new ColumnTransformer([["p", "passthrough", [0.5]]])).toThrow(
      InvalidParameterError
    );
    expect(() => new ColumnTransformer([["p", "keep" as "drop", [0]]])).toThrow(
      InvalidParameterError
    );
    expect(
      () => new ColumnTransformer([["p", "passthrough", [0]]], { remainder: "x" as "drop" })
    ).toThrow(InvalidParameterError);
    expect(() => new ColumnTransformer([["a__b", "passthrough", [0]]])).toThrow(
      InvalidParameterError
    );
    expect(() => new ColumnTransformer([["p", "passthrough", [5]]]).fit(X)).toThrow(
      InvalidParameterError
    );
    const ct = new ColumnTransformer([["p", "passthrough", [0]]]).fit(X);
    expect(() => ct.transform(f64([[1, 2]]))).toThrow(ShapeError);
  });

  it("ColumnTransformer / FeatureUnion setParams and clone", () => {
    const ct = new ColumnTransformer([
      ["s", new StandardScaler(), [0]],
      ["p", "passthrough", [1]],
    ]);
    ct.setParams({ s__withMean: false, remainder: "passthrough" });
    expect(ct.getParams()).toMatchObject({ remainder: "passthrough", s: { withMean: false } });
    ct.setParams({ p: "drop" });
    expect(ct.getParams()["p"]).toBe("drop");
    expect(() => ct.setParams({ p: "keep" })).toThrow(InvalidParameterError);
    expect(() => ct.setParams({ unknown: 1 })).toThrow(InvalidParameterError);
    expect(() => ct.setParams({ remainder: "x" })).toThrow(InvalidParameterError);

    const copy = ct.clone();
    expect(copy).toBeInstanceOf(ColumnTransformer);
    expect(copy.getParams()).toEqual(ct.getParams());
    expect(() => copy.transform(X)).toThrow(NotFittedError);

    const union = new FeatureUnion([["a", new StandardScaler()]]);
    union.setParams({ a: { withStd: false } });
    expect(union.getParams()).toEqual({ a: { withMean: true, withStd: false } });
    expect(() => union.setParams({ b__x: 1 })).toThrow(InvalidParameterError);
    expect(() => union.setParams({ a: 3 })).toThrow(InvalidParameterError);
    const unionCopy = union.clone();
    expect(unionCopy.transformerNames).toEqual(["a"]);
    expect(unionCopy.getParams()).toEqual(union.getParams());
  });

  it("Pipeline.setParams routes step parameters (name__param and nested objects)", () => {
    const pipe = new Pipeline([
      ["sc", new StandardScaler()],
      ["lr", new LinearRegression()],
    ]);
    pipe.setParams({ sc__withMean: false });
    expect(pipe.getStep("sc").getParams()["withMean"]).toBe(false);
    pipe.setParams({ sc: { withStd: false } });
    expect(pipe.getStep("sc").getParams()["withStd"]).toBe(false);
    expect(() => pipe.setParams({ nope__x: 1 })).toThrow(InvalidParameterError);
    expect(() => pipe.setParams({ nope: {} })).toThrow(InvalidParameterError);
    expect(() => pipe.setParams({ sc: 1 })).toThrow(InvalidParameterError);
  });

  it("GridSearchCV can tune a Pipeline through step__param keys", () => {
    const pipe = new Pipeline([
      ["sc", new StandardScaler()],
      ["lr", new LinearRegression()],
    ]);
    const gs = new GridSearchCV(pipe, { sc__withMean: [true, false] }, { cv: 3 });
    gs.fit(X, y);
    expect(typeof gs.bestParams["sc__withMean"]).toBe("boolean");
  });

  it("Pipeline reports the tags of its final step", () => {
    const reg = new Pipeline([
      ["sc", new StandardScaler()],
      ["lr", new LinearRegression()],
    ]);
    expect(getEstimatorTags(reg).estimatorType).toBe("regressor");
    const clf = new Pipeline([
      ["sc", new StandardScaler()],
      ["lr", new LogisticRegression({ maxIter: 50 })],
    ]);
    expect(getEstimatorTags(clf).estimatorType).toBe("classifier");
    expect(getEstimatorTags(clf).hasPredictProba).toBe(true);
  });

  it("a failed refit leaves the Pipeline unfitted", () => {
    const pipe = new Pipeline([
      ["sc", new StandardScaler()],
      ["lr", new LinearRegression()],
    ]);
    pipe.fit(X, y);
    expect(() => pipe.predict(X)).not.toThrow();
    expect(() => pipe.fit(X, f64([1, 2]))).toThrow();
    expect(() => pipe.predict(X)).toThrow(NotFittedError);
  });

  it("fitTransform checks the final step before fitting anything", () => {
    const scaler = new StandardScaler();
    const pipe = new Pipeline([
      ["sc", scaler],
      ["lr", new LinearRegression()],
    ]);
    expect(() => pipe.fitTransform(X, y)).toThrow(InvalidParameterError);
    expect(() => scaler.transform(X)).toThrow();
  });

  it("Pipeline.clone works for steps that are ColumnTransformers", () => {
    const pipe = new Pipeline([
      [
        "ct",
        new ColumnTransformer([
          ["s", new StandardScaler(), [0, 1]],
          ["p", "passthrough", [2]],
        ]),
      ],
      ["lr", new LinearRegression()],
    ]);
    const copy = pipe.clone();
    expect(copy.getStep("ct")).not.toBe(pipe.getStep("ct"));
    copy.fit(X, y);
    expect(copy.predict(X).shape).toEqual([6]);
    expect(() => pipe.predict(X)).toThrow(NotFittedError);
  });

  it("Pipeline supports inverseTransform, fitPredict and classes", () => {
    const inv = new Pipeline([
      ["a", new StandardScaler()],
      ["b", new StandardScaler()],
    ]);
    const Z = inv.fitTransform(X);
    expectClose(flat(inv.inverseTransform(Z)), flat(X), 9);

    const km = new Pipeline([
      ["sc", new StandardScaler()],
      ["km", new KMeans({ nClusters: 2, randomState: 0 })],
    ]);
    const labels = km.fitPredict(X);
    expect(labels.shape).toEqual([6]);
    expect(() => km.predict(X)).not.toThrow();

    const clf = new Pipeline([
      ["sc", new StandardScaler()],
      ["lr", new LogisticRegression({ maxIter: 50 })],
    ]);
    expect(() => clf.classes).toThrow(NotFittedError);
    clf.fit(X, yc);
    expect(clf.classes?.size).toBe(2);
  });

  it("validates step names and the final step", () => {
    expect(() => new Pipeline([["a__b", new StandardScaler()]])).toThrow(InvalidParameterError);
    expect(() => new Pipeline([["", new StandardScaler()]])).toThrow(InvalidParameterError);
    expect(() => new Pipeline([["a", {} as unknown as StandardScaler]])).toThrow(
      InvalidParameterError
    );
    expect(() => new FeatureUnion([["a", {} as unknown as StandardScaler]])).toThrow(
      InvalidParameterError
    );
  });

  it("makePipeline gives repeated estimators unique names", () => {
    const pipe = makePipeline(new StandardScaler(), new StandardScaler(), new StandardScaler());
    expect(pipe.stepNames).toEqual(["standardscaler", "standardscaler_1", "standardscaler_2"]);
  });
});

// ---------------------------------------------------------------------------
// GaussianRandomProjection
// ---------------------------------------------------------------------------
describe("GaussianRandomProjection (c24)", () => {
  afterEach(() => clearSeed());

  const data = (n: number, d: number): Tensor => {
    const arr = new Float64Array(n * d);
    for (let i = 0; i < arr.length; i++) arr[i] = Math.sin(i * 1.7) * 3;
    return tensor(arr).reshape([n, d]);
  };

  it("johnsonLindenstraussMinDim matches scikit-learn", () => {
    expect(johnsonLindenstraussMinDim(1e6, 0.5)).toBe(663);
    expect(johnsonLindenstraussMinDim(100, 0.1)).toBe(3947);
    expect(johnsonLindenstraussMinDim(1000, 0.3)).toBe(767);
    expect(johnsonLindenstraussMinDim(200, 0.5)).toBe(254);
    expect(() => johnsonLindenstraussMinDim(0)).toThrow(InvalidParameterError);
    expect(() => johnsonLindenstraussMinDim(10, 1)).toThrow(InvalidParameterError);
  });

  it('nComponents "auto" uses the Johnson-Lindenstrauss bound', () => {
    const proj = new GaussianRandomProjection({ nComponents: "auto", eps: 0.5, randomState: 1 });
    const X = data(200, 300);
    const out = proj.fitTransform(X);
    expect(out.shape).toEqual([200, 254]);
    expect(proj.components.shape).toEqual([254, 300]);
    // eps = 0.1 needs far more components than the 20 features available.
    expect(() => new GaussianRandomProjection({ nComponents: "auto" }).fit(data(100, 20))).toThrow(
      InvalidParameterError
    );
  });

  it("transform uses the fitted dimension even if setParams changed nComponents", () => {
    const proj = new GaussianRandomProjection({ nComponents: 3, randomState: 2 });
    const X = data(4, 6);
    proj.fit(X);
    proj.setParams({ nComponents: 5 });
    expect(proj.transform(X).shape).toEqual([4, 3]);
    expect(proj.fit(X).transform(X).shape).toEqual([4, 5]);
  });

  it("projection entries are N(0, 1/nComponents) and the output is float64", () => {
    const proj = new GaussianRandomProjection({ nComponents: 4, randomState: 7 });
    proj.fit(data(3, 2000));
    const comps = flat(proj.components);
    const mean = comps.reduce((a, b) => a + b, 0) / comps.length;
    const variance = comps.reduce((a, b) => a + (b - mean) ** 2, 0) / comps.length;
    expect(Math.abs(mean)).toBeLessThan(0.02);
    expect(Math.sqrt(variance)).toBeGreaterThan(0.5 * 0.95);
    expect(Math.sqrt(variance)).toBeLessThan(0.5 * 1.05);
    expect(proj.transform(data(3, 2000)).dtype).toBe("float64");
  });

  it("transform equals X @ components^T", () => {
    const proj = new GaussianRandomProjection({ nComponents: 2, randomState: 3 });
    const X = f64([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const out = rows(proj.fit(X).transform(X));
    const comps = rows(proj.components);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const expected = [0, 1, 2].reduce(
          (s, f) => s + (rows(X)[i]?.[f] as number) * ((comps[j] as number[])[f] as number),
          0
        );
        expect((out[i] as number[])[j]).toBeCloseTo(expected, 12);
      }
    }
  });

  it("randomState and the deprecated seed give the same stream; setSeed controls the global stream", () => {
    const X = data(5, 8);
    const a = new GaussianRandomProjection({ nComponents: 3, randomState: 11 }).fit(X).components;
    const b = new GaussianRandomProjection({ nComponents: 3, seed: 11 }).fit(X).components;
    expect(flat(a)).toEqual(flat(b));
    const c = new GaussianRandomProjection({ nComponents: 3, randomState: 12 }).fit(X).components;
    expect(flat(c)).not.toEqual(flat(a));
    expect(() => new GaussianRandomProjection({ randomState: 1, seed: 2 })).toThrow(
      InvalidParameterError
    );

    setSeed(5);
    const g1 = new GaussianRandomProjection({ nComponents: 3 }).fit(X).components;
    setSeed(5);
    const g2 = new GaussianRandomProjection({ nComponents: 3 }).fit(X).components;
    expect(flat(g1)).toEqual(flat(g2));
  });

  it("validates parameters", () => {
    expect(() => new GaussianRandomProjection({ nComponents: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new GaussianRandomProjection({ eps: 1 })).toThrow(InvalidParameterError);
    expect(() => new GaussianRandomProjection({ randomState: Number.NaN })).toThrow(
      InvalidParameterError
    );
    const proj = new GaussianRandomProjection();
    expect(() => proj.setParams({ nComponents: 0 })).toThrow(InvalidParameterError);
    expect(() => proj.setParams({ bogus: 1 })).toThrow(InvalidParameterError);
    expect(() => proj.components).toThrow(NotFittedError);
    expect(proj.clone().getParams()).toEqual(proj.getParams());
  });
});

// ---------------------------------------------------------------------------
// Semi-supervised
// ---------------------------------------------------------------------------
describe("LabelPropagation / LabelSpreading (c24)", () => {
  const X = f64([
    [0, 0],
    [0.5, 0],
    [0, 0.5],
    [1, 1],
    [2, 2],
    [3, 3],
    [3.5, 3],
    [4, 4],
    [1.5, 1.2],
    [2.5, 2.8],
  ]);
  const y = tensor([0, -1, -1, -1, -1, 1, -1, -1, -1, -1]);
  const Xt = f64([
    [0.2, 0.1],
    [2.2, 2.1],
  ]);

  // Reference values from scikit-learn 1.8 (gamma=0.5).
  it("LabelPropagation matches scikit-learn", () => {
    const lp = new LabelPropagation({ gamma: 0.5 }).fit(X, y);
    expectClose(
      flat(lp.labelDistributions),
      [
        1.0, 0.0, 0.826123, 0.173877, 0.828831, 0.171169, 0.698618, 0.301382, 0.377101, 0.622899,
        0.0, 1.0, 0.114877, 0.885123, 0.091846, 0.908154, 0.603994, 0.396006, 0.190964, 0.809036,
      ],
      5
    );
    expectClose(flat(lp.predictProba(Xt)), [0.837864, 0.162136, 0.325705, 0.674295], 5);
    expect(flat(lp.predict(Xt))).toEqual([0, 1]);
    expect(lp.nIter).toBe(24);
    expect(flat(lp.transduction)).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 0, 1]);
  });

  it("LabelSpreading matches scikit-learn", () => {
    const ls = new LabelSpreading({ gamma: 0.5, alpha: 0.2 }).fit(X, y);
    expectClose(
      flat(ls.labelDistributions),
      [
        0.999635, 0.000365, 0.992751, 0.007249, 0.993042, 0.006958, 0.914789, 0.085211, 0.106725,
        0.893275, 0.000365, 0.999635, 0.002534, 0.997466, 0.001079, 0.998921, 0.695616, 0.304384,
        0.010692, 0.989308,
      ],
      5
    );
    expectClose(flat(ls.predictProba(Xt)), [0.956071, 0.043929, 0.230935, 0.769065], 5);
    expect(ls.nIter).toBe(4);
  });

  it("predictProba is finite for samples far from all training data (raw kernel underflows)", () => {
    const lp = new LabelPropagation({ gamma: 20 }).fit(X, y);
    const proba = rows(
      lp.predictProba(
        f64([
          [10, 10],
          [-5, -5],
        ])
      )
    );
    for (const row of proba) {
      expect(row.every(Number.isFinite)).toBe(true);
      expect(row[0]! + row[1]!).toBeCloseTo(1, 12);
    }
    // The nearest training samples are (4, 4) -> class 1 and (0, 0) -> class 0.
    expectClose(proba[0] as number[], [0, 1], 12);
    expectClose(proba[1] as number[], [1, 0], 12);
    expect(
      flat(
        lp.predict(
          f64([
            [10, 10],
            [-5, -5],
          ])
        )
      )
    ).toEqual([1, 0]);
  });

  it("uses scikit-learn's default maxIter for LabelPropagation", () => {
    expect(new LabelPropagation().getParams()["maxIter"]).toBe(1000);
    expect(new LabelSpreading().getParams()["maxIter"]).toBe(30);
  });

  it("needs at least one labeled sample and validates parameters", () => {
    const allUnlabeled = tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]);
    expect(() => new LabelPropagation().fit(X, allUnlabeled)).toThrow(DataValidationError);
    expect(() => new LabelSpreading().fit(X, allUnlabeled)).toThrow(DataValidationError);
    expect(() => new LabelPropagation({ gamma: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new LabelPropagation({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => new LabelSpreading({ alpha: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new LabelSpreading().setParams({ nope: 1 })).toThrow(InvalidParameterError);
    expect(() => new LabelPropagation().setParams({ gamma: -1 })).toThrow(InvalidParameterError);
    const ls = new LabelSpreading({ alpha: 0.4 });
    expect(ls.setParams({ alpha: 0.6 }).getParams()["alpha"]).toBe(0.6);
    expect(ls.clone().getParams()).toEqual(ls.getParams());
  });

  it("score checks the target shape", () => {
    const lp = new LabelPropagation({ gamma: 0.5 }).fit(X, y);
    expect(() => lp.score(Xt, tensor([0]))).toThrow(ShapeError);
    expect(lp.score(Xt, tensor([0, 1]))).toBe(1);
    expect(() => new LabelPropagation().predict(Xt)).toThrow(NotFittedError);
  });
});

/** Base classifier with a fixed second-class probability that records what it was trained on. */
class FixedProbaClassifier {
  static lastX: Tensor | undefined;
  static lastY: Tensor | undefined;
  fitCalls = 0;
  constructor(private readonly options: { p?: number } = {}) {}
  fit(X: Tensor, y: Tensor): this {
    this.fitCalls++;
    FixedProbaClassifier.lastX = X;
    FixedProbaClassifier.lastY = y;
    return this;
  }
  predict(X: Tensor): Tensor {
    return tensor(new Float64Array(X.shape[0] ?? 0).fill(1));
  }
  predictProba(X: Tensor): Tensor {
    const n = X.shape[0] ?? 0;
    const p = this.options.p ?? 0.75;
    const out = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      out[2 * i] = 1 - p;
      out[2 * i + 1] = p;
    }
    return tensor(out).reshape([n, 2]);
  }
  score(): number {
    return 1;
  }
  getParams(): Record<string, unknown> {
    return { p: this.options.p };
  }
  setParams(): this {
    return this;
  }
}

describe("SelfTrainingClassifier (c24)", () => {
  const X = f64([[0.1], [0.9], [0.3], [0.7]]);
  const y = tensor([0, 1, -1, -1]);
  const make = (p: number, extra: Record<string, unknown> = {}) => {
    const base = new FixedProbaClassifier({ p });
    const st = new SelfTrainingClassifier({
      baseEstimator: base as unknown as Classifier,
      ...extra,
    });
    return { base, st };
  };

  it("only accepts probabilities strictly above the threshold, like scikit-learn", () => {
    const { st } = make(0.75, { threshold: 0.75 });
    st.fit(X, y);
    expect(st.terminationCondition).toBe("no_change");
    expect(flat(st.transduction)).toEqual([0, 1, -1, -1]);
    expect(Array.from(st.labeledIterations)).toEqual([0, 0, -1, -1]);
    expect(st.nIterations).toBe(1);
  });

  it("records labeledIterations / transduction / terminationCondition", () => {
    const { st } = make(0.75, { threshold: 0.7 });
    st.fit(X, y);
    expect(st.terminationCondition).toBe("all_labeled");
    expect(flat(st.transduction)).toEqual([0, 1, 1, 1]);
    expect(Array.from(st.labeledIterations)).toEqual([0, 0, 1, 1]);
    expect(st.nIterations).toBe(1);
    // Deprecated accessor keeps its original encoding.
    expect(Array.from(st.transductionLabels ?? [])).toEqual([-1, -1, 0, 0]);
  });

  it("stops at maxIter and supports the kBest criterion", () => {
    const { st } = make(0.9, { criterion: "kBest", kBest: 1, maxIter: 1 });
    st.fit(X, y);
    expect(st.terminationCondition).toBe("max_iter");
    expect(Array.from(st.labeledIterations)).toEqual([0, 0, 1, -1]);
    expect(flat(st.transduction)).toEqual([0, 1, 1, -1]);
  });

  it("trains the base estimator on float64 data and leaves the given instance untouched", () => {
    const { base, st } = make(0.9);
    st.fit(f64([[0.1 + 1e-9], [0.9], [0.3], [0.7]]), y);
    expect(base.fitCalls).toBe(0);
    const trained = FixedProbaClassifier.lastX as Tensor;
    expect(trained.dtype).toBe("float64");
    expect(Number(trained.data[trained.offset])).toBe(0.1 + 1e-9);
    expect(FixedProbaClassifier.lastY?.dtype).toBe("float64");
    expect((st.estimator as unknown as FixedProbaClassifier).fitCalls).toBeGreaterThan(0);
  });

  it("does not modify the labels passed in", () => {
    const labels = tensor([0, 1, -1, -1]);
    make(0.9).st.fit(X, labels);
    expect(Array.from(labels.data as Int32Array)).toEqual([0, 1, -1, -1]);
  });

  it("works with a real base estimator and exposes clone/setParams", () => {
    const Xr = f64([
      [0, 0],
      [0.2, 0.1],
      [0.1, 0.3],
      [0.3, 0.2],
      [5, 5],
      [5.2, 5.1],
      [5.1, 5.3],
      [5.3, 5.2],
    ]);
    const yr = tensor([0, -1, -1, -1, 1, -1, -1, -1]);
    const st = new SelfTrainingClassifier({
      baseEstimator: new KNeighborsClassifier({ nNeighbors: 1 }),
      threshold: 0.5,
    }).fit(Xr, yr);
    expect(flat(st.predict(Xr))).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(flat(st.transduction)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(st.score(Xr, tensor([0, 0, 0, 0, 1, 1, 1, 1]))).toBe(1);
    expect(flat(st.classes)).toEqual([0, 1]);

    st.setParams({ baseEstimator__nNeighbors: 2, threshold: 0.6 });
    expect(st.getParams()["threshold"]).toBe(0.6);
    expect(
      (st.getParams()["baseEstimator"] as KNeighborsClassifier).getParams()["nNeighbors"]
    ).toBe(2);
    const copy = st.clone();
    expect(copy.getParams()["baseEstimator"]).not.toBe(st.getParams()["baseEstimator"]);
    expect(() => copy.predict(Xr)).toThrow(NotFittedError);
  });

  it("validates parameters and labels", () => {
    const base = new KNeighborsClassifier();
    expect(() => new SelfTrainingClassifier({ baseEstimator: base, threshold: 2 })).toThrow(
      InvalidParameterError
    );
    expect(
      () => new SelfTrainingClassifier({ baseEstimator: base, criterion: "x" as "kBest" })
    ).toThrow(InvalidParameterError);
    expect(() => new SelfTrainingClassifier({ baseEstimator: base, kBest: 0 })).toThrow(
      InvalidParameterError
    );
    expect(
      () => new SelfTrainingClassifier({ baseEstimator: {} as unknown as Classifier })
    ).toThrow(InvalidParameterError);
    expect(() => make(0.9).st.fit(X, tensor([0, 0, -1, -1]))).toThrow(InvalidParameterError);
    expect(() => make(0.9).st.setParams({ nope: 1 })).toThrow(InvalidParameterError);
    expect(() => make(0.9).st.predict(X)).toThrow(NotFittedError);
  });
});
