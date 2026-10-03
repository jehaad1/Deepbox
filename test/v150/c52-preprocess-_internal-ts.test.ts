import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import { CSRMatrix, type Tensor, tensor, zeros } from "../../src/ndarray";
import {
  KBinsDiscretizer,
  LabelBinarizer,
  LabelEncoder,
  MultiLabelBinarizer,
  OneHotEncoder,
  OrdinalEncoder,
  TargetEncoder,
} from "../../src/preprocess";
import { assertNumericTensor, createSeededRandom } from "../../src/preprocess/_internal";

// Reference values come from scikit-learn 1.8 / NumPy 2.4 / SciPy 1.17.

const COL = [
  6.072, -0.398, 1.098, 2.223, -1.367, 1.006, 0.997, -4.264, 4.053, 2.801, -0.876, 0.485, 2.516,
  0.216, 0.272, -3.36, 2.664,
];

function column(values: number[]): Tensor {
  return tensor(
    values.map((v) => [v]),
    { dtype: "float64" }
  );
}

function flat(t: Tensor): number[] {
  return (t.toArray() as number[][]).map((r) => r[0] as number);
}

describe("createSeededRandom", () => {
  it("matches the exact 31-bit LCG (Python integer arithmetic)", () => {
    // state = (1103515245 * state + 12345) % 2**31, seed 42, value = state / 2**31
    const expected = [
      0.5823075897060335, 0.5198187492787838, 0.46597642498090863, 0.7770372582599521,
      0.42286502895876765,
    ];
    const rnd = createSeededRandom(42);
    for (const e of expected) expect(rnd()).toBe(e);
  });

  it("does not collapse into a short cycle", () => {
    // The old floating-point multiply lost low bits and repeated after ~15.8k draws.
    const rnd = createSeededRandom(42);
    const seen = new Set<number>();
    for (let i = 0; i < 100000; i++) seen.add(rnd());
    expect(seen.size).toBe(100000);
  });

  it("produces odd and even states (low bits are alive)", () => {
    const rnd = createSeededRandom(1);
    let odd = 0;
    for (let i = 0; i < 2000; i++) if ((rnd() * 2 ** 31) % 2 === 1) odd++;
    expect(odd).toBeGreaterThan(800);
    expect(odd).toBeLessThan(1200);
  });

  it("rejects invalid seeds", () => {
    expect(() => createSeededRandom(-1)).toThrow(InvalidParameterError);
    expect(() => createSeededRandom(1.5)).toThrow(InvalidParameterError);
    expect(() => createSeededRandom(Number.NaN)).toThrow(InvalidParameterError);
  });
});

describe("assertNumericTensor", () => {
  it("rejects string tensors and accepts numeric ones", () => {
    expect(() => assertNumericTensor(tensor(["a"]), "X")).toThrow(DTypeError);
    expect(() => assertNumericTensor(tensor([1, 2]), "X")).not.toThrow();
  });
});

describe("KBinsDiscretizer: scikit-learn parity", () => {
  it("puts values on an inner edge into the upper bin", () => {
    // sklearn: edges [0, 2, 4] -> transform [0, 0, 1, 1, 1]
    const X = column([0, 1, 2, 3, 4]);
    for (const strategy of ["quantile", "uniform"] as const) {
      const kbd = new KBinsDiscretizer({ nBins: 2, strategy }).fit(X);
      expect(kbd.binEdges).toEqual([[0, 2, 4]]);
      expect(flat(kbd.transform(X))).toEqual([0, 0, 1, 1, 1]);
      expect(flat(kbd.transform(column([-5, 9])))).toEqual([0, 1]);
    }
  });

  it("matches np.percentile edges and bins (quantile)", () => {
    const kbd = new KBinsDiscretizer({ nBins: 4, strategy: "quantile" }).fit(column(COL));
    const edges = kbd.binEdges[0] as number[];
    const ref = [-4.264, -0.398, 0.997, 2.516, 6.072];
    for (let i = 0; i < ref.length; i++) expect(edges[i]).toBeCloseTo(ref[i] as number, 12);
    expect(flat(kbd.transform(column(COL)))).toEqual([
      3, 1, 2, 2, 0, 2, 2, 0, 3, 3, 0, 1, 3, 1, 1, 0, 3,
    ]);
    const inv = flat(kbd.inverseTransform(column([0, 1, 2, 3])));
    const refInv = [-2.331, 0.2995, 1.7565, 4.294];
    for (let i = 0; i < 4; i++) expect(inv[i]).toBeCloseTo(refInv[i] as number, 12);
  });

  it("matches np.linspace edges and bins (uniform) with the last edge equal to max", () => {
    const kbd = new KBinsDiscretizer({ nBins: 4, strategy: "uniform" }).fit(column(COL));
    const edges = kbd.binEdges[0] as number[];
    const ref = [-4.264, -1.68, 0.904, 3.488, 6.072];
    for (let i = 0; i < ref.length; i++) expect(edges[i]).toBeCloseTo(ref[i] as number, 12);
    expect(edges[4]).toBe(6.072);
    expect(flat(kbd.transform(column(COL)))).toEqual([
      3, 1, 2, 2, 1, 2, 2, 0, 3, 2, 1, 1, 2, 1, 1, 0, 2,
    ]);
  });

  it("uses np.percentile's interpolation position so edge values land in the same bin", () => {
    // sklearn: edges [-1.6, -0.5, 0.5, 0.6, 1.4000000000000001, 1.7], bins [1, 3, 3, 2, 4, 1, 0]
    const x = [-0.5, 0.6, 1.4, 0.5, 1.7, -0.5, -1.6];
    const kbd = new KBinsDiscretizer({ nBins: 6 }).fit(column(x));
    expect(kbd.binEdges).toEqual([[-1.6, -0.5, 0.5, 0.6, 1.4000000000000001, 1.7]]);
    expect(flat(kbd.transform(column(x)))).toEqual([1, 3, 3, 2, 4, 1, 0]);
  });

  it("merges collapsed quantile bins", () => {
    // sklearn: edges [1, 2, 5], n_bins_ 2, transform [0, 0, 0, 1, 1]
    const X = column([1, 1, 1, 2, 5]);
    const kbd = new KBinsDiscretizer({ nBins: 4 }).fit(X);
    expect(kbd.binEdges).toEqual([[1, 2, 5]]);
    expect(kbd.nBinsPerFeature).toEqual([2]);
    expect(flat(kbd.transform(X))).toEqual([0, 0, 0, 1, 1]);
  });

  it("gives a constant feature a single bin with infinite edges", () => {
    // sklearn: edges [-inf, inf], n_bins_ 1, transform zeros
    const X = column([3, 3, 3]);
    for (const strategy of ["quantile", "uniform"] as const) {
      const kbd = new KBinsDiscretizer({ nBins: 3, strategy }).fit(X);
      expect(kbd.binEdges).toEqual([[Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]]);
      expect(flat(kbd.transform(X))).toEqual([0, 0, 0]);
    }
  });

  it("supports per-feature nBins and onehot-dense output", () => {
    const X = tensor(
      COL.map((v, i) => [v, i * i]),
      { dtype: "float64" }
    );
    const kbd = new KBinsDiscretizer({
      nBins: [3, 2],
      strategy: "uniform",
      encode: "onehot-dense",
    });
    const out = kbd.fitTransform(X);
    // sklearn onehot-dense reference (3 + 2 columns)
    expect(out.shape).toEqual([17, 5]);
    const rows = out.toArray() as number[][];
    expect(rows[0]).toEqual([0, 0, 1, 1, 0]);
    expect(rows[4]).toEqual([1, 0, 0, 1, 0]);
    expect(rows[16]).toEqual([0, 0, 1, 0, 1]);
    const edges = kbd.binEdges;
    expect(edges[1]).toEqual([0, 128, 256]);
    expect((edges[0] as number[])[1]).toBeCloseTo(-0.8186666666666667, 12);
    // inverse of the one-hot blocks gives the bin centers
    const inv = kbd.inverseTransform(out).toArray() as number[][];
    expect(inv[0]?.[1]).toBe(64);
    expect(inv[16]?.[1]).toBe(192);
  });

  it("rejects NaN and Infinity instead of producing garbage bins", () => {
    expect(() => new KBinsDiscretizer().fit(column([1, Number.NaN, 3]))).toThrow(
      DataValidationError
    );
    expect(() => new KBinsDiscretizer({ nBins: 2 }).fit(column([1, Infinity]))).toThrow(
      DataValidationError
    );
    const fitted = new KBinsDiscretizer({ nBins: 2 }).fit(column([1, 2, 3]));
    expect(() => fitted.transform(column([Number.NaN]))).toThrow(DataValidationError);
  });

  it("validates options and input shapes", () => {
    expect(() => new KBinsDiscretizer({ nBins: 2.5 })).toThrow(InvalidParameterError);
    expect(() => new KBinsDiscretizer({ nBins: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new KBinsDiscretizer({ nBins: [] })).toThrow(InvalidParameterError);
    expect(() => new KBinsDiscretizer({ nBins: [3, 1] })).toThrow(InvalidParameterError);
    expect(() => new KBinsDiscretizer({ encode: "sparse" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => new KBinsDiscretizer({ nBins: [2, 3] }).fit(column([1, 2, 3]))).toThrow(
      InvalidParameterError
    );
    expect(() => new KBinsDiscretizer().fit(tensor([[]]))).toThrow(InvalidParameterError);
    expect(() => new KBinsDiscretizer().fit(tensor([[1], [2]]).reshape([2, 1, 1]))).toThrow(
      ShapeError
    );
    expect(() => new KBinsDiscretizer().inverseTransform(column([0]))).toThrow(NotFittedError);
  });

  it("does not expose internal edge arrays and keeps old state after a failed refit", () => {
    const kbd = new KBinsDiscretizer({ nBins: 2, strategy: "uniform" }).fit(column([0, 4]));
    kbd.binEdges[0]?.fill(99);
    expect(kbd.binEdges).toEqual([[0, 2, 4]]);
    expect(() => kbd.fit(column([1, Number.NaN]))).toThrow(DataValidationError);
    expect(kbd.binEdges).toEqual([[0, 2, 4]]);
  });

  it("rejects invalid bin indices in inverseTransform", () => {
    const kbd = new KBinsDiscretizer({ nBins: 2, strategy: "uniform" }).fit(column([0, 4]));
    expect(() => kbd.inverseTransform(column([2]))).toThrow(InvalidParameterError);
    expect(() => kbd.inverseTransform(column([0.5]))).toThrow(InvalidParameterError);
  });
});

describe("Encoders: category ordering and input coercion", () => {
  it("sorts string classes by code point like NumPy, not by locale", () => {
    // sklearn: LabelEncoder().fit(["b","A","a","B"]).classes_ == ["A","B","a","b"]
    const le = new LabelEncoder().fit(["b", "A", "a", "B"]);
    expect(le.classes?.toArray()).toEqual(["A", "B", "a", "b"]);
    expect((le.transform(["a", "B"]).toArray() as number[]).map(Number)).toEqual([2, 1]);
  });

  it("orders astral characters above BMP characters", () => {
    const le = new LabelEncoder().fit(["\u{1F600}", "￮", "a"]);
    expect(le.classes?.toArray()).toEqual(["a", "￮", "\u{1F600}"]);
  });

  it("keeps bigint arrays exact (no float64 round trip)", () => {
    const big = 9007199254740993n;
    const le = new LabelEncoder().fit([big, 1n, big + 2n]);
    const back = le.inverseTransform([0, 1, 2]);
    expect(back.dtype).toBe("int64");
    expect(Array.from(back.data as BigInt64Array)).toEqual([1n, big, big + 2n]);
    expect((le.transform([big]).toArray() as number[]).map(Number)).toEqual([1]);
  });

  it("treats strings mixed with other values as strings, in either order", () => {
    const a = new LabelEncoder().fit([1, "a"] as never);
    const b = new LabelEncoder().fit(["a", 1] as never);
    expect(a.classes?.toArray()).toEqual(["1", "a"]);
    expect(b.classes?.toArray()).toEqual(["1", "a"]);
    const oh = new OneHotEncoder().fit([
      ["red", 1],
      ["blue", 2],
    ] as never);
    expect(oh.categories).toEqual([
      ["blue", "red"],
      ["1", "2"],
    ]);
  });

  it("rejects number/bigint mixtures, non-array input and ragged rows", () => {
    expect(() => new LabelEncoder().fit([1, 2n] as never)).toThrow(InvalidParameterError);
    expect(() => new OneHotEncoder().fit([[1, 2n]] as never)).toThrow(InvalidParameterError);
    expect(() => new LabelEncoder().fit([{}] as never)).toThrow(InvalidParameterError);
    expect(() => new LabelEncoder().fit(null as never)).toThrow(InvalidParameterError);
    expect(() => new OneHotEncoder().fit([[1, 2], [3]] as never)).toThrow(ShapeError);
    expect(() => new OneHotEncoder().fit([1, 2, 3] as never)).toThrow(ShapeError);
  });

  it("treats booleans as 0/1", () => {
    const le = new LabelEncoder().fit([true, false, true]);
    expect(le.classes?.toArray()).toEqual([0, 1]);
  });

  it("exposes learned categories", () => {
    const oh = new OneHotEncoder();
    expect(oh.categories).toBeUndefined();
    oh.fit([
      ["red", "S"],
      ["blue", "M"],
    ]);
    expect(oh.categories).toEqual([
      ["blue", "red"],
      ["M", "S"],
    ]);
    const ord = new OrdinalEncoder().fit([[3], [1], [2]]);
    expect(ord.categories).toEqual([[1, 2, 3]]);
    // the getter returns copies
    ord.categories?.[0]?.push(99);
    expect(ord.categories).toEqual([[1, 2, 3]]);
  });
});

describe("OneHotEncoder", () => {
  it("matches the documented example", () => {
    // sklearn: categories [blue, red] and [L, M, S]
    const out = new OneHotEncoder().fitTransform([
      ["red", "S"],
      ["blue", "M"],
      ["red", "L"],
    ]) as Tensor;
    expect(out.toArray()).toEqual([
      [0, 1, 0, 0, 1],
      [1, 0, 0, 1, 0],
      [0, 1, 1, 0, 0],
    ]);
  });

  it("allows unlisted training values with explicit categories when handleUnknown is ignore", () => {
    // sklearn: OneHotEncoder(handle_unknown="ignore", categories=[["a","b"]]).fit([["a"],["c"]])
    // transform([["a"],["c"]]) == [[1, 0], [0, 0]]
    const enc = new OneHotEncoder({ handleUnknown: "ignore", categories: [["a", "b"]] });
    const out = enc.fit([["a"], ["c"]]).transform([["a"], ["c"]]) as Tensor;
    expect(out.toArray()).toEqual([
      [1, 0],
      [0, 0],
    ]);
    const strict = new OneHotEncoder({ categories: [["a", "b"]] });
    expect(() => strict.fit([["a"], ["c"]])).toThrow(InvalidParameterError);
  });

  it("inverse transform is not fooled by a leading NaN", () => {
    const enc = new OneHotEncoder().fit([["a"], ["b"], ["c"]]);
    const dec = enc.inverseTransform(tensor([[Number.NaN, 0, 1]], { dtype: "float64" }));
    expect(dec.toArray()).toEqual([["c"]]);
    expect(() =>
      enc.inverseTransform(tensor([[Number.NaN, Number.NaN, Number.NaN]], { dtype: "float64" }))
    ).toThrow(InvalidParameterError);
  });

  it("keeps the previous fit when a refit fails", () => {
    const enc = new OneHotEncoder({ categories: [["a", "b"]] }).fit([["a"], ["b"]]);
    expect(() => enc.fit([["z"]])).toThrow(InvalidParameterError);
    expect((enc.transform([["b"]]) as Tensor).toArray()).toEqual([[0, 1]]);
  });

  it("reports the feature of an unknown category", () => {
    const enc = new OneHotEncoder().fit([
      ["a", "x"],
      ["b", "y"],
    ]);
    expect(() => enc.transform([["a", "q"]])).toThrow(/feature 1/);
  });

  it("produces identical dense and sparse output with drop", () => {
    const X = [
      ["a", 1],
      ["b", 2],
      ["c", 1],
    ].map((r) => r.map(String));
    const dense = new OneHotEncoder({ drop: "first" }).fitTransform(X) as Tensor;
    const sparse = new OneHotEncoder({ drop: "first", sparse: true }).fitTransform(X);
    expect(sparse).toBeInstanceOf(CSRMatrix);
    expect((sparse as CSRMatrix).toDense().toArray()).toEqual(dense.toArray());
    // sklearn drop="first": [[0,0,0],[1,0,1],[0,1,0]]
    expect(dense.toArray()).toEqual([
      [0, 0, 0],
      [1, 0, 1],
      [0, 1, 0],
    ]);
  });

  it("round-trips non-contiguous input views", () => {
    const base = tensor([
      ["a", "x", "p"],
      ["b", "y", "q"],
      ["a", "x", "r"],
    ]);
    const view = base.slice({ start: 0, end: 3 }, { start: 0, end: 3, step: 2 });
    const enc = new OneHotEncoder();
    const out = enc.fitTransform(view) as Tensor;
    expect(enc.inverseTransform(out).toArray()).toEqual([
      ["a", "p"],
      ["b", "q"],
      ["a", "r"],
    ]);
  });
});

describe("OrdinalEncoder", () => {
  it("allows unlisted training values with explicit categories when unknowns are encoded", () => {
    const enc = new OrdinalEncoder({
      handleUnknown: "useEncodedValue",
      unknownValue: -1,
      categories: [["a", "b"]],
    });
    const out = enc.fit([["a"], ["c"]]).transform([["b"], ["c"]]);
    expect(out.toArray()).toEqual([[1], [-1]]);
  });

  it("rejects zero-feature input like the other encoders", () => {
    expect(() => new OrdinalEncoder().fit(tensor([[]]))).toThrow(InvalidParameterError);
  });

  it("keeps the previous fit when a refit fails", () => {
    const enc = new OrdinalEncoder({ categories: [["a", "b"]] }).fit([["a"]]);
    expect(() => enc.fit([["q"]])).toThrow(InvalidParameterError);
    expect(enc.transform([["b"]]).toArray()).toEqual([[1]]);
  });

  it("round-trips bigint categories", () => {
    const enc = new OrdinalEncoder();
    const out = enc.fitTransform([[5n], [2n], [5n]]);
    expect(out.toArray()).toEqual([[1], [0], [1]]);
    expect(enc.inverseTransform(out).dtype).toBe("int64");
  });
});

describe("LabelBinarizer / MultiLabelBinarizer", () => {
  it("inverse transform skips NaN instead of returning class 0", () => {
    const lb = new LabelBinarizer().fit([0, 1, 2]);
    const dec = lb.inverseTransform(tensor([[Number.NaN, 0, 1]], { dtype: "float64" }));
    expect(dec.toArray()).toEqual([2]);
    expect(() =>
      lb.inverseTransform(tensor([[Number.NaN, Number.NaN, Number.NaN]], { dtype: "float64" }))
    ).toThrow(InvalidParameterError);
  });

  it("honors posLabel/negLabel for dense output and exposes classes", () => {
    const lb = new LabelBinarizer({ posLabel: 5, negLabel: -1 });
    const out = lb.fitTransform(["b", "a", "b"]) as Tensor;
    expect(out.toArray()).toEqual([
      [-1, 5],
      [5, -1],
      [-1, 5],
    ]);
    expect(lb.classes?.toArray()).toEqual(["a", "b"]);
    expect(new LabelBinarizer().classes).toBeUndefined();
  });

  it("transforms to sparse with the right values", () => {
    const lb = new LabelBinarizer({ sparse: true, posLabel: 3 });
    const out = lb.fitTransform([0, 2, 1, 2]) as CSRMatrix;
    expect(out.toDense().toArray()).toEqual([
      [3, 0, 0],
      [0, 0, 3],
      [0, 3, 0],
      [0, 0, 3],
    ]);
  });

  it("MultiLabelBinarizer keeps the previous fit when a refit fails", () => {
    const mlb = new MultiLabelBinarizer({ classes: ["a", "b"] }).fit([["a"], ["b"]]);
    expect(() => mlb.fit([["a", "z"]])).toThrow(InvalidParameterError);
    expect((mlb.transform([["a", "a", "b"]]) as Tensor).toArray()).toEqual([[1, 1]]);
    expect(mlb.classes?.toArray()).toEqual(["a", "b"]);
  });

  it("MultiLabelBinarizer matches sklearn on a multi-label example", () => {
    // sklearn: classes_ ['action', 'comedy', 'drama', 'sci-fi']
    const y = [["sci-fi", "action"], ["comedy"], ["action", "drama"]];
    const mlb = new MultiLabelBinarizer();
    const out = mlb.fitTransform(y) as Tensor;
    expect(out.toArray()).toEqual([
      [1, 0, 0, 1],
      [0, 1, 0, 0],
      [1, 0, 1, 0],
    ]);
    expect(mlb.inverseTransform(out)).toEqual([
      ["action", "sci-fi"],
      ["comedy"],
      ["action", "drama"],
    ]);
    const sparse = new MultiLabelBinarizer({ sparse: true }).fitTransform(y) as CSRMatrix;
    expect(sparse.toDense().toArray()).toEqual(out.toArray());
  });
});

describe("TargetEncoder", () => {
  const X = [[0], [1], [0], [1], [2], [0], [1], [2], [0], [1]];
  const y = [10, 20, 12, 18, 15, 11, 19, 14, 13, 17];

  it("encodes with smoothed category means in float64", () => {
    // NumPy: global mean 14.9; (sum + 5 * 14.9) / (count + 5) per category
    const enc = new TargetEncoder({ smooth: 5 }).fit(X, y);
    const out = enc.transform([[0], [1], [2], [7]]);
    expect(out.dtype).toBe("float64");
    const vals = flat(out);
    expect(vals[0]).toBeCloseTo(13.38888888888889, 12);
    expect(vals[1]).toBeCloseTo(16.5, 12);
    expect(vals[2]).toBeCloseTo(14.785714285714286, 12);
    // unseen category falls back to the target mean
    expect(vals[3]).toBeCloseTo(14.9, 12);
    expect(enc.targetMean).toBeCloseTo(14.9, 12);
  });

  it("uses out-of-fold statistics and the training-fold mean in fitTransform", () => {
    // NumPy reference with folds i % 5, per-fold training mean as the fallback:
    //   for each held-out fold h, m = mean(y[train]); encoding = (sum_c + 5 m) / (n_c + 5)
    const expected = [
      15.0, 14.821428571428571, 13.859375, 16.21875, 14.520833333333334, 15.0, 14.821428571428571,
      15.3125, 13.34375, 16.265625,
    ];
    const enc = new TargetEncoder({ smooth: 5 });
    const got = flat(enc.fitTransform(X, y));
    for (let i = 0; i < expected.length; i++) {
      expect(got[i]).toBeCloseTo(expected[i] as number, 12);
    }
    // the encoder is fitted on the full data afterwards
    expect(flat(enc.transform([[0]]))[0]).toBeCloseTo(13.38888888888889, 12);
  });

  it("does not leak a row's own target through the fallback mean", () => {
    // Each category appears once, so every encoding must come from the fold mean only.
    const enc = new TargetEncoder({ smooth: 1, cv: 2 });
    const got = flat(enc.fitTransform([[0], [1], [2], [3]], [0, 0, 0, 100]));
    // folds: {0,2} and {1,3}. Held-out rows have unseen categories, so the value is the
    // training-fold mean: rows 0,2 -> mean(y[1], y[3]) = 50, rows 1,3 -> mean(y[0], y[2]) = 0.
    expect(got).toEqual([50, 0, 50, 0]);
  });

  it("supports string categories and array targets", () => {
    const enc = new TargetEncoder({ smooth: 0 }).fit([["a"], ["b"], ["a"]], [1, 5, 3]);
    expect(flat(enc.transform([["a"], ["b"], ["zzz"]]))).toEqual([2, 5, 3]);
    expect(enc.encodings?.[0]?.get("a")).toBe(2);
  });

  it("validates options, targets and shapes", () => {
    expect(() => new TargetEncoder({ smooth: -1 })).toThrow(InvalidParameterError);
    expect(() => new TargetEncoder({ smooth: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new TargetEncoder({ cv: 1 })).toThrow(InvalidParameterError);
    expect(() => new TargetEncoder().fit([[1], [2]], [1, Number.NaN])).toThrow(DataValidationError);
    expect(() => new TargetEncoder().fit([[1], [2]], [1])).toThrow(ShapeError);
    expect(() => new TargetEncoder().fit(tensor([[1], [2]]), tensor([[1], [2]]))).toThrow(
      ShapeError
    );
    expect(() => new TargetEncoder().fit(zeros([0, 1]), [])).toThrow(InvalidParameterError);
    expect(() => new TargetEncoder().fitTransform([[1]], [1])).toThrow(InvalidParameterError);
    expect(() => new TargetEncoder().transform([[1]])).toThrow(NotFittedError);
    const fitted = new TargetEncoder().fit(
      [
        [1, 2],
        [2, 3],
      ],
      [1, 2]
    );
    expect(() => fitted.transform([[1]])).toThrow(ShapeError);
  });

  it("keeps the previous fit when a refit fails", () => {
    const enc = new TargetEncoder({ smooth: 0 }).fit([[1], [2]], [10, 20]);
    expect(() => enc.fit([[1], [2]], [1, Number.NaN])).toThrow(DataValidationError);
    expect(flat(enc.transform([[1]]))).toEqual([10]);
  });
});
