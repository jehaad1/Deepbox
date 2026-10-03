import { describe, expect, it } from "vitest";
import { DeepboxError, DTypeError, InvalidParameterError, MemoryError } from "../../src/core";
import { CSRMatrix, type Tensor, tensor, zeros } from "../../src/ndarray";
import {
  type CountVectorizerOptions,
  type HashingVectorizerOptions,
  LabelBinarizer,
  LeaveOneGroupOut,
  LeavePGroupsOut,
  MinMaxScaler,
  MultiLabelBinarizer,
  PolynomialFeatures,
  PredefinedSplit,
  RepeatedKFold,
  ShuffleSplit,
  type SplineExtrapolation,
  type SplineKnots,
  StratifiedGroupKFold,
  StratifiedShuffleSplit,
  type TfidfVectorizerOptions,
} from "../../src/preprocess";
import {
  assertNumericTensor,
  createRandomStream,
  deriveSeed,
} from "../../src/preprocess/_internal";

const pairs = (splits: { trainIndex: number[]; testIndex: number[] }[]): number[][][] =>
  splits.map((s) => [s.trainIndex, s.testIndex]);

describe("wave2 preprocess: sparse inverseTransform with repeated entries", () => {
  // Row 0 stores column 0 twice (0.6 + 0.6 = 1.2) and column 1 once (1.0).
  const nonCanonical = new CSRMatrix({
    data: new Float64Array([0.6, 0.6, 1.0, 1.0]),
    indices: new Int32Array([0, 0, 1, 2]),
    indptr: new Int32Array([0, 3, 4]),
    shape: [2, 3],
  });

  it("builds a non-canonical matrix for the checks below", () => {
    expect(nonCanonical.hasCanonicalFormat).toBe(false);
  });

  it("LabelBinarizer sums repeated entries before picking the active class", () => {
    const lb = new LabelBinarizer();
    lb.fit(tensor([0, 1, 2]));
    const out = lb.inverseTransform(nonCanonical);
    expect(Array.from(out.data as ArrayLike<number>)).toEqual([0, 2]);
  });

  it("MultiLabelBinarizer sums repeated entries before testing activity", () => {
    const mlb = new MultiLabelBinarizer();
    mlb.fit([["a"], ["b"], ["c"]]);
    // Row 0: column 1 stored as +1 and -1 (sum 0, inactive) plus column 2 once.
    const m = new CSRMatrix({
      data: new Float64Array([1, -1, 1]),
      indices: new Int32Array([1, 1, 2]),
      indptr: new Int32Array([0, 3]),
      shape: [1, 3],
    });
    expect(mlb.inverseTransform(m)).toEqual([["c"]]);
  });
});

describe("wave2 preprocess: seeded random streams", () => {
  it("createRandomStream is deterministic and stays in [0, 1)", () => {
    const a = createRandomStream(42);
    const b = createRandomStream(42);
    let sum = 0;
    for (let i = 0; i < 20000; i++) {
      const x = a();
      expect(x).toBe(b());
      expect(x >= 0 && x < 1).toBe(true);
      sum += x;
    }
    expect(Math.abs(sum / 20000 - 0.5)).toBeLessThan(0.01);
  });

  it("rejects negative and non-integer seeds", () => {
    expect(() => createRandomStream(-1)).toThrow(InvalidParameterError);
    expect(() => createRandomStream(1, 0.5)).toThrow(InvalidParameterError);
    expect(() => createRandomStream(1, -1)).toThrow(InvalidParameterError);
    expect(() => createRandomStream(1.5)).toThrow(InvalidParameterError);
    expect(() => deriveSeed(-1, 0)).toThrow(InvalidParameterError);
  });

  it("does not repeat a short cycle", () => {
    const next = createRandomStream(1);
    const seen = new Set<number>();
    for (let i = 0; i < 200000; i++) seen.add(next());
    expect(seen.size).toBeGreaterThan(199000);
  });

  it("adjacent seeds give uncorrelated streams", () => {
    const n = 5000;
    const a = createRandomStream(10);
    const b = createRandomStream(11);
    let sa = 0;
    let sb = 0;
    let saa = 0;
    let sbb = 0;
    let sab = 0;
    for (let i = 0; i < n; i++) {
      const x = a();
      const y = b();
      sa += x;
      sb += y;
      saa += x * x;
      sbb += y * y;
      sab += x * y;
    }
    const cov = sab / n - (sa / n) * (sb / n);
    const r = cov / Math.sqrt((saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2));
    expect(Math.abs(r)).toBeLessThan(0.06);
  });

  it("deriveSeed gives distinct safe integers per (seed, index)", () => {
    const seen = new Set<number>();
    for (let s = 0; s < 40; s++) {
      for (let i = 0; i < 40; i++) {
        const d = deriveSeed(s, i);
        expect(Number.isSafeInteger(d) && d >= 0).toBe(true);
        seen.add(d);
      }
    }
    expect(seen.size).toBe(1600);
  });

  it("ShuffleSplit iteration i+1 of seed s differs from iteration i of seed s+1", () => {
    const X = zeros([30, 1]);
    const a = new ShuffleSplit({ nSplits: 3, testSize: 0.3, randomState: 5 }).split(X);
    const b = new ShuffleSplit({ nSplits: 3, testSize: 0.3, randomState: 6 }).split(X);
    expect(a[1]?.testIndex).not.toEqual(b[0]?.testIndex);
    expect(a[2]?.testIndex).not.toEqual(b[1]?.testIndex);
    // Same seed still reproduces the same splits.
    const again = new ShuffleSplit({ nSplits: 3, testSize: 0.3, randomState: 5 }).split(X);
    expect(pairs(again)).toEqual(pairs(a));
  });

  it("StratifiedShuffleSplit and RepeatedKFold decorrelate neighbouring seeds", () => {
    const X = zeros([40, 1]);
    const y = tensor(Array.from({ length: 40 }, (_, i) => i % 2));
    const a = new StratifiedShuffleSplit({ nSplits: 3, testSize: 0.25, randomState: 3 }).split(
      X,
      y
    );
    const b = new StratifiedShuffleSplit({ nSplits: 3, testSize: 0.25, randomState: 4 }).split(
      X,
      y
    );
    expect(a[1]?.testIndex).not.toEqual(b[0]?.testIndex);

    const r1 = new RepeatedKFold({ nSplits: 2, nRepeats: 3, randomState: 1 }).split(X);
    const r2 = new RepeatedKFold({ nSplits: 2, nRepeats: 3, randomState: 2 }).split(X);
    // Repetition r has two folds, so repetition 1 of seed 1 sits at indices 2 and 3.
    expect(r1[2]?.testIndex).not.toEqual(r2[0]?.testIndex);
    expect(r1[3]?.testIndex).not.toEqual(r2[1]?.testIndex);
  });
});

describe("wave2 preprocess: complex dtype guard", () => {
  it("assertNumericTensor rejects complex and string dtypes", () => {
    const fake = (dtype: string): Tensor => ({ dtype }) as unknown as Tensor;
    expect(() => assertNumericTensor(fake("complex128"), "X")).toThrow(DTypeError);
    expect(() => assertNumericTensor(fake("complex64"), "X")).toThrow(DTypeError);
    expect(() => assertNumericTensor(fake("string"), "X")).toThrow(DTypeError);
    expect(() => assertNumericTensor(fake("float64"), "X")).not.toThrow();
  });
});

describe("wave2 preprocess: PolynomialFeatures column definitions", () => {
  it("matches scikit-learn powers and values", () => {
    const poly = new PolynomialFeatures({ degree: 3 }).fit(tensor([[1, 1]]));
    expect(poly.powers).toEqual([
      [0, 0],
      [1, 0],
      [0, 1],
      [2, 0],
      [1, 1],
      [0, 2],
      [3, 0],
      [2, 1],
      [1, 2],
      [0, 3],
    ]);
    const out = poly.transform(
      tensor([
        [2, 3],
        [-1, 0.5],
      ])
    );
    expect(out.toArray()).toEqual([
      [1, 2, 3, 4, 6, 9, 8, 12, 18, 27],
      [1, -1, 0.5, 1, -0.5, 0.25, -1, 0.5, -0.25, 0.125],
    ]);
    expect(poly.nOutputFeatures).toBe(10);
    expect(poly.getFeatureNamesOut().slice(0, 6)).toEqual([
      "1",
      "x0",
      "x1",
      "x0^2",
      "x0 x1",
      "x1^2",
    ]);
  });

  it("matches scikit-learn for interactionOnly without a bias column", () => {
    const poly = new PolynomialFeatures({ degree: 2, interactionOnly: true }).fit(
      tensor([[1, 1, 1]])
    );
    expect(poly.powers).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 0],
      [1, 0, 1],
      [0, 1, 1],
    ]);
    const nb = new PolynomialFeatures({ degree: 2, includeBias: false }).fit(tensor([[1, 1]]));
    expect(nb.nOutputFeatures).toBe(5);
  });

  it("refuses requests whose column definitions would be enormous", () => {
    // One feature gives a single column per degree, so the column count alone stays small
    // while the stored index count grows with the square of the degree.
    expect(() => new PolynomialFeatures({ degree: 100000 }).fit(tensor([[1]]))).toThrow(
      MemoryError
    );
    expect(() => new PolynomialFeatures({ degree: 2 }).fit(zeros([1, 100000]))).toThrow(
      MemoryError
    );
  });
});

describe("wave2 preprocess: group and predefined splitters", () => {
  const X = zeros([10, 1]);
  const y = tensor([0, 1, 0, 0, 1, 1, 0, 1, 0, 1]);
  const groups = [0, 0, 1, 1, 1, 2, 3, 3, 4, 4];

  it("LeaveOneGroupOut matches scikit-learn", () => {
    const logo = new LeaveOneGroupOut();
    expect(pairs(logo.split(X, undefined, groups))).toEqual([
      [
        [2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1],
      ],
      [
        [0, 1, 5, 6, 7, 8, 9],
        [2, 3, 4],
      ],
      [[0, 1, 2, 3, 4, 6, 7, 8, 9], [5]],
      [
        [0, 1, 2, 3, 4, 5, 8, 9],
        [6, 7],
      ],
      [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9],
      ],
    ]);
    expect(logo.getNSplits(groups)).toBe(5);
    const strGroups = ["b", "b", "a", "a", "c", "c", "d", "d", "e", "e"];
    expect(logo.split(X, y, strGroups).map((s) => s.testIndex)).toEqual([
      [2, 3],
      [0, 1],
      [4, 5],
      [6, 7],
      [8, 9],
    ]);
    expect(() => logo.split(X, y, new Array<number>(10).fill(1))).toThrow(InvalidParameterError);
    expect(() => logo.split(X, y, [0, 1])).toThrow(InvalidParameterError);
  });

  it("LeavePGroupsOut matches scikit-learn", () => {
    const lpgo = new LeavePGroupsOut(2);
    const splits = lpgo.split(X, undefined, tensor(groups));
    expect(splits.length).toBe(10);
    expect(pairs(splits.slice(0, 4))).toEqual([
      [
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
      ],
      [
        [2, 3, 4, 6, 7, 8, 9],
        [0, 1, 5],
      ],
      [
        [2, 3, 4, 5, 8, 9],
        [0, 1, 6, 7],
      ],
      [
        [2, 3, 4, 5, 6, 7],
        [0, 1, 8, 9],
      ],
    ]);
    expect(lpgo.getNSplits(groups)).toBe(10);
    expect(() => new LeavePGroupsOut(5).split(X, undefined, groups)).toThrow(InvalidParameterError);
    expect(() => new LeavePGroupsOut(0)).toThrow(InvalidParameterError);
  });

  it("PredefinedSplit matches scikit-learn and skips -1", () => {
    const ps = new PredefinedSplit([0, 0, 1, 1, -1, 2, 2, -1, 1, 0]);
    expect(ps.getNSplits()).toBe(3);
    expect(pairs(ps.split())).toEqual([
      [
        [2, 3, 4, 5, 6, 7, 8],
        [0, 1, 9],
      ],
      [
        [0, 1, 4, 5, 6, 7, 9],
        [2, 3, 8],
      ],
      [
        [0, 1, 2, 3, 4, 7, 8, 9],
        [5, 6],
      ],
    ]);
    expect(() => ps.split(zeros([3, 1]))).toThrow(InvalidParameterError);
    expect(() => new PredefinedSplit([-1, -1])).toThrow(InvalidParameterError);
    expect(() => new PredefinedSplit([0, 1.5])).toThrow(InvalidParameterError);
    expect(new PredefinedSplit(tensor([0, 1, 1])).getNSplits()).toBe(2);
  });

  it("StratifiedGroupKFold matches scikit-learn without shuffling", () => {
    const sgk = new StratifiedGroupKFold({ nSplits: 3 });
    expect(pairs(sgk.split(X, y, groups))).toEqual([
      [
        [0, 1, 5, 6, 7, 8, 9],
        [2, 3, 4],
      ],
      [
        [2, 3, 4, 5, 6, 7],
        [0, 1, 8, 9],
      ],
      [
        [0, 1, 2, 3, 4, 8, 9],
        [5, 6, 7],
      ],
    ]);
    expect(sgk.getNSplits()).toBe(3);
  });

  it("StratifiedGroupKFold keeps groups whole and is reproducible with shuffle", () => {
    const opts = { nSplits: 3, shuffle: true, randomState: 7 };
    const a = new StratifiedGroupKFold(opts).split(X, y, groups);
    const b = new StratifiedGroupKFold(opts).split(X, y, groups);
    expect(pairs(a)).toEqual(pairs(b));
    const seen = new Set<number>();
    for (const s of a) {
      for (const i of s.testIndex) {
        expect(seen.has(i)).toBe(false);
        seen.add(i);
      }
      const testGroups = new Set(s.testIndex.map((i) => groups[i]));
      for (const i of s.trainIndex) expect(testGroups.has(groups[i])).toBe(false);
    }
    expect(seen.size).toBe(10);
  });

  it("StratifiedGroupKFold validates its input", () => {
    expect(() => new StratifiedGroupKFold({ nSplits: 1 })).toThrow(InvalidParameterError);
    expect(() => new StratifiedGroupKFold({ nSplits: 6 }).split(X, y, groups)).toThrow(
      InvalidParameterError
    );
    expect(() => new StratifiedGroupKFold({ nSplits: 2 }).split(X, tensor([0, 1]), groups)).toThrow(
      InvalidParameterError
    );
    expect(() => new StratifiedGroupKFold({ nSplits: 2 }).split(X, y, [0, 1])).toThrow(
      InvalidParameterError
    );
  });
});

describe("wave2 preprocess: public type exports", () => {
  it("re-exports the spline and text option types", () => {
    const knots: SplineKnots = "uniform";
    const extrapolation: SplineExtrapolation = "constant";
    const count: CountVectorizerOptions = { lowercase: true };
    const tfidf: TfidfVectorizerOptions = { norm: "l2" };
    const hashing: HashingVectorizerOptions = { nFeatures: 8 };
    expect([knots, extrapolation, count.lowercase, tfidf.norm, hashing.nFeatures]).toEqual([
      "uniform",
      "constant",
      true,
      "l2",
      8,
    ]);
    expect(new DeepboxError("x")).toBeInstanceOf(Error);
  });
});

describe("wave2 preprocess: MinMaxScaler when max - min overflows", () => {
  it("scales a range wider than the float64 maximum", () => {
    const X = tensor([[-1e308], [0], [1e308]], { dtype: "float64" });
    const scaler = new MinMaxScaler().fit(X);
    const out = scaler.transform(X).toArray();
    expect(out).toEqual([[0], [0.5], [1]]);
    const back = scaler.inverseTransform(tensor([[0], [0.5], [1]], { dtype: "float64" })).toArray();
    expect(back).toEqual([[-1e308], [0], [1e308]]);
  });
});
