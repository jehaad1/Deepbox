import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  type Classifier,
  cross_val_score,
  DecisionTreeRegressor,
  GaussianProcessClassifier,
  GaussianProcessRegressor,
  LinearRegression,
  permutationImportance,
  type Regressor,
  VotingClassifier,
  VotingRegressor,
} from "../../src/ml";
import { type Tensor, tensor, transpose } from "../../src/ndarray";
import { clearSeed, setSeed } from "../../src/random";

const f64 = (rows: number | number[] | number[][]): Tensor => tensor(rows, { dtype: "float64" });
const arr = (t: Tensor): number[] =>
  Array.from(t.toArray() as ArrayLike<number>).flat() as number[];
const expectClose = (actual: number[], expected: number[], digits: number): void => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, digits);
  }
};
const rows = (t: Tensor): number[][] => t.toArray() as number[][];

/** Classifier stub that always predicts one label with a fixed probability row. */
class ConstClassifier implements Classifier {
  private labels: number[] = [];
  constructor(
    private readonly label: number,
    private readonly proba: number[],
    private readonly exposeClasses = true
  ) {}
  fit(_X: Tensor, y: Tensor): this {
    this.labels = [...new Set(arr(y))].sort((a, b) => a - b);
    return this;
  }
  predict(X: Tensor): Tensor {
    return tensor(new Array<number>(X.shape[0] ?? 0).fill(this.label), { dtype: "float64" });
  }
  predictProba(X: Tensor): Tensor {
    const n = X.shape[0] ?? 0;
    const flat = new Float64Array(n * this.proba.length);
    for (let i = 0; i < n; i++) flat.set(this.proba, i * this.proba.length);
    return tensor(flat).reshape([n, this.proba.length]);
  }
  score(): number {
    return 0;
  }
  get classes(): Tensor | undefined {
    return this.exposeClasses ? tensor(this.labels, { dtype: "int32" }) : undefined;
  }
  getParams(): Record<string, unknown> {
    return { label: this.label, proba: this.proba, exposeClasses: this.exposeClasses };
  }
  setParams(): this {
    return this;
  }
}

/** Regressor stub that predicts a constant. */
class ConstRegressor implements Regressor {
  constructor(private readonly value: number) {}
  fit(): this {
    return this;
  }
  predict(X: Tensor): Tensor {
    return tensor(new Array<number>(X.shape[0] ?? 0).fill(this.value), { dtype: "float64" });
  }
  score(): number {
    return 0;
  }
  getParams(): Record<string, unknown> {
    return { value: this.value };
  }
  setParams(): this {
    return this;
  }
}

const X4 = f64([[0], [1], [2], [3]]);
const y3 = tensor([0, 1, 2, 0], { dtype: "int32" });

describe("VotingClassifier", () => {
  it("hard voting breaks ties toward the smallest label (scikit-learn: [1 1 1 1] and [0 0 0 0])", () => {
    const tie = new VotingClassifier({
      estimators: [new ConstClassifier(2, [0, 0, 1]), new ConstClassifier(1, [0, 1, 0])],
    }).fit(X4, y3);
    expect(arr(tie.predict(X4))).toEqual([1, 1, 1, 1]);

    const three = new VotingClassifier({
      estimators: [
        new ConstClassifier(2, [0, 0, 1]),
        new ConstClassifier(1, [0, 1, 0]),
        new ConstClassifier(0, [1, 0, 0]),
      ],
    }).fit(X4, y3);
    expect(arr(three.predict(X4))).toEqual([0, 0, 0, 0]);
  });

  it("weights decide a hard vote", () => {
    const clf = new VotingClassifier({
      estimators: [new ConstClassifier(2, [0, 0, 1]), new ConstClassifier(1, [0, 1, 0])],
      weights: [3, 1],
    }).fit(X4, y3);
    expect(arr(clf.predict(X4))).toEqual([2, 2, 2, 2]);
  });

  it("soft voting matches scikit-learn probabilities for weights [1, 3]", () => {
    const clf = new VotingClassifier({
      estimators: [
        new ConstClassifier(2, [0.1, 0.2, 0.7]),
        new ConstClassifier(1, [0.3, 0.5, 0.2]),
      ],
      voting: "soft",
      weights: [1, 3],
    }).fit(X4, y3);
    const p = clf.predictProba(X4);
    expect(p.dtype).toBe("float64");
    const first = rows(p)[0] as number[];
    expect(first[0]).toBeCloseTo(0.25, 14);
    expect(first[1]).toBeCloseTo(0.425, 14);
    expect(first[2]).toBeCloseTo(0.325, 14);
    expect(arr(clf.predict(X4))).toEqual([1, 1, 1, 1]);
  });

  it("predictProba returns an (0, n_classes) tensor for zero rows", () => {
    const clf = new VotingClassifier({
      estimators: [new ConstClassifier(0, [0.5, 0.5, 0])],
    }).fit(X4, y3);
    expect(
      clf.predictProba(tensor([] as number[][], { dtype: "float64" }).reshape([0, 1])).shape
    ).toEqual([0, 3]);
  });

  it("rejects negative, non-finite, all-zero and mis-sized weights", () => {
    const est = () => [new ConstClassifier(0, [1, 0]), new ConstClassifier(1, [0, 1])];
    expect(() => new VotingClassifier({ estimators: est(), weights: [1, -1] })).toThrow(
      InvalidParameterError
    );
    expect(() => new VotingClassifier({ estimators: est(), weights: [1, Number.NaN] })).toThrow(
      /finite/
    );
    expect(() => new VotingClassifier({ estimators: est(), weights: [0, 0] })).toThrow(/positive/);
    expect(
      () => new VotingRegressor({ estimators: [new ConstRegressor(1)], weights: [0] })
    ).toThrow(InvalidParameterError);
    const clf = new VotingClassifier({ estimators: est() });
    expect(() => clf.setParams({ weights: [1] })).toThrow(/length/);
    expect(() => clf.setParams({ weights: [1, 2, 3] })).toThrow(InvalidParameterError);
    const reg = new VotingRegressor({ estimators: [new ConstRegressor(1)] });
    expect(() => reg.setParams({ weights: [1, 1] })).toThrow(/length/);
  });

  it("rejects an unknown voting mode in the constructor", () => {
    expect(
      () =>
        new VotingClassifier({
          estimators: [new ConstClassifier(0, [1, 0])],
          voting: "median" as unknown as "hard",
        })
    ).toThrow(InvalidParameterError);
  });

  it("does not alias the weights array and applies setParams atomically", () => {
    const weights = [1, 2];
    const clf = new VotingClassifier({
      estimators: [new ConstClassifier(0, [1, 0]), new ConstClassifier(1, [0, 1])],
      weights,
    });
    weights[0] = 100;
    expect(clf.getParams().weights).toEqual([1, 2]);
    (clf.getParams().weights as number[])[1] = 50;
    expect(clf.getParams().weights).toEqual([1, 2]);

    expect(() => clf.setParams({ voting: "soft", weights: [1, -4] })).toThrow(
      InvalidParameterError
    );
    expect(clf.getParams().voting).toBe("hard");
    expect(clf.getParams().weights).toEqual([1, 2]);
  });

  it("does not truncate fractional class labels", () => {
    const y = tensor([0.5, 1.5, 0.5, 1.5], { dtype: "float64" });
    const clf = new VotingClassifier({ estimators: [new ConstClassifier(1.5, [0.2, 0.8])] }).fit(
      X4,
      y
    );
    const pred = clf.predict(X4);
    expect(pred.dtype).toBe("float64");
    expect(arr(pred)).toEqual([1.5, 1.5, 1.5, 1.5]);
    expect(arr(clf.classes as Tensor)).toEqual([0.5, 1.5]);
  });

  it("score validates the sample count of X against y", () => {
    const clf = new VotingClassifier({ estimators: [new ConstClassifier(0, [1, 0, 0])] }).fit(
      X4,
      y3
    );
    expect(() => clf.score(X4, tensor([0, 1], { dtype: "int32" }))).toThrow(ShapeError);
    expect(() => clf.score(X4, tensor([], { dtype: "int32" }))).toThrow(DataValidationError);
    expect(clf.score(X4, y3)).toBe(0.5);
  });

  it("score accepts int64 targets", () => {
    const clf = new VotingClassifier({ estimators: [new ConstClassifier(0, [1, 0, 0])] }).fit(
      X4,
      y3
    );
    const y64 = tensor([0, 1, 2, 0], { dtype: "int64" });
    expect(clf.score(X4, y64)).toBe(0.5);
  });

  it("a failed refit leaves the model unfitted", () => {
    class Failing extends ConstClassifier {
      override fit(_X: Tensor, _y: Tensor): this {
        throw new Error("boom");
      }
    }
    const good = new ConstClassifier(0, [1, 0, 0]);
    const clf = new VotingClassifier({ estimators: [good, new Failing(0, [1, 0, 0])] });
    expect(() => clf.fit(X4, y3)).toThrow("boom");
    expect(() => clf.predict(X4)).toThrow(NotFittedError);
  });

  it("soft voting reports a mismatch between probability columns and classes", () => {
    const clf = new VotingClassifier({
      estimators: [new ConstClassifier(0, [0.5, 0.5], false)],
      voting: "soft",
    }).fit(X4, y3);
    expect(() => clf.predict(X4)).toThrow(ShapeError);
  });

  it("predictProba reports an estimator without predictProba", () => {
    const noProba = new ConstClassifier(0, [1, 0, 0]) as unknown as { predictProba?: unknown };
    noProba.predictProba = undefined;
    const clf = new VotingClassifier({ estimators: [noProba as unknown as Classifier] }).fit(
      X4,
      y3
    );
    expect(() => clf.predictProba(X4)).toThrow(InvalidParameterError);
    expect(() =>
      new VotingClassifier({ estimators: [noProba as unknown as Classifier], voting: "soft" }).fit(
        X4,
        y3
      )
    ).toThrow(/predictProba/);
  });

  it("clone returns an unfitted copy with independent estimators", () => {
    const clf = new VotingClassifier({
      estimators: [new ConstClassifier(0, [1, 0, 0])],
      weights: [2],
    }).fit(X4, y3);
    const copy = clf.clone();
    expect(copy.classes).toBeUndefined();
    expect(copy.getParams().weights).toEqual([2]);
    expect(copy.getParams().estimators).not.toEqual(clf.getParams().estimators);
    expect(() => copy.predict(X4)).toThrow(NotFittedError);
  });
});

describe("VotingRegressor", () => {
  it("returns float64 weighted means", () => {
    const reg = new VotingRegressor({
      estimators: [new ConstRegressor(1), new ConstRegressor(2)],
      weights: [1, 2],
    }).fit(X4, f64([1, 2, 3, 4]));
    const pred = reg.predict(X4);
    expect(pred.dtype).toBe("float64");
    expect(arr(pred)[0]).toBe(5 / 3);
  });

  it("score returns 1 for a perfect fit of a constant target and 0 otherwise", () => {
    const reg = new VotingRegressor({ estimators: [new ConstRegressor(2)] }).fit(
      X4,
      f64([2, 2, 2, 2])
    );
    expect(reg.score(X4, f64([2, 2, 2, 2]))).toBe(1);
    expect(reg.score(X4, f64([3, 3, 3, 3]))).toBe(0);
    expect(() => reg.score(X4, f64([2, 2]))).toThrow(ShapeError);
  });

  it("works with cross_val_score through clone()", () => {
    const X = f64([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const y = f64([2, 4, 6, 8, 10, 12, 14, 16]);
    const reg = new VotingRegressor({
      estimators: [new LinearRegression(), new DecisionTreeRegressor({ maxDepth: 3 })],
    });
    const scores = cross_val_score(reg, X, y, 2);
    expect(scores.length).toBe(2);
    expect(scores.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("GaussianProcessRegressor", () => {
  const X = f64([[1], [2], [3.5], [4], [6]]);
  const y = f64([1, 4, 9, 16, 25]);
  const Xt = f64([[0], [2.5], [5], [10]]);
  const opts = { alpha: 1e-6, lengthScale: 1.5, kernelVariance: 2 };

  it("matches scikit-learn (RBF, optimizer=None): mean, std and log marginal likelihood", () => {
    const gpr = new GaussianProcessRegressor(opts).fit(X, y);
    const { mean, std } = gpr.predictWithStd(Xt);
    const refMean = [
      -5.1439758029075655, 3.738627798388734, 28.118842008108615, 0.20441459352055996,
    ];
    const refStd = [
      0.5334169436222664, 0.05357737537963583, 0.17379997007234121, 1.413218497745831,
    ];
    expectClose(arr(mean), refMean, 10);
    expectClose(arr(std), refStd, 10);
    expect(mean.dtype).toBe("float64");
    expect(gpr.logMarginalLikelihood()).toBeCloseTo(-294.43763236376697, 8);
    expect(arr(gpr.predict(Xt))).toEqual(arr(mean));
  });

  it("normalizeY matches scikit-learn normalize_y=True", () => {
    const gpr = new GaussianProcessRegressor({ ...opts, normalizeY: true }).fit(X, y);
    const { mean, std } = gpr.predictWithStd(Xt);
    const refMean = [-3.084888620364824, 3.8023053376777307, 27.93410240298545, 10.89420734387446];
    const refStd = [4.613362759809849, 0.4633746102369729, 1.5031436837064425, 12.222501866367196];
    expectClose(arr(mean), refMean, 10);
    expectClose(arr(std), refStd, 10);
    expect(gpr.logMarginalLikelihood()).toBeCloseTo(-6.817347152850687, 8);
  });

  it("throws a typed error instead of returning garbage for a singular kernel matrix", () => {
    const gpr = new GaussianProcessRegressor({ alpha: 0 });
    expect(() => gpr.fit(f64([[1], [1]]), f64([1, 2]))).toThrow(DataValidationError);
  });

  it("keeps the previous model when a refit fails", () => {
    const gpr = new GaussianProcessRegressor(opts).fit(X, y);
    const before = arr(gpr.predict(Xt));
    gpr.setParams({ alpha: 0 });
    expect(() => gpr.fit(f64([[1], [1]]), f64([1, 2]))).toThrow(DataValidationError);
    expect(arr(gpr.predict(Xt))).toEqual(before);
  });

  it("setParams validates and takes effect on the next fit only", () => {
    const gpr = new GaussianProcessRegressor(opts).fit(X, y);
    const before = arr(gpr.predict(Xt));
    gpr.setParams({ lengthScale: 0.2 });
    expect(gpr.getParams().lengthScale).toBe(0.2);
    expect(arr(gpr.predict(Xt))).toEqual(before);
    gpr.fit(X, y);
    expect(arr(gpr.predict(Xt))).not.toEqual(before);

    expect(() => gpr.setParams({ lengthScale: -1 })).toThrow(InvalidParameterError);
    expect(() => gpr.setParams({ alpha: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => gpr.setParams({ bogus: 1 })).toThrow(/Unknown parameter/);
    expect(gpr.getParams().lengthScale).toBe(0.2);
  });

  it("rejects NaN and infinite hyperparameters", () => {
    expect(() => new GaussianProcessRegressor({ alpha: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new GaussianProcessRegressor({ lengthScale: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(
      () => new GaussianProcessRegressor({ kernelVariance: Number.POSITIVE_INFINITY })
    ).toThrow(InvalidParameterError);
    expect(() => new GaussianProcessRegressor({ lengthScale: 1e-200 })).toThrow(
      InvalidParameterError
    );
  });

  it("does not alias the training data", () => {
    const Xm = f64([[1], [2], [3.5], [4], [6]]);
    const gpr = new GaussianProcessRegressor(opts).fit(Xm, y);
    const before = arr(gpr.predict(Xt));
    (Xm.data as Float64Array)[0] = 100;
    expect(arr(gpr.predict(Xt))).toEqual(before);
  });

  it("score returns 1 for a perfectly fitted constant target and validates y", () => {
    const Xc = f64([[0], [1], [2]]);
    const yc = f64([3, 3, 3]);
    const gpr = new GaussianProcessRegressor({ alpha: 1e-12, normalizeY: true }).fit(Xc, yc);
    expect(gpr.score(Xc, yc)).toBe(1);
    expect(() => gpr.score(Xc, f64([3, 3]))).toThrow(ShapeError);
    expect(() => gpr.score(Xc, f64([[3, 3, 3]]))).toThrow(ShapeError);
  });

  it("accepts integer and transposed-copy inputs through the shared validation", () => {
    const Xi = tensor([[1], [2], [3]], { dtype: "int32" });
    const gpr = new GaussianProcessRegressor({ alpha: 1e-8 }).fit(Xi, f64([1, 2, 3]));
    expect(arr(gpr.predict(Xi)).length).toBe(3);
    // the transpose of a row is a dense (3, 1) column, so it is read in place
    const Xw = transpose(f64([[1, 2, 3]]));
    expect(arr(gpr.predict(Xw))).toEqual(arr(gpr.predict(f64([[1], [2], [3]]))));
    // a genuinely strided view is still rejected
    const strided = transpose(
      f64([
        [1, 2, 3],
        [4, 5, 6],
      ])
    );
    expect(() => gpr.predict(strided)).toThrow();
  });

  it("clone is unfitted with the same hyperparameters", () => {
    const gpr = new GaussianProcessRegressor({ ...opts, normalizeY: true }).fit(X, y);
    const copy = gpr.clone();
    expect(copy.getParams()).toEqual(gpr.getParams());
    expect(() => copy.predict(Xt)).toThrow(NotFittedError);
  });
});

describe("GaussianProcessClassifier", () => {
  const X2 = f64([
    [0.264052345967664, -1.0998427916327767],
    [-0.5212620158942608, 0.7408931992014578],
    [0.3675579901499675, -2.477277879876411],
    [-0.5499115824744106, -1.651357208297698],
    [-1.603218851793558, -1.0894014980616276],
    [-1.355956428839122, -0.04572649303702492],
    [-0.7389622748530066, -1.3783249835071716],
    [-1.0561367672545743, -1.1663256726257332],
    [-0.00592092684239387, -1.705158263765801],
    [-1.1869322983490986, -2.3540957393017248],
    [-1.0529898158340787, 2.153618595440361],
    [2.3644361988595053, 0.7578349795935583],
    [3.7697546239876076, 0.04563432540123524],
    [1.545758517301446, 1.3128161499741664],
    [3.0327792143584578, 2.969358769900285],
    [1.6549474256969163, 1.8781625196021736],
    [0.6122142523698872, -0.48079646822392696],
    [1.1520878506738474, 1.65634896910398],
    [2.730290680727721, 2.7023798487844113],
    [1.1126731825920477, 1.1976972494246645],
  ]);
  const y2 = tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], {
    dtype: "int32",
  });
  const Xt2 = f64([
    [0, 0],
    [1, 1],
    [-2, -2],
    [3, -3],
    [10, 10],
  ]);
  const ref2 = [
    [0.5875442813757559, 0.4124557186242441],
    [0.26795078565612584, 0.7320492143438742],
    [0.7888943979321539, 0.21110560206784612],
    [0.5126549381183416, 0.4873450618816584],
    [0.49999963916895895, 0.500000360831041],
  ];

  it("binary predictProba matches scikit-learn's Laplace GPC (RBF, optimizer=None)", () => {
    const gpc = new GaussianProcessClassifier({
      alpha: 0,
      lengthScale: 2,
      kernelVariance: 1.5,
    }).fit(X2, y2);
    const p = gpc.predictProba(Xt2);
    expect(p.dtype).toBe("float64");
    expect(p.shape).toEqual([5, 2]);
    expectClose(rows(p).flat(), ref2.flat(), 10);
    expect(arr(gpc.predict(Xt2))).toEqual([0, 1, 0, 0, 1]);
  });

  it("multi-class (one-vs-rest) predictProba matches scikit-learn", () => {
    const X3 = f64([
      [-1.0485529650670926, -1.4200179371789752],
      [-1.7062701906250126, 1.9507753952317897],
      [-0.5096521817516535, -0.4380743016111864],
      [-1.2527953600499264, 0.7774903558319103],
      [-1.6138978475579515, -0.2127402802139687],
      [-0.8954665611936756, 0.386902497859262],
      [3.489194862431127, -1.180632184122412],
      [3.971817771661345, 0.42833187053041766],
      [4.066517222383168, 0.3024718977397814],
      [3.3656779063190365, -0.3627411659871381],
      [3.327539552224049, -0.3595531615405413],
      [3.186853717955546, -1.7262826023316769],
      [2.177426142253753, 3.598219063791738],
      [0.3698016530339554, 4.462782255525775],
      [1.0927016356167578, 4.051945395796139],
      [2.7290905621775368, 4.1289829107574105],
      [3.1394006845433005, 2.7651741796463476],
      [2.402341641177549, 3.315189909059687],
    ]);
    const y3m = tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], { dtype: "int32" });
    const Xt3 = f64([
      [0, 0],
      [4, 0],
      [2, 4],
      [2, 1.5],
      [-3, -3],
    ]);
    const ref3 = [
      [0.5208823617038846, 0.24671430343119788, 0.23240333486491757],
      [0.20261541996308155, 0.5851361585124701, 0.21224842152444828],
      [0.20621807494359573, 0.20423609291147718, 0.5895458321449271],
      [0.2724744917146683, 0.341907315388614, 0.3856181928967177],
      [0.35752887610369677, 0.32115001146676325, 0.32132111242954],
    ];
    const gpc = new GaussianProcessClassifier({ alpha: 0, lengthScale: 1.5 }).fit(X3, y3m);
    const p = gpc.predictProba(Xt3);
    expectClose(rows(p).flat(), ref3.flat(), 10);
    for (const r of rows(p)) expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(arr(gpc.predict(Xt3))).toEqual([0, 1, 2, 2, 0]);
  });

  it("probabilities far from the data move toward 0.5 instead of saturating", () => {
    const gpc = new GaussianProcessClassifier({
      alpha: 0,
      lengthScale: 2,
      kernelVariance: 1.5,
    }).fit(X2, y2);
    const far = rows(gpc.predictProba(f64([[1000, 1000]])))[0] as number[];
    expect(far[1]).toBeGreaterThan(0.499);
    expect(far[1]).toBeLessThan(0.501);
  });

  it("requires at least two classes", () => {
    const gpc = new GaussianProcessClassifier();
    expect(() => gpc.fit(f64([[1], [2]]), tensor([1, 1], { dtype: "int32" }))).toThrow(
      DataValidationError
    );
  });

  it("keeps fractional class labels", () => {
    const gpc = new GaussianProcessClassifier().fit(
      f64([[0], [0.1], [5], [5.1]]),
      f64([0.5, 0.5, 1.5, 1.5])
    );
    expect(arr(gpc.classes as Tensor)).toEqual([0.5, 1.5]);
    expect(arr(gpc.predict(f64([[0.05], [5.05]])))).toEqual([0.5, 1.5]);
  });

  it("validates parameters and implements setParams / clone", () => {
    expect(() => new GaussianProcessClassifier({ alpha: -1 })).toThrow(InvalidParameterError);
    expect(() => new GaussianProcessClassifier({ alpha: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new GaussianProcessClassifier({ maxLaplaceIter: 0 })).toThrow(/maxLaplaceIter/);
    expect(() => new GaussianProcessClassifier({ maxLaplaceIter: 2.5 })).toThrow(
      InvalidParameterError
    );
    expect(new GaussianProcessClassifier().getParams().maxLaplaceIter).toBe(100);

    const gpc = new GaussianProcessClassifier();
    gpc.setParams({ lengthScale: 0.5, maxLaplaceIter: 30 });
    expect(gpc.getParams().lengthScale).toBe(0.5);
    expect(gpc.getParams().maxLaplaceIter).toBe(30);
    expect(() => gpc.setParams({ lengthScale: 0 })).toThrow(InvalidParameterError);
    expect(() => gpc.setParams({ nope: 1 })).toThrow(/Unknown parameter/);
    expect(gpc.clone().getParams()).toEqual(gpc.getParams());
  });

  it("score checks sample counts and does not alias the training data", () => {
    const Xm = f64([[0], [0.1], [5], [5.1]]);
    const ym = tensor([0, 0, 1, 1], { dtype: "int32" });
    const gpc = new GaussianProcessClassifier().fit(Xm, ym);
    const before = rows(gpc.predictProba(f64([[2.5]])));
    (Xm.data as Float64Array)[0] = 50;
    expect(rows(gpc.predictProba(f64([[2.5]])))).toEqual(before);
    expect(gpc.score(f64([[0], [5]]), tensor([0, 1], { dtype: "int32" }))).toBe(1);
    expect(() => gpc.score(f64([[0], [5]]), tensor([0], { dtype: "int32" }))).toThrow(ShapeError);
  });
});

describe("permutationImportance", () => {
  const X = f64([
    [1, 10, 100],
    [2, 20, 200],
    [3, 30, 300],
    [4, 40, 400],
    [5, 50, 500],
    [6, 60, 600],
  ]);
  const y = f64([1, 2, 3, 4, 5, 6]);

  /** Scores 1 when column 0 equals y, so shuffling column 0 lowers it and others do not. */
  const model = {
    score(Xs: Tensor, ys: Tensor): number {
      const xs = rows(Xs);
      const yv = arr(ys);
      return xs.filter((r, i) => r[0] === yv[i]).length / xs.length;
    },
  };

  it("never modifies X and returns float64 results with the documented shapes", () => {
    const copy = arr(X);
    const res = permutationImportance(model, X, y, { nRepeats: 4, randomState: 7 });
    expect(arr(X)).toEqual(copy);
    expect(res.importances.shape).toEqual([4, 3]);
    expect(res.importancesMean.dtype).toBe("float64");
    expect(res.importancesStd.shape).toEqual([3]);
    const mean = arr(res.importancesMean);
    expect(mean[0]).toBeGreaterThan(0);
    expect(mean[1]).toBe(0);
    expect(mean[2]).toBe(0);
  });

  it("each evaluation shuffles exactly one column by a true permutation", () => {
    const seen: number[][][] = [];
    const scoring = (_est: unknown, Xs: Tensor): number => {
      seen.push(rows(Xs));
      return 0;
    };
    permutationImportance(model, X, y, { nRepeats: 3, randomState: 1, scoring });
    expect(seen.length).toBe(1 + 3 * 3);
    expect(seen[0]).toEqual(rows(X));
    for (let k = 1; k < seen.length; k++) {
      const feature = (k - 1) % 3;
      const cur = seen[k] as number[][];
      for (let col = 0; col < 3; col++) {
        const orig = rows(X).map((r) => r[col] as number);
        const got = cur.map((r) => r[col] as number);
        if (col === feature) expect([...got].sort((a, b) => a - b)).toEqual(orig);
        else expect(got).toEqual(orig);
      }
    }
  });

  it("is reproducible for equal seeds and differs across seeds", () => {
    const a = arr(permutationImportance(model, X, y, { nRepeats: 6, randomState: 3 }).importances);
    const b = arr(permutationImportance(model, X, y, { nRepeats: 6, randomState: 3 }).importances);
    const c = arr(permutationImportance(model, X, y, { nRepeats: 6, randomState: 4 }).importances);
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  it("follows the global seed when randomState is omitted", () => {
    setSeed(11);
    const a = arr(permutationImportance(model, X, y, { nRepeats: 6 }).importances);
    setSeed(11);
    const b = arr(permutationImportance(model, X, y, { nRepeats: 6 }).importances);
    clearSeed();
    expect(a).toEqual(b);
  });

  it("uses the population standard deviation like numpy.std", () => {
    let call = 0;
    const scores = [10, 7, 8, 9, 3];
    const scoring = (): number => scores[call++] as number;
    const res = permutationImportance(model, f64([[1], [2]]), f64([1, 2]), {
      nRepeats: 4,
      scoring,
    });
    // importances = 10 - [7, 8, 9, 3] = [3, 2, 1, 7]
    expect(arr(res.importances)).toEqual([3, 2, 1, 7]);
    expect(arr(res.importancesMean)[0]).toBe(3.25);
    expect(arr(res.importancesStd)[0]).toBeCloseTo(
      Math.sqrt((0.25 ** 2 + 1.5625 + 5.0625 + 14.0625) / 4),
      12
    );
  });

  it("validates shapes, views and options", () => {
    expect(() => permutationImportance(model, f64([1, 2, 3]), y)).toThrow(ShapeError);
    expect(() => permutationImportance(model, X, f64([[1, 2]]))).toThrow(ShapeError);
    expect(() => permutationImportance(model, X, f64([1, 2]))).toThrow(ShapeError);
    expect(() =>
      permutationImportance(
        model,
        tensor([] as number[][], { dtype: "float64" }).reshape([0, 2]),
        f64([])
      )
    ).toThrow(DataValidationError);
    expect(() =>
      permutationImportance(
        model,
        transpose(
          f64([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
          ])
        ),
        y
      )
    ).toThrow(DataValidationError);
    expect(() => permutationImportance(model, X, y, { nRepeats: 1.5 })).toThrow(/nRepeats/);
    expect(() =>
      permutationImportance({} as { score(X: Tensor, y: Tensor): number }, X, y)
    ).toThrow(InvalidParameterError);
  });

  it("accepts integer feature matrices", () => {
    const Xi = tensor(
      [
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8],
      ],
      { dtype: "int32" }
    );
    const res = permutationImportance(
      { score: (Xs: Tensor) => arr(Xs).length },
      Xi,
      f64([1, 2, 3, 4]),
      { nRepeats: 2, randomState: 0 }
    );
    expect(arr(res.importancesMean)).toEqual([0, 0]);
  });
});
