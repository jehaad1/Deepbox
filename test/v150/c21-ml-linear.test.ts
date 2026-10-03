import { describe, expect, it } from "vitest";
import {
  ConvergenceError,
  catchWarnings,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  type LinearRegression,
  LogisticRegression,
  QuantileRegressor,
  RANSACRegressor,
  type Regressor,
  Ridge,
  SGDClassifier,
  SGDRegressor,
} from "../../src/ml";
import { toFloat64View } from "../../src/ml/_validation";
import { type Tensor, tensor } from "../../src/ndarray";

const f64 = (rows: number | number[] | number[][]): Tensor => tensor(rows, { dtype: "float64" });
const vals = (t: Tensor): number[] => Array.from(toFloat64View(t));

function expectClose(actual: ArrayLike<number>, expected: number[], tol: number): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThanOrEqual(tol);
  }
}

// ---------------------------------------------------------------------------
// Ridge
// ---------------------------------------------------------------------------
describe("Ridge (v1.5.0)", () => {
  // Ill-conditioned design: columns 0 and 1 are almost identical.
  const X = f64([
    [1.0, 1.0002, 3.0],
    [2.0, 2.0001, 1.0],
    [3.0, 3.0004, 4.0],
    [4.0, 4.0003, 1.0],
    [5.0, 5.0006, 5.0],
    [6.0, 6.0005, 9.0],
    [7.0, 7.0008, 2.0],
    [8.0, 8.0007, 6.0],
  ]);
  const y = f64([10.1, 12.2, 9.9, 15.3, 17.0, 21.5, 16.2, 24.9]);

  it("matches scikit-learn on an ill-conditioned problem (float64 normal equations)", () => {
    // sklearn.linear_model.Ridge(alpha, solver="cholesky"). The previous float32 Gram
    // matrix gave 364.3 / -362.7 for alpha = 1e-6 instead of 608.2 / -606.5.
    const tight = new Ridge({ alpha: 1e-6 }).fit(X, y);
    expectClose(
      vals(tight.coef),
      [608.1507398440292, -606.4819496156936, 0.5451037185644999],
      1e-2
    );
    expect(Math.abs(tight.intercept - 6.5385839403793256)).toBeLessThan(1e-3);

    const model = new Ridge({ alpha: 0.5 }).fit(X, y);
    expectClose(
      vals(model.coef),
      [0.8034089257361124, 0.8009596779824709, 0.5449675792966708],
      1e-9
    );
    expect(model.intercept).toBeCloseTo(6.555731481636682, 9);
  });

  it("all direct and iterative solvers agree with scikit-learn", () => {
    const expected = [0.8034089257361124, 0.8009596779824709, 0.5449675792966708];
    for (const solver of ["auto", "cholesky", "svd", "lsqr"] as const) {
      const model = new Ridge({ alpha: 0.5, solver, tol: 1e-12 }).fit(X, y);
      expectClose(vals(model.coef), expected, 1e-7);
      expect(model.intercept).toBeCloseTo(6.555731481636682, 6);
    }
    const sag = new Ridge({ alpha: 0.5, solver: "sag", tol: 1e-10, maxIter: 200000 }).fit(X, y);
    expectClose(vals(sag.coef), expected, 1e-3);
    expect(sag.nIter).toBeGreaterThan(0);
  });

  it("fitIntercept=false matches scikit-learn", () => {
    const model = new Ridge({ alpha: 0.5, fitIntercept: false }).fit(X, y);
    expectClose(vals(model.coef), [1.239325055322334, 1.237831412721552, 0.9013312429822942], 1e-9);
    expect(model.intercept).toBe(0);
  });

  it("svd solver returns the minimum-norm solution for p > n and alpha = 0", () => {
    // 3 samples, 5 features: the system is underdetermined. The minimum-norm solution
    // with a free intercept equals pinv(Xc) @ yc (checked against numpy).
    const Xw = f64([
      [1, 2, 3, 4, 5],
      [2, 1, 0, 1, 2],
      [5, 4, 3, 2, 1],
    ]);
    const yw = f64([1, 0, 1]);
    const model = new Ridge({ alpha: 0, solver: "svd" }).fit(Xw, yw);
    const pred = vals(model.predict(Xw));
    expectClose(pred, [1, 0, 1], 1e-10);
    // Coefficients lie in the row space of the centered data (minimum norm).
    const c = vals(model.coef);
    const norm = Math.sqrt(c.reduce((s, v) => s + v * v, 0));
    const ols = new Ridge({ alpha: 1e-12, solver: "svd" }).fit(Xw, yw);
    const normOls = Math.sqrt(vals(ols.coef).reduce((s, v) => s + v * v, 0));
    expect(norm).toBeLessThanOrEqual(normOls + 1e-9);
  });

  it("normalize=true matches the explicit center / L2-scale / rescale recipe", () => {
    // numpy: Xc = X - mean; s = ||Xc_col||; ridge(alpha=0.7, no intercept) on Xc/s; coef / s
    const Xn = f64([
      [1, 10, 2],
      [2, 7, 1],
      [4, 8, 5],
      [3, 1, 3],
      [5, 4, 9],
    ]);
    const yn = f64([1, 3, 2, 5, 4]);
    const model = new Ridge({ alpha: 0.7, normalize: true }).fit(Xn, yn);
    const manual = new Ridge({ alpha: 0.7, fitIntercept: false });
    const colMean = [3, 6, 4];
    const rows = [
      [1, 10, 2],
      [2, 7, 1],
      [4, 8, 5],
      [3, 1, 3],
      [5, 4, 9],
    ];
    const scale = [0, 1, 2].map((j) =>
      Math.sqrt(rows.reduce((s, r) => s + ((r[j] as number) - (colMean[j] as number)) ** 2, 0))
    );
    const scaled = rows.map((r) =>
      r.map((v, j) => (v - (colMean[j] as number)) / (scale[j] as number))
    );
    manual.fit(f64(scaled), f64([1, 3, 2, 5, 4].map((v) => v - 3)));
    const expectedCoef = vals(manual.coef).map((c, j) => c / (scale[j] as number));
    expectClose(vals(model.coef), expectedCoef, 1e-12);
    const expectedIntercept =
      3 - expectedCoef.reduce((s, c, j) => s + c * (colMean[j] as number), 0);
    expect(model.intercept).toBeCloseTo(expectedIntercept, 12);
  });

  it("returns float64 coefficients and predictions", () => {
    const model = new Ridge().fit(X, y);
    expect(model.coef.dtype).toBe("float64");
    expect(model.predict(X).dtype).toBe("float64");
  });

  it("validates every option at fit time", () => {
    const small = f64([[1], [2], [3]]);
    const t = f64([1, 2, 3]);
    expect(() => new Ridge({ alpha: -1 }).fit(small, t)).toThrow("alpha must be >= 0");
    expect(() => new Ridge({ alpha: Number.POSITIVE_INFINITY }).fit(small, t)).toThrow(
      InvalidParameterError
    );
    expect(() => new Ridge({ maxIter: 0 }).fit(small, t)).toThrow(InvalidParameterError);
    expect(() => new Ridge({ tol: -1 }).fit(small, t)).toThrow(InvalidParameterError);
    expect(() => new Ridge({ solver: "bogus" as unknown as "auto" }).fit(small, t)).toThrow(
      InvalidParameterError
    );
  });

  it("keeps the previous fit when a later fit fails", () => {
    const model = new Ridge({ alpha: 0.5 }).fit(X, y);
    const before = vals(model.coef);
    model.setParams({ alpha: -2 });
    expect(() => model.fit(X, y)).toThrow(InvalidParameterError);
    expectClose(vals(model.coef), before, 0);
  });

  it("getParams reports effective defaults and round-trips through the constructor", () => {
    const params = new Ridge().getParams();
    expect(params).toEqual({
      alpha: 1,
      fitIntercept: true,
      normalize: false,
      solver: "auto",
      maxIter: 1000,
      tol: 1e-4,
    });
    const clone = new Ridge({ alpha: 3, solver: "svd" }).clone();
    expect(clone.getParams()["alpha"]).toBe(3);
    expect(clone.getParams()["solver"]).toBe("svd");
  });

  it("score rejects empty and mismatched targets, and 1.0 for exact constant targets", () => {
    const model = new Ridge({ alpha: 0 }).fit(f64([[1], [2], [3]]), f64([5, 5, 5]));
    expect(model.score(f64([[1], [2]]), f64([5, 5]))).toBe(1);
    expect(model.score(f64([[1], [2]]), f64([5, 6]))).toBeLessThan(1);
    expect(() => model.score(f64([[1], [2]]), f64([5]))).toThrow(ShapeError);
    expect(() => model.score(tensor([[1]], { dtype: "float64" }).reshape([1, 1]), f64([]))).toThrow(
      DataValidationError
    );
  });

  it("falls back to the minimum-norm solution on singular systems with the direct solvers", () => {
    // Rank-deficient X (second column is twice the first) with alpha = 0. scikit-learn catches
    // the failed Cholesky solve and returns the SVD (minimum-norm) solution:
    // Ridge(alpha=0, fit_intercept=False).fit(X, y).coef_ == [0.2, 0.4]
    const Xs = f64([
      [1, 2],
      [2, 4],
      [3, 6],
    ]);
    const ys = f64([1, 2, 3]);
    for (const solver of ["auto", "cholesky"] as const) {
      const model = new Ridge({ alpha: 0, fitIntercept: false, solver }).fit(Xs, ys);
      const coef = vals(model.coef);
      expect(coef[0]).toBeCloseTo(0.2, 10);
      expect(coef[1]).toBeCloseTo(0.4, 10);
    }
  });

  it("score accepts int64 (BigInt) targets", () => {
    const X1 = f64([[1], [2], [3], [4]]);
    const y64 = tensor(new BigInt64Array([2n, 4n, 6n, 8n]), { dtype: "int64" });
    const model = new Ridge({ alpha: 0 }).fit(X1, y64);
    expect(model.score(X1, y64)).toBeCloseTo(1, 10);
    expect(new QuantileRegressor({ alpha: 0 }).fit(X1, y64).score(X1, y64)).toBeCloseTo(1, 6);
    expect(new SGDRegressor({ maxIter: 50 }).fit(X1, y64).score(X1, y64)).toBeLessThanOrEqual(1);
  });

  it("is not fitted before fit", () => {
    const m = new Ridge();
    expect(() => m.coef).toThrow(NotFittedError);
    expect(() => m.intercept).toThrow(NotFittedError);
    expect(() => m.nIter).toThrow(NotFittedError);
    expect(() => m.score(X, y)).toThrow(NotFittedError);
  });
});

// ---------------------------------------------------------------------------
// LogisticRegression
// ---------------------------------------------------------------------------
describe("LogisticRegression (v1.5.0)", () => {
  const X = f64([
    [0.03, 4.08],
    [1.22, -1.53],
    [-0.3, -1.58],
    [0.57, -0.17],
    [0.75, -5.54],
    [1.57, -0.29],
    [0.68, -0.41],
    [-0.38, 1.39],
    [0.82, -0.61],
    [-0.15, 2.06],
    [-0.87, -4.54],
    [0.39, -2.01],
    [-1.92, -2.44],
    [-0.47, -3.58],
  ]);
  const yb = f64([2, 7, 7, 7, 7, 7, 7, 2, 7, 2, 7, 7, 2, 7]);
  const ym = f64([5, 25, 15, 15, 25, 25, 25, 5, 25, 15, 15, 15, 5, 25]);
  const tight = { tol: 1e-12, maxIter: 5000 };
  const first3 = f64([
    [0.03, 4.08],
    [1.22, -1.53],
    [-0.3, -1.58],
  ]);
  const first2 = f64([
    [0.03, 4.08],
    [1.22, -1.53],
  ]);

  it("binary L2 matches scikit-learn (C=1)", () => {
    const m = new LogisticRegression(tight).fit(X, yb);
    expectClose(vals(m.coef), [1.3585080720780633, -0.8408773113828126], 1e-6);
    expect(m.intercept as number).toBeCloseTo(0.44343602561947726, 6);
    const proba = vals(m.predictProba(first3));
    expectClose(
      proba,
      [
        0.9501019875412171, 0.04989801245878293, 0.032692897984406, 0.967307102015594,
        0.20351481011361283, 0.7964851898863872,
      ],
      1e-6
    );
    expectClose(
      vals(m.decisionFunction(first3)),
      [-2.9465881626600563, 3.387358159970418, 1.3644697559809025],
      1e-5
    );
    // Labels are mapped back: classes_[1] is the positive class.
    expect(vals(m.classes as Tensor)).toEqual([2, 7]);
    expect(vals(m.predict(first3))).toEqual([2, 7, 7]);
  });

  it("class weights follow scikit-learn (balanced and explicit dictionary)", () => {
    const balanced = new LogisticRegression({ ...tight, C: 0.2, classWeight: "balanced" }).fit(
      X,
      yb
    );
    expectClose(vals(balanced.coef), [0.5639852806297874, -0.505921337481407], 1e-6);
    expect(balanced.intercept as number).toBeCloseTo(-0.14076480190999957, 6);

    const dict = new LogisticRegression({ ...tight, classWeight: { 2: 1, 7: 4 } }).fit(X, yb);
    expectClose(vals(dict.coef), [1.7415146065187372, -1.0315573129990616], 1e-6);
    expect(dict.intercept as number).toBeCloseTo(1.3025574774258377, 6);
  });

  it("fitIntercept=false matches scikit-learn", () => {
    const m = new LogisticRegression({ ...tight, fitIntercept: false }).fit(X, yb);
    expectClose(vals(m.coef), [1.3579207226934764, -0.9149168229880318], 1e-6);
    expect(m.intercept).toBe(0);
  });

  it("multinomial softmax matches scikit-learn and is the default for three classes", () => {
    const expectedCoef = [
      -1.0976585466549076, 0.4188398120523105, -0.06079758251448511, -0.05726777615135473,
      1.158456129169393, -0.3615720359009563,
    ];
    const expectedIntercept = [-0.24602220574539294, 0.4417075422999395, -0.19568533655454604];
    for (const multiClass of ["auto", "multinomial"] as const) {
      const m = new LogisticRegression({ ...tight, multiClass }).fit(X, ym);
      expectClose(vals(m.coef), expectedCoef, 1e-5);
      expectClose(m.intercept as number[], expectedIntercept, 1e-5);
      expect(m.coef.shape).toEqual([3, 2]);
      const proba = m.predictProba(first2);
      expectClose(
        vals(proba),
        [
          0.7458516838550528, 0.21938833579655112, 0.03475998034839608, 0.014280210654822096,
          0.20851676439716493, 0.777203024948013,
        ],
        1e-5
      );
      expect(vals(m.predict(X))).toEqual([5, 25, 15, 25, 25, 25, 25, 5, 25, 5, 15, 25, 5, 15]);
    }
  });

  it("multinomial with balanced class weights matches scikit-learn", () => {
    const m = new LogisticRegression({ ...tight, classWeight: "balanced" }).fit(X, ym);
    expectClose(
      vals(m.coef),
      [
        -1.1951928250616324, 0.4448423972853114, 0.058665817261756605, -0.08532178207827927,
        1.1365270077998757, -0.35952061520703255,
      ],
      1e-5
    );
    expectClose(
      m.intercept as number[],
      [-0.014488898134497444, 0.3506556851651547, -0.3361667870306575],
      1e-5
    );
  });

  it("one-vs-rest probabilities are normalized and predict uses the raw scores", () => {
    const m = new LogisticRegression({ multiClass: "ovr", tol: 1e-10 }).fit(X, ym);
    const proba = vals(m.predictProba(X));
    for (let i = 0; i < 14; i++) {
      expect(proba[i * 3]! + proba[i * 3 + 1]! + proba[i * 3 + 2]!).toBeCloseTo(1, 12);
    }
    // Scaled-up inputs saturate the logistic function; the argmax of the raw
    // scores must still decide the label.
    const big = f64(
      X.shape[0] === 14
        ? vals(X).reduce<number[][]>((rows, v, k) => {
            if (k % 2 === 0) rows.push([v * 1e4]);
            else (rows[rows.length - 1] as number[]).push(v * 1e4);
            return rows;
          }, [])
        : []
    );
    const scores = vals(m.decisionFunction(big));
    const pred = vals(m.predict(big));
    const labels = vals(m.classes as Tensor);
    for (let i = 0; i < 14; i++) {
      const row = scores.slice(i * 3, i * 3 + 3);
      expect(pred[i]).toBe(labels[row.indexOf(Math.max(...row))]);
    }
  });

  it("L1 matches scikit-learn (saga) and produces exact zeros", () => {
    const m = new LogisticRegression({
      penalty: "l1",
      solver: "saga",
      tol: 1e-7,
      maxIter: 20000,
      C: 1,
    }).fit(X, yb);
    expectClose(vals(m.coef), [1.6739698751167622, -0.8686907673775465], 1e-5);
    expect(m.intercept as number).toBeCloseTo(0.4979385955753699, 5);

    const sparse = new LogisticRegression({ penalty: "l1", solver: "saga", C: 0.1, tol: 1e-7 }).fit(
      X,
      yb
    );
    expect(vals(sparse.coef)).toEqual([0, 0]);
    // With every weight at zero the intercept is the log-odds of the positive class: log(10 / 4).
    expect(sparse.intercept as number).toBeCloseTo(Math.log(10 / 4), 5);
  });

  it("accepts multiClass='multinomial' and rejects it with liblinear", () => {
    expect(() => new LogisticRegression({ multiClass: "multinomial" })).not.toThrow();
    expect(() =>
      new LogisticRegression({ solver: "liblinear", multiClass: "multinomial" }).fit(X, ym)
    ).toThrow(InvalidParameterError);
    // liblinear + auto falls back to one-vs-rest.
    const m = new LogisticRegression({ solver: "liblinear" }).fit(X, ym);
    expect(m.coef.shape).toEqual([3, 2]);
  });

  it("keeps large labels exact (float64 classes and predictions)", () => {
    const m = new LogisticRegression().fit(
      f64([[0], [1], [2], [3]]),
      f64([16777217, 16777217, 16777219, 16777219])
    );
    expect(m.classes?.dtype).toBe("float64");
    expect(vals(m.classes as Tensor)).toEqual([16777217, 16777219]);
    expect(vals(m.predict(f64([[0], [3]])))).toEqual([16777217, 16777219]);
  });

  it("accepts int64 labels", () => {
    const y64 = tensor(new BigInt64Array([0n, 0n, 1n, 1n]), { dtype: "int64" });
    const m = new LogisticRegression().fit(f64([[0], [1], [2], [3]]), y64);
    expect(vals(m.predict(f64([[0], [3]])))).toEqual([0, 1]);
  });

  it("single-class data gives a constant model with one probability column", () => {
    const m = new LogisticRegression().fit(f64([[0], [1]]), f64([3, 3]));
    expect(vals(m.predict(f64([[5], [6]])))).toEqual([3, 3]);
    const proba = m.predictProba(f64([[5], [6]]));
    expect(proba.shape).toEqual([2, 1]);
    expect(vals(proba)).toEqual([1, 1]);
  });

  it("emits a ConvergenceWarning when maxIter is exhausted", () => {
    const warnings = catchWarnings(() => {
      new LogisticRegression({ maxIter: 2, tol: 1e-12 }).fit(X, yb);
    });
    expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
    const quiet = catchWarnings(() => {
      new LogisticRegression().fit(X, yb);
    });
    expect(quiet.length).toBe(0);
  });

  it("reports the iteration count", () => {
    const m = new LogisticRegression().fit(X, yb);
    expect(m.nIter).toBeGreaterThan(0);
    expect(m.nIter).toBeLessThan(100);
  });

  it("handles unscaled features (1e6) without diverging", () => {
    const m = new LogisticRegression().fit(
      f64([[0], [1e6], [2e6], [3e6], [1.5e6]]),
      f64([0, 0, 1, 1, 1])
    );
    expect(m.score(f64([[0], [1e6], [2e6], [3e6], [1.5e6]]), f64([0, 0, 1, 1, 1]))).toBe(1);
  });

  it("setParams accepts classWeight, so getParams output can be applied back", () => {
    const a = new LogisticRegression({ classWeight: { 2: 1, 7: 3 }, C: 2 });
    const b = new LogisticRegression().setParams(a.getParams());
    expect(b.getParams()).toEqual(a.getParams());
    b.setParams({ classWeight: undefined });
    expect(b.getParams()["classWeight"]).toBeUndefined();
    expect(() => b.setParams({ classWeight: { 2: -1 } })).toThrow(InvalidParameterError);
    expect(() => b.setParams({ classWeight: { abc: 1 } })).toThrow(InvalidParameterError);
    expect(() => b.setParams({ classWeight: "heavy" })).toThrow(InvalidParameterError);
  });

  it("copies classWeight so later edits to the caller's object do not leak in", () => {
    const cw: Record<number, number> = { 2: 1, 7: 4 };
    const m = new LogisticRegression({ ...tight, classWeight: cw });
    cw[7] = 1;
    const fitted = m.fit(X, yb);
    expect(fitted.intercept as number).toBeCloseTo(1.3025574774258377, 5);
  });

  it("reports effective defaults and clones", () => {
    const params = new LogisticRegression().getParams();
    expect(params["maxIter"]).toBe(1000);
    expect(params["tol"]).toBe(1e-4);
    expect(params["C"]).toBe(1);
    expect(params["fitIntercept"]).toBe(true);
    expect(params["learningRate"]).toBe(0.1);
    const clone = new LogisticRegression({ C: 5, classWeight: "balanced" }).clone();
    expect(clone.getParams()["C"]).toBe(5);
    expect(clone.getParams()["classWeight"]).toBe("balanced");
  });

  it("validates inputs and options", () => {
    expect(() => new LogisticRegression({ C: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new LogisticRegression({ C: 0 })).toThrow("C must be > 0");
    expect(() => new LogisticRegression({ classWeight: { 1: Number.NaN } })).toThrow(
      InvalidParameterError
    );
    expect(() => new LogisticRegression({ multiClass: "bogus" as unknown as "ovr" })).toThrow(
      InvalidParameterError
    );
    const m = new LogisticRegression().fit(X, yb);
    expect(() => m.predict(f64([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => m.score(X, f64([]))).toThrow(DataValidationError);
    expect(() => m.score(X, f64([1, 2]))).toThrow(ShapeError);
    expect(() => new LogisticRegression().decisionFunction(X)).toThrow(NotFittedError);
  });

  it("an all-zero class weight vector is rejected", () => {
    expect(() => new LogisticRegression({ classWeight: { 2: 0, 7: 0 } }).fit(X, yb)).toThrow(
      DataValidationError
    );
  });

  it("keeps the previous model when a refit fails", () => {
    const m = new LogisticRegression().fit(X, yb);
    const before = vals(m.coef);
    expect(() => m.fit(f64([[Number.NaN, 1]]), f64([1]))).toThrow(DataValidationError);
    expectClose(vals(m.coef), before, 0);
    expect(vals(m.classes as Tensor)).toEqual([2, 7]);
  });
});

// ---------------------------------------------------------------------------
// QuantileRegressor
// ---------------------------------------------------------------------------
describe("QuantileRegressor (v1.5.0)", () => {
  const X = f64([
    [0.36, 6.04],
    [-1.79, 6.75],
    [-0.05, -3.2],
    [-0.8, -4.33],
    [-0.22, 3.34],
    [0.58, 2.55],
    [-1.69, -6.28],
    [1.55, 3.88],
    [2.18, 4.84],
    [-1.02, 5.14],
    [0.63, 0.86],
    [-0.82, 0.01],
  ]);
  const y = f64([-1.42, -3.25, 3.4, 2.49, -0.48, 0.25, 0.49, 1.42, 3.04, -2.73, 3.28, -2.38]);

  it("matches the scikit-learn linear program (quantile 0.8, no penalty)", () => {
    const m = new QuantileRegressor({ quantile: 0.8, alpha: 0 }).fit(X, y);
    expectClose(m.coef, [1.3788452129879678, -0.4272170180302448], 1e-6);
    expect(m.intercept).toBeCloseTo(2.101847802952615, 6);
  });

  it("alpha is an L1 penalty on the mean pinball loss, as in scikit-learn", () => {
    // sklearn QuantileRegressor(quantile=0.3, alpha=0.05, solver="highs")
    const m = new QuantileRegressor({ quantile: 0.3, alpha: 0.05 }).fit(X, y);
    expectClose(m.coef, [1.2841212957270596, -0.35729958565123726], 1e-6);
    expect(m.intercept).toBeCloseTo(0.4163235918889605, 6);

    // A strong L1 penalty produces exact zeros; the intercept alone is the 0.5 quantile.
    const strong = new QuantileRegressor({ alpha: 1000 }).fit(X, y);
    expect(Array.from(strong.coef)).toEqual([0, 0]);
  });

  it("supports fitIntercept=false", () => {
    const m = new QuantileRegressor({ quantile: 0.5, alpha: 0.02, fitIntercept: false }).fit(X, y);
    expectClose(m.coef, [1.4772849414209452, -0.2379706925584894], 1e-6);
    expect(m.intercept).toBe(0);
  });

  it("recovers an exact linear relation and is scale invariant", () => {
    const x = f64([[1e6], [2e6], [3e6], [4e6], [5e6]]);
    const yy = f64([3e6 + 1, 6e6 + 1, 9e6 + 1, 12e6 + 1, 15e6 + 1]);
    const m = new QuantileRegressor({ alpha: 0 }).fit(x, yy);
    expect(m.coef[0]).toBeCloseTo(3, 6);
    expect(m.intercept).toBeCloseTo(1, 2);
  });

  it("handles more features than samples and duplicate columns", () => {
    const wide = new QuantileRegressor({ alpha: 0 }).fit(
      f64([
        [1, 2, 3],
        [2, 1, 0],
      ]),
      f64([1, 2])
    );
    expectClose(
      vals(
        wide.predict(
          f64([
            [1, 2, 3],
            [2, 1, 0],
          ])
        )
      ),
      [1, 2],
      1e-6
    );

    const dup = new QuantileRegressor({ alpha: 0 }).fit(
      f64([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
      ]),
      f64([2, 4, 6, 8])
    );
    expect(dup.coef[0]! + dup.coef[1]!).toBeCloseTo(2, 5);
  });

  it("predicts float64 and scores with R2", () => {
    const m = new QuantileRegressor({ alpha: 0 }).fit(X, y);
    expect(m.predict(X).dtype).toBe("float64");
    expect(m.coefTensor.dtype).toBe("float64");
    expect(m.score(X, y)).toBeLessThanOrEqual(1);
    expect(m.nIter).toBeGreaterThan(0);
    expect(() => m.score(X, f64([1]))).toThrow(ShapeError);
  });

  it("setParams changes the fit and validates", () => {
    const m = new QuantileRegressor({ alpha: 0 });
    m.setParams({ quantile: 0.8 });
    expect(m.getParams()["quantile"]).toBe(0.8);
    m.fit(X, y);
    expect(m.intercept).toBeCloseTo(2.101847802952615, 6);
    expect(() => m.setParams({ quantile: 1.5 })).toThrow(InvalidParameterError);
    expect(m.getParams()["quantile"]).toBe(0.8);
    expect(() => m.setParams({ nope: 1 })).toThrow(InvalidParameterError);
    expect(() => m.setParams({ fitIntercept: "yes" })).toThrow(InvalidParameterError);
    expect(m.clone().getParams()).toEqual(m.getParams());
  });

  it("validates constructor options", () => {
    expect(() => new QuantileRegressor({ quantile: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new QuantileRegressor({ alpha: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new QuantileRegressor({ maxIter: 0 })).toThrow(InvalidParameterError);
    expect(() => new QuantileRegressor({ tol: 0 })).toThrow(InvalidParameterError);
    expect(() => new QuantileRegressor().predict(X)).toThrow(NotFittedError);
    expect(() => new QuantileRegressor().nIter).toThrow(NotFittedError);
  });
});

// ---------------------------------------------------------------------------
// RANSACRegressor
// ---------------------------------------------------------------------------
describe("RANSACRegressor (v1.5.0)", () => {
  const xs = Array.from({ length: 20 }, (_, i) => [i]);
  const ys = xs.map(([x]) => 2 * (x as number) + 1);
  ys[3] = 80;
  ys[11] = -40;
  ys[16] = 95;
  const X = f64(xs);
  const y = f64(ys);

  it("finds the line and flags the outliers (default MAD threshold, default minSamples)", () => {
    const m = new RANSACRegressor({ randomState: 7 }).fit(X, y);
    const mask = Array.from(m.inlierMask);
    expect(mask.filter((v) => v === 0)).toHaveLength(3);
    expect(mask[3]).toBe(0);
    expect(mask[11]).toBe(0);
    expect(mask[16]).toBe(0);
    const base = m.estimator as LinearRegression;
    expect(vals(base.coef)[0]).toBeCloseTo(2, 8);
    expect(vals(m.predict(f64([[100]])))[0]).toBeCloseTo(201, 6);
    expect(m.nTrials).toBeGreaterThan(0);
  });

  it("is reproducible with randomState and honors a custom threshold and loss", () => {
    const a = new RANSACRegressor({ randomState: 3, minSamples: 2, residualThreshold: 1 }).fit(
      X,
      y
    );
    const b = new RANSACRegressor({ randomState: 3, minSamples: 2, residualThreshold: 1 }).fit(
      X,
      y
    );
    expect(Array.from(a.inlierMask)).toEqual(Array.from(b.inlierMask));
    const sq = new RANSACRegressor({
      randomState: 3,
      minSamples: 2,
      residualThreshold: 1,
      loss: "squared_error",
    }).fit(X, y);
    expect(Array.from(sq.inlierMask).filter((v) => v === 0)).toHaveLength(3);
  });

  it("minSamples may be a fraction and must not exceed the sample count", () => {
    const frac = new RANSACRegressor({ randomState: 1, minSamples: 0.2 }).fit(X, y);
    expect(frac.inlierMask.length).toBe(20);
    expect(() => new RANSACRegressor({ minSamples: 25 }).fit(X, y)).toThrow(InvalidParameterError);
    expect(() => new RANSACRegressor({ minSamples: 1.5 })).toThrow(InvalidParameterError);
    // Default minSamples is n_features + 1.
    expect(() =>
      new RANSACRegressor().fit(
        f64([
          [1, 2, 3, 4],
          [2, 3, 4, 5],
        ]),
        f64([1, 2])
      )
    ).toThrow(InvalidParameterError);
  });

  it("throws ConvergenceError when no trial yields an inlier", () => {
    const hopeless = (): Regressor => ({
      fit() {
        return hopeless();
      },
      predict(Z: Tensor) {
        return f64(new Array<number>(Z.shape[0] ?? 0).fill(1e9));
      },
      score: () => 0,
      getParams: () => ({}),
      setParams() {
        return hopeless();
      },
      clone: () => hopeless(),
    });
    expect(() => new RANSACRegressor({ estimator: hopeless(), randomState: 0 }).fit(X, y)).toThrow(
      ConvergenceError
    );
  });

  it("stopNInliers ends the search early", () => {
    const m = new RANSACRegressor({
      randomState: 5,
      minSamples: 2,
      residualThreshold: 0.5,
      stopNInliers: 17,
      maxTrials: 500,
    }).fit(X, y);
    expect(m.nTrials).toBeLessThan(500);
    expect(m.inlierMask.reduce((s, v) => s + v, 0)).toBeGreaterThanOrEqual(17);
  });

  it("uses a custom base estimator and clones it", () => {
    const m = new RANSACRegressor({
      estimator: new Ridge({ alpha: 1e-9 }),
      randomState: 2,
      minSamples: 3,
      residualThreshold: 0.5,
    }).fit(X, y);
    expect(m.estimator).toBeInstanceOf(Ridge);
    expect(vals(m.predict(f64([[50]])))[0]).toBeCloseTo(101, 4);
    const clone = m.clone();
    expect(clone.getParams()["estimator"]).toBeInstanceOf(Ridge);
    expect(clone.getParams()["estimator"]).not.toBe(m.getParams()["estimator"]);
    expect(() => new RANSACRegressor({ estimator: {} as never })).toThrow(InvalidParameterError);
  });

  it("setParams changes the configuration and validates atomically", () => {
    const m = new RANSACRegressor({ randomState: 1 });
    m.setParams({ maxTrials: 7, residualThreshold: 2 });
    expect(m.getParams()["maxTrials"]).toBe(7);
    expect(m.getParams()["residualThreshold"]).toBe(2);
    expect(() => m.setParams({ maxTrials: 0, residualThreshold: 9 })).toThrow(
      InvalidParameterError
    );
    expect(m.getParams()["maxTrials"]).toBe(7);
    expect(m.getParams()["residualThreshold"]).toBe(2);
    expect(() => m.setParams({ unknown: 1 })).toThrow(InvalidParameterError);
    expect(() => new RANSACRegressor({ loss: "huber" as never })).toThrow(InvalidParameterError);
    expect(() => new RANSACRegressor({ residualThreshold: -1 })).toThrow(InvalidParameterError);
    expect(() => new RANSACRegressor({ stopProbability: 2 })).toThrow(InvalidParameterError);
  });

  it("score validates its arguments and requires a fit", () => {
    const m = new RANSACRegressor({ randomState: 7 });
    expect(() => m.score(X, y)).toThrow(NotFittedError);
    expect(() => m.estimator).toThrow(NotFittedError);
    expect(() => m.nTrials).toThrow(NotFittedError);
    m.fit(X, y);
    expect(() => m.score(X, f64([1]))).toThrow(ShapeError);
    expect(m.score(f64([[1], [2], [3]]), f64([3, 5, 7]))).toBeCloseTo(1, 8);
  });

  it("does not lose precision on large offsets (float64 subsets)", () => {
    const xo = f64(Array.from({ length: 12 }, (_, i) => [i]));
    const yo = f64(Array.from({ length: 12 }, (_, i) => 1e8 + 0.25 * i));
    const m = new RANSACRegressor({ randomState: 4, minSamples: 2 }).fit(xo, yo);
    const base = m.estimator as LinearRegression;
    expect(vals(base.coef)[0]).toBeCloseTo(0.25, 6);
  });
});

// ---------------------------------------------------------------------------
// SGDClassifier / SGDRegressor
// ---------------------------------------------------------------------------
describe("SGD (v1.5.0)", () => {
  const Xs = f64([
    [-0.802, -1.324],
    [-0.248, 0.42],
    [1.136, 0.11],
    [-0.553, -0.785],
    [0.749, 1.635],
    [0.273, -1.233],
    [-0.958, 1.6],
    [0.203, -1.732],
    [-0.084, -1.163],
    [-0.629, -0.488],
  ]);
  const ysb = f64([5, 1, 5, 5, 1, 5, 1, 5, 5, 1]);
  const ysm = f64([1, 0, 2, 1, 0, 2, 0, 2, 2, 1]);
  const ysr = f64([0.22, -0.416, 2.662, 0.179, 0.363, 2.279, -3.016, 2.638, 1.495, -0.27]);
  // Without shuffling the update sequence is deterministic and identical to scikit-learn.
  const det = { shuffle: false, tol: 0, maxIter: 12 };

  it("hinge with the 'optimal' schedule matches scikit-learn", () => {
    const m = new SGDClassifier(det).fit(Xs, ysb);
    expectClose(m.coef, [10.822162645218935, -10.241286863270773], 1e-9);
    expect(m.intercept).toBeCloseTo(0.0792271551249133, 9);
    expect(m.nIter).toBe(12);
  });

  it("log_loss, modified_huber, elasticnet and l1 follow scikit-learn's update rules", () => {
    const log = new SGDClassifier({
      ...det,
      loss: "log_loss",
      learningRate: "constant",
      eta0: 0.05,
    }).fit(Xs, ysb);
    expectClose(log.coef, [0.4009289854158356, -1.319365230671691], 1e-9);
    expect(log.intercept).toBeCloseTo(0.13361082888698328, 9);

    const mh = new SGDClassifier({
      ...det,
      loss: "modified_huber",
      penalty: "l1",
      alpha: 0.01,
      learningRate: "constant",
      eta0: 0.02,
    }).fit(Xs, ysb);
    expectClose(mh.coef, [0.5080547255105077, -0.823215671296231], 1e-9);
    expect(mh.intercept).toBeCloseTo(-0.01836114178249314, 9);

    const enet = new SGDClassifier({
      ...det,
      penalty: "elasticnet",
      alpha: 0.02,
      l1Ratio: 0.5,
      learningRate: "invscaling",
      eta0: 0.05,
    }).fit(Xs, ysb);
    expectClose(enet.coef, [0.07644878025891795, -0.7396788275852625], 1e-9);
    expect(enet.intercept).toBeCloseTo(0.15365950303046533, 9);
  });

  it("modified_huber probabilities are (clip(score, -1, 1) + 1) / 2", () => {
    const m = new SGDClassifier({
      ...det,
      loss: "modified_huber",
      learningRate: "constant",
      eta0: 0.02,
    }).fit(Xs, ysb);
    const first3 = f64([
      [-0.802, -1.324],
      [-0.248, 0.42],
      [1.136, 0.11],
    ]);
    expectClose(
      vals(m.decisionFunction(first3)),
      [0.6653833980307549, -0.5016550934371261, 0.48573632173359405],
      1e-9
    );
    expectClose(
      vals(m.predictProba(first3)),
      [
        0.16730830098462257, 0.8326916990153774, 0.750827546718563, 0.24917245328143695,
        0.2571318391332029, 0.7428681608667971,
      ],
      1e-9
    );
  });

  it("multiclass exposes per-class coefficients and intercepts", () => {
    const m = new SGDClassifier({ ...det, learningRate: "constant", eta0: 0.03 }).fit(Xs, ysm);
    expectClose(
      vals(m.coefTensor),
      [
        -0.1816235529529634, 1.0587703270576425, -0.6588620419610524, -0.3395933686968077,
        1.1347424529930994, -0.9120935193252562,
      ],
      1e-9
    );
    expect(m.coefTensor.shape).toEqual([3, 2]);
    expect(m.coef.length).toBe(6);
    expectClose(
      m.interceptArray,
      [-0.5400000000000003, -0.9600000000000006, -0.4800000000000002],
      1e-9
    );
    expect(() => m.intercept).toThrow(DataValidationError);
    expect(m.decisionFunction(Xs).shape).toEqual([10, 3]);
  });

  it("partialFit continues the schedule exactly like scikit-learn's partial_fit", () => {
    const opts = {
      loss: "log_loss" as const,
      shuffle: false,
      learningRate: "constant" as const,
      eta0: 0.05,
    };
    const m = new SGDClassifier(opts);
    for (let k = 0; k < 3; k++) m.partialFit(Xs, ysb, [1, 5]);
    expectClose(m.coef, [0.09702616303822685, -0.5449834816902441], 1e-9);
    expect(m.intercept).toBeCloseTo(0.09218389779124728, 9);
    expect(() => m.partialFit(Xs, f64([9, 1, 5, 5, 1, 5, 1, 5, 5, 1]))).toThrow(
      DataValidationError
    );

    const r = new SGDRegressor({ shuffle: false, learningRate: "constant", eta0: 0.02 });
    for (let k = 0; k < 3; k++) r.partialFit(Xs, ysr);
    expectClose(r.coef, [0.410504002418929, -0.5689242951753365], 1e-9);
    expect(r.intercept).toBeCloseTo(0.23588048775058168, 9);
  });

  it("regressor matches scikit-learn for squared_error elasticnet and huber", () => {
    const enet = new SGDRegressor({
      ...det,
      eta0: 0.02,
      penalty: "elasticnet",
      alpha: 0.01,
      l1Ratio: 0.3,
    }).fit(Xs, ysr);
    expectClose(enet.coef, [0.6083101993012039, -0.696684658167991], 1e-9);
    expect(enet.intercept).toBeCloseTo(0.3097215977954226, 9);

    const huber = new SGDRegressor({
      ...det,
      loss: "huber",
      learningRate: "constant",
      eta0: 0.02,
    }).fit(Xs, ysr);
    expectClose(huber.coef, [0.07260803042746139, -0.13453004528834267], 1e-9);
    expect(huber.intercept).toBeCloseTo(0.08646649255233678, 9);
  });

  it("l1 penalty drives weights to exactly zero and matches scikit-learn", () => {
    const r = new SGDRegressor({
      penalty: "l1",
      alpha: 1,
      learningRate: "constant",
      eta0: 0.01,
      shuffle: false,
      maxIter: 200,
      tol: 0,
    }).fit(Xs, ysr);
    expect(r.coef[0]).toBe(0);
    expect(r.coef[1]).toBeCloseTo(-0.16521800082942217, 9);
    expect(r.intercept).toBeCloseTo(0.5556129548551818, 9);
    expect(r.nIter).toBe(200);
  });

  it("stops on the mean training objective like scikit-learn (same epoch count)", () => {
    const log = new SGDClassifier({
      loss: "log_loss",
      shuffle: false,
      learningRate: "constant",
      eta0: 0.05,
    }).fit(Xs, ysb);
    expect(log.nIter).toBe(75);
    expectClose(log.coef, [1.820207528800025, -2.8079964026440236], 1e-9);
    expect(log.intercept).toBeCloseTo(-0.18006926021180847, 9);

    const hinge = new SGDClassifier({ shuffle: false }).fit(Xs, ysb);
    expect(hinge.nIter).toBe(7);
    expectClose(hinge.coef, [11.328344246959764, -10.720299345182408], 1e-9);

    const reg = new SGDRegressor({
      shuffle: false,
      penalty: "elasticnet",
      alpha: 0.05,
      l1Ratio: 0.5,
      learningRate: "constant",
      eta0: 0.01,
      tol: 1e-4,
      nIterNoChange: 3,
    }).fit(Xs, ysr);
    expect(reg.nIter).toBe(78);
    expectClose(reg.coef, [1.7672589125592308, -0.9537352822834806], 1e-9);
    expect(reg.intercept).toBeCloseTo(0.4880522625787769, 9);
  });

  it("one-vs-rest class weights follow scikit-learn (own class weight vs 1 for the rest)", () => {
    const opts = {
      shuffle: false,
      tol: 0,
      maxIter: 30,
      learningRate: "constant" as const,
      eta0: 0.01,
    };
    const m = new SGDClassifier({ ...opts, classWeight: { 0: 1, 1: 3, 2: 2 } }).fit(Xs, ysm);
    expectClose(
      vals(m.coefTensor),
      [
        -0.12559831481266498, 0.98625368888682, -1.0547650384459921, -0.6502098440655482,
        1.1119456788824318, -0.9689438322547528,
      ],
      1e-9
    );
    expectClose(
      m.interceptArray,
      [-0.5000000000000002, -0.06999999999999973, -0.15999999999999975],
      1e-9
    );

    const bal = new SGDClassifier({ ...opts, classWeight: "balanced" }).fit(Xs, ysb);
    expectClose(bal.coef, [0.35497341697900386, -0.973574611272785], 1e-9);
    expect(bal.intercept).toBeCloseTo(-0.008333333333333385, 9);
    expect(() => new SGDClassifier({ classWeight: "balanced" }).partialFit(Xs, ysb)).toThrow(
      InvalidParameterError
    );
  });

  it("is reproducible with randomState, and different seeds shuffle differently", () => {
    const run = (seed: number) =>
      Array.from(new SGDClassifier({ randomState: seed, maxIter: 3, tol: 0 }).fit(Xs, ysb).coef);
    expect(run(1)).toEqual(run(1));
    expect(run(1)).not.toEqual(run(2));
  });

  it("rejects unknown losses, penalties and schedules instead of silently ignoring them", () => {
    expect(() => new SGDClassifier({ loss: "nope" as never })).toThrow(InvalidParameterError);
    expect(() => new SGDClassifier({ loss: "huber" as never })).toThrow(InvalidParameterError);
    expect(() => new SGDRegressor({ loss: "hinge" as never })).toThrow(InvalidParameterError);
    expect(() => new SGDClassifier({ penalty: "l3" as never })).toThrow(InvalidParameterError);
    expect(() => new SGDClassifier({ learningRate: "cosine" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => new SGDClassifier({ l1Ratio: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new SGDClassifier({ eta0: 0, learningRate: "constant" })).toThrow(
      InvalidParameterError
    );
    expect(() => new SGDClassifier({ alpha: 0 })).toThrow(InvalidParameterError); // 'optimal' needs alpha > 0
    expect(() => new SGDClassifier({ alpha: 0, learningRate: "constant" })).not.toThrow();
  });

  it("setParams really updates the estimator and is atomic", () => {
    const clf = new SGDClassifier({ loss: "hinge" });
    clf.setParams({ loss: "log_loss", alpha: 0.01 });
    expect(clf.getParams()["loss"]).toBe("log_loss");
    expect(clf.getParams()["alpha"]).toBe(0.01);
    clf.fit(Xs, ysb);
    expect(() => clf.predictProba(Xs)).not.toThrow();
    expect(() => clf.setParams({ alpha: 0.5, loss: "bogus" })).toThrow(InvalidParameterError);
    expect(clf.getParams()["alpha"]).toBe(0.01);
    expect(() => clf.setParams({ epsilon: 1 })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ whatever: 1 })).toThrow(InvalidParameterError);
    expect(clf.clone().getParams()).toEqual(clf.getParams());

    const reg = new SGDRegressor();
    reg.setParams({ epsilon: 0.3, loss: "huber" });
    expect(reg.getParams()["epsilon"]).toBe(0.3);
    expect(() => reg.setParams({ classWeight: "balanced" })).toThrow(InvalidParameterError);
    expect(reg.clone().getParams()).toEqual(reg.getParams());
  });

  it("decides with score > 0 and exposes decisionFunction for two classes", () => {
    const m = new SGDClassifier(det).fit(Xs, ysb);
    const d = vals(m.decisionFunction(Xs));
    const pred = vals(m.predict(Xs));
    for (let i = 0; i < d.length; i++) expect(pred[i]).toBe((d[i] as number) > 0 ? 5 : 1);
    expect(m.decisionFunction(Xs).shape).toEqual([10]);
  });

  it("classWeight scales the per-sample updates", () => {
    const base = new SGDClassifier({ ...det, learningRate: "constant", eta0: 0.01 }).fit(Xs, ysb);
    const weighted = new SGDClassifier({
      ...det,
      learningRate: "constant",
      eta0: 0.01,
      classWeight: { 1: 5, 5: 1 },
    }).fit(Xs, ysb);
    expect(Array.from(weighted.coef)).not.toEqual(Array.from(base.coef));
    expect(() => new SGDClassifier({ classWeight: { 1: -1 } })).toThrow(InvalidParameterError);
  });

  it("raises a clear error on overflow instead of returning NaN weights", () => {
    const huge = f64([[1e200], [-1e200], [1e200], [-1e200]]);
    expect(() =>
      new SGDRegressor({ learningRate: "constant", eta0: 1, shuffle: false }).fit(
        huge,
        f64([1, -1, 1, -1])
      )
    ).toThrow(DataValidationError);
  });

  it("predictions are float64 and metrics validate their arguments", () => {
    const reg = new SGDRegressor({ ...det, eta0: 0.02 }).fit(Xs, ysr);
    expect(reg.predict(Xs).dtype).toBe("float64");
    expect(reg.coefTensor.dtype).toBe("float64");
    expect(() => reg.score(Xs, f64([1, 2]))).toThrow(ShapeError);
    expect(() => reg.score(Xs, f64([]))).toThrow(DataValidationError);
    const clf = new SGDClassifier(det).fit(Xs, ysb);
    expect(clf.predict(Xs).dtype).toBe("float64");
    expect(() => clf.score(Xs, f64([1, 2]))).toThrow(ShapeError);
    expect(() => new SGDClassifier().predict(Xs)).toThrow(NotFittedError);
    expect(() => new SGDRegressor().coef).toThrow(NotFittedError);
    expect(() => new SGDClassifier().interceptArray).toThrow(NotFittedError);
  });

  it("a constant target gives R2 = 1 only for exact predictions", () => {
    const reg = new SGDRegressor({ ...det, eta0: 0.02, penalty: "none" }).fit(
      Xs,
      f64(new Array(10).fill(3))
    );
    expect(reg.score(Xs, f64(new Array(10).fill(3)))).toBeLessThanOrEqual(1);
    expect(reg.score(Xs, f64(new Array(10).fill(4)))).toBe(0);
  });

  it("emits a ConvergenceWarning when the loss keeps improving until maxIter", () => {
    const warnings = catchWarnings(() => {
      new SGDRegressor({ maxIter: 2, tol: 0, shuffle: false, eta0: 0.02 }).fit(Xs, ysr);
    });
    expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Review additions
// ---------------------------------------------------------------------------
describe("c21 review additions", () => {
  const Xl = f64([
    [0.03, 4.08],
    [1.22, -1.53],
    [-0.3, -1.58],
    [0.57, -0.17],
    [0.75, -5.54],
    [1.57, -0.29],
    [0.68, -0.41],
    [-0.38, 1.39],
    [0.82, -0.61],
    [-0.15, 2.06],
    [-0.87, -4.54],
    [0.39, -2.01],
    [-1.92, -2.44],
    [-0.47, -3.58],
  ]);
  const yl = f64([2, 7, 7, 7, 7, 7, 7, 2, 7, 2, 7, 7, 2, 7]);

  it("L1 solver reaches very tight tolerances without stalling or warning", () => {
    // sklearn LogisticRegression(penalty="l1", solver="saga", C=1, tol=1e-10, max_iter=100000)
    const warnings = catchWarnings(() => {
      const m = new LogisticRegression({ penalty: "l1", solver: "saga", tol: 1e-9 }).fit(Xl, yl);
      expectClose(vals(m.coef), [1.6739698751167622, -0.8686907673775465], 1e-7);
      expect(m.intercept as number).toBeCloseTo(0.4979385955753699, 7);
      expect(m.nIter).toBeLessThan(500);
    });
    expect(warnings.length).toBe(0);
  });

  it("one-vs-rest 'balanced' balances each binary problem (scikit-learn OneVsRestClassifier)", () => {
    // scikit-learn's OneVsRestClassifier(LogisticRegression(class_weight="balanced")) fits
    // each class against the rest with balanced weights of that binary problem.
    const X3 = f64([
      [0.1, 1.2],
      [1.5, -0.3],
      [-0.7, 0.4],
      [2.2, 1.1],
      [-1.4, -0.9],
      [0.3, -1.6],
      [1.9, 0.2],
      [-0.2, 2.3],
      [0.8, 0.9],
      [-1.1, 1.4],
    ]);
    const y3 = f64([0, 1, 0, 2, 0, 1, 2, 0, 1, 0]);
    const m = new LogisticRegression({
      multiClass: "ovr",
      classWeight: "balanced",
      tol: 1e-12,
      maxIter: 5000,
    }).fit(X3, y3);
    // Each binary fit uses the balanced weights n / (2 * count) (the binary 'balanced'
    // rule is checked against scikit-learn in the class weight test above).
    const ref = [0, 1, 2].map((k) => {
      const yk = f64(vals(y3).map((v) => (v === k ? 1 : 0)));
      return new LogisticRegression({ classWeight: "balanced", tol: 1e-12, maxIter: 5000 }).fit(
        X3,
        yk
      );
    });
    const coef = vals(m.coef);
    const b = m.intercept as number[];
    for (let k = 0; k < 3; k++) {
      expectClose(coef.slice(k * 2, k * 2 + 2), vals((ref[k] as LogisticRegression).coef), 1e-8);
      expect(b[k]).toBeCloseTo((ref[k] as LogisticRegression).intercept as number, 8);
    }
  });

  it("SGD predictions do not change when fitIntercept or loss is modified after fit", () => {
    const Xs = f64([
      [-0.8, -1.3],
      [-0.2, 0.4],
      [1.1, 0.1],
      [-0.5, -0.8],
      [0.7, 1.6],
      [0.3, -1.2],
    ]);
    const ys = f64([5, 1, 5, 5, 1, 5]);
    const clf = new SGDClassifier({ loss: "log_loss", shuffle: false, maxIter: 20, tol: 0 });
    clf.fit(Xs, ys);
    const before = vals(clf.decisionFunction(Xs));
    const proba = vals(clf.predictProba(Xs));
    clf.setParams({ fitIntercept: false, loss: "hinge" });
    expectClose(vals(clf.decisionFunction(Xs)), before, 0);
    expectClose(vals(clf.predictProba(Xs)), proba, 0);

    const reg = new SGDRegressor({ shuffle: false, maxIter: 20, tol: 0 });
    reg.fit(Xs, f64([1, 2, 3, 4, 5, 6]));
    const predBefore = vals(reg.predict(Xs));
    const interceptBefore = reg.intercept;
    reg.setParams({ fitIntercept: false });
    expectClose(vals(reg.predict(Xs)), predBefore, 0);
    expect(reg.intercept).toBe(interceptBefore);
  });

  it("partialFit rejects 'balanced' before touching the model", () => {
    const clf = new SGDClassifier({ classWeight: "balanced" });
    expect(() => clf.partialFit(f64([[1], [2]]), f64([0, 1]))).toThrow(InvalidParameterError);
    expect(clf.classes).toBeUndefined();
    expect(() => clf.decisionFunction(f64([[1]]))).toThrow(NotFittedError);
  });

  it("SGD class weights multiply the clipped gradient, like scikit-learn", () => {
    // Unscaled data with the 'optimal' schedule drives the weights to ~1e13, where the
    // gradient clip at 1e12 is active. scikit-learn clips first and then applies the class
    // weight; reference from SGDClassifier(loss="squared_hinge", shuffle=False, tol=None,
    // max_iter=3, class_weight={1: 2, 3: 3}).
    const Xb = f64([
      [1, 2],
      [2, 1],
      [3, 5],
      [4, 3],
      [5, 8],
      [6, 2],
    ]);
    const yb = f64([1, 3, 1, 3, 1, 3]);
    const m = new SGDClassifier({
      loss: "squared_hinge",
      shuffle: false,
      maxIter: 3,
      tol: 0,
      classWeight: { 1: 2, 3: 3 },
    }).fit(Xb, yb);
    expect(m.nIter).toBe(3);
    const expected = [91949659886856.0, -77101597391288.11];
    for (let j = 0; j < 2; j++) {
      expect(Math.abs((m.coef[j] as number) - (expected[j] as number))).toBeLessThan(
        1e-6 * Math.abs(expected[j] as number)
      );
    }
    expect(Math.abs(m.intercept - 12624150780428.559)).toBeLessThan(1e-6 * 12624150780428.559);
  });

  it("Ridge keeps nIter of the previous fit when a refit fails", () => {
    const X = f64([[1], [2], [3], [4]]);
    const y = f64([1, 2, 3, 5]);
    const model = new Ridge({ solver: "lsqr", tol: 1e-12 }).fit(X, y);
    const iters = model.nIter;
    expect(iters).toBeGreaterThan(0);
    model.setParams({ alpha: -1 });
    expect(() => model.fit(X, y)).toThrow(InvalidParameterError);
    expect(model.nIter).toBe(iters);
  });
});
