import { describe, expect, it } from "vitest";
import {
  catchWarnings,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  BayesianRidge,
  ElasticNet,
  HuberRegressor,
  IsotonicRegression,
  KernelRidge,
  Lasso,
  LinearRegression,
} from "../../src/ml";
import { type Tensor, tensor } from "../../src/ndarray";

const f64 = (rows: number | number[] | number[][]): Tensor => tensor(rows, { dtype: "float64" });
const arr = (t: Tensor): number[] =>
  Array.from(t.toArray() as ArrayLike<number>).flat() as number[];
const expectClose = (actual: ArrayLike<number>, expected: number[], digits: number): void => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, digits);
  }
};

// Reference data (numpy RandomState(0), rounded to 3 decimals); every expected value below was
// computed with scikit-learn 1.8 / scipy 1.17 on exactly these numbers.
const XA = [
  [1.764, 0.4, 1.607],
  [2.241, 1.868, 1.597],
  [0.95, -0.151, 0.739],
  [0.411, 0.144, 0.62],
  [0.761, 0.122, 0.698],
  [0.334, 1.494, 0.226],
  [0.313, -0.854, -0.26],
  [0.654, 0.864, 0.375],
  [2.27, -1.454, 1.825],
  [-0.187, 1.533, 0.144],
  [0.155, 0.378, -0.054],
  [-1.981, -0.348, -1.554],
];
const YA = [8.301, 7.773, 6.304, 4.897, 5.434, 2.861, 4.838, 5.217, 10.754, 2.034, 3.529, -0.158];
const YO = [
  33.301, 7.773, 6.304, 4.897, 5.434, -15.139, 4.838, 5.217, 10.754, 2.034, 3.529, -0.158,
];
const X = f64(XA);
const y = f64(YA);

describe("c20 LinearRegression", () => {
  it("returns float64 predictions with full precision", () => {
    const model = new LinearRegression().fit(X, y);
    expectClose(arr(model.coef), [1.854073720450394, -0.9110088441708383, 0.7420854128595116], 10);
    expect(Number(model.intercept?.data[0])).toBeCloseTo(3.8958982902303436, 10);
    const pred = model.predict(X);
    expect(pred.dtype).toBe("float64");
    expect(pred.shape).toEqual([12]);
    // The old float32 output differed from float64 in the 7th digit.
    expect(arr(pred)[0]).toBeCloseTo(7.994612053901737, 10);
  });

  it("is accurate when the features have a large offset", () => {
    const shifted = f64(XA.map((row) => row.map((v) => v + 1e6)));
    const model = new LinearRegression().fit(shifted, y);
    expectClose(arr(model.coef), [1.8540737203096886, -0.9110088441868124, 0.7420854130232463], 6);
    expect(Number(model.intercept?.data[0])).toBeCloseTo(-1685146.3932478318, 1);
  });

  it("normalizes columns also when fitIntercept is false", () => {
    const plain = new LinearRegression({ fitIntercept: false }).fit(X, y);
    const scaled = new LinearRegression({ fitIntercept: false, normalize: true }).fit(X, y);
    expectClose(arr(plain.coef), [4.137477828348354, 0.08461358868918181, -0.4038532065931079], 9);
    // Least squares is scale equivariant, so both give the same coefficients in original units.
    expectClose(arr(scaled.coef), arr(plain.coef), 9);
    expect(scaled.intercept).toBeUndefined();
  });

  it("clears the intercept when refitted with fitIntercept false", () => {
    const model = new LinearRegression().fit(X, y);
    expect(model.intercept).toBeDefined();
    model.setParams({ fitIntercept: false });
    model.fit(X, y);
    expect(model.intercept).toBeUndefined();
    // Prediction must not reuse the previous intercept.
    const expected = XA.map(
      (r) => r[0] * 4.137477828348354 + r[1] * 0.08461358868918181 + r[2] * -0.4038532065931079
    );
    expectClose(arr(model.predict(X)), expected, 9);
  });

  it("exposes rank and singular values and keeps the minimum-norm solution for wide data", () => {
    const wide = f64([
      [1, 2, 3, 4],
      [2, 1, 0, 1],
    ]);
    const model = new LinearRegression().fit(wide, f64([1, 2]));
    expect(model.rank).toBe(1);
    expect(model.singular.length).toBe(2);
    expectClose(arr(model.predict(wide)), [1, 2], 10);
    expect(model.nFeaturesIn).toBe(4);
  });

  it("scores an int64 target and a constant target like sklearn's r2_score", () => {
    const Xi = f64([[1], [2], [3], [4]]);
    const model = new LinearRegression().fit(Xi, f64([2, 4, 6, 8]));
    const yInt = tensor([2, 4, 6, 8], { dtype: "int64" });
    expect(model.score(Xi, yInt)).toBeCloseTo(1, 12);
    const constant = new LinearRegression().fit(Xi, f64([5, 5, 5, 5]));
    expect(constant.score(Xi, f64([5, 5, 5, 5]))).toBe(1);
    expect(constant.score(Xi, f64([5, 5, 5, 6]))).toBeLessThan(1);
    expect(constant.score(f64([[1], [2]]), f64([6, 6]))).toBe(0);
  });

  it("validates score inputs", () => {
    const model = new LinearRegression().fit(X, y);
    expect(() => model.score(X, f64(YA.slice(0, 5)))).toThrow(ShapeError);
    expect(() => model.score(X, f64([YA]))).toThrow(ShapeError);
    expect(() => model.score(X, f64(YA.map((v, i) => (i === 3 ? Number.NaN : v))))).toThrow(
      DataValidationError
    );
  });

  it("copyX false overwrites a float32 input with the centered values", () => {
    const Xf = tensor(
      [
        [1, 10],
        [2, 20],
        [3, 30],
      ],
      { dtype: "float32" }
    );
    const model = new LinearRegression({ copyX: false }).fit(Xf, tensor([1, 2, 3]));
    expect(arr(Xf)).toEqual([-1, -10, 0, 0, 1, 10]);
    expect(model.coef.size).toBe(2);
  });

  it("setParams is atomic and clone gives an unfitted copy", () => {
    const model = new LinearRegression();
    expect(() => model.setParams({ normalize: true, copyX: "no" })).toThrow(InvalidParameterError);
    expect(model.getParams().normalize).toBe(false);
    model.fit(X, y);
    const copy = model.clone();
    expect(() => copy.predict(X)).toThrow(NotFittedError);
  });
});

describe("c20 Lasso", () => {
  const exact = { tol: 1e-10, maxIter: 100000 };

  it("matches scikit-learn", () => {
    const model = new Lasso({ alpha: 0.1, ...exact }).fit(X, y);
    expectClose(arr(model.coef), [2.2075712207770897, -0.7750349226943789, 0.18469905051696422], 8);
    expect(model.intercept).toBeCloseTo(3.9012078567651782, 8);
    expectClose(arr(model.predict(X)).slice(0, 2), [7.782160895318974, 7.695574110609129], 8);
    expect(model.predict(X).dtype).toBe("float64");
  });

  it("reproduces scikit-learn's iteration count at the default tolerance", () => {
    const model = new Lasso({ alpha: 0.1 }).fit(X, y);
    expectClose(arr(model.coef), [2.2105560053938538, -0.7748803523110249, 0.1811160925164259], 10);
    expect(model.nIter).toBe(65);
    expect(model.dualGap).toBeLessThan(1e-4 * 12 * 9);
  });

  it("supports positive coefficients", () => {
    const model = new Lasso({ alpha: 0.05, positive: true, ...exact }).fit(X, y);
    expectClose(arr(model.coef), [2.377971396346237, 0, 0], 8);
    expect(model.intercept).toBeCloseTo(3.6257741515899298, 8);
  });

  it("normalize matches fitting on unit-norm centered columns", () => {
    const model = new Lasso({ alpha: 0.05, normalize: true, ...exact, tol: 1e-12 }).fit(X, y);
    expectClose(arr(model.coef), [1.8905100351932007, -0.7139028778265473, 0.49116663513824654], 8);
    expect(model.intercept).toBeCloseTo(3.9316133028338145, 8);
  });

  it("does not stop early on a target with a tiny scale", () => {
    // The old criterion compared absolute coefficient changes with tol and quit after one pass.
    const tiny = f64(YA.map((v) => v * 1e-6));
    const model = new Lasso({ alpha: 1e-8, tol: 1e-4 }).fit(X, tiny);
    expectClose(
      arr(model.coef).map((v) => v * 1e6),
      [1.891874116293, -0.897284542611, 0.683405002883],
      6
    );
    expect(model.nIter).toBe(92);
  });

  it("random selection honors randomState, also for negative and huge seeds", () => {
    const opts = { alpha: 0.05, selection: "random" as const, ...exact, tol: 1e-12 };
    for (const randomState of [-1e6, 7, 3.5, 2 ** 40]) {
      const a = new Lasso({ ...opts, randomState }).fit(X, y);
      const b = new Lasso({ ...opts, randomState }).fit(X, y);
      expect(arr(a.coef)).toEqual(arr(b.coef));
      // Any visiting order has to reach the unique minimizer.
      expectClose(arr(a.coef), [2.030822469187888, -0.8430218835064484, 0.4633922333998436], 6);
    }
  });

  it("warm start with normalize starts from the previous solution", () => {
    const model = new Lasso({ alpha: 0.05, normalize: true, warmStart: true, tol: 1e-10 });
    model.fit(X, y);
    expect(model.nIter as number).toBeGreaterThan(50);
    model.fit(X, y);
    expect(model.nIter).toBeLessThanOrEqual(2);
  });

  it("warns when maxIter is exhausted", () => {
    const warnings = catchWarnings(() => {
      new Lasso({ alpha: 0.001, maxIter: 1, tol: 1e-12 }).fit(X, y);
    });
    expect(warnings.length).toBe(1);
    expect(warnings[0]?.category).toBe("ConvergenceWarning");
  });

  it("validates hyper-parameters at fit and in setParams", () => {
    expect(() => new Lasso({ maxIter: 0 }).fit(X, y)).toThrow(InvalidParameterError);
    expect(() => new Lasso({ maxIter: 2.5 }).fit(X, y)).toThrow(InvalidParameterError);
    expect(() => new Lasso({ tol: Number.NaN }).fit(X, y)).toThrow(InvalidParameterError);
    expect(() => new Lasso({ alpha: Number.NaN }).fit(X, y)).toThrow(InvalidParameterError);
    expect(() => new Lasso({ selection: "bogus" as "cyclic" }).fit(X, y)).toThrow(
      InvalidParameterError
    );
    const model = new Lasso();
    expect(() => model.setParams({ alpha: -1 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ maxIter: -3 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ l1Ratio: 0.5 })).toThrow(/Unknown parameter/);
    // A failing update leaves every parameter untouched.
    expect(() => model.setParams({ alpha: 0.5, tol: "x" })).toThrow(InvalidParameterError);
    expect(model.getParams().alpha).toBe(1.0);
  });

  it("keeps the previous fit intact when a refit is rejected", () => {
    const model = new Lasso({ alpha: 0.1 }).fit(X, y);
    const before = arr(model.coef);
    const n = model.nIter;
    model.setParams({ alpha: 0.2 });
    expect(() => model.fit(X, f64(YA.slice(0, 3)))).toThrow(ShapeError);
    expect(arr(model.coef)).toEqual(before);
    expect(model.nIter).toBe(n);
  });

  it("getParams reports defaults and round-trips through the constructor", () => {
    const params = new Lasso({ alpha: 0.3, randomState: 4 }).getParams();
    expect(params).toEqual({
      alpha: 0.3,
      fitIntercept: true,
      normalize: false,
      maxIter: 1000,
      tol: 1e-4,
      warmStart: false,
      positive: false,
      selection: "cyclic",
      randomState: 4,
    });
    const clone = new Lasso({ alpha: 0.3 }).clone();
    expect(clone.getParams().alpha).toBe(0.3);
  });

  it("handles a single sample and constant columns", () => {
    const one = new Lasso({ alpha: 0.1 }).fit(f64([[1, 2]]), f64([3]));
    expect(arr(one.coef)).toEqual([0, 0]);
    expect(one.intercept).toBe(3);
    const constant = new Lasso({ alpha: 0.01 }).fit(
      f64([
        [1, 5],
        [2, 5],
        [3, 5],
      ]),
      f64([1, 2, 3])
    );
    expect(arr(constant.coef)[1]).toBe(0);
  });
});

describe("c20 ElasticNet", () => {
  const exact = { tol: 1e-10, maxIter: 100000 };

  it("matches scikit-learn", () => {
    const model = new ElasticNet({ alpha: 0.2, l1Ratio: 0.3, ...exact }).fit(X, y);
    expectClose(arr(model.coef), [1.4703544183265744, -0.7328265429320154, 0.9482673659461155], 8);
    expect(model.intercept).toBeCloseTo(3.979848571464993, 8);
    expectClose(arr(model.predict(X)).slice(0, 2), [7.804288805295672, 7.420375824153789], 8);
  });

  it("reproduces scikit-learn's iteration count at the default tolerance", () => {
    const model = new ElasticNet({ alpha: 0.2, l1Ratio: 0.3 }).fit(X, y);
    expectClose(arr(model.coef), [1.4706954058285935, -0.732808563109252, 0.9479188997779483], 10);
    expect(model.nIter).toBe(24);
  });

  it("works without an intercept", () => {
    const model = new ElasticNet({
      alpha: 0.2,
      l1Ratio: 0.7,
      fitIntercept: false,
      ...exact,
    }).fit(X, y);
    expectClose(arr(model.coef), [2.9608190651294746, 0, 0.8646896583316713], 8);
    expect(model.intercept).toBe(0);
  });

  it("normalize matches fitting on unit-norm centered columns", () => {
    const model = new ElasticNet({
      alpha: 0.05,
      l1Ratio: 0.4,
      normalize: true,
      tol: 1e-12,
      maxIter: 100000,
    }).fit(X, y);
    expectClose(arr(model.coef), [1.068049390071117, -0.6196405316675102, 1.1654037787497076], 8);
    expect(model.intercept).toBeCloseTo(4.0919017724301945, 8);
  });

  it("l1Ratio is validated, including NaN", () => {
    expect(() => new ElasticNet({ l1Ratio: Number.NaN }).fit(X, y)).toThrow(/l1Ratio/);
    expect(() => new ElasticNet().setParams({ l1Ratio: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new ElasticNet({ maxIter: -1 }).fit(X, y)).toThrow(InvalidParameterError);
  });

  it("setParams accepts undefined randomState to drop the seed", () => {
    const model = new ElasticNet({ randomState: 3 });
    model.setParams({ randomState: undefined });
    expect("randomState" in model.getParams()).toBe(false);
  });

  it("round-trips getParams through clone", () => {
    const model = new ElasticNet({ alpha: 0.4, l1Ratio: 0.2, positive: true });
    const clone = model.clone();
    expect(clone.getParams()).toEqual(model.getParams());
    expect(() => clone.coef).toThrow(NotFittedError);
  });
});

describe("c20 BayesianRidge", () => {
  it("matches scikit-learn (coefficients, precisions, iterations)", () => {
    const model = new BayesianRidge().fit(X, y);
    expectClose(model.coef, [1.7827145095709873, -0.9072699850466494, 0.8204561390639016], 9);
    expect(model.intercept).toBeCloseTo(3.901409158079611, 9);
    expect(model.alpha).toBeCloseTo(7.547913354502954, 7);
    expect(model.lambda).toBeCloseTo(0.5987208466041194, 8);
    expect(model.nIter).toBe(5);
  });

  it("honors the hyper-prior and initial value options", () => {
    const model = new BayesianRidge({
      alpha1: 1e-3,
      lambda2: 1e-2,
      maxIter: 5,
      tol: 1e-12,
      alphaInit: 0.5,
      lambdaInit: 2,
    }).fit(X, y);
    expectClose(model.coef, [1.7829664274988517, -0.9072885642128545, 0.8201871751623091], 9);
    expect(model.alpha).toBeCloseTo(7.549217846921987, 7);
    expect(model.lambda).toBeCloseTo(0.5962572840060761, 8);
    expect(model.nIter).toBe(5);
  });

  it("matches scikit-learn without an intercept", () => {
    const model = new BayesianRidge({ fitIntercept: false }).fit(X, y);
    expectClose(model.coef, [2.308956128142714, 0.08414826986120731, 1.4652191598911266], 9);
    expect(model.intercept).toBe(0);
    expect(model.nIter).toBe(4);
  });

  it("handles more features than samples", () => {
    const model = new BayesianRidge().fit(f64(XA.slice(0, 4)), f64(YA.slice(0, 4)));
    expectClose(model.coef, [1.830822659754658, -0.9464946533980257, 1.184595131747735], 7);
    expect(model.intercept).toBeCloseTo(3.546380608231132, 7);
    expect(model.lambda).toBeCloseTo(0.5308725201052701, 7);
  });

  it("exposes the posterior covariance and log marginal likelihood", () => {
    const model = new BayesianRidge({ computeScore: true }).fit(X, y);
    const sigma = model.sigma;
    expect(sigma.shape).toEqual([3, 3]);
    expectClose(
      arr(sigma),
      [
        0.12833766653262216, 0.005937881987427967, -0.1527478053369206, 0.005937881987427967,
        0.012490435477416347, -0.007917038718289759, -0.15274780533692062, -0.007917038718289759,
        0.19552501344328738,
      ],
      9
    );
    // scikit-learn evaluates the last score with the coefficients of the previous iteration,
    // so only the value up to the convergence tolerance is comparable.
    expect(model.scores.length).toBe(model.nIter + 1);
    expect(model.scores[model.scores.length - 1]).toBeCloseTo(-11.044598749701292, 4);
    expect(new BayesianRidge().fit(X, y).scores.length).toBe(0);
  });

  it("predictWithStd uses the centered predictive variance", () => {
    const model = new BayesianRidge().fit(X, y);
    const { mean, std } = model.predictWithStd(f64(XA.slice(0, 2)));
    expectClose(arr(mean), [8.00168257, 7.5119605], 7);
    expectClose(arr(std), [0.3926634483319979, 0.4380102366999947], 9);
    expect(mean.dtype).toBe("float64");
  });

  it("setParams validates and changes the options", () => {
    const model = new BayesianRidge();
    model.setParams({ maxIter: 3, alpha1: 0.5 });
    expect(model.getParams().maxIter).toBe(3);
    expect(model.getParams().alpha1).toBe(0.5);
    expect(() => model.setParams({ maxIter: 0 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ nope: 1 })).toThrow(/Unknown parameter/);
    expect(() => model.setParams({ fitIntercept: 1 })).toThrow(InvalidParameterError);
    expect(model.getParams().maxIter).toBe(3);
    model.fit(X, y);
    expect(model.nIter).toBeLessThanOrEqual(3);
  });

  it("round-trips getParams through the constructor and clone", () => {
    const model = new BayesianRidge({ alphaInit: 2, computeScore: true });
    const clone = model.clone();
    expect(clone.getParams()).toEqual(model.getParams());
    expect("alphaInit" in new BayesianRidge().getParams()).toBe(false);
  });

  it("validates constructor options", () => {
    expect(() => new BayesianRidge({ tol: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new BayesianRidge({ alpha2: -1 })).toThrow(InvalidParameterError);
    expect(() => new BayesianRidge({ alphaInit: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("scores a perfectly predicted constant target as 1", () => {
    const Xc = f64([[1], [2], [3]]);
    const model = new BayesianRidge().fit(Xc, f64([4, 4, 4]));
    expect(model.score(Xc, f64([4, 4, 4]))).toBe(1);
    expect(model.predict(Xc).dtype).toBe("float64");
  });

  it("copes with a single sample", () => {
    const model = new BayesianRidge().fit(f64([[1]]), f64([2]));
    expect(model.predict(f64([[5]])).toArray()).toEqual([2]);
  });
});

describe("c20 HuberRegressor", () => {
  it("matches scikit-learn on clean data", () => {
    const model = new HuberRegressor().fit(X, y);
    expectClose(model.coef, [1.935992047198306, -0.8753273815024956, 0.7836405687550237], 3);
    expect(model.intercept).toBeCloseTo(3.7203420688237174, 3);
    expect(model.scale).toBeCloseTo(0.11662036482226001, 3);
    expect(model.outliers.map((o) => (o ? 1 : 0)).join("")).toBe("100011010001");
  });

  it("is not moved by gross outliers (sklearn parity)", () => {
    const model = new HuberRegressor().fit(X, f64(YO));
    expectClose(model.coef, [1.9360159906921903, -0.8753220032084772, 0.7836141633971934], 3);
    expect(model.intercept).toBeCloseTo(3.7203458327358176, 3);
    expect(model.outliers[0]).toBe(true);
    expect(model.outliers[5]).toBe(true);
  });

  it("applies the L2 penalty like sklearn", () => {
    const model = new HuberRegressor({ alpha: 0.5 }).fit(X, y);
    expectClose(model.coef, [1.815315311754907, -0.8854484632266008, 0.8822055006164411], 3);
    expect(model.intercept).toBeCloseTo(3.7743870701270197, 3);
    expect(model.scale).toBeCloseTo(0.1712965598065045, 3);
  });

  it("works without an intercept", () => {
    const model = new HuberRegressor({ fitIntercept: false }).fit(X, f64(YO));
    expectClose(model.coef, [2.819755144480061, -0.7648000438372923, 2.4092314372613215], 3);
    expect(model.intercept).toBe(0);
    expect(model.scale).toBeCloseTo(3.098636890146594, 3);
  });

  it("recovers the line when one target is far off, whatever the units of y", () => {
    const Xo = f64([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const base = [3.1, 4.8, 7.15, 8.95, 11.2, 12.9, 15.05, 16.85, 19.0, 321.1];
    for (const unit of [1, 1e-3, 1e6]) {
      const model = new HuberRegressor({ alpha: 0, tol: 1e-9, maxIter: 1000 }).fit(
        Xo,
        f64(base.map((v) => v * unit))
      );
      expect((model.coef[0] as number) / unit).toBeCloseTo(2.00656, 3);
      expect(model.intercept / unit).toBeCloseTo(0.99073, 3);
      expect(model.scale / unit).toBeCloseTo(0.13714, 3);
      expect(model.outliers.map((o) => (o ? 1 : 0)).join("")).toBe("0100000101");
    }
  });

  it("leaves a high-leverage outlier as influential as scikit-learn does", () => {
    // Same fit as sklearn: slope 5.128, intercept -14.33, scale 6.095.
    const model = new HuberRegressor().fit(
      f64([[1], [2], [3], [4], [5], [6], [7], [8], [9], [100]]),
      f64([3, 5, 7, 9, 11, 13, 15, 17, 19, 500])
    );
    expect(model.coef[0]).toBeCloseTo(5.12788975, 2);
    expect(model.intercept).toBeCloseTo(-14.330844, 2);
    expect(model.scale).toBeCloseTo(6.0954626, 2);
  });

  it("gives the same fit when X and y carry a large offset", () => {
    const shifted = new HuberRegressor().fit(
      f64(XA.map((row) => [row[0] as number, (row[1] as number) + 1e6, row[2] as number])),
      f64(YO.map((v) => v + 1e8))
    );
    const plain = new HuberRegressor().fit(X, f64(YO));
    expectClose(shifted.coef, Array.from(plain.coef), 6);
    expect(shifted.scale).toBeCloseTo(plain.scale, 6);
    expect(shifted.outliers).toEqual(plain.outliers);
    // intercept_shifted = intercept_plain + 1e8 - 1e6 * coef[1]
    expect(shifted.intercept).toBeCloseTo(
      plain.intercept + 1e8 - 1e6 * (plain.coef[1] as number),
      1
    );
  });

  it("fits the documented example", () => {
    const model = new HuberRegressor({ epsilon: 1.35 }).fit(
      f64([[1], [2], [3], [4], [100]]),
      f64([2, 4, 6, 8, 200])
    );
    expect(model.coef[0]).toBeCloseTo(2, 8);
    expect(model.intercept).toBeCloseTo(0, 8);
  });

  it("still accepts the deprecated learningRate option", () => {
    const model = new HuberRegressor({ learningRate: 0.5 }).fit(X, y);
    expect(model.getParams().learningRate).toBe(0.5);
    expect(model.coef.length).toBe(3);
  });

  it("setParams validates and changes the options", () => {
    const model = new HuberRegressor();
    model.setParams({ epsilon: 2, alpha: 0.1 });
    expect(model.getParams().epsilon).toBe(2);
    expect(() => model.setParams({ epsilon: 1 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ maxIter: 1.5 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ fitIntercept: "yes" })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ nope: 1 })).toThrow(/Unknown parameter/);
    expect(model.getParams().epsilon).toBe(2);
    const first = new HuberRegressor({ epsilon: 1.35 }).fit(X, y).coef[0] as number;
    const second = model.setParams({ epsilon: 1.35, alpha: 1e-4 }).fit(X, y).coef[0] as number;
    expect(second).toBeCloseTo(first, 10);
  });

  it("rejects NaN options and scores a constant target", () => {
    expect(() => new HuberRegressor({ epsilon: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new HuberRegressor({ tol: Number.NaN })).toThrow(InvalidParameterError);
    const Xc = f64([[1], [2], [3]]);
    const model = new HuberRegressor().fit(Xc, f64([4, 4, 4]));
    expect(model.score(Xc, f64([4, 4, 4]))).toBeGreaterThanOrEqual(0);
    expect(model.predict(Xc).dtype).toBe("float64");
    const clone = model.clone();
    expect(() => clone.predict(Xc)).toThrow(NotFittedError);
  });

  it("warns when maxIter is exhausted", () => {
    const warnings = catchWarnings(() => {
      new HuberRegressor({ epsilon: 1.1, maxIter: 1, tol: 1e-12 }).fit(X, y);
    });
    expect(warnings[0]?.category).toBe("ConvergenceWarning");
  });
});

describe("c20 IsotonicRegression", () => {
  const XI = f64([1, 2, 2, 3, 4, 5, 6, 7, 8]);
  const YI = f64([1, 3, 0.5, 2, 2, 5, 4, 3, 7]);

  it("averages samples that share the same X (sklearn parity)", () => {
    const model = new IsotonicRegression().fit(XI, YI);
    expect(Array.from(model.xThresholds)).toEqual([1, 2, 3, 4, 5, 7, 8]);
    expect(Array.from(model.yThresholds)).toEqual([1, 1.75, 2, 2, 4, 4, 7]);
    expectClose(arr(model.predict(f64([1.5, 2, 2.5]))), [1.375, 1.75, 1.875], 12);
    expect(model.score(XI, YI)).toBeCloseTo(0.8457357859531772, 12);
  });

  it("does not split tied X values whose y values arrive in increasing order", () => {
    // Old behavior kept y=1 for x=1 and dropped the larger tied value.
    const model = new IsotonicRegression().fit(f64([1, 1, 2]), f64([1, 3, 5]));
    expect(arr(model.predict(f64([1])))).toEqual([2]);
  });

  it("returns the fitted value, not NaN, at the ends of the training range with outOfBounds nan", () => {
    const model = new IsotonicRegression({ outOfBounds: "nan" }).fit(XI, YI);
    const pred = arr(model.predict(f64([0, 1, 1.5, 8, 9])));
    expect(pred[0]).toBeNaN();
    expect(pred[1]).toBe(1);
    expect(pred[2]).toBe(1.375);
    expect(pred[3]).toBe(7);
    expect(pred[4]).toBeNaN();
  });

  it("clips by default and raises with outOfBounds raise", () => {
    const clip = new IsotonicRegression().fit(XI, YI);
    expect(arr(clip.predict(f64([0, 9])))).toEqual([1, 7]);
    const strict = new IsotonicRegression({ outOfBounds: "raise" }).fit(XI, YI);
    expect(arr(strict.predict(f64([1, 8])))).toEqual([1, 7]);
    expect(() => strict.predict(f64([9]))).toThrow(DataValidationError);
  });

  it("supports sample weights and decreasing fits", () => {
    const model = new IsotonicRegression({ increasing: false }).fit(
      XI,
      YI,
      f64([1, 2, 1, 3, 1, 1, 0, 2, 1])
    );
    expect(Array.from(model.xThresholds)).toEqual([1, 8]);
    expect(model.yThresholds[0]).toBeCloseTo(2.7916666666666665, 12);
    expect(model.increasing).toBe(false);
  });

  it("validates sample weights", () => {
    const model = new IsotonicRegression();
    expect(() => model.fit(XI, YI, f64([1, 2]))).toThrow(ShapeError);
    expect(() => model.fit(XI, YI, f64([-1, 1, 1, 1, 1, 1, 1, 1, 1]))).toThrow(DataValidationError);
    expect(() => model.fit(XI, YI, f64([0, 0, 0, 0, 0, 0, 0, 0, 0]))).toThrow(DataValidationError);
  });

  it("picks the direction automatically from the Spearman correlation", () => {
    const decreasing = new IsotonicRegression({ increasing: "auto" }).fit(
      XI,
      f64([-1, -3, -0.5, -2, -2, -5, -4, -3, -7])
    );
    expect(decreasing.increasing).toBe(false);
    expect(Array.from(decreasing.yThresholds)).toEqual([-1, -1.75, -2, -2, -4, -4, -7]);
    expect(new IsotonicRegression({ increasing: "auto" }).fit(XI, YI).increasing).toBe(true);
  });

  it("clips the fitted values with yMin and yMax", () => {
    const model = new IsotonicRegression({ yMin: 1.5, yMax: 4 }).fit(XI, YI);
    expect(Array.from(model.xThresholds)).toEqual([1, 2, 3, 4, 5, 8]);
    expect(Array.from(model.yThresholds)).toEqual([1.5, 1.75, 2, 2, 4, 4]);
  });

  it("handles degenerate inputs", () => {
    const same = new IsotonicRegression().fit(f64([1, 1, 1]), f64([1, 2, 3]));
    expect(Array.from(same.yThresholds)).toEqual([2]);
    expect(arr(same.predict(f64([1, 5])))).toEqual([2, 2]);
    const flat = new IsotonicRegression().fit(f64([1, 2, 3]), f64([2, 2, 2]));
    expect(Array.from(flat.xThresholds)).toEqual([1, 3]);
    expect(arr(flat.predict(f64([0, 2, 5])))).toEqual([2, 2, 2]);
    expect(flat.score(f64([1, 2, 3]), f64([2, 2, 2]))).toBe(1);
  });

  it("rejects bad shapes and values with clear errors", () => {
    const model = new IsotonicRegression();
    expect(() => model.fit(f64(3), f64([1, 2, 3]))).toThrow(ShapeError);
    expect(() => model.fit(tensor(["a", "b"]), f64([1, 2]))).toThrow();
    expect(() => model.fit(f64([[1, 2]]), f64([1]))).toThrow(ShapeError);
    expect(() => model.fit(f64([1, 2]), f64([1]))).toThrow(ShapeError);
    expect(() => model.fit(f64([1, Number.NaN]), f64([1, 2]))).toThrow(DataValidationError);
    expect(() => model.predict(f64([1]))).toThrow(/must be fitted/);
  });

  it("validates options in the constructor and setParams", () => {
    expect(() => new IsotonicRegression({ outOfBounds: "x" as "nan" })).toThrow(
      InvalidParameterError
    );
    expect(() => new IsotonicRegression({ increasing: "up" as "auto" })).toThrow(
      InvalidParameterError
    );
    expect(() => new IsotonicRegression({ yMin: 3, yMax: 1 })).toThrow(InvalidParameterError);
    const model = new IsotonicRegression();
    expect(() => model.setParams({ outOfBounds: "x" })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ nope: 1 })).toThrow(/Unknown parameter/);
    model.setParams({ increasing: false, yMax: 10 });
    expect(model.getParams()).toEqual({ increasing: false, outOfBounds: "clip", yMax: 10 });
    expect(model.clone().getParams()).toEqual(model.getParams());
  });

  it("accepts single-column y in fit and score", () => {
    const model = new IsotonicRegression().fit(f64([[1], [2], [3]]), f64([[1], [2], [3]]));
    expect(model.score(f64([[1], [2], [3]]), f64([[1], [2], [3]]))).toBeCloseTo(1, 12);
  });
});

describe("c20 KernelRidge", () => {
  const XK = f64([
    [1.764, 0.4],
    [2.241, 1.868],
    [0.95, -0.151],
    [0.411, 0.144],
    [0.761, 0.122],
    [0.334, 1.494],
    [0.313, -0.854],
    [0.654, 0.864],
  ]);
  const yK = f64([8.301, 7.773, 6.304, 4.897, 5.434, 2.861, 4.838, 5.217]);
  const T = f64([
    [0.5, 0.5],
    [1.5, -0.2],
  ]);

  it("matches scikit-learn for the rbf kernel", () => {
    const model = new KernelRidge({ kernel: "rbf", gamma: 0.5, alpha: 0.1 }).fit(XK, yK);
    expectClose(
      model.dualCoef.slice(0, 3),
      [5.224516108384989, 5.440838935443509, 2.2336187161147487],
      9
    );
    expectClose(arr(model.predict(T)), [4.815608006440724, 6.384002974917107], 9);
    expect(model.predict(T).dtype).toBe("float64");
  });

  it("matches scikit-learn for the linear and polynomial kernels", () => {
    const lin = new KernelRidge({ kernel: "linear", alpha: 0.5 }).fit(XK, yK);
    expectClose(arr(lin.predict(T)), [2.1370951773516342, 7.517609131239951], 8);
    const poly = new KernelRidge({
      kernel: "poly",
      degree: 2,
      gamma: 0.3,
      coef0: 1,
      alpha: 0.2,
    }).fit(XK, yK);
    expectClose(arr(poly.predict(T)), [4.60869118828508, 7.686209888884798], 8);
  });

  it("supports the sigmoid and laplacian kernels", () => {
    const sig = new KernelRidge({ kernel: "sigmoid", gamma: 0.1, coef0: 0.5, alpha: 1 }).fit(
      XK,
      yK
    );
    expectClose(arr(sig.predict(T)), [4.379212860523005, 5.108065498381444], 9);
    const lap = new KernelRidge({ kernel: "laplacian", gamma: 0.4, alpha: 0.3 }).fit(XK, yK);
    expectClose(arr(lap.predict(T)), [5.023682238360692, 6.053464424127033], 9);
  });

  it("rejects unknown kernels and invalid numbers instead of returning zeros", () => {
    expect(() => new KernelRidge({ kernel: "bogus" as "rbf" })).toThrow(InvalidParameterError);
    expect(() => new KernelRidge({ gamma: -1 })).toThrow(InvalidParameterError);
    expect(() => new KernelRidge({ degree: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new KernelRidge({ alpha: Number.NaN })).toThrow(InvalidParameterError);
    const model = new KernelRidge();
    expect(() => model.setParams({ kernel: "bogus" })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ alpha: -1 })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ gamma: "x" })).toThrow(InvalidParameterError);
    expect(() => model.setParams({ nope: 1 })).toThrow(/Unknown parameter/);
    expect(model.getParams().kernel).toBe("rbf");
  });

  it("predicts with the parameters it was fitted with", () => {
    const model = new KernelRidge({ kernel: "rbf", gamma: 0.5, alpha: 0.1 }).fit(XK, yK);
    const before = arr(model.predict(T));
    model.setParams({ kernel: "linear", gamma: 5 });
    expect(arr(model.predict(T))).toEqual(before);
  });

  it("falls back to the minimum-norm solution for a singular system", () => {
    const dup = f64([[1], [1], [2]]);
    let model: KernelRidge | undefined;
    const warnings = catchWarnings(() => {
      model = new KernelRidge({ kernel: "linear", alpha: 0 }).fit(dup, f64([1, 1, 2]));
    });
    expect(warnings.length).toBe(1);
    expectClose(arr((model as KernelRidge).predict(dup)), [1, 1, 2], 8);
  });

  it("setParams ignores undefined values", () => {
    const model = new KernelRidge({ kernel: "linear", alpha: 0.5 });
    model.setParams({ alpha: undefined, gamma: 2 });
    expect(model.getParams()).toMatchObject({ kernel: "linear", alpha: 0.5, gamma: 2 });
  });

  it("keeps the previous model when a refit fails", () => {
    const model = new KernelRidge({ kernel: "linear", alpha: 0.5 }).fit(XK, yK);
    const before = arr(model.predict(T));
    expect(() => model.fit(XK, f64([1, 2, 3]))).toThrow(ShapeError);
    expect(arr(model.predict(T))).toEqual(before);
  });

  it("reports the feature count and validates predict inputs", () => {
    const Xc = f64([[1], [2], [3]]);
    const model = new KernelRidge({ kernel: "rbf", alpha: 1e-3 }).fit(Xc, f64([4, 4, 4]));
    expect(Number.isFinite(model.score(Xc, f64([4, 4, 4])))).toBe(true);
    expect(model.nFeaturesIn).toBe(1);
    expect(() => new KernelRidge().predict(Xc)).toThrow(/must be fitted/);
    expect(() => model.predict(f64([[1, 2]]))).toThrow(ShapeError);
  });

  it("clone keeps parameters and drops the fit", () => {
    const model = new KernelRidge({ kernel: "poly", degree: 2 }).fit(XK, yK);
    const clone = model.clone();
    expect(clone.getParams()).toEqual(model.getParams());
    expect(() => clone.dualCoef).toThrow(NotFittedError);
  });
});
