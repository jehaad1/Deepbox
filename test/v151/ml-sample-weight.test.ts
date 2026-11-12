import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError, ShapeError } from "../../src/core";
import {
  AdaBoostClassifier,
  AdaBoostRegressor,
  BayesianRidge,
  ElasticNet,
  GaussianNB,
  HuberRegressor,
  KernelRidge,
  KNeighborsClassifier,
  KNeighborsRegressor,
  Lasso,
  LinearDiscriminantAnalysis,
  LinearRegression,
  LinearSVR,
  LogisticRegression,
  MLPClassifier,
  MLPRegressor,
  QuadraticDiscriminantAnalysis,
  Ridge,
  SGDClassifier,
  SGDRegressor,
} from "../../src/ml";
import { type Tensor, tensor } from "../../src/ndarray";

// Reference values: scikit-learn 1.9.1, fitted with sample_weight on makeData(48) below.
// Linear models: LinearRegression(), Ridge(alpha=0.7), Lasso(alpha=0.05, tol=1e-12),
// ElasticNet(alpha=0.05, l1_ratio=0.4, tol=1e-12), LogisticRegression(C=2, tol=1e-10),
// GaussianNB(), BayesianRidge(tol=1e-12), HuberRegressor(epsilon=1.35, alpha=0.01, tol=1e-12),
// KernelRidge(alpha=0.5, kernel="rbf", gamma=0.3), LinearSVR(C=1, epsilon=0.05, dual=True),
// SGDRegressor / SGDClassifier(loss="log_loss", penalty="l2", alpha=0.001, learning_rate="constant",
// eta0=0.01, max_iter=4, tol=None, shuffle=False).
const REF = {
  lr: {
    coef: [1.465791627341571, -1.0016823639981038, 0.5293422832904477, 0.07907900694428689],
    intercept: -0.0029645060298489068,
  },
  ridge: {
    coef: [1.4430340582743053, -0.9870976377616052, 0.5189315390971105, 0.07205339800009829],
    intercept: -0.004150970225953482,
  },
  lasso: {
    coef: [1.3880545584361828, -0.9288280086955651, 0.4266370351270774, 0.0],
    intercept: -0.00971229135606605,
  },
  enet: {
    coef: [1.3703663595308286, -0.9313008559054683, 0.4606836526085812, 0.0],
    intercept: -0.006768786603926241,
  },
  logit2: {
    coef: [[3.553273519488568, -1.1296932422124364, -0.1469756971860239, -0.6318418912617975]],
    intercept: [-0.2032989217055457],
  },
  logit3: {
    coef: [
      [-2.0495215543233036, -1.1159896506886011, -2.53211284803695, 2.3694975691055107],
      [0.25667307949448165, 0.5624502550932422, 0.07996584275951865, -0.6731561863649935],
      [1.7928484748288214, 0.5535393955953587, 2.4521470052774332, -1.6963413827405167],
    ],
    intercept: [0.5644233583990264, 1.2400300033628409, -1.8044533617618645],
    proba: [
      [0.00013778870768576577, 0.343946785869053, 0.6559154254232612],
      [9.531220390688388e-5, 0.1767228107870793, 0.8231818770090138],
      [0.0021951635278159433, 0.2647126133933365, 0.7330922230788475],
      [0.055247550982234044, 0.6475704721597305, 0.2971819768580354],
    ],
  },
  gnb: {
    theta: [
      [-0.38285230629828654, -0.13642827796674759, -0.40753329108677006, 0.2520833333333333],
      [0.22274568037472847, 0.46693055818524065, 0.02117968712443707, 0.0016260162601626428],
      [0.8383422099978246, -0.17864713745476293, 0.9009057658940937, -0.27424242424242423],
    ],
    var: [
      [0.22818116193779733, 0.5631245103221142, 0.19851472471507584, 0.12864149368407873],
      [0.7475670258758523, 0.4557845320803603, 0.47311321965399017, 0.180729064016718],
      [0.146510824506356, 0.3539763579071481, 0.03839457598744886, 0.028351699434767445],
    ],
    prior: [0.43478260869565216, 0.44565217391304346, 0.11956521739130435],
    proba: [
      [0.000515084278938502, 0.6233301999061986, 0.3761547158148629],
      [1.1168188485563837e-5, 0.06259427743502405, 0.9373945543764903],
      [5.4209702246226626e-5, 0.037723926381698684, 0.9622218639160551],
      [0.04165952945522171, 0.9394715705479209, 0.01886889999685758],
    ],
  },
  bayes: {
    coef: [1.3593526495169588, -1.0229955619188627, 0.4521157745025574, 0.38925597213650764],
    intercept: 0.2789302117458453,
    alpha: 0.8660354192823152,
    lambda: 1.16694404007695,
  },
  huber: {
    coef: [1.4412888609376373, -1.0276311228047397, 0.5395283102470572, 0.1202487258231048],
    intercept: 0.028829484547111334,
    scale: 0.10341303522941298,
  },
  kr: {
    pred: [
      -0.22774625949182825, 1.579097327944701, 2.739828876623017, 2.9432389945737394,
      0.48191005963512,
    ],
    dual: [0.0, 0.5406359833729475, 1.1245869162518702, -1.1128055950607674, -1.5833113985072098],
  },
  lsvr: {
    coef: [1.378406282442515, -1.0400278241571692, 0.45943678799380067, 0.42880558494110543],
    intercept: 0.3105258817893378,
  },
  lsvr1: {
    coef: [1.4244068432399242, -1.0143589780028572, 0.5401441452595919, 0.09953407857913402],
    intercept: 0.029254893419645874,
  },
  sgdr: {
    coef: [1.1534893690269885, -0.8519795897210999, 0.33807967229908836, 0.13093665936285045],
    intercept: 0.26845631584110397,
  },
  sgdc: {
    coef: [[0.7105870722274051, -0.22354354446294344, -0.011394313010060737, -0.10535897236024526]],
    intercept: -0.015439609795898734,
  },
};

const N = 48;
const f64 = { dtype: "float64" } as const;

function makeData() {
  const rows: number[][] = [];
  const y: number[] = [];
  const yOut: number[] = [];
  const w: number[] = [];
  const yc: number[] = [];
  const yb: number[] = [];
  for (let i = 0; i < N; i++) {
    const x = [
      Math.sin(i * 0.7) + 0.4 * Math.cos(i * 0.37),
      Math.cos(i * 1.3) + 0.3 * Math.sin(i * 0.11),
      Math.sin(i * 0.53 + 1) + 0.2 * Math.cos(i * 0.9),
      (i % 6) * 0.25 - 0.6,
    ];
    rows.push(x);
    const clean = 1.5 * x[0]! - x[1]! + 0.5 * x[2]! + 0.2 * Math.sin(i * 3.1);
    y.push(clean);
    yOut.push(clean + (i % 11 === 0 ? 4 : 0));
    w.push((i % 7 === 0 ? 0 : i % 5 === 0 ? 3 : 1) + 0.25 * (i % 3));
    yc.push((x[0]! + 0.5 * x[1]! > 0.3 ? 1 : 0) + (x[2]! - x[3]! > 0.2 ? 1 : 0));
    yb.push(x[0]! - 0.4 * x[1]! + 0.3 * Math.sin(i * 2.3) > 0 ? 1 : 0);
  }
  return { rows, y, yOut, w, yc, yb };
}

const D = makeData();
const X = tensor(D.rows, f64);
const W = tensor(D.w, f64);
const Y = tensor(D.y, f64);
const YOUT = tensor(D.yOut, f64);
const YC = tensor(D.yc, f64);
const YB = tensor(D.yb, f64);
const XHEAD = tensor(D.rows.slice(0, 4), f64);

function vec(t: Tensor | ArrayLike<number>): number[] {
  const data = (t as Tensor).data ?? (t as ArrayLike<number>);
  return Array.from(data as ArrayLike<number>);
}

function expectNear(actual: ArrayLike<number>, expected: readonly number[], tol: number): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThan(tol);
  }
}

describe("ML-8: sampleWeight matches scikit-learn", () => {
  it("LinearRegression", () => {
    const m = new LinearRegression().fit(X, Y, W);
    expectNear(vec(m.coef), REF.lr.coef, 1e-10);
    expect(Math.abs(vec(m.intercept as Tensor)[0]! - REF.lr.intercept)).toBeLessThan(1e-10);
  });

  it("Ridge", () => {
    const m = new Ridge({ alpha: 0.7 }).fit(X, Y, W);
    expectNear(vec(m.coef), REF.ridge.coef, 1e-10);
    expect(Math.abs(m.intercept - REF.ridge.intercept)).toBeLessThan(1e-10);
  });

  it("Lasso", () => {
    const m = new Lasso({ alpha: 0.05, tol: 1e-12, maxIter: 100000 }).fit(X, Y, W);
    expectNear(vec(m.coef), REF.lasso.coef, 1e-8);
    expect(Math.abs(m.intercept - REF.lasso.intercept)).toBeLessThan(1e-8);
  });

  it("ElasticNet", () => {
    const m = new ElasticNet({ alpha: 0.05, l1Ratio: 0.4, tol: 1e-12, maxIter: 100000 }).fit(
      X,
      Y,
      W
    );
    expectNear(vec(m.coef), REF.enet.coef, 1e-8);
    expect(Math.abs(m.intercept - REF.enet.intercept)).toBeLessThan(1e-8);
  });

  it("LogisticRegression (binary and multinomial)", () => {
    const b = new LogisticRegression({ C: 2, tol: 1e-10, maxIter: 10000 }).fit(X, YB, W);
    expectNear(vec(b.coef), REF.logit2.coef[0]!, 1e-6);
    expect(Math.abs((b.intercept as number) - REF.logit2.intercept[0]!)).toBeLessThan(1e-6);
    const m = new LogisticRegression({ C: 2, tol: 1e-10, maxIter: 10000 }).fit(X, YC, W);
    expectNear(vec(m.predictProba(XHEAD)), REF.logit3.proba.flat(), 1e-5);
  });

  it("GaussianNB", () => {
    const m = new GaussianNB().fit(X, YC, W);
    expectNear(vec(m.predictProba(XHEAD)), REF.gnb.proba.flat(), 1e-10);
  });

  it("BayesianRidge", () => {
    const m = new BayesianRidge({ tol: 1e-12, maxIter: 1000 }).fit(X, YOUT, W);
    expectNear(vec(m.coef), REF.bayes.coef, 1e-8);
    expect(Math.abs(m.intercept - REF.bayes.intercept)).toBeLessThan(1e-8);
  });

  it("HuberRegressor", () => {
    const m = new HuberRegressor({ epsilon: 1.35, alpha: 0.01, tol: 1e-12, maxIter: 5000 }).fit(
      X,
      YOUT,
      W
    );
    expectNear(vec(m.coef), REF.huber.coef, 1e-5);
    expect(Math.abs(m.intercept - REF.huber.intercept)).toBeLessThan(1e-5);
    expect(Math.abs(m.scale - REF.huber.scale)).toBeLessThan(1e-5);
  });

  it("KernelRidge", () => {
    const m = new KernelRidge({ alpha: 0.5, kernel: "rbf", gamma: 0.3 }).fit(X, YOUT, W);
    expectNear(vec(m.predict(tensor(D.rows.slice(0, 5), f64))), REF.kr.pred, 1e-10);
    expectNear(vec(m.dualCoef).slice(0, 5), REF.kr.dual, 1e-10);
  });

  it("LinearSVR (both losses)", () => {
    const sq = new LinearSVR({
      C: 1,
      epsilon: 0.05,
      loss: "squaredEpsilonInsensitive",
      tol: 1e-10,
      maxIter: 200000,
      randomState: 0,
    }).fit(X, YOUT, W);
    expectNear(vec(sq.coef), REF.lsvr.coef, 1e-6);
    const l1 = new LinearSVR({
      C: 1,
      epsilon: 0.05,
      loss: "epsilonInsensitive",
      tol: 1e-10,
      maxIter: 200000,
      randomState: 0,
    }).fit(X, YOUT, W);
    expectNear(vec(l1.coef), REF.lsvr1.coef, 1e-6);
  });

  it("SGDRegressor and SGDClassifier follow scikit-learn's weighted updates", () => {
    const opts = {
      penalty: "l2",
      alpha: 0.001,
      learningRate: "constant",
      eta0: 0.01,
      maxIter: 4,
      tol: 0,
      shuffle: false,
      randomState: 0,
    } as const;
    const r = new SGDRegressor(opts).fit(X, YOUT, W);
    expectNear(vec(r.coef), REF.sgdr.coef, 1e-10);
    expect(Math.abs(r.intercept - REF.sgdr.intercept)).toBeLessThan(1e-10);
    const c = new SGDClassifier({ ...opts, loss: "log_loss" }).fit(X, YB, W);
    expectNear(vec(c.coef), REF.sgdc.coef[0]!, 1e-10);
  });
});

describe("ML-8: weights behave like sample multiplicities", () => {
  const rowsWithWeights = D.rows.map((r, i) => ({
    r,
    y: D.y[i]!,
    yb: D.yb[i]!,
    w: i % 4 === 0 ? 2 : 1,
  }));
  const doubled = rowsWithWeights.flatMap((e) => (e.w === 2 ? [e, e] : [e]));
  const Xw = tensor(
    rowsWithWeights.map((e) => e.r),
    f64
  );
  const Xd = tensor(
    doubled.map((e) => e.r),
    f64
  );
  const yw = tensor(
    rowsWithWeights.map((e) => e.y),
    f64
  );
  const yd = tensor(
    doubled.map((e) => e.y),
    f64
  );
  const ybw = tensor(
    rowsWithWeights.map((e) => e.yb),
    f64
  );
  const ybd = tensor(
    doubled.map((e) => e.yb),
    f64
  );
  const wv = tensor(
    rowsWithWeights.map((e) => e.w),
    f64
  );

  it("integer weights equal repeating the rows (LinearRegression, Ridge, GaussianNB)", () => {
    expectNear(
      vec(new LinearRegression().fit(Xw, yw, wv).coef),
      vec(new LinearRegression().fit(Xd, yd).coef),
      1e-10
    );
    expectNear(vec(new Ridge().fit(Xw, yw, wv).coef), vec(new Ridge().fit(Xd, yd).coef), 1e-10);
    expectNear(
      vec(new GaussianNB().fit(Xw, ybw, wv).predictProba(XHEAD)),
      vec(new GaussianNB().fit(Xd, ybd).predictProba(XHEAD)),
      1e-10
    );
  });

  it("zero weights equal dropping the rows", () => {
    const keep = D.w.map((w) => w > 0);
    const Xk = tensor(
      D.rows.filter((_, i) => keep[i]),
      f64
    );
    const yk = tensor(
      D.y.filter((_, i) => keep[i]),
      f64
    );
    const wk = tensor(
      D.w.filter((_, i) => keep[i]),
      f64
    );
    expectNear(
      vec(new LinearRegression().fit(X, Y, W).coef),
      vec(new LinearRegression().fit(Xk, yk, wk).coef),
      1e-10
    );
    expectNear(vec(new Ridge().fit(X, Y, W).coef), vec(new Ridge().fit(Xk, yk, wk).coef), 1e-10);
  });

  it("unit weights change nothing", () => {
    const ones = tensor(new Array<number>(N).fill(1), f64);
    expectNear(vec(new Ridge().fit(X, Y, ones).coef), vec(new Ridge().fit(X, Y).coef), 1e-12);
    expectNear(
      vec(new Lasso({ alpha: 0.05 }).fit(X, Y, ones).coef),
      vec(new Lasso({ alpha: 0.05 }).fit(X, Y).coef),
      1e-12
    );
    expectNear(
      vec(new BayesianRidge().fit(X, Y, ones).coef),
      vec(new BayesianRidge().fit(X, Y).coef),
      1e-12
    );
  });

  it("AdaBoost and MLP draw on the weights", () => {
    const heavy = tensor(
      D.w.map((_, i) => (i < 24 ? 50 : 0.001)),
      f64
    );
    const a = new AdaBoostRegressor({ nEstimators: 5, randomState: 0 }).fit(X, Y);
    const b = new AdaBoostRegressor({ nEstimators: 5, randomState: 0 }).fit(X, Y, heavy);
    expect(vec(a.predict(X))).not.toEqual(vec(b.predict(X)));
    const ac = new AdaBoostClassifier({ nEstimators: 5, randomState: 0 }).fit(X, YB);
    const bc = new AdaBoostClassifier({ nEstimators: 5, randomState: 0 }).fit(X, YB, heavy);
    expect(vec(ac.predict(X))).not.toEqual(vec(bc.predict(X)));
    const mc = new MLPClassifier({ maxIter: 20, randomState: 0 }).fit(X, YB);
    const mw = new MLPClassifier({ maxIter: 20, randomState: 0 }).fit(X, YB, heavy);
    expect(vec(mc.predictProba(X))).not.toEqual(vec(mw.predictProba(X)));
    const rc = new MLPRegressor({ maxIter: 20, randomState: 0 }).fit(X, Y);
    const rw = new MLPRegressor({ maxIter: 20, randomState: 0 }).fit(X, Y, heavy);
    expect(vec(rc.predict(X))).not.toEqual(vec(rw.predict(X)));
  });
});

describe("ML-8: invalid weights and estimators without weight support", () => {
  it("rejects a wrong length, negative weights and all-zero weights", () => {
    expect(() => new Ridge().fit(X, Y, tensor([1, 2, 3], f64))).toThrow(ShapeError);
    expect(() => new Ridge().fit(X, Y, tensor(new Array<number>(N).fill(-1), f64))).toThrow(
      DataValidationError
    );
    expect(() => new GaussianNB().fit(X, YC, tensor(new Array<number>(N).fill(0), f64))).toThrow(
      DataValidationError
    );
  });

  it("estimators that cannot use weights throw instead of ignoring them", () => {
    const w = W as unknown as undefined;
    expect(() => new KNeighborsClassifier().fit(X, YC, w)).toThrow(/does not support sampleWeight/);
    expect(() => new KNeighborsRegressor().fit(X, Y, w)).toThrow(InvalidParameterError);
    expect(() => new LinearDiscriminantAnalysis().fit(X, YC, w)).toThrow(
      /does not support sampleWeight/
    );
    expect(() => new QuadraticDiscriminantAnalysis().fit(X, YC, w)).toThrow(InvalidParameterError);
  });
});
