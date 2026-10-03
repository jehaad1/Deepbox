import { describe, expect, it } from "vitest";
import {
  catchWarnings,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import { LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR } from "../../src/ml";
import { getEstimatorTags } from "../../src/ml/base";
import { type Tensor, tensor, transpose } from "../../src/ndarray";
import { clearSeed, setSeed } from "../../src/random";

// Reference values come from scikit-learn 1.8 (SVC, SVR, NuSVC, NuSVR, OneClassSVM,
// LinearSVC, LinearSVR) fitted on the same data with tol=1e-9 / 1e-10.
const REF = {
  X: [
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.2, 0.1],
    [0.9, 0.8],
    [0.1, 0.9],
    [0.8, 0.2],
    [0.5, 0.45],
    [0.45, 0.5],
  ],
  y: [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
  X3: [
    [0.0, 0.0],
    [0.5, 0.3],
    [0.2, 0.6],
    [4.0, 0.0],
    [4.5, 0.4],
    [3.8, 0.7],
    [0.0, 4.0],
    [0.4, 4.5],
    [0.9, 3.7],
    [2.0, 2.1],
  ],
  y3: [0, 0, 0, 1, 1, 1, 2, 2, 2, 1],
  yr: [0.1, 1.9, 1.1, 0.8, 0.25, 1.6, 0.95, 0.9, 0.7, 0.6],
  sw: [1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0],
  svc_rbf: {
    dual: [[-0.427966827, -0.427966827, -2.0, -2.0, -2.0, 0.855933654, 2.0, 2.0, 2.0]],
    icpt: [0.035217727],
    sv: [0, 1, 4, 5, 8, 3, 6, 7, 9],
    dec: [
      -0.999999945, -0.999999945, 1.144540601, 0.99999995, -0.666565287, -0.664407419, 0.959510856,
      0.512339159, 0.039411508, 0.061063228,
    ],
    pred: [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
  },
  svc_poly3: {
    dual: [
      [0.02201527, 0.004275891, -0.001057341, -0.025233821, -0.009128669],
      [0.0, 0.009128669, 0.0, 0.018984643, -0.018984643],
    ],
    icpt: [1.137519645, 1.087635222, 1.440003207],
    nsup: [2, 2, 1],
    dec: [
      [2.229979123, 1.077411997, -0.238841412],
      [2.223783886, 1.102955796, -0.237813217],
      [2.222222223, 1.087306991, -0.233975241],
      [0.976470257, 2.260490215, -0.259260802],
      [0.818216103, 2.275871692, -0.2608927],
      [0.853605685, 2.265275275, -0.252322311],
      [-0.223065152, 0.864534616, 2.243426902],
      [-0.260900114, 0.895591783, 2.267431396],
      [-0.238377155, 1.11263996, 2.222222222],
      [0.845141832, 2.222222216, -0.177009546],
    ],
    pred: [0, 0, 0, 1, 1, 1, 2, 2, 2, 1],
  },
  svc_w: {
    dual: [[-1.0, -2.0, -1.0, -1.0, -1.0, 0.5, 0.5, 3.0, 1.0, 1.0]],
    icpt: [0.45000007],
    sv: [0, 1, 4, 5, 8, 2, 3, 6, 7, 9],
  },
  svc_sig: {
    dual: [[-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    icpt: [-0.049814481],
  },
  svr: {
    dual: [[-0.605710099, 1.179681038, 0.260212216, 0.094913224, -0.929096379]],
    icpt: [1.093185089],
    sv: [0, 1, 2, 7, 9],
    pred: [
      0.200000022, 1.800000007, 1.000000018, 0.892465491, 0.226756756, 1.5565561, 0.936219074,
      0.799999983, 0.693629586, 0.699999994,
    ],
  },
  svr_w: {
    dual: [[2.150000036, 3.649999964, 0.2, -3.0, -3.0]],
    icpt: [-0.0],
    pred: [
      -0.0, 1.799999928, 0.999999964, 0.799999964, 0.259999989, 1.519999939, 0.979999964,
      0.839999964, 0.849999966, 0.859999966,
    ],
  },
  nusvc: {
    dual: [[-4.100588891, -4.123633788, -16.448445357, 1.539027641, 6.685195038, 16.448445357]],
    icpt: [-0.191531728],
    sv: [4, 5, 8, 6, 7, 9],
    dec: [
      -1.603810921, -1.622572586, 1.146185277, 1.60890071, -1.000000378, -0.999999871, 1.000000366,
      1.000000186, 0.216688657, 0.203249744,
    ],
  },
  nusvc3: {
    icpt: [-0.318739161, -0.112142048, 0.215408931],
    dec: [
      [2.222222223, 0.853469456, -0.182884114],
      [2.222222223, 0.853797631, -0.183096312],
      [2.222222222, 0.853793691, -0.183093771],
      [-0.175515543, 2.222222221, 0.843233474],
      [-0.175515661, 2.222222221, 0.843233621],
      [-0.175515197, 2.222222222, 0.843233039],
      [-0.189597693, 0.864969857, 2.222222221],
      [-0.189587354, 0.864950184, 2.222222221],
      [-0.190545344, 0.866802367, 2.222222222],
      [-0.176258562, 2.222222222, 0.844172892],
    ],
    pred: [0, 0, 0, 1, 1, 1, 2, 2, 2, 1],
    nsup: [3, 4, 3],
  },
  nusvr: {
    dual: [[-0.746303904, 1.212198301, 0.625015209, -1.253696096, 0.16278649, 2.0, -2.0]],
    icpt: [1.068600845],
    pred: [
      0.151080508, 1.848919507, 1.048919541, 0.851080548, 0.198919535, 1.609046644, 0.966140651,
      0.81642263, 0.712577572, 0.715008156,
    ],
  },
  oc: {
    dual: [[0.750000001, 0.75, 0.749999999, 0.75]],
    offset: 1.403320637,
    score: [
      1.403320624, 1.403320624, 1.403320624, 1.403320624, 1.601445316, 1.601445316, 1.544220511,
      1.660790724, 1.817317964, 1.817317964,
    ],
    dec: [
      -1.2e-8, -1.2e-8, -1.3e-8, -1.2e-8, 0.19812468, 0.198124679, 0.140899874, 0.257470087,
      0.413997328, 0.413997328,
    ],
    sv: [0, 1, 2, 3],
    score_te: [1.819591979, 0.000254999, 0.451153546],
    pred_te: [1, -1, -1],
  },
  Xte: [
    [0.5, 0.5],
    [3.0, 3.0],
    [-1.0, 0.5],
  ],
  lsvc_h: {
    coef: [[-0.25, 0.25]],
    icpt: [-0.0],
    dec: [-0.0, 0.0, 0.25, -0.25, -0.025, -0.025, 0.2, -0.15, -0.0125, 0.0125],
  },
  lsvc_sh: {
    coef: [[-0.112269447, 0.112269447]],
    icpt: [0.0],
    dec: [
      0.0, 0.0, 0.112269447, -0.112269447, -0.011226945, -0.011226945, 0.089815557, -0.067361668,
      -0.005613472, 0.005613472,
    ],
  },
  lsvc_3: {
    coef: [
      [-0.625, -0.625],
      [1.411481361, -0.279606564],
      [-0.662251656, 0.794701987],
    ],
    icpt: [1.5, -1.235788937, -1.344370861],
    dec: [
      [1.5, -1.235788937, -1.344370861],
      [1.0, -0.613930226, -1.437086093],
      [1.0, -1.121256603, -1.0],
      [-1.0, 4.410136507, -3.993377483],
      [-1.5625, 5.004034562, -4.006622516],
      [-1.3125, 3.93211564, -3.304635762],
      [-1.0, -2.354215194, 1.834437086],
      [-1.5625, -1.929425932, 1.966887417],
      [-1.375, -1.0, 1.0],
      [-1.0625, 1.0, -1.0],
    ],
    pred: [0, 0, 0, 1, 1, 1, 2, 2, 2, 1],
  },
  lsvc_sw: { coef: [[-0.740438271, 0.382624976]], icpt: [0.132164051] },
  lsvr: {
    coef: [0.765662651, 0.945481928],
    icpt: [0.054518072],
    pred: [
      0.054518072, 1.765662651, 1.0, 0.820180723, 0.302198795, 1.5, 0.982018072, 0.856144578,
      0.862816265, 0.871807229,
    ],
  },
  lsvr_sq: {
    coef: [0.667320661, 0.848741676],
    icpt: [0.137022299],
    pred: [
      0.137022299, 1.653084636, 0.985763975, 0.80434296, 0.355360599, 1.416604235, 0.967621873,
      0.840627163, 0.852616384, 0.861687434,
    ],
  },
};

const f64 = (rows: number | number[] | number[][]): Tensor => tensor(rows, { dtype: "float64" });
const X = f64(REF.X);
const y = tensor(REF.y);
const X3 = f64(REF.X3);
const y3 = tensor(REF.y3);
const yr = f64(REF.yr);
const sw = f64(REF.sw);

const flat = (a: unknown): number[] =>
  Array.isArray(a) ? (a as unknown[]).flatMap(flat) : [a as number];
const close = (actual: Tensor | number[], expected: unknown, tol: number): void => {
  const a = flat(Array.isArray(actual) ? actual : actual.toArray());
  const e = flat(expected);
  expect(a.length).toBe(e.length);
  for (let i = 0; i < e.length; i++) {
    expect(Math.abs((a[i] as number) - (e[i] as number))).toBeLessThan(tol);
  }
};

describe("SVC matches scikit-learn", () => {
  it("rbf, two classes: dual coefficients, intercept, support set, decision values", () => {
    const m = new SVC({ C: 2, gamma: 1.5, tol: 1e-9 }).fit(X, y);
    close(m.dualCoef, REF.svc_rbf.dual, 1e-5);
    close(m.intercept, REF.svc_rbf.icpt, 1e-5);
    expect(m.supportIndices.toArray()).toEqual(REF.svc_rbf.sv);
    close(m.decisionFunction(X), REF.svc_rbf.dec, 1e-5);
    expect(m.predict(X).toArray()).toEqual(REF.svc_rbf.pred);
  });

  it("poly kernel with three classes uses one-vs-one voting and the ovr decision shape", () => {
    const m = new SVC({ C: 1, kernel: "poly", degree: 2, gamma: 1, coef0: 1, tol: 1e-9 }).fit(
      X3,
      y3
    );
    close(m.dualCoef, REF.svc_poly3.dual, 1e-5);
    close(m.intercept, REF.svc_poly3.icpt, 1e-5);
    expect(m.nSupport.toArray()).toEqual(REF.svc_poly3.nsup);
    expect(m.decisionFunction(X3).shape).toEqual([10, 3]);
    close(m.decisionFunction(X3), REF.svc_poly3.dec, 1e-5);
    expect(m.predict(X3).toArray()).toEqual(REF.svc_poly3.pred);
  });

  it("classWeight and sampleWeight scale C per sample", () => {
    const m = new SVC({ kernel: "linear", tol: 1e-9, classWeight: "balanced" }).fit(X, y, sw);
    close(m.dualCoef, REF.svc_w.dual, 1e-5);
    close(m.intercept, REF.svc_w.icpt, 1e-5);
    expect(m.supportIndices.toArray()).toEqual(REF.svc_w.sv);
  });

  it("sigmoid kernel", () => {
    const m = new SVC({ kernel: "sigmoid", gamma: 0.5, coef0: 0.1, tol: 1e-9 }).fit(X, y);
    close(m.dualCoef, REF.svc_sig.dual, 1e-5);
    close(m.intercept, REF.svc_sig.icpt, 1e-5);
  });

  it("does not depend on the global random seed", () => {
    setSeed(1);
    const a = new SVC({ C: 2, gamma: 1.5 }).fit(X, y).decisionFunction(X).toArray();
    setSeed(999);
    const b = new SVC({ C: 2, gamma: 1.5 }).fit(X, y).decisionFunction(X).toArray();
    clearSeed();
    expect(a).toEqual(b);
  });
});

describe("SVR matches scikit-learn", () => {
  it("rbf epsilon-SVR", () => {
    const m = new SVR({ C: 3, epsilon: 0.1, gamma: 1, tol: 1e-9 }).fit(X, yr);
    close(m.dualCoef, REF.svr.dual, 1e-5);
    close(m.intercept, REF.svr.icpt, 1e-5);
    expect(m.supportIndices.toArray()).toEqual(REF.svr.sv);
    close(m.predict(X), REF.svr.pred, 1e-5);
  });

  it("linear kernel with sample weights", () => {
    const m = new SVR({ C: 3, epsilon: 0.1, kernel: "linear", tol: 1e-9 }).fit(X, yr, sw);
    close(m.dualCoef, REF.svr_w.dual, 1e-5);
    close(m.predict(X), REF.svr_w.pred, 1e-5);
  });

  it("a very wide tube leaves no support vectors and a constant model", () => {
    const m = new SVR({ epsilon: 100 }).fit(f64([[0], [1], [2]]), f64([1, 2, 3]));
    expect(m.supportVectors.shape).toEqual([0, 1]);
    const p = m.predict(f64([[0], [1]])).toArray() as number[];
    expect(p[0]).toBe(p[1]);
  });

  it("a single sample is reproduced", () => {
    const m = new SVR().fit(f64([[1]]), f64([2]));
    close(m.predict(f64([[1], [5]])), [2, 2], 1e-9);
  });
});

describe("NuSVC, NuSVR and OneClassSVM match scikit-learn", () => {
  it("NuSVC two classes", () => {
    const m = new NuSVC({ nu: 0.3, gamma: 1.5, tol: 1e-9 }).fit(X, y);
    close(m.dualCoef, REF.nusvc.dual, 1e-4);
    close(m.intercept, REF.nusvc.icpt, 1e-5);
    expect(m.supportIndices.toArray()).toEqual(REF.nusvc.sv);
    close(m.decisionFunction(X), REF.nusvc.dec, 1e-4);
  });

  it("NuSVC three classes", () => {
    const m = new NuSVC({ nu: 0.2, gamma: 1, tol: 1e-9 }).fit(X3, y3);
    close(m.intercept, REF.nusvc3.icpt, 1e-5);
    close(m.decisionFunction(X3), REF.nusvc3.dec, 1e-4);
    expect(m.predict(X3).toArray()).toEqual(REF.nusvc3.pred);
    expect(m.nSupport.toArray()).toEqual(REF.nusvc3.nsup);
  });

  it("NuSVC rejects a nu that is infeasible for the class sizes", () => {
    const Xi = f64([[0], [1], [2], [3], [4], [5]]);
    const yi = tensor([0, 0, 0, 0, 0, 1]);
    expect(() => new NuSVC({ nu: 0.9 }).fit(Xi, yi)).toThrow(InvalidParameterError);
    expect(() => new NuSVC({ nu: 0.9 }).fit(Xi, yi)).toThrow(/infeasible/);
  });

  it("NuSVC reports identical samples with different labels instead of returning NaN", () => {
    const Xd = f64([
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
    ]);
    expect(() => new NuSVC({ nu: 0.5 }).fit(Xd, tensor([0, 1, 0, 1]))).toThrow(DataValidationError);
  });

  it("NuSVR", () => {
    const m = new NuSVR({ nu: 0.4, C: 2, gamma: 1, tol: 1e-9 }).fit(X, yr);
    close(m.dualCoef, REF.nusvr.dual, 1e-4);
    close(m.intercept, REF.nusvr.icpt, 1e-5);
    close(m.predict(X), REF.nusvr.pred, 1e-4);
  });

  it("OneClassSVM: dual coefficients, offset, score_samples and decision_function", () => {
    const m = new OneClassSVM({ nu: 0.3, gamma: 1, tol: 1e-9 }).fit(X);
    close(m.dualCoef, REF.oc.dual, 1e-5);
    expect(m.offset).toBeCloseTo(REF.oc.offset, 5);
    expect(m.supportIndices.toArray()).toEqual(REF.oc.sv);
    close(m.scoreSamples(X), REF.oc.score, 1e-5);
    close(m.decisionFunction(X), REF.oc.dec, 1e-5);
    const Xte = f64(REF.Xte);
    close(m.scoreSamples(Xte), REF.oc.score_te, 1e-5);
    expect(m.predict(Xte).toArray()).toEqual(REF.oc.pred_te);
  });

  it("OneClassSVM: scoreSamples minus offset is decisionFunction, and sum(alpha) = nu * n", () => {
    const m = new OneClassSVM({ nu: 0.3 }).fit(X);
    const s = m.scoreSamples(X).toArray() as number[];
    const d = m.decisionFunction(X).toArray() as number[];
    for (let i = 0; i < s.length; i++) expect((s[i] as number) - m.offset).toBeCloseTo(d[i]!, 12);
    const total = (m.dualCoef.toArray() as number[][])[0]!.reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(0.3 * 10, 9);
  });

  it("OneClassSVM keeps the inlier label for a single training sample", () => {
    const m = new OneClassSVM().fit(f64([[1, 2]]));
    expect(m.predict(f64([[1, 2]])).toArray()).toEqual([1]);
  });

  it("OneClassSVM is tagged as an outlier detector", () => {
    expect(getEstimatorTags(new OneClassSVM()).estimatorType).toBe("outlier_detector");
  });
});

describe("LinearSVC matches scikit-learn / LIBLINEAR", () => {
  const opts = { tol: 1e-10, maxIter: 200000, randomState: 0 };

  it("hinge loss", () => {
    const m = new LinearSVC({ ...opts, loss: "hinge", C: 1 }).fit(X, y);
    close(m.coef, REF.lsvc_h.coef, 1e-6);
    close(m.intercept, REF.lsvc_h.icpt, 1e-6);
    close(m.decisionFunction(X), REF.lsvc_h.dec, 1e-6);
  });

  it("squared hinge loss", () => {
    const m = new LinearSVC({ ...opts, loss: "squaredHinge", C: 0.7 }).fit(X, y);
    close(m.coef, REF.lsvc_sh.coef, 1e-6);
    close(m.intercept, REF.lsvc_sh.icpt, 1e-6);
  });

  it("one-vs-rest with three classes", () => {
    const m = new LinearSVC({ ...opts, loss: "hinge", C: 2 }).fit(X3, y3);
    expect(m.coef.shape).toEqual([3, 2]);
    close(m.coef, REF.lsvc_3.coef, 1e-6);
    close(m.intercept, REF.lsvc_3.icpt, 1e-6);
    close(m.decisionFunction(X3), REF.lsvc_3.dec, 1e-6);
    expect(m.predict(X3).toArray()).toEqual(REF.lsvc_3.pred);
  });

  it("sampleWeight", () => {
    const m = new LinearSVC({ ...opts, loss: "squaredHinge" }).fit(X, y, sw);
    close(m.coef, REF.lsvc_sw.coef, 1e-6);
    close(m.intercept, REF.lsvc_sw.icpt, 1e-6);
  });

  it("applies C to the summed loss, not to each sample (the old solver halved this fit)", () => {
    const rows: number[][] = [];
    const labels: number[] = [];
    for (let i = 0; i < 100; i++) {
      rows.push([i / 100, 0.3]);
      labels.push(0);
      rows.push([2 + i / 100, 0.7]);
      labels.push(1);
    }
    // scikit-learn: LinearSVC(loss="hinge", C=1, dual=True, tol=1e-10)
    const m = new LinearSVC({ tol: 1e-9, maxIter: 200000, randomState: 0 }).fit(
      f64(rows),
      tensor(labels)
    );
    close(m.coef, [1.541664601132318, 0.760422346673858], 1e-5);
    close(m.intercept, [-2.6618747830552136], 1e-5);
  });

  it("randomState makes fits reproducible and fitIntercept=false drops the intercept", () => {
    const a = new LinearSVC({ randomState: 3 }).fit(X, y).coef.toArray();
    const b = new LinearSVC({ randomState: 3 }).fit(X, y).coef.toArray();
    expect(a).toEqual(b);
    const m = new LinearSVC({ fitIntercept: false }).fit(X, y);
    expect(m.intercept.toArray()).toEqual([0]);
  });

  it("keeps classWeight in getParams, clone and setParams", () => {
    const m = new LinearSVC({ classWeight: { 0: 1, 1: 3 } });
    expect(m.getParams()["classWeight"]).toEqual({ 0: 1, 1: 3 });
    expect(m.clone().getParams()).toEqual(m.getParams());
    m.setParams({ classWeight: undefined });
    expect("classWeight" in m.getParams()).toBe(false);
  });

  it("warns when the iteration budget runs out", () => {
    const warnings = catchWarnings(() => {
      new LinearSVC({ maxIter: 1, tol: 1e-12 }).fit(X, y);
    });
    expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
  });
});

describe("LinearSVR matches scikit-learn / LIBLINEAR", () => {
  const opts = { tol: 1e-10, maxIter: 200000, randomState: 0 };

  it("epsilon-insensitive loss", () => {
    const m = new LinearSVR({ ...opts, epsilon: 0.1, C: 1 }).fit(X, yr);
    close(m.coef, REF.lsvr.coef, 1e-6);
    close(m.intercept, REF.lsvr.icpt, 1e-6);
    close(m.predict(X), REF.lsvr.pred, 1e-6);
  });

  it("squared epsilon-insensitive loss", () => {
    const m = new LinearSVR({
      ...opts,
      loss: "squaredEpsilonInsensitive",
      epsilon: 0.05,
      C: 2,
    }).fit(X, yr);
    close(m.coef, REF.lsvr_sq.coef, 1e-6);
    close(m.intercept, REF.lsvr_sq.icpt, 1e-6);
  });

  it("recovers a noiseless line", () => {
    const Xl = f64([[1], [2], [3], [4], [5], [6]]);
    const yl = f64([3, 5, 7, 9, 11, 13]);
    const m = new LinearSVR({ C: 100, epsilon: 0, maxIter: 20000, randomState: 1 }).fit(Xl, yl);
    expect(m.score(Xl, yl)).toBeGreaterThan(0.999);
  });

  it("exposes coef, intercept and the unfitted error", () => {
    const m = new LinearSVR();
    expect(() => m.coef).toThrow(NotFittedError);
    expect(() => m.predict(X)).toThrow(NotFittedError);
    m.fit(X, yr);
    expect(m.coef.shape).toEqual([2]);
    expect(m.intercept.shape).toEqual([1]);
  });
});

describe("input handling", () => {
  it("keeps fractional class labels instead of truncating them to int32", () => {
    const Xl = f64([[0], [1], [2], [3]]);
    const yl = f64([0.5, 0.5, 1.5, 1.5]);
    for (const clf of [new SVC(), new NuSVC({ nu: 0.4 }), new LinearSVC({ randomState: 0 })]) {
      clf.fit(Xl, yl);
      expect(clf.predict(f64([[0], [3]])).toArray()).toEqual([0.5, 1.5]);
      expect(clf.classes?.toArray()).toEqual([0.5, 1.5]);
    }
  });

  it("returns int32 labels for integer classes and accepts int64 targets", () => {
    const Xl = f64([[0], [1], [2], [3]]);
    const m = new SVC().fit(Xl, tensor([0, 0, 1, 1], { dtype: "int64" }));
    const p = m.predict(f64([[0], [3]]));
    expect(p.dtype).toBe("int32");
    expect(p.toArray()).toEqual([0, 1]);
  });

  it("rejects non-contiguous views instead of reading wrong memory", () => {
    const view = transpose(
      f64([
        [0, 1],
        [1, 2],
        [2, 3],
      ])
    );
    expect(() => new SVC().fit(view, tensor([0, 1]))).toThrow(DataValidationError);
    const m = new SVC().fit(f64([[0], [1], [2], [3]]), tensor([0, 0, 1, 1]));
    // the transpose of a row is a dense (4, 1) column and is read in place
    expect(m.predict(transpose(f64([[0, 1, 2, 3]]))).toArray()).toEqual([0, 0, 1, 1]);
  });

  it("reads samples in order when the classes are interleaved", () => {
    const Xl = f64([[0], [10], [1], [11], [2], [12]]);
    const yl = tensor([0, 1, 0, 1, 0, 1]);
    for (const clf of [new SVC({ kernel: "linear" }), new NuSVC({ nu: 0.4, kernel: "linear" })]) {
      clf.fit(Xl, yl);
      expect(clf.score(Xl, yl)).toBe(1);
    }
  });

  it("score rejects a y with a different number of samples", () => {
    const Xl = f64([[0], [1], [2], [3]]);
    const yl = tensor([0, 0, 1, 1]);
    const models = [new SVC(), new NuSVC({ nu: 0.4 }), new LinearSVC()];
    for (const clf of models) {
      clf.fit(Xl, yl);
      expect(() => clf.score(f64([[0], [1]]), tensor([0, 0, 1]))).toThrow(ShapeError);
    }
    const reg = new NuSVR().fit(Xl, f64([0, 1, 2, 3]));
    expect(() => reg.score(f64([[0], [1]]), f64([0, 1, 2]))).toThrow(ShapeError);
    expect(() => reg.score(f64([[0], [1]]), f64([0, Number.NaN]))).toThrow(DataValidationError);
  });

  it("validates sampleWeight", () => {
    const Xl = f64([[0], [1], [2], [3]]);
    const yl = tensor([0, 0, 1, 1]);
    expect(() => new SVC().fit(Xl, yl, f64([1, 1, 1]))).toThrow(ShapeError);
    expect(() => new SVC().fit(Xl, yl, f64([1, -1, 1, 1]))).toThrow(DataValidationError);
    expect(() => new SVR().fit(Xl, f64([0, 1, 2, 3]), f64([1, 1, Number.NaN, 1]))).toThrow(
      DataValidationError
    );
  });

  it("reports kernel overflow as a typed error", () => {
    const big = f64([
      [1e60, 1],
      [2e60, 2],
      [1, 1],
      [2, 2],
    ]);
    expect(() =>
      new SVC({ kernel: "poly", degree: 8, gamma: 1 }).fit(big, tensor([0, 0, 1, 1]))
    ).toThrow(DataValidationError);
  });

  it("rejects classWeight entries for labels that are not in y", () => {
    const Xl = f64([[0], [1], [2], [3]]);
    expect(() => new SVC({ classWeight: { 7: 2 } }).fit(Xl, tensor([0, 0, 1, 1]))).toThrow(
      InvalidParameterError
    );
  });

  it("a zero class weight removes that class from the penalty", () => {
    const Xl = f64([[0], [1], [2], [3]]);
    const m = new SVC({ classWeight: { 0: 0 } }).fit(Xl, tensor([0, 0, 1, 1]));
    expect(m.supportIndices.size).toBe(0);
  });
});

describe("probabilities and decision values", () => {
  it("predictProba rows sum to one and agree with predict for two and three classes", () => {
    const models: Array<[SVC | NuSVC | LinearSVC, Tensor, Tensor]> = [
      [new SVC({ gamma: 1 }), X, y],
      [new SVC({ gamma: 1 }), X3, y3],
      [new NuSVC({ nu: 0.2 }), X3, y3],
      [new LinearSVC({ randomState: 0 }), X3, y3],
    ];
    for (const [clf, Xa, ya] of models) {
      clf.fit(Xa, ya);
      const proba = clf.predictProba(Xa).toArray() as number[][];
      const pred = clf.predict(Xa).toArray() as number[];
      proba.forEach((row, i) => {
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
        expect(row.indexOf(Math.max(...row))).toBe(pred[i]);
      });
    }
  });

  it("predictProba stays finite for very large decision values", () => {
    const m = new LinearSVC({ C: 1000, randomState: 0 }).fit(
      f64([[0], [1], [2], [3]]),
      tensor([0, 0, 1, 1])
    );
    const p = m.predictProba(f64([[1e8], [-1e8]])).toArray() as number[][];
    for (const row of p) for (const v of row) expect(Number.isFinite(v)).toBe(true);
  });
});

describe("parameters", () => {
  it("rejects NaN and infinite values that the old comparisons let through", () => {
    expect(() => new SVC().setParams({ C: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new SVC().setParams({ C: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    expect(() => new SVC().setParams({ tol: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new SVC().setParams({ gamma: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new SVC().setParams({ coef0: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    expect(() => new SVR().setParams({ epsilon: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new NuSVC().setParams({ nu: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new LinearSVC().setParams({ tol: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("validates every option in the constructor", () => {
    expect(() => new SVC({ kernel: "bad" as never })).toThrow(/kernel/);
    expect(() => new SVC({ gamma: -1 })).toThrow(/gamma/);
    expect(() => new SVC({ degree: 0 })).toThrow(/degree/);
    expect(() => new SVC({ tol: -1 })).toThrow(/tol/);
    expect(() => new SVR({ degree: 1.5 })).toThrow(/degree/);
    expect(() => new SVR({ maxIter: 0 })).toThrow(/maxIter/);
    expect(() => new NuSVC({ kernel: "bad" as never })).toThrow(/kernel/);
    expect(() => new NuSVR({ tol: -1 })).toThrow(/tol/);
    expect(() => new OneClassSVM({ gamma: 0 })).toThrow(/gamma/);
    expect(() => new LinearSVC({ loss: "bad" as never })).toThrow(/loss/);
    expect(() => new LinearSVC({ interceptScaling: 0 })).toThrow(/interceptScaling/);
    expect(() => new LinearSVR({ loss: "bad" as never })).toThrow(/loss/);
    expect(() => new SVC({ classWeight: { 0: -1 } })).toThrow(/classWeight/);
    expect(() => new SVC({ classWeight: "heavy" as never })).toThrow(/classWeight/);
  });

  it("setParams is atomic and rejects unknown keys", () => {
    const m = new SVC({ C: 2 });
    expect(() => m.setParams({ C: 5, kernel: "bad" })).toThrow(InvalidParameterError);
    expect(m.getParams()["C"]).toBe(2);
    expect(() => m.setParams({ nu: 0.5 })).toThrow(/Unknown parameter: nu/);
    const nu = new NuSVC({ nu: 0.3 });
    expect(() => nu.setParams({ nu: 0.7, gamma: "bad" })).toThrow(InvalidParameterError);
    expect(nu.getParams()["nu"]).toBe(0.3);
  });

  it("setParams reaches every option (NuSVC and NuSVR used to ignore most of them)", () => {
    const nu = new NuSVC().setParams({ gamma: 0.25, coef0: 2, degree: 4, maxIter: 77, tol: 0.5 });
    expect(nu.getParams()).toMatchObject({
      gamma: 0.25,
      coef0: 2,
      degree: 4,
      maxIter: 77,
      tol: 0.5,
    });
    const nur = new NuSVR().setParams({
      kernel: "linear",
      gamma: "auto",
    });
    expect(nur.getParams()["kernel"]).toBe("linear");
  });

  it("getParams returns every option and round-trips through the constructor", () => {
    const all: Array<[{ getParams(): Record<string, unknown> }, new (o: never) => unknown]> = [
      [new SVC({ C: 3, classWeight: "balanced", cacheSize: 50 }), SVC as never],
      [new SVR({ epsilon: 0.3 }), SVR as never],
      [new NuSVC({ nu: 0.4, maxIter: 9, tol: 0.1 }), NuSVC as never],
      [new NuSVR({ nu: 0.4, C: 3, degree: 2 }), NuSVR as never],
      [new OneClassSVM({ nu: 0.4, kernel: "poly", degree: 5 }), OneClassSVM as never],
      [new LinearSVC({ C: 2, loss: "squaredHinge", fitIntercept: false }), LinearSVC as never],
      [new LinearSVR({ epsilon: 0.2, loss: "squaredEpsilonInsensitive" }), LinearSVR as never],
    ];
    for (const [model, Ctor] of all) {
      const p = model.getParams();
      const copy = new Ctor(p as never) as { getParams(): Record<string, unknown> };
      expect(copy.getParams()).toEqual(p);
    }
    expect(new NuSVC({ maxIter: 9 }).getParams()["maxIter"]).toBe(9);
    expect(new OneClassSVM({ degree: 5 }).getParams()["degree"]).toBe(5);
  });

  it("clone returns an unfitted estimator with the same parameters", () => {
    const m = new SVC({ C: 4 }).fit(X, y);
    const c = m.clone();
    expect(c.getParams()).toEqual(m.getParams());
    expect(() => c.predict(X)).toThrow(NotFittedError);
    expect(new NuSVR({ nu: 0.2 }).clone().getParams()["nu"]).toBe(0.2);
    expect(new OneClassSVM({ nu: 0.2 }).clone().getParams()["nu"]).toBe(0.2);
  });

  it("fitted attributes throw NotFittedError before fit", () => {
    expect(() => new SVC().supportVectors).toThrow(NotFittedError);
    expect(() => new NuSVC().decisionFunction(X)).toThrow(NotFittedError);
    expect(() => new OneClassSVM().offset).toThrow(NotFittedError);
    expect(() => new OneClassSVM().decisionFunction(X)).toThrow(NotFittedError);
    expect(() => new SVR().intercept).toThrow(NotFittedError);
    expect(new SVC().classes).toBeUndefined();
  });

  it("a failed refit keeps the previously fitted model", () => {
    const m = new SVC({ kernel: "linear" }).fit(f64([[0], [1], [2], [3]]), tensor([0, 0, 1, 1]));
    expect(() => m.fit(f64([[0], [1]]), tensor([0, 0]))).toThrow(InvalidParameterError);
    expect(m.predict(f64([[0], [3]])).toArray()).toEqual([0, 1]);
  });

  it("SVC and NuSVC honour the reported estimator tags", () => {
    expect(getEstimatorTags(new SVC()).hasDecisionFunction).toBe(true);
    expect(getEstimatorTags(new LinearSVC()).hasDecisionFunction).toBe(true);
  });
});

describe("solver budget", () => {
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < 60; i++) {
    const a = Math.sin(i);
    const b = Math.cos(i * 1.3);
    rows.push([a, b]);
    labels.push(a * b + 0.15 * Math.sin(i * 7) > 0 ? 1 : 0);
  }
  const Xc = f64(rows);
  const yc = tensor(labels);

  it("warns when SMO stops before converging", () => {
    const warnings = catchWarnings(() => {
      new SVC({ maxIter: 1, tol: 1e-12, gamma: 2 }).fit(Xc, yc);
    });
    expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
  });

  it("gives the same model when the kernel cache holds only two rows", () => {
    const small = new SVC({ cacheSize: 0.0001, gamma: 2, tol: 1e-6 }).fit(Xc, yc);
    const big = new SVC({ gamma: 2, tol: 1e-6 }).fit(Xc, yc);
    close(small.decisionFunction(Xc), big.decisionFunction(Xc).toArray() as number[], 1e-9);
    const svrSmall = new SVR({ cacheSize: 0.0001, gamma: 2, tol: 1e-6 }).fit(Xc, f64(labels));
    const svrBig = new SVR({ gamma: 2, tol: 1e-6 }).fit(Xc, f64(labels));
    close(svrSmall.predict(Xc), svrBig.predict(Xc).toArray() as number[], 1e-9);
  });
});

describe("zero sample weights and degenerate dual problems", () => {
  const Xz = f64([
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0],
    [0.2, 0.1],
    [0.9, 0.8],
    [0.1, 0.9],
    [0.8, 0.2],
  ]);
  const swz = f64([1, 1, 0, 1, 1, 1, 1, 0]);

  it("SVR ignores samples with zero weight (scikit-learn: SVR(gamma=1.5).fit(X, y, sw))", () => {
    const yz = f64([0, 1, 2, 3, 4, 5, 6, 7]);
    const m = new SVR({ gamma: 1.5, tol: 1e-9 }).fit(Xz, yz, swz);
    expect(m.supportIndices.toArray()).toEqual([0, 1, 3, 4, 5, 6]);
    close(m.dualCoef, [[-1, -1, -1, 1, 1, 1]], 1e-6);
    close(m.intercept, [3.70763248], 1e-6);
    close(
      m.predict(
        f64([
          [0.5, 0.5],
          [0.3, 0.1],
        ])
      ),
      [4.28389477, 3.85742173],
      1e-6
    );
  });

  it("SVC ignores samples with zero weight", () => {
    const yz = tensor([0, 0, 1, 1, 0, 0, 1, 1]);
    const m = new SVC({ gamma: 1.5, tol: 1e-9 }).fit(Xz, yz, swz);
    expect(m.supportIndices.toArray()).toEqual([4, 5, 3, 6]);
    close(m.dualCoef, [[-1, -1, 1, 1]], 1e-6);
    close(m.intercept, [-0.49926515], 1e-6);
  });

  it("all-zero weights give a constant SVR instead of looping on fixed variables", () => {
    const m = new SVR().fit(f64([[0], [1], [2]]), f64([1, 2, 3]), f64([0, 0, 0]));
    expect(m.supportIndices.size).toBe(0);
    close(m.predict(f64([[0], [5]])), [0, 0], 1e-12);
  });

  it("OneClassSVM with nu = 1 keeps a finite offset", () => {
    const m = new OneClassSVM({ nu: 1 }).fit(f64([[0], [1], [2], [3]]));
    expect(Number.isFinite(m.offset)).toBe(true);
    const scores = m.decisionFunction(f64([[0], [1], [2], [3]])).toArray() as number[];
    expect(Math.max(...scores)).toBeCloseTo(0, 12);
  });

  it("NuSVC explains a nu that is too small for overlapping classes", () => {
    // Interleaved classes: no separating hyperplane exists once nu is this small, and
    // scikit-learn returns dual coefficients around 1e7 after hitting its iteration limit.
    const Xo = f64([
      [1.0, -1.7535],
      [1.0, 0.9718],
      [1.0, -0.3357],
      [1.0, 2.1552],
      [1.0, -0.6229],
      [1.0, -1.2884],
      [1.0, -1.6296],
      [1.0, 2.0056],
      [1.0, -0.4691],
      [1.0, 1.6205],
    ]);
    const yo = tensor([0, 0, 1, 0, 1, 1, 1, 1, 1, 1]);
    expect(() => new NuSVC({ nu: 0.223, kernel: "linear" }).fit(Xo, yo)).toThrow(/increase nu/);
  });

  it("getParams returns a copy of classWeight", () => {
    const m = new SVC({ classWeight: { 0: 1, 1: 2 } });
    const p = m.getParams()["classWeight"] as Record<number, number>;
    p[0] = 99;
    expect((m.getParams()["classWeight"] as Record<number, number>)[0]).toBe(1);
    const l = new LinearSVC({ classWeight: { 0: 1, 1: 2 } });
    (l.getParams()["classWeight"] as Record<number, number>)[1] = 99;
    expect((l.getParams()["classWeight"] as Record<number, number>)[1]).toBe(2);
  });
});
