import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  BernoulliNB,
  CategoricalNB,
  ComplementNB,
  GaussianNB,
  MultinomialNB,
  OneVsOneClassifier,
  OneVsRestClassifier,
} from "../../src/ml";
import type { Classifier } from "../../src/ml/base";
import { type Tensor, tensor } from "../../src/ndarray";

// Reference values computed with scikit-learn 1.8 (see the generating inputs below).
const REF = {
  mnb: {
    proba: [
      [0.8455944456349316, 0.007072010123590911, 0.14733354424147743],
      [0.5768954124843062, 0.10785928781931714, 0.31524529969637677],
      [0.2944551946097061, 0.2438008228050229, 0.461743982585271],
    ],
    logp: [
      [-0.16771541299672243, -4.95161052214458, -1.9150562540703326],
      [-0.5500942897628605, -2.226927791919654, -1.1544042140139164],
      [-1.222628428009334, -1.4114036870226014, -0.7727446917467251],
    ],
    pred: [0, 0, 2],
  },
  mnb_prior: {
    proba: [
      [0.4926727598595374, 0.021588393197466502, 0.4857388469429962],
      [0.30756780011897444, 0.09448968186009733, 0.5979425180209283],
      [0.1380415147073861, 0.18146494427212922, 0.6804935410204848],
    ],
  },
  bnb: {
    proba: [
      [0.8634064080944351, 0.13659359190556475],
      [0.6781456953642389, 0.3218543046357612],
      [0.2313601446000903, 0.7686398553999094],
    ],
    logp: [
      [-0.14686977395821765, -1.990745244325288],
      [-0.3883931242095162, -1.1336563059084774],
      [-1.4637797150115421, -0.2631327476551881],
    ],
  },
  bnb_prior: {
    proba: [
      [0.9884169884169884, 0.01158301158301158],
      [0.9660377358490566, 0.03396226415094336],
      [0.8025078369905956, 0.1974921630094046],
    ],
  },
  cnbF: {
    proba: [
      [0.6657106541238815, 0.10171075432336271, 0.23257859155275593],
      [0.5826272446967051, 0.20059275906167792, 0.21677999624161648],
      [0.3666980991641832, 0.3486065509187982, 0.28469534991701834],
    ],
  },
  cnbT: {
    proba: [
      [0.4169231290233469, 0.24861930062789203, 0.33445757034876084],
      [0.3807544271709923, 0.289344418906864, 0.3299011539221437],
      [0.3333333333333332, 0.3333333333333332, 0.3333333333333333],
    ],
  },
  cat_gap: {
    ncat: [3, 2],
    proba: [
      [0.39999999999999997, 0.6000000000000001],
      [0.39999999999999997, 0.6000000000000001],
    ],
  },
  cat_min: {
    ncat: [4, 4],
    proba: [
      [0.38461538461538447, 0.6153846153846155],
      [0.3846153846153846, 0.6153846153846154],
    ],
  },
  cat_minarr: {
    ncat: [3, 5],
    proba: [
      [0.3786982248520709, 0.6213017751479294],
      [0.3786982248520709, 0.6213017751479294],
    ],
  },
  gnb: {
    eps: 2e-5,
    proba: [
      [0.07592619090707296, 0.9240738090929274],
      [0.000271636803080469, 0.999728363196919],
      [0.009859595943402086, 0.9901404040565974],
    ],
    logp: [
      [-2.577993582864507, -0.07896333055998728],
      [-8.211044666745128, -0.00027167370303970984],
      [-4.619310090578256, -0.00990852362993877],
    ],
  },
  gnb_s: {
    eps: 10000.0,
    proba: [
      [0.44166181499933166, 0.5583381850006689],
      [0.12125233855733557, 0.8787476614426643],
      [0.5483672929687475, 0.45163270703125175],
    ],
  },
  gnb_p: {
    proba: [
      [0.5258916779959587, 0.47410832200404224],
      [0.0036546874962396334, 0.996345312503761],
      [0.11850001668804638, 0.881499983311954],
    ],
  },
  ovo: {
    dec: [
      [0.8301678250378827, -0.029428965389281646, 2.1772460891280163],
      [0.83536846359491, 2.0217370631266984, 0.15845734982752738],
      [0.8306375380689816, -0.0730888400039239, 2.1892657066015917],
      [-0.18602378628784613, 0.9979170020782601, 2.1864320213790593],
      [0.8325007035368609, -0.07999710931528571, 2.1900144693278794],
      [-0.1872328851631377, 1.004410838280339, 2.186369087444975],
    ],
    pred: [2, 1, 2, 2, 2, 2],
  },
  ovr: {
    proba: [
      [0.3105843653328972, 0.29179673803672695, 0.39761889663037586],
      [0.34990109419790244, 0.37400845045846326, 0.27609045534363436],
      [0.323810666096035, 0.22098345388840204, 0.45520588001556295],
      [0.22655733264274275, 0.33511063958492093, 0.4383320277723364],
      [0.32960463862576517, 0.2149907148931631, 0.4554046464810717],
      [0.2235155691697103, 0.3403519196456997, 0.4361325111845901],
    ],
    pred: [2, 1, 2, 2, 2, 2],
  },
} as const;

const rows = (t: Tensor): number[][] => {
  const [n = 0, k = 0] = t.shape;
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    const r: number[] = [];
    for (let j = 0; j < k; j++) r.push(Number(t.data[t.offset + i * k + j]));
    out.push(r);
  }
  return out;
};
const flat = (t: Tensor): number[] =>
  Array.from({ length: t.size }, (_, i) => Number(t.data[t.offset + i]));

function expectMatrix(actual: Tensor, expected: readonly (readonly number[])[], digits = 9): void {
  const got = rows(actual);
  expect(got.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    for (let j = 0; j < (expected[i] as readonly number[]).length; j++) {
      expect(got[i]?.[j]).toBeCloseTo((expected[i] as readonly number[])[j] as number, digits);
    }
  }
}

const X = tensor(
  [
    [3, 0, 1],
    [0, 2, 1],
    [1, 0, 3],
    [0, 3, 0],
    [2, 2, 2],
  ],
  { dtype: "float64" }
);
const y = tensor([0, 1, 0, 1, 2]);
const Xt = tensor(
  [
    [2, 0, 1],
    [0, 1, 5],
    [1, 1, 1],
  ],
  { dtype: "float64" }
);

describe("MultinomialNB (reference: scikit-learn)", () => {
  it("matches predict_proba and predict_log_proba", () => {
    const clf = new MultinomialNB({ alpha: 0.5 }).fit(X, y);
    expectMatrix(clf.predictProba(Xt), REF.mnb.proba);
    expectMatrix(clf.predictLogProba(Xt), REF.mnb.logp);
    expect(flat(clf.predict(Xt))).toEqual(REF.mnb.pred);
  });

  it("honors an explicit classPrior", () => {
    const clf = new MultinomialNB({ classPrior: [0.2, 0.3, 0.5] }).fit(X, y);
    expectMatrix(clf.predictProba(Xt), REF.mnb_prior.proba);
    expect(clf.getParams().classPrior).toEqual([0.2, 0.3, 0.5]);
  });

  it("rejects a classPrior of the wrong length at fit", () => {
    expect(() => new MultinomialNB({ classPrior: [0.5, 0.5] }).fit(X, y)).toThrow(
      InvalidParameterError
    );
  });

  it("does not produce NaN when alpha is 0 and a feature is absent from a class", () => {
    const Xa = tensor([
      [2, 0],
      [3, 0],
      [0, 2],
      [0, 4],
    ]);
    const clf = new MultinomialNB({ alpha: 0 }).fit(Xa, tensor([0, 0, 1, 1]));
    const p = rows(
      clf.predictProba(
        tensor([
          [1, 0],
          [0, 1],
        ])
      )
    );
    expect(p[0]).toEqual([1, 0]);
    expect(p[1]).toEqual([0, 1]);
  });

  it("returns uniform probabilities when every class has zero likelihood (alpha = 0)", () => {
    const Xa = tensor([
      [2, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]);
    const clf = new MultinomialNB({ alpha: 0 }).fit(Xa, tensor([0, 1, 2]));
    const p = rows(clf.predictProba(tensor([[1, 1, 1]])));
    for (const v of p[0] as number[]) expect(v).toBeCloseTo(1 / 3, 12);
  });

  it("keeps the previous model intact when a refit fails validation", () => {
    const clf = new MultinomialNB().fit(X, y);
    const before = flat(clf.predict(Xt));
    expect(() => clf.fit(tensor([[-1, 2]]), tensor([0]))).toThrow(DataValidationError);
    expect(flat(clf.predict(Xt))).toEqual(before);
  });

  it("keeps non-integer labels instead of truncating them", () => {
    const clf = new MultinomialNB().fit(
      tensor([
        [3, 0],
        [0, 3],
      ]),
      tensor([0.5, 1.5])
    );
    expect(flat(clf.classes as Tensor)).toEqual([0.5, 1.5]);
    expect(
      flat(
        clf.predict(
          tensor([
            [2, 0],
            [0, 2],
          ])
        )
      )
    ).toEqual([0.5, 1.5]);
    expect(
      clf.score(
        tensor([
          [2, 0],
          [0, 2],
        ]),
        tensor([0.5, 1.5])
      )
    ).toBe(1);
  });

  it("scores int64 targets and reports a sample count mismatch", () => {
    const clf = new MultinomialNB().fit(X, y);
    const y64 = tensor([0, 1, 0, 1, 2], { dtype: "int64" });
    expect(clf.score(X, y64)).toBeGreaterThanOrEqual(0);
    expect(() => clf.score(X, tensor([0, 1, 0]))).toThrow(ShapeError);
    expect(() => clf.score(X, tensor([], { dtype: "int32" }))).toThrow(DataValidationError);
  });

  it("validates alpha, fitPrior and classPrior in setParams", () => {
    const clf = new MultinomialNB();
    expect(() => clf.setParams({ alpha: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ alpha: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ classPrior: [0.5, -0.5] })).toThrow(InvalidParameterError);
    expect(() => new MultinomialNB({ fitPrior: 1 as unknown as boolean })).toThrow(
      InvalidParameterError
    );
  });

  it("omits classPrior from getParams unless it was set", () => {
    expect(new MultinomialNB().getParams()).toEqual({ alpha: 1, fitPrior: true });
  });
});

describe("BernoulliNB (reference: scikit-learn)", () => {
  const Xb = tensor(
    [
      [1, 0, 1, 0.2],
      [0, 1, 1, 0.9],
      [1, 0, 0, 0.7],
      [0, 1, 0, 0.4],
      [1, 1, 1, 1],
    ],
    { dtype: "float64" }
  );
  const yb = tensor([0, 1, 0, 1, 1]);
  const Xbt = tensor(
    [
      [1, 0, 1, 0.8],
      [0, 0, 0, 0],
      [1, 1, 0, 0.6],
    ],
    { dtype: "float64" }
  );

  it("matches predict_proba and predict_log_proba with binarize=0.5", () => {
    const clf = new BernoulliNB({ alpha: 0.5, binarize: 0.5 }).fit(Xb, yb);
    expectMatrix(clf.predictProba(Xbt), REF.bnb.proba);
    expectMatrix(clf.predictLogProba(Xbt), REF.bnb.logp);
  });

  it("honors an explicit classPrior", () => {
    const clf = new BernoulliNB({ alpha: 0.5, binarize: 0.5, classPrior: [0.9, 0.1] }).fit(Xb, yb);
    expectMatrix(clf.predictProba(Xbt), REF.bnb_prior.proba);
  });

  it("stays finite for alpha = 0 when a feature is always present in a class", () => {
    const Xa = tensor([
      [1, 0],
      [1, 1],
      [0, 1],
      [0, 1],
    ]);
    const clf = new BernoulliNB({ alpha: 0 }).fit(Xa, tensor([0, 0, 1, 1]));
    const p = rows(
      clf.predictProba(
        tensor([
          [1, 0],
          [0, 1],
        ])
      )
    );
    expect(p[0]).toEqual([1, 0]);
    expect(p[1]).toEqual([0, 1]);
  });

  it("computes log(1 - p) without cancellation", () => {
    // 1e9 samples are not needed: a tiny alpha makes p extremely close to 1.
    const Xa = tensor([[1], [1], [0], [0]]);
    const clf = new BernoulliNB({ alpha: 1e-18 }).fit(Xa, tensor([0, 0, 1, 1]));
    const logp = rows(clf.predictLogProba(tensor([[0]])));
    // Class 0 never shows the feature as absent: its likelihood is ~alpha / 2.
    expect(logp[0]?.[0]).toBeLessThan(-30);
    expect(Number.isFinite(logp[0]?.[0])).toBe(true);
  });

  it("rejects a NaN binarize threshold", () => {
    expect(() => new BernoulliNB({ binarize: Number.NaN })).toThrow(InvalidParameterError);
  });
});

describe("CategoricalNB (reference: scikit-learn)", () => {
  const Xc = tensor([
    [0, 0],
    [2, 1],
    [0, 1],
    [2, 0],
    [0, 0],
  ]);
  const yc = tensor([0, 0, 1, 1, 1]);
  const Xct = tensor([
    [1, 0],
    [0, 1],
  ]);

  it("counts unused ordinal codes as categories (0..max)", () => {
    const clf = new CategoricalNB().fit(Xc, yc);
    expect(flat(clf.nCategories)).toEqual(REF.cat_gap.ncat);
    expectMatrix(clf.predictProba(Xct), REF.cat_gap.proba);
  });

  it("supports minCategories as a number and as an array", () => {
    const a = new CategoricalNB({ minCategories: 4, alpha: 0.5 }).fit(Xc, yc);
    expect(flat(a.nCategories)).toEqual(REF.cat_min.ncat);
    expectMatrix(a.predictProba(Xct), REF.cat_min.proba);
    const b = new CategoricalNB({ minCategories: [3, 5] }).fit(Xc, yc);
    expect(flat(b.nCategories)).toEqual(REF.cat_minarr.ncat);
    expectMatrix(b.predictProba(Xct), REF.cat_minarr.proba);
  });

  it("validates minCategories", () => {
    expect(() => new CategoricalNB({ minCategories: 0 })).toThrow(InvalidParameterError);
    expect(() => new CategoricalNB({ minCategories: [2, 2, 2] }).fit(Xc, yc)).toThrow(
      InvalidParameterError
    );
  });

  it("handles unseen and non-ordinal category values", () => {
    const Xf = tensor([
      [-1.5, 0.25],
      [-1.5, 0.75],
      [2.5, 0.25],
      [2.5, 0.75],
    ]);
    const clf = new CategoricalNB().fit(Xf, tensor([0, 0, 1, 1]));
    expect(flat(clf.nCategories)).toEqual([2, 2]);
    expect(
      flat(
        clf.predict(
          tensor([
            [-1.5, 0.25],
            [2.5, 0.75],
          ])
        )
      )
    ).toEqual([0, 1]);
    const p = rows(clf.predictProba(tensor([[100, 100]])));
    expect(p[0]?.[0]).toBeCloseTo(0.5, 12);
  });

  it("keeps the previous model after a failed refit", () => {
    const clf = new CategoricalNB().fit(Xc, yc);
    const before = flat(clf.predict(Xct));
    expect(() => clf.fit(tensor([[1, 2]]), tensor([0, 1]))).toThrow(ShapeError);
    expect(flat(clf.predict(Xct))).toEqual(before);
  });
});

describe("ComplementNB (reference: scikit-learn)", () => {
  it("matches predict_proba without norm", () => {
    expectMatrix(new ComplementNB().fit(X, y).predictProba(Xt), REF.cnbF.proba);
  });

  it("normalizes weights by their sum (L1) when norm is true", () => {
    expectMatrix(new ComplementNB({ norm: true }).fit(X, y).predictProba(Xt), REF.cnbT.proba);
  });

  it("raises a clear error when alpha is 0 and the complement is empty", () => {
    expect(() =>
      new ComplementNB({ alpha: 0 }).fit(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 0])
      )
    ).toThrow(DataValidationError);
  });

  it("uses the prior only for a single class", () => {
    const clf = new ComplementNB().fit(
      tensor([
        [1, 2],
        [3, 4],
      ]),
      tensor([7, 7])
    );
    expectMatrix(clf.predictProba(tensor([[1, 1]])), [[1]]);
    expect(flat(clf.predict(tensor([[1, 1]])))).toEqual([7]);
  });
});

describe("GaussianNB (reference: scikit-learn)", () => {
  const Xg = tensor(
    [
      [1, 200],
      [2, 300],
      [3, 400],
      [4, 500],
      [2, 100],
    ],
    { dtype: "float64" }
  );
  const yg = tensor([0, 0, 1, 1, 1]);
  const Xgt = tensor(
    [
      [2.5, 350],
      [1, 500],
      [3, 150],
    ],
    { dtype: "float64" }
  );

  it("scales varSmoothing by the largest feature variance", () => {
    const clf = new GaussianNB().fit(Xg, yg);
    expectMatrix(clf.predictProba(Xgt), REF.gnb.proba, 10);
    expectMatrix(clf.predictLogProba(Xgt), REF.gnb.logp, 9);
    const wide = new GaussianNB({ varSmoothing: 0.5 }).fit(Xg, yg);
    expectMatrix(wide.predictProba(Xgt), REF.gnb_s.proba, 10);
  });

  it("honors fixed priors", () => {
    const clf = new GaussianNB({ priors: [0.9, 0.1] }).fit(Xg, yg);
    expectMatrix(clf.predictProba(Xgt), REF.gnb_p.proba, 10);
    expect(clf.getParams().priors).toEqual([0.9, 0.1]);
    expect(() => new GaussianNB({ priors: [1] }).fit(Xg, yg)).toThrow(InvalidParameterError);
  });

  it("rejects non-finite varSmoothing in setParams", () => {
    const clf = new GaussianNB();
    expect(() => clf.setParams({ varSmoothing: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => clf.setParams({ varSmoothing: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
  });

  it("works when every feature is constant", () => {
    const clf = new GaussianNB().fit(
      tensor([
        [1, 1],
        [1, 1],
        [1, 1],
      ]),
      tensor([0, 0, 1])
    );
    const p = rows(clf.predictProba(tensor([[1, 1]])));
    expect(p[0]?.[0]).toBeCloseTo(2 / 3, 12);
  });

  it("still throws on zero variance with varSmoothing = 0", () => {
    expect(() =>
      new GaussianNB({ varSmoothing: 0 }).fit(
        tensor([
          [1, 1],
          [1, 1],
          [2, 2],
          [2, 2],
        ]),
        tensor([0, 0, 1, 1])
      )
    ).toThrow(/zero variance/i);
  });

  it("keeps non-integer labels and accepts int64 targets in score", () => {
    const clf = new GaussianNB().fit(
      tensor([[0], [0.1], [5], [5.1]]),
      tensor([0.25, 0.25, 0.75, 0.75])
    );
    expect(flat(clf.predict(tensor([[0.05], [5.05]])))).toEqual([0.25, 0.75]);
    const y64 = tensor([0, 0, 1, 1], { dtype: "int64" });
    const clf2 = new GaussianNB().fit(tensor([[0], [0.1], [5], [5.1]]), y64);
    expect(clf2.score(tensor([[0.05], [5.05]]), tensor([0, 1], { dtype: "int64" }))).toBe(1);
  });

  it("does not mutate X or y", () => {
    const Xcopy = flat(Xg);
    new GaussianNB().fit(Xg, yg);
    expect(flat(Xg)).toEqual(Xcopy);
  });
});

/**
 * Binary base estimator: class means in feature space, probabilities from a softmax of the
 * negative distances. The same model is used for the scikit-learn reference values.
 * `seen` records every tensor passed to fit.
 */
class MeanClassifier implements Classifier {
  static readonly seen: Array<{ X: Tensor; y: Tensor }> = [];
  private means: number[][] = [];
  private labels: number[] = [];
  fit(X: Tensor, y: Tensor): this {
    MeanClassifier.seen.push({ X, y });
    const [n = 0, d = 0] = X.shape;
    const yy = flat(y);
    const xx = flat(X);
    this.labels = [...new Set(yy)].sort((a, b) => a - b);
    this.means = this.labels.map((lab) => {
      const m = new Array<number>(d).fill(0);
      let cnt = 0;
      for (let i = 0; i < n; i++) {
        if (yy[i] === lab) {
          cnt++;
          for (let j = 0; j < d; j++) m[j] = (m[j] as number) + (xx[i * d + j] as number);
        }
      }
      return m.map((v) => v / cnt);
    });
    return this;
  }
  private scores(X: Tensor): number[][] {
    const [n = 0, d = 0] = X.shape;
    const xx = flat(X);
    const out: number[][] = [];
    for (let i = 0; i < n; i++) {
      const neg = this.means.map((m) => {
        let s = 0;
        for (let j = 0; j < d; j++) s += ((xx[i * d + j] as number) - (m[j] as number)) ** 2;
        return -Math.sqrt(s);
      });
      const mx = Math.max(...neg);
      const e = neg.map((v) => Math.exp(v - mx));
      const t = e.reduce((a, b) => a + b, 0);
      out.push(e.map((v) => v / t));
    }
    return out;
  }
  predictProba(X: Tensor): Tensor {
    return tensor(this.scores(X), { dtype: "float64" });
  }
  predict(X: Tensor): Tensor {
    return tensor(
      this.scores(X).map((p) => {
        let b = 0;
        for (let k = 1; k < p.length; k++) if ((p[k] as number) > (p[b] as number)) b = k;
        return this.labels[b] as number;
      })
    );
  }
  score(): number {
    return 0;
  }
  getParams(): Record<string, unknown> {
    return {};
  }
  setParams(): this {
    return this;
  }
}

const Xm = tensor(
  [
    [0, 0],
    [0.2, 0.1],
    [1, 0],
    [1.1, 0.2],
    [0.5, 1],
    [0.4, 1.2],
    [0.5, 0.4],
  ],
  { dtype: "float64" }
);
const ym = tensor([0, 0, 1, 1, 2, 2, 2]);
const Xmt = tensor(
  [
    [0.5, 0.5],
    [0.6, 0.2],
    [0.2, 0.8],
    [0.9, 0.9],
    [0, 1],
    [1, 1],
  ],
  { dtype: "float64" }
);

describe("OneVsRestClassifier (reference: scikit-learn)", () => {
  it("matches predict and predict_proba of scikit-learn", () => {
    const ovr = new OneVsRestClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym);
    expect(flat(ovr.predict(Xmt))).toEqual(REF.ovr.pred);
    expectMatrix(ovr.predictProba(Xmt), REF.ovr.proba);
    expect(ovr.predict(Xmt).dtype).toBe("int32");
  });

  it("exposes per-class confidences through decisionFunction", () => {
    const ovr = new OneVsRestClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym);
    const dec = rows(ovr.decisionFunction(Xmt));
    const proba = rows(
      new OneVsRestClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym).predictProba(Xmt)
    );
    // Raw confidences are the unnormalized positive-class probabilities.
    for (let i = 0; i < dec.length; i++) {
      const sum = (dec[i] as number[]).reduce((a, b) => a + b, 0);
      for (let c = 0; c < 3; c++) {
        expect((dec[i] as number[])[c]! / sum).toBeCloseTo((proba[i] as number[])[c] as number, 12);
      }
    }
  });

  it("keeps non-integer labels in predict, classes and score", () => {
    const labels = tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3], { dtype: "float64" });
    const ovr = new OneVsRestClassifier({ estimator: new MeanClassifier() }).fit(Xm, labels);
    expect(flat(ovr.classes)).toEqual([0.1, 0.2, 0.3]);
    expect(flat(ovr.predict(Xm))).toEqual([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3]);
    expect(ovr.score(Xm, labels)).toBe(1);
  });

  it("keeps labels that float32 cannot represent", () => {
    const big = tensor([16777216, 16777216, 16777217, 16777217, 16777219, 16777219, 16777219], {
      dtype: "int32",
    });
    const ovr = new OneVsRestClassifier({ estimator: new MeanClassifier() }).fit(Xm, big);
    expect(flat(ovr.classes)).toEqual([16777216, 16777217, 16777219]);
    expect(ovr.score(Xm, big)).toBe(1);
  });

  it("requires at least two classes", () => {
    const ovr = new OneVsRestClassifier({ estimator: new MeanClassifier() });
    expect(() => ovr.fit(Xm, tensor([1, 1, 1, 1, 1, 1, 1]))).toThrow(DataValidationError);
  });

  it("validates the estimator option", () => {
    expect(() => new OneVsRestClassifier({} as never)).toThrow(InvalidParameterError);
    expect(() => new OneVsRestClassifier(undefined as never)).toThrow(InvalidParameterError);
    expect(() => new OneVsRestClassifier({ estimator: {} as never })).toThrow(
      InvalidParameterError
    );
  });

  it("does not hide errors from predictProba behind a silent fallback", () => {
    class Broken extends MeanClassifier {
      override predictProba(): Tensor {
        throw new RangeError("boom");
      }
    }
    const ovr = new OneVsRestClassifier({ estimator: new Broken() }).fit(Xm, ym);
    expect(() => ovr.predict(Xmt)).toThrow(RangeError);
  });

  it("falls back to hard predictions when the estimator has no probabilities", () => {
    class HardOnly extends MeanClassifier {
      override predictProba = undefined as unknown as (X: Tensor) => Tensor;
    }
    const ovr = new OneVsRestClassifier({ estimator: new HardOnly() }).fit(Xm, ym);
    expect(ovr.predict(Xm).size).toBe(7);
  });

  it("returns uniform probabilities when all estimators give zero", () => {
    class Zero extends MeanClassifier {
      override predictProba(X: Tensor): Tensor {
        return tensor(Array.from({ length: X.shape[0] ?? 0 }, () => [0, 0]));
      }
    }
    const ovr = new OneVsRestClassifier({ estimator: new Zero() }).fit(Xm, ym);
    expectMatrix(
      ovr.predictProba(Xmt),
      Array.from({ length: 6 }, () => [1 / 3, 1 / 3, 1 / 3])
    );
  });

  it("is unfitted after a failed refit and reports a clear error", () => {
    const ovr = new OneVsRestClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym);
    expect(() => ovr.fit(Xm, tensor([1, 1, 1, 1, 1, 1, 1]))).toThrow(DataValidationError);
    expect(() => ovr.predict(Xmt)).toThrow(NotFittedError);
  });

  it("checks the feature count in predictProba and decisionFunction", () => {
    const ovr = new OneVsRestClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym);
    expect(() => ovr.predictProba(tensor([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => ovr.decisionFunction(tensor([[1, 2, 3]]))).toThrow(ShapeError);
  });

  it("exposes the estimator in getParams, supports setParams and clone", () => {
    const est = new MeanClassifier();
    const ovr = new OneVsRestClassifier({ estimator: est });
    expect(ovr.getParams().estimator).toBe(est);
    expect(() => ovr.setParams({ bogus: 1 })).toThrow(InvalidParameterError);
    expect(() => ovr.setParams({ estimator: 3 })).toThrow(InvalidParameterError);
    const copy = ovr.clone();
    expect(copy).not.toBe(ovr);
    expect(copy.fit(Xm, ym).predict(Xmt).size).toBe(6);
    // Rebuilding from getParams (how CalibratedClassifierCV clones) must work too.
    const rebuilt = new OneVsRestClassifier(ovr.getParams() as { estimator: Classifier });
    expect(rebuilt.fit(Xm, ym).predict(Xmt).size).toBe(6);
  });

  it("ignores a decisionFunction that takes more than the samples (private helpers)", () => {
    class PrivateHelper extends MeanClassifier {
      // Same name and runtime shape as the private helper of some estimators.
      decisionFunction(xi: number[], model: { bias: number }): number {
        return model.bias + xi.length;
      }
    }
    const ovr = new OneVsRestClassifier({ estimator: new PrivateHelper() }).fit(Xm, ym);
    expect(flat(ovr.predict(Xmt))).toEqual(REF.ovr.pred);
    const ovo = new OneVsOneClassifier({ estimator: new PrivateHelper() }).fit(Xm, ym);
    expect(flat(ovo.predict(Xmt))).toEqual(REF.ovo.pred);
  });

  it("score validates the target", () => {
    const ovr = new OneVsRestClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym);
    expect(() => ovr.score(Xm, tensor([0, 1]))).toThrow(ShapeError);
    expect(() => ovr.score(Xm, tensor([[0, 1]]))).toThrow(ShapeError);
    expect(() => ovr.score(Xm, tensor([0, 1, 2, 0, 1, 2, Number.NaN]))).toThrow(
      DataValidationError
    );
    expect(ovr.score(Xm, tensor([0, 0, 1, 1, 2, 2, 2], { dtype: "int64" }))).toBeGreaterThan(0);
  });
});

describe("OneVsOneClassifier (reference: scikit-learn)", () => {
  it("matches decision_function and predict of scikit-learn", () => {
    const ovo = new OneVsOneClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym);
    expectMatrix(ovo.decisionFunction(Xmt), REF.ovo.dec);
    expect(flat(ovo.predict(Xmt))).toEqual(REF.ovo.pred);
  });

  /** Base estimator whose answers are scripted per class pair (identified by the feature value). */
  class Scripted implements Classifier {
    private key = "";
    fit(X: Tensor): this {
      this.key = [...new Set(flat(X))].sort((a, b) => a - b).join(",");
      return this;
    }
    private table(): { pred: number; conf: number } {
      const t: Record<string, { pred: number; conf: number }> = {
        "0,1": { pred: 0, conf: 0.2 },
        "0,2": { pred: 1, conf: 0.7 },
        "1,2": { pred: 0, conf: 0.4 },
      };
      return t[this.key] as { pred: number; conf: number };
    }
    predict(X: Tensor): Tensor {
      return tensor(Array.from({ length: X.shape[0] ?? 0 }, () => this.table().pred));
    }
    predictProba(X: Tensor): Tensor {
      const c = this.table().conf;
      return tensor(
        Array.from({ length: X.shape[0] ?? 0 }, () => [1 - c, c]),
        { dtype: "float64" }
      );
    }
    score(): number {
      return 0;
    }
    getParams(): Record<string, unknown> {
      return {};
    }
    setParams(): this {
      return this;
    }
  }

  it("breaks vote ties with the pairwise confidences like scikit-learn", () => {
    const Xs = tensor([[0], [0], [1], [1], [2], [2]]);
    const ys = tensor([0, 0, 1, 1, 2, 2]);
    const ovo = new OneVsOneClassifier({ estimator: new Scripted() }).fit(Xs, ys);
    // sklearn _ovr_decision_function: votes [1, 1, 1] -> [0.8421..., 0.9444..., 1.1746...]
    expectMatrix(ovo.decisionFunction(tensor([[0]])), [[0.84210526, 0.94444444, 1.17460317]], 7);
    expect(flat(ovo.predict(tensor([[0]])))).toEqual([2]);
    // predictProba keeps plain vote shares.
    expectMatrix(ovo.predictProba(tensor([[0]])), [[1 / 3, 1 / 3, 1 / 3]], 12);
  });

  it("passes the exact float64 rows to every pairwise estimator", () => {
    MeanClassifier.seen.length = 0;
    const exact = 0.1 + 1e-9;
    const Xe = tensor(
      [
        [exact, 1],
        [2, 3],
        [4, 5],
        [6, 7],
      ],
      { dtype: "float64" }
    );
    new OneVsOneClassifier({ estimator: new MeanClassifier() }).fit(Xe, tensor([0, 1, 2, 2]));
    // Pair (0, 1): rows 0 and 1 in the original order, labels 0 and 1.
    const first = MeanClassifier.seen[0] as { X: Tensor; y: Tensor };
    expect(first.X.dtype).toBe("float64");
    expect(flat(first.X)).toEqual([exact, 1, 2, 3]);
    expect(flat(first.y)).toEqual([0, 1]);
    // Pair (1, 2): rows 1, 2, 3.
    const last = MeanClassifier.seen[2] as { X: Tensor; y: Tensor };
    expect(flat(last.X)).toEqual([2, 3, 4, 5, 6, 7]);
    expect(flat(last.y)).toEqual([0, 1, 1]);
  });

  it("keeps non-integer and float32-unsafe labels", () => {
    const labels = tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3], { dtype: "float64" });
    const ovo = new OneVsOneClassifier({ estimator: new MeanClassifier() }).fit(Xm, labels);
    expect(flat(ovo.classes)).toEqual([0.1, 0.2, 0.3]);
    expect(flat(ovo.predict(Xm))).toEqual([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3]);
    expect(ovo.score(Xm, labels)).toBe(1);
    const big = tensor([16777216, 16777216, 16777217, 16777217, 16777219, 16777219, 16777219], {
      dtype: "int32",
    });
    const ovo2 = new OneVsOneClassifier({ estimator: new MeanClassifier() }).fit(Xm, big);
    expect(ovo2.score(Xm, big)).toBe(1);
  });

  it("validates input in predictProba (feature count)", () => {
    const ovo = new OneVsOneClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym);
    expect(() => ovo.predictProba(tensor([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => ovo.decisionFunction(tensor([[1, 2, 3]]))).toThrow(ShapeError);
  });

  it("requires at least two classes and an estimator", () => {
    const ovo = new OneVsOneClassifier({ estimator: new MeanClassifier() });
    expect(() => ovo.fit(Xm, tensor([4, 4, 4, 4, 4, 4, 4]))).toThrow(DataValidationError);
    expect(() => new OneVsOneClassifier({} as never)).toThrow(InvalidParameterError);
  });

  it("is unfitted after a failed refit", () => {
    const ovo = new OneVsOneClassifier({ estimator: new MeanClassifier() }).fit(Xm, ym);
    expect(() => ovo.fit(Xm, tensor([4, 4, 4, 4, 4, 4, 4]))).toThrow(DataValidationError);
    expect(() => ovo.predictProba(Xmt)).toThrow(NotFittedError);
  });

  it("works for two classes and without confidence outputs", () => {
    class HardOnly extends MeanClassifier {
      override predictProba = undefined as unknown as (X: Tensor) => Tensor;
    }
    const ovo = new OneVsOneClassifier({ estimator: new HardOnly() });
    ovo.fit(
      tensor([
        [0, 0],
        [0.1, 0],
        [1, 1],
        [1.1, 1],
      ]),
      tensor([3, 3, 9, 9])
    );
    expect(
      flat(
        ovo.predict(
          tensor([
            [0, 0],
            [1, 1],
          ])
        )
      )
    ).toEqual([3, 9]);
  });

  it("exposes the estimator in getParams, supports setParams and clone", () => {
    const est = new MeanClassifier();
    const ovo = new OneVsOneClassifier({ estimator: est });
    expect(ovo.getParams().estimator).toBe(est);
    expect(() => ovo.setParams({ nope: 1 })).toThrow(InvalidParameterError);
    expect(ovo.clone().fit(Xm, ym).predict(Xmt).size).toBe(6);
    expect(
      new OneVsOneClassifier(ovo.getParams() as { estimator: Classifier }).fit(Xm, ym).classes.size
    ).toBe(3);
  });
});
