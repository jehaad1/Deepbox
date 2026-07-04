import { describe, expect, it } from "vitest";
import { GaussianNB } from "../src/ml/naive_bayes";
import { BernoulliNB } from "../src/ml/naive_bayes/BernoulliNB";
import { ComplementNB } from "../src/ml/naive_bayes/ComplementNB";
import { MultinomialNB } from "../src/ml/naive_bayes/MultinomialNB";
import { tensor } from "../src/ndarray";

// ---- GaussianNB extended branch coverage ----

describe("GaussianNB extended branches", () => {
  it("constructor with custom varSmoothing", () => {
    const nb = new GaussianNB({ varSmoothing: 0.01 });
    expect(nb.getParams().varSmoothing).toBe(0.01);
  });

  it("constructor rejects negative varSmoothing", () => {
    expect(() => new GaussianNB({ varSmoothing: -1 })).toThrow();
  });

  it("constructor rejects NaN varSmoothing", () => {
    expect(() => new GaussianNB({ varSmoothing: NaN })).toThrow();
  });

  it("classes returns undefined before fit", () => {
    const nb = new GaussianNB();
    expect(nb.classes).toBeUndefined();
  });

  it("getParams returns correct defaults", () => {
    const nb = new GaussianNB();
    expect(nb.getParams()).toEqual({ varSmoothing: 1e-9 });
  });

  it("setParams updates varSmoothing", () => {
    const nb = new GaussianNB();
    nb.setParams({ varSmoothing: 0.1 });
    expect(nb.getParams().varSmoothing).toBe(0.1);
  });

  it("setParams rejects invalid varSmoothing", () => {
    const nb = new GaussianNB();
    expect(() => nb.setParams({ varSmoothing: -1 })).toThrow();
    expect(() => nb.setParams({ varSmoothing: "bad" as unknown as number })).toThrow();
  });

  it("setParams rejects unknown parameter", () => {
    const nb = new GaussianNB();
    expect(() => nb.setParams({ unknownParam: 42 })).toThrow(/unknown/i);
  });

  it("score rejects non-1D y", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const nb = new GaussianNB();
    nb.fit(X, y);
    expect(() => nb.score(X, tensor([[0, 1]]))).toThrow(/1-dimensional/i);
  });

  it("score rejects non-finite y values", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const nb = new GaussianNB();
    nb.fit(X, y);
    expect(() => nb.score(X, tensor([0, NaN, 1, 1]))).toThrow(/non-finite/i);
  });

  it("multiclass classification", () => {
    const X = tensor([
      [1, 0],
      [1, 1],
      [0, 2],
      [5, 0],
      [5, 1],
      [5, 2],
      [10, 0],
      [10, 1],
      [10, 2],
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]);
    const nb = new GaussianNB();
    nb.fit(X, y);
    const preds = nb.predict(
      tensor([
        [1, 1],
        [5, 1],
        [10, 1],
      ])
    );
    expect(preds.toArray()).toEqual([0, 1, 2]);
    const classes = nb.classes;
    expect(classes?.toArray()).toEqual([0, 1, 2]);
  });

  it("predictProba sums to 1 for each sample", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [4, 5],
      [5, 6],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const nb = new GaussianNB();
    nb.fit(X, y);
    const proba = nb.predictProba(
      tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ])
    );
    const data = proba.toArray() as number[][];
    for (const row of data) {
      const sum = (row as number[]).reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("score computes accuracy", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [4, 5],
      [5, 6],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const nb = new GaussianNB();
    nb.fit(X, y);
    const score = nb.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("zero variance with varSmoothing=0 throws", () => {
    const X = tensor([
      [1, 1],
      [1, 1],
      [2, 2],
      [2, 2],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const nb = new GaussianNB({ varSmoothing: 0 });
    expect(() => nb.fit(X, y)).toThrow(/zero variance/i);
  });
});

// ---- BernoulliNB extended branch coverage ----

describe("BernoulliNB extended branches", () => {
  const X = tensor([
    [1, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
  ]);
  const y = tensor([0, 1, 0, 1]);

  it("fit and predict binary features", () => {
    const clf = new BernoulliNB();
    clf.fit(X, y);
    const preds = clf.predict(tensor([[1, 0, 1]]));
    expect(preds.shape).toEqual([1]);
  });

  it("predictProba sums to 1", () => {
    const clf = new BernoulliNB();
    clf.fit(X, y);
    const proba = clf.predictProba(
      tensor([
        [1, 0, 1],
        [0, 1, 0],
      ])
    );
    const data = proba.toArray() as number[][];
    for (const row of data) {
      const sum = (row as number[]).reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("score computes accuracy", () => {
    const clf = new BernoulliNB();
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("classes accessor", () => {
    const clf = new BernoulliNB();
    expect(clf.classes).toBeUndefined();
    clf.fit(X, y);
    expect(clf.classes?.toArray()).toEqual([0, 1]);
  });

  it("constructor rejects invalid alpha", () => {
    expect(() => new BernoulliNB({ alpha: -1 })).toThrow();
    expect(() => new BernoulliNB({ alpha: NaN })).toThrow();
  });

  it("throws when predicting before fit", () => {
    const clf = new BernoulliNB();
    expect(() => clf.predict(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
    expect(() => clf.predictProba(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
  });

  it("fitPrior=false uses uniform priors", () => {
    const clf = new BernoulliNB({ fitPrior: false });
    clf.fit(X, y);
    const preds = clf.predict(tensor([[1, 0, 1]]));
    expect(preds.shape).toEqual([1]);
  });

  it("binarize=null assumes pre-binarized data", () => {
    const clf = new BernoulliNB({ binarize: null });
    clf.fit(X, y);
    const preds = clf.predict(tensor([[1, 0, 1]]));
    expect(preds.shape).toEqual([1]);
  });

  it("binarize with custom threshold", () => {
    const Xcont = tensor([
      [0.5, 0.1, 0.9],
      [0.1, 0.7, 0.8],
      [0.6, 0.2, 0.3],
      [0.2, 0.8, 0.1],
    ]);
    const clf = new BernoulliNB({ binarize: 0.5 });
    clf.fit(Xcont, y);
    const preds = clf.predict(tensor([[0.6, 0.1, 0.9]]));
    expect(preds.shape).toEqual([1]);
  });

  it("getParams returns correct values", () => {
    const clf = new BernoulliNB({ alpha: 0.5, binarize: 0.5, fitPrior: false });
    expect(clf.getParams()).toEqual({ alpha: 0.5, binarize: 0.5, fitPrior: false });
  });

  it("setParams updates alpha", () => {
    const clf = new BernoulliNB();
    clf.setParams({ alpha: 2.0 });
    expect(clf.getParams().alpha).toBe(2.0);
  });

  it("setParams updates binarize", () => {
    const clf = new BernoulliNB();
    clf.setParams({ binarize: 0.5 });
    expect(clf.getParams().binarize).toBe(0.5);
  });

  it("setParams updates binarize to null", () => {
    const clf = new BernoulliNB();
    clf.setParams({ binarize: null });
    expect(clf.getParams().binarize).toBeNull();
  });

  it("setParams updates fitPrior", () => {
    const clf = new BernoulliNB();
    clf.setParams({ fitPrior: false });
    expect(clf.getParams().fitPrior).toBe(false);
  });

  it("setParams rejects invalid alpha", () => {
    const clf = new BernoulliNB();
    expect(() => clf.setParams({ alpha: -1 })).toThrow();
  });

  it("setParams rejects invalid binarize", () => {
    const clf = new BernoulliNB();
    expect(() => clf.setParams({ binarize: "bad" as unknown as number })).toThrow();
  });

  it("setParams rejects invalid fitPrior", () => {
    const clf = new BernoulliNB();
    expect(() => clf.setParams({ fitPrior: 42 as unknown as boolean })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const clf = new BernoulliNB();
    expect(() => clf.setParams({ bad: 1 })).toThrow(/unknown/i);
  });

  it("score rejects non-1D y", () => {
    const clf = new BernoulliNB();
    clf.fit(X, y);
    expect(() => clf.score(X, tensor([[0, 1]]))).toThrow(/1-dimensional/i);
  });

  it("score rejects non-finite y", () => {
    const clf = new BernoulliNB();
    clf.fit(X, y);
    expect(() => clf.score(X, tensor([0, NaN, 1, 1]))).toThrow(/non-finite/i);
  });
});

// ---- MultinomialNB extended branch coverage ----

describe("MultinomialNB extended branches", () => {
  const X = tensor([
    [3, 0, 1],
    [0, 2, 1],
    [1, 0, 3],
    [0, 3, 0],
  ]);
  const y = tensor([0, 1, 0, 1]);

  it("fit and predict word counts", () => {
    const clf = new MultinomialNB();
    clf.fit(X, y);
    const preds = clf.predict(tensor([[2, 0, 1]]));
    expect(preds.shape).toEqual([1]);
  });

  it("predictProba sums to 1", () => {
    const clf = new MultinomialNB();
    clf.fit(X, y);
    const proba = clf.predictProba(
      tensor([
        [2, 0, 1],
        [0, 3, 0],
      ])
    );
    const data = proba.toArray() as number[][];
    for (const row of data) {
      const sum = (row as number[]).reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("score computes accuracy", () => {
    const clf = new MultinomialNB();
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("rejects negative features", () => {
    const clf = new MultinomialNB();
    expect(() =>
      clf.fit(
        tensor([
          [-1, 0],
          [0, 1],
        ]),
        tensor([0, 1])
      )
    ).toThrow(/non-negative/i);
  });

  it("constructor rejects invalid alpha", () => {
    expect(() => new MultinomialNB({ alpha: -1 })).toThrow();
    expect(() => new MultinomialNB({ alpha: NaN })).toThrow();
  });

  it("classes accessor", () => {
    const clf = new MultinomialNB();
    expect(clf.classes).toBeUndefined();
    clf.fit(X, y);
    expect(clf.classes?.toArray()).toEqual([0, 1]);
  });

  it("throws when predicting before fit", () => {
    const clf = new MultinomialNB();
    expect(() => clf.predict(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
    expect(() => clf.predictProba(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
  });

  it("fitPrior=false uses uniform priors", () => {
    const clf = new MultinomialNB({ fitPrior: false });
    clf.fit(X, y);
    const preds = clf.predict(tensor([[2, 0, 1]]));
    expect(preds.shape).toEqual([1]);
  });

  it("getParams returns correct values", () => {
    const clf = new MultinomialNB({ alpha: 0.5, fitPrior: false });
    expect(clf.getParams()).toEqual({ alpha: 0.5, fitPrior: false });
  });

  it("setParams updates alpha", () => {
    const clf = new MultinomialNB();
    clf.setParams({ alpha: 2.0 });
    expect(clf.getParams().alpha).toBe(2.0);
  });

  it("setParams updates fitPrior", () => {
    const clf = new MultinomialNB();
    clf.setParams({ fitPrior: false });
    expect(clf.getParams().fitPrior).toBe(false);
  });

  it("setParams rejects invalid alpha", () => {
    const clf = new MultinomialNB();
    expect(() => clf.setParams({ alpha: -1 })).toThrow();
  });

  it("setParams rejects invalid fitPrior", () => {
    const clf = new MultinomialNB();
    expect(() => clf.setParams({ fitPrior: 42 as unknown as boolean })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const clf = new MultinomialNB();
    expect(() => clf.setParams({ bad: 1 })).toThrow(/unknown/i);
  });

  it("score rejects non-1D y", () => {
    const clf = new MultinomialNB();
    clf.fit(X, y);
    expect(() => clf.score(X, tensor([[0, 1]]))).toThrow(/1-dimensional/i);
  });

  it("score rejects non-finite y", () => {
    const clf = new MultinomialNB();
    clf.fit(X, y);
    expect(() => clf.score(X, tensor([0, NaN, 1, 1]))).toThrow(/non-finite/i);
  });
});

// ---- ComplementNB extended branch coverage ----

describe("ComplementNB extended branches", () => {
  const X = tensor([
    [3, 0, 1],
    [0, 2, 1],
    [1, 0, 3],
    [0, 3, 0],
  ]);
  const y = tensor([0, 1, 0, 1]);

  it("fit and predict", () => {
    const clf = new ComplementNB();
    clf.fit(X, y);
    const preds = clf.predict(tensor([[2, 0, 1]]));
    expect(preds.shape).toEqual([1]);
  });

  it("predictProba sums to 1", () => {
    const clf = new ComplementNB();
    clf.fit(X, y);
    const proba = clf.predictProba(
      tensor([
        [2, 0, 1],
        [0, 3, 0],
      ])
    );
    const data = proba.toArray() as number[][];
    for (const row of data) {
      const sum = (row as number[]).reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("score computes accuracy", () => {
    const clf = new ComplementNB();
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("rejects negative features", () => {
    const clf = new ComplementNB();
    expect(() =>
      clf.fit(
        tensor([
          [-1, 0],
          [0, 1],
        ]),
        tensor([0, 1])
      )
    ).toThrow(/non-negative/i);
  });

  it("constructor rejects invalid alpha", () => {
    expect(() => new ComplementNB({ alpha: -1 })).toThrow();
  });

  it("classes accessor", () => {
    const clf = new ComplementNB();
    expect(clf.classes).toBeUndefined();
    clf.fit(X, y);
    expect(clf.classes?.toArray()).toEqual([0, 1]);
  });

  it("throws when predicting before fit", () => {
    const clf = new ComplementNB();
    expect(() => clf.predict(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
    expect(() => clf.predictProba(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
  });

  it("norm=true normalizes weights", () => {
    const clf = new ComplementNB({ norm: true });
    clf.fit(X, y);
    const preds = clf.predict(tensor([[2, 0, 1]]));
    expect(preds.shape).toEqual([1]);
  });

  it("fitPrior=false uses uniform priors", () => {
    const clf = new ComplementNB({ fitPrior: false });
    clf.fit(X, y);
    const preds = clf.predict(tensor([[2, 0, 1]]));
    expect(preds.shape).toEqual([1]);
  });

  it("getParams returns correct values", () => {
    const clf = new ComplementNB({ alpha: 0.5, norm: true, fitPrior: false });
    const p = clf.getParams();
    expect(p.alpha).toBe(0.5);
    expect(p.norm).toBe(true);
    expect(p.fitPrior).toBe(false);
  });

  it("setParams updates alpha", () => {
    const clf = new ComplementNB();
    clf.setParams({ alpha: 2.0 });
    expect(clf.getParams().alpha).toBe(2.0);
  });

  it("setParams rejects invalid alpha", () => {
    const clf = new ComplementNB();
    expect(() => clf.setParams({ alpha: -1 })).toThrow();
  });

  it("setParams rejects unknown param", () => {
    const clf = new ComplementNB();
    expect(() => clf.setParams({ bad: 1 })).toThrow(/unknown/i);
  });

  it("score rejects non-1D y", () => {
    const clf = new ComplementNB();
    clf.fit(X, y);
    expect(() => clf.score(X, tensor([[0, 1]]))).toThrow(/1-dimensional/i);
  });

  it("score rejects non-finite y", () => {
    const clf = new ComplementNB();
    clf.fit(X, y);
    expect(() => clf.score(X, tensor([0, NaN, 1, 1]))).toThrow(/non-finite/i);
  });

  it("multiclass classification", () => {
    const Xm = tensor([
      [3, 0, 0],
      [2, 0, 0],
      [0, 3, 0],
      [0, 2, 0],
      [0, 0, 3],
      [0, 0, 2],
    ]);
    const ym = tensor([0, 0, 1, 1, 2, 2]);
    const clf = new ComplementNB();
    clf.fit(Xm, ym);
    const classes = clf.classes;
    expect(classes?.toArray()).toEqual([0, 1, 2]);
  });
});
