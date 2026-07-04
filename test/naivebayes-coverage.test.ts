import { describe, expect, it } from "vitest";
import { GaussianNB } from "../src/ml/naive_bayes";
import { BernoulliNB } from "../src/ml/naive_bayes/BernoulliNB";
import { ComplementNB } from "../src/ml/naive_bayes/ComplementNB";
import { MultinomialNB } from "../src/ml/naive_bayes/MultinomialNB";
import { tensor } from "../src/ndarray";

const X = tensor([
  [1, 2],
  [1, 3],
  [2, 1],
  [4, 5],
  [4, 6],
  [3, 5],
]);
const y = tensor([0, 0, 0, 1, 1, 1]);

describe("GaussianNB setParams", () => {
  it("sets valid varSmoothing", () => {
    const nb = new GaussianNB();
    nb.setParams({ varSmoothing: 1e-8 });
    expect(nb.getParams().varSmoothing).toBe(1e-8);
  });

  it("rejects invalid varSmoothing", () => {
    const nb = new GaussianNB();
    expect(() => nb.setParams({ varSmoothing: -1 })).toThrow(/varSmoothing/);
    expect(() => nb.setParams({ varSmoothing: "1e-9" })).toThrow(/varSmoothing/);
  });

  it("rejects unknown params", () => {
    const nb = new GaussianNB();
    expect(() => nb.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });

  it("score validates y shape", () => {
    const nb = new GaussianNB();
    nb.fit(X, y);
    expect(() => nb.score(X, tensor([[0, 1]]))).toThrow(/1-dimensional/);
  });
});

describe("BernoulliNB", () => {
  const Xbin = tensor([
    [1, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
  ]);
  const ybin = tensor([0, 1, 0, 1]);

  it("fits, predicts, and scores", () => {
    const nb = new BernoulliNB();
    nb.fit(Xbin, ybin);
    const preds = nb.predict(tensor([[1, 0, 1]]));
    expect(preds.shape).toEqual([1]);
    const score = nb.score(Xbin, ybin);
    expect(score).toBeGreaterThan(0.5);
  });

  it("predictProba sums to 1", () => {
    const nb = new BernoulliNB();
    nb.fit(Xbin, ybin);
    const proba = nb.predictProba(tensor([[1, 0, 1]]));
    const row = proba.toArray() as number[][];
    const sum = (row[0] as number[]).reduce((a: number, b: number) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it("classes getter", () => {
    const nb = new BernoulliNB();
    expect(nb.classes).toBeUndefined();
    nb.fit(Xbin, ybin);
    expect(nb.classes?.toArray()).toEqual([0, 1]);
  });

  it("works with fitPrior=false", () => {
    const nb = new BernoulliNB({ fitPrior: false });
    nb.fit(Xbin, ybin);
    const preds = nb.predict(Xbin);
    expect(preds.shape).toEqual([4]);
  });

  it("works with binarize=null", () => {
    const nb = new BernoulliNB({ binarize: null });
    nb.fit(Xbin, ybin);
    const preds = nb.predict(Xbin);
    expect(preds.shape).toEqual([4]);
  });

  it("works with custom binarize threshold", () => {
    const Xcont = tensor([
      [0.8, 0.2, 0.9],
      [0.1, 0.7, 0.6],
      [0.9, 0.1, 0.3],
      [0.2, 0.8, 0.1],
    ]);
    const nb = new BernoulliNB({ binarize: 0.5 });
    nb.fit(Xcont, ybin);
    const preds = nb.predict(Xcont);
    expect(preds.shape).toEqual([4]);
  });

  it("throws before fit", () => {
    const nb = new BernoulliNB();
    expect(() => nb.predict(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
    expect(() => nb.predictProba(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
  });

  it("validates feature count on predict", () => {
    const nb = new BernoulliNB();
    nb.fit(Xbin, ybin);
    expect(() => nb.predict(tensor([[1, 0]]))).toThrow(/features/i);
  });

  it("validates constructor params", () => {
    expect(() => new BernoulliNB({ alpha: -1 })).toThrow(/alpha/);
    expect(() => new BernoulliNB({ alpha: Infinity })).toThrow(/alpha/);
  });

  it("getParams", () => {
    const nb = new BernoulliNB({ alpha: 0.5, binarize: 0.3, fitPrior: false });
    const p = nb.getParams();
    expect(p.alpha).toBe(0.5);
    expect(p.binarize).toBe(0.3);
    expect(p.fitPrior).toBe(false);
  });

  it("score validates y", () => {
    const nb = new BernoulliNB();
    nb.fit(Xbin, ybin);
    expect(() => nb.score(Xbin, tensor([[0, 1]]))).toThrow(/1-dimensional/);
  });

  describe("setParams", () => {
    it("sets valid params", () => {
      const nb = new BernoulliNB();
      nb.setParams({ alpha: 2.0, binarize: 0.5, fitPrior: false });
      const p = nb.getParams();
      expect(p.alpha).toBe(2.0);
      expect(p.binarize).toBe(0.5);
      expect(p.fitPrior).toBe(false);
    });

    it("accepts binarize=null", () => {
      const nb = new BernoulliNB();
      nb.setParams({ binarize: null });
      expect(nb.getParams().binarize).toBeNull();
    });

    it("rejects invalid alpha", () => {
      const nb = new BernoulliNB();
      expect(() => nb.setParams({ alpha: -1 })).toThrow(/alpha/);
      expect(() => nb.setParams({ alpha: "1" })).toThrow(/alpha/);
    });

    it("rejects invalid binarize", () => {
      const nb = new BernoulliNB();
      expect(() => nb.setParams({ binarize: "0.5" })).toThrow(/binarize/);
      expect(() => nb.setParams({ binarize: true })).toThrow(/binarize/);
    });

    it("rejects invalid fitPrior", () => {
      const nb = new BernoulliNB();
      expect(() => nb.setParams({ fitPrior: 1 })).toThrow(/fitPrior/);
    });

    it("rejects unknown params", () => {
      const nb = new BernoulliNB();
      expect(() => nb.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
    });
  });
});

describe("MultinomialNB", () => {
  const Xcount = tensor([
    [3, 0, 1],
    [0, 2, 1],
    [1, 0, 3],
    [0, 3, 0],
  ]);
  const ycount = tensor([0, 1, 0, 1]);

  it("fits, predicts, and scores", () => {
    const nb = new MultinomialNB();
    nb.fit(Xcount, ycount);
    const preds = nb.predict(tensor([[2, 0, 1]]));
    expect(preds.shape).toEqual([1]);
    expect(preds.toArray()).toEqual([0]);
    const score = nb.score(Xcount, ycount);
    expect(score).toBeGreaterThan(0.5);
  });

  it("predictProba sums to 1", () => {
    const nb = new MultinomialNB();
    nb.fit(Xcount, ycount);
    const proba = nb.predictProba(tensor([[2, 0, 1]]));
    const row = proba.toArray() as number[][];
    const sum = (row[0] as number[]).reduce((a: number, b: number) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it("classes getter", () => {
    const nb = new MultinomialNB();
    expect(nb.classes).toBeUndefined();
    nb.fit(Xcount, ycount);
    expect(nb.classes?.toArray()).toEqual([0, 1]);
  });

  it("works with fitPrior=false", () => {
    const nb = new MultinomialNB({ fitPrior: false });
    nb.fit(Xcount, ycount);
    const preds = nb.predict(Xcount);
    expect(preds.shape).toEqual([4]);
  });

  it("throws before fit", () => {
    const nb = new MultinomialNB();
    expect(() => nb.predict(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
    expect(() => nb.predictProba(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
  });

  it("validates feature count", () => {
    const nb = new MultinomialNB();
    nb.fit(Xcount, ycount);
    expect(() => nb.predict(tensor([[1, 0]]))).toThrow(/features/i);
  });

  it("validates constructor params", () => {
    expect(() => new MultinomialNB({ alpha: -1 })).toThrow(/alpha/);
  });

  it("getParams", () => {
    const nb = new MultinomialNB({ alpha: 0.5, fitPrior: false });
    const p = nb.getParams();
    expect(p.alpha).toBe(0.5);
    expect(p.fitPrior).toBe(false);
  });

  it("score validates y", () => {
    const nb = new MultinomialNB();
    nb.fit(Xcount, ycount);
    expect(() => nb.score(Xcount, tensor([[0, 1]]))).toThrow(/1-dimensional/);
  });

  describe("setParams", () => {
    it("sets valid params", () => {
      const nb = new MultinomialNB();
      nb.setParams({ alpha: 2.0, fitPrior: false });
      const p = nb.getParams();
      expect(p.alpha).toBe(2.0);
      expect(p.fitPrior).toBe(false);
    });

    it("rejects invalid alpha", () => {
      const nb = new MultinomialNB();
      expect(() => nb.setParams({ alpha: -1 })).toThrow(/alpha/);
      expect(() => nb.setParams({ alpha: "1" })).toThrow(/alpha/);
    });

    it("rejects invalid fitPrior", () => {
      const nb = new MultinomialNB();
      expect(() => nb.setParams({ fitPrior: 1 })).toThrow(/fitPrior/);
    });

    it("rejects unknown params", () => {
      const nb = new MultinomialNB();
      expect(() => nb.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
    });
  });
});

describe("ComplementNB", () => {
  const Xcount = tensor([
    [3, 0, 1],
    [0, 2, 1],
    [1, 0, 3],
    [0, 3, 0],
  ]);
  const ycount = tensor([0, 1, 0, 1]);

  it("fits, predicts, and scores", () => {
    const nb = new ComplementNB();
    nb.fit(Xcount, ycount);
    const preds = nb.predict(tensor([[2, 0, 1]]));
    expect(preds.shape).toEqual([1]);
    const score = nb.score(Xcount, ycount);
    expect(score).toBeGreaterThan(0.5);
  });

  it("predictProba sums to 1", () => {
    const nb = new ComplementNB();
    nb.fit(Xcount, ycount);
    const proba = nb.predictProba(tensor([[2, 0, 1]]));
    const row = proba.toArray() as number[][];
    const sum = (row[0] as number[]).reduce((a: number, b: number) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it("classes getter", () => {
    const nb = new ComplementNB();
    expect(nb.classes).toBeUndefined();
    nb.fit(Xcount, ycount);
    expect(nb.classes?.toArray()).toEqual([0, 1]);
  });

  it("works with fitPrior=false", () => {
    const nb = new ComplementNB({ fitPrior: false });
    nb.fit(Xcount, ycount);
    const preds = nb.predict(Xcount);
    expect(preds.shape).toEqual([4]);
  });

  it("works with norm=true", () => {
    const nb = new ComplementNB({ norm: true });
    nb.fit(Xcount, ycount);
    const preds = nb.predict(Xcount);
    expect(preds.shape).toEqual([4]);
  });

  it("throws before fit", () => {
    const nb = new ComplementNB();
    expect(() => nb.predict(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
    expect(() => nb.predictProba(tensor([[1, 0, 1]]))).toThrow(/fitted/i);
  });

  it("validates feature count", () => {
    const nb = new ComplementNB();
    nb.fit(Xcount, ycount);
    expect(() => nb.predict(tensor([[1, 0]]))).toThrow(/features/i);
  });

  it("validates constructor params", () => {
    expect(() => new ComplementNB({ alpha: -1 })).toThrow(/alpha/);
  });

  it("getParams", () => {
    const nb = new ComplementNB({ alpha: 0.5, fitPrior: false, norm: true });
    const p = nb.getParams();
    expect(p.alpha).toBe(0.5);
    expect(p.fitPrior).toBe(false);
    expect(p.norm).toBe(true);
  });

  it("score validates y", () => {
    const nb = new ComplementNB();
    nb.fit(Xcount, ycount);
    expect(() => nb.score(Xcount, tensor([[0, 1]]))).toThrow(/1-dimensional/);
  });

  describe("setParams", () => {
    it("sets valid params", () => {
      const nb = new ComplementNB();
      nb.setParams({ alpha: 2.0, fitPrior: false, norm: true });
      const p = nb.getParams();
      expect(p.alpha).toBe(2.0);
      expect(p.fitPrior).toBe(false);
      expect(p.norm).toBe(true);
    });

    it("rejects invalid alpha", () => {
      const nb = new ComplementNB();
      expect(() => nb.setParams({ alpha: -1 })).toThrow(/alpha/);
    });

    it("rejects invalid fitPrior", () => {
      const nb = new ComplementNB();
      expect(() => nb.setParams({ fitPrior: 1 })).toThrow(/fitPrior/);
    });

    it("rejects invalid norm", () => {
      const nb = new ComplementNB();
      expect(() => nb.setParams({ norm: 1 })).toThrow(/norm/);
    });

    it("rejects unknown params", () => {
      const nb = new ComplementNB();
      expect(() => nb.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
    });
  });
});
