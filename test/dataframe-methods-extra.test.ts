import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe/DataFrame";
import {
  makeBiclusters,
  makeCheckerboard,
  makeLowRankMatrix,
  makeSPDMatrix,
  makeSparseUncorrelated,
} from "../src/datasets";
import { StackingClassifier, StackingRegressor } from "../src/ml/ensemble/Stacking";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../src/ml/tree/DecisionTree";
import { parameter, tensor } from "../src/ndarray";
import { LBFGS } from "../src/optim/optimizers/lbfgs";
import { getPalette, getPaletteColor, listPalettes } from "../src/plot";
import { GroupShuffleSplit, TargetEncoder } from "../src/preprocess";

// ─── DataFrame.query() ───
describe("DataFrame.query()", () => {
  it("filters with > operator", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4], b: [10, 20, 30, 40] });
    const result = df.query("a > 2");
    expect(result.shape).toEqual([2, 2]);
    expect(result.get("a").data).toEqual([3, 4]);
  });

  it("filters with == operator", () => {
    const df = new DataFrame({ x: [1, 2, 3], y: ["a", "b", "c"] });
    const result = df.query("x == 2");
    expect(result.shape).toEqual([1, 2]);
  });

  it("supports 'and' connector", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4], b: [10, 20, 30, 40] });
    const result = df.query("a > 1 and b < 40");
    expect(result.get("a").data).toEqual([2, 3]);
  });

  it("supports 'or' connector", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [10, 20, 30] });
    const result = df.query("a == 1 or a == 3");
    expect(result.get("a").data).toEqual([1, 3]);
  });

  it("handles string values in quotes", () => {
    const df = new DataFrame({
      name: ["alice", "bob", "charlie"],
      age: [25, 30, 35],
    });
    const result = df.query("name == 'bob'");
    expect(result.shape[0]).toBe(1);
    expect(result.get("age").data).toEqual([30]);
  });
});

// ─── DataFrame.memory_usage() ───
describe("DataFrame.memory_usage()", () => {
  it("returns memory estimates per column", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: ["hello", "world", "!"] });
    const mem = df.memory_usage();
    expect(mem.columns).toContain("column");
    expect(mem.columns).toContain("bytes");
    expect(mem.shape[0]).toBe(2);
    // numeric column: 3 * 8 = 24
    expect(mem.get("bytes").data[0]).toBe(24);
  });
});

// ─── DataFrame.interpolate() ───
describe("DataFrame.interpolate()", () => {
  it("linearly interpolates NaN values", () => {
    const df = new DataFrame({ a: [1, null, 3, null, 5] });
    const result = df.interpolate("linear");
    const data = result.get("a").data;
    expect(data[1]).toBe(2);
    expect(data[3]).toBe(4);
  });

  it("nearest interpolation picks closest value", () => {
    const df = new DataFrame({ a: [10, null, null, 40] });
    const result = df.interpolate("nearest");
    const data = result.get("a").data;
    // Index 1 is closer to index 0 (10); index 2 is closer to index 3 (40)
    expect(data[1]).toBe(10);
    expect(data[2]).toBe(40);
  });
});

// ─── DataFrame.pivot_table() ───
describe("DataFrame.pivot_table()", () => {
  it("aggregates with mean by default", () => {
    const df = new DataFrame({
      region: ["east", "east", "west", "west"],
      product: ["A", "B", "A", "B"],
      sales: [10, 20, 30, 40],
    });
    const result = df.pivot_table({
      index: "region",
      columns: "product",
      values: "sales",
    });
    expect(result.shape[0]).toBe(2);
    expect(result.columns.sort()).toEqual(["A", "B"]);
  });

  it("supports sum aggFunc", () => {
    const df = new DataFrame({
      g: ["a", "a", "b"],
      c: ["x", "x", "x"],
      v: [1, 2, 3],
    });
    const result = df.pivot_table({
      index: "g",
      columns: "c",
      values: "v",
      aggFunc: "sum",
    });
    // group 'a', col 'x' => 1+2=3
    expect(result.get("x").data[0]).toBe(3);
  });
});

// ─── DataFrame.expanding() ───
describe("DataFrame.expanding()", () => {
  it("computes expanding mean", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
    const result = df.expanding().mean();
    const data = result.get("a").data as number[];
    expect(data[0]).toBe(1);
    expect(data[1]).toBe(1.5);
    expect(data[2]).toBe(2);
    expect(data[4]).toBe(3);
  });

  it("computes expanding sum", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.expanding().sum();
    expect(result.get("a").data).toEqual([1, 3, 6]);
  });

  it("computes expanding min/max", () => {
    const df = new DataFrame({ a: [3, 1, 4, 1, 5] });
    expect(df.expanding().min().get("a").data).toEqual([3, 1, 1, 1, 1]);
    expect(df.expanding().max().get("a").data).toEqual([3, 3, 4, 4, 5]);
  });

  it("respects minPeriods", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.expanding(2).mean();
    expect(result.get("a").data[0]).toBeNull();
    expect(result.get("a").data[1]).toBe(1.5);
  });
});

// ─── DataFrame.ewm() ───
describe("DataFrame.ewm()", () => {
  it("computes EWM mean with span", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
    const result = df.ewm({ span: 3 }).mean();
    const data = result.get("a").data as number[];
    expect(data[0]).toBe(1);
    expect(data[1]).toBeGreaterThan(1);
    expect(data[4]).toBeGreaterThan(3);
  });

  it("computes EWM mean with alpha", () => {
    const df = new DataFrame({ a: [10, 20, 30] });
    const result = df.ewm({ alpha: 0.5 }).mean();
    const data = result.get("a").data as number[];
    expect(data[0]).toBe(10);
    expect(data[1]).toBe(15); // 0.5*20 + 0.5*10
    expect(data[2]).toBe(22.5); // 0.5*30 + 0.5*15
  });

  it("computes EWM std", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
    const result = df.ewm({ span: 3 }).std();
    const data = result.get("a").data;
    expect(data[0]).toBeNull();
    expect(typeof data[2]).toBe("number");
  });

  it("throws if no span or alpha", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(() => df.ewm({})).toThrow();
  });
});

// ─── Dataset generators ───
describe("makeSparseUncorrelated()", () => {
  it("generates correct shapes", () => {
    const [X, y] = makeSparseUncorrelated({
      nSamples: 50,
      nFeatures: 8,
      randomState: 42,
    });
    expect(X.shape).toEqual([50, 8]);
    expect(y.shape).toEqual([50]);
  });

  it("is reproducible with seed", () => {
    const [X1] = makeSparseUncorrelated({ nSamples: 10, randomState: 1 });
    const [X2] = makeSparseUncorrelated({ nSamples: 10, randomState: 1 });
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        expect(X1.at(i, j)).toBe(X2.at(i, j));
      }
    }
  });
});

describe("makeLowRankMatrix()", () => {
  it("generates correct shape", () => {
    const M = makeLowRankMatrix({
      nSamples: 30,
      nFeatures: 20,
      effectiveRank: 5,
      randomState: 42,
    });
    expect(M.shape).toEqual([30, 20]);
  });
});

describe("makeSPDMatrix()", () => {
  it("generates symmetric matrix", () => {
    const M = makeSPDMatrix({ nDim: 4, randomState: 42 });
    expect(M.shape).toEqual([4, 4]);
    // Check symmetry
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        expect(M.at(i, j)).toBeCloseTo(M.at(j, i) as number, 10);
      }
    }
  });

  it("generates positive definite matrix (all diagonal > 0)", () => {
    const M = makeSPDMatrix({ nDim: 3, randomState: 0 });
    for (let i = 0; i < 3; i++) {
      expect(M.at(i, i)).toBeGreaterThan(0);
    }
  });
});

describe("makeBiclusters()", () => {
  it("generates correct shapes", () => {
    const [X, rows, cols] = makeBiclusters({
      shape: [20, 15],
      nClusters: 3,
      randomState: 42,
    });
    expect(X.shape).toEqual([20, 15]);
    expect(rows.shape).toEqual([20]);
    expect(cols.shape).toEqual([15]);
  });
});

describe("makeCheckerboard()", () => {
  it("generates correct shapes", () => {
    const [X, rows, cols] = makeCheckerboard({
      shape: [20, 20],
      nClusters: [4, 4],
      randomState: 42,
    });
    expect(X.shape).toEqual([20, 20]);
    expect(rows.shape).toEqual([20]);
    expect(cols.shape).toEqual([20]);
  });

  it("has checkerboard pattern without noise", () => {
    const [X] = makeCheckerboard({
      shape: [10, 10],
      nClusters: [2, 2],
      noise: 0,
      randomState: 0,
    });
    // Top-left and bottom-right should be 1, top-right and bottom-left should be 0 (or vice versa)
    const topLeft = X.at(0, 0) as number;
    const topRight = X.at(0, 9) as number;
    expect(topLeft).not.toBe(topRight);
  });
});

// ─── Color palettes ───
describe("Color palettes", () => {
  it("getPalette returns known palettes", () => {
    const viridis = getPalette("viridis");
    expect(viridis.length).toBe(10);
    expect(viridis[0]).toBe("#440154");
  });

  it("getPaletteColor wraps around", () => {
    const c0 = getPaletteColor("tab10", 0);
    const c10 = getPaletteColor("tab10", 10);
    expect(c0).toBe(c10);
  });

  it("listPalettes returns all palette names", () => {
    const names = listPalettes();
    expect(names).toContain("viridis");
    expect(names).toContain("plasma");
    expect(names).toContain("tab10");
    expect(names).toContain("Set1");
  });

  it("throws for unknown palette", () => {
    expect(() => getPalette("nonexistent")).toThrow();
  });
});

// ─── TargetEncoder ───
describe("TargetEncoder", () => {
  it("encodes categories by target mean with smoothing", () => {
    const X = tensor([[0], [1], [0], [1], [0]]);
    const y = tensor([10, 20, 12, 22, 8]);
    const enc = new TargetEncoder({ smooth: 0 });
    enc.fit(X, y);
    const result = enc.transform(X);
    expect(result.shape).toEqual([5, 1]);
    // Category 0 mean = (10+12+8)/3 = 10
    // Category 1 mean = (20+22)/2 = 21
    expect(result.at(0, 0)).toBeCloseTo(10, 5);
    expect(result.at(1, 0)).toBeCloseTo(21, 5);
  });

  it("fitTransform works", () => {
    const X = tensor([[0], [1], [0]]);
    const y = tensor([5, 10, 15]);
    const enc = new TargetEncoder();
    const result = enc.fitTransform(X, y);
    expect(result.shape).toEqual([3, 1]);
  });

  it("unseen category maps to global mean", () => {
    const X = tensor([[0], [1]]);
    const y = tensor([10, 20]);
    const enc = new TargetEncoder({ smooth: 0 });
    enc.fit(X, y);
    const Xnew = tensor([[0], [1], [99]]);
    const result = enc.transform(Xnew);
    // Unseen category 99 should get global mean = 15
    expect(result.at(2, 0)).toBeCloseTo(15, 5);
  });

  it("throws if not fitted", () => {
    const enc = new TargetEncoder();
    expect(() => enc.transform(tensor([[0]]))).toThrow();
  });
});

// ─── GroupShuffleSplit ───
describe("GroupShuffleSplit", () => {
  it("splits by groups", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6]]);
    const groups = [0, 0, 1, 1, 2, 2];
    const gss = new GroupShuffleSplit({
      nSplits: 3,
      testSize: 0.33,
      randomState: 42,
    });
    const splits = gss.split(X, undefined, groups);
    expect(splits.length).toBe(3);

    for (const split of splits) {
      // Groups should not overlap between train and test
      const trainGroups = new Set(split.trainIndex.map((i) => groups[i]));
      const testGroups = new Set(split.testIndex.map((i) => groups[i]));
      for (const g of testGroups) {
        expect(trainGroups.has(g)).toBe(false);
      }
    }
  });

  it("getNSplits returns correct value", () => {
    const gss = new GroupShuffleSplit({ nSplits: 7 });
    expect(gss.getNSplits()).toBe(7);
  });

  it("throws without groups", () => {
    const X = tensor([[1], [2]]);
    const gss = new GroupShuffleSplit();
    expect(() => gss.split(X)).toThrow();
  });
});

// ─── StackingClassifier ───
describe("StackingClassifier", () => {
  it("fits and predicts", () => {
    const X = tensor([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const y = tensor([0, 1, 1, 0, 0, 1, 1, 0]);

    const clf = new StackingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
    });
    clf.fit(X, y);
    const predictions = clf.predict(X);
    expect(predictions.shape).toEqual([8]);
  });

  it("score returns a number between 0 and 1", () => {
    const X = tensor([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 0],
      [3, 0],
      [2, 1],
      [3, 1],
    ]);
    const y = tensor([0, 0, 1, 1, 0, 0, 1, 1]);

    const clf = new StackingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 2 }),
        new DecisionTreeClassifier({ maxDepth: 3 }),
      ],
    });
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("throws if not fitted", () => {
    const clf = new StackingClassifier({
      estimators: [new DecisionTreeClassifier()],
    });
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow();
  });
});

// ─── StackingRegressor ───
describe("StackingRegressor", () => {
  it("fits and predicts", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const y = tensor([2, 4, 6, 8, 10, 12, 14, 16]);

    const reg = new StackingRegressor({
      estimators: [
        new DecisionTreeRegressor({ maxDepth: 2 }),
        new DecisionTreeRegressor({ maxDepth: 3 }),
      ],
    });
    reg.fit(X, y);
    const predictions = reg.predict(X);
    expect(predictions.shape).toEqual([8]);
  });

  it("score is reasonable on linear data", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const y = tensor([2, 4, 6, 8, 10, 12, 14, 16]);

    const reg = new StackingRegressor({
      estimators: [
        new DecisionTreeRegressor({ maxDepth: 3 }),
        new DecisionTreeRegressor({ maxDepth: 5 }),
      ],
    });
    reg.fit(X, y);
    const score = reg.score(X, y);
    expect(score).toBeGreaterThan(0.5);
  });
});

// ─── LBFGS ───
describe("LBFGS optimizer", () => {
  it("minimizes a simple quadratic", () => {
    // Minimize f(x) = (x - 3)^2, gradient = 2(x - 3)
    const p = parameter(tensor([0.0], { dtype: "float64" }));
    const optimizer = new LBFGS([p], { lr: 1, maxIter: 20 });

    for (let step = 0; step < 10; step++) {
      optimizer.step(() => {
        optimizer.zeroGrad();
        const x = Number(p.tensor.data[p.tensor.offset]);
        const grad = 2 * (x - 3);
        p.setGrad(tensor([grad], { dtype: "float64" }));
        return (x - 3) ** 2;
      });
    }

    const finalX = Number(p.tensor.data[p.tensor.offset]);
    expect(finalX).toBeCloseTo(3, 1);
  });

  it("throws without closure", () => {
    const p = parameter(tensor([0.0], { dtype: "float64" }));
    const optimizer = new LBFGS([p]);
    expect(() => optimizer.step()).toThrow();
  });

  it("respects maxIter option", () => {
    const p = parameter(tensor([0.0], { dtype: "float64" }));
    const optimizer = new LBFGS([p], { maxIter: 1 });
    let evalCount = 0;
    optimizer.step(() => {
      evalCount++;
      optimizer.zeroGrad();
      const x = Number(p.tensor.data[p.tensor.offset]);
      const grad = 2 * (x - 3);
      p.setGrad(tensor([grad], { dtype: "float64" }));
      return (x - 3) ** 2;
    });
    // With maxIter=1, evaluations should be bounded
    expect(evalCount).toBeLessThanOrEqual(15);
  });
});
