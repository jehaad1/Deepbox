import { describe, expect, it } from "vitest";
import { kron, matrix_power } from "../src/linalg";
import {
  brierScoreLoss,
  hingeLoss,
  meanSquaredLogError,
  smape,
  topKAccuracyScore,
  zeroOneLoss,
} from "../src/metrics";
import {
  cross_val_score,
  cross_validate,
  LogisticRegression,
  makePipeline,
  Pipeline,
} from "../src/ml";
import { parameter, tensor, zeros } from "../src/ndarray";
import {
  clip_grad_norm_,
  clip_grad_value_,
  constant_,
  kaiming_normal_,
  kaiming_uniform_,
  normal_,
  ones_,
  orthogonal_,
  sparse_,
  uniform_,
  xavier_normal_,
  xavier_uniform_,
  zeros_,
} from "../src/nn";
import { SimpleImputer, StandardScaler } from "../src/preprocess";
import { dirichlet, multinomial, multivariate_normal, setSeed } from "../src/random";

// ── nn.init tests ──

describe("nn.init: weight initialization", () => {
  it("uniform_ fills tensor within bounds", () => {
    const t = zeros([10, 10]);
    uniform_(t, -2, 2);
    for (let i = 0; i < t.size; i++) {
      const v = Number(t.data[t.offset + i]);
      expect(v).toBeGreaterThanOrEqual(-2);
      expect(v).toBeLessThanOrEqual(2);
    }
  });

  it("normal_ fills tensor with normal-distributed values", () => {
    const t = zeros([100, 10]);
    normal_(t, 0, 1);
    let sum = 0;
    for (let i = 0; i < t.size; i++) {
      sum += Number(t.data[t.offset + i]);
    }
    const mean = sum / t.size;
    expect(Math.abs(mean)).toBeLessThan(0.3); // should be close to 0
  });

  it("constant_ fills with value", () => {
    const t = zeros([2, 3]);
    constant_(t, 42);
    for (let i = 0; i < t.size; i++) {
      expect(Number(t.data[t.offset + i])).toBe(42);
    }
  });

  it("zeros_ fills with zeros", () => {
    const t = zeros([2, 3]);
    ones_(t);
    zeros_(t);
    for (let i = 0; i < t.size; i++) {
      expect(Number(t.data[t.offset + i])).toBe(0);
    }
  });

  it("ones_ fills with ones", () => {
    const t = zeros([2, 3]);
    ones_(t);
    for (let i = 0; i < t.size; i++) {
      expect(Number(t.data[t.offset + i])).toBe(1);
    }
  });

  it("xavier_uniform_ fills 2D tensor", () => {
    const t = zeros([10, 10]);
    xavier_uniform_(t);
    let allZero = true;
    for (let i = 0; i < t.size; i++) {
      if (Number(t.data[t.offset + i]) !== 0) allZero = false;
    }
    expect(allZero).toBe(false);
  });

  it("xavier_normal_ fills 2D tensor", () => {
    const t = zeros([10, 10]);
    xavier_normal_(t);
    let allZero = true;
    for (let i = 0; i < t.size; i++) {
      if (Number(t.data[t.offset + i]) !== 0) allZero = false;
    }
    expect(allZero).toBe(false);
  });

  it("kaiming_uniform_ fills 2D tensor", () => {
    const t = zeros([10, 10]);
    kaiming_uniform_(t);
    let allZero = true;
    for (let i = 0; i < t.size; i++) {
      if (Number(t.data[t.offset + i]) !== 0) allZero = false;
    }
    expect(allZero).toBe(false);
  });

  it("kaiming_normal_ fills 2D tensor", () => {
    const t = zeros([10, 10]);
    kaiming_normal_(t);
    let allZero = true;
    for (let i = 0; i < t.size; i++) {
      if (Number(t.data[t.offset + i]) !== 0) allZero = false;
    }
    expect(allZero).toBe(false);
  });

  it("orthogonal_ fills 2D tensor with orthogonal rows", () => {
    const t = zeros([5, 5]);
    orthogonal_(t);
    // Check rows are approximately unit norm
    for (let i = 0; i < 5; i++) {
      let norm = 0;
      for (let j = 0; j < 5; j++) {
        const v = Number(t.data[t.offset + i * 5 + j]);
        norm += v * v;
      }
      expect(Math.abs(Math.sqrt(norm) - 1)).toBeLessThan(0.2);
    }
  });

  it("sparse_ fills 2D tensor with sparse values", () => {
    const t = zeros([10, 10]);
    sparse_(t, 0.5);
    let zeroCount = 0;
    for (let i = 0; i < t.size; i++) {
      if (Number(t.data[t.offset + i]) === 0) zeroCount++;
    }
    expect(zeroCount).toBeGreaterThan(20); // roughly half should be zero
  });
});

// ── Gradient clipping tests ──

describe("Gradient clipping", () => {
  it("clip_grad_norm_ clips gradients by L2 norm", () => {
    const p = parameter(tensor([3.0, 4.0]));
    // Manually set grad
    const g = tensor([3.0, 4.0]);
    p.setGrad(g);

    const totalNorm = clip_grad_norm_([p], 1.0);
    expect(totalNorm).toBeCloseTo(5.0, 5); // sqrt(9+16) = 5
  });

  it("clip_grad_value_ clips gradient elements", () => {
    const p = parameter(tensor([10.0, -20.0, 0.5]));
    const g = tensor([10.0, -20.0, 0.5]);
    p.setGrad(g);

    clip_grad_value_([p], 5.0);
    expect(Number(g.data[g.offset + 0])).toBe(5.0);
    expect(Number(g.data[g.offset + 1])).toBe(-5.0);
    expect(Number(g.data[g.offset + 2])).toBe(0.5);
  });
});

// ── SimpleImputer tests ──

describe("SimpleImputer", () => {
  it("imputes missing values with mean", () => {
    const X = tensor([
      [1, 2],
      [NaN, 3],
      [7, NaN],
    ]);
    const imp = new SimpleImputer({ strategy: "mean" });
    const result = imp.fitTransform(X);
    expect(result.shape).toEqual([3, 2]);
    // col 0 mean = (1+7)/2 = 4, col 1 mean = (2+3)/2 = 2.5
    expect(Number(result.data[result.offset + 2])).toBeCloseTo(4, 5);
    expect(Number(result.data[result.offset + 5])).toBeCloseTo(2.5, 5);
  });

  it("imputes missing values with median", () => {
    const X = tensor([
      [1, 10],
      [NaN, 20],
      [5, NaN],
      [3, 30],
    ]);
    const imp = new SimpleImputer({ strategy: "median" });
    const result = imp.fitTransform(X);
    expect(result.shape).toEqual([4, 2]);
    // col 0 median of [1,5,3] = 3, col 1 median of [10,20,30] = 20
    expect(Number(result.data[result.offset + 2])).toBeCloseTo(3, 5);
    expect(Number(result.data[result.offset + 5])).toBeCloseTo(20, 5);
  });

  it("imputes missing values with constant", () => {
    const X = tensor([
      [1, NaN],
      [NaN, 3],
    ]);
    const imp = new SimpleImputer({ strategy: "constant", fillValue: -1 });
    const result = imp.fitTransform(X);
    expect(Number(result.data[result.offset + 1])).toBe(-1);
    expect(Number(result.data[result.offset + 2])).toBe(-1);
  });

  it("imputes with most_frequent", () => {
    const X = tensor([
      [1, 10],
      [2, 10],
      [1, NaN],
      [NaN, 20],
    ]);
    const imp = new SimpleImputer({ strategy: "most_frequent" });
    const result = imp.fitTransform(X);
    // col 0 most frequent = 1, col 1 most frequent = 10
    expect(Number(result.data[result.offset + 6])).toBe(1);
    expect(Number(result.data[result.offset + 5])).toBe(10);
  });

  it("most_frequent breaks ties by choosing the smallest value (sklearn parity)", () => {
    // Column has 5 and 2 each appearing twice; sklearn picks the smallest (2).
    const X = tensor([[5], [2], [5], [2], [NaN]]);
    const imp = new SimpleImputer({ strategy: "most_frequent" });
    const result = imp.fitTransform(X);
    expect(imp.statistics[0]).toBe(2);
    // The NaN at row 4 is imputed with the tie-break statistic.
    expect(Number(result.data[result.offset + 4])).toBe(2);
  });

  it("throws before fitting", () => {
    const imp = new SimpleImputer();
    expect(() => imp.transform(tensor([[1, 2]]))).toThrow(/fitted/i);
  });

  it("getParams returns strategy", () => {
    const imp = new SimpleImputer({ strategy: "median" });
    expect(imp.getParams().strategy).toBe("median");
  });
});

// ── Pipeline tests ──

describe("Pipeline", () => {
  it("chains scaler + classifier", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 1],
      [4, 2],
      [5, 3],
      [6, 1],
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1]);

    const pipe = new Pipeline([
      ["scaler", new StandardScaler()],
      ["clf", new LogisticRegression({ maxIter: 200 })],
    ]);
    pipe.fit(X, y);
    const pred = pipe.predict(X);
    expect(pred.shape).toEqual([6]);
  });

  it("makePipeline auto-generates step names", () => {
    const pipe = makePipeline(new StandardScaler(), new LogisticRegression({ maxIter: 100 }));
    expect(pipe.stepNames.length).toBe(2);
    expect(pipe.stepNames[0]).toBe("standardscaler");
  });

  it("score works through pipeline", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 1],
      [4, 2],
      [5, 3],
      [6, 1],
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1]);

    const pipe = new Pipeline([
      ["scaler", new StandardScaler()],
      ["clf", new LogisticRegression({ maxIter: 200 })],
    ]);
    pipe.fit(X, y);
    const score = pipe.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("clone() returns a fresh unfitted pipeline with independent steps", () => {
    const scaler = new StandardScaler();
    const pipe = new Pipeline([
      ["scaler", scaler],
      ["clf", new LogisticRegression()],
    ]);
    const cloned = pipe.clone();
    expect(cloned).toBeInstanceOf(Pipeline);
    // Steps must be fresh instances, not shared references with the original.
    expect(cloned.getStep("scaler")).not.toBe(scaler);
    // Cloning before fit must not throw (fresh, unfitted).
    expect(() =>
      cloned.fit(
        tensor([
          [1, 0],
          [2, 1],
          [3, 0],
          [4, 1],
        ]),
        tensor([0, 1, 0, 1])
      )
    ).not.toThrow();
  });

  it("cross_validate works on a Pipeline (clone path, no fitted-state leak)", () => {
    const X = tensor(Array.from({ length: 30 }, (_, i) => [i % 7, (i * 3) % 5, i % 2]));
    const y = tensor(Array.from({ length: 30 }, (_, i) => i % 2));
    const pipe = new Pipeline([
      ["scaler", new StandardScaler()],
      ["clf", new LogisticRegression({ maxIter: 50 })],
    ]);
    const res = cross_validate(pipe, X, y, { cv: 3 });
    expect(res.testScores.score).toHaveLength(3);
    for (const s of res.testScores.score) expect(Number.isFinite(s)).toBe(true);
    // The pipeline passed in must remain usable/refittable afterwards.
    expect(() => pipe.fit(X, y)).not.toThrow();
  });

  it("getStep returns named step", () => {
    const scaler = new StandardScaler();
    const pipe = new Pipeline([
      ["scaler", scaler],
      ["clf", new LogisticRegression()],
    ]);
    expect(pipe.getStep("scaler")).toBe(scaler);
  });

  it("throws on duplicate step names", () => {
    expect(
      () =>
        new Pipeline([
          ["a", new StandardScaler()],
          ["a", new LogisticRegression()],
        ])
    ).toThrow(/duplicate/i);
  });

  it("throws when not fitted", () => {
    const pipe = new Pipeline([
      ["scaler", new StandardScaler()],
      ["clf", new LogisticRegression()],
    ]);
    expect(() => pipe.predict(tensor([[1, 2]]))).toThrow(/fitted/i);
  });
});

// ── cross_val_score tests ──

describe("cross_val_score", () => {
  it("returns array of cv scores", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 1],
      [4, 2],
      [5, 3],
      [6, 1],
      [7, 2],
      [8, 3],
      [9, 1],
      [10, 2],
    ]);
    const y = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    const scores = cross_val_score(new LogisticRegression({ maxIter: 200 }), X, y, 3);
    expect(scores.length).toBe(3);
    for (const s of scores) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("throws for cv < 2", () => {
    const X = tensor([[1, 2]]);
    const y = tensor([0]);
    expect(() => cross_val_score(new LogisticRegression(), X, y, 1)).toThrow();
  });
});

// ── linalg: matrix_power, kron tests ──

describe("matrix_power", () => {
  it("A^0 = identity", () => {
    const A = tensor([
      [2, 3],
      [1, 4],
    ]);
    const I = matrix_power(A, 0);
    expect(I.shape).toEqual([2, 2]);
    expect(Number(I.data[I.offset + 0])).toBeCloseTo(1, 10);
    expect(Number(I.data[I.offset + 1])).toBeCloseTo(0, 10);
    expect(Number(I.data[I.offset + 2])).toBeCloseTo(0, 10);
    expect(Number(I.data[I.offset + 3])).toBeCloseTo(1, 10);
  });

  it("A^1 = A", () => {
    const A = tensor([
      [2, 3],
      [1, 4],
    ]);
    const A1 = matrix_power(A, 1);
    expect(Number(A1.data[A1.offset + 0])).toBeCloseTo(2, 10);
    expect(Number(A1.data[A1.offset + 3])).toBeCloseTo(4, 10);
  });

  it("A^2 = A*A", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const A2 = matrix_power(A, 2);
    // [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7, 10], [15, 22]]
    expect(Number(A2.data[A2.offset + 0])).toBeCloseTo(7, 10);
    expect(Number(A2.data[A2.offset + 1])).toBeCloseTo(10, 10);
    expect(Number(A2.data[A2.offset + 2])).toBeCloseTo(15, 10);
    expect(Number(A2.data[A2.offset + 3])).toBeCloseTo(22, 10);
  });

  it("A^(-1) = inv(A)", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const Ainv = matrix_power(A, -1);
    expect(Ainv.shape).toEqual([2, 2]);
    // inv([[1,2],[3,4]]) = [[-2, 1], [1.5, -0.5]]
    expect(Number(Ainv.data[Ainv.offset + 0])).toBeCloseTo(-2, 5);
    expect(Number(Ainv.data[Ainv.offset + 1])).toBeCloseTo(1, 5);
  });

  it("throws for non-square", () => {
    const A = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(() => matrix_power(A, 2)).toThrow();
  });
});

describe("kron", () => {
  it("computes Kronecker product", () => {
    const A = tensor([
      [1, 0],
      [0, 1],
    ]);
    const B = tensor([
      [1, 2],
      [3, 4],
    ]);
    const K = kron(A, B);
    expect(K.shape).toEqual([4, 4]);
    // Top-left 2x2 = 1*B, top-right = 0*B, bottom-left = 0*B, bottom-right = 1*B
    expect(Number(K.data[K.offset + 0])).toBeCloseTo(1, 10);
    expect(Number(K.data[K.offset + 1])).toBeCloseTo(2, 10);
    expect(Number(K.data[K.offset + 2])).toBeCloseTo(0, 10);
    expect(Number(K.data[K.offset + 15])).toBeCloseTo(4, 10);
  });

  it("handles rectangular matrices", () => {
    const A = tensor([[1, 2]]); // 1x2
    const B = tensor([[3], [4]]); // 2x1
    const K = kron(A, B);
    expect(K.shape).toEqual([2, 2]);
  });
});

// ── metrics tests ──

describe("smape", () => {
  it("perfect predictions give 0", () => {
    const y = tensor([1, 2, 3]);
    expect(smape(y, y)).toBeCloseTo(0, 10);
  });

  it("returns value in [0, 2]", () => {
    const yt = tensor([1, 2, 3]);
    const yp = tensor([2, 3, 4]);
    const s = smape(yt, yp);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(2);
  });
});

describe("meanSquaredLogError", () => {
  it("perfect predictions give 0", () => {
    const y = tensor([1, 2, 3]);
    expect(meanSquaredLogError(y, y)).toBeCloseTo(0, 10);
  });

  it("throws for negative values", () => {
    const yt = tensor([1, -1]);
    const yp = tensor([1, 1]);
    expect(() => meanSquaredLogError(yt, yp)).toThrow();
  });
});

describe("brierScoreLoss", () => {
  it("perfect predictions give 0", () => {
    const yt = tensor([1, 0, 1]);
    const yp = tensor([1, 0, 1]);
    expect(brierScoreLoss(yt, yp)).toBeCloseTo(0, 10);
  });

  it("worst case gives 1", () => {
    const yt = tensor([1, 0]);
    const yp = tensor([0, 1]);
    expect(brierScoreLoss(yt, yp)).toBeCloseTo(1, 10);
  });
});

describe("hingeLoss", () => {
  it("computes average hinge loss", () => {
    const yt = tensor([1, -1, 1]);
    const yd = tensor([2, -1, 0.5]);
    const loss = hingeLoss(yt, yd);
    // max(0, 1-2)=0, max(0, 1-1)=0, max(0, 1-0.5)=0.5 => mean=0.5/3
    expect(loss).toBeCloseTo(0.5 / 3, 5);
  });
});

describe("zeroOneLoss", () => {
  it("returns 0 for perfect predictions", () => {
    const y = tensor([0, 1, 2]);
    expect(zeroOneLoss(y, y)).toBe(0);
  });

  it("returns fraction of misclassifications", () => {
    const yt = tensor([0, 1, 2, 3]);
    const yp = tensor([0, 1, 1, 3]);
    expect(zeroOneLoss(yt, yp)).toBeCloseTo(0.25, 10);
  });
});

describe("topKAccuracyScore", () => {
  it("top-1 accuracy matches standard accuracy", () => {
    const yt = tensor([0, 1, 2]);
    const yp = tensor([
      [0.9, 0.05, 0.05],
      [0.1, 0.8, 0.1],
      [0.2, 0.3, 0.5],
    ]);
    expect(topKAccuracyScore(yt, yp, 1)).toBeCloseTo(1.0, 10);
  });

  it("top-2 is more lenient than top-1", () => {
    const yt = tensor([0, 1]);
    const yp = tensor([
      [0.3, 0.4, 0.3],
      [0.1, 0.2, 0.7],
    ]);
    const top1 = topKAccuracyScore(yt, yp, 1);
    const top2 = topKAccuracyScore(yt, yp, 2);
    expect(top2).toBeGreaterThanOrEqual(top1);
  });
});

// ── random distribution tests ──

describe("multinomial", () => {
  it("returns correct shape", () => {
    setSeed(42);
    const probs = tensor([0.2, 0.5, 0.3]);
    const result = multinomial(10, probs, 5);
    expect(result.shape).toEqual([5, 3]);
  });

  it("counts sum to n", () => {
    setSeed(42);
    const probs = tensor([0.5, 0.5]);
    const result = multinomial(20, probs, 3);
    for (let i = 0; i < 3; i++) {
      let rowSum = 0;
      for (let j = 0; j < 2; j++) {
        rowSum += Number(result.data[result.offset + i * 2 + j]);
      }
      expect(rowSum).toBe(20);
    }
  });
});

describe("multivariate_normal", () => {
  it("returns correct shape", () => {
    setSeed(42);
    const mean = tensor([0, 0]);
    const cov = tensor([
      [1, 0],
      [0, 1],
    ]);
    const result = multivariate_normal(mean, cov, 100);
    expect(result.shape).toEqual([100, 2]);
  });

  it("mean is approximately correct for large samples", () => {
    setSeed(42);
    const mean = tensor([5, -3]);
    const cov = tensor([
      [1, 0],
      [0, 1],
    ]);
    const result = multivariate_normal(mean, cov, 1000);
    let sum0 = 0,
      sum1 = 0;
    for (let i = 0; i < 1000; i++) {
      sum0 += Number(result.data[result.offset + i * 2]);
      sum1 += Number(result.data[result.offset + i * 2 + 1]);
    }
    expect(Math.abs(sum0 / 1000 - 5)).toBeLessThan(0.3);
    expect(Math.abs(sum1 / 1000 + 3)).toBeLessThan(0.3);
  });
});

describe("dirichlet", () => {
  it("returns correct shape", () => {
    setSeed(42);
    const alpha = tensor([1, 1, 1]);
    const result = dirichlet(alpha, 10);
    expect(result.shape).toEqual([10, 3]);
  });

  it("each row sums to approximately 1", () => {
    setSeed(42);
    const alpha = tensor([2, 3, 5]);
    const result = dirichlet(alpha, 5);
    for (let i = 0; i < 5; i++) {
      let rowSum = 0;
      for (let j = 0; j < 3; j++) {
        rowSum += Number(result.data[result.offset + i * 3 + j]);
      }
      expect(rowSum).toBeCloseTo(1, 5);
    }
  });

  it("throws for non-positive alpha", () => {
    expect(() => dirichlet(tensor([1, -1, 1]))).toThrow();
  });
});

// ── Optimizer tests (Adamax, RAdam) ──

describe("Adamax optimizer", async () => {
  const { Adamax } = await import("../src/optim");

  it("minimizes a simple quadratic", () => {
    // Minimize f(x) = x^2 where x starts at 5
    const x = parameter(tensor([5.0]));
    const opt = new Adamax([x], { lr: 0.1 });

    for (let i = 0; i < 100; i++) {
      opt.zeroGrad();
      // grad of x^2 = 2x
      const xVal = Number(x.tensor.data[x.tensor.offset]);
      x.grad!.data[x.grad!.offset] = 2 * xVal;
      opt.step();
    }

    const finalVal = Number(x.tensor.data[x.tensor.offset]);
    expect(Math.abs(finalVal)).toBeLessThan(1.0);
  });
});

describe("RAdam optimizer", async () => {
  const { RAdam } = await import("../src/optim");

  it("minimizes a simple quadratic", () => {
    const x = parameter(tensor([5.0]));
    const opt = new RAdam([x], { lr: 0.1 });

    for (let i = 0; i < 100; i++) {
      opt.zeroGrad();
      const xVal = Number(x.tensor.data[x.tensor.offset]);
      x.grad!.data[x.grad!.offset] = 2 * xVal;
      opt.step();
    }

    const finalVal = Number(x.tensor.data[x.tensor.offset]);
    expect(Math.abs(finalVal)).toBeLessThan(1.0);
  });
});
