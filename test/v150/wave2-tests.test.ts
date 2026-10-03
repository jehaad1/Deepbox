/**
 * Wave 2 regression tests for behaviour that shared test files were adjusted
 * to in wave 1. Each block pins the new behaviour against the reference
 * library (NumPy, pandas, PyTorch, scikit-learn) so the edited assertions in
 * the older test files cannot silently drift from the intended semantics.
 */

import { describe, expect, it } from "vitest";
import { ConvergenceError } from "../../src/core";
import { DataFrame } from "../../src/dataframe";
import { cond, eig, norm } from "../../src/linalg";
import { arange, parameter, tensor } from "../../src/ndarray";
import { RMSprop } from "../../src/optim";
import { trainTestSplit } from "../../src/preprocess";

const numbers = (t: { toArray(): unknown }): number[] =>
  ([t.toArray()].flat(Infinity) as number[]).map(Number);

describe("DataFrame.interpolate('nearest') matches pandas", () => {
  it("picks the closer valid neighbour for interior gaps", () => {
    // pandas: pd.Series([10, nan, nan, 40]).interpolate(method="nearest") -> [10, 10, 40, 40]
    const df = new DataFrame({ a: [10, null, null, 40] });
    expect(Array.from(df.interpolate("nearest").get("a").data as ArrayLike<number>)).toEqual([
      10, 10, 40, 40,
    ]);
  });

  it("linear interpolation fills a longer gap evenly", () => {
    const df = new DataFrame({ a: [0, null, null, null, 4] });
    expect(Array.from(df.interpolate("linear").get("a").data as ArrayLike<number>)).toEqual([
      0, 1, 2, 3, 4,
    ]);
  });
});

describe("eig convergence on blocks that need QR sweeps", () => {
  // Real eigenvalues 1..5 (numpy.linalg.eigvals gives {1, 2, 3, 4, 5}).
  const A = () =>
    tensor([
      [4.75, -1.5, 0.75, -0.75, 1.0],
      [-0.25, 1.5, 0.75, -0.75, 1.0],
      [-1.0, 0.0, 3.0, 1.0, 0.0],
      [-1.75, -0.5, 0.25, 3.75, 3.0],
      [3.75, -1.5, 0.75, -0.75, 2.0],
    ]);

  it("throws ConvergenceError when the sweep budget is too small", () => {
    expect(() => eig(A(), { maxIter: 1, tol: 1e-12 })).toThrow(ConvergenceError);
  });

  it("converges with the default budget to eigenvalues 1..5", () => {
    const [values] = eig(A());
    const re = numbers(values).sort((x, y) => x - y);
    expect(re.length).toBe(5);
    for (let i = 0; i < 5; i++) expect(re[i]).toBeCloseTo(i + 1, 6);
  });

  it("solves a 2x2 matrix in closed form even with maxIter 1", () => {
    const [values] = eig(
      tensor([
        [4, 1],
        [2, 3],
      ]),
      { maxIter: 1 }
    );
    const re = numbers(values).sort((x, y) => x - y);
    expect(re[0]).toBeCloseTo(2, 10);
    expect(re[1]).toBeCloseTo(5, 10);
  });
});

describe("cond and norm orders match NumPy", () => {
  const B = tensor([
    [4, 1],
    [2, 3],
  ]);

  it("supports every NumPy condition-number order", () => {
    // numpy.linalg.cond(B, p) for p in [1, -1, 2, -2, inf, -inf, 'fro', 'nuc']
    const expected: [number | "fro" | "nuc", number][] = [
      [1, 3],
      [-1, 2],
      [2, 2.6180339887498945],
      [-2, 0.3819660112501052],
      [Number.POSITIVE_INFINITY, 3],
      [Number.NEGATIVE_INFINITY, 2],
      ["fro", 3],
      ["nuc", 5],
    ];
    for (const [p, value] of expected) {
      expect(cond(B, p)).toBeCloseTo(value, 10);
    }
  });

  it("rejects an invalid condition-number order like NumPy", () => {
    expect(() => cond(B, 3)).toThrow(/Unsupported norm order/);
  });

  it("flattens N-D input when no order is given and rejects 'fro' on N-D input", () => {
    // numpy.linalg.norm(np.arange(8.).reshape(2, 2, 2)) = 11.832159566199232
    const t = arange(0, 8).reshape([2, 2, 2]);
    expect(norm(t)).toBeCloseTo(11.832159566199232, 10);
    expect(() => norm(t, "fro")).toThrow(/1D or 2D/i);
  });
});

describe("RMSprop centered variant follows PyTorch", () => {
  it("matches torch.optim.RMSprop(centered=True) over three steps", () => {
    // torch float64: lr=0.01, alpha=0.9, eps=1e-3, loss = sum(w * p^2), w = [1, 3],
    // start [2, -1] -> [1.9192366953736613, -0.9197377837460131]
    const p = parameter(tensor([2, -1], { dtype: "float64" }));
    const opt = new RMSprop([p], {
      lr: 0.01,
      alpha: 0.9,
      eps: 1e-3,
      weightDecay: 0,
      momentum: 0,
      centered: true,
    });
    for (let i = 0; i < 3; i++) {
      const v = numbers(p.tensor);
      p.setGrad(tensor([2 * 1 * (v[0] ?? 0), 2 * 3 * (v[1] ?? 0)], { dtype: "float64" }));
      opt.step();
    }
    const out = numbers(p.tensor);
    expect(out[0]).toBeCloseTo(1.9192366953736613, 10);
    expect(out[1]).toBeCloseTo(-0.9197377837460131, 10);
  });
});

describe("stratified trainTestSplit tie handling", () => {
  it("keeps class totals fixed and breaks the 1.5/1.5 tie randomly, like scikit-learn", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6]]);
    const y = tensor([0, 0, 0, 1, 1, 1]);
    const trainZeros = new Set<number>();
    for (let seed = 0; seed < 40; seed++) {
      const [, , yTrain, yTest] = trainTestSplit(X, y, {
        testSize: 0.5,
        stratify: y,
        randomState: seed,
        shuffle: true,
      });
      const tr = numbers(yTrain);
      const te = numbers(yTest);
      const train0 = tr.filter((v) => v === 0).length;
      const test0 = te.filter((v) => v === 0).length;
      expect(tr.length).toBe(3);
      expect(te.length).toBe(3);
      expect(train0 + test0).toBe(3);
      expect(tr.filter((v) => v === 1).length + te.filter((v) => v === 1).length).toBe(3);
      expect([1, 2]).toContain(train0);
      trainZeros.add(train0);
    }
    // Both tie outcomes occur across seeds (a deterministic tie-break would give one).
    expect(trainZeros.size).toBe(2);
  });
});
