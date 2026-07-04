import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../src/dataframe";
import { eigh } from "../src/linalg";
import {
  BallTree,
  FastICA,
  GaussianProcessClassifier,
  LocalOutlierFactor,
  PCA,
  Ridge,
} from "../src/ml";
import * as db from "../src/ndarray";
import { beta, binom, chi2, f as fDist, norm } from "../src/stats/distributions";

/**
 * Cross-module regression tests for the 2026-07-02 correctness audit. Each
 * asserts a value/behavior against an external reference (NumPy/SciPy/sklearn/
 * PyTorch) — the class of check the original suite was missing.
 */

function flat(t: { toArray(): unknown }): number[] {
  const o: number[] = [];
  const rec = (a: unknown) => (Array.isArray(a) ? a.forEach(rec) : o.push(a as number));
  rec(t.toArray());
  return o;
}

describe("audit regression — ndarray", () => {
  it("non-power-of-2 FFT matches NumPy (not the conjugate)", () => {
    const im = flat(db.fft(db.tensor([0, 1, 0], { dtype: "float64" })).imag);
    // numpy.fft.fft([0,1,0]).imag == [0, -0.8660, +0.8660]
    expect(im[0]).toBeCloseTo(0, 6);
    expect(im[1]).toBeCloseTo(-0.8660254, 5);
    expect(im[2]).toBeCloseTo(0.8660254, 5);
  });

  it("allclose rejects NaN and mismatched infinities", () => {
    expect(db.allclose(db.tensor([NaN]), db.tensor([5]))).toBe(false);
    expect(db.allclose(db.tensor([Infinity]), db.tensor([-Infinity]))).toBe(false);
    expect(db.allclose(db.tensor([Infinity]), db.tensor([Infinity]))).toBe(true);
  });
});

describe("audit regression — linalg", () => {
  it("eigh is accurate for a 20x20 symmetric matrix", () => {
    const n = 20;
    let seed = 42;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    const A: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++)
      for (let j = i; j < n; j++) {
        const v = rng();
        A[i]![j] = v;
        A[j]![i] = v;
      }
    const [wT, VT] = eigh(db.tensor(A, { dtype: "float64" }));
    const V = VT.toArray() as number[][];
    const w = wT.toArray() as number[];
    let maxRes = 0;
    for (let k = 0; k < n; k++)
      for (let i = 0; i < n; i++) {
        let av = 0;
        for (let j = 0; j < n; j++) av += A[i]![j]! * V[j]![k]!;
        maxRes = Math.max(maxRes, Math.abs(av - w[k]! * V[i]![k]!));
      }
    expect(maxRes).toBeLessThan(1e-9);
  });
});

describe("audit regression — stats", () => {
  it("F.ppf lower tail matches scipy", () => {
    expect(fDist(5, 10).ppf(0.025)).toBeCloseTo(0.15108, 4);
  });
  it("beta.ppf tails match scipy", () => {
    expect(beta(2, 5).ppf(0.001)).toBeCloseTo(0.0082555, 5);
  });
  it("chi2 tiny quantile is not clamped", () => {
    expect(chi2(0.5).ppf(0.001)).toBeLessThan(1e-10);
  });
  it("binom.pmf handles degenerate p", () => {
    expect(binom(5, 0).pmf(0)).toBe(1);
    expect(binom(5, 1).pmf(5)).toBe(1);
  });
  it("norm.sf resolves the far upper tail", () => {
    expect(norm().sf(10)).toBeGreaterThan(0);
    expect(norm().sf(10)).toBeLessThan(1e-20);
  });
});

describe("audit regression — ml", () => {
  it("BallTree.query returns the true nearest neighbors", () => {
    let seed = 1;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const n = 120;
    const d = 3;
    const flatData = new Float64Array(n * d);
    const pts: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < d; j++) {
        const v = rng();
        flatData[i * d + j] = v;
        row.push(v);
      }
      pts.push(row);
    }
    const bt = new BallTree(flatData, n, d, 40);
    const q = pts[7]!;
    const brute = pts
      .map((p, i) => ({ i, dd: p.reduce((s, v, j) => s + (v - q[j]!) ** 2, 0) }))
      .sort((a, b) => a.dd - b.dd)
      .slice(0, 5)
      .map((x) => x.i);
    expect(bt.query(new Float64Array(q), 5).indices).toEqual(brute);
  });

  it("Ridge sag converges (does not diverge to Infinity)", () => {
    let seed = 7;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    const w = [1, -2, 0.5, 3, -1];
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 40; i++) {
      const r: number[] = [];
      let t = 2;
      for (let j = 0; j < 5; j++) {
        const v = rng();
        r.push(v);
        t += v * w[j]!;
      }
      X.push(r);
      y.push(t);
    }
    const Xt = db.tensor(X, { dtype: "float64" });
    const yt = db.tensor(y, { dtype: "float64" });
    const sag = flat(new Ridge({ alpha: 0.5, solver: "sag", maxIter: 5000 }).fit(Xt, yt).coef);
    const chol = flat(new Ridge({ alpha: 0.5, solver: "cholesky" }).fit(Xt, yt).coef);
    expect(sag.every((v) => Number.isFinite(v))).toBe(true);
    for (let i = 0; i < sag.length; i++) expect(sag[i]).toBeCloseTo(chol[i]!, 2);
  });

  it("GaussianProcessClassifier separates linearly separable blobs", () => {
    let seed = 3;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 30; i++) {
      X.push([rng() - 3, rng() - 3]);
      y.push(0);
    }
    for (let i = 0; i < 30; i++) {
      X.push([rng() + 3, rng() + 3]);
      y.push(1);
    }
    const gpc = new GaussianProcessClassifier({ lengthScale: 1 }).fit(
      db.tensor(X, { dtype: "float64" }),
      db.tensor(y, { dtype: "int32" })
    );
    expect(
      gpc.score(db.tensor(X, { dtype: "float64" }), db.tensor(y, { dtype: "int32" }))
    ).toBeGreaterThan(0.9);
  });

  it("FastICA recovers independent sources", () => {
    let seed = 42;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const n = 500;
    const S: number[][] = [];
    for (let i = 0; i < n; i++) S.push([Math.sign(Math.sin(i * 0.3)), rng() * 2 - 1]);
    const A = [
      [1, 1],
      [0.5, 2],
    ];
    const X = S.map((s) => [
      A[0]![0]! * s[0]! + A[0]![1]! * s[1]!,
      A[1]![0]! * s[0]! + A[1]![1]! * s[1]!,
    ]);
    const rec = new FastICA({ nComponents: 2, randomState: 1 })
      .fit(db.tensor(X, { dtype: "float64" }))
      .transform(db.tensor(X, { dtype: "float64" }))
      .toArray() as number[][];
    const corr = (a: number[], b: number[]) => {
      const ma = a.reduce((x, y) => x + y, 0) / a.length;
      const mb = b.reduce((x, y) => x + y, 0) / b.length;
      let c = 0;
      let va = 0;
      let vb = 0;
      for (let i = 0; i < a.length; i++) {
        c += (a[i]! - ma) * (b[i]! - mb);
        va += (a[i]! - ma) ** 2;
        vb += (b[i]! - mb) ** 2;
      }
      return Math.abs(c / Math.sqrt(va * vb));
    };
    const r0 = rec.map((r) => r[0]!);
    const s0 = S.map((s) => s[0]!);
    const s1 = S.map((s) => s[1]!);
    expect(Math.max(corr(r0, s0), corr(r0, s1))).toBeGreaterThan(0.9);
  });

  it("PCA randomized explainedVarianceRatio matches the full solver", () => {
    let seed = 1;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    const X: number[][] = [];
    for (let i = 0; i < 10; i++) X.push([rng(), rng() * 0.3]);
    const full = flat(
      new PCA({ nComponents: 1, svdSolver: "full" }).fit(db.tensor(X, { dtype: "float64" }))
        .explainedVarianceRatio
    );
    const rand = flat(
      new PCA({ nComponents: 1, svdSolver: "randomized" }).fit(db.tensor(X, { dtype: "float64" }))
        .explainedVarianceRatio
    );
    expect(rand[0]).toBeCloseTo(full[0]!, 4);
    expect(rand[0]).toBeLessThan(1); // NOT spuriously 1
  });

  it("LocalOutlierFactor novelty predict flags far points, not by row count", () => {
    const train: number[][] = [];
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) train.push([i * 10, j * 10]);
    const lof = new LocalOutlierFactor({ nNeighbors: 8 }).fit(
      db.tensor(train, { dtype: "float64" })
    );
    expect(
      flat(
        lof.predict(
          db.tensor(
            [
              [10, 10],
              [11, 10],
              [10000, 10000],
            ],
            { dtype: "float64" }
          )
        )
      )
    ).toEqual([1, 1, -1]);
  });
});

describe("audit regression — dataframe", () => {
  it("Date values hash distinctly in groupby", () => {
    const df = new DataFrame({
      d: [new Date("2024-01-01"), new Date("2025-06-30"), new Date("2024-01-01")],
      v: [1, 2, 3],
    });
    expect(df.drop_duplicates(["d"]).shape[0]).toBe(2);
  });

  it("eval('a == 3') filters instead of overwriting the column", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [3, 2, 1] });
    expect(df.eval("a == 2").get("a").data).toEqual([2]);
    expect(df.get("a").data).toEqual([1, 2, 3]); // original untouched
  });

  it("query supports column-vs-column comparison", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [3, 2, 1] });
    expect(df.query("a > b").get("a").data).toEqual([3]);
  });

  it("CSV keeps '007' string and whitespace-as-null, strips BOM", () => {
    expect(DataFrame.fromCsvString("﻿a,b\n1,2").columns).toEqual(["a", "b"]);
    const df = DataFrame.fromCsvString("x,y\n 007 ,z\n  ,q");
    expect(df.get("x").data).toEqual([" 007 ", null]);
  });

  it("datetime accessor parses date-only strings without a timezone shift", () => {
    const s = new Series(["2024-01-01"]);
    expect(s.dt.year().data).toEqual([2024]);
    expect(s.dt.day().data).toEqual([1]);
  });
});
