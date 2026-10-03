/**
 * Regression tests for src/ml/decomposition and src/ml/discriminant_analysis (v1.5.0 review).
 *
 * Reference values come from scikit-learn 1.8 / SciPy 1.17 / NumPy 2.4
 * (`PCA`, `TruncatedSVD`, `NMF`, `FastICA`, `LinearDiscriminantAnalysis`,
 * `QuadraticDiscriminantAnalysis`, `scipy.special.digamma`, `sklearn.covariance.ledoit_wolf_shrinkage`).
 */
import { describe, expect, it } from "vitest";
import {
  ConvergenceError,
  catchWarnings,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../src/core";
import {
  FastICA,
  LatentDirichletAllocation,
  LinearDiscriminantAnalysis,
  NMF,
  PCA,
  QuadraticDiscriminantAnalysis,
  TruncatedSVD,
} from "../../src/ml";
import { digamma, jacobiEigenSymmetric } from "../../src/ml/decomposition";
import { type Tensor, tensor } from "../../src/ndarray";
import { setSeed } from "../../src/random";
import { Generator } from "../../src/random/Generator";

/** Float64 tensor from nested rows. */
function mat(rows: number[][]): Tensor {
  return tensor(rows, { dtype: "float64" });
}

/** Flat numbers of a contiguous tensor. */
function flat(t: Tensor): number[] {
  return Array.from(t.data as ArrayLike<number | bigint>, Number).slice(
    t.offset,
    t.offset + t.size
  );
}

function expectClose(actual: ArrayLike<number>, expected: number[], digits = 10): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, digits);
  }
}

const PCA_X = mat([
  [2.5, 2.4],
  [0.5, 0.7],
  [2.2, 2.9],
  [1.9, 2.2],
  [3.1, 3.0],
  [2.3, 2.7],
  [2.0, 1.6],
  [1.0, 1.1],
  [1.5, 1.6],
  [1.1, 0.9],
]);

/** Low-rank plus noise matrix used to compare solvers. */
function lowRank(n: number, f: number, rank: number, seed: number): Tensor {
  const g = new Generator(seed);
  const U = g.normalArray(0, 1, n * rank);
  const V = g.normalArray(0, 1, rank * f);
  const noise = g.normalArray(0, 0.1, n * f);
  const data = new Float64Array(n * f);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < f; j++) {
      let s = 0;
      for (let r = 0; r < rank; r++) s += (U[i * rank + r] ?? 0) * (V[r * f + j] ?? 0) * (rank - r);
      data[i * f + j] = s + (noise[i * f + j] ?? 0);
    }
  }
  return tensor(data).reshape([n, f]);
}

describe("PCA (v1.5.0)", () => {
  it("matches scikit-learn components, variances, singular values and noise variance", () => {
    const pca = new PCA({ nComponents: 2 }).fit(PCA_X);
    expectClose(
      flat(pca.components),
      [0.6778733985280119, 0.735178655544408, 0.735178655544408, -0.6778733985280119]
    );
    expectClose(flat(pca.explainedVariance), [1.2840277121727839, 0.04908339893832733]);
    expectClose(flat(pca.explainedVarianceRatio), [0.963181314348646, 0.03681868565135406]);
    expectClose(flat(pca.singularValues), [3.399448397836781, 0.6646432053703295]);
    expectClose(flat(pca.mean), [1.81, 1.91]);
    expect(pca.noiseVariance).toBe(0);
    expect(new PCA({ nComponents: 1 }).fit(PCA_X).noiseVariance).toBeCloseTo(
      0.04908339893832733,
      12
    );
  });

  it("signs each component so its largest entry is positive (svd_flip convention)", () => {
    const flipped = mat([
      [-2, 1],
      [2, -1.5],
      [1, 2],
      [-1, -2.5],
      [0.5, 0.3],
    ]);
    const comps = new PCA({ nComponents: 2 }).fit(flipped).components.toArray() as number[][];
    for (const row of comps) {
      const idx = row.reduce(
        (best, v, i) => (Math.abs(v) > Math.abs(row[best] ?? 0) ? i : best),
        0
      );
      expect(row[idx]).toBeGreaterThan(0);
    }
  });

  it("returns float64 outputs for float64 input and keeps float32 for float32 input", () => {
    const pca64 = new PCA({ nComponents: 1 }).fit(PCA_X);
    expect(pca64.components.dtype).toBe("float64");
    expect(pca64.transform(PCA_X).dtype).toBe("float64");
    // Previously every output was float32 (the library default), losing ~1e-7 of precision.
    expectClose(
      flat(pca64.transform(PCA_X)).slice(0, 3),
      [0.8279701862010884, -1.7775803252804288, 0.992197494414889]
    );

    const X32 = tensor([
      [2.5, 2.4],
      [0.5, 0.7],
      [2.2, 2.9],
    ]);
    expect(X32.dtype).toBe("float32");
    expect(new PCA({ nComponents: 1 }).fit(X32).transform(X32).dtype).toBe("float32");

    const Xint = tensor(
      [
        [1, 2],
        [3, 5],
        [4, 4],
      ],
      { dtype: "int32" }
    );
    expect(new PCA({ nComponents: 1 }).fit(Xint).transform(Xint).dtype).toBe("float64");
  });

  it("whitening divides by the exact component standard deviation", () => {
    const pca = new PCA({ nComponents: 2, whiten: true }).fit(PCA_X);
    const Z = pca.transform(PCA_X);
    expectClose(
      flat(Z).slice(0, 6),
      [
        0.7306804716270696, 0.7904179519115544, -1.56870772894648, -0.644814655697935,
        0.875610432898115, -1.7349533664437808,
      ]
    );
    // Unit variance (ddof = 1) in every column; the old 1e-12 offset made it 1 - 5e-13.
    const rows = Z.toArray() as number[][];
    for (let c = 0; c < 2; c++) {
      const col = rows.map((r) => r[c] as number);
      const m = col.reduce((a, b) => a + b, 0) / col.length;
      const v = col.reduce((a, b) => a + (b - m) ** 2, 0) / (col.length - 1);
      expect(v).toBeCloseTo(1, 12);
    }
    expectClose(flat(pca.inverseTransform(Z)), flat(PCA_X), 10);
  });

  it("does not amplify numerically zero-variance components when whitening", () => {
    const Xc = mat([
      [1, 5, 2],
      [2, 5, 3],
      [3, 5, 5],
      [4, 5, 4.5],
    ]);
    const pca = new PCA({ nComponents: 3, whiten: true }).fit(Xc);
    const Z = flat(pca.transform(Xc));
    expect(Z.every((v) => Number.isFinite(v) && Math.abs(v) < 10)).toBe(true);
    expect(flat(pca.explainedVarianceRatio)[2]).toBeCloseTo(0, 12);
  });

  it("randomized solver uses power iterations and matches the exact solver", () => {
    const X = lowRank(400, 60, 5, 3);
    const full = new PCA({ nComponents: 5, svdSolver: "full" }).fit(X);
    const rand = new PCA({ nComponents: 5, svdSolver: "randomized", randomState: 1 }).fit(X);
    const fv = flat(full.explainedVariance);
    const rv = flat(rand.explainedVariance);
    for (let i = 0; i < 5; i++) expect(rv[i]).toBeCloseTo(fv[i] as number, 6);
    expect(rand.noiseVariance).toBeCloseTo(full.noiseVariance, 6);
    // Same seed gives the same result.
    const again = new PCA({ nComponents: 5, svdSolver: "randomized", randomState: 1 }).fit(X);
    expect(flat(again.components)).toEqual(flat(rand.components));
  });

  it("randomized solver handles more oversampled vectors than samples", () => {
    const X = mat([
      [1, 2, 3, 4, 5, 6],
      [2, 1, 0, 5, 3, 1],
      [0, 4, 2, 1, 1, 7],
    ]);
    const full = new PCA({ nComponents: 2, svdSolver: "full" }).fit(X);
    const rand = new PCA({ nComponents: 2, svdSolver: "randomized", randomState: 0 }).fit(X);
    expectClose(flat(rand.explainedVariance), flat(full.explainedVariance), 8);
  });

  it("accepts a variance fraction as nComponents like scikit-learn", () => {
    expect(new PCA({ nComponents: 0.9 }).fit(PCA_X).components.shape).toEqual([1, 2]);
    expect(new PCA({ nComponents: 0.99 }).fit(PCA_X).components.shape).toEqual([2, 2]);
    expect(() => new PCA({ nComponents: 0 })).toThrow(InvalidParameterError);
    expect(() => new PCA({ nComponents: 1.5 })).toThrow(/>= 1/);
    expect(() => new PCA({ nComponents: 0.5, svdSolver: "randomized" }).fit(PCA_X)).toThrow(
      InvalidParameterError
    );
  });

  it("getParams reports every option so clone-by-params keeps them", () => {
    const options = {
      nComponents: 1,
      whiten: true,
      svdSolver: "randomized" as const,
      nOversamples: 4,
      randomState: 7,
    };
    const params = new PCA(options).getParams();
    expect(params).toEqual(options);
    const clone = new PCA(params as typeof options);
    expect(clone.getParams()).toEqual(options);
    const pca = new PCA();
    pca.setParams({ svdSolver: "full", nOversamples: 2, randomState: 3 });
    expect(pca.getParams()).toMatchObject({ svdSolver: "full", nOversamples: 2, randomState: 3 });
    expect(() => pca.setParams({ svdSolver: "arpack" })).toThrow(InvalidParameterError);
  });

  it("keeps the previous fit usable when a refit fails", () => {
    const pca = new PCA({ nComponents: 2 }).fit(PCA_X);
    const before = flat(pca.transform(PCA_X));
    // 3-feature data makes the state check meaningful: nFeaturesIn must not change on failure.
    const bad = mat([[1, 2, 3]]);
    expect(() => pca.fit(bad)).toThrow(DataValidationError);
    expect(flat(pca.transform(PCA_X))).toEqual(before);
    expect(pca.nFeaturesIn).toBe(2);
  });

  it("does not modify the input and exposes fresh tensors from getters", () => {
    const X = mat([
      [1, 2],
      [3, 5],
      [4, 4],
    ]);
    const snapshot = flat(X);
    const pca = new PCA({ nComponents: 1 }).fit(X);
    expect(flat(X)).toEqual(snapshot);
    const comps = pca.components;
    (comps.data as Float64Array)[0] = 99;
    expect(flat(pca.components)[0]).not.toBe(99);
  });

  it("validates inverseTransform input", () => {
    const pca = new PCA({ nComponents: 1 }).fit(PCA_X);
    expect(() => pca.inverseTransform(mat([[1, 2]]))).toThrow(ShapeError);
    expect(() => pca.inverseTransform(tensor([1, 2, 3]))).toThrow(ShapeError);
    expect(() => pca.inverseTransform(mat([[Number.NaN]]))).toThrow(DataValidationError);
    expectClose(
      flat(pca.inverseTransform(pca.transform(PCA_X))).slice(0, 4),
      [2.3712589640000026, 2.518706008322169, 0.6050255837456271, 0.6031608863381426]
    );
  });

  it("requires fitting before reading attributes", () => {
    const pca = new PCA();
    expect(() => pca.singularValues).toThrow(NotFittedError);
    expect(() => pca.mean).toThrow(NotFittedError);
    expect(() => pca.noiseVariance).toThrow(NotFittedError);
  });
});

describe("TruncatedSVD (v1.5.0)", () => {
  const X = mat([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12.5],
    [2, 3, 4, 5],
    [6, 7, 8, 9.5],
  ]);

  it("matches scikit-learn explained variance (variance of the transformed columns)", () => {
    const t = new TruncatedSVD({ nComponents: 2 }).fit(X);
    expectClose(flat(t.explainedVariance), [33.522778509672285, 0.6509142419122897], 8);
    expectClose(flat(t.explainedVarianceRatio), [0.9807717527698153, 0.01904371684939408], 10);
    expectClose(flat(t.singularValues), [30.8468191566309, 1.9854865641709458], 9);
    expectClose(
      flat(t.components),
      [
        0.38998525019286523, 0.4557810216392128, 0.5215767930855604, 0.6067394942313434,
        0.7605953106166021, 0.2902524990284719, -0.1800903125596558, -0.5521012041264959,
      ]
    );
    expectClose(
      flat(t.transform(X)).slice(0, 4),
      [5.2932356496533455, -1.4075754455114051, 13.189565886249273, -0.1329502736757151],
      9
    );
  });

  it("explained variance ratio is finite for a single sample", () => {
    const one = mat([[1, 2, 3]]);
    const t = new TruncatedSVD({ nComponents: 1 }).fit(one);
    // Previously divided by n_samples - 1 = 0.
    expect(Number.isFinite(flat(t.explainedVariance)[0] as number)).toBe(true);
    expect(flat(t.explainedVarianceRatio)).toEqual([0]);
  });

  it("is not affected by setParams after fit", () => {
    const t = new TruncatedSVD({ nComponents: 2 }).fit(X);
    t.setParams({ nComponents: 3 });
    expect(t.transform(X).shape).toEqual([5, 2]);
    expect(t.components.shape).toEqual([2, 4]);
    expect(t.inverseTransform(t.transform(X)).shape).toEqual([5, 4]);
  });

  it("keeps the previous fit when a refit fails and validates inverseTransform", () => {
    const t = new TruncatedSVD({ nComponents: 2 }).fit(X);
    expect(() => t.fit(mat([[1, 2, 3]]))).toThrow(InvalidParameterError);
    expect(t.nFeaturesIn).toBe(4);
    expect(() => t.inverseTransform(mat([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => t.inverseTransform(mat([[1, Number.POSITIVE_INFINITY]]))).toThrow(
      DataValidationError
    );
  });

  it("rejects invalid nComponents in setParams", () => {
    expect(() => new TruncatedSVD().setParams({ nComponents: 0 })).toThrow(InvalidParameterError);
  });
});

describe("NMF (v1.5.0)", () => {
  const X = mat([
    [1, 2, 0, 3],
    [5, 0, 7, 1],
    [0, 3, 4, 2],
    [2, 1, 3, 5],
    [4, 3, 2, 1],
  ]);

  it("fitTransform returns W of the fit, and transform is deterministic", () => {
    const nmf = new NMF({ nComponents: 2, randomState: 0, maxIter: 2000, tol: 1e-9 });
    const W = nmf.fitTransform(X);
    // Previously fitTransform re-solved W from fresh random numbers.
    const W2 = nmf.transform(X);
    const W3 = nmf.transform(X);
    expect(flat(W2)).toEqual(flat(W3));
    const a = flat(W);
    const b = flat(W2);
    for (let i = 0; i < a.length; i++) expect(b[i]).toBeCloseTo(a[i] as number, 2);
  });

  it("reaches the optimum reconstruction error reported by scikit-learn", () => {
    const nmf = new NMF({ nComponents: 2, randomState: 0, maxIter: 1000, tol: 1e-9 });
    nmf.fit(X);
    // sklearn NMF(n_components=2, init="nndsvda", solver="cd"/"mu").reconstruction_err_ = 4.4949379
    expect(nmf.reconstructionErr).toBeCloseTo(4.4949379, 3);
    const W = nmf.transform(X);
    const recon = flat(nmf.inverseTransform(W));
    let err = 0;
    const xs = flat(X);
    for (let i = 0; i < xs.length; i++) err += ((xs[i] as number) - (recon[i] as number)) ** 2;
    expect(Math.sqrt(err)).toBeCloseTo(4.4949379, 2);
  });

  it("supports the nndsvd/nndsvda initializations deterministically", () => {
    const opts = { nComponents: 2, init: "nndsvda" as const, maxIter: 3000, tol: 1e-10 };
    const a = new NMF(opts).fit(X);
    const b = new NMF(opts).fit(X);
    expect(flat(a.components)).toEqual(flat(b.components));
    expect(a.reconstructionErr).toBeCloseTo(4.4949379, 3);
    const plain = new NMF({ nComponents: 2, init: "nndsvd", maxIter: 3000, tol: 1e-10 }).fit(X);
    expect(plain.reconstructionErr).toBeLessThan(4.6);
    expect(() => new NMF({ init: "bogus" as "random" })).toThrow(InvalidParameterError);
    expect(() => new NMF({ nComponents: 5, init: "nndsvd" }).fit(X)).toThrow(InvalidParameterError);
  });

  it("each multiplicative update uses the factors from before the update", () => {
    // Reference: W *= (X H^T) / (W H H^T + 1e-12); H *= (W^T X) / (W^T W H + 1e-12), starting from
    // scikit-learn's nndsvda initialization (numpy, 2 iterations). Updating W in place within a
    // row would change the second and later components.
    const expected: Record<number, number[]> = {
      1: [
        1.6947951967830484, 1.1441800406550242, 2.4735189682742575, 1.3188625412123336,
        2.1425416884570794, 0.728618591894407, 2.7345766500664848, 1.6556953066841205,
      ],
      2: [
        1.732763490429617, 1.3621541425753616, 2.3787221380386647, 1.272493570529835,
        2.051205706274797, 0.5148705501522072, 2.8759244520791114, 1.6935379323662627,
      ],
    };
    for (const iters of [1, 2]) {
      const nmf = new NMF({ nComponents: 2, init: "nndsvda", maxIter: iters, tol: 0 });
      catchWarnings(() => nmf.fit(X));
      expectClose(flat(nmf.components), expected[iters] as number[], 8);
    }
  });

  it("scales the random start to the data so large-valued matrices reach the optimum", () => {
    const big = mat([
      [1000, 2000, 0],
      [5000, 0, 7000],
      [0, 3000, 4000],
      [2000, 1000, 3000],
    ]);
    const nmf = new NMF({ nComponents: 2, randomState: 1, maxIter: 5000, tol: 1e-12 });
    nmf.fit(big);
    const X = flat(big);
    const R = flat(nmf.inverseTransform(nmf.transform(big)));
    const relErr = Math.sqrt(
      X.reduce((a, v, i) => a + (v - (R[i] as number)) ** 2, 0) / X.reduce((a, v) => a + v * v, 0)
    );
    // Optimal rank-2 relative error from sklearn NMF(init="nndsvda", tol=1e-12): 0.168791200
    expect(relErr).toBeCloseTo(0.1687912, 3);
  });

  it("seeded runs are reproducible with a real PRNG and differ across seeds", () => {
    const a = new NMF({ nComponents: 2, randomState: 1, maxIter: 5 }).fit(X);
    const b = new NMF({ nComponents: 2, randomState: 1, maxIter: 5 }).fit(X);
    const c = new NMF({ nComponents: 2, randomState: 2, maxIter: 5 }).fit(X);
    expect(flat(a.components)).toEqual(flat(b.components));
    expect(flat(a.components)).not.toEqual(flat(c.components));
  });

  it("unseeded runs follow the global seed", () => {
    setSeed(11);
    const a = new NMF({ nComponents: 2, maxIter: 5 }).fit(X);
    setSeed(11);
    const b = new NMF({ nComponents: 2, maxIter: 5 }).fit(X);
    expect(flat(a.components)).toEqual(flat(b.components));
    setSeed(11);
    const p1 = new PCA({ nComponents: 1, svdSolver: "randomized" }).fit(PCA_X);
    setSeed(11);
    const p2 = new PCA({ nComponents: 1, svdSolver: "randomized" }).fit(PCA_X);
    expect(flat(p1.components)).toEqual(flat(p2.components));
  });

  it("warns when it stops at maxIter without converging", () => {
    const warnings = catchWarnings(() => {
      new NMF({ nComponents: 2, randomState: 0, maxIter: 2, tol: 0 }).fit(X);
    });
    expect(warnings.some((w) => w.category === "ConvergenceWarning")).toBe(true);
    const quiet = catchWarnings(() => {
      new NMF({ nComponents: 2, randomState: 0, maxIter: 5000, tol: 1e-4 }).fit(X);
    });
    expect(quiet).toEqual([]);
  });

  it("is not affected by setParams after fit and keeps state when a refit fails", () => {
    const nmf = new NMF({ nComponents: 2, randomState: 0 }).fit(X);
    nmf.setParams({ nComponents: 3 });
    expect(nmf.components.shape).toEqual([2, 4]);
    expect(nmf.transform(X).shape).toEqual([5, 2]);
    expect(() => nmf.fit(mat([[1, -1, 2]]))).toThrow(DataValidationError);
    expect(nmf.nFeaturesIn).toBe(4);
    expect(nmf.transform(X).shape).toEqual([5, 2]);
  });

  it("validates tol, randomState and inverseTransform input", () => {
    expect(() => new NMF({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => new NMF({ tol: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new NMF({ randomState: Number.NaN })).toThrow(InvalidParameterError);
    const nmf = new NMF({ nComponents: 2, randomState: 0 }).fit(X);
    expect(() => nmf.inverseTransform(mat([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => nmf.transform(mat([[1, -1, 0, 0]]))).toThrow(DataValidationError);
    expect(() => new NMF().inverseTransform(mat([[1, 2]]))).toThrow(NotFittedError);
  });

  it("handles an all-zero matrix", () => {
    const zeros = mat([
      [0, 0],
      [0, 0],
    ]);
    const nmf = new NMF({ nComponents: 1, randomState: 0 }).fit(zeros);
    expect(flat(nmf.transform(zeros)).every((v) => v === 0)).toBe(true);
    expect(nmf.reconstructionErr).toBeCloseTo(0, 8);
  });
});

describe("FastICA (v1.5.0)", () => {
  function sources(n: number, seed: number): { S: number[][]; X: Tensor } {
    const g = new Generator(seed);
    const u = g.randomArray(n);
    const S: number[][] = [];
    for (let i = 0; i < n; i++) S.push([Math.sign(Math.sin(i * 0.3)), (u[i] as number) * 2 - 1]);
    const A = [
      [1, 1],
      [0.5, 2],
    ];
    const X = S.map((s) => [
      (A[0]?.[0] ?? 0) * (s[0] ?? 0) + (A[0]?.[1] ?? 0) * (s[1] ?? 0),
      (A[1]?.[0] ?? 0) * (s[0] ?? 0) + (A[1]?.[1] ?? 0) * (s[1] ?? 0),
    ]);
    return { S, X: mat(X) };
  }

  function corr(a: number[], b: number[]): number {
    const ma = a.reduce((x, y) => x + y, 0) / a.length;
    const mb = b.reduce((x, y) => x + y, 0) / b.length;
    let c = 0;
    let va = 0;
    let vb = 0;
    for (let i = 0; i < a.length; i++) {
      c += ((a[i] as number) - ma) * ((b[i] as number) - mb);
      va += ((a[i] as number) - ma) ** 2;
      vb += ((b[i] as number) - mb) ** 2;
    }
    return Math.abs(c / Math.sqrt(va * vb));
  }

  it("components has shape (n_components, n_features) and is inverted by mixingMatrix", () => {
    const { X } = sources(600, 0);
    const ica = new FastICA({ nComponents: 2, randomState: 0 }).fit(X);
    const C = ica.components.toArray() as number[][];
    const M = ica.mixingMatrix.toArray() as number[][];
    expect(ica.components.shape).toEqual([2, 2]);
    expect(ica.mixingMatrix.shape).toEqual([2, 2]);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        let s = 0;
        for (let k = 0; k < 2; k++) s += (C[i]?.[k] ?? 0) * (M[k]?.[j] ?? 0);
        expect(s).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
  });

  it("recovers independent sources with unit variance for every contrast function", () => {
    const { S, X } = sources(1000, 0);
    for (const fun of ["logcosh", "exp", "cube"] as const) {
      const ica = new FastICA({ nComponents: 2, randomState: 0, fun }).fit(X);
      const Z = ica.transform(X).toArray() as number[][];
      const z0 = Z.map((r) => r[0] as number);
      const z1 = Z.map((r) => r[1] as number);
      const s0 = S.map((r) => r[0] as number);
      const s1 = S.map((r) => r[1] as number);
      expect(Math.max(corr(z0, s0), corr(z0, s1))).toBeGreaterThan(0.99);
      expect(Math.max(corr(z1, s0), corr(z1, s1))).toBeGreaterThan(0.99);
      for (const col of [z0, z1]) {
        const m = col.reduce((a, b) => a + b, 0) / col.length;
        const v = col.reduce((a, b) => a + (b - m) ** 2, 0) / col.length;
        expect(v).toBeCloseTo(1, 8);
      }
    }
  });

  it("inverseTransform reconstructs X exactly and handles n_components < n_features", () => {
    const { X } = sources(300, 1);
    const ica = new FastICA({ nComponents: 2, randomState: 0 }).fit(X);
    expectClose(flat(ica.inverseTransform(ica.transform(X))), flat(X), 9);

    const g = new Generator(4);
    const wide = tensor(g.normalArray(0, 1, 200 * 4), { dtype: "float64" }).reshape([200, 4]);
    const reduced = new FastICA({ nComponents: 2, randomState: 0 }).fit(wide);
    expect(reduced.components.shape).toEqual([2, 4]);
    expect(reduced.mixingMatrix.shape).toEqual([4, 2]);
    const S = reduced.transform(wide);
    expect(S.shape).toEqual([200, 2]);
    // Projecting back and forth is the identity on the source space.
    expectClose(flat(reduced.transform(reduced.inverseTransform(S))), flat(S), 8);
  });

  it("without whitening the unmixing matrix has orthonormal rows", () => {
    const { X } = sources(200, 2);
    const warnings = catchWarnings(() => {
      const ica = new FastICA({ nComponents: 2, randomState: 0, whiten: false, maxIter: 30 }).fit(
        X
      );
      const C = ica.components.toArray() as number[][];
      const dot = (C[0]?.[0] ?? 0) * (C[1]?.[0] ?? 0) + (C[0]?.[1] ?? 0) * (C[1]?.[1] ?? 0);
      expect(dot).toBeCloseTo(0, 10);
      expectClose(flat(ica.inverseTransform(ica.transform(X))), flat(X), 9);
    });
    expect(warnings.length).toBeLessThanOrEqual(1);
  });

  it("uses a seeded PRNG: same seed same result, different seed different init", () => {
    const { X } = sources(300, 3);
    const a = new FastICA({ nComponents: 2, randomState: 5 }).fit(X);
    const b = new FastICA({ nComponents: 2, randomState: 5 }).fit(X);
    expect(flat(a.components)).toEqual(flat(b.components));
  });

  it("clamps nComponents to min(n_samples, n_features) with a warning", () => {
    const { X } = sources(50, 3);
    let ica: FastICA | undefined;
    const warnings = catchWarnings(() => {
      ica = new FastICA({ nComponents: 5, randomState: 0 }).fit(X);
    });
    expect(warnings.some((w) => w.message.includes("exceeds"))).toBe(true);
    expect(ica?.components.shape).toEqual([2, 2]);
    // Previously transform() reused the requested count and the getters used a stale one.
    expect(ica?.transform(X).shape).toEqual([50, 2]);
    expect(ica?.inverseTransform(ica.transform(X)).shape).toEqual([50, 2]);
  });

  it("is not affected by setParams after fit", () => {
    const { X } = sources(100, 3);
    const ica = new FastICA({ nComponents: 2, randomState: 0 }).fit(X);
    ica.setParams({ nComponents: 1 });
    expect(ica.transform(X).shape).toEqual([100, 2]);
    expect(ica.inverseTransform(ica.transform(X)).shape).toEqual([100, 2]);
  });

  it("rejects rank-deficient data and single samples instead of returning noise", () => {
    const collinear = mat([
      [1, 2],
      [3, 6],
      [5, 10],
      [7, 14],
    ]);
    expect(() => new FastICA({ nComponents: 2 }).fit(collinear)).toThrow(DataValidationError);
    expect(() => new FastICA({ nComponents: 1 }).fit(mat([[1, 2]]))).toThrow(DataValidationError);
    expect(new FastICA({ nComponents: 1, randomState: 0 }).fit(collinear).components.shape).toEqual(
      [1, 2]
    );
  });

  it("validates inverseTransform input and option values", () => {
    const { X } = sources(100, 3);
    const ica = new FastICA({ nComponents: 2, randomState: 0 }).fit(X);
    expect(() => ica.inverseTransform(mat([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => ica.inverseTransform(mat([[Number.NaN, 1]]))).toThrow(DataValidationError);
    expect(() => new FastICA({ tol: -1 })).toThrow(InvalidParameterError);
    expect(() => new FastICA({ fun: "tanh" as "exp" })).toThrow(InvalidParameterError);
    expect(() => new FastICA({ randomState: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    expect(ica.mean.shape).toEqual([2]);
  });

  it("reports divergence as a ConvergenceError", () => {
    // Huge unwhitened data overflows the cube nonlinearity.
    const g = new Generator(0);
    const big = tensor(g.normalArray(0, 1e120, 200 * 2), { dtype: "float64" }).reshape([200, 2]);
    expect(() =>
      new FastICA({ nComponents: 2, whiten: false, fun: "cube", randomState: 0 }).fit(big)
    ).toThrow(ConvergenceError);
  });
});

describe("LatentDirichletAllocation (v1.5.0)", () => {
  function corpus(): Tensor {
    const g = new Generator(3);
    const rows: number[][] = [];
    for (let d = 0; d < 40; d++) {
      const topic = d % 2;
      const row = new Array<number>(10).fill(0);
      for (let w = 0; w < 60; w++) {
        const own = g.random() < 0.9 ? topic : 1 - topic;
        const v = own * 5 + Math.floor(g.random() * 5);
        row[v] = (row[v] ?? 0) + 1;
      }
      rows.push(row);
    }
    return mat(rows);
  }

  it("digamma matches scipy.special.digamma", () => {
    const xs = [0.1, 1, 2.5, 6, 50, 1e-3, 9.99, 10, 1e5];
    const expected = [
      -10.423754940411076, -0.5772156649015329, 0.7031566406452432, 1.7061176684318005,
      3.9019896734278925, -1000.5755719318103, 2.250700372831201, 2.251752589066721,
      11.512920464961896,
    ];
    // The previous 4-term series after a shift to x >= 6 was only accurate to ~1e-7.
    for (let i = 0; i < xs.length; i++) {
      expect(digamma(xs[i] as number)).toBeCloseTo(expected[i] as number, 11);
    }
  });

  it("recovers planted topics", () => {
    const X = corpus();
    const lda = new LatentDirichletAllocation({ nComponents: 2, maxIter: 30, randomState: 0 });
    const dt = lda.fitTransform(X).toArray() as number[][];
    let agree = 0;
    for (let d = 0; d < 40; d++) {
      const pred = (dt[d]?.[0] ?? 0) > (dt[d]?.[1] ?? 0) ? 0 : 1;
      if (pred === d % 2) agree++;
    }
    expect(Math.max(agree, 40 - agree)).toBe(40);
    const comps = lda.components.toArray() as number[][];
    const mass = (row: number[], from: number, to: number) =>
      row.slice(from, to).reduce((a, b) => a + b, 0) / row.reduce((a, b) => a + b, 0);
    const first = comps[0] as number[];
    const dominantFirst = mass(first, 0, 5) > 0.5 ? 0 : 5;
    expect(mass(first, dominantFirst, dominantFirst + 5)).toBeGreaterThan(0.9);
  });

  it("transform is deterministic and rows are probability vectors", () => {
    const X = corpus();
    const lda = new LatentDirichletAllocation({ nComponents: 3, maxIter: 5, randomState: 1 }).fit(
      X
    );
    const a = flat(lda.transform(X));
    expect(flat(lda.transform(X))).toEqual(a);
    for (let d = 0; d < 40; d++) {
      const row = a.slice(d * 3, d * 3 + 3);
      expect(row.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12);
      expect(row.every((v) => v >= 0)).toBe(true);
    }
  });

  it("resolves default priors at fit time and ignores later setParams for fitted state", () => {
    const lda = new LatentDirichletAllocation({ nComponents: 2, randomState: 0 });
    lda.setParams({ nComponents: 4 });
    // Default priors follow the current nComponents (they were frozen at 1/2 before).
    expect(lda.getParams()).toMatchObject({ docTopicPrior: 0.25, topicWordPrior: 0.25 });
    const X = corpus();
    lda.fit(X);
    lda.setParams({ nComponents: 2 });
    expect(lda.components.shape).toEqual([4, 10]);
    expect(lda.transform(X).shape).toEqual([40, 4]);
    expect(lda.inverseTransform(lda.transform(X)).shape).toEqual([40, 10]);
  });

  it("validates priors, tol and inverseTransform input", () => {
    expect(() => new LatentDirichletAllocation({ docTopicPrior: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() => new LatentDirichletAllocation({ topicWordPrior: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new LatentDirichletAllocation({ tol: -1 })).toThrow(InvalidParameterError);
    const lda = new LatentDirichletAllocation({ nComponents: 2, maxIter: 2, randomState: 0 }).fit(
      corpus()
    );
    expect(() => lda.inverseTransform(mat([[0.5, 0.25, 0.25]]))).toThrow(ShapeError);
    expect(() => lda.inverseTransform(mat([[Number.NaN, 0.5]]))).toThrow(DataValidationError);
    expect(() => lda.transform(mat([[1, -1, 0, 0, 0, 0, 0, 0, 0, 0]]))).toThrow(
      DataValidationError
    );
  });

  it("handles empty documents", () => {
    const lda = new LatentDirichletAllocation({ nComponents: 2, maxIter: 2, randomState: 0 }).fit(
      corpus()
    );
    const out = flat(lda.transform(mat([new Array<number>(10).fill(0)])));
    expect(out[0]).toBeCloseTo(0.5, 12);
    expect(out[1]).toBeCloseTo(0.5, 12);
  });
});

describe("jacobiEigenSymmetric", () => {
  it("returns eigenpairs in descending order matching numpy.linalg.eigvalsh", () => {
    const A = Float64Array.from([2, -1, 0, -1, 2, -1, 0, -1, 2]);
    const { values, vectors } = jacobiEigenSymmetric(A, 3);
    expectClose(values, [2 + Math.SQRT2, 2, 2 - Math.SQRT2], 12);
    // A v = lambda v for each eigenvector (columns).
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 3; i++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += (A[i * 3 + k] ?? 0) * (vectors[k * 3 + j] ?? 0);
        expect(s).toBeCloseTo((values[j] ?? 0) * (vectors[i * 3 + j] ?? 0), 12);
      }
    }
  });
});

describe("LinearDiscriminantAnalysis (v1.5.0)", () => {
  const X = mat([
    [1, 1],
    [1, 2],
    [2, 1],
    [10, 1],
    [10, 2],
    [11, 1],
    [5, 10],
    [5, 11],
    [6, 10.5],
    [4, 9],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]);
  const T = mat([
    [3, 3],
    [8, 5],
    [5, 5],
  ]);

  it("matches scikit-learn: explained variance ratio, projection and probabilities", () => {
    const lda = new LinearDiscriminantAnalysis().fit(X, y);
    const ratio = lda.explainedVarianceRatio as Float64Array;
    expectClose(ratio, [0.69274736, 0.30725264], 8);
    // Previously the ratio was normalized over only the kept components.
    expect(
      new LinearDiscriminantAnalysis({ nComponents: 1 }).fit(X, y).explainedVarianceRatio
    ).toEqual(expect.objectContaining({ 0: expect.closeTo(0.69274736, 8) }));
    const proj = (lda.transform(X).toArray() as number[][]).flat().map(Math.abs);
    expectClose(
      proj.slice(0, 6),
      [
        0.8032638006999623, 7.5875987017592, 0.5117072682915546, 6.967874057352414,
        1.7497915373482682, 6.43167451936074,
      ],
      9
    );
    expectClose(
      flat(lda.predictProba(T)),
      [
        1.0, 6.906858245859581e-29, 1.3937444793664788e-20, 4.6623130192829265e-12,
        0.9999999990966497, 8.986879729717611e-10, 0.9953091692543008, 1.4326791542053899e-15,
        0.004690830745697873,
      ],
      9
    );
  });

  it("projected data has identity within-class covariance", () => {
    const lda = new LinearDiscriminantAnalysis().fit(X, y);
    const P = lda.transform(X).toArray() as number[][];
    const labels = flat(y);
    const means = [0, 1, 2].map((c) => {
      const rows = P.filter((_, i) => labels[i] === c);
      return [0, 1].map((k) => rows.reduce((a, r) => a + (r[k] as number), 0) / rows.length);
    });
    const cov = [
      [0, 0],
      [0, 0],
    ];
    P.forEach((r, i) => {
      const m = means[labels[i] as number] as number[];
      for (let a = 0; a < 2; a++) {
        for (let b = 0; b < 2; b++) {
          (cov[a] as number[])[b] =
            ((cov[a] as number[])[b] as number) +
            (((r[a] as number) - (m[a] as number)) * ((r[b] as number) - (m[b] as number))) /
              (P.length - 3);
        }
      }
    });
    expect(cov[0]?.[0]).toBeCloseTo(1, 10);
    expect(cov[1]?.[1]).toBeCloseTo(1, 10);
    expect(cov[0]?.[1]).toBeCloseTo(0, 10);
  });

  it("custom priors shift the projection centre and the posteriors like scikit-learn", () => {
    const lda = new LinearDiscriminantAnalysis({ priors: [0.2, 0.3, 0.5] }).fit(X, y);
    expectClose(
      lda.explainedVarianceRatio as Float64Array,
      [0.7663949460388098, 0.2336050539611902],
      8
    );
    expectClose(
      flat(lda.predictProba(T)),
      [
        1.0, 1.036028736878977e-28, 2.6132708988121528e-20, 3.1082086788284514e-12,
        0.9999999988735304, 1.123359965964055e-9, 0.9912406449224115, 2.140234189511324e-15,
        0.008759355077586621,
      ],
      9
    );
    const proj = (lda.transform(X).toArray() as number[][]).slice(0, 2).flat().map(Math.abs);
    expectClose(
      proj,
      [1.7848215762359407, 8.521958318146295, 0.45759819534944246, 7.928927421482364],
      9
    );
    expectClose(flat(lda.classPriors), [0.2, 0.3, 0.5], 12);
  });

  it("rejects invalid priors, including NaN which used to pass the sum check", () => {
    expect(() => new LinearDiscriminantAnalysis({ priors: [0.5, Number.NaN, 0.5] })).toThrow(
      InvalidParameterError
    );
    expect(() => new LinearDiscriminantAnalysis({ priors: [0.5, 0.6] })).toThrow(/sum to 1/);
    expect(() => new LinearDiscriminantAnalysis({ priors: [1.5, -0.5] })).toThrow(
      InvalidParameterError
    );
    expect(() => new LinearDiscriminantAnalysis({ priors: [0.5, 0.5] }).fit(X, y)).toThrow(
      /length 3/
    );
    expect(() => new LinearDiscriminantAnalysis().setParams({ priors: [0.1, Number.NaN] })).toThrow(
      InvalidParameterError
    );
    expect(() => new LinearDiscriminantAnalysis().setParams({ shrinkage: Number.NaN })).toThrow(
      InvalidParameterError
    );
  });

  it("failed fits leave the previous model untouched", () => {
    const lda = new LinearDiscriminantAnalysis().fit(X, y);
    const before = flat(lda.predictProba(T));
    lda.setParams({ priors: [0.5, 0.5] });
    expect(() => lda.fit(X, y)).toThrow(InvalidParameterError);
    expect(flat(lda.predictProba(T))).toEqual(before);
    expect(lda.nFeaturesIn).toBe(2);
  });

  it("Ledoit-Wolf shrinkage 'auto' matches the reference construction", () => {
    const n = 45;
    const rows: number[][] = [];
    const labels: number[] = [];
    for (let i = 0; i < n; i++) {
      const c = i % 3;
      const a = Math.sin(i * 0.7);
      const b = Math.cos(i * 1.3);
      rows.push([
        a + 0.5 * b + 0.9 * c,
        a - 0.3 * b + 0.4 * Math.sin(i * 2.1) + 0.5 * c,
        0.7 * a + 0.2 * Math.cos(i * 0.37) - 0.6 * c,
      ]);
      labels.push(c);
    }
    const Xd = mat(rows);
    const yd = tensor(labels);
    const Td = mat([
      [0.5, 1.0, -0.5],
      [1.0, 1.0, -1.0],
      [2.0, 1.5, -1.5],
    ]);
    // delta = sklearn.covariance.ledoit_wolf_shrinkage on standardized within-class data = 0.0294551731...
    // Posteriors from (1 - delta) * Sigma + delta * diag(Sigma).
    const auto = new LinearDiscriminantAnalysis({ shrinkage: "auto" }).fit(Xd, yd);
    expectClose(
      flat(auto.predictProba(Td)),
      [
        1.217035848197707e-7, 0.999999508333315, 3.6996310005732765e-7, 9.32393228978548e-18,
        0.32968107654575085, 0.6703189234542493, 4.391145085513369e-43, 2.681242435326743e-11,
        0.9999999999731877,
      ],
      9
    );
    expectClose(
      auto.explainedVarianceRatio as Float64Array,
      [0.9900122945407761, 0.00998770545922392],
      8
    );
    const none = new LinearDiscriminantAnalysis().fit(Xd, yd);
    expectClose(
      flat(none.predictProba(Td)).slice(3, 6),
      [1.8537739436121192e-28, 0.09731647829850043, 0.9026835217014996],
      9
    );
  });

  it("numeric shrinkage blends with mean variance times identity", () => {
    const Xs = mat([
      [1, 1, 2],
      [1, 2, 1.5],
      [2, 1, 0.2],
      [10, 1, 3],
      [10, 2, 2.5],
      [11, 1, 1],
      [5, 10, 4],
      [5, 11, 0],
      [6, 10.5, 2],
      [4, 9, 5],
      [3, 3, 3.3],
      [8, 6, 1],
    ]);
    const ys = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 1]);
    const Ts = mat([
      [3, 3, 1],
      [8, 5, 2],
      [5, 5, 3],
    ]);
    const lda = new LinearDiscriminantAnalysis({ shrinkage: 0.3 }).fit(Xs, ys);
    expectClose(
      flat(lda.predictProba(Ts)),
      [
        0.999999898014745, 5.170807508888144e-8, 5.027718014067772e-8, 4.905122779927546e-9,
        0.9972772735119529, 0.0027227215829240758, 0.0428641117270142, 0.055749938206912944,
        0.9013859500660734,
      ],
      9
    );
  });

  it("handles collinear features with finite probabilities (was NaN) and warns", () => {
    const Xc = mat([
      [1, 2, 1],
      [2, 4, 3],
      [3, 6, 2],
      [8, 16, 7],
      [9, 18, 9],
      [10, 20, 8],
    ]);
    const yc = tensor([0, 0, 0, 1, 1, 1]);
    let lda: LinearDiscriminantAnalysis | undefined;
    const warnings = catchWarnings(() => {
      lda = new LinearDiscriminantAnalysis().fit(Xc, yc);
    });
    expect(warnings.some((w) => w.message.includes("collinear"))).toBe(true);
    const proba = flat(lda?.predictProba(Xc) as Tensor);
    expect(proba.every((v) => Number.isFinite(v))).toBe(true);
    expect(flat(lda?.predict(Xc) as Tensor)).toEqual([0, 0, 0, 1, 1, 1]);
    expect(flat(lda?.transform(Xc) as Tensor).every((v) => Number.isFinite(v))).toBe(true);
  });

  it("requires more samples than classes and a non-degenerate covariance", () => {
    expect(() =>
      new LinearDiscriminantAnalysis().fit(
        mat([
          [1, 2],
          [3, 4],
        ]),
        tensor([0, 1])
      )
    ).toThrow(DataValidationError);
    const same = mat([
      [1, 1],
      [1, 1],
      [2, 2],
      [2, 2],
    ]);
    expect(() => new LinearDiscriminantAnalysis().fit(same, tensor([0, 0, 1, 1]))).toThrow(
      /Within-class covariance is zero/
    );
  });

  it("keeps empty batches 2-D, uses float64 probabilities and exposes log probabilities", () => {
    const lda = new LinearDiscriminantAnalysis().fit(X, y);
    const empty = tensor(new Float64Array(0)).reshape([0, 2]);
    expect(lda.predictProba(empty).shape).toEqual([0, 3]);
    expect(lda.transform(empty).shape).toEqual([0, 2]);
    expect(lda.predict(empty).shape).toEqual([0]);
    const proba = lda.predictProba(T);
    expect(proba.dtype).toBe("float64");
    const logp = flat(lda.predictLogProba(T));
    const p = flat(proba);
    for (let i = 0; i < p.length; i++)
      expect(Math.exp(logp[i] as number)).toBeCloseTo(p[i] as number, 12);
    // Very far points keep valid log probabilities.
    expect(flat(lda.predictLogProba(mat([[1e6, -1e6]]))).every((v) => v <= 0)).toBe(true);
  });

  it("score validates the label vector length", () => {
    const lda = new LinearDiscriminantAnalysis().fit(X, y);
    expect(lda.score(X, y)).toBe(1);
    expect(() => lda.score(X, tensor([0, 1]))).toThrow(ShapeError);
    expect(() => lda.score(X, mat([[0, 1]]))).toThrow(ShapeError);
  });

  it("keeps non-integer labels instead of truncating them to int32", () => {
    const yf = tensor([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5], { dtype: "float64" });
    const lda = new LinearDiscriminantAnalysis().fit(X, yf);
    expect(lda.predict(T).dtype).toBe("float64");
    expect(flat(lda.predict(T))).toEqual([0.5, 1.5, 0.5]);
    expect(flat(lda.classes as Tensor)).toEqual([0.5, 1.5, 2.5]);
    expect(lda.score(X, yf)).toBe(1);
    expect(new LinearDiscriminantAnalysis().fit(X, y).predict(T).dtype).toBe("int32");
  });

  it("exposes fitted attributes and validates accessor state", () => {
    const lda = new LinearDiscriminantAnalysis();
    expect(() => lda.classMeans).toThrow(NotFittedError);
    expect(() => lda.scalings).toThrow(NotFittedError);
    lda.fit(X, y);
    expect(lda.classMeans.shape).toEqual([3, 2]);
    expect(lda.scalings.shape).toEqual([2, 2]);
    expectClose(flat(lda.classMeans).slice(0, 2), [4 / 3, 4 / 3], 12);
    const ratio = lda.explainedVarianceRatio;
    if (ratio) ratio[0] = 5;
    expect((lda.explainedVarianceRatio as Float64Array)[0]).toBeCloseTo(0.69274736, 8);
  });
});

describe("QuadraticDiscriminantAnalysis (v1.5.0)", () => {
  const X = mat([
    [1, 1],
    [1, 2],
    [2, 1],
    [10, 1],
    [10, 2],
    [11, 1],
    [5, 10],
    [5, 11],
    [6, 10.5],
    [4, 9],
  ]);
  const y = tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]);
  const T = mat([
    [3, 3],
    [8, 5],
    [5, 5],
  ]);

  it("matches scikit-learn posteriors", () => {
    const qda = new QuadraticDiscriminantAnalysis().fit(X, y);
    expectClose(
      flat(qda.predictProba(T)),
      [
        0.9999999999999727, 5.380186160020833e-32, 2.7201729329086912e-14, 2.8946403116480536e-63,
        1.0, 3.932359784119651e-28, 1.4854781232461616e-19, 0.0006404240147488485,
        0.9993595759852513,
      ],
      9
    );
    expect(qda.predictProba(T).dtype).toBe("float64");
  });

  it("regParam adds to the covariance diagonal (documented behavior)", () => {
    const qda = new QuadraticDiscriminantAnalysis({ regParam: 0.1 }).fit(X, y);
    expectClose(
      flat(qda.predictProba(T)),
      [
        0.9999999999949765, 6.199956226920847e-24, 5.0236174038000856e-12, 1.3644255994013601e-38,
        1.0, 9.491727628188826e-18, 3.2181808984050494e-11, 0.00004119804470302206,
        0.9999588019231165,
      ],
      9
    );
  });

  it("rejects single-sample classes unless regParam > 0", () => {
    const Xs = mat([
      [1, 1],
      [1, 2],
      [2, 1],
      [9, 9],
    ]);
    const ys = tensor([0, 0, 0, 1]);
    expect(() => new QuadraticDiscriminantAnalysis().fit(Xs, ys)).toThrow(DataValidationError);
    const qda = new QuadraticDiscriminantAnalysis({ regParam: 0.5 }).fit(Xs, ys);
    expect(
      flat(
        qda.predict(
          mat([
            [9, 9.5],
            [1, 1.5],
          ])
        )
      )
    ).toEqual([1, 0]);
    expect(flat(qda.predictProba(Xs)).every((v) => Number.isFinite(v))).toBe(true);
  });

  it("gives finite, correct predictions for collinear classes (was +Infinity / NaN)", () => {
    const Xc = mat([
      [1, 2],
      [2, 4],
      [3, 6],
      [8, 16.5],
      [9, 17.5],
      [10, 21],
    ]);
    const yc = tensor([0, 0, 0, 1, 1, 1]);
    let qda: QuadraticDiscriminantAnalysis | undefined;
    const warnings = catchWarnings(() => {
      qda = new QuadraticDiscriminantAnalysis().fit(Xc, yc);
    });
    expect(warnings.some((w) => w.message.includes("collinear"))).toBe(true);
    const proba = flat(qda?.predictProba(Xc) as Tensor);
    expect(proba.every((v) => Number.isFinite(v))).toBe(true);
    for (let i = 0; i < 6; i++) {
      expect((proba[i * 2] as number) + (proba[i * 2 + 1] as number)).toBeCloseTo(1, 12);
    }
    expect(flat(qda?.predict(Xc) as Tensor)).toEqual([0, 0, 0, 1, 1, 1]);
  });

  it("a zero prior gives exactly zero probability", () => {
    const qda = new QuadraticDiscriminantAnalysis({ priors: [0, 0.5, 0.5] }).fit(X, y);
    const proba = flat(qda.predictProba(mat([[1.5, 1.5]])));
    expect(proba[0]).toBe(0);
    expect(flat(qda.predict(mat([[1.5, 1.5]])))[0]).not.toBe(0);
    expectClose(flat(qda.classPriors), [0, 0.5, 0.5], 12);
  });

  it("validates priors like LDA and keeps empty batches 2-D", () => {
    expect(() => new QuadraticDiscriminantAnalysis({ priors: [0.5, Number.NaN, 0.5] })).toThrow(
      InvalidParameterError
    );
    expect(() => new QuadraticDiscriminantAnalysis({ priors: [0.5, 0.5] }).fit(X, y)).toThrow(
      /length 3/
    );
    const qda = new QuadraticDiscriminantAnalysis().fit(X, y);
    const empty = tensor(new Float64Array(0)).reshape([0, 2]);
    expect(qda.predictProba(empty).shape).toEqual([0, 3]);
    expect(qda.predictLogProba(empty).shape).toEqual([0, 3]);
    expect(qda.classMeans.shape).toEqual([3, 2]);
    expect(() => qda.score(X, tensor([0, 1]))).toThrow(ShapeError);
    expect(qda.score(X, y)).toBe(1);
  });

  it("failed fits leave the previous model untouched", () => {
    const qda = new QuadraticDiscriminantAnalysis().fit(X, y);
    const before = flat(qda.predictProba(T));
    qda.setParams({ priors: [0.5, 0.5] });
    expect(() => qda.fit(X, y)).toThrow(InvalidParameterError);
    expect(flat(qda.predictProba(T))).toEqual(before);
    expect(qda.nFeaturesIn).toBe(2);
  });
});
