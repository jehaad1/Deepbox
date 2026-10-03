/**
 * Wave 2 regression tests for `deepbox/linalg` and `deepbox/optim` (group linalg-optim):
 * complex dtype rejection and function names in dense conversion errors, det and slogdet
 * when LU elimination overflows, eigenvalue error details, the sparse solver bridge to the
 * ndarray `CSRMatrix`, the public export surface, optimizer option validation per group,
 * frozen parameters, atomic steps, the shared step counter and `maximize` on every optimizer
 * that PyTorch gives it to.
 */

import { describe, expect, it } from "vitest";
import { DataValidationError, DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import {
  blockDiag,
  cholesky,
  denseToCSR,
  det,
  eig,
  eigh,
  eigvals,
  expm,
  hessenberg,
  inv,
  logm,
  lstsq,
  lu,
  lyapunov,
  matrixPower,
  qr,
  type SolveTriangularOptions,
  type SparseMatrixInput,
  schur,
  slogdet,
  solve,
  solveBanded,
  solveTriangular,
  sparseCholeskySolve,
  sparseSolve,
  sqrtm,
  svd,
  sylvester,
} from "../../src/linalg";
import {
  CSRMatrix,
  GradTensor,
  parameter,
  type Tensor,
  tensor,
  transpose,
} from "../../src/ndarray";
import {
  AdaDelta,
  Adagrad,
  Adam,
  Adamax,
  AdamW,
  ASGD,
  LAMB,
  LARS,
  LBFGS,
  Lion,
  Nadam,
  type Optimizer,
  type PlateauStateDict,
  RAdam,
  RMSprop,
  Rprop,
  type SchedulerStateDict,
  SGD,
  SparseAdam,
} from "../../src/optim";

const f64 = (data: number | number[] | number[][]): Tensor => tensor(data, { dtype: "float64" });
const rows = (t: Tensor): number[][] => t.toArray() as number[][];
const vec = (t: Tensor): number[] => t.toArray() as number[];

/** A tensor that reports a complex dtype, as an interleaved complex buffer would. */
function asComplex(t: Tensor): Tensor {
  return Object.create(t, { dtype: { value: "complex128" } }) as Tensor;
}

describe("linalg dense conversion", () => {
  it("rejects complex tensors instead of reading the real parts", () => {
    const m = asComplex(
      f64([
        [1, 2],
        [3, 4],
      ])
    );
    const v = asComplex(f64([1, 2]));
    expect(() => inv(m)).toThrow(DTypeError);
    expect(() => solve(m, v)).toThrow(DTypeError);
    expect(() =>
      solve(
        f64([
          [1, 0],
          [0, 1],
        ]),
        v
      )
    ).toThrow(DTypeError);
    expect(() => svd(m)).toThrow(DTypeError);
    expect(() => eig(m)).toThrow(DTypeError);
    expect(() => det(m)).toThrow(DTypeError);
    expect(() => lstsq(m, v)).toThrow(DTypeError);
    expect(() => schur(m)).toThrow(DTypeError);
    expect(() => cholesky(m)).toThrow(DTypeError);
  });

  it("names the failing function when an input is not finite", () => {
    const bad = f64([
      [1, Number.NaN],
      [3, 4],
    ]);
    const good = f64([
      [1, 2],
      [3, 4],
    ]);
    const badVec = f64([1, Number.POSITIVE_INFINITY]);
    expect(() => inv(bad)).toThrow(/inv\(\): input contains non-finite values/);
    expect(() => solve(bad, f64([1, 2]))).toThrow(/solve\(\)/);
    expect(() => solve(good, badVec)).toThrow(/solve\(\)/);
    expect(() => solveTriangular(good, badVec)).toThrow(/solveTriangular\(\)/);
    expect(() => svd(bad)).toThrow(/svd\(\)/);
    expect(() => det(bad)).toThrow(/det\(\)/);
    expect(() => slogdet(bad)).toThrow(/slogdet\(\)/);
    expect(() => lu(bad)).toThrow(/lu\(\)/);
    expect(() => qr(bad)).toThrow(/qr\(\)/);
    expect(() => eig(bad)).toThrow(/eig\(\)/);
    expect(() => eigh(bad)).toThrow(/eigh\(\)/);
    expect(() => schur(bad)).toThrow(/schur\(\)/);
    expect(() => hessenberg(bad)).toThrow(/hessenberg\(\)/);
    expect(() => lstsq(bad, f64([1, 2]))).toThrow(/lstsq\(\)/);
    expect(() => sylvester(bad, good, good)).toThrow(/sylvester\(\)/);
    expect(() => sylvester(good, good, bad)).toThrow(/sylvester\(\)/);
    expect(() => lyapunov(bad, good)).toThrow(/lyapunov\(\)/);
    expect(() => denseToCSR(bad)).toThrow(/denseToCSR\(\)/);
    expect(() => inv(bad)).toThrow(DataValidationError);
  });
});

describe("det and slogdet when elimination overflows", () => {
  const big = f64([
    [1e308, -1e308],
    [1e308, 1e308],
  ]);

  it("returns Infinity for a determinant above the float64 range, as NumPy does", () => {
    expect(det(big)).toBe(Number.POSITIVE_INFINITY);
    expect(
      det(
        f64([
          [1e308, 1e308],
          [1e308, -1e308],
        ])
      )
    ).toBe(Number.NEGATIVE_INFINITY);
  });

  it("computes log|det| by scaling instead of returning Infinity", () => {
    const [sign, logdet] = slogdet(big);
    expect(sign.item()).toBe(1);
    // log(2e616) = log(2) + 616 * log(10)
    expect(logdet.item()).toBeCloseTo(Math.LN2 + 616 * Math.LN10, 9);
  });

  it("keeps the finite determinant of a matrix whose elimination overflows", () => {
    // True determinant: 1e-10 * 1e308 + 1e-10 * 1e308 = 2e298, but the second pivot of
    // the LU factorization is 1e308 + 1e308, which overflows.
    const a = f64([
      [1e-10, 1e308],
      [-1e-10, 1e308],
    ]);
    expect(det(a) / 2e298).toBeCloseTo(1, 12);
    const [sign, logdet] = slogdet(a);
    expect(sign.item()).toBe(1);
    expect(logdet.item()).toBeCloseTo(Math.log(2e298), 9);
    const [signNeg] = slogdet(
      f64([
        [-1e-10, 1e308],
        [1e-10, 1e308],
      ])
    );
    expect(signNeg.item()).toBe(-1);
  });

  it("still reports an exactly singular matrix as zero", () => {
    expect(
      det(
        f64([
          [1, 2],
          [2, 4],
        ])
      )
    ).toBe(0);
    expect(
      slogdet(
        f64([
          [1, 2],
          [2, 4],
        ])
      )[1].item()
    ).toBe(Number.NEGATIVE_INFINITY);
  });
});

describe("eig complex spectrum error", () => {
  it("reports the offending eigenvalue pair in the message and in error.value", () => {
    let caught: unknown;
    try {
      eig(
        f64([
          [0, 2],
          [-2, 0],
        ])
      );
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(InvalidParameterError);
    const error = caught as InvalidParameterError;
    expect(error.message).toMatch(/complex eigenvalues/);
    expect(error.message).toMatch(/\+\/- 2(\.\d+)?i/);
    const value = error.value as { real: number; imag: number };
    expect(value.real).toBeCloseTo(0, 12);
    expect(value.imag).toBeCloseTo(2, 12);
  });

  it("reports the value in the units of the input for badly scaled matrices", () => {
    let caught: unknown;
    try {
      eigvals(
        f64([
          [0, 2e-200],
          [-2e-200, 0],
        ])
      );
    } catch (err) {
      caught = err;
    }
    const value = (caught as InvalidParameterError).value as { real: number; imag: number };
    expect(value.imag / 2e-200).toBeCloseTo(1, 10);
  });
});

describe("eig on a nearly symmetric matrix", () => {
  it("keeps the eigenpair residual at rounding level when the asymmetry is 1e-10", () => {
    // NumPy gives a residual of about 3e-15. Treating this matrix as symmetric would
    // move the eigenvectors by the asymmetry and leave a residual near 1e-10.
    const A = [
      [2, 1, 0],
      [1 + 3e-10, 3, 1],
      [0, 1 + 2e-10, 4],
    ];
    const [w, V] = eig(f64(A));
    const values = w.toArray() as number[];
    const vectors = V.toArray() as number[][];
    let residual = 0;
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 3; i++) {
        let sum = 0;
        for (let k = 0; k < 3; k++) sum += (A[i]?.[k] ?? 0) * (vectors[k]?.[j] ?? 0);
        residual = Math.max(residual, Math.abs(sum - (values[j] ?? 0) * (vectors[i]?.[j] ?? 0)));
      }
    }
    expect(residual).toBeLessThan(1e-13);
  });

  it("still takes the symmetric path for rounding-level asymmetry", () => {
    const [w] = eig(
      f64([
        [2, 1 + 1e-15],
        [1, 2],
      ])
    );
    const values = w.toArray() as number[];
    expect(values[0]).toBeCloseTo(1, 12);
    expect(values[1]).toBeCloseTo(3, 12);
  });
});

describe("matrix functions with complex spectra and defective matrices", () => {
  it("expm of a rotation generator is a rotation", () => {
    const e = rows(
      expm(
        f64([
          [0, 2],
          [-2, 0],
        ])
      )
    );
    expect(e[0]?.[0]).toBeCloseTo(-0.41614683654714235, 12);
    expect(e[0]?.[1]).toBeCloseTo(0.9092974268256818, 12);
    expect(e[1]?.[0]).toBeCloseTo(-0.9092974268256817, 12);
    expect(e[1]?.[1]).toBeCloseTo(-0.41614683654714213, 12);
  });

  it("sqrtm and logm of Jordan blocks match SciPy", () => {
    const s = rows(
      sqrtm(
        f64([
          [4, 1],
          [0, 4],
        ])
      )
    );
    expect(s[0]?.[0]).toBeCloseTo(2, 12);
    expect(s[0]?.[1]).toBeCloseTo(0.25, 12);
    expect(s[1]?.[0]).toBeCloseTo(0, 12);
    const l = rows(
      logm(
        f64([
          [2, 1],
          [0, 2],
        ])
      )
    );
    expect(l[0]?.[0]).toBeCloseTo(Math.LN2, 12);
    expect(l[0]?.[1]).toBeCloseTo(0.5, 12);
  });
});

describe("schur and sylvester at extreme scales", () => {
  const base = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10],
  ];

  it.each([1e-160, 1e-20, 1, 1e20, 1e150])("schur is scale invariant (scale %s)", (scale) => {
    const a = f64(base.map((r) => r.map((v) => v * scale)));
    const [t, q] = schur(a);
    const tq = rows(t);
    // The eigenvalues are real here, so T is exactly upper triangular.
    expect(tq[1]?.[0]).toBe(0);
    expect(tq[2]?.[0]).toBe(0);
    expect(tq[2]?.[1]).toBe(0);
    expect((tq[0]?.[0] ?? 0) / scale).toBeCloseTo(16.707493, 5);
    const qq = rows(q);
    const recon = qq.map((qr, i) =>
      (tq[0] ?? []).map((_, j) =>
        qr.reduce((acc, _v, k) => {
          // (Q T Q^T)[i][j] = sum_k sum_l Q[i][k] T[k][l] Q[j][l]
          let inner = 0;
          for (let l2 = 0; l2 < 3; l2++) {
            inner += (tq[k]?.[l2] ?? 0) * (qq[j]?.[l2] ?? 0);
          }
          return acc + (qq[i]?.[k] ?? 0) * inner;
        }, 0)
      )
    );
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(((recon[i]?.[j] ?? 0) - (base[i]?.[j] ?? 0) * scale) / scale).toBeCloseTo(0, 10);
      }
    }
  });

  it("schur leaves a non-zero sub-diagonal entry only for a complex pair", () => {
    const [t] = schur(
      f64([
        [0, 1e-20],
        [-1e-20, 0],
      ])
    );
    const tq = rows(t);
    expect(tq[1]?.[0]).not.toBe(0);
    const [t2] = schur(
      f64([
        [2, 1],
        [0, 3],
      ])
    );
    expect(rows(t2)[1]?.[0]).toBe(0);
  });

  it("sylvester with quasi-triangular factors matches SciPy at tiny scales", () => {
    const a0 = [
      [1, 2, 0],
      [-2, 1, 0.5],
      [0, 0, 3],
    ];
    const b0 = [
      [0.5, 1.5],
      [-1.5, 0.5],
    ];
    const c0 = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const expected = [
      [-0.9448275862068964, 0.42068965517241363],
      [1.5241379310344825, 1.393103448275862],
      [1.8275862068965516, 0.9310344827586207],
    ];
    for (const scale of [1, 1e-30, 1e30]) {
      const x = rows(
        sylvester(
          f64(a0.map((r) => r.map((v) => v * scale))),
          f64(b0.map((r) => r.map((v) => v * scale))),
          f64(c0.map((r) => r.map((v) => v * scale)))
        )
      );
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
          expect(x[i]?.[j]).toBeCloseTo(expected[i]?.[j] ?? 0, 10);
        }
      }
    }
  });
});

describe("sparse solvers accept the ndarray CSRMatrix", () => {
  const dense = [
    [4, 1, 0],
    [1, 3, 1],
    [0, 1, 2],
  ];
  const b = f64([1, 2, 3]);
  const nd = new CSRMatrix({
    data: new Float64Array([4, 1, 1, 3, 1, 1, 2]),
    indices: new Int32Array([0, 1, 0, 1, 2, 1, 2]),
    indptr: new Int32Array([0, 2, 5, 7]),
    shape: [3, 3],
  });

  it("gives the same answer as the dense solver", () => {
    const expected = vec(solve(f64(dense), b));
    const viaLu = vec(sparseSolve(nd, b));
    const viaChol = vec(sparseCholeskySolve(nd, b));
    const viaRecord = vec(sparseSolve(denseToCSR(f64(dense)), b));
    for (let i = 0; i < 3; i++) {
      expect(viaLu[i]).toBeCloseTo(expected[i] ?? 0, 12);
      expect(viaChol[i]).toBeCloseTo(expected[i] ?? 0, 12);
      expect(viaRecord[i]).toBeCloseTo(expected[i] ?? 0, 12);
    }
  });

  it("rejects a non-square ndarray CSRMatrix", () => {
    const rect = new CSRMatrix({
      data: new Float64Array([1, 2]),
      indices: new Int32Array([0, 1]),
      indptr: new Int32Array([0, 1, 2]),
      shape: [2, 3],
    });
    expect(() => sparseSolve(rect, f64([1, 2]))).toThrow(ShapeError);
    expect(() => sparseCholeskySolve(rect, f64([1, 2]))).toThrow(/square/);
  });

  it("exports the input type from deepbox/linalg", () => {
    const input: SparseMatrixInput = nd;
    expect(input.shape).toEqual([3, 3]);
  });
});

describe("deepbox/linalg export surface", () => {
  it("exposes the camelCase aliases and the option type", () => {
    const options: SolveTriangularOptions = { trans: true };
    const x = vec(
      solveTriangular(
        f64([
          [2, 0],
          [3, 4],
        ]),
        f64([6, 18]),
        true,
        options
      )
    );
    expect(x[0]).toBeCloseTo(-3.75, 12);
    expect(x[1]).toBeCloseTo(4.5, 12);
    expect(typeof matrixPower).toBe("function");
    expect(typeof blockDiag).toBe("function");
    expect(typeof solveBanded).toBe("function");
    const p = rows(
      matrixPower(
        f64([
          [1, 1],
          [0, 1],
        ]),
        3
      )
    );
    expect(p[0]?.[1]).toBeCloseTo(3, 12);
    expect(rows(blockDiag(f64([[1]]), f64([[2]])))).toEqual([
      [1, 0],
      [0, 2],
    ]);
  });
});

describe("optim export surface", () => {
  it("exports the scheduler state types", () => {
    const state: SchedulerStateDict | undefined = undefined;
    const plateau: PlateauStateDict | undefined = undefined;
    expect(state).toBeUndefined();
    expect(plateau).toBeUndefined();
  });
});

const stepFactories: ReadonlyArray<readonly [string, (p: GradTensor[]) => { step(): unknown }]> = [
  ["SGD", (p) => new SGD(p, { lr: 0.1, momentum: 0.9 })],
  ["Adam", (p) => new Adam(p, { lr: 0.1 })],
  ["AdamW", (p) => new AdamW(p, { lr: 0.1 })],
  ["Adagrad", (p) => new Adagrad(p, { lr: 0.1 })],
  ["Adamax", (p) => new Adamax(p, { lr: 0.1 })],
  ["Nadam", (p) => new Nadam(p, { lr: 0.1 })],
  ["AdaDelta", (p) => new AdaDelta(p, { lr: 0.5 })],
  ["LAMB", (p) => new LAMB(p, { lr: 0.1 })],
  ["LARS", (p) => new LARS(p, { lr: 0.1 })],
  ["Lion", (p) => new Lion(p, { lr: 0.1 })],
  ["RAdam", (p) => new RAdam(p, { lr: 0.1 })],
  ["RMSprop", (p) => new RMSprop(p, { lr: 0.1 })],
  ["Rprop", (p) => new Rprop(p, { lr: 0.1 })],
  ["ASGD", (p) => new ASGD(p, { lr: 0.1 })],
  ["SparseAdam", (p) => new SparseAdam(p, { lr: 0.1 })],
];

const badGroupFactories: ReadonlyArray<readonly [string, (p: GradTensor[]) => unknown]> = [
  ["SGD", (p) => new SGD([{ params: p, lr: -1 }])],
  ["SGD nesterov", (p) => new SGD([{ params: p, nesterov: true }])],
  ["Adam", (p) => new Adam([{ params: p, eps: 0 }])],
  ["AdamW", (p) => new AdamW([{ params: p, beta1: 1 }])],
  ["Adagrad", (p) => new Adagrad([{ params: p, eps: 0 }])],
  ["Adamax", (p) => new Adamax([{ params: p, lr: -1 }])],
  ["Nadam", (p) => new Nadam([{ params: p, beta2: 1 }])],
  ["AdaDelta", (p) => new AdaDelta([{ params: p, eps: 0 }])],
  ["LAMB", (p) => new LAMB([{ params: p, eps: 0 }])],
  ["LARS", (p) => new LARS([{ params: p, eta: 0 }])],
  ["Lion", (p) => new Lion([{ params: p, beta1: 1 }])],
  ["RAdam", (p) => new RAdam([{ params: p, lr: -1 }])],
  ["RMSprop", (p) => new RMSprop([{ params: p, alpha: 2 }])],
  ["Rprop", (p) => new Rprop([{ params: p, etaMinus: 2 }])],
  ["ASGD", (p) => new ASGD([{ params: p, lambda: -1 }])],
  ["SparseAdam", (p) => new SparseAdam([{ params: p, eps: 0 }])],
  ["LBFGS", (p) => new LBFGS([{ params: p, lr: -1 }])],
];

function makeParam(values: number[]): GradTensor {
  return parameter(f64(values));
}

describe("optimizer per-group option validation", () => {
  it.each(badGroupFactories)("%s rejects a bad group override at construction", (_name, make) => {
    expect(() => make([makeParam([1, 2])])).toThrow(InvalidParameterError);
  });

  it("rejects a bad group in addParamGroup and a bad group in loadStateDict", () => {
    const p = makeParam([1]);
    const opt = new Adam([p]);
    expect(() => opt.addParamGroup({ params: [makeParam([2])], lr: -1 })).toThrow(
      InvalidParameterError
    );
    expect(opt.paramGroups).toHaveLength(1);
    const sd = opt.stateDict();
    const broken = {
      ...sd,
      paramGroups: sd.paramGroups.map((g) => ({ ...g, options: { ...g.options, eps: 0 } })),
    };
    expect(() => opt.loadStateDict(broken)).toThrow(InvalidParameterError);
    expect(opt.paramGroups[0]?.options.eps).toBe(1e-8);
  });

  it("still accepts valid per-group overrides", () => {
    const a = makeParam([1]);
    const b = makeParam([2]);
    const opt = new Adam([{ params: [a], lr: 0.5 }, { params: [b] }], { lr: 0.01 });
    expect(opt.getLearningRate(0)).toBe(0.5);
    expect(opt.getLearningRate(1)).toBe(0.01);
  });
});

describe("optimizer learning rate accessors", () => {
  it("share one implementation and one set of error messages", () => {
    const opt = new SGD([makeParam([1])], { lr: 0.1 });
    expect(opt.getLearningRate()).toBe(0.1);
    opt.setLearningRate(0.5);
    expect(opt.getLearningRate()).toBe(0.5);
    expect(() => opt.getLearningRate(3)).toThrow(/Invalid group index: 3/);
    expect(() => opt.setLearningRate(-1)).toThrow(InvalidParameterError);
    expect(() => opt.setLearningRate(Number.NaN)).toThrow(InvalidParameterError);
    // The same code serves every optimizer.
    const lbfgs = new LBFGS([makeParam([1])]);
    expect(() => lbfgs.getLearningRate(1)).toThrow(/Invalid group index: 1/);
    expect(Object.hasOwn(SGD.prototype, "getLearningRate")).toBe(false);
    expect(Object.hasOwn(Adam.prototype, "setLearningRate")).toBe(false);
  });
});

describe("frozen parameters", () => {
  it.each(stepFactories)("%s skips a parameter with requiresGrad=false", (_name, make) => {
    const trained = makeParam([1, -2, 0.5]);
    const frozen = makeParam([3, 4, 5]);
    trained.setGrad(f64([0.5, -1, 0.25]));
    frozen.setGrad(f64([1, 1, 1]));
    const opt = make([trained, frozen]);
    frozen.setRequiresGrad(false);

    expect(() => opt.step()).not.toThrow();
    expect(Array.from(frozen.tensor.data as Float64Array)).toEqual([3, 4, 5]);
    expect(Array.from(trained.tensor.data as Float64Array)).not.toEqual([1, -2, 0.5]);
  });

  it("LBFGS leaves a frozen parameter out of the optimization", () => {
    const x = makeParam([3, -2]);
    const frozen = makeParam([7]);
    const opt = new LBFGS([x, frozen], { lr: 1, maxIter: 50 });
    frozen.setRequiresGrad(false);
    const closure = (): number => {
      opt.zeroGrad();
      const [a, b] = Array.from(x.tensor.data as Float64Array) as [number, number];
      x.setGrad(f64([2 * a, 2 * b]));
      return a * a + b * b;
    };
    opt.step(closure);
    const [a, b] = Array.from(x.tensor.data as Float64Array) as [number, number];
    expect(Math.abs(a)).toBeLessThan(1e-6);
    expect(Math.abs(b)).toBeLessThan(1e-6);
    expect(Array.from(frozen.tensor.data as Float64Array)).toEqual([7]);
  });

  it("still updates the parameter once it is unfrozen", () => {
    const p = makeParam([1]);
    p.setGrad(f64([1]));
    const opt = new SGD([p], { lr: 0.5 });
    p.setRequiresGrad(false);
    opt.step();
    expect(p.tensor.item()).toBe(1);
    p.setRequiresGrad(true);
    p.setGrad(f64([1]));
    opt.step();
    expect(p.tensor.item()).toBe(0.5);
  });

  it("skips a trainable parameter whose backward() never ran (PyTorch semantics)", () => {
    const p = makeParam([1]);
    const opt = new SGD([p]);
    expect(() => opt.step()).not.toThrow();
    expect(p.tensor.item()).toBe(1);
  });
});

describe("atomic optimizer steps", () => {
  it.each(
    stepFactories
  )("%s changes nothing when a later gradient is not finite", (_name, make) => {
    const a = makeParam([1, 2]);
    const b = makeParam([3, 4, 5]);
    a.setGrad(f64([0.5, 0.5]));
    b.setGrad(f64([1, Number.NaN, 1]));
    const opt = make([a, b]) as unknown as Optimizer<never, never> & { step(): unknown };

    expect(() => opt.step()).toThrow(InvalidParameterError);
    expect(Array.from(a.tensor.data as Float64Array)).toEqual([1, 2]);
    expect(Array.from(b.tensor.data as Float64Array)).toEqual([3, 4, 5]);
    expect(opt.stateDict().state).toHaveLength(0);
    expect(opt.stepCount).toBe(0);

    // The optimizer is still usable after the failed step.
    b.setGrad(f64([1, 1, 1]));
    opt.step();
    expect(opt.stepCount).toBe(1);
    expect(Array.from(a.tensor.data as Float64Array)).not.toEqual([1, 2]);
  });

  it("changes nothing when a parameter value is not finite", () => {
    const a = makeParam([1, 2]);
    const b = makeParam([Number.POSITIVE_INFINITY]);
    a.setGrad(f64([1, 1]));
    b.setGrad(f64([1]));
    const opt = new Adam([a, b]);
    expect(() => opt.step()).toThrow(InvalidParameterError);
    expect(Array.from(a.tensor.data as Float64Array)).toEqual([1, 2]);
    expect(opt.stepCount).toBe(0);
  });

  it("changes nothing when a group option was corrupted after construction", () => {
    const a = makeParam([1]);
    const b = makeParam([2]);
    a.setGrad(f64([1]));
    b.setGrad(f64([1]));
    const opt = new SGD([{ params: [a] }, { params: [b] }], { lr: 0.1 });
    const second = opt.paramGroups[1];
    if (second === undefined) throw new Error("missing group");
    second.options.lr = -1;
    expect(() => opt.step()).toThrow(InvalidParameterError);
    expect(a.tensor.item()).toBe(1);
    expect(opt.stepCount).toBe(0);
  });
});

describe("optimizer step counter", () => {
  it("survives stateDict and loadStateDict", () => {
    const p = makeParam([1, 2]);
    const opt = new Adam([p], { lr: 0.1 });
    for (let i = 0; i < 3; i++) {
      p.setGrad(f64([1, 1]));
      opt.step();
    }
    expect(opt.stepCount).toBe(3);
    const sd = opt.stateDict();
    expect(sd.stepCount).toBe(3);

    const q = makeParam([1, 2]);
    const resumed = new Adam([q], { lr: 0.1 });
    expect(resumed.stepCount).toBe(0);
    resumed.loadStateDict(sd);
    expect(resumed.stepCount).toBe(3);
    q.setGrad(f64([1, 1]));
    resumed.step();
    expect(resumed.stepCount).toBe(4);
  });

  it("is tracked for every optimizer through the base class", () => {
    for (const [, make] of stepFactories) {
      const p = makeParam([1, 2]);
      p.setGrad(f64([1, -1]));
      const opt = make([p]) as unknown as Optimizer<never, never> & { step(): unknown };
      opt.step();
      opt.step();
      expect(opt.stepCount).toBe(2);
      expect(opt.stateDict().stepCount).toBe(2);
    }
  });

  it("rejects an invalid stepCount without changing the optimizer", () => {
    const p = makeParam([1]);
    const opt = new SGD([p]);
    const sd = opt.stateDict();
    expect(() => opt.loadStateDict({ ...sd, stepCount: -1 })).toThrow(DataValidationError);
    expect(() => opt.loadStateDict({ ...sd, stepCount: 1.5 })).toThrow(DataValidationError);
    expect(opt.stepCount).toBe(0);
  });

  it("keeps the LBFGS count and history in one checkpoint", () => {
    const x = makeParam([3, -2]);
    const opt = new LBFGS([x], { lr: 1, maxIter: 5 });
    const closure = (): number => {
      opt.zeroGrad();
      const [a, b] = Array.from(x.tensor.data as Float64Array) as [number, number];
      x.setGrad(f64([2 * a, 20 * b]));
      return a * a + 10 * b * b;
    };
    opt.step(closure);
    opt.step(closure);
    expect(opt.stepCount).toBe(2);
    const sd = opt.stateDict();
    expect(sd.lbfgs.sHistory.length).toBeGreaterThan(0);
    const y = makeParam([3, -2]);
    const other = new LBFGS([y], { lr: 1, maxIter: 5 });
    other.loadStateDict(sd);
    expect(other.stepCount).toBe(2);
  });
});

describe("contiguity of parameters and gradients", () => {
  it("reads a transposed gradient in logical order", () => {
    const p = parameter(
      f64([
        [0, 0, 0],
        [0, 0, 0],
      ])
    );
    const g = transpose(
      f64([
        [1, 2],
        [3, 4],
        [5, 6],
      ])
    );
    expect(g.strides).toEqual([1, 2]);
    p.setGrad(g);
    new SGD([p], { lr: 1 }).step();
    expect(Array.from(p.tensor.data as Float64Array)).toEqual([-1, -3, -5, -2, -4, -6]);
  });

  it("rejects a strided parameter", () => {
    const view = GradTensor.fromTensor(
      transpose(
        f64([
          [1, 2, 3],
          [4, 5, 6],
        ])
      ),
      {
        requiresGrad: true,
      }
    );
    view.setGrad(
      f64([
        [1, 1],
        [1, 1],
        [1, 1],
      ])
    );
    expect(() => new SGD([view], { lr: 1 }).step()).toThrow(ShapeError);
  });
});

describe("maximize on every PyTorch optimizer that has it", () => {
  const grads = [
    [0.5, -1.0, 0.25],
    [0.3, 0.8, -0.6],
    [-0.2, 0.1, 0.9],
  ];

  function run(make: (p: GradTensor[]) => { step(): unknown }, sign: number): number[][] {
    const p = makeParam([1, -2, 0.5]);
    const opt = make([p]);
    const out: number[][] = [];
    for (const g of grads) {
      p.setGrad(f64(g.map((v) => v * sign)));
      opt.step();
      out.push(Array.from(p.tensor.data as Float64Array));
    }
    return out;
  }

  // Reference trajectories from torch 2.12 (float64, maximize=True) on the gradients above.
  const torchReference: Record<string, number[][]> = {
    Adam: [
      [1.0999999979591837, -2.099999998979592, 0.5999999959183675],
      [1.1955253094986753, -2.1035516119709254, 0.5561104536564672],
      [1.242391991547418, -2.1002590916814294, 0.5876869189991726],
    ],
    AdamW: [
      [1.094999998, -2.089999999, 0.5974999960000001],
      [1.185274013301124, -2.085362503670944, 0.5515783506435693],
      [1.2286097771799145, -2.0744379772670882, 0.5811938850513692],
    ],
    Adagrad: [
      [1.0999999999795917, -2.099999999989796, 0.5999999999591836],
      [1.1508018461280312, -2.0357817401492873, 0.5072901244744646],
      [1.1159526644993418, -2.026408848679621, 0.5880435544648307],
    ],
    Adamax: [
      [1.0999999979591837, -2.099999998979592, 0.5999999959183675],
      [1.1784887966528772, -2.103279325534275, 0.5665190167952338],
      [1.2120837112247687, -2.100784210811039, 0.5891041840315558],
    ],
    NAdam: [
      [1.1056451756795147, -2.105645176757527, 0.6056451735234909],
      [1.1641762904193345, -2.0433494995107453, 0.5111198984673075],
      [1.1335978853580726, -2.0331041970291315, 0.5997258386521229],
    ],
    ASGD: [
      [1.04899, -2.09798, 0.524495],
      [1.0779303030479985, -2.0158616560868095, 0.46396571401907744],
      [1.0568419097670285, -2.0038258163513065, 0.5534957657005003],
    ],
    Adadelta: [
      [1.00158110590444, -2.0015811305984803, 0.501581007139847],
      [1.0027645727307022, -2.0001020290316407, 0.4994937816053293],
      [1.0018354559742952, -1.9998346578346713, 0.5019810147007674],
    ],
  };

  const makers: Record<string, (p: GradTensor[]) => { step(): unknown }> = {
    Adam: (p) => new Adam(p, { lr: 0.1, weightDecay: 0.01, maximize: true }),
    AdamW: (p) => new AdamW(p, { lr: 0.1, weightDecay: 0.05, maximize: true }),
    Adagrad: (p) => new Adagrad(p, { lr: 0.1, weightDecay: 0.01, maximize: true }),
    Adamax: (p) => new Adamax(p, { lr: 0.1, weightDecay: 0.01, maximize: true }),
    NAdam: (p) => new Nadam(p, { lr: 0.1, weightDecay: 0.01, maximize: true }),
    ASGD: (p) => new ASGD(p, { lr: 0.1, weightDecay: 0.01, t0: 1, maximize: true }),
    Adadelta: (p) => new AdaDelta(p, { lr: 0.5, weightDecay: 0.01, maximize: true }),
  };

  it.each(Object.keys(torchReference))("%s matches torch with maximize=true", (name) => {
    const make = makers[name];
    const expected = torchReference[name];
    if (make === undefined || expected === undefined) throw new Error("missing reference");
    const actual = run(make, 1);
    for (let step = 0; step < expected.length; step++) {
      for (let i = 0; i < 3; i++) {
        expect(actual[step]?.[i]).toBeCloseTo(expected[step]?.[i] ?? 0, 12);
      }
    }
  });

  it("Lion with maximize follows the update of the negated gradient", () => {
    const maximized = run((p) => new Lion(p, { lr: 0.05, weightDecay: 0.1, maximize: true }), 1);
    const negated = run((p) => new Lion(p, { lr: 0.05, weightDecay: 0.1 }), -1);
    expect(maximized).toEqual(negated);
    expect(maximized).not.toEqual(run((p) => new Lion(p, { lr: 0.05, weightDecay: 0.1 }), 1));
  });

  it("maximize can be set per group and defaults to false", () => {
    const a = makeParam([1]);
    const b = makeParam([1]);
    a.setGrad(f64([1]));
    b.setGrad(f64([1]));
    const opt = new Adam([{ params: [a], maximize: true }, { params: [b] }], { lr: 0.1 });
    opt.step();
    expect(a.tensor.item()).toBeGreaterThan(1);
    expect(b.tensor.item()).toBeLessThan(1);
    expect(() => new Adam([makeParam([1])], { maximize: true })).not.toThrow();
  });
});
