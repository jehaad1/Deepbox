import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  DTypeError,
  getDtype,
  InvalidParameterError,
  ShapeError,
  setDtype,
} from "../../src/core";
import {
  block_diag,
  circulant,
  companion,
  cond,
  denseToCSR,
  det,
  expm,
  hadamard,
  hankel,
  kron,
  logm,
  lstsq,
  matrix_power,
  matrixRank,
  norm,
  solve,
  solve_banded,
  solveTriangular,
  sparseCholeskySolve,
  sparseSolve,
  sqrtm,
  sylvester,
  toeplitz,
  trace,
  vandermonde,
} from "../../src/linalg";
import { blockDiag, matrixPower } from "../../src/linalg/matrix_ops";
import { solveBanded } from "../../src/linalg/solvers";
import { type Tensor, tensor, transpose } from "../../src/ndarray";

const f64 = (data: number[] | number[][] | number[][][]): Tensor =>
  tensor(data as number[][], { dtype: "float64" });

function expectClose(actual: unknown, expected: unknown, tol = 1e-12): void {
  const a = (actual as Tensor).toArray() as number[] | number[][];
  const flatA = (a as unknown[]).flat(Infinity) as number[];
  const flatE = (expected as unknown[]).flat(Infinity) as number[];
  expect(flatA.length).toBe(flatE.length);
  for (let i = 0; i < flatE.length; i++) {
    const e = flatE[i] as number;
    expect(Math.abs((flatA[i] as number) - e)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(e)));
  }
}

describe("v1.5.0 matrix_ops", () => {
  it("matrix_power keeps float64 precision (was rounded to float32)", () => {
    const A = f64([
      [1.1234567890123, 2],
      [3, 4.5],
    ]);
    const P = matrix_power(A, 3);
    expect(P.dtype).toBe("float64");
    // numpy.linalg.matrix_power
    expectClose(
      P,
      [
        [41.89945824781653, 65.13542141466635],
        [97.70313212199953, 151.8657407340738],
      ],
      1e-13
    );
    expectClose(
      matrix_power(A, -2),
      [
        [29.429065432329896, -12.608996403788401],
        [-18.9134946056826, 8.141654830039874],
      ],
      1e-12
    );
    expect(matrix_power(A, 0).dtype).toBe("float64");
  });

  it("matrix_power rejects string dtype and handles an empty matrix", () => {
    expect(() =>
      matrix_power(
        tensor([
          ["a", "b"],
          ["c", "d"],
        ]),
        2
      )
    ).toThrow(DTypeError);
    expect(matrix_power(tensor([[]]).reshape([0, 0]), 3).shape).toEqual([0, 0]);
  });

  it("matrix_power does not alias or modify its input", () => {
    const A = f64([
      [2, 0],
      [0, 3],
    ]);
    const P1 = matrix_power(A, 1);
    expect(P1.data).not.toBe(A.data);
    matrix_power(A, 5);
    expect(A.toArray()).toEqual([
      [2, 0],
      [0, 3],
    ]);
  });

  it("expm returns float64 (cos 1 / sin 1 to full precision)", () => {
    const E = expm(
      f64([
        [0, 1],
        [-1, 0],
      ])
    );
    expect(E.dtype).toBe("float64");
    // scipy.linalg.expm
    expectClose(
      E,
      [
        [0.5403023058681397, 0.8414709848078966],
        [-0.8414709848078965, 0.5403023058681397],
      ],
      1e-15
    );
  });

  it("expm rejects non-finite input instead of looping or returning NaN", () => {
    expect(() => expm(f64([[Number.POSITIVE_INFINITY]]))).toThrow(DataValidationError);
    expect(() => expm(f64([[Number.NaN]]))).toThrow(DataValidationError);
  });

  it("sqrtm of a singular positive semi-definite matrix is real (was NaN-prone)", () => {
    const S = sqrtm(
      f64([
        [1, 1],
        [1, 1],
      ])
    );
    expect(S.dtype).toBe("float64");
    expectClose(
      S,
      [
        [Math.SQRT1_2, Math.SQRT1_2],
        [Math.SQRT1_2, Math.SQRT1_2],
      ],
      1e-15
    );
  });

  it("sqrtm and logm throw a typed error for eigenvalues without a real result (was NaN)", () => {
    const M = f64([
      [-1, 0],
      [0, 4],
    ]);
    expect(() => sqrtm(M)).toThrow(DataValidationError);
    expect(() => logm(M)).toThrow(DataValidationError);
    // zero eigenvalue: log is undefined
    expect(() =>
      logm(
        f64([
          [1, 1],
          [1, 1],
        ])
      )
    ).toThrow(DataValidationError);
  });

  it("sqrtm and logm handle defective and complex-eigenvalue matrices", () => {
    // Jordan block: eig-based evaluation failed ("Matrix is singular")
    expectClose(
      sqrtm(
        f64([
          [1, 1],
          [0, 1],
        ])
      ),
      [
        [1, 0.5],
        [0, 1],
      ],
      1e-14
    );
    expectClose(
      logm(
        f64([
          [1, 1],
          [0, 1],
        ])
      ),
      [
        [0, 1],
        [0, 0],
      ],
      1e-14
    );
    const J = f64([
      [4, 1, 0],
      [0, 4, 1],
      [0, 0, 4],
    ]);
    // scipy.linalg.sqrtm / logm
    expectClose(
      sqrtm(J),
      [
        [2, 0.25, -0.015625],
        [0, 2, 0.25],
        [0, 0, 2],
      ],
      1e-14
    );
    expectClose(
      logm(J),
      [
        [1.3862943611198906, 0.25, -0.03125000000000007],
        [0, 1.3862943611198906, 0.25],
        [0, 0, 1.3862943611198906],
      ],
      1e-13
    );
    // 90 degree rotation: complex eigenvalues +-i
    const R = f64([
      [0, -1],
      [1, 0],
    ]);
    expectClose(
      sqrtm(R),
      [
        [Math.SQRT1_2, -Math.SQRT1_2],
        [Math.SQRT1_2, Math.SQRT1_2],
      ],
      1e-14
    );
    expectClose(
      logm(R),
      [
        [0, -Math.PI / 2],
        [Math.PI / 2, 0],
      ],
      1e-14
    );
  });

  it("sqrtm and logm match scipy for a general matrix with positive eigenvalues", () => {
    const M = f64([
      [2, 1],
      [0.5, 3],
    ]);
    expectClose(
      sqrtm(M),
      [
        [1.395851935785898, 0.3212393916155419],
        [0.1606196958077709, 1.7170913274014399],
      ],
      1e-13
    );
    expectClose(
      logm(M),
      [
        [0.6437435629862357, 0.4172609662659538],
        [0.20863048313297688, 1.0610045292521895],
      ],
      1e-13
    );
    expectClose(
      logm(
        f64([
          [2, 1],
          [1, 2],
        ])
      ),
      [
        [0.5493061443340548, 0.5493061443340548],
        [0.5493061443340548, 0.5493061443340548],
      ],
      1e-14
    );
  });

  it("sqrtm and logm of non-symmetric matrices do not depend on the absolute scale", () => {
    // The symmetry test used an absolute floor of 1, so tiny non-symmetric matrices were
    // symmetrised, and the Denman-Beavers iteration needed hundreds of steps for huge ones.
    const base = [
      [2, 1],
      [0.5, 3],
    ];
    const sqrtBase = [
      [1.395851935785898, 0.3212393916155419],
      [0.1606196958077709, 1.7170913274014399],
    ];
    const logBase = [
      [0.6437435629862357, 0.4172609662659538],
      [0.20863048313297688, 1.0610045292521895],
    ];
    for (const [scale, rootScale, logShift] of [
      [1e-30, 1e-15, -30 * Math.LN10],
      [1e100, 1e50, 100 * Math.LN10],
    ] as const) {
      const A = f64(base.map((r) => r.map((v) => v * scale)));
      const S = (sqrtm(A).toArray() as number[][]).map((r) => r.map((v) => v / rootScale));
      expectClose(f64(S), sqrtBase, 1e-13);
      const L = logm(A).toArray() as number[][];
      expectClose(
        f64(L),
        logBase.map((r, i) => r.map((v, j) => v + (i === j ? logShift : 0))),
        1e-12
      );
    }
    // scipy.linalg.sqrtm of a scaled Jordan block: 1e100 * (I + 0.1 N) -> 1e100 * (I + 0.05 N)
    const J = sqrtm(
      f64([
        [1e200, 1e199],
        [0, 1e200],
      ])
    ).toArray() as number[][];
    expectClose(
      f64(J.map((r) => r.map((v) => v / 1e100))),
      [
        [1, 0.05],
        [0, 1],
      ],
      1e-13
    );
  });

  it("matrix functions reject string dtype and non-square input", () => {
    expect(() => expm(tensor([["a"]]))).toThrow(DTypeError);
    expect(() => sqrtm(f64([[1, 2, 3]]))).toThrow(ShapeError);
    expect(() => logm(f64([1, 2]))).toThrow(ShapeError);
  });

  it("kron and block_diag reject string dtype and return float64", () => {
    expect(() => kron(tensor([["a"]]), f64([[1]]))).toThrow(DTypeError);
    expect(() => block_diag(tensor([["a"]]))).toThrow(DTypeError);
    expect(kron(tensor([[1, 2]]), tensor([[3], [4]])).dtype).toBe("float64");
    // rectangular blocks are supported
    const D = block_diag(f64([[1, 2, 3]]), f64([[4], [5]]));
    expect(D.shape).toEqual([3, 4]);
    expect(D.toArray()).toEqual([
      [1, 2, 3, 0],
      [0, 0, 0, 4],
      [0, 0, 0, 5],
    ]);
  });

  it("toeplitz keeps c[0] on the diagonal for non-float default dtypes (was r[0])", () => {
    const prev = getDtype();
    try {
      setDtype("int32");
      expect(toeplitz([1, 2, 3], [9, 5, 6, 7]).toArray()).toEqual([
        [1, 5, 6, 7],
        [2, 1, 5, 6],
        [3, 2, 1, 5],
      ]);
    } finally {
      setDtype(prev);
    }
    // scipy.linalg.toeplitz
    expect(toeplitz([1, 2, 3], [9, 5, 6, 7]).toArray()).toEqual([
      [1, 5, 6, 7],
      [2, 1, 5, 6],
      [3, 2, 1, 5],
    ]);
  });

  it("hadamard rejects non powers of two even when 32-bit masking would accept them", () => {
    // (2^32 + 1) & 2^32 is 0 in 32-bit arithmetic; this used to exhaust memory
    expect(() => hadamard(4294967297)).toThrow(InvalidParameterError);
    expect(() => hadamard(2 ** 53 - 1)).toThrow(InvalidParameterError);
    expect(() => hadamard(2 ** 40)).toThrow(InvalidParameterError);
    // scipy.linalg.hadamard(4)
    expect(hadamard(4).toArray()).toEqual([
      [1, 1, 1, 1],
      [1, -1, 1, -1],
      [1, 1, -1, -1],
      [1, -1, -1, 1],
    ]);
    expect(hadamard(1).toArray()).toEqual([[1]]);
  });

  it("vandermonde matches numpy.vander and rejects a non-integer N", () => {
    expect(vandermonde([1, 2, 3], 4).toArray()).toEqual([
      [1, 1, 1, 1],
      [8, 4, 2, 1],
      [27, 9, 3, 1],
    ]);
    expect(vandermonde([1.5, -2, 0], 3, true).toArray()).toEqual([
      [1, 1.5, 2.25],
      [1, -2, 4],
      [1, 0, 0],
    ]);
    expect(() => vandermonde([1, 2, 3], 2.5)).toThrow(InvalidParameterError);
  });

  it("hankel, circulant and companion match scipy", () => {
    expect(hankel([1, 2, 3], [3, 4, 5, 6]).toArray()).toEqual([
      [1, 2, 3, 4],
      [2, 3, 4, 5],
      [3, 4, 5, 6],
    ]);
    expect(hankel([1, 2, 3]).toArray()).toEqual([
      [1, 2, 3],
      [2, 3, 0],
      [3, 0, 0],
    ]);
    expect(circulant([1, 2, 3]).toArray()).toEqual([
      [1, 3, 2],
      [2, 1, 3],
      [3, 2, 1],
    ]);
    expect(companion([2, -10, 31, -30]).toArray()).toEqual([
      [5, -15.5, 15],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    expect(() => companion([Number.NaN, 1, 2])).toThrow(InvalidParameterError);
  });
});

describe("v1.5.0 norms", () => {
  it("2-norm does not overflow or underflow", () => {
    expect(norm(f64([3e200, 4e200])) / 5e200).toBeCloseTo(1, 14);
    expect(norm(f64([3e-200, 4e-200]))).toBeCloseTo(5e-200, 214);
    expect(norm(f64([3e-200, 4e-200])) / 5e-200).toBeCloseTo(1, 14);
    expect(
      norm(
        f64([
          [3e200, 0],
          [0, 4e200],
        ]),
        "fro"
      ) / 5e200
    ).toBeCloseTo(1, 14);
    expect(
      norm(
        f64([
          [3e200, 0],
          [0, 4e200],
        ])
      ) / 5e200
    ).toBeCloseTo(1, 14);
    expect(
      norm(
        f64([
          [3e-200, 0],
          [0, 4e-200],
        ])
      ) / 5e-200
    ).toBeCloseTo(1, 14);
    // along an axis, and as a matrix norm over two axes
    const big = f64([
      [3e200, 4e200],
      [6e200, 8e200],
    ]);
    expectClose(norm(big, 2, 1), [5e200, 1e201], 1e-14);
    expect((norm(big, "fro", [0, 1]) as number) / 1.118033988749895e201).toBeCloseTo(1, 14);
  });

  it("general p-norms stay finite for extreme values", () => {
    const x = f64([1e200, 1e200]);
    expect(norm(x, 3) / (1e200 * 2 ** (1 / 3))).toBeCloseTo(1, 14);
    // p < 0: smallest element dominates; value is huge but finite
    expect(norm(f64([1e200, 2e200]), -2) / 8.94427190999916e199).toBeCloseTo(1, 14);
    // reference values from numpy.linalg.norm
    expect(norm(f64([1, -2, 3]), 3)).toBeCloseTo(3.3019272488946263, 14);
    expect(norm(f64([1, -2, 3]), -2)).toBeCloseTo(0.8571428571428571, 14);
    expect(norm(f64([1, 2, 3]), 0.5)).toBeCloseTo(17.191508225450303, 12);
  });

  it("inputs with more than two dimensions give the flat 2-norm when ord is omitted", () => {
    const data: number[][][] = [];
    let v = 1;
    for (let i = 0; i < 2; i++) {
      const plane: number[][] = [];
      for (let j = 0; j < 3; j++) {
        const row: number[] = [];
        for (let k = 0; k < 4; k++) row.push(v++ * 0.37);
        plane.push(row);
      }
      data.push(plane);
    }
    const X = tensor(data as unknown as number[][], { dtype: "float64" });
    // numpy.linalg.norm(X) on arange(1, 25).reshape(2, 3, 4) * 0.37
    expect(norm(X)).toBeCloseTo(25.9, 12);
    // an explicit ord on a 3-D array is still an error without axes
    expect(() => norm(X, "fro")).toThrow(ShapeError);
    expect(() => norm(X, 1)).toThrow(ShapeError);
    // nuclear norm over two axes: numpy.linalg.norm(X, 'nuc', axis=(0, 2))
    expectClose(
      norm(X, "nuc", [0, 2]),
      [11.60250188079048, 15.05373209186834, 18.80579830514372],
      1e-12
    );
  });

  it("keepdims is honoured when every axis is reduced", () => {
    const A = f64([
      [1, 2],
      [3, 4],
    ]);
    const k = norm(A, undefined, undefined, true) as Tensor;
    expect(k.shape).toEqual([1, 1]);
    expect(k.dtype).toBe("float64");
    expect(k.toArray()).toEqual([[Math.sqrt(30)]]);
    const kv = norm(f64([3, 4]), undefined, undefined, true) as Tensor;
    expect(kv.shape).toEqual([1]);
    expect(kv.toArray()).toEqual([5]);
    expect((norm(A, 2, [], true) as Tensor).shape).toEqual([1, 1]);
  });

  it("axis results are float64 tensors", () => {
    const r = norm(
      f64([
        [1.123456789012, 2],
        [3, 4],
      ]),
      2,
      0
    ) as Tensor;
    expect(r.dtype).toBe("float64");
    expect((r.toArray() as number[])[0]).toBeCloseTo(Math.hypot(1.123456789012, 3), 14);
    const kd = norm(
      f64([
        [1, 2],
        [3, 4],
      ]),
      1,
      [0, 1],
      true
    ) as Tensor;
    expect(kd.shape).toEqual([1, 1]);
    expect(kd.toArray()).toEqual([[6]]);
  });

  it("nuclear norm rejects a single axis with a clear message", () => {
    expect(() =>
      norm(
        f64([
          [1, 2],
          [3, 4],
        ]),
        "nuc",
        0
      )
    ).toThrow(/axis is only supported/);
  });

  it("cond supports the numpy orders (reference: numpy.linalg.cond)", () => {
    const M = f64([
      [1, 2],
      [3, 4],
    ]);
    expect(cond(M)).toBeCloseTo(14.933034373659263, 10);
    expect(cond(M, 2)).toBeCloseTo(14.933034373659263, 10);
    expect(cond(M, -2)).toBeCloseTo(0.0669656263407472, 12);
    expect(cond(M, 1)).toBeCloseTo(21, 10);
    expect(cond(M, -1)).toBeCloseTo(6, 10);
    expect(cond(M, Number.POSITIVE_INFINITY)).toBeCloseTo(21, 10);
    expect(cond(M, Number.NEGATIVE_INFINITY)).toBeCloseTo(6, 10);
    expect(cond(M, "fro")).toBeCloseTo(15, 10);
    expect(cond(M, "nuc")).toBeCloseTo(17, 10);
  });

  it("cond edge cases", () => {
    const S = f64([
      [1, 2],
      [2, 4],
    ]);
    expect(cond(S, 1)).toBe(Number.POSITIVE_INFINITY);
    expect(cond(S, -2)).toBe(0);
    expect(cond(S)).toBe(Number.POSITIVE_INFINITY);
    // p = 1 needs an inverse, so the matrix must be square
    expect(() =>
      cond(
        f64([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        1
      )
    ).toThrow(ShapeError);
    expect(() =>
      cond(
        f64([
          [1, 2],
          [3, 4],
        ]),
        3
      )
    ).toThrow(InvalidParameterError);
    expect(() =>
      cond(
        f64([
          [1, 2],
          [3, 4],
        ]),
        Number.NaN
      )
    ).toThrow(InvalidParameterError);
    // rectangular matrices still work for the singular-value based orders
    expect(
      cond(
        f64([
          [1, 2],
          [3, 4],
          [5, 6],
        ])
      )
    ).toBeCloseTo(18.52130534125813, 10);
    // singular values of a tiny matrix do not underflow in the Frobenius condition number
    expect(
      cond(
        f64([
          [1e-200, 0],
          [0, 2e-200],
        ]),
        "fro"
      )
    ).toBeCloseTo(Math.sqrt(5) * Math.sqrt(1.25), 12);
  });
});

describe("v1.5.0 properties", () => {
  it("trace keeps float64 precision (was rounded to float32)", () => {
    const t = trace(
      f64([
        [1.1234567890123, 2],
        [3, 4.5],
      ])
    );
    expect(t.dtype).toBe("float64");
    expect(t.shape).toEqual([1]);
    expect((t.toArray() as number[])[0]).toBe(1.1234567890123 + 4.5);
    const batched = trace(
      f64([
        [
          [1.1234567890123, 0],
          [0, 1],
        ],
        [
          [2, 0],
          [0, 2.5],
        ],
      ]),
      0,
      1,
      2
    );
    expect(batched.dtype).toBe("float64");
    expect((batched.toArray() as number[])[0]).toBe(1.1234567890123 + 1);
  });

  it("trace rejects string dtype with DTypeError", () => {
    expect(() =>
      trace(
        tensor([
          ["a", "b"],
          ["c", "d"],
        ])
      )
    ).toThrow(DTypeError);
  });

  it("det survives transient overflow and underflow of the pivot product", () => {
    const D = f64([
      [1e200, 0, 0, 0],
      [0, 1e200, 0, 0],
      [0, 0, 1e-200, 0],
      [0, 0, 0, 1e-200],
    ]);
    // numpy.linalg.det(diag([1e200, 1e200, 1e-200, 1e-200])) == 1.0
    expect(det(D)).toBeCloseTo(1, 12);
    const E = f64([
      [1e-200, 0, 0],
      [0, 1e-200, 0],
      [0, 0, 1e300],
    ]);
    expect(det(E) / 1e-100).toBeCloseTo(1, 12);
    // a subnormal partial product (1e-160 * 1e-160) must not cost precision
    expect(
      det(
        f64([
          [1e-160, 0, 0],
          [0, 1e-160, 0],
          [0, 0, 1e300],
        ])
      ) / 1e-20
    ).toBeCloseTo(1, 14);
    // ordinary values are returned exactly as the plain product
    expect(
      det(
        f64([
          [1, 2],
          [3, 4],
        ])
      )
    ).toBe(-2);
    // a true overflow is still Infinity, a true underflow 0
    expect(
      det(
        f64([
          [1e200, 0],
          [0, 1e200],
        ])
      )
    ).toBe(Number.POSITIVE_INFINITY);
    expect(
      det(
        f64([
          [1e-200, 0],
          [0, 1e-200],
        ])
      )
    ).toBe(0);
  });

  it("matrixRank default tolerance follows the dtype like numpy (float32 vs float64)", () => {
    const data = [
      [0.1, 0.7],
      [0.3, 2.1],
    ];
    // numpy.linalg.matrix_rank: 1 for float32 input, 2 for the same (float32-rounded)
    // values held in float64
    expect(matrixRank(tensor(data, { dtype: "float32" }))).toBe(1);
    expect(
      matrixRank(
        tensor(
          data.map((r) => r.map(Math.fround)),
          { dtype: "float64" }
        )
      )
    ).toBe(2);
    // an explicit tolerance always wins
    expect(matrixRank(tensor(data, { dtype: "float32" }), 1e-12)).toBe(2);
  });
});

describe("v1.5.0 solvers", () => {
  const L = [
    [2, 0, 0],
    [3, 4, 0],
    [5, 6, 7],
  ];
  const B = [
    [1, 2],
    [3, 4],
    [5, 6],
  ];

  it("solveTriangular supports trans and unitDiagonal (scipy.linalg.solve_triangular)", () => {
    expectClose(
      solveTriangular(f64(L), f64(B), true, { trans: true }),
      [
        [-0.8035714285714288, -0.7142857142857144],
        [-0.3214285714285712, -0.2857142857142856],
        [0.7142857142857142, 0.8571428571428571],
      ],
      1e-13
    );
    expectClose(
      solveTriangular(f64(L), f64(B), true, { unitDiagonal: true }),
      [
        [1, 2],
        [0, -2],
        [0, 8],
      ],
      1e-13
    );
    const U = L[0]?.map((_, j) => L.map((row) => row[j] as number)) as number[][];
    expectClose(
      solveTriangular(f64(U), f64(B), false, { trans: true }),
      [
        [0.5, 1],
        [0.375, 0.25],
        [0.03571428571428571, -0.07142857142857142],
      ],
      1e-13
    );
  });

  it("solveTriangular multi-RHS equals column-by-column solves and ignores the other triangle", () => {
    const dirty = [
      [2, 99, 99],
      [3, 4, 99],
      [5, 6, 7],
    ];
    const X = solveTriangular(f64(dirty), f64(B), true);
    for (let c = 0; c < 2; c++) {
      const col = f64(B.map((r) => r[c] as number));
      const x = solveTriangular(f64(L), col, true).toArray() as number[];
      for (let i = 0; i < 3; i++) {
        expect((X.toArray() as number[][])[i]?.[c]).toBeCloseTo(x[i] as number, 14);
      }
    }
  });

  it("solveTriangular only reports a zero diagonal when the diagonal is read", () => {
    const Z = f64([
      [0, 0],
      [1, 1],
    ]);
    expect(() => solveTriangular(Z, f64([1, 1]), true)).toThrow(/singular/i);
    expect(() => solveTriangular(Z, f64([1, 1]), true, { unitDiagonal: true })).not.toThrow();
  });

  it("solve does not modify b", () => {
    const b = f64([9, 8]);
    solve(
      f64([
        [3, 1],
        [1, 2],
      ]),
      b
    );
    expect(b.toArray()).toEqual([9, 8]);
  });

  it("lstsq residuals for a problem with no unknowns are ||b||^2 (was 0)", () => {
    // numpy.linalg.lstsq(np.zeros((3, 0)), [1, 2, 3]) -> residuals [14]
    const r = lstsq(f64([[], [], []]).reshape([3, 0]), f64([1, 2, 3]));
    expect(r.x.shape).toEqual([0]);
    expect(r.rank).toBe(0);
    expect(r.residuals.toArray()).toEqual([14]);
    const r2 = lstsq(
      f64([[], [], []]).reshape([3, 0]),
      f64([
        [1, 0],
        [2, 1],
        [3, 1],
      ])
    );
    expect(r2.x.shape).toEqual([0, 2]);
    expect(r2.residuals.toArray()).toEqual([14, 2]);
    // no equations: zero solution, zero residual
    const r3 = lstsq(f64([[], []]).reshape([0, 2]), f64([]));
    expect(r3.x.toArray()).toEqual([0, 0]);
    expect(r3.residuals.toArray()).toEqual([0]);
  });

  it("lstsq returns the minimum-norm solution for rank-deficient multi-RHS input", () => {
    const A = f64([
      [1, 2],
      [2, 4],
      [3, 6],
    ]);
    const b = f64([
      [1, 2],
      [2, 4],
      [3, 7],
    ]);
    const r = lstsq(A, b);
    // numpy.linalg.lstsq
    expect(r.rank).toBe(1);
    expectClose(
      r.x,
      [
        [0.1999999999999999, 0.4428571428571426],
        [0.3999999999999999, 0.8857142857142855],
      ],
      1e-12
    );
    const res = r.residuals.toArray() as number[];
    expect(res[0]).toBeLessThan(1e-20);
    expect(res[1]).toBeCloseTo(0.35714285714285715, 12);
  });

  it("lstsq with 1-D b matches numpy and keeps float64 residuals", () => {
    const r = lstsq(
      f64([
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
      ]),
      f64([1, 3, 2, 5])
    );
    expectClose(r.x, [0, 1.1], 1e-12);
    expect(r.residuals.dtype).toBe("float64");
    expect((r.residuals.toArray() as number[])[0]).toBeCloseTo(2.7, 12);
    expectClose(r.s, [5.779378813233887, 0.773809106397227], 1e-12);
  });

  describe("solve_banded", () => {
    it("handles systems with tiny entries (absolute 1e-15 pivot test removed)", () => {
      const x = solve_banded([0, 0], f64([[1e-20, 2e-20]]), f64([1e-20, 4e-20]));
      expectClose(x, [1, 2], 1e-14);
      const ab = f64([
        [0, -1e-20, -1e-20],
        [4e-20, 4e-20, 4e-20],
        [-1e-20, -1e-20, 0],
      ]);
      const y = solve_banded([1, 1], ab, f64([3e-20, 2e-20, 3e-20]));
      expectClose(y, [1, 1, 1], 1e-14);
    });

    it("uses pivoting for non-dominant tridiagonal and general bands (scipy reference)", () => {
      // tridiagonal with a tiny leading pivot
      const x = solve_banded(
        [1, 1],
        f64([
          [0, 1, 1],
          [1e-10, 1, 1],
          [1, 1, 0],
        ]),
        f64([1, 2, 3])
      );
      expectClose(x, [-1, 1.0000000001, 1.9999999999], 1e-12);

      // l=2, u=1 with zero diagonal entries; two right-hand sides
      const ab = f64([
        [0, 2, 3, 1, 2],
        [0, 0, 0, 0, 0.5],
        [1, 1, 5, 3, 0],
        [4, 2, 1, 0, 0],
      ]);
      const b = f64([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
      ]);
      expectClose(
        solve_banded([2, 1], ab, b),
        [
          [0.4825174825174825, 0.5174825174825174],
          [0.5, 1.0],
          [0.839160839160839, 1.1608391608391606],
          [2.56993006993007, 2.9300699300699304],
          [0.9020979020979024, 0.09790209790209832],
        ],
        1e-12
      );
    });

    it("multi-RHS solve equals the single-RHS solves", () => {
      const ab = f64([
        [0, 2, 3, 1, 2],
        [0, 0, 0, 0, 0.5],
        [1, 1, 5, 3, 0],
        [4, 2, 1, 0, 0],
      ]);
      const cols = [
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 10],
      ];
      const both = solve_banded(
        [2, 1],
        ab,
        f64([0, 1, 2, 3, 4].map((i) => [cols[0]?.[i] as number, cols[1]?.[i] as number]))
      ).toArray() as number[][];
      for (let c = 0; c < 2; c++) {
        const single = solve_banded([2, 1], ab, f64(cols[c] as number[])).toArray() as number[];
        for (let i = 0; i < 5; i++) {
          expect(both[i]?.[c]).toBeCloseTo(single[i] as number, 13);
        }
      }
    });

    it("empty system keeps the number of right-hand sides (was shape [0, 0])", () => {
      const empty = tensor([[], [], []]).reshape([3, 0]);
      expect(solve_banded([1, 1], empty, tensor([[], []]).reshape([0, 2])).shape).toEqual([0, 2]);
      expect(solve_banded([1, 1], empty, tensor([])).shape).toEqual([0]);
      expect(() => solve_banded([1, 1], empty, f64([1]))).toThrow(ShapeError);
    });

    it("validates the band widths as parameters", () => {
      expect(() => solve_banded([-1, 0], f64([[1]]), f64([1]))).toThrow(InvalidParameterError);
      expect(() => solve_banded([0.5, 0], f64([[1]]), f64([1]))).toThrow(InvalidParameterError);
      expect(() => solve_banded([0, 0], f64([[0, 0]]), f64([1, 1]))).toThrow(/Singular banded/);
    });
  });

  describe("sparse solvers", () => {
    it("reject malformed CSR input instead of silently dropping entries", () => {
      const bad = {
        n: 2,
        values: new Float64Array([1, 1, 1]),
        colIndices: new Int32Array([0, 5, 1]),
        rowPointers: new Int32Array([0, 2, 3]),
      };
      expect(() => sparseSolve(bad, f64([1, 2]))).toThrow(/outside/);
      expect(() => sparseCholeskySolve(bad, f64([1, 2]))).toThrow(/outside/);
      expect(() =>
        sparseSolve(
          { ...bad, colIndices: new Int32Array([0, 1, 1]), rowPointers: new Int32Array([0, 2]) },
          f64([1, 2])
        )
      ).toThrow(ShapeError);
      expect(() =>
        sparseSolve(
          { ...bad, colIndices: new Int32Array([0, 1, 1]), rowPointers: new Int32Array([0, 3, 2]) },
          f64([1, 2])
        )
      ).toThrow(DataValidationError);
      expect(() =>
        sparseSolve(
          {
            ...bad,
            colIndices: new Int32Array([0, 1, 1]),
            values: new Float64Array([1, Number.NaN, 1]),
          },
          f64([1, 2])
        )
      ).toThrow(DataValidationError);
    });

    it("accept spare capacity in values and colIndices past rowPointers[n]", () => {
      const spare = {
        n: 1,
        values: new Float64Array([2, 99]),
        colIndices: new Int32Array([0, 7]),
        rowPointers: new Int32Array([0, 1]),
      };
      expect(sparseSolve(spare, f64([4])).toArray()).toEqual([2]);
      expect((sparseCholeskySolve(spare, f64([4])).toArray() as number[])[0]).toBeCloseTo(2, 14);
      // rowPointers promising more entries than the arrays hold is still an error
      expect(() =>
        sparseSolve({ ...spare, rowPointers: new Int32Array([0, 3]) }, f64([4]))
      ).toThrow(DataValidationError);
    });

    it("sum repeated CSR entries like scipy", () => {
      const dup = {
        n: 1,
        values: new Float64Array([1, 1]),
        colIndices: new Int32Array([0, 0]),
        rowPointers: new Int32Array([0, 2]),
      };
      expect(sparseSolve(dup, f64([1])).toArray()).toEqual([0.5]);
      expect((sparseCholeskySolve(dup, f64([2])).toArray() as number[])[0]).toBeCloseTo(1, 14);
    });

    it("sparseSolve handles a matrix needing pivoting and tiny scaling", () => {
      const A = [
        [0, 2, 0, 1],
        [3, 0, 0, 0],
        [0, 1, 4, 0],
        [1, 0, 0, 5],
      ];
      const x = sparseSolve(denseToCSR(f64(A)), f64([1, 2, 3, 4]));
      // numpy.linalg.solve
      expectClose(
        x,
        [0.6666666666666666, 0.16666666666666663, 0.7083333333333334, 0.6666666666666667],
        1e-14
      );
      // an absolute 1e-15 pivot threshold used to call this singular
      const tiny = sparseSolve(
        denseToCSR(
          f64([
            [1e-20, 0],
            [0, 1e-20],
          ])
        ),
        f64([1e-20, 2e-20])
      );
      expectClose(tiny, [1, 2], 1e-14);
      expect(() =>
        sparseSolve(
          denseToCSR(
            f64([
              [1, 2],
              [2, 4],
            ])
          ),
          f64([1, 2])
        )
      ).toThrow(/Singular sparse matrix/);
    });

    it("sparseCholeskySolve matches numpy on a banded SPD matrix and handles tiny scaling", () => {
      const A = [
        [4, -1, 0.5, 0, 0, 0],
        [-1, 4, -1, 0.5, 0, 0],
        [0.5, -1, 4, -1, 0.5, 0],
        [0, 0.5, -1, 4, -1, 0.5],
        [0, 0, 0.5, -1, 4, -1],
        [0, 0, 0, 0.5, -1, 4],
      ];
      const b = f64([1, 2, 3, 4, 5, 6]);
      // numpy.linalg.solve
      expectClose(
        sparseCholeskySolve(denseToCSR(f64(A)), b),
        [
          0.2865785152536514, 0.640739780485697, 0.988851438942183, 1.4249416645060928,
          1.9339728631924638, 1.8053755077348543,
        ],
        1e-13
      );
      // an absolute 1e-15 diagonal threshold used to reject this valid SPD matrix
      const tiny = sparseCholeskySolve(
        denseToCSR(
          f64([
            [1e-40, 0],
            [0, 1e-40],
          ])
        ),
        f64([1e-20, 2e-20])
      );
      expectClose(tiny, [1e20, 2e20], 1e-14);
      expect(() =>
        sparseCholeskySolve(
          denseToCSR(
            f64([
              [1, 2],
              [2, 1],
            ])
          ),
          f64([1, 1])
        )
      ).toThrow(/not positive definite/);
    });

    it("sparseCholeskySolve reads only the lower triangle", () => {
      const lowerOnly = {
        n: 2,
        values: new Float64Array([4, 1, 3]),
        colIndices: new Int32Array([0, 0, 1]),
        rowPointers: new Int32Array([0, 1, 3]),
      };
      const full = denseToCSR(
        f64([
          [4, 1],
          [1, 3],
        ])
      );
      expectClose(
        sparseCholeskySolve(lowerOnly, f64([1, 2])),
        sparseCholeskySolve(full, f64([1, 2])).toArray(),
        1e-15
      );
    });
  });

  describe("sylvester", () => {
    it("detects a singular equation whose shared eigenvalue is only exact up to rounding", () => {
      // B = -S A S^-1, so A and -B share all eigenvalues up to ~1e-16.
      // The old absolute 1e-30 threshold returned a solution of size 1e16.
      const A = f64([
        [0.03419276725318417, 1.3597475403099617, 1.2247210785859324],
        [-0.5103070767876675, -0.2979695111064471, -0.5273841930334252],
        [0.5697263575719601, -0.056064439045617594, 0.7468856162565439],
      ]);
      const Bm = f64([
        [-2.905348191018389, -11.459767018175768, 7.228048769293259],
        [0.4134734033188285, 2.3842586060077804, -1.5878529983379717],
        [-0.27246237236227683, -0.0008952009260177959, 0.0379807126073282],
      ]);
      const C = f64([
        [-0.15278617857019708, 0.685698610809258, -0.8703406419471712],
        [-1.5143835037313955, 0.39498186274953, -0.6705658236878794],
        [-1.9203405901180286, -0.8140536639453595, -0.467597558892747],
      ]);
      expect(() => sylvester(A, Bm, C)).toThrow(/singular/);
    });

    it("solves problems with tiny entries and complex Schur blocks", () => {
      // diag-scaled problem: X_ij = C_ij / (a_i + b_j)
      const X = sylvester(
        f64([
          [1e-20, 0],
          [0, 2e-20],
        ]),
        f64([
          [3e-20, 0],
          [0, 4e-20],
        ]),
        f64([
          [4e-20, 5e-20],
          [6e-20, 12e-20],
        ])
      );
      expectClose(
        X,
        [
          [1, 1],
          [1.2, 2],
        ],
        1e-13
      );

      // rotation blocks on both sides (scipy.linalg.solve_sylvester reference)
      const A = f64([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 2],
      ]);
      const Bm = f64([
        [1, -2],
        [2, 1],
      ]);
      const C = f64([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const Y = sylvester(A, Bm, C);
      // residual check A Y + Y B = C
      const y = Y.toArray() as number[][];
      const a = A.toArray() as number[][];
      const bb = Bm.toArray() as number[][];
      const c = C.toArray() as number[][];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
          let s = 0;
          for (let k = 0; k < 3; k++) s += (a[i]?.[k] as number) * (y[k]?.[j] as number);
          for (let k = 0; k < 2; k++) s += (y[i]?.[k] as number) * (bb[k]?.[j] as number);
          expect(s).toBeCloseTo(c[i]?.[j] as number, 12);
        }
      }
    });

    it("handles empty dimensions", () => {
      const X = sylvester(
        tensor([[]]).reshape([0, 0]),
        f64([
          [1, 0],
          [0, 1],
        ]),
        tensor([[], []]).reshape([0, 2])
      );
      expect(X.shape).toEqual([0, 2]);
    });
  });
});

describe("v1.5.0 non-contiguous views and aliases", () => {
  const A = f64([
    [1, 2],
    [3, 4],
  ]);
  const T = transpose(A); // strides [1, 2]: [[1, 3], [2, 4]]

  it("read transposed views through their strides", () => {
    expect(matrix_power(T, 2).toArray()).toEqual([
      [7, 15],
      [10, 22],
    ]);
    expect(kron(T, f64([[1, 1]])).toArray()).toEqual([
      [1, 1, 3, 3],
      [2, 2, 4, 4],
    ]);
    expect(block_diag(T).toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
    expect(norm(T, 1)).toBe(7);
    expectClose(norm(T, 2, 0), [Math.sqrt(5), 5], 1e-14);
    expect((trace(T, 1).toArray() as number[])[0]).toBe(3);
    // only the upper triangle [[1, 3], [., 4]] of the view is read
    expect(solveTriangular(T, f64([1, 2]), false).toArray()).toEqual([-0.5, 0.5]);
    // scipy.linalg.expm([[1, 3], [2, 4]])
    expectClose(
      expm(T),
      [
        [51.968956198705, 112.10484685050483],
        [74.7365645670032, 164.07380304920986],
      ],
      1e-13
    );
  });

  it("camelCase aliases point at the same functions", () => {
    expect(matrixPower).toBe(matrix_power);
    expect(blockDiag).toBe(block_diag);
    expect(solveBanded).toBe(solve_banded);
  });
});
