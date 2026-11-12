import { describe, expect, it } from "vitest";
import { DataValidationError } from "../../src/core";
import { cond, det, eig, inv, matrixRank, norm, svd } from "../../src/linalg";
import { tensor } from "../../src/ndarray";

// Reference: numpy.linalg.norm / cond / det with NaN and Infinity inputs (numpy 2.x).
// numpy.linalg.matrix_rank, svd and the SVD-based norms (ord 2, -2, 'nuc') raise LinAlgError
// ("SVD did not converge"), so those keep throwing here.

const f = (a: number[] | number[][]) => tensor(a, { dtype: "float64" });
const N = Number.NaN;
const I = Number.POSITIVE_INFINITY;

describe("norm with NaN and Infinity", () => {
  it("vector norms follow IEEE", () => {
    expect(norm(f([N, 1]))).toBeNaN();
    expect(norm(f([N, 0]))).toBeNaN();
    expect(norm(f([I, 1]))).toBe(I);
    expect(norm(f([N, I]))).toBeNaN();
    expect(norm(f([N, 1]), 1)).toBeNaN();
    expect(norm(f([I, 1]), 1)).toBe(I);
    expect(norm(f([N, 1]), I)).toBeNaN();
    expect(norm(f([I, 1]), I)).toBe(I);
    expect(norm(f([I, 1]), 0)).toBe(2);
    expect(norm(f([I, 1]), 3)).toBe(I);
    expect(norm(f([N, 0]), 3)).toBeNaN();
  });

  it("matrix norms that need no SVD follow IEEE", () => {
    expect(
      norm(
        f([
          [N, 1],
          [1, 1],
        ])
      )
    ).toBeNaN();
    expect(
      norm(
        f([
          [I, 1],
          [1, 1],
        ])
      )
    ).toBe(I);
    expect(
      norm(
        f([
          [N, 1],
          [1, 1],
        ]),
        1
      )
    ).toBeNaN();
    expect(
      norm(
        f([
          [I, 1],
          [1, 1],
        ]),
        I
      )
    ).toBe(I);
    expect(
      norm(
        f([
          [N, 0],
          [0, 0],
        ]),
        "fro"
      )
    ).toBeNaN();
  });

  it("norms that need the SVD still throw", () => {
    expect(() =>
      norm(
        f([
          [N, 1],
          [1, 1],
        ]),
        2
      )
    ).toThrow(DataValidationError);
    expect(() =>
      norm(
        f([
          [I, 1],
          [1, 1],
        ]),
        "nuc"
      )
    ).toThrow(DataValidationError);
  });

  it("applies along an axis", () => {
    expect(
      norm(
        f([
          [N, 1],
          [3, 4],
        ]),
        2,
        1
      )
    ).toBeDefined();
    const out = norm(
      f([
        [N, 1],
        [3, 4],
      ]),
      2,
      1
    );
    expect((out as { toArray(): unknown }).toArray()).toEqual([Number.NaN, 5]);
  });
});

describe("cond with NaN and Infinity", () => {
  it("is NaN for NaN input and Infinity for Infinity input", () => {
    for (const p of [undefined, 1, I, "fro"] as const) {
      expect(
        cond(
          f([
            [N, 1],
            [1, 1],
          ]),
          p
        )
      ).toBeNaN();
      expect(
        cond(
          f([
            [I, 1],
            [1, 1],
          ]),
          p
        )
      ).toBe(I);
    }
  });
});

describe("det with NaN and Infinity", () => {
  it("returns the IEEE result of the LU factorization", () => {
    expect(
      det(
        f([
          [N, 1],
          [1, 1],
        ])
      )
    ).toBeNaN();
    expect(
      det(
        f([
          [1, 0],
          [0, N],
        ])
      )
    ).toBeNaN();
    expect(
      det(
        f([
          [I, 1],
          [1, 1],
        ])
      )
    ).toBe(I);
    expect(
      det(
        f([
          [I, I],
          [1, 1],
        ])
      )
    ).toBeNaN();
    expect(
      det(
        f([
          [1, I],
          [I, 1],
        ])
      )
    ).toBe(-I);
    expect(
      det(
        f([
          [-I, 1],
          [1, 1],
        ])
      )
    ).toBe(-I);
    expect(
      det(
        f([
          [0, I],
          [1, 0],
        ])
      )
    ).toBe(-I);
    expect(
      det(
        f([
          [1, 2],
          [3, I],
        ])
      )
    ).toBe(I);
    expect(det(f([[I]]))).toBe(I);
  });

  it("is unchanged for finite input", () => {
    expect(
      det(
        f([
          [1, 2],
          [3, 4],
        ])
      )
    ).toBeCloseTo(-2, 12);
  });
});

describe("decompositions keep rejecting non-finite input", () => {
  it("matrixRank, svd, eig and inv throw", () => {
    const bad = f([
      [N, 1],
      [1, 1],
    ]);
    expect(() => matrixRank(bad)).toThrow(DataValidationError);
    expect(() => svd(bad)).toThrow(DataValidationError);
    expect(() => eig(bad)).toThrow(DataValidationError);
    expect(() => inv(bad)).toThrow(DataValidationError);
  });
});
