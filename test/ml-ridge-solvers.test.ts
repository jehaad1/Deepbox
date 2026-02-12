import { describe, expect, it } from "vitest";
import { Ridge } from "../src/ml/linear/Ridge";
import { tensor } from "../src/ndarray";

describe("Ridge Regression Solvers", () => {
  // Simple dataset
  const X = tensor([
    [0, 0],
    [0, 0],
    [1, 1],
  ]);
  const y = tensor([0, 0.1, 1]);

  it("works with 'auto' solver", () => {
    const ridge = new Ridge({ solver: "auto", alpha: 0.5 });
    ridge.fit(X, y);
    expect(ridge.coef.shape).toEqual([2]);
    expect(typeof ridge.intercept).toBe("number");
    const score = ridge.score(X, y);
    expect(score).toBeGreaterThan(0.0);
  });

  it("works with 'cholesky' solver", () => {
    const ridge = new Ridge({ solver: "cholesky", alpha: 0.5 });
    ridge.fit(X, y);
    expect(ridge.coef.shape).toEqual([2]);
    expect(typeof ridge.intercept).toBe("number");
    const score = ridge.score(X, y);
    expect(score).toBeGreaterThan(0.0);
  });

  it("works with 'svd' solver", () => {
    const ridge = new Ridge({ solver: "svd", alpha: 0.5 });
    ridge.fit(X, y);
    expect(ridge.coef.shape).toEqual([2]);
    expect(typeof ridge.intercept).toBe("number");
    const score = ridge.score(X, y);
    expect(score).toBeGreaterThan(0.0);
  });

  it("works with 'lsqr' solver", () => {
    const ridge = new Ridge({ solver: "lsqr", alpha: 0.5, maxIter: 5000, tol: 1e-6 });
    ridge.fit(X, y);
    expect(ridge.coef.shape).toEqual([2]);
    expect(typeof ridge.intercept).toBe("number");
    const score = ridge.score(X, y);
    expect(score).toBeGreaterThan(0.0);
  });

  it("works with 'sag' solver", () => {
    const ridge = new Ridge({ solver: "sag", alpha: 0.5, maxIter: 2000, tol: 1e-6 });
    ridge.fit(X, y);
    expect(ridge.coef.shape).toEqual([2]);
    expect(typeof ridge.intercept).toBe("number");
    const score = ridge.score(X, y);
    expect(score).toBeGreaterThan(0.0);
  });

  it("produces consistent results across solvers", () => {
    // With regularization, different solvers should give very similar results
    // for this well-behaved problem.
    const ridgeChol = new Ridge({ solver: "cholesky", alpha: 0.1 });
    ridgeChol.fit(X, y);

    const ridgeSvd = new Ridge({ solver: "svd", alpha: 0.1 });
    ridgeSvd.fit(X, y);

    const coefChol = ridgeChol.coef;
    const coefSvd = ridgeSvd.coef;

    for (let i = 0; i < coefChol.size; i++) {
      expect(coefChol.data[i]).toBeCloseTo(Number(coefSvd.data[i]), 5);
    }
    expect(ridgeChol.intercept).toBeCloseTo(ridgeSvd.intercept, 5);
  });
});
