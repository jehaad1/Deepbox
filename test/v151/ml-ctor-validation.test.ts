import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import {
  ElasticNet,
  KernelRidge,
  Lasso,
  RANSACRegressor,
  Ridge,
  SGDClassifier,
  SGDRegressor,
} from "../../src/ml";

type Ctor = new (
  options?: Record<string, unknown>
) => { setParams(p: Record<string, unknown>): unknown };

describe("ML-12: constructors validate like setParams", () => {
  const cases: Array<[string, Ctor, Record<string, unknown>]> = [
    ["Lasso alpha -1", Lasso as unknown as Ctor, { alpha: -1 }],
    ["Lasso alpha NaN", Lasso as unknown as Ctor, { alpha: Number.NaN }],
    ["Lasso alpha Infinity", Lasso as unknown as Ctor, { alpha: Number.POSITIVE_INFINITY }],
    ["Lasso maxIter 0", Lasso as unknown as Ctor, { maxIter: 0 }],
    ["Lasso maxIter 1.5", Lasso as unknown as Ctor, { maxIter: 1.5 }],
    ["Lasso tol -1", Lasso as unknown as Ctor, { tol: -1 }],
    ["Lasso selection", Lasso as unknown as Ctor, { selection: "sideways" }],
    ["ElasticNet alpha -1", ElasticNet as unknown as Ctor, { alpha: -1 }],
    ["ElasticNet l1Ratio 2", ElasticNet as unknown as Ctor, { l1Ratio: 2 }],
    ["ElasticNet l1Ratio NaN", ElasticNet as unknown as Ctor, { l1Ratio: Number.NaN }],
    ["ElasticNet maxIter -1", ElasticNet as unknown as Ctor, { maxIter: -1 }],
    ["ElasticNet tol NaN", ElasticNet as unknown as Ctor, { tol: Number.NaN }],
    ["Ridge alpha -1", Ridge as unknown as Ctor, { alpha: -1 }],
    ["Ridge alpha NaN", Ridge as unknown as Ctor, { alpha: Number.NaN }],
    ["Ridge maxIter 0", Ridge as unknown as Ctor, { maxIter: 0 }],
    ["Ridge maxIter 1.5", Ridge as unknown as Ctor, { maxIter: 1.5 }],
    ["Ridge tol -1", Ridge as unknown as Ctor, { tol: -1 }],
    ["Ridge solver", Ridge as unknown as Ctor, { solver: "magic" }],
    ["Ridge fitIntercept", Ridge as unknown as Ctor, { fitIntercept: "yes" }],
    ["RANSAC stopNInliers -1", RANSACRegressor as unknown as Ctor, { stopNInliers: -1 }],
    ["RANSAC stopNInliers 1.5", RANSACRegressor as unknown as Ctor, { stopNInliers: 1.5 }],
    ["SGDClassifier eta0 -1", SGDClassifier as unknown as Ctor, { eta0: -1 }],
    ["SGDRegressor eta0 0", SGDRegressor as unknown as Ctor, { eta0: 0 }],
    ["SGDClassifier powerT NaN", SGDClassifier as unknown as Ctor, { powerT: Number.NaN }],
    ["KernelRidge gamma -1", KernelRidge as unknown as Ctor, { gamma: -1 }],
    ["KernelRidge degree NaN", KernelRidge as unknown as Ctor, { degree: Number.NaN }],
  ];

  for (const [name, C, bad] of cases) {
    it(`${name} is rejected by the constructor and by setParams`, () => {
      expect(() => new C(bad)).toThrow(InvalidParameterError);
      expect(() => new C().setParams(bad)).toThrow(InvalidParameterError);
    });
  }

  it("accepts the edge values that scikit-learn accepts", () => {
    expect(() => new Lasso({ alpha: 0 })).not.toThrow();
    expect(() => new ElasticNet({ l1Ratio: 0 })).not.toThrow();
    expect(() => new ElasticNet({ l1Ratio: 1 })).not.toThrow();
    expect(() => new Ridge({ alpha: 0, tol: 0 })).not.toThrow();
    expect(() => new RANSACRegressor({ stopNInliers: 0 })).not.toThrow();
    expect(() => new RANSACRegressor({ stopNInliers: Number.POSITIVE_INFINITY })).not.toThrow();
    expect(() => new KernelRidge({ gamma: 0, degree: 0 })).not.toThrow();
  });

  it("an invalid Ridge setParams leaves the estimator unchanged", () => {
    const ridge = new Ridge({ alpha: 2 });
    expect(() => ridge.setParams({ alpha: 5, tol: -1 })).toThrow(InvalidParameterError);
    expect(ridge.getParams()["alpha"]).toBe(2);
  });
});
