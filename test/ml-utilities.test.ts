import { describe, expect, it } from "vitest";
import {
  check_array,
  check_is_fitted,
  check_X_y,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
} from "../src/core";
import { ColumnTransformer, cross_validate, LinearRegression } from "../src/ml";
import { parameter, randn, tensor } from "../src/ndarray";
import { ParameterDict, ParameterList } from "../src/nn";
import { ShuffleSplit, StandardScaler, VarianceThreshold } from "../src/preprocess";
import {
  meanConfidenceInterval,
  meanConfidenceIntervalZ,
  meanDiffConfidenceInterval,
  proportionConfidenceInterval,
} from "../src/stats";

// ============================================================
// ParameterList / ParameterDict
// ============================================================
describe("ParameterList", () => {
  it("should store and retrieve parameters", () => {
    const p1 = parameter(randn([3, 4]));
    const p2 = parameter(randn([5]));
    const list = new ParameterList([p1, p2]);

    expect(list.length).toBe(2);
    expect(list.get(0)).toBe(p1);
    expect(list.get(1)).toBe(p2);
  });

  it("should support negative indexing", () => {
    const p1 = parameter(randn([2]));
    const p2 = parameter(randn([3]));
    const list = new ParameterList([p1, p2]);

    expect(list.get(-1)).toBe(p2);
    expect(list.get(-2)).toBe(p1);
  });

  it("should append parameters", () => {
    const list = new ParameterList();
    expect(list.length).toBe(0);

    const p = parameter(randn([4]));
    list.append(p);
    expect(list.length).toBe(1);
    expect(list.get(0)).toBe(p);
  });

  it("should iterate over parameters", () => {
    const p1 = parameter(randn([2]));
    const p2 = parameter(randn([3]));
    const list = new ParameterList([p1, p2]);

    const items = [...list];
    expect(items).toHaveLength(2);
    expect(items[0]).toBe(p1);
    expect(items[1]).toBe(p2);
  });

  it("should register parameters for Module.parameters()", () => {
    const p1 = parameter(randn([2]));
    const p2 = parameter(randn([3]));
    const list = new ParameterList([p1, p2]);

    const params = [...list.parameters()];
    expect(params.length).toBe(2);
  });

  it("should throw on out of range index", () => {
    const list = new ParameterList();
    expect(() => list.get(0)).toThrow(InvalidParameterError);
  });

  it("should throw on forward call", () => {
    const list = new ParameterList();
    expect(() => list.forward(tensor([1]))).toThrow(InvalidParameterError);
  });

  it("should have toString", () => {
    const p1 = parameter(randn([2, 3]));
    const list = new ParameterList([p1]);
    const str = list.toString();
    expect(str).toContain("ParameterList");
    expect(str).toContain("2, 3");
  });
});

describe("ParameterDict", () => {
  it("should store and retrieve parameters by key", () => {
    const w = parameter(randn([3, 4]));
    const b = parameter(randn([4]));
    const dict = new ParameterDict({ weight: w, bias: b });

    expect(dict.length).toBe(2);
    expect(dict.get("weight")).toBe(w);
    expect(dict.get("bias")).toBe(b);
  });

  it("should support has/set/delete", () => {
    const dict = new ParameterDict();
    const p = parameter(randn([5]));

    expect(dict.has("test")).toBe(false);
    dict.set("test", p);
    expect(dict.has("test")).toBe(true);
    expect(dict.get("test")).toBe(p);

    dict.delete("test");
    expect(dict.has("test")).toBe(false);
  });

  it("should iterate over entries", () => {
    const w = parameter(randn([2]));
    const b = parameter(randn([3]));
    const dict = new ParameterDict({ w, b });

    const entries = [...dict];
    expect(entries).toHaveLength(2);
  });

  it("should expose keys and values", () => {
    const w = parameter(randn([2]));
    const dict = new ParameterDict({ weight: w });

    expect([...dict.keys()]).toEqual(["weight"]);
    expect([...dict.values()]).toHaveLength(1);
  });

  it("should register parameters for Module.parameters()", () => {
    const w = parameter(randn([2]));
    const b = parameter(randn([3]));
    const dict = new ParameterDict({ w, b });

    const params = [...dict.parameters()];
    expect(params.length).toBe(2);
  });

  it("should throw on missing key", () => {
    const dict = new ParameterDict();
    expect(() => dict.get("missing")).toThrow(InvalidParameterError);
  });

  it("should throw on forward call", () => {
    const dict = new ParameterDict();
    expect(() => dict.forward(tensor([1]))).toThrow(InvalidParameterError);
  });

  it("should have toString", () => {
    const w = parameter(randn([3, 4]));
    const dict = new ParameterDict({ weight: w });
    const str = dict.toString();
    expect(str).toContain("ParameterDict");
    expect(str).toContain("weight");
  });
});

// ============================================================
// ColumnTransformer
// ============================================================
describe("ColumnTransformer", () => {
  it("should apply different transformers to different columns", () => {
    const X = tensor([
      [1, 100, 0.5],
      [2, 200, 0.6],
      [3, 300, 0.7],
      [4, 400, 0.8],
    ]);

    const ct = new ColumnTransformer([
      ["num", new StandardScaler(), [0, 1]],
      ["pass", "passthrough", [2]],
    ]);

    const Xt = ct.fitTransform(X);
    expect(Xt.shape[0]).toBe(4);
    expect(Xt.shape[1]).toBe(3);
  });

  it("should support remainder=passthrough", () => {
    const X = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    const ct = new ColumnTransformer([["scale", new StandardScaler(), [0]]], {
      remainder: "passthrough",
    });

    const Xt = ct.fitTransform(X);
    expect(Xt.shape[0]).toBe(2);
    expect(Xt.shape[1]).toBe(3); // 1 scaled + 2 remainder
  });

  it("should support drop transformer", () => {
    const X = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    const ct = new ColumnTransformer([
      ["keep", "passthrough", [0, 2]],
      ["drop", "drop", [1]],
    ]);

    const Xt = ct.fitTransform(X);
    expect(Xt.shape[0]).toBe(2);
    expect(Xt.shape[1]).toBe(2);
  });

  it("should throw on empty transformers", () => {
    expect(() => new ColumnTransformer([])).toThrow(InvalidParameterError);
  });

  it("should throw on duplicate names", () => {
    expect(
      () =>
        new ColumnTransformer([
          ["a", "passthrough", [0]],
          ["a", "passthrough", [1]],
        ])
    ).toThrow(InvalidParameterError);
  });

  it("should throw when not fitted", () => {
    const ct = new ColumnTransformer([["a", new StandardScaler(), [0]]]);
    expect(() => ct.transform(tensor([[1]]))).toThrow();
  });

  it("should expose transformer names", () => {
    const ct = new ColumnTransformer([["num", new StandardScaler(), [0]]]);
    expect(ct.transformerNames).toEqual(["num"]);
  });
});

// ============================================================
// cross_validate
// ============================================================
describe("cross_validate", () => {
  it("should return testScores, fitTime, scoreTime", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
    ]);
    const y = tensor([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);

    const result = cross_validate(new LinearRegression(), X, y, { cv: 3 });

    expect(result.testScores).toHaveProperty("score");
    expect(result.testScores["score"]).toHaveLength(3);
    expect(result.fitTime).toHaveLength(3);
    expect(result.scoreTime).toHaveLength(3);
  });

  it("should support multiple scoring functions", () => {
    const X = tensor([
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
    ]);
    const y = tensor([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);

    const result = cross_validate(new LinearRegression(), X, y, {
      cv: 3,
      scoring: {
        default: (est, Xv, yv) => (est as LinearRegression).score(Xv, yv),
        custom: () => 0.5,
      },
    });

    expect(result.testScores["default"]).toHaveLength(3);
    expect(result.testScores["custom"]).toHaveLength(3);
    expect(result.testScores["custom"]!.every((s: number) => s === 0.5)).toBe(true);
  });

  it("should throw on invalid cv", () => {
    const X = tensor([[1], [2]]);
    const y = tensor([1, 2]);

    expect(() => cross_validate(new LinearRegression(), X, y, { cv: 1 })).toThrow(
      InvalidParameterError
    );
  });
});

// ============================================================
// VarianceThreshold
// ============================================================
describe("VarianceThreshold", () => {
  it("should remove constant features", () => {
    const X = tensor([
      [0, 2, 0],
      [0, 3, 0],
      [0, 4, 0],
      [0, 5, 0],
    ]);

    const vt = new VarianceThreshold();
    const Xt = vt.fitTransform(X);

    expect(Xt.shape[0]).toBe(4);
    expect(Xt.shape[1]).toBe(1); // only column 1 has variance > 0
  });

  it("should remove features below threshold", () => {
    const X = tensor([
      [1, 2, 100],
      [1.1, 3, 200],
      [0.9, 4, 300],
      [1, 5, 400],
    ]);

    const vt = new VarianceThreshold({ threshold: 1.0 });
    vt.fit(X);
    const Xt = vt.transform(X);

    // Only features with variance > 1.0 should remain
    expect(Xt.shape[0]).toBe(4);
    expect(Xt.shape[1]).toBeGreaterThanOrEqual(1);
  });

  it("should return correct variances", () => {
    const X = tensor([
      [1, 5],
      [3, 5],
    ]);

    const vt = new VarianceThreshold();
    vt.fit(X);

    const variances = vt.variances;
    expect(variances).toHaveLength(2);
    expect(variances[0]).toBeGreaterThan(0); // col 0: variance of [1,3]
    expect(variances[1]).toBe(0); // col 1: constant
  });

  it("should return support mask", () => {
    const X = tensor([
      [0, 1, 0],
      [0, 2, 0],
    ]);

    const vt = new VarianceThreshold();
    vt.fit(X);

    const support = vt.getSupport();
    expect(support).toEqual([false, true, false]);
  });

  it("should throw when not fitted", () => {
    const vt = new VarianceThreshold();
    expect(() => vt.transform(tensor([[1]]))).toThrow(NotFittedError);
    expect(() => vt.variances).toThrow(NotFittedError);
    expect(() => vt.getSupport()).toThrow(NotFittedError);
  });

  it("should throw on negative threshold", () => {
    expect(() => new VarianceThreshold({ threshold: -1 })).toThrow(InvalidParameterError);
  });

  it("should throw on feature count mismatch in transform", () => {
    const vt = new VarianceThreshold();
    vt.fit(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    expect(() =>
      vt.transform(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
      )
    ).toThrow(InvalidParameterError);
  });
});

// ============================================================
// ShuffleSplit
// ============================================================
describe("ShuffleSplit", () => {
  it("should produce nSplits splits", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const ss = new ShuffleSplit({ nSplits: 3, testSize: 0.2, randomState: 42 });
    const splits = ss.split(X);

    expect(splits).toHaveLength(3);
    for (const split of splits) {
      expect(split.testIndex.length).toBe(2); // 20% of 10
      expect(split.trainIndex.length).toBe(8);
    }
  });

  it("should produce deterministic splits with randomState", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const ss1 = new ShuffleSplit({
      nSplits: 2,
      testSize: 0.3,
      randomState: 99,
    });
    const ss2 = new ShuffleSplit({
      nSplits: 2,
      testSize: 0.3,
      randomState: 99,
    });

    const splits1 = ss1.split(X);
    const splits2 = ss2.split(X);

    expect(splits1[0]!.testIndex).toEqual(splits2[0]!.testIndex);
    expect(splits1[0]!.trainIndex).toEqual(splits2[0]!.trainIndex);
  });

  it("should return correct getNSplits", () => {
    const ss = new ShuffleSplit({ nSplits: 7 });
    expect(ss.getNSplits()).toBe(7);
  });

  it("should throw on invalid nSplits", () => {
    expect(() => new ShuffleSplit({ nSplits: 0 })).toThrow(InvalidParameterError);
  });
});

// ============================================================
// check_is_fitted / check_array / check_X_y
// ============================================================
describe("check_is_fitted", () => {
  it("should pass when fitted attribute exists", () => {
    const est = { coef_: [1, 2], intercept_: 0.5 };
    expect(() => check_is_fitted(est)).not.toThrow();
  });

  it("should throw when no fitted attribute", () => {
    const est = { param: 5 };
    expect(() => check_is_fitted(est)).toThrow(NotFittedError);
  });

  it("should check specific attributes", () => {
    const est = { coef_: [1, 2] };
    expect(() => check_is_fitted(est, ["coef_"])).not.toThrow();
    expect(() => check_is_fitted(est, ["intercept_"])).toThrow(NotFittedError);
  });

  it("should support custom message", () => {
    const est = { param: 5 };
    expect(() => check_is_fitted(est, undefined, "Custom msg")).toThrow("Custom msg");
  });
});

describe("check_array", () => {
  it("should pass for valid tensor", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => check_array(t, { ensureNdim: 2 })).not.toThrow();
  });

  it("should throw on null", () => {
    expect(() => check_array(null)).toThrow(DataValidationError);
  });

  it("should throw on wrong ndim", () => {
    const t = tensor([1, 2, 3]);
    expect(() => check_array(t, { ensureNdim: 2 })).toThrow(DataValidationError);
  });

  it("should throw on empty when not allowed", () => {
    const t = tensor([]).reshape([0, 2]);
    expect(() => check_array(t)).toThrow(DataValidationError);
  });

  it("should pass on empty when allowed", () => {
    const t = tensor([]).reshape([0, 2]);
    expect(() => check_array(t, { allowEmpty: true })).not.toThrow();
  });
});

describe("check_X_y", () => {
  it("should pass for consistent X and y", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([0, 1]);
    expect(() => check_X_y(X, y)).not.toThrow();
  });

  it("should throw on inconsistent samples", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([0, 1, 2]);
    expect(() => check_X_y(X, y)).toThrow(DataValidationError);
  });

  it("should throw if X is not 2D", () => {
    const X = tensor([1, 2, 3]);
    const y = tensor([0, 1, 2]);
    expect(() => check_X_y(X, y)).toThrow(DataValidationError);
  });
});

// ============================================================
// Confidence Intervals
// ============================================================
describe("meanConfidenceInterval", () => {
  it("should compute 95% CI", () => {
    const data = [2.3, 2.5, 2.1, 2.4, 2.6, 2.2, 2.3, 2.5];
    const ci = meanConfidenceInterval(data, 0.95);

    expect(ci.confidenceLevel).toBe(0.95);
    expect(ci.lower).toBeLessThan(ci.mean);
    expect(ci.upper).toBeGreaterThan(ci.mean);
    expect(ci.marginOfError).toBeGreaterThan(0);
    expect(ci.upper - ci.lower).toBeCloseTo(2 * ci.marginOfError, 10);
  });

  it("should narrow with larger sample", () => {
    const small = [1, 2, 3, 4, 5];
    const large = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5];

    const ciSmall = meanConfidenceInterval(small);
    const ciLarge = meanConfidenceInterval(large);

    expect(ciLarge.marginOfError).toBeLessThan(ciSmall.marginOfError);
  });

  it("should throw on too few data points", () => {
    expect(() => meanConfidenceInterval([1])).toThrow(InvalidParameterError);
  });

  it("should throw on invalid confidence level", () => {
    expect(() => meanConfidenceInterval([1, 2, 3], 0)).toThrow(InvalidParameterError);
    expect(() => meanConfidenceInterval([1, 2, 3], 1)).toThrow(InvalidParameterError);
  });
});

describe("meanConfidenceIntervalZ", () => {
  it("should compute z-interval with known std", () => {
    const data = [10, 12, 11, 13, 9, 11, 10, 12];
    const ci = meanConfidenceIntervalZ(data, 2.0, 0.95);

    expect(ci.confidenceLevel).toBe(0.95);
    expect(ci.lower).toBeLessThan(ci.mean);
    expect(ci.upper).toBeGreaterThan(ci.mean);
  });

  it("should throw on non-positive std", () => {
    expect(() => meanConfidenceIntervalZ([1, 2], 0)).toThrow(InvalidParameterError);
    expect(() => meanConfidenceIntervalZ([1, 2], -1)).toThrow(InvalidParameterError);
  });
});

describe("proportionConfidenceInterval", () => {
  it("should compute CI for proportion", () => {
    const ci = proportionConfidenceInterval(45, 100, 0.95);

    expect(ci.mean).toBeCloseTo(0.45, 5);
    expect(ci.lower).toBeGreaterThanOrEqual(0);
    expect(ci.upper).toBeLessThanOrEqual(1);
    expect(ci.lower).toBeLessThan(ci.mean);
    expect(ci.upper).toBeGreaterThan(ci.mean);
  });

  it("should clamp bounds to [0, 1]", () => {
    // Very small sample should still be clamped
    const ci = proportionConfidenceInterval(0, 2, 0.99);
    expect(ci.lower).toBeGreaterThanOrEqual(0);
  });

  it("should throw on invalid inputs", () => {
    expect(() => proportionConfidenceInterval(-1, 10)).toThrow(InvalidParameterError);
    expect(() => proportionConfidenceInterval(11, 10)).toThrow(InvalidParameterError);
    expect(() => proportionConfidenceInterval(5, 0)).toThrow(InvalidParameterError);
  });
});

describe("meanDiffConfidenceInterval", () => {
  it("should compute CI for difference of means", () => {
    const data1 = [10, 12, 11, 13, 14, 12, 11];
    const data2 = [8, 9, 10, 7, 8, 9, 10];
    const ci = meanDiffConfidenceInterval(data1, data2, 0.95);

    expect(ci.mean).toBeCloseTo(
      data1.reduce((a, b) => a + b, 0) / data1.length -
        data2.reduce((a, b) => a + b, 0) / data2.length,
      5
    );
    expect(ci.lower).toBeLessThan(ci.upper);
  });

  it("should throw on too few data points", () => {
    expect(() => meanDiffConfidenceInterval([1], [1, 2])).toThrow(InvalidParameterError);
    expect(() => meanDiffConfidenceInterval([1, 2], [1])).toThrow(InvalidParameterError);
  });
});
