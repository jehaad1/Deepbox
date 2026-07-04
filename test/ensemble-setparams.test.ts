import { describe, expect, it } from "vitest";
import { AdaBoostClassifier, AdaBoostRegressor } from "../src/ml/ensemble/AdaBoost";
import { BaggingClassifier, BaggingRegressor } from "../src/ml/ensemble/Bagging";
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from "../src/ml/ensemble/GradientBoosting";
import { StackingClassifier, StackingRegressor } from "../src/ml/ensemble/Stacking";
import { VotingClassifier, VotingRegressor } from "../src/ml/ensemble/Voting";
import { LinearRegression } from "../src/ml/linear/LinearRegression";
import { LogisticRegression } from "../src/ml/linear/LogisticRegression";
import { Ridge } from "../src/ml/linear/Ridge";

describe("AdaBoostClassifier setParams", () => {
  it("sets valid params and chains", () => {
    const m = new AdaBoostClassifier();
    expect(m.setParams({ nEstimators: 20 })).toBe(m);
    expect(m.getParams().nEstimators).toBe(20);
    m.setParams({ learningRate: 0.5 });
    expect(m.getParams().learningRate).toBe(0.5);
    m.setParams({ maxDepth: 3 });
    expect(m.getParams().maxDepth).toBe(3);
  });

  it("rejects invalid nEstimators", () => {
    const m = new AdaBoostClassifier();
    expect(() => m.setParams({ nEstimators: 0 })).toThrow(/nEstimators/);
    expect(() => m.setParams({ nEstimators: -1 })).toThrow(/nEstimators/);
    expect(() => m.setParams({ nEstimators: 1.5 })).toThrow(/nEstimators/);
    expect(() => m.setParams({ nEstimators: "10" })).toThrow(/nEstimators/);
  });

  it("rejects invalid learningRate", () => {
    const m = new AdaBoostClassifier();
    expect(() => m.setParams({ learningRate: 0 })).toThrow(/learningRate/);
    expect(() => m.setParams({ learningRate: -1 })).toThrow(/learningRate/);
    expect(() => m.setParams({ learningRate: "0.1" })).toThrow(/learningRate/);
  });

  it("rejects invalid maxDepth", () => {
    const m = new AdaBoostClassifier();
    expect(() => m.setParams({ maxDepth: 0 })).toThrow(/maxDepth/);
    expect(() => m.setParams({ maxDepth: 1.5 })).toThrow(/maxDepth/);
  });

  it("rejects unknown params", () => {
    const m = new AdaBoostClassifier();
    expect(() => m.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("AdaBoostRegressor setParams", () => {
  it("sets valid params including loss", () => {
    const m = new AdaBoostRegressor();
    m.setParams({
      nEstimators: 30,
      learningRate: 0.2,
      maxDepth: 5,
      loss: "square",
    });
    const p = m.getParams();
    expect(p.nEstimators).toBe(30);
    expect(p.learningRate).toBe(0.2);
    expect(p.maxDepth).toBe(5);
    expect(p.loss).toBe("square");
  });

  it("accepts all valid loss values", () => {
    const m = new AdaBoostRegressor();
    m.setParams({ loss: "linear" });
    expect(m.getParams().loss).toBe("linear");
    m.setParams({ loss: "exponential" });
    expect(m.getParams().loss).toBe("exponential");
  });

  it("rejects invalid loss", () => {
    const m = new AdaBoostRegressor();
    expect(() => m.setParams({ loss: "bad" })).toThrow(/loss/);
  });

  it("rejects unknown params", () => {
    const m = new AdaBoostRegressor();
    expect(() => m.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("BaggingClassifier setParams", () => {
  it("sets all valid params", () => {
    const m = new BaggingClassifier();
    m.setParams({
      nEstimators: 20,
      maxSamples: 0.8,
      maxFeatures: 0.5,
      bootstrap: false,
      maxDepth: 5,
      randomState: 42,
    });
    const p = m.getParams();
    expect(p.nEstimators).toBe(20);
    expect(p.maxSamples).toBe(0.8);
    expect(p.maxFeatures).toBe(0.5);
    expect(p.bootstrap).toBe(false);
    expect(p.maxDepth).toBe(5);
    expect(p.randomState).toBe(42);
  });

  it("rejects invalid maxSamples", () => {
    const m = new BaggingClassifier();
    expect(() => m.setParams({ maxSamples: 0 })).toThrow(/maxSamples/);
    expect(() => m.setParams({ maxSamples: 1.1 })).toThrow(/maxSamples/);
    expect(() => m.setParams({ maxSamples: "0.5" })).toThrow(/maxSamples/);
  });

  it("rejects invalid maxFeatures", () => {
    const m = new BaggingClassifier();
    expect(() => m.setParams({ maxFeatures: 0 })).toThrow(/maxFeatures/);
    expect(() => m.setParams({ maxFeatures: 1.1 })).toThrow(/maxFeatures/);
  });

  it("rejects invalid bootstrap", () => {
    const m = new BaggingClassifier();
    expect(() => m.setParams({ bootstrap: 1 })).toThrow(/bootstrap/);
  });

  it("rejects unknown params", () => {
    const m = new BaggingClassifier();
    expect(() => m.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("BaggingRegressor setParams", () => {
  it("sets and validates params", () => {
    const m = new BaggingRegressor();
    m.setParams({ nEstimators: 15, bootstrap: true, maxDepth: 8 });
    const p = m.getParams();
    expect(p.nEstimators).toBe(15);
    expect(p.bootstrap).toBe(true);
    expect(p.maxDepth).toBe(8);
  });

  it("rejects invalid nEstimators", () => {
    const m = new BaggingRegressor();
    expect(() => m.setParams({ nEstimators: 0 })).toThrow(/nEstimators/);
  });

  it("rejects invalid randomState", () => {
    const m = new BaggingRegressor();
    expect(() => m.setParams({ randomState: "42" })).toThrow(/randomState/);
  });

  it("accepts undefined randomState", () => {
    const m = new BaggingRegressor();
    m.setParams({ randomState: undefined });
    expect(m.getParams().randomState).toBeUndefined();
  });

  it("rejects unknown params", () => {
    const m = new BaggingRegressor();
    expect(() => m.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("GradientBoostingRegressor setParams", () => {
  it("sets all params", () => {
    const m = new GradientBoostingRegressor();
    m.setParams({
      nEstimators: 50,
      learningRate: 0.05,
      maxDepth: 4,
      minSamplesSplit: 5,
      warmStart: true,
      subsample: 0.8,
    });
    const p = m.getParams();
    expect(p.nEstimators).toBe(50);
    expect(p.learningRate).toBe(0.05);
    expect(p.maxDepth).toBe(4);
    expect(p.minSamplesSplit).toBe(5);
    expect(p.warmStart).toBe(true);
    expect(p.subsample).toBe(0.8);
  });

  it("handles maxFeatures variants", () => {
    const m = new GradientBoostingRegressor();
    m.setParams({ maxFeatures: "sqrt" });
    expect(m.getParams().maxFeatures).toBe("sqrt");
    m.setParams({ maxFeatures: "log2" });
    expect(m.getParams().maxFeatures).toBe("log2");
    m.setParams({ maxFeatures: 5 });
    expect(m.getParams().maxFeatures).toBe(5);
    m.setParams({ maxFeatures: undefined });
    expect(m.getParams().maxFeatures).toBeUndefined();
  });

  it("rejects invalid maxFeatures", () => {
    const m = new GradientBoostingRegressor();
    expect(() => m.setParams({ maxFeatures: "bad" })).toThrow(/maxFeatures/);
    expect(() => m.setParams({ maxFeatures: 0 })).toThrow(/maxFeatures/);
  });

  it("handles validationFraction", () => {
    const m = new GradientBoostingRegressor();
    m.setParams({ validationFraction: 0.2 });
    expect(m.getParams().validationFraction).toBe(0.2);
  });

  it("rejects invalid validationFraction", () => {
    const m = new GradientBoostingRegressor();
    expect(() => m.setParams({ validationFraction: 0 })).toThrow(/validationFraction/);
    expect(() => m.setParams({ validationFraction: 1 })).toThrow(/validationFraction/);
  });

  it("handles nIterNoChange", () => {
    const m = new GradientBoostingRegressor();
    m.setParams({ nIterNoChange: 10 });
    expect(m.getParams().nIterNoChange).toBe(10);
    m.setParams({ nIterNoChange: undefined });
    expect(m.getParams().nIterNoChange).toBeUndefined();
  });

  it("rejects invalid nIterNoChange", () => {
    const m = new GradientBoostingRegressor();
    expect(() => m.setParams({ nIterNoChange: 0 })).toThrow(/nIterNoChange/);
    expect(() => m.setParams({ nIterNoChange: 1.5 })).toThrow(/nIterNoChange/);
  });

  it("handles loss", () => {
    const m = new GradientBoostingRegressor();
    m.setParams({ loss: "huber" });
    expect(m.getParams().loss).toBe("huber");
    m.setParams({ loss: "quantile" });
    expect(m.getParams().loss).toBe("quantile");
  });

  it("rejects invalid loss", () => {
    const m = new GradientBoostingRegressor();
    expect(() => m.setParams({ loss: "bad" })).toThrow(/loss/);
  });

  it("handles alpha", () => {
    const m = new GradientBoostingRegressor();
    m.setParams({ alpha: 0.5 });
    expect(m.getParams().alpha).toBe(0.5);
  });

  it("rejects invalid alpha", () => {
    const m = new GradientBoostingRegressor();
    expect(() => m.setParams({ alpha: 0 })).toThrow(/alpha/);
    expect(() => m.setParams({ alpha: 1 })).toThrow(/alpha/);
  });

  it("rejects invalid subsample", () => {
    const m = new GradientBoostingRegressor();
    expect(() => m.setParams({ subsample: 0 })).toThrow(/subsample/);
    expect(() => m.setParams({ subsample: 1.1 })).toThrow(/subsample/);
  });

  it("rejects invalid warmStart", () => {
    const m = new GradientBoostingRegressor();
    expect(() => m.setParams({ warmStart: 1 })).toThrow(/warmStart/);
  });

  it("rejects unknown params", () => {
    const m = new GradientBoostingRegressor();
    expect(() => m.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("GradientBoostingClassifier setParams", () => {
  it("sets valid params", () => {
    const m = new GradientBoostingClassifier();
    m.setParams({ nEstimators: 25, learningRate: 0.1, maxDepth: 3 });
    const p = m.getParams();
    expect(p.nEstimators).toBe(25);
    expect(p.learningRate).toBe(0.1);
    expect(p.maxDepth).toBe(3);
  });

  it("rejects unknown params", () => {
    const m = new GradientBoostingClassifier();
    expect(() => m.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("StackingClassifier setParams", () => {
  it("sets passthrough", () => {
    const m = new StackingClassifier({
      estimators: [new LogisticRegression()],
      finalEstimator: new LogisticRegression(),
    });
    m.setParams({ passthrough: true });
    expect(m.getParams().passthrough).toBe(true);
    m.setParams({ passthrough: false });
    expect(m.getParams().passthrough).toBe(false);
  });

  it("rejects invalid passthrough", () => {
    const m = new StackingClassifier({
      estimators: [new LogisticRegression()],
      finalEstimator: new LogisticRegression(),
    });
    expect(() => m.setParams({ passthrough: 1 })).toThrow(/passthrough/);
  });

  it("rejects unknown params", () => {
    const m = new StackingClassifier({
      estimators: [new LogisticRegression()],
      finalEstimator: new LogisticRegression(),
    });
    expect(() => m.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("StackingRegressor setParams", () => {
  it("sets passthrough", () => {
    const m = new StackingRegressor({
      estimators: [new LinearRegression()],
      finalEstimator: new LinearRegression(),
    });
    m.setParams({ passthrough: true });
    expect(m.getParams().passthrough).toBe(true);
  });

  it("rejects invalid passthrough", () => {
    const m = new StackingRegressor({
      estimators: [new LinearRegression()],
      finalEstimator: new LinearRegression(),
    });
    expect(() => m.setParams({ passthrough: "true" })).toThrow(/passthrough/);
  });

  it("rejects unknown params", () => {
    const m = new StackingRegressor({
      estimators: [new LinearRegression()],
      finalEstimator: new LinearRegression(),
    });
    expect(() => m.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("VotingClassifier setParams", () => {
  it("sets voting and weights", () => {
    const m = new VotingClassifier({
      estimators: [new LogisticRegression(), new LogisticRegression()],
    });
    m.setParams({ voting: "soft" });
    expect(m.getParams().voting).toBe("soft");
    m.setParams({ voting: "hard" });
    expect(m.getParams().voting).toBe("hard");
    m.setParams({ weights: [0.6, 0.4] });
    expect(m.getParams().weights).toEqual([0.6, 0.4]);
  });

  it("rejects invalid voting", () => {
    const m = new VotingClassifier({
      estimators: [new LogisticRegression()],
    });
    expect(() => m.setParams({ voting: "bad" })).toThrow(/voting/);
  });

  it("rejects invalid weights", () => {
    const m = new VotingClassifier({
      estimators: [new LogisticRegression()],
    });
    expect(() => m.setParams({ weights: "bad" })).toThrow(/weights/);
    expect(() => m.setParams({ weights: [1, "two"] })).toThrow(/weights/);
  });

  it("rejects unknown params", () => {
    const m = new VotingClassifier({
      estimators: [new LogisticRegression()],
    });
    expect(() => m.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });
});

describe("VotingRegressor setParams", () => {
  it("sets weights", () => {
    const m = new VotingRegressor({
      estimators: [new LinearRegression(), new Ridge()],
    });
    m.setParams({ weights: [0.7, 0.3] });
    expect(m.getParams().weights).toEqual([0.7, 0.3]);
  });

  it("rejects invalid weights", () => {
    const m = new VotingRegressor({
      estimators: [new LinearRegression()],
    });
    expect(() => m.setParams({ weights: 123 })).toThrow(/weights/);
  });

  it("rejects unknown params", () => {
    const m = new VotingRegressor({
      estimators: [new LinearRegression()],
    });
    expect(() => m.setParams({ foo: 1 })).toThrow(/Unknown parameter/);
  });
});
