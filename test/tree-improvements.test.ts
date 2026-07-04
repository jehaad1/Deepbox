import { describe, expect, it } from "vitest";
import { DecisionTreeClassifier, DecisionTreeRegressor, export_text } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("DecisionTreeClassifier criterion", () => {
  const X = tensor([
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 5],
    [5, 4],
    [6, 6],
    [7, 2],
    [8, 3],
  ]);
  const y = tensor([0, 0, 0, 0, 1, 1, 1, 1]);

  it("defaults to gini criterion", () => {
    const clf = new DecisionTreeClassifier({ maxDepth: 3 });
    expect(clf.getParams().criterion).toBe("gini");
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0.8);
  });

  it("supports entropy criterion", () => {
    const clf = new DecisionTreeClassifier({ maxDepth: 3, criterion: "entropy" });
    expect(clf.getParams().criterion).toBe("entropy");
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0.8);
  });

  it("supports log_loss criterion", () => {
    const clf = new DecisionTreeClassifier({ maxDepth: 3, criterion: "log_loss" });
    expect(clf.getParams().criterion).toBe("log_loss");
    clf.fit(X, y);
    const score = clf.score(X, y);
    expect(score).toBeGreaterThanOrEqual(0.8);
  });

  it("all criteria produce valid predictions", () => {
    for (const criterion of ["gini", "entropy", "log_loss"] as const) {
      const clf = new DecisionTreeClassifier({ maxDepth: 5, criterion });
      clf.fit(X, y);
      const pred = clf.predict(X);
      expect(pred.shape).toEqual([8]);
      const proba = clf.predictProba(X);
      expect(proba.shape[0]).toBe(8);
      expect(proba.shape[1]).toBe(2);
    }
  });
});

describe("export_text", () => {
  it("exports classifier tree as text", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const clf = new DecisionTreeClassifier({ maxDepth: 3 });
    clf.fit(X, y);
    const text = export_text(clf);
    expect(text).toContain("feature_");
    expect(text).toContain("<=");
    expect(text).toContain("class:");
  });

  it("exports regressor tree as text", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([1.0, 2.0, 3.0, 4.0]);
    const reg = new DecisionTreeRegressor({ maxDepth: 3 });
    reg.fit(X, y);
    const text = export_text(reg);
    expect(text).toContain("feature_");
    expect(text).toContain("<=");
    expect(text).toContain("value:");
  });

  it("uses custom feature names", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const clf = new DecisionTreeClassifier({ maxDepth: 3 });
    clf.fit(X, y);
    const text = export_text(clf, { featureNames: ["height", "weight"] });
    const hasCustomName = text.includes("height") || text.includes("weight");
    expect(hasCustomName).toBe(true);
  });

  it("throws on unfitted tree", () => {
    const clf = new DecisionTreeClassifier();
    expect(() => export_text(clf)).toThrow();
  });

  it("respects decimals option", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const clf = new DecisionTreeClassifier({ maxDepth: 3 });
    clf.fit(X, y);
    const text = export_text(clf, { decimals: 4 });
    // Should have 4 decimal places in threshold
    expect(text).toMatch(/\d+\.\d{4}/);
  });
});
