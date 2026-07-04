import { describe, expect, it } from "vitest";
import { DBSCAN } from "../src/ml/clustering/DBSCAN";
import { LinearSVC } from "../src/ml/svm/SVM";
import { RandomForestClassifier, RandomForestRegressor } from "../src/ml/tree/RandomForest";
import { tensor } from "../src/ndarray";
import { toNumArr } from "./_helpers";

describe("Logic Audit ML Fixes Verification", () => {
  describe("SVM", () => {
    it("should respect C parameter (Large C = Hard Margin = Higher Training Accuracy)", () => {
      // Linearly separable data
      // AND function: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1
      // Linearly separable? Yes, x1+x2 > 1.5
      const X_and = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]);
      const y_and = tensor([0, 0, 0, 1]);

      // SVM with large C should fit well
      const svmHard = new LinearSVC({ C: 100.0, maxIter: 1000 });
      svmHard.fit(X_and, y_and);
      const scoreHard = svmHard.score(X_and, y_and);
      expect(scoreHard).toBeGreaterThan(0.7); // Should be perfect or close

      // SVM with very small C should underfit (strong regularization)
      // With C=0.001, it cares very little about errors, so it minimizes ||w||^2.
      // Weights -> 0.
      // Decision -> bias.
      // If bias -> 0 (or whatever), it predicts one class.
      // Accuracy should be lower than hard margin (which gets 1.0).
      const svmSoft = new LinearSVC({ C: 0.001, maxIter: 1000 });
      svmSoft.fit(X_and, y_and);
      const scoreSoft = svmSoft.score(X_and, y_and);

      // Hard margin should generally do better or equal to soft margin on training data
      expect(scoreHard).toBeGreaterThanOrEqual(scoreSoft);
    });
  });

  describe("RandomForest", () => {
    it("should support maxFeatures parameter in Classifier", () => {
      const X = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
      ]);
      const y = tensor([0, 0, 1, 1]);

      const rf = new RandomForestClassifier({
        nEstimators: 5,
        maxFeatures: 1, // Force selection of 1 feature per split
        randomState: 42,
      });
      rf.fit(X, y);
      const preds = rf.predict(X);
      expect(preds.shape).toEqual([4]);
      expect(rf.getParams().maxFeatures).toBe(1);
    });

    it("should support maxFeatures parameter in Regressor", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const y = tensor([1.5, 3.5, 5.5]);

      const rf = new RandomForestRegressor({
        nEstimators: 5,
        maxFeatures: "sqrt",
        randomState: 42,
      });
      rf.fit(X, y);
      const preds = rf.predict(X);
      expect(preds.shape).toEqual([3]);
    });
  });

  describe("DBSCAN", () => {
    it("should cluster correctly with Set-based implementation", () => {
      const X = tensor([
        [1, 1],
        [1, 2],
        [2, 1],
        [10, 10],
        [10, 11],
        [11, 10],
      ]);
      // Should find 2 clusters: one around (1,1), one around (10,10)
      const dbscan = new DBSCAN({ eps: 2.0, minSamples: 2 });
      const labels = dbscan.fitPredict(X);

      const l = toNumArr(labels.toArray());
      // First 3 should be same cluster
      expect(l[0]).toBe(l[1]);
      expect(l[1]).toBe(l[2]);

      // Last 3 should be same cluster
      expect(l[3]).toBe(l[4]);
      expect(l[4]).toBe(l[5]);

      // Clusters should be different
      expect(l[0]).not.toBe(l[3]);

      // No noise
      expect(l).not.toContain(-1);
    });

    it("should handle noise correctly", () => {
      const X = tensor([
        [1, 1],
        [1, 2],
        [100, 100], // Outlier
      ]);
      const dbscan = new DBSCAN({ eps: 2.0, minSamples: 2 });
      const labels = dbscan.fitPredict(X);
      const l = toNumArr(labels.toArray());

      expect(l[0]).toBe(l[1]); // Cluster 0
      expect(l[2]).toBe(-1); // Noise
    });
  });
});
