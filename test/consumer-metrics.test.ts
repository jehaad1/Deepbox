import { describe, expect, it } from "vitest";
import {
  accuracy,
  adjustedMutualInfoScore,
  adjustedR2Score,
  adjustedRandScore,
  averagePrecisionScore,
  balancedAccuracyScore,
  calinskiHarabaszScore,
  classificationReport,
  cohenKappaScore,
  completenessScore,
  confusionMatrix,
  daviesBouldinScore,
  explainedVarianceScore,
  f1Score,
  fbetaScore,
  fowlkesMallowsScore,
  hammingLoss,
  homogeneityScore,
  jaccardScore,
  logLoss,
  mae,
  mape,
  matthewsCorrcoef,
  maxError,
  medianAbsoluteError,
  mse,
  normalizedMutualInfoScore,
  precision,
  precisionRecallCurve,
  r2Score,
  recall,
  rmse,
  rocAucScore,
  rocCurve,
  silhouetteScore,
  vMeasureScore,
} from "../src/metrics";
import { tensor } from "../src/ndarray";

describe("consumer API: metrics", () => {
  const yTrue = tensor([0, 1, 1, 0, 1, 0, 1, 1]);
  const yPred = tensor([0, 1, 0, 0, 1, 1, 1, 1]);

  describe("classification", () => {
    it("accuracy, precision, recall, f1Score, fbetaScore", () => {
      const acc = accuracy(yTrue, yPred);
      expect(acc).toBeGreaterThan(0.5);
      expect(acc).toBeLessThanOrEqual(1);
      expect(precision(yTrue, yPred)).toBeGreaterThanOrEqual(0);
      expect(recall(yTrue, yPred)).toBeGreaterThanOrEqual(0);
      const f1 = f1Score(yTrue, yPred);
      expect(f1).toBeGreaterThanOrEqual(0);
      expect(f1).toBeLessThanOrEqual(1);
      expect(typeof fbetaScore(yTrue, yPred, 0.5)).toBe("number");
    });

    it("f1Score accepts { average } object form", () => {
      const score = f1Score(yTrue, yPred, { average: "macro" });
      expect(score).toBeGreaterThanOrEqual(0);
    });

    it("confusionMatrix, classificationReport", () => {
      const cm = confusionMatrix(yTrue, yPred);
      expect(cm.shape).toEqual([2, 2]);
      const report = classificationReport(yTrue, yPred);
      expect(typeof report).toBe("string");
      expect(report.length).toBeGreaterThan(0);
    });

    it("balancedAccuracy, cohenKappa, hammingLoss, jaccard, logLoss, mcc", () => {
      expect(typeof balancedAccuracyScore(yTrue, yPred)).toBe("number");
      expect(cohenKappaScore(yTrue, yPred)).toBeLessThanOrEqual(1);
      const hamming = hammingLoss(yTrue, yPred);
      expect(hamming).toBeGreaterThanOrEqual(0);
      expect(hamming).toBeLessThanOrEqual(1);
      expect(typeof jaccardScore(yTrue, yPred)).toBe("number");
      const yProba = tensor([0.1, 0.9, 0.4, 0.2, 0.8, 0.7, 0.9, 0.95]);
      expect(logLoss(yTrue, yProba)).toBeGreaterThanOrEqual(0);
      const mcc = matthewsCorrcoef(yTrue, yPred);
      expect(mcc).toBeGreaterThanOrEqual(-1);
      expect(mcc).toBeLessThanOrEqual(1);
    });

    it("ROC curve and AUC", () => {
      const yProba = tensor([0.1, 0.9, 0.4, 0.2, 0.8, 0.7, 0.9, 0.95]);
      const [fpr, tpr] = rocCurve(yTrue, yProba);
      expect(fpr.size).toBeGreaterThanOrEqual(2);
      expect(tpr.size).toBeGreaterThanOrEqual(2);
      const auc = rocAucScore(yTrue, yProba);
      expect(auc).toBeGreaterThanOrEqual(0);
      expect(auc).toBeLessThanOrEqual(1);
    });

    it("precision-recall curve and average precision", () => {
      const yProba = tensor([0.1, 0.9, 0.4, 0.2, 0.8, 0.7, 0.9, 0.95]);
      const [prcPrec, prcRec] = precisionRecallCurve(yTrue, yProba);
      expect(prcPrec.size).toBeGreaterThanOrEqual(2);
      expect(prcRec.size).toBeGreaterThanOrEqual(2);
      expect(typeof averagePrecisionScore(yTrue, yProba)).toBe("number");
    });
  });

  describe("regression", () => {
    const yTrueR = tensor([3, -0.5, 2, 7]);
    const yPredR = tensor([2.5, 0, 2, 8]);

    it("mse, rmse, mae, mape", () => {
      expect(mse(yTrueR, yPredR)).toBeGreaterThanOrEqual(0);
      expect(rmse(yTrueR, yPredR)).toBeGreaterThanOrEqual(0);
      expect(mae(yTrueR, yPredR)).toBeGreaterThanOrEqual(0);
      expect(mape(tensor([3, 2, 7]), tensor([2.5, 2, 8]))).toBeGreaterThanOrEqual(0);
    });

    it("r2Score, adjustedR2Score, explainedVarianceScore", () => {
      expect(r2Score(yTrueR, yPredR)).toBeLessThanOrEqual(1);
      expect(typeof adjustedR2Score(yTrueR, yPredR, 1)).toBe("number");
      expect(typeof explainedVarianceScore(yTrueR, yPredR)).toBe("number");
    });

    it("maxError, medianAbsoluteError", () => {
      expect(maxError(yTrueR, yPredR)).toBeGreaterThanOrEqual(0);
      expect(medianAbsoluteError(yTrueR, yPredR)).toBeGreaterThanOrEqual(0);
    });
  });

  describe("clustering", () => {
    const X = tensor([
      [1, 0],
      [1, 1],
      [0, 1],
      [5, 5],
      [5, 6],
      [6, 5],
    ]);
    const labels = tensor([0, 0, 0, 1, 1, 1]);
    const labelsTrue = tensor([0, 0, 0, 1, 1, 1]);

    it("silhouetteScore, daviesBouldinScore, calinskiHarabaszScore", () => {
      const sil = silhouetteScore(X, labels);
      expect(sil).toBeGreaterThanOrEqual(-1);
      expect(sil).toBeLessThanOrEqual(1);
      expect(daviesBouldinScore(X, labels)).toBeGreaterThanOrEqual(0);
      expect(calinskiHarabaszScore(X, labels)).toBeGreaterThanOrEqual(0);
    });

    it("external clustering metrics", () => {
      expect(typeof adjustedRandScore(labelsTrue, labels)).toBe("number");
      expect(typeof adjustedMutualInfoScore(labelsTrue, labels)).toBe("number");
      const nmi = normalizedMutualInfoScore(labelsTrue, labels);
      expect(nmi).toBeGreaterThanOrEqual(0);
      expect(nmi).toBeLessThanOrEqual(1);
      expect(typeof homogeneityScore(labelsTrue, labels)).toBe("number");
      expect(typeof completenessScore(labelsTrue, labels)).toBe("number");
      expect(typeof vMeasureScore(labelsTrue, labels)).toBe("number");
      expect(typeof fowlkesMallowsScore(labelsTrue, labels)).toBe("number");
    });
  });
});
