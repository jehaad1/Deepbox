import { describe, expect, it } from "vitest";
import {
  accuracy,
  averagePrecisionScore,
  balancedAccuracyScore,
  classificationReport,
  cohenKappaScore,
  confusionMatrix,
  f1Score,
  fbetaScore,
  hammingLoss,
  jaccardScore,
  logLoss,
  matthewsCorrcoef,
  precision,
  precisionRecallCurve,
  recall,
  rocAucScore,
  rocCurve,
} from "../src/metrics";
import { tensor } from "../src/ndarray";

const toNumberArray = (data: unknown): number[] => {
  if (data instanceof BigInt64Array) {
    return Array.from(data, (v) => Number(v));
  }
  if (
    data instanceof Float64Array ||
    data instanceof Float32Array ||
    data instanceof Int32Array ||
    data instanceof Uint8Array
  ) {
    return Array.from(data);
  }
  throw new Error("Expected numeric tensor data");
};

describe("Classification Metrics - Comprehensive Tests", () => {
  describe("accuracy", () => {
    it("should calculate perfect accuracy", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 1, 0, 1]);
      expect(accuracy(yTrue, yPred)).toBe(1.0);
    });

    it("should calculate zero accuracy", () => {
      const yTrue = tensor([0, 0, 0, 0, 0]);
      const yPred = tensor([1, 1, 1, 1, 1]);
      expect(accuracy(yTrue, yPred)).toBe(0.0);
    });

    it("should calculate partial accuracy", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(accuracy(yTrue, yPred)).toBeCloseTo(0.8);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(accuracy(empty, empty)).toBe(0);
    });

    it("should handle single element", () => {
      expect(accuracy(tensor([1]), tensor([1]))).toBe(1);
      expect(accuracy(tensor([1]), tensor([0]))).toBe(0);
    });

    it("should handle multiclass", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 1, 2, 0, 1, 2]);
      expect(accuracy(yTrue, yPred)).toBe(1.0);
    });

    it("should handle multiclass with errors", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      expect(accuracy(yTrue, yPred)).toBeCloseTo(2 / 6);
    });

    it("should throw on size mismatch", () => {
      expect(() => accuracy(tensor([1, 2]), tensor([1]))).toThrow();
    });

    it("should handle large datasets", () => {
      const size = 10000;
      const yTrue = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i % 2)
      );
      const yPred = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i % 2)
      );
      expect(accuracy(yTrue, yPred)).toBe(1.0);
    });

    it("should handle all same predictions", () => {
      const yTrue = tensor([0, 1, 0, 1, 0]);
      const yPred = tensor([0, 0, 0, 0, 0]);
      expect(accuracy(yTrue, yPred)).toBeCloseTo(0.6);
    });
  });

  describe("precision", () => {
    it("should calculate binary precision correctly", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(precision(yTrue, yPred, "binary")).toBeCloseTo(1.0);
    });

    it("should handle no positive predictions", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 0, 0, 0, 0]);
      expect(precision(yTrue, yPred, "binary")).toBe(0);
    });

    it("should handle all positive predictions", () => {
      const yTrue = tensor([1, 1, 1, 1, 1]);
      const yPred = tensor([1, 1, 1, 1, 1]);
      expect(precision(yTrue, yPred, "binary")).toBe(1.0);
    });

    it("should handle false positives", () => {
      const yTrue = tensor([0, 0, 0, 0, 0]);
      const yPred = tensor([1, 1, 1, 1, 1]);
      expect(precision(yTrue, yPred, "binary")).toBe(0);
    });

    it("should calculate multiclass micro precision", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const prec = precision(yTrue, yPred, "micro");
      expect(typeof prec).toBe("number");
      expect(prec).toBeCloseTo(2 / 6);
    });

    it("should calculate multiclass macro precision", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const prec = precision(yTrue, yPred, "macro");
      expect(typeof prec).toBe("number");
    });

    it("should calculate multiclass weighted precision", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const prec = precision(yTrue, yPred, "weighted");
      expect(typeof prec).toBe("number");
    });

    it("should return array for null average", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const prec = precision(yTrue, yPred, null);
      expect(Array.isArray(prec)).toBe(true);
      if (!Array.isArray(prec)) throw new Error("unreachable");
      expect(prec.length).toBeGreaterThan(0);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(precision(empty, empty, "binary")).toBe(0);
      expect(Array.isArray(precision(empty, empty, null))).toBe(true);
    });

    it("should handle imbalanced classes", () => {
      const yTrue = tensor([0, 0, 0, 0, 1]);
      const yPred = tensor([0, 0, 0, 0, 1]);
      expect(precision(yTrue, yPred, "binary")).toBe(1.0);
    });
  });

  describe("recall", () => {
    it("should calculate binary recall correctly", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(recall(yTrue, yPred, "binary")).toBeCloseTo(2 / 3);
    });

    it("should handle no true positives", () => {
      const yTrue = tensor([0, 0, 0, 0, 0]);
      const yPred = tensor([1, 1, 1, 1, 1]);
      expect(recall(yTrue, yPred, "binary")).toBe(0);
    });

    it("should handle perfect recall", () => {
      const yTrue = tensor([1, 1, 1, 1, 1]);
      const yPred = tensor([1, 1, 1, 1, 1]);
      expect(recall(yTrue, yPred, "binary")).toBe(1.0);
    });

    it("should handle all false negatives", () => {
      const yTrue = tensor([1, 1, 1, 1, 1]);
      const yPred = tensor([0, 0, 0, 0, 0]);
      expect(recall(yTrue, yPred, "binary")).toBe(0);
    });

    it("should calculate multiclass micro recall", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const rec = recall(yTrue, yPred, "micro");
      expect(typeof rec).toBe("number");
    });

    it("should calculate multiclass macro recall", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const rec = recall(yTrue, yPred, "macro");
      expect(typeof rec).toBe("number");
    });

    it("should calculate multiclass weighted recall", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const rec = recall(yTrue, yPred, "weighted");
      expect(typeof rec).toBe("number");
    });

    it("should return array for null average", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const rec = recall(yTrue, yPred, null);
      expect(Array.isArray(rec)).toBe(true);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(recall(empty, empty, "binary")).toBe(0);
    });
  });

  describe("f1Score", () => {
    it("should calculate binary F1 correctly", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(f1Score(yTrue, yPred, "binary")).toBeCloseTo(0.8);
    });

    it("should handle zero F1", () => {
      const yTrue = tensor([0, 0, 0, 0, 0]);
      const yPred = tensor([1, 1, 1, 1, 1]);
      expect(f1Score(yTrue, yPred, "binary")).toBe(0);
    });

    it("should handle perfect F1", () => {
      const yTrue = tensor([1, 1, 1, 1, 1]);
      const yPred = tensor([1, 1, 1, 1, 1]);
      expect(f1Score(yTrue, yPred, "binary")).toBe(1.0);
    });

    it("should calculate multiclass macro F1", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const f1 = f1Score(yTrue, yPred, "macro");
      expect(typeof f1).toBe("number");
    });

    it("should return array for null average", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const f1 = f1Score(yTrue, yPred, null);
      expect(Array.isArray(f1)).toBe(true);
    });

    it("should handle edge case with zero precision and recall", () => {
      const yTrue = tensor([0, 0, 0]);
      const yPred = tensor([0, 0, 0]);
      expect(f1Score(yTrue, yPred, "binary")).toBe(0);
    });
  });

  describe("fbetaScore", () => {
    it("should calculate F-beta with beta=1 (equivalent to F1)", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      const fb1 = fbetaScore(yTrue, yPred, 1, "binary");
      const f1 = f1Score(yTrue, yPred, "binary");
      expect(fb1).toBeCloseTo(Number(f1));
    });

    it("should calculate F-beta with beta=2 (favor recall)", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      const fb2 = fbetaScore(yTrue, yPred, 2, "binary");
      expect(fb2).toBeCloseTo(5 / 7);
    });

    it("should calculate F-beta with beta=0.5 (favor precision)", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      const fb05 = fbetaScore(yTrue, yPred, 0.5, "binary");
      expect(typeof fb05).toBe("number");
    });

    it("should handle multiclass", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const fb = fbetaScore(yTrue, yPred, 1.5, "macro");
      expect(typeof fb).toBe("number");
    });

    it("should return array for null average", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const fb = fbetaScore(yTrue, yPred, 1, null);
      expect(Array.isArray(fb)).toBe(true);
    });
  });

  describe("confusionMatrix", () => {
    it("should create binary confusion matrix", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      const cm = confusionMatrix(yTrue, yPred);
      expect(cm.shape[0]).toBe(2);
      expect(cm.shape[1]).toBe(2);
      const d = toNumberArray(cm.data);
      expect(d).toEqual([2, 0, 1, 2]);
    });

    it("should create multiclass confusion matrix", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const cm = confusionMatrix(yTrue, yPred);
      expect(cm.shape[0]).toBe(3);
      expect(cm.shape[1]).toBe(3);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 1, 2, 0, 1, 2]);
      const cm = confusionMatrix(yTrue, yPred);
      const d = toNumberArray(cm.data);
      expect(d[0]).toBe(2); // True class 0, pred class 0
      expect(d[4]).toBe(2); // True class 1, pred class 1
      expect(d[8]).toBe(2); // True class 2, pred class 2
    });

    it("should handle single class", () => {
      const yTrue = tensor([0, 0, 0, 0]);
      const yPred = tensor([0, 0, 0, 0]);
      const cm = confusionMatrix(yTrue, yPred);
      expect(cm.shape[0]).toBe(1);
      expect(cm.shape[1]).toBe(1);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      const cm = confusionMatrix(empty, empty);
      expect(cm.shape[0]).toBe(0);
    });
  });

  describe("classificationReport", () => {
    it("should generate report string", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      const report = classificationReport(yTrue, yPred);
      expect(typeof report).toBe("string");
      expect(report).toContain("Precision");
      expect(report).toContain("Recall");
      expect(report).toContain("F1-Score");
      expect(report).toContain("Accuracy");
    });

    it("should format numbers correctly", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      const report = classificationReport(yTrue, yPred);
      expect(report).toMatch(/\d+\.\d{4}/);
    });
  });

  describe("rocCurve", () => {
    it("should compute ROC curve", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yScore = tensor([0.1, 0.9, 0.4, 0.2, 0.8]);
      const [fpr, tpr, thresholds] = rocCurve(yTrue, yScore);

      expect(fpr.size).toBeGreaterThan(1);
      expect(tpr.size).toBe(fpr.size);
      expect(thresholds.size).toBe(fpr.size);

      expect(Number(fpr.data[fpr.offset + 0])).toBeCloseTo(0);
      expect(Number(tpr.data[tpr.offset + 0])).toBeCloseTo(0);
    });

    it("should handle perfect classifier", () => {
      const yTrue = tensor([0, 0, 1, 1]);
      const yScore = tensor([0.1, 0.2, 0.8, 0.9]);
      const [fpr, tpr] = rocCurve(yTrue, yScore);

      const lastFpr = Number(fpr.data[fpr.offset + (fpr.size - 1)]);
      const lastTpr = Number(tpr.data[tpr.offset + (tpr.size - 1)]);
      expect(lastFpr).toBeCloseTo(1);
      expect(lastTpr).toBeCloseTo(1);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      const [fpr, tpr, thresholds] = rocCurve(empty, empty);
      expect(fpr.size).toBe(0);
      expect(tpr.size).toBe(0);
      expect(thresholds.size).toBe(0);
    });

    it("should handle all same class", () => {
      const yTrue = tensor([1, 1, 1, 1]);
      const yScore = tensor([0.1, 0.2, 0.8, 0.9]);
      const [fpr, _tpr, _thresholds] = rocCurve(yTrue, yScore);
      expect(fpr.size).toBe(0);
    });

    it("should handle tied scores", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yScore = tensor([0.5, 0.5, 0.5, 0.5, 0.5]);
      const [fpr, _tpr] = rocCurve(yTrue, yScore);
      expect(fpr.size).toBeGreaterThan(0);
    });
  });

  describe("rocAucScore", () => {
    it("should calculate AUC for perfect classifier", () => {
      const yTrue = tensor([0, 0, 1, 1]);
      const yScore = tensor([0.1, 0.2, 0.8, 0.9]);
      const auc = rocAucScore(yTrue, yScore);
      expect(auc).toBeCloseTo(1.0);
    });

    it("should calculate AUC for random classifier", () => {
      const yTrue = tensor([0, 1, 0, 1]);
      const yScore = tensor([0.5, 0.5, 0.5, 0.5]);
      const auc = rocAucScore(yTrue, yScore);
      expect(auc).toBeCloseTo(0.5, 1);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(rocAucScore(empty, empty)).toBe(0.5);
    });

    it("should calculate AUC correctly", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yScore = tensor([0.1, 0.9, 0.4, 0.2, 0.8]);
      const auc = rocAucScore(yTrue, yScore);
      expect(auc).toBeCloseTo(1.0);
    });
  });

  describe("precisionRecallCurve", () => {
    it("should compute PR curve", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yScore = tensor([0.1, 0.9, 0.4, 0.2, 0.8]);
      const [prec, rec, thresh] = precisionRecallCurve(yTrue, yScore);

      expect(prec.size).toBeGreaterThan(1);
      expect(rec.size).toBe(prec.size);
      expect(thresh.size).toBe(prec.size);

      expect(Number(rec.data[rec.offset + 0])).toBeCloseTo(0);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      const [prec, _rec, _thresh] = precisionRecallCurve(empty, empty);
      expect(prec.size).toBe(0);
    });

    it("should handle no positives", () => {
      const yTrue = tensor([0, 0, 0, 0]);
      const yScore = tensor([0.1, 0.2, 0.8, 0.9]);
      const [prec, _rec, _thresh] = precisionRecallCurve(yTrue, yScore);
      expect(prec.size).toBe(0);
    });
  });

  describe("averagePrecisionScore", () => {
    it("should calculate AP for perfect classifier", () => {
      const yTrue = tensor([0, 0, 1, 1]);
      const yScore = tensor([0.1, 0.2, 0.8, 0.9]);
      const ap = averagePrecisionScore(yTrue, yScore);
      expect(ap).toBeCloseTo(1.0);
    });

    it("should calculate AP correctly", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yScore = tensor([0.1, 0.9, 0.4, 0.2, 0.8]);
      const ap = averagePrecisionScore(yTrue, yScore);
      expect(ap).toBeCloseTo(1.0);
    });

    it("should handle tied scores deterministically", () => {
      const yTrue = tensor([1, 0]);
      const yScore = tensor([0.5, 0.5]);
      const ap = averagePrecisionScore(yTrue, yScore);
      expect(ap).toBeCloseTo(0.5);
    });

    it("should handle no positives", () => {
      const yTrue = tensor([0, 0, 0, 0]);
      const yScore = tensor([0.1, 0.2, 0.8, 0.9]);
      expect(averagePrecisionScore(yTrue, yScore)).toBe(0);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(averagePrecisionScore(empty, empty)).toBe(0);
    });
  });

  describe("logLoss", () => {
    it("should calculate log loss", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yScore = tensor([0.1, 0.9, 0.4, 0.2, 0.8]);
      const loss = logLoss(yTrue, yScore);
      expect(loss).toBeGreaterThan(0);
      expect(typeof loss).toBe("number");
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yScore = tensor([0, 1, 1, 0, 1]);
      const loss = logLoss(yTrue, yScore);
      expect(loss).toBeGreaterThan(0); // Due to epsilon clipping
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(logLoss(empty, empty)).toBe(0);
    });

    it("should clip extreme values", () => {
      const yTrue = tensor([1, 0]);
      const yScore = tensor([1.0, 0.0]);
      const loss = logLoss(yTrue, yScore);
      expect(Number.isFinite(loss)).toBe(true);
    });
  });

  describe("hammingLoss", () => {
    it("should calculate hamming loss", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(hammingLoss(yTrue, yPred)).toBeCloseTo(0.2);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 1, 0, 1]);
      expect(hammingLoss(yTrue, yPred)).toBe(0);
    });

    it("should handle all wrong predictions", () => {
      const yTrue = tensor([0, 0, 0, 0, 0]);
      const yPred = tensor([1, 1, 1, 1, 1]);
      expect(hammingLoss(yTrue, yPred)).toBe(1.0);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(hammingLoss(empty, empty)).toBe(0);
    });
  });

  describe("jaccardScore", () => {
    it("should calculate Jaccard score", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(jaccardScore(yTrue, yPred)).toBeCloseTo(2 / 3);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 1, 0, 1]);
      expect(jaccardScore(yTrue, yPred)).toBe(1.0);
    });

    it("should handle no overlap", () => {
      const yTrue = tensor([0, 0, 0]);
      const yPred = tensor([1, 1, 1]);
      expect(jaccardScore(yTrue, yPred)).toBe(0);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(jaccardScore(empty, empty)).toBe(1);
    });

    it("should handle all zeros", () => {
      const yTrue = tensor([0, 0, 0]);
      const yPred = tensor([0, 0, 0]);
      expect(jaccardScore(yTrue, yPred)).toBe(1);
    });
  });

  describe("matthewsCorrcoef", () => {
    it("should calculate MCC", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(matthewsCorrcoef(yTrue, yPred)).toBeCloseTo(2 / 3);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 1, 0, 1]);
      expect(matthewsCorrcoef(yTrue, yPred)).toBe(1.0);
    });

    it("should handle completely wrong predictions", () => {
      const yTrue = tensor([0, 0, 1, 1]);
      const yPred = tensor([1, 1, 0, 0]);
      expect(matthewsCorrcoef(yTrue, yPred)).toBe(-1.0);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(matthewsCorrcoef(empty, empty)).toBe(0);
    });

    it("should handle edge case with zero denominator", () => {
      const yTrue = tensor([0, 0, 0]);
      const yPred = tensor([0, 0, 0]);
      expect(matthewsCorrcoef(yTrue, yPred)).toBe(0);
    });
  });

  describe("cohenKappaScore", () => {
    it("should calculate Cohen's kappa", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(cohenKappaScore(yTrue, yPred)).toBeCloseTo(0.6153846153846154);
    });

    it("should handle perfect agreement", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 1, 0, 1]);
      expect(cohenKappaScore(yTrue, yPred)).toBe(1.0);
    });

    it("should handle no agreement", () => {
      const yTrue = tensor([0, 0, 1, 1]);
      const yPred = tensor([1, 1, 0, 0]);
      const kappa = cohenKappaScore(yTrue, yPred);
      expect(kappa).toBeLessThan(0);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(cohenKappaScore(empty, empty)).toBe(0);
    });

    it("should handle multiclass", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const kappa = cohenKappaScore(yTrue, yPred);
      expect(typeof kappa).toBe("number");
    });
  });

  describe("balancedAccuracyScore", () => {
    it("should calculate balanced accuracy", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 0, 0, 1]);
      expect(balancedAccuracyScore(yTrue, yPred)).toBeCloseTo(5 / 6);
    });

    it("should handle perfect predictions", () => {
      const yTrue = tensor([0, 1, 1, 0, 1]);
      const yPred = tensor([0, 1, 1, 0, 1]);
      expect(balancedAccuracyScore(yTrue, yPred)).toBe(1.0);
    });

    it("should handle imbalanced classes", () => {
      const yTrue = tensor([0, 0, 0, 0, 1]);
      const yPred = tensor([0, 0, 0, 0, 0]);
      const bacc = balancedAccuracyScore(yTrue, yPred);
      expect(bacc).toBeCloseTo(0.5);
    });

    it("should handle empty tensors", () => {
      const empty = tensor([]);
      expect(balancedAccuracyScore(empty, empty)).toBe(0);
    });

    it("should handle multiclass", () => {
      const yTrue = tensor([0, 1, 2, 0, 1, 2]);
      const yPred = tensor([0, 2, 1, 0, 0, 1]);
      const bacc = balancedAccuracyScore(yTrue, yPred);
      expect(typeof bacc).toBe("number");
    });
  });

  describe("Edge Cases and Error Handling", () => {
    it("should throw on size mismatch for all metrics", () => {
      const short = tensor([1, 2]);
      const long = tensor([1, 2, 3]);

      expect(() => accuracy(short, long)).toThrow();
      expect(() => precision(short, long)).toThrow();
      expect(() => recall(short, long)).toThrow();
      expect(() => f1Score(short, long)).toThrow();
      expect(() => confusionMatrix(short, long)).toThrow();
      expect(() => rocCurve(short, long)).toThrow();
      expect(() => hammingLoss(short, long)).toThrow();
      expect(() => jaccardScore(short, long)).toThrow();
      expect(() => matthewsCorrcoef(short, long)).toThrow();
      expect(() => cohenKappaScore(short, long)).toThrow();
      expect(() => balancedAccuracyScore(short, long)).toThrow();
    });

    it("should handle string labels with auto-detected weighted average", () => {
      const yTrue = tensor(["cat", "dog", "cat"]);
      const yPred = tensor(["cat", "cat", "dog"]);
      expect(typeof precision(yTrue, yPred)).toBe("number");
      expect(typeof recall(yTrue, yPred)).toBe("number");
      expect(typeof f1Score(yTrue, yPred)).toBe("number");
    });

    it("should reject string labels for explicit binary average", () => {
      const yTrue = tensor(["cat", "dog", "cat"]);
      const yPred = tensor(["cat", "cat", "dog"]);
      expect(() => precision(yTrue, yPred, "binary")).toThrow(/string/i);
      expect(() => recall(yTrue, yPred, "binary")).toThrow(/string/i);
      expect(() => f1Score(yTrue, yPred, "binary")).toThrow(/string/i);
    });

    it("should handle very large datasets efficiently", () => {
      const size = 100000;
      const yTrue = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i % 2)
      );
      const yPred = tensor(
        Array(size)
          .fill(0)
          .map((_, i) => i % 2)
      );

      const start = Date.now();
      const acc = accuracy(yTrue, yPred);
      const duration = Date.now() - start;

      expect(acc).toBe(1.0);
      expect(duration).toBeLessThan(1000); // Should complete in under 1 second
    });

    it("should handle floating point scores correctly", () => {
      const yTrue = tensor([0, 1, 1, 0]);
      const yScore = tensor([0.123456789, 0.987654321, 0.456789123, 0.234567891]);
      const auc = rocAucScore(yTrue, yScore);
      expect(Number.isFinite(auc)).toBe(true);
      expect(auc).toBeGreaterThanOrEqual(0);
      expect(auc).toBeLessThanOrEqual(1);
    });
  });

  describe("P0 Edge Cases and Error Handling", () => {
    describe("precision micro-average correctness", () => {
      it("should calculate micro-precision correctly for multiclass", () => {
        // Test case where micro != accuracy
        const yTrue = tensor([0, 0, 1, 1, 2, 2]);
        const yPred = tensor([0, 1, 1, 2, 2, 0]);

        const microPrec = precision(yTrue, yPred, "micro");

        // Micro-precision: sum all TP and FP across classes
        // Class 0: pred=[0,1,0] -> TP=1, FP=1
        // Class 1: pred=[1,2] -> TP=1, FP=1
        // Class 2: pred=[2,0] -> TP=1, FP=1
        // Total: TP=3, FP=3 -> precision = 3/6 = 0.5
        expect(microPrec).toBeCloseTo(0.5);
      });
    });

    describe("rocCurve validation", () => {
      it("should throw DTypeError for non-binary yTrue", () => {
        const yTrue = tensor([0, 1, 2, 1]); // Contains 2, not binary
        const yScore = tensor([0.1, 0.4, 0.35, 0.8]);

        expect(() => rocCurve(yTrue, yScore)).toThrow("binary values");
      });

      it("should handle all same class gracefully", () => {
        const yTrue = tensor([1, 1, 1, 1]); // All positive
        const yScore = tensor([0.1, 0.4, 0.35, 0.8]);

        const [fpr, tpr, thresholds] = rocCurve(yTrue, yScore);
        expect(fpr.size).toBe(0);
        expect(tpr.size).toBe(0);
        expect(thresholds.size).toBe(0);
      });
    });

    describe("logLoss validation", () => {
      it("should throw DTypeError for predictions > 1", () => {
        const yTrue = tensor([0, 1, 1, 0]);
        const yPred = tensor([0.1, 1.5, 0.9, 0.2]); // 1.5 > 1

        expect(() => logLoss(yTrue, yPred)).toThrow("range [0, 1]");
      });

      it("should throw DTypeError for predictions < 0", () => {
        const yTrue = tensor([0, 1, 1, 0]);
        const yPred = tensor([0.1, -0.1, 0.9, 0.2]); // -0.1 < 0

        expect(() => logLoss(yTrue, yPred)).toThrow("range [0, 1]");
      });

      it("should handle boundary values 0 and 1", () => {
        const yTrue = tensor([0, 1, 1, 0]);
        const yPred = tensor([0.0, 1.0, 1.0, 0.0]);

        const loss = logLoss(yTrue, yPred);
        expect(Number.isFinite(loss)).toBe(true);
        expect(loss).toBeGreaterThanOrEqual(0);
      });
    });

    describe("error type validation", () => {
      it("should throw ShapeError for size mismatch", () => {
        const yTrue = tensor([0, 1, 1]);
        const yPred = tensor([0, 1]);

        expect(() => accuracy(yTrue, yPred)).toThrow("size");
      });

      it("should throw InvalidParameterError for invalid average", () => {
        const yTrue = tensor([0, 1, 1, 0]);
        const yPred = tensor([0, 1, 0, 0]);

        // @ts-expect-error Testing invalid parameter
        expect(() => precision(yTrue, yPred, "invalid")).toThrow("Invalid average parameter");
      });
    });
  });
});
