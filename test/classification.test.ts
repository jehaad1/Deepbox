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
import { numRawData } from "./_helpers";

describe("Classification Metrics", () => {
  const yTrue = tensor([0, 1, 1, 0, 1]);
  const yPred = tensor([0, 1, 0, 0, 1]);
  const yScore = tensor([0.1, 0.9, 0.4, 0.2, 0.8]);

  it("should calculate accuracy", () => {
    const acc = accuracy(yTrue, yPred);
    expect(acc).toBeCloseTo(0.8);
  });

  it("should calculate precision", () => {
    const prec = precision(yTrue, yPred);
    expect(prec).toBeCloseTo(1);
  });

  it("should calculate recall", () => {
    const rec = recall(yTrue, yPred);
    expect(rec).toBeCloseTo(2 / 3);
  });

  it("should calculate F1 score", () => {
    const f1 = f1Score(yTrue, yPred);
    expect(f1).toBeCloseTo(0.8);
  });

  it("should calculate F-beta score", () => {
    const fb = fbetaScore(yTrue, yPred, 2);
    expect(fb).toBeCloseTo(5 / 7);
  });

  it("should create confusion matrix", () => {
    const cm = confusionMatrix(yTrue, yPred);
    expect(cm.shape[0]).toBe(2);
    expect(cm.shape[1]).toBe(2);
    const d = numRawData(cm.data);
    expect(d).toEqual([2, 0, 1, 2]);
  });

  it("should generate classification report", () => {
    const report = classificationReport(yTrue, yPred);
    expect(report).toBeDefined();
    expect(typeof report).toBe("string");
  });

  it("should calculate ROC curve", () => {
    const [fpr, tpr, thresholds] = rocCurve(yTrue, yScore);
    expect(fpr.size).toBeGreaterThan(1);
    expect(tpr.size).toBe(fpr.size);
    expect(thresholds.size).toBe(fpr.size);
    expect(Number(fpr.data[fpr.offset + 0])).toBeCloseTo(0);
    expect(Number(tpr.data[tpr.offset + 0])).toBeCloseTo(0);
    expect(Number(fpr.data[fpr.offset + (fpr.size - 1)])).toBeCloseTo(1);
    expect(Number(tpr.data[tpr.offset + (tpr.size - 1)])).toBeCloseTo(1);
  });

  it("should calculate ROC AUC score", () => {
    const auc = rocAucScore(yTrue, yScore);
    expect(auc).toBeCloseTo(1);
  });

  it("should calculate precision-recall curve", () => {
    const [prec, rec, thresh] = precisionRecallCurve(yTrue, yScore);
    expect(prec.size).toBeGreaterThan(1);
    expect(rec.size).toBe(prec.size);
    expect(thresh.size).toBe(prec.size);
    expect(Number(rec.data[rec.offset + 0])).toBeCloseTo(0);
    expect(Number(rec.data[rec.offset + (rec.size - 1)])).toBeCloseTo(1);
  });

  it("should calculate average precision score", () => {
    const ap = averagePrecisionScore(yTrue, yScore);
    expect(ap).toBeCloseTo(1);
  });

  it("should calculate log loss", () => {
    const loss = logLoss(yTrue, yScore);
    expect(loss).toBeDefined();
    expect(typeof loss).toBe("number");
  });

  it("should calculate Hamming loss", () => {
    const loss = hammingLoss(yTrue, yPred);
    expect(loss).toBeCloseTo(0.2);
  });

  it("should calculate Jaccard score", () => {
    const score = jaccardScore(yTrue, yPred);
    expect(score).toBeCloseTo(2 / 3);
  });

  it("should calculate Matthews correlation coefficient", () => {
    const mcc = matthewsCorrcoef(yTrue, yPred);
    expect(mcc).toBeCloseTo(2 / 3);
  });

  it("should calculate Cohen's kappa score", () => {
    const kappa = cohenKappaScore(yTrue, yPred);
    expect(kappa).toBeCloseTo(0.6153846153846154);
  });

  it("should calculate balanced accuracy score", () => {
    const bacc = balancedAccuracyScore(yTrue, yPred);
    expect(bacc).toBeCloseTo(5 / 6);
  });

  it("should handle empty inputs", () => {
    const empty = tensor([]);
    expect(accuracy(empty, empty)).toBe(0);
    expect(precision(empty, empty)).toBe(0);
    expect(recall(empty, empty)).toBe(0);
    expect(hammingLoss(empty, empty)).toBe(0);
    expect(jaccardScore(empty, empty)).toBe(1);
    expect(rocAucScore(empty, empty)).toBe(0.5);
  });
});
