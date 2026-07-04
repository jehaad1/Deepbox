import { describe, expect, it } from "vitest";
import {
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
} from "../src/metrics/classification";
import {
  ndcgScore,
  pairwiseCosine,
  pairwiseEuclidean,
  pairwiseManhattan,
  reciprocalRank,
} from "../src/metrics/pairwise";
import {
  adjustedR2Score,
  explainedVarianceScore,
  mae,
  mape,
  maxError,
  medianAbsoluteError,
  mse,
  r2Score,
  rmse,
} from "../src/metrics/regression";
import { tensor } from "../src/ndarray";

// ---- precision/recall/f1 with all average modes ----

describe("precision average modes", () => {
  const yTrue = tensor([0, 0, 1, 1, 2, 2]);
  const yPred = tensor([0, 1, 1, 2, 2, 0]);

  it("macro", () => {
    const p = precision(yTrue, yPred, "macro");
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("micro", () => {
    const p = precision(yTrue, yPred, "micro");
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("weighted", () => {
    const p = precision(yTrue, yPred, "weighted");
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("null returns array", () => {
    const p = precision(yTrue, yPred, null);
    expect(Array.isArray(p)).toBe(true);
    expect((p as number[]).length).toBe(3);
  });

  it("binary with binary data", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0, 1, 0, 0]);
    const p = precision(yt, yp, "binary");
    expect(p).toBe(1.0);
  });

  it("binary with no positive predictions returns 0", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0, 0, 0, 0]);
    const p = precision(yt, yp, "binary");
    expect(p).toBe(0);
  });

  it("empty input returns 0", () => {
    const yt = tensor([] as number[]);
    const yp = tensor([] as number[]);
    expect(precision(yt, yp, "macro")).toBe(0);
    expect(precision(yt, yp, null)).toEqual([]);
  });
});

describe("recall average modes", () => {
  const yTrue = tensor([0, 0, 1, 1, 2, 2]);
  const yPred = tensor([0, 1, 1, 2, 2, 0]);

  it("macro", () => {
    const r = recall(yTrue, yPred, "macro");
    expect(r).toBeGreaterThanOrEqual(0);
    expect(r).toBeLessThanOrEqual(1);
  });

  it("micro", () => {
    const r = recall(yTrue, yPred, "micro");
    expect(r).toBeGreaterThanOrEqual(0);
  });

  it("weighted", () => {
    const r = recall(yTrue, yPred, "weighted");
    expect(r).toBeGreaterThanOrEqual(0);
  });

  it("null returns array", () => {
    const r = recall(yTrue, yPred, null);
    expect(Array.isArray(r)).toBe(true);
    expect((r as number[]).length).toBe(3);
  });

  it("binary with no positives returns 0", () => {
    const yt = tensor([0, 0, 0, 0]);
    const yp = tensor([0, 1, 0, 1]);
    const r = recall(yt, yp, "binary");
    expect(r).toBe(0);
  });

  it("empty input", () => {
    const yt = tensor([] as number[]);
    const yp = tensor([] as number[]);
    expect(recall(yt, yp, "micro")).toBe(0);
    expect(recall(yt, yp, null)).toEqual([]);
  });
});

describe("f1Score average modes", () => {
  const yTrue = tensor([0, 0, 1, 1, 2, 2]);
  const yPred = tensor([0, 1, 1, 2, 2, 0]);

  it("macro", () => {
    const f = f1Score(yTrue, yPred, "macro");
    expect(f).toBeGreaterThanOrEqual(0);
    expect(f).toBeLessThanOrEqual(1);
  });

  it("micro", () => {
    const f = f1Score(yTrue, yPred, "micro");
    expect(f).toBeGreaterThanOrEqual(0);
  });

  it("weighted", () => {
    const f = f1Score(yTrue, yPred, "weighted");
    expect(f).toBeGreaterThanOrEqual(0);
  });

  it("null returns array", () => {
    const f = f1Score(yTrue, yPred, null);
    expect(Array.isArray(f)).toBe(true);
  });

  it("binary", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0, 1, 0, 0]);
    const f = f1Score(yt, yp, "binary");
    expect(f).toBeGreaterThan(0);
  });

  it("object average param", () => {
    const f = f1Score(yTrue, yPred, { average: "macro" });
    expect(f).toBeGreaterThanOrEqual(0);
  });

  it("auto-detect multiclass", () => {
    const f = f1Score(yTrue, yPred);
    expect(f).toBeGreaterThanOrEqual(0);
  });

  it("auto-detect binary", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0, 1, 0, 0]);
    const f = f1Score(yt, yp);
    expect(f).toBeGreaterThan(0);
  });
});

describe("fbetaScore", () => {
  const yTrue = tensor([0, 1, 1, 0, 1]);
  const yPred = tensor([0, 1, 0, 0, 1]);

  it("beta=1 equals f1", () => {
    const fb = fbetaScore(yTrue, yPred, 1.0);
    const f1 = f1Score(yTrue, yPred);
    expect(fb).toBeCloseTo(f1, 5);
  });

  it("beta=0.5 weights precision more", () => {
    const fb = fbetaScore(yTrue, yPred, 0.5);
    expect(fb).toBeGreaterThanOrEqual(0);
    expect(fb).toBeLessThanOrEqual(1);
  });

  it("beta=2 weights recall more", () => {
    const fb = fbetaScore(yTrue, yPred, 2.0);
    expect(fb).toBeGreaterThanOrEqual(0);
    expect(fb).toBeLessThanOrEqual(1);
  });

  it("multiclass macro", () => {
    const yt = tensor([0, 0, 1, 1, 2, 2]);
    const yp = tensor([0, 1, 1, 2, 2, 0]);
    const fb = fbetaScore(yt, yp, 1.0, "macro");
    expect(fb).toBeGreaterThanOrEqual(0);
  });

  it("multiclass micro", () => {
    const yt = tensor([0, 0, 1, 1, 2, 2]);
    const yp = tensor([0, 1, 1, 2, 2, 0]);
    const fb = fbetaScore(yt, yp, 1.0, "micro");
    expect(fb).toBeGreaterThanOrEqual(0);
  });

  it("multiclass null", () => {
    const yt = tensor([0, 0, 1, 1, 2, 2]);
    const yp = tensor([0, 1, 1, 2, 2, 0]);
    const fb = fbetaScore(yt, yp, 1.0, null);
    expect(Array.isArray(fb)).toBe(true);
  });
});

// ---- Other classification metrics branches ----

describe("confusionMatrix branches", () => {
  it("binary", () => {
    const cm = confusionMatrix(tensor([0, 1, 1, 0]), tensor([0, 1, 0, 0]));
    expect(cm.shape).toEqual([2, 2]);
  });

  it("multiclass", () => {
    const cm = confusionMatrix(tensor([0, 1, 2, 0, 1, 2]), tensor([0, 2, 1, 0, 0, 2]));
    expect(cm.shape).toEqual([3, 3]);
  });
});

describe("classificationReport", () => {
  it("binary report", () => {
    const report = classificationReport(tensor([0, 1, 1, 0, 1]), tensor([0, 1, 0, 0, 1]));
    expect(typeof report).toBe("string");
    expect(report.length).toBeGreaterThan(0);
  });
});

describe("balancedAccuracyScore", () => {
  it("perfect predictions", () => {
    expect(balancedAccuracyScore(tensor([0, 1, 1, 0]), tensor([0, 1, 1, 0]))).toBeCloseTo(1.0);
  });

  it("imbalanced dataset", () => {
    const ba = balancedAccuracyScore(tensor([0, 0, 0, 1]), tensor([0, 0, 0, 0]));
    expect(ba).toBeLessThan(1.0);
  });
});

describe("cohenKappaScore", () => {
  it("perfect agreement", () => {
    expect(cohenKappaScore(tensor([0, 1, 1, 0]), tensor([0, 1, 1, 0]))).toBeCloseTo(1.0);
  });

  it("partial agreement", () => {
    const k = cohenKappaScore(tensor([0, 1, 1, 0]), tensor([0, 1, 0, 0]));
    expect(k).toBeGreaterThan(0);
    expect(k).toBeLessThan(1);
  });
});

describe("matthewsCorrcoef", () => {
  it("perfect", () => {
    expect(matthewsCorrcoef(tensor([0, 1, 1, 0]), tensor([0, 1, 1, 0]))).toBeCloseTo(1.0);
  });

  it("partial", () => {
    const m = matthewsCorrcoef(tensor([0, 1, 1, 0]), tensor([0, 1, 0, 0]));
    expect(m).toBeGreaterThan(0);
  });
});

describe("hammingLoss", () => {
  it("perfect", () => {
    expect(hammingLoss(tensor([0, 1, 1, 0]), tensor([0, 1, 1, 0]))).toBe(0);
  });

  it("all wrong", () => {
    expect(hammingLoss(tensor([0, 1]), tensor([1, 0]))).toBe(1);
  });
});

describe("jaccardScore", () => {
  it("perfect", () => {
    expect(jaccardScore(tensor([0, 1, 1, 0]), tensor([0, 1, 1, 0]))).toBeCloseTo(1.0);
  });

  it("partial overlap", () => {
    const j = jaccardScore(tensor([0, 1, 1, 0]), tensor([0, 1, 0, 0]));
    expect(j).toBeGreaterThanOrEqual(0);
    expect(j).toBeLessThanOrEqual(1);
  });
});

describe("logLoss", () => {
  it("good predictions have low loss", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0.1, 0.9, 0.8, 0.2]);
    const loss = logLoss(yt, yp);
    expect(loss).toBeGreaterThan(0);
    expect(loss).toBeLessThan(1);
  });

  it("bad predictions have high loss", () => {
    const yt = tensor([0, 1, 1, 0]);
    const yp = tensor([0.9, 0.1, 0.1, 0.9]);
    const loss = logLoss(yt, yp);
    expect(loss).toBeGreaterThan(1);
  });
});

describe("rocAucScore", () => {
  it("perfect separation", () => {
    const yt = tensor([0, 0, 1, 1]);
    const yp = tensor([0.1, 0.2, 0.8, 0.9]);
    const auc = rocAucScore(yt, yp);
    expect(auc).toBe(1.0);
  });

  it("random predictions around 0.5", () => {
    const yt = tensor([0, 1, 0, 1]);
    const yp = tensor([0.5, 0.5, 0.5, 0.5]);
    const auc = rocAucScore(yt, yp);
    expect(auc).toBeGreaterThanOrEqual(0);
    expect(auc).toBeLessThanOrEqual(1);
  });
});

describe("rocCurve", () => {
  it("returns [fpr, tpr, thresholds] tuple", () => {
    const yt = tensor([0, 0, 1, 1]);
    const yp = tensor([0.1, 0.4, 0.6, 0.9]);
    const [fpr, tpr, thresholds] = rocCurve(yt, yp);
    expect(fpr.size).toBeGreaterThan(0);
    expect(tpr.size).toBe(fpr.size);
    expect(thresholds.size).toBe(fpr.size);
  });
});

describe("precisionRecallCurve", () => {
  it("returns [precision, recall, thresholds] tuple", () => {
    const yt = tensor([0, 0, 1, 1]);
    const yp = tensor([0.1, 0.4, 0.6, 0.9]);
    const [prec, rec, thresholds] = precisionRecallCurve(yt, yp);
    expect(prec.size).toBeGreaterThan(0);
    expect(rec.size).toBeGreaterThan(0);
    expect(thresholds.size).toBeGreaterThan(0);
  });
});

describe("averagePrecisionScore", () => {
  it("perfect predictions", () => {
    const yt = tensor([0, 0, 1, 1]);
    const yp = tensor([0.1, 0.2, 0.8, 0.9]);
    const ap = averagePrecisionScore(yt, yp);
    expect(ap).toBeCloseTo(1.0, 1);
  });
});

// ---- Pairwise metrics ----

describe("pairwiseCosine", () => {
  it("computes cosine similarity matrix", () => {
    const X = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const result = pairwiseCosine(X);
    expect(result.shape).toEqual([3, 3]);
  });
});

describe("pairwiseEuclidean", () => {
  it("computes distance matrix", () => {
    const X = tensor([
      [0, 0],
      [1, 0],
      [0, 1],
    ]);
    const result = pairwiseEuclidean(X);
    expect(result.shape).toEqual([3, 3]);
    // Diagonal should be 0
    expect(Number(result.data[0])).toBeCloseTo(0);
  });
});

describe("pairwiseManhattan", () => {
  it("computes distance matrix", () => {
    const X = tensor([
      [0, 0],
      [1, 0],
      [0, 1],
    ]);
    const result = pairwiseManhattan(X);
    expect(result.shape).toEqual([3, 3]);
  });
});

describe("ndcgScore", () => {
  it("perfect ranking", () => {
    const yTrue = tensor([3, 2, 1, 0]);
    const yScore = tensor([3, 2, 1, 0]);
    const ndcg = ndcgScore(yTrue, yScore);
    expect(ndcg).toBeCloseTo(1.0, 3);
  });

  it("with k parameter", () => {
    const yTrue = tensor([3, 2, 1, 0]);
    const yScore = tensor([3, 2, 1, 0]);
    const ndcg = ndcgScore(yTrue, yScore, 2);
    expect(ndcg).toBeGreaterThan(0);
  });
});

describe("reciprocalRank", () => {
  it("first result correct", () => {
    const yTrue = tensor([1, 0, 0]);
    const yScore = tensor([0.9, 0.5, 0.1]);
    const rr = reciprocalRank(yTrue, yScore);
    expect(rr).toBe(1.0);
  });

  it("second result correct", () => {
    const yTrue = tensor([0, 1, 0]);
    const yScore = tensor([0.9, 0.5, 0.1]);
    const rr = reciprocalRank(yTrue, yScore);
    expect(rr).toBeCloseTo(0.5);
  });
});

// ---- Regression metrics branches ----

describe("regression metrics branches", () => {
  const yTrue = tensor([1, 2, 3, 4, 5]);
  const yPred = tensor([1.1, 2.1, 2.9, 4.1, 5.1]);

  it("mae", () => {
    const m = mae(yTrue, yPred);
    expect(m).toBeCloseTo(0.1, 1);
  });

  it("mse", () => {
    const m = mse(yTrue, yPred);
    expect(m).toBeGreaterThan(0);
    expect(m).toBeLessThan(0.1);
  });

  it("rmse", () => {
    const r = rmse(yTrue, yPred);
    expect(r).toBeGreaterThan(0);
  });

  it("r2Score perfect", () => {
    const r2 = r2Score(tensor([1, 2, 3]), tensor([1, 2, 3]));
    expect(r2).toBeCloseTo(1.0);
  });

  it("r2Score with predictions", () => {
    const r2 = r2Score(yTrue, yPred);
    expect(r2).toBeGreaterThan(0.9);
  });

  it("adjustedR2Score", () => {
    const ar2 = adjustedR2Score(yTrue, yPred, 1);
    expect(ar2).toBeLessThanOrEqual(1);
  });

  it("mape", () => {
    const m = mape(yTrue, yPred);
    expect(m).toBeGreaterThan(0);
  });

  it("maxError", () => {
    const me = maxError(yTrue, yPred);
    expect(me).toBeCloseTo(0.1, 1);
  });

  it("medianAbsoluteError", () => {
    const med = medianAbsoluteError(yTrue, yPred);
    expect(med).toBeCloseTo(0.1, 1);
  });

  it("explainedVarianceScore", () => {
    const ev = explainedVarianceScore(yTrue, yPred);
    expect(ev).toBeGreaterThan(0.9);
  });
});
