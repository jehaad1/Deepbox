import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError, ShapeError } from "../../src/core";
import {
  accuracy,
  f1Score,
  fbetaScore,
  hammingLoss,
  jaccardScore,
  precision,
  recall,
} from "../../src/metrics";
import { tensor } from "../../src/ndarray";

// Expected values come from scikit-learn 1.9.1 with 2-D multilabel-indicator targets:
// precision_score, recall_score, f1_score, fbeta_score (beta=0.5), jaccard_score with
// average in micro, macro, weighted, samples, None, zero_division 0 or 1 and sample_weight;
// accuracy_score (subset accuracy) and hamming_loss.

type Reference = {
  yt: number[][];
  yp: number[][];
  sw: number[];
  accuracy: number;
  accuracy_w: number;
  hamming: number;
  hamming_w: number;
  hamming_1d_w: number;
  binary_err: string;
  res: {
    fn: string;
    avg: string | null;
    zd: number;
    w: boolean;
    labels?: number[];
    val: number | number[];
  }[];
};

const REF: Reference = {
  yt: [
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
  ],
  yp: [
    [1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
  ],
  sw: [1, 2, 0.5, 3, 1, 1, 2],
  res: [
    { fn: "precision", avg: "micro", zd: 0, w: false, val: 0.6666666666666666 },
    { fn: "precision", avg: "micro", zd: 0, w: true, val: 0.52 },
    { fn: "precision", avg: "micro", zd: 1, w: false, val: 0.6666666666666666 },
    { fn: "precision", avg: "micro", zd: 1, w: true, val: 0.52 },
    { fn: "precision", avg: "macro", zd: 0, w: false, val: 0.5333333333333333 },
    { fn: "precision", avg: "macro", zd: 0, w: true, val: 0.5 },
    { fn: "precision", avg: "macro", zd: 1, w: false, val: 0.7333333333333333 },
    { fn: "precision", avg: "macro", zd: 1, w: true, val: 0.7 },
    { fn: "precision", avg: "weighted", zd: 0, w: false, val: 0.9 },
    { fn: "precision", avg: "weighted", zd: 0, w: true, val: 0.8636363636363636 },
    { fn: "precision", avg: "weighted", zd: 1, w: false, val: 0.9 },
    { fn: "precision", avg: "weighted", zd: 1, w: true, val: 0.8636363636363636 },
    { fn: "precision", avg: "samples", zd: 0, w: false, val: 0.5714285714285714 },
    { fn: "precision", avg: "samples", zd: 0, w: true, val: 0.38095238095238093 },
    { fn: "precision", avg: "samples", zd: 1, w: false, val: 0.7142857142857143 },
    { fn: "precision", avg: "samples", zd: 1, w: true, val: 0.5714285714285714 },
    { fn: "precision", avg: null, zd: 0, w: false, val: [1.0, 1.0, 0.6666666666666666, 0.0, 0.0] },
    { fn: "precision", avg: null, zd: 0, w: true, val: [1.0, 1.0, 0.5, 0.0, 0.0] },
    { fn: "precision", avg: null, zd: 1, w: false, val: [1.0, 1.0, 0.6666666666666666, 1.0, 0.0] },
    { fn: "precision", avg: null, zd: 1, w: true, val: [1.0, 1.0, 0.5, 1.0, 0.0] },
    { fn: "precision", avg: "micro", zd: 0, w: false, labels: [0, 2, 4], val: 0.625 },
    { fn: "precision", avg: "macro", zd: 0, w: false, labels: [0, 2, 4], val: 0.5555555555555555 },
    {
      fn: "precision",
      avg: null,
      zd: 0,
      w: false,
      labels: [0, 2, 4],
      val: [1.0, 0.6666666666666666, 0.0],
    },
    { fn: "recall", avg: "micro", zd: 0, w: false, val: 0.6 },
    { fn: "recall", avg: "micro", zd: 0, w: true, val: 0.5909090909090909 },
    { fn: "recall", avg: "micro", zd: 1, w: false, val: 0.6 },
    { fn: "recall", avg: "micro", zd: 1, w: true, val: 0.5909090909090909 },
    { fn: "recall", avg: "macro", zd: 0, w: false, val: 0.35 },
    { fn: "recall", avg: "macro", zd: 0, w: true, val: 0.3587301587301587 },
    { fn: "recall", avg: "macro", zd: 1, w: false, val: 0.75 },
    { fn: "recall", avg: "macro", zd: 1, w: true, val: 0.7587301587301587 },
    { fn: "recall", avg: "weighted", zd: 0, w: false, val: 0.6 },
    { fn: "recall", avg: "weighted", zd: 0, w: true, val: 0.5909090909090909 },
    { fn: "recall", avg: "weighted", zd: 1, w: false, val: 0.6 },
    { fn: "recall", avg: "weighted", zd: 1, w: true, val: 0.5909090909090909 },
    { fn: "recall", avg: "samples", zd: 0, w: false, val: 0.5 },
    { fn: "recall", avg: "samples", zd: 0, w: true, val: 0.40476190476190477 },
    { fn: "recall", avg: "samples", zd: 1, w: false, val: 0.6428571428571429 },
    { fn: "recall", avg: "samples", zd: 1, w: true, val: 0.6904761904761905 },
    {
      fn: "recall",
      avg: null,
      zd: 0,
      w: false,
      val: [0.75, 0.3333333333333333, 0.6666666666666666, 0.0, 0.0],
    },
    {
      fn: "recall",
      avg: null,
      zd: 0,
      w: true,
      val: [0.5555555555555556, 0.5714285714285714, 0.6666666666666666, 0.0, 0.0],
    },
    {
      fn: "recall",
      avg: null,
      zd: 1,
      w: false,
      val: [0.75, 0.3333333333333333, 0.6666666666666666, 1.0, 1.0],
    },
    {
      fn: "recall",
      avg: null,
      zd: 1,
      w: true,
      val: [0.5555555555555556, 0.5714285714285714, 0.6666666666666666, 1.0, 1.0],
    },
    { fn: "recall", avg: "micro", zd: 0, w: false, labels: [0, 2, 4], val: 0.7142857142857143 },
    { fn: "recall", avg: "macro", zd: 0, w: false, labels: [0, 2, 4], val: 0.47222222222222215 },
    {
      fn: "recall",
      avg: null,
      zd: 0,
      w: false,
      labels: [0, 2, 4],
      val: [0.75, 0.6666666666666666, 0.0],
    },
    { fn: "f1", avg: "micro", zd: 0, w: false, val: 0.631578947368421 },
    { fn: "f1", avg: "micro", zd: 0, w: true, val: 0.5531914893617021 },
    { fn: "f1", avg: "micro", zd: 1, w: false, val: 0.631578947368421 },
    { fn: "f1", avg: "micro", zd: 1, w: true, val: 0.5531914893617021 },
    { fn: "f1", avg: "macro", zd: 0, w: false, val: 0.40476190476190477 },
    { fn: "f1", avg: "macro", zd: 0, w: true, val: 0.4025974025974026 },
    { fn: "f1", avg: "macro", zd: 1, w: false, val: 0.6047619047619047 },
    { fn: "f1", avg: "macro", zd: 1, w: true, val: 0.6025974025974026 },
    { fn: "f1", avg: "weighted", zd: 0, w: false, val: 0.6928571428571428 },
    { fn: "f1", avg: "weighted", zd: 0, w: true, val: 0.6794569067296341 },
    { fn: "f1", avg: "weighted", zd: 1, w: false, val: 0.6928571428571428 },
    { fn: "f1", avg: "weighted", zd: 1, w: true, val: 0.6794569067296341 },
    { fn: "f1", avg: "samples", zd: 0, w: false, val: 0.49999999999999994 },
    { fn: "f1", avg: "samples", zd: 0, w: true, val: 0.36507936507936506 },
    { fn: "f1", avg: "samples", zd: 1, w: false, val: 0.49999999999999994 },
    { fn: "f1", avg: "samples", zd: 1, w: true, val: 0.36507936507936506 },
    {
      fn: "f1",
      avg: null,
      zd: 0,
      w: false,
      val: [0.8571428571428571, 0.5, 0.6666666666666666, 0.0, 0.0],
    },
    {
      fn: "f1",
      avg: null,
      zd: 0,
      w: true,
      val: [0.7142857142857143, 0.7272727272727273, 0.5714285714285714, 0.0, 0.0],
    },
    {
      fn: "f1",
      avg: null,
      zd: 1,
      w: false,
      val: [0.8571428571428571, 0.5, 0.6666666666666666, 1.0, 0.0],
    },
    {
      fn: "f1",
      avg: null,
      zd: 1,
      w: true,
      val: [0.7142857142857143, 0.7272727272727273, 0.5714285714285714, 1.0, 0.0],
    },
    { fn: "f1", avg: "micro", zd: 0, w: false, labels: [0, 2, 4], val: 0.6666666666666666 },
    { fn: "f1", avg: "macro", zd: 0, w: false, labels: [0, 2, 4], val: 0.5079365079365079 },
    {
      fn: "f1",
      avg: null,
      zd: 0,
      w: false,
      labels: [0, 2, 4],
      val: [0.8571428571428571, 0.6666666666666666, 0.0],
    },
    { fn: "fbeta", avg: "micro", zd: 0, w: false, val: 0.6521739130434783 },
    { fn: "fbeta", avg: "micro", zd: 0, w: true, val: 0.5327868852459017 },
    { fn: "fbeta", avg: "micro", zd: 1, w: false, val: 0.6521739130434783 },
    { fn: "fbeta", avg: "micro", zd: 1, w: true, val: 0.5327868852459017 },
    { fn: "fbeta", avg: "macro", zd: 0, w: false, val: 0.4636904761904762 },
    { fn: "fbeta", avg: "macro", zd: 0, w: true, val: 0.451589994476446 },
    { fn: "fbeta", avg: "macro", zd: 1, w: false, val: 0.6636904761904762 },
    { fn: "fbeta", avg: "macro", zd: 1, w: true, val: 0.6515899944764459 },
    { fn: "fbeta", avg: "weighted", zd: 0, w: false, val: 0.7892857142857143 },
    { fn: "fbeta", avg: "weighted", zd: 0, w: true, val: 0.7728850885562003 },
    { fn: "fbeta", avg: "weighted", zd: 1, w: false, val: 0.7892857142857143 },
    { fn: "fbeta", avg: "weighted", zd: 1, w: true, val: 0.7728850885562003 },
    { fn: "fbeta", avg: "samples", zd: 0, w: false, val: 0.5317460317460317 },
    { fn: "fbeta", avg: "samples", zd: 0, w: true, val: 0.36772486772486773 },
    { fn: "fbeta", avg: "samples", zd: 1, w: false, val: 0.5317460317460317 },
    { fn: "fbeta", avg: "samples", zd: 1, w: true, val: 0.36772486772486773 },
    {
      fn: "fbeta",
      avg: null,
      zd: 0,
      w: false,
      val: [0.9375, 0.7142857142857143, 0.6666666666666666, 0.0, 0.0],
    },
    {
      fn: "fbeta",
      avg: null,
      zd: 0,
      w: true,
      val: [0.8620689655172413, 0.8695652173913043, 0.5263157894736842, 0.0, 0.0],
    },
    {
      fn: "fbeta",
      avg: null,
      zd: 1,
      w: false,
      val: [0.9375, 0.7142857142857143, 0.6666666666666666, 1.0, 0.0],
    },
    {
      fn: "fbeta",
      avg: null,
      zd: 1,
      w: true,
      val: [0.8620689655172413, 0.8695652173913043, 0.5263157894736842, 1.0, 0.0],
    },
    { fn: "fbeta", avg: "micro", zd: 0, w: false, labels: [0, 2, 4], val: 0.6410256410256411 },
    { fn: "fbeta", avg: "macro", zd: 0, w: false, labels: [0, 2, 4], val: 0.5347222222222222 },
    {
      fn: "fbeta",
      avg: null,
      zd: 0,
      w: false,
      labels: [0, 2, 4],
      val: [0.9375, 0.6666666666666666, 0.0],
    },
    { fn: "jaccard", avg: "micro", zd: 0, w: false, val: 0.46153846153846156 },
    { fn: "jaccard", avg: "micro", zd: 0, w: true, val: 0.38235294117647056 },
    { fn: "jaccard", avg: "micro", zd: 1, w: false, val: 0.46153846153846156 },
    { fn: "jaccard", avg: "micro", zd: 1, w: true, val: 0.38235294117647056 },
    { fn: "jaccard", avg: "macro", zd: 0, w: false, val: 0.31666666666666665 },
    { fn: "jaccard", avg: "macro", zd: 0, w: true, val: 0.3053968253968254 },
    { fn: "jaccard", avg: "macro", zd: 1, w: false, val: 0.5166666666666666 },
    { fn: "jaccard", avg: "macro", zd: 1, w: true, val: 0.5053968253968254 },
    { fn: "jaccard", avg: "weighted", zd: 0, w: false, val: 0.55 },
    { fn: "jaccard", avg: "weighted", zd: 0, w: true, val: 0.5181818181818182 },
    { fn: "jaccard", avg: "weighted", zd: 1, w: false, val: 0.55 },
    { fn: "jaccard", avg: "weighted", zd: 1, w: true, val: 0.5181818181818182 },
    { fn: "jaccard", avg: "samples", zd: 0, w: false, val: 0.4047619047619047 },
    { fn: "jaccard", avg: "samples", zd: 0, w: true, val: 0.2936507936507936 },
    { fn: "jaccard", avg: "samples", zd: 1, w: false, val: 0.4047619047619047 },
    { fn: "jaccard", avg: "samples", zd: 1, w: true, val: 0.2936507936507936 },
    { fn: "jaccard", avg: null, zd: 0, w: false, val: [0.75, 0.3333333333333333, 0.5, 0.0, 0.0] },
    {
      fn: "jaccard",
      avg: null,
      zd: 0,
      w: true,
      val: [0.5555555555555556, 0.5714285714285714, 0.4, 0.0, 0.0],
    },
    { fn: "jaccard", avg: null, zd: 1, w: false, val: [0.75, 0.3333333333333333, 0.5, 1.0, 0.0] },
    {
      fn: "jaccard",
      avg: null,
      zd: 1,
      w: true,
      val: [0.5555555555555556, 0.5714285714285714, 0.4, 1.0, 0.0],
    },
    { fn: "jaccard", avg: "micro", zd: 0, w: false, labels: [0, 2, 4], val: 0.5 },
    { fn: "jaccard", avg: "macro", zd: 0, w: false, labels: [0, 2, 4], val: 0.4166666666666667 },
    { fn: "jaccard", avg: null, zd: 0, w: false, labels: [0, 2, 4], val: [0.75, 0.5, 0.0] },
  ],
  accuracy: 0.14285714285714285,
  accuracy_w: 0.09523809523809523,
  hamming: 0.2,
  hamming_w: 0.2,
  hamming_1d_w: 0.2,
  binary_err:
    "Target is multilabel-indicator but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted', 'samples'].",
};

const yTrue = tensor(REF.yt, { dtype: "int32" });
const yPred = tensor(REF.yp, { dtype: "int32" });

type Average = "micro" | "macro" | "weighted" | "samples" | null;
type Options = {
  average: Average;
  zeroDivision: number;
  sampleWeight?: number[];
  labels?: number[];
};

const SCORES: Record<string, (o: Options) => number | number[]> = {
  precision: (o) => precision(yTrue, yPred, o as never),
  recall: (o) => recall(yTrue, yPred, o as never),
  f1: (o) => f1Score(yTrue, yPred, o as never),
  fbeta: (o) => fbetaScore(yTrue, yPred, 0.5, o as never),
  jaccard: (o) => jaccardScore(yTrue, yPred, o as never),
};

describe("multilabel indicator targets match scikit-learn", () => {
  for (const r of REF.res) {
    const label = `${r.fn} average=${String(r.avg)} zeroDivision=${r.zd}${r.w ? " weighted" : ""}${r.labels ? ` labels=${JSON.stringify(r.labels)}` : ""}`;
    it(label, () => {
      const options: Options = {
        average: r.avg as Average,
        zeroDivision: r.zd,
        ...(r.w ? { sampleWeight: REF.sw } : {}),
        ...(r.labels ? { labels: r.labels } : {}),
      };
      const actual = (SCORES[r.fn] as (o: Options) => number | number[])(options);
      if (Array.isArray(r.val)) {
        expect(Array.isArray(actual)).toBe(true);
        expect((actual as number[]).length).toBe(r.val.length);
        for (const [i, e] of r.val.entries()) expect((actual as number[])[i]).toBeCloseTo(e, 12);
      } else {
        expect(actual as number).toBeCloseTo(r.val, 12);
      }
    });
  }

  it("accuracy is the subset accuracy", () => {
    expect(accuracy(yTrue, yPred)).toBeCloseTo(REF.accuracy, 12);
    expect(accuracy(yTrue, yPred, { sampleWeight: REF.sw })).toBeCloseTo(REF.accuracy_w, 12);
  });

  it("hammingLoss is the fraction of wrong labels", () => {
    expect(hammingLoss(yTrue, yPred)).toBeCloseTo(REF.hamming, 12);
    expect(hammingLoss(yTrue, yPred, { sampleWeight: REF.sw })).toBeCloseTo(REF.hamming_w, 12);
    expect(
      hammingLoss(tensor([0, 1, 1, 0, 1]), tensor([0, 1, 0, 0, 1]), {
        sampleWeight: [1, 2, 3, 4, 5],
      })
    ).toBeCloseTo(REF.hamming_1d_w, 12);
  });

  it("uses the weighted average by default and rejects the binary average", () => {
    expect(precision(yTrue, yPred)).toBeCloseTo(
      precision(yTrue, yPred, { average: "weighted" }) as number,
      12
    );
    // scikit-learn: Target is multilabel-indicator but average='binary'.
    expect(REF.binary_err).toContain("multilabel-indicator but average='binary'");
    expect(() => precision(yTrue, yPred, "binary")).toThrow(
      /multilabel-indicator but average='binary'/
    );
  });

  it("rejects the samples average for single-label targets", () => {
    expect(() =>
      precision(tensor([0, 1, 1]), tensor([0, 1, 0]), { average: "samples" as never })
    ).toThrow(InvalidParameterError);
  });

  it("rejects mismatched shapes, non-indicator values and out-of-range label columns", () => {
    expect(() => accuracy(yTrue, tensor(REF.yp.map((row) => row.slice(0, 4))))).toThrow(ShapeError);
    expect(() => accuracy(yTrue, tensor(REF.yp.map((row) => row[0] as number)))).toThrow(
      ShapeError
    );
    const bad = tensor(REF.yp.map((row) => row.map((v) => v * 2)));
    expect(() => precision(yTrue, bad)).toThrow(InvalidParameterError);
    expect(() => precision(yTrue, yPred, { average: null, labels: [0, 9] })).toThrow(
      InvalidParameterError
    );
    expect(() => precision(yTrue, tensor(REF.yp.map((row) => row.map(() => Number.NaN))))).toThrow(
      DataValidationError
    );
  });
});
