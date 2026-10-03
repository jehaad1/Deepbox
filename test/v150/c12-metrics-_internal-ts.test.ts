import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  ShapeError,
} from "../../src/core/errors";
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
} from "../../src/metrics";
import {
  denseFloat64,
  denseLabels,
  isRowMajorContiguous,
  tryDenseNumeric,
} from "../../src/metrics/_internal";
import { tensor } from "../../src/ndarray";
import { Tensor } from "../../src/ndarray/tensor";

const f64 = { dtype: "float64" } as const;

function toArray(t: Tensor): number[] {
  const out: number[] = [];
  const data = t.data as Float64Array;
  for (let i = 0; i < t.size; i++) out.push(data[t.offset + i] as number);
  return out;
}

function int64(values: bigint[]): Tensor {
  return tensor(BigInt64Array.from(values));
}

function strided1D(logical: number[], stride: number, offset: number): Tensor {
  const buf = new Float64Array(offset + logical.length * stride + 1).fill(-99);
  logical.forEach((v, i) => {
    buf[offset + i * stride] = v;
  });
  return Tensor.fromTypedArray({
    data: buf,
    shape: [logical.length],
    dtype: "float64",
    device: "cpu",
    offset,
    strides: [stride],
  });
}

// Multiclass fixture; references from scikit-learn 1.8:
//   yt3 = [0,1,2,0,1,2,2,1], yp3 = [0,2,1,0,0,1,2,1]
const yt3 = tensor([0, 1, 2, 0, 1, 2, 2, 1]);
const yp3 = tensor([0, 2, 1, 0, 0, 1, 2, 1]);

describe("v1.5.0 metrics/_internal", () => {
  it("denseFloat64 / tryDenseNumeric read contiguous offset views without a gather", () => {
    const base = tensor([9, 1, 2, 3, 9], f64);
    const view = Tensor.fromTypedArray({
      data: base.data as Float64Array,
      shape: [3],
      dtype: "float64",
      device: "cpu",
      offset: 1,
    });
    expect(Array.from(denseFloat64(view, "x"))).toEqual([1, 2, 3]);
    expect(Array.from(tryDenseNumeric(view) ?? [])).toEqual([1, 2, 3]);
  });

  it("denseFloat64 handles strided 1-D and 2-D views", () => {
    expect(Array.from(denseFloat64(strided1D([1, 2, 3, 4], 2, 1), "x"))).toEqual([1, 2, 3, 4]);

    const matrix = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      f64
    );
    const transposed = Tensor.fromTypedArray({
      data: matrix.data as Float64Array,
      shape: [3, 2],
      dtype: "float64",
      device: "cpu",
      strides: [1, 3],
    });
    expect(Array.from(denseFloat64(transposed, "x"))).toEqual([1, 4, 2, 5, 3, 6]);
    expect(isRowMajorContiguous([3, 2], [1, 3])).toBe(false);
    expect(isRowMajorContiguous([3, 1], [1, 1])).toBe(true);
    expect(isRowMajorContiguous([3, 1], [5, 7])).toBe(false);
  });

  it("denseFloat64 reports the offending index with the shared finite-number message", () => {
    expect(() => denseFloat64(tensor([1, Number.NaN, 3]), "scores")).toThrow(
      /scores must contain only finite numbers; found NaN at index 1/
    );
    expect(() => denseFloat64(tensor([1, Infinity]), "scores")).toThrow(DataValidationError);
    expect(Array.from(denseFloat64(tensor([1, Number.NaN]), "x", false))[1]).toBeNaN();
  });

  it("denseLabels returns number, string and bigint arrays for strided data", () => {
    expect(Array.from(denseLabels(strided1D([3, 1, 2], 3, 2), "y") as Float64Array)).toEqual([
      3, 1, 2,
    ]);
    expect(denseLabels(tensor(["a", "b", "c"]), "y")).toEqual(["a", "b", "c"]);
    expect(denseLabels(int64([5n, 7n]), "y")).toEqual([5n, 7n]);
    const strided = Tensor.fromStringArray({
      data: ["a", "x", "b", "x", "c"],
      shape: [3],
      device: "cpu",
      offset: 0,
      strides: [2],
    });
    expect(denseLabels(strided, "y")).toEqual(["a", "b", "c"]);
  });

  it("shape errors name the offending shape", () => {
    expect(() => accuracy(tensor([[1, 2]]), tensor([[1, 2]]))).toThrow(
      /1D or a column vector; got shape \[1, 2\]/
    );
  });
});

describe("v1.5.0 metrics/classification: input validation", () => {
  it("accuracy and hammingLoss reject NaN labels instead of silently counting them wrong", () => {
    // Previously the dense fast path skipped validation: NaN !== NaN gave accuracy 0.5.
    const yTrue = tensor([0, Number.NaN]);
    const yPred = tensor([0, Number.NaN]);
    expect(() => accuracy(yTrue, yPred)).toThrow(DataValidationError);
    expect(() => hammingLoss(yTrue, yPred)).toThrow(DataValidationError);
    expect(() => confusionMatrix(yTrue, yPred)).toThrow(DataValidationError);
    expect(() => accuracy(tensor([0, 1]), tensor([0, Infinity]))).toThrow(DataValidationError);
  });

  it("auto-detected average validates the inputs first", () => {
    // Previously the multiclass probe read past the shorter tensor before the size check.
    expect(() => precision(tensor([0, 1, 2]), tensor([0, 1]))).toThrow(ShapeError);
    expect(() => recall(tensor([0, 1]), tensor([0, 1, 2]))).toThrow(ShapeError);
    expect(() => f1Score(tensor([0, 1]), tensor([[0, 1]]))).toThrow(ShapeError);
  });

  it("rejects invalid average values even for empty input", () => {
    const empty = tensor([]);
    expect(() => precision(empty, empty, "nope" as never)).toThrow(/Invalid average parameter/);
    expect(precision(empty, empty, null)).toEqual([]);
    expect(precision(empty, empty, "macro")).toBe(0);
  });

  it("mismatched label kinds raise DTypeError", () => {
    expect(() => accuracy(tensor(["a"]), tensor([1]))).toThrow(DTypeError);
    expect(() => precision(tensor(["a", "b"]), tensor([1, 0]))).toThrow(DTypeError);
    expect(() => confusionMatrix(int64([1n]), tensor(["1"]))).toThrow(DTypeError);
    // int64 labels may be mixed with integer-valued numeric labels since 1.5.0.
    expect(Array.from(confusionMatrix(int64([1n]), tensor([1])).data as Float64Array)).toEqual([1]);
  });

  it("score inputs reject strings with DTypeError and ignore nothing", () => {
    expect(() => rocAucScore(tensor([0, 1]), tensor(["a", "b"]))).toThrow(DTypeError);
    expect(() => rocCurve(tensor(["a", "b"]), tensor([0.1, 0.2]))).toThrow(DTypeError);
    expect(() => rocAucScore(tensor([0, 1]), tensor([0.1, Number.NaN]))).toThrow(
      DataValidationError
    );
    expect(() => logLoss(tensor([0, 1]), tensor([0.1, 1.5]))).toThrow(/range \[0, 1\]/);
  });
});

describe("v1.5.0 metrics/classification: int64, bool and strided inputs", () => {
  it("binary metrics accept int64 and bool labels", () => {
    const yTrue = int64([0n, 1n, 1n, 0n, 1n]);
    const yPred = int64([0n, 1n, 0n, 0n, 1n]);
    expect(precision(yTrue, yPred)).toBe(1);
    expect(recall(yTrue, yPred)).toBeCloseTo(2 / 3, 12);
    expect(f1Score(yTrue, yPred)).toBeCloseTo(0.8, 12);
    expect(jaccardScore(yTrue, yPred)).toBeCloseTo(2 / 3, 12);
    expect(matthewsCorrcoef(yTrue, yPred)).toBeCloseTo(0.6666666666666666, 12);
    expect(() => precision(int64([0n, 2n]), int64([0n, 1n]), "binary")).toThrow(/binary values/);

    const bTrue = tensor([true, false, true]);
    const bPred = tensor([true, false, false]);
    expect(accuracy(bTrue, bPred)).toBeCloseTo(2 / 3, 12);
    expect(precision(bTrue, bPred)).toBe(1);
  });

  it("int64 auto-detects multiclass averages like numeric labels", () => {
    const t = int64([0n, 1n, 2n, 0n, 1n, 2n, 2n, 1n]);
    const p = int64([0n, 2n, 1n, 0n, 0n, 1n, 2n, 1n]);
    expect(precision(t, p)).toBeCloseTo(0.47916666666666663, 12);
    expect(precision(t, p, null)).toHaveLength(3);
    expect(Array.from(confusionMatrix(t, p).data as Float64Array)).toEqual([
      2, 0, 0, 1, 1, 1, 0, 2, 1,
    ]);
  });

  it("strided and column-vector inputs give the same results as dense ones", () => {
    const yTrue = [0, 1, 1, 0, 1, 0];
    const yPred = [0, 1, 0, 0, 1, 1];
    const yScore = [0.2, 0.9, 0.4, 0.1, 0.8, 0.6];
    const dense = (a: number[]) => tensor(a, f64);
    const view = (a: number[]) => strided1D(a, 3, 2);
    expect(accuracy(view(yTrue), view(yPred))).toBe(accuracy(dense(yTrue), dense(yPred)));
    expect(f1Score(view(yTrue), view(yPred))).toBe(f1Score(dense(yTrue), dense(yPred)));
    expect(rocAucScore(view(yTrue), view(yScore))).toBe(rocAucScore(dense(yTrue), dense(yScore)));
    expect(logLoss(view(yTrue), view(yScore))).toBe(logLoss(dense(yTrue), dense(yScore)));
    const column = Tensor.fromTypedArray({
      data: new Float64Array(yTrue.flatMap((v) => [v, -1])),
      shape: [6, 1],
      dtype: "float64",
      device: "cpu",
      strides: [2, 1],
    });
    expect(accuracy(column, dense(yPred))).toBe(accuracy(dense(yTrue), dense(yPred)));
  });
});

describe("v1.5.0 metrics/classification: precision, recall, F-scores", () => {
  // sklearn: precision_score(yt3, yp3, average=...) and friends
  it("matches scikit-learn for every averaging mode", () => {
    expect(precision(yt3, yp3, "micro")).toBeCloseTo(0.5, 12);
    expect(precision(yt3, yp3, "macro")).toBeCloseTo(0.5, 12);
    expect(precision(yt3, yp3, "weighted")).toBeCloseTo(0.47916666666666663, 12);
    expect(recall(yt3, yp3, "macro")).toBeCloseTo(0.5555555555555555, 12);
    expect(recall(yt3, yp3, "weighted")).toBeCloseTo(0.5, 12);
    expect(f1Score(yt3, yp3, "macro")).toBeCloseTo(0.5111111111111111, 12);
    expect(f1Score(yt3, yp3, "weighted")).toBeCloseTo(0.47500000000000003, 12);
    expect(f1Score(yt3, yp3, "micro")).toBeCloseTo(0.5, 12);
    const perClass = f1Score(yt3, yp3, null);
    [0.8, 1 / 3, 0.4].forEach((v, i) => {
      expect(perClass[i]).toBeCloseTo(v, 12);
    });
  });

  it("fbetaScore matches scikit-learn, including beta = 0 (precision) and the auto average", () => {
    const f2 = fbetaScore(yt3, yp3, 2, null);
    [0.9090909090909091, 0.3333333333333333, 0.35714285714285715].forEach((v, i) => {
      expect(f2[i]).toBeCloseTo(v, 12);
    });
    expect(fbetaScore(yt3, yp3, 2, "macro")).toBeCloseTo(0.5331890331890332, 12);
    expect(fbetaScore(yt3, yp3, 2, "weighted")).toBeCloseTo(0.4862012987012987, 12);
    expect(fbetaScore(yt3, yp3, 0, "macro")).toBeCloseTo(0.5, 12);
    expect(() => fbetaScore(yt3, yp3, -1)).toThrow(InvalidParameterError);
    expect(() => fbetaScore(yt3, yp3, Number.NaN)).toThrow(InvalidParameterError);
  });

  it("fbetaScore without an average works on multiclass input like f1Score", () => {
    // Previously defaulted to "binary" and threw for labels outside {0, 1}.
    expect(fbetaScore(yt3, yp3, 1)).toBeCloseTo(f1Score(yt3, yp3), 12);
    expect(fbetaScore(yt3, yp3, 1)).toBeCloseTo(0.47500000000000003, 12);
    const yTrue = tensor([0, 1, 1, 0, 1]);
    const yPred = tensor([0, 1, 0, 0, 1]);
    expect(fbetaScore(yTrue, yPred, 2)).toBeCloseTo(0.7142857142857143, 12);
  });

  it("fbetaScore with a beta whose square overflows falls back to recall", () => {
    const yTrue = tensor([0, 1, 1, 0, 1]);
    const yPred = tensor([0, 1, 0, 0, 1]);
    expect(fbetaScore(yTrue, yPred, 1e200)).toBe(recall(yTrue, yPred));
    expect(fbetaScore(yt3, yp3, 1e200, "macro")).toBeCloseTo(recall(yt3, yp3, "macro"), 12);
  });

  it("f1Score is exact for simple counts", () => {
    // 2TP / (2TP + FP + FN) = 4 / 5 with no intermediate rounding.
    expect(f1Score(tensor([0, 1, 1, 0, 1]), tensor([0, 1, 0, 0, 1]))).toBe(0.8);
    expect(f1Score(tensor([0, 1, 1, 0, 1]), tensor([0, 1, 0, 0, 1]), { average: "binary" })).toBe(
      0.8
    );
  });

  it("string labels match scikit-learn", () => {
    const ys = tensor(["cat", "Dog", "cat", "bird", "Dog"]);
    const ps = tensor(["cat", "cat", "cat", "Dog", "Dog"]);
    expect(f1Score(ys, ps, "weighted")).toBeCloseTo(0.52, 12);
    expect(balancedAccuracyScore(ys, ps)).toBeCloseTo(0.5, 12);
    expect(() => f1Score(ys, ps, "binary")).toThrow(/string/i);
  });

  it("classes seen only in the predictions count for macro but not weighted support", () => {
    const yTrue = tensor([0, 0, 1, 1]);
    const yPred = tensor([0, 2, 1, 1]);
    // sklearn: precision macro = (1 + 1 + 0) / 3, recall macro = (0.5 + 1 + 0) / 3
    expect(precision(yTrue, yPred, "macro")).toBeCloseTo(2 / 3, 12);
    expect(recall(yTrue, yPred, "macro")).toBeCloseTo(0.5, 12);
    expect(recall(yTrue, yPred, null)).toEqual([0.5, 1, 0]);
  });
});

describe("v1.5.0 metrics/classification: confusionMatrix", () => {
  it("returns float64 counts for numeric, string and int64 labels", () => {
    expect(confusionMatrix(yt3, yp3).dtype).toBe("float64");
    // The string path used to return the default float32 dtype.
    const cm = confusionMatrix(
      tensor(["cat", "Dog", "cat", "bird", "Dog"]),
      tensor(["cat", "cat", "cat", "Dog", "Dog"])
    );
    expect(cm.dtype).toBe("float64");
    // numpy sorts strings by code unit: "Dog" < "bird" < "cat" (not locale order).
    expect(Array.from(cm.data as Float64Array)).toEqual([1, 0, 1, 1, 0, 0, 0, 0, 2]);
    expect(confusionMatrix(int64([1n, 2n]), int64([1n, 1n])).dtype).toBe("float64");
  });

  it("empty input gives a 0x0 float64 matrix", () => {
    const cm = confusionMatrix(tensor([]), tensor([]));
    expect(cm.shape).toEqual([0, 0]);
    expect(cm.dtype).toBe("float64");
  });

  it("supports normalize like scikit-learn", () => {
    const rows = Array.from(confusionMatrix(yt3, yp3, { normalize: "true" }).data as Float64Array);
    [1, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 2 / 3, 1 / 3].forEach((v, i) => {
      expect(rows[i]).toBeCloseTo(v, 12);
    });
    const cols = Array.from(confusionMatrix(yt3, yp3, { normalize: "pred" }).data as Float64Array);
    [2 / 3, 0, 0, 1 / 3, 1 / 3, 0.5, 0, 2 / 3, 0.5].forEach((v, i) => {
      expect(cols[i]).toBeCloseTo(v, 12);
    });
    const all = Array.from(confusionMatrix(yt3, yp3, { normalize: "all" }).data as Float64Array);
    [0.25, 0, 0, 0.125, 0.125, 0.125, 0, 0.25, 0.125].forEach((v, i) => {
      expect(all[i]).toBeCloseTo(v, 12);
    });
    expect(() => confusionMatrix(yt3, yp3, { normalize: "bad" as never })).toThrow(
      InvalidParameterError
    );
  });

  it("supports a labels subset and order like scikit-learn", () => {
    // sklearn: confusion_matrix(yt3, yp3, labels=[2, 0]) -> [[1, 0], [0, 2]]
    const cm = confusionMatrix(yt3, yp3, { labels: [2, 0] });
    expect(cm.shape).toEqual([2, 2]);
    expect(Array.from(cm.data as Float64Array)).toEqual([1, 0, 0, 2]);
    // Labels that never occur give zero rows/columns.
    expect(Array.from(confusionMatrix(yt3, yp3, { labels: [0, 7] }).data as Float64Array)).toEqual([
      2, 0, 0, 0,
    ]);
    // int64 tensors accept plain integer labels.
    expect(
      Array.from(
        confusionMatrix(int64([1n, 2n]), int64([1n, 1n]), { labels: [2, 1] }).data as Float64Array
      )
    ).toEqual([0, 1, 0, 1]);
    expect(() => confusionMatrix(yt3, yp3, { labels: [] })).toThrow(InvalidParameterError);
    expect(() => confusionMatrix(yt3, yp3, { labels: [0, 0] })).toThrow(/unique/);
    expect(() => confusionMatrix(yt3, yp3, { labels: ["a"] })).toThrow(DTypeError);
    // With labels, empty input yields an all-zero matrix.
    expect(
      Array.from(confusionMatrix(tensor([]), tensor([]), { labels: [0, 1] }).data as Float64Array)
    ).toEqual([0, 0, 0, 0]);
  });
});

describe("v1.5.0 metrics/classification: classificationReport", () => {
  it("keeps the label column wide enough for 'Weighted Avg'", () => {
    // Previously "Weighted Avg" ran straight into the first number.
    const report = classificationReport(tensor([0, 1, 1, 0, 1]), tensor([0, 1, 0, 0, 1]));
    const lines = report.split("\n");
    expect(lines).toContain("Class         Precision   Recall      F1-Score    Support");
    expect(lines).toContain("0             0.6667      1.0000      0.8000      2");
    expect(lines).toContain("Accuracy                              0.8000      5");
    expect(lines).toContain("Macro Avg     0.8333      0.8333      0.8000      5");
    expect(lines).toContain("Weighted Avg  0.8667      0.8000      0.8000      5");
    // sklearn: weighted precision 0.8667, recall 0.8, f1 0.8; macro 0.8333/0.8333/0.8
  });

  it("handles one-class input and int64 labels", () => {
    const report = classificationReport(tensor([1, 1]), tensor([1, 1]));
    expect(report).toContain("Accuracy");
    expect(report).toContain("1.0000");
    expect(classificationReport(int64([0n, 1n]), int64([0n, 1n]))).toContain("Weighted Avg");
    expect(() => classificationReport(tensor(["a"]), tensor(["a"]))).toThrow(InvalidParameterError);
  });
});

describe("v1.5.0 metrics/classification: ranking metrics", () => {
  const y = tensor([0, 1, 1, 0, 1, 0, 1, 1]);
  const s = tensor([0.5, 0.5, 0.8, 0.2, 0.2, 0.9, 0.5, 0.1], f64);

  it("rocAucScore equals scikit-learn with ties and is exact for simple cases", () => {
    // sklearn: roc_auc_score(y, s) = 0.3666666666666667
    expect(rocAucScore(y, s)).toBeCloseTo(0.3666666666666667, 14);
    expect(rocAucScore(tensor([0, 0, 1, 1]), tensor([0.1, 0.4, 0.35, 0.8]))).toBe(0.75);
    expect(rocAucScore(tensor([0, 1]), tensor([0.5, 0.5]))).toBe(0.5);
    expect(rocAucScore(tensor([1, 1]), tensor([0.5, 0.2]))).toBe(0.5);
    expect(rocAucScore(tensor([]), tensor([]))).toBe(0.5);
  });

  it("rocCurve matches roc_curve(drop_intermediate=False) and is float64 even when empty", () => {
    // sklearn: fpr = [0, 1/3, 1/3, 2/3, 1, 1], tpr = [0, 0, .2, .6, .8, 1],
    //          thresholds = [inf, .9, .8, .5, .2, .1]
    const [fpr, tpr, thr] = rocCurve(y, s);
    [0, 1 / 3, 1 / 3, 2 / 3, 1, 1].forEach((v, i) => {
      expect(toArray(fpr)[i]).toBeCloseTo(v, 14);
    });
    [0, 0, 0.2, 0.6, 0.8, 1].forEach((v, i) => {
      expect(toArray(tpr)[i]).toBeCloseTo(v, 14);
    });
    expect(toArray(thr)).toEqual([Infinity, 0.9, 0.8, 0.5, 0.2, 0.1]);
    const empties = [rocCurve(tensor([]), tensor([])), rocCurve(tensor([1, 1]), tensor([1, 2]))];
    for (const empty of empties) {
      for (const t of empty) {
        expect(t.size).toBe(0);
        // Empty results used to come back as float32 while non-empty ones were float64.
        expect(t.dtype).toBe("float64");
      }
    }
  });

  it("rocCurve on the textbook example", () => {
    // sklearn roc_curve([0,0,1,1], [.1,.4,.35,.8], drop_intermediate=False)
    const [fpr, tpr, thr] = rocCurve(tensor([0, 0, 1, 1]), tensor([0.1, 0.4, 0.35, 0.8], f64));
    expect(toArray(fpr)).toEqual([0, 0, 0.5, 0.5, 1]);
    expect(toArray(tpr)).toEqual([0, 0.5, 0.5, 1, 1]);
    expect(toArray(thr)).toEqual([Infinity, 0.8, 0.4, 0.35, 0.1]);
  });

  it("averagePrecisionScore and precisionRecallCurve match scikit-learn", () => {
    expect(averagePrecisionScore(tensor([0, 0, 1, 1]), tensor([0.1, 0.4, 0.35, 0.8]))).toBeCloseTo(
      0.8333333333333333,
      14
    );
    // sklearn: average_precision_score(y, s) = 0.5792857142857143
    expect(averagePrecisionScore(y, s)).toBeCloseTo(0.5792857142857143, 14);
    // sklearn precision_recall_curve(y, s) read from the highest threshold down, preceded by (1, 0):
    //   precision = [1, 0, .5, .6, 4/7, .625], recall = [0, 0, .2, .6, .8, 1]
    const [prec, rec, thr] = precisionRecallCurve(y, s);
    expect(thr.dtype).toBe("float64");
    [1, 0, 0.5, 0.6, 4 / 7, 0.625].forEach((v, i) => {
      expect(toArray(prec)[i]).toBeCloseTo(v, 14);
    });
    [0, 0, 0.2, 0.6, 0.8, 1].forEach((v, i) => {
      expect(toArray(rec)[i]).toBeCloseTo(v, 14);
    });
    expect(toArray(thr)).toEqual([Infinity, 0.9, 0.8, 0.5, 0.2, 0.1]);
    for (const t of precisionRecallCurve(tensor([0, 0]), tensor([0.1, 0.2]))) {
      expect(t.dtype).toBe("float64");
      expect(t.size).toBe(0);
    }
  });

  it("works with int64 scores, bool labels and large inputs (radix path)", () => {
    expect(rocAucScore(tensor([false, false, true, true]), int64([1n, 4n, 3n, 8n]))).toBe(0.75);
    const n = 5000;
    const labels = new Float64Array(n);
    const scores = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      labels[i] = i % 2;
      scores[i] = ((i * 7919) % 1000) / 1000 + labels[i] * 0.05;
    }
    const auc = rocAucScore(tensor(labels), tensor(scores, f64));
    // Cross-check against the pairwise definition on the same data.
    let wins = 0;
    let pairs = 0;
    const pos: number[] = [];
    const neg: number[] = [];
    for (let i = 0; i < n; i++) (labels[i] === 1 ? pos : neg).push(scores[i] as number);
    for (const p of pos) {
      for (const q of neg) {
        pairs++;
        if (p > q) wins++;
        else if (p === q) wins += 0.5;
      }
    }
    expect(auc).toBeCloseTo(wins / pairs, 12);
  });
});

describe("v1.5.0 metrics/classification: logLoss", () => {
  it("uses the dtype epsilon for clipping like scikit-learn", () => {
    // sklearn: log_loss([1,0], [0.0, 1.0] as float64, labels=[0,1]) = 36.04365338911715
    expect(logLoss(tensor([1, 0]), tensor([0, 1], f64))).toBeCloseTo(36.04365338911715, 10);
    // float32 predictions clip at 2^-23: 15.942385152878742
    expect(logLoss(tensor([1, 0]), tensor([0, 1], { dtype: "float32" }))).toBeCloseTo(
      15.942385152878742,
      10
    );
  });

  it("matches scikit-learn on ordinary probabilities", () => {
    // sklearn: log_loss([0,1,1,0], [0.1,0.9,0.8,0.2]) = 0.164252033486018
    expect(logLoss(tensor([0, 1, 1, 0]), tensor([0.1, 0.9, 0.8, 0.2], f64))).toBeCloseTo(
      0.164252033486018,
      13
    );
  });

  it("stays accurate for tiny probabilities of the negative class", () => {
    // -log(1 - 1e-20) is 1e-20; a naive log(1 - p) rounds to exactly 0 because 1 - 1e-20 === 1.
    const loss = logLoss(tensor([0]), tensor([1e-20], f64));
    expect(loss).toBeLessThan(1e-15);
    expect(loss).toBeGreaterThanOrEqual(0);
  });
});

describe("v1.5.0 metrics/classification: other label metrics", () => {
  it("cohenKappaScore matches scikit-learn, with weights and for string labels", () => {
    expect(cohenKappaScore(yt3, yp3)).toBeCloseTo(0.2558139534883721, 12);
    expect(cohenKappaScore(yt3, yp3, "linear")).toBeCloseTo(0.4285714285714286, 12);
    expect(cohenKappaScore(yt3, yp3, "quadratic")).toBeCloseTo(0.6097560975609756, 12);
    expect(
      cohenKappaScore(
        tensor(["cat", "Dog", "cat", "bird", "Dog"]),
        tensor(["cat", "cat", "cat", "Dog", "Dog"])
      )
    ).toBeCloseTo(0.33333333333333337, 12);
    expect(cohenKappaScore(int64([0n, 1n, 1n]), int64([0n, 1n, 1n]))).toBe(1);
    expect(cohenKappaScore(tensor([1, 1]), tensor([1, 1]), "linear")).toBe(1);
    expect(() => cohenKappaScore(yt3, yp3, "cubic" as never)).toThrow(InvalidParameterError);
  });

  it("balancedAccuracyScore and hammingLoss match scikit-learn", () => {
    expect(balancedAccuracyScore(yt3, yp3)).toBeCloseTo(0.5555555555555555, 12);
    expect(hammingLoss(yt3, yp3)).toBe(0.5);
    expect(accuracy(yt3, yp3)).toBe(0.5);
  });

  it("matthewsCorrcoef never leaves [-1, 1] for very large samples", () => {
    // 3e6 samples per class: the product of the four marginal sums exceeds 2^53.
    const n = 6_000_000;
    const t = new Uint8Array(n);
    for (let i = 0; i < n; i++) t[i] = i % 2;
    const perfect = tensor(t);
    const inverted = tensor(t.map((v) => 1 - v));
    expect(matthewsCorrcoef(perfect, perfect)).toBe(1);
    expect(matthewsCorrcoef(perfect, inverted)).toBe(-1);
    expect(matthewsCorrcoef(tensor([0, 1, 1, 0, 1]), tensor([0, 1, 0, 0, 1]))).toBeCloseTo(
      0.6666666666666666,
      12
    );
    expect(matthewsCorrcoef(tensor([1, 1]), tensor([1, 1]))).toBe(0);
  });
});
