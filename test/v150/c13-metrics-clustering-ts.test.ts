import { describe, expect, it } from "vitest";
import {
  adjustedMutualInfoScore,
  adjustedRandScore,
  brierScoreLoss,
  calinskiHarabaszScore,
  completenessScore,
  coverageError,
  d2TweedieScore,
  daviesBouldinScore,
  detCurve,
  explainedVarianceScore,
  fowlkesMallowsScore,
  hingeLoss,
  homogeneityScore,
  labelRankingLoss,
  mae,
  maxError,
  meanAbsoluteError,
  meanGammaDeviance,
  meanPoissonDeviance,
  meanSquaredError,
  meanSquaredLogError,
  medianAbsoluteError,
  mse,
  multilabelConfusionMatrix,
  mutualInfoScore,
  ndcgScore,
  normalizedMutualInfoScore,
  pairwiseCosine,
  pairwiseEuclidean,
  pairwiseManhattan,
  r2Score,
  randScore,
  reciprocalRank,
  rmse,
  rootMeanSquaredError,
  silhouetteSamples,
  silhouetteScore,
  smape,
  topKAccuracyScore,
  vMeasureScore,
  zeroOneLoss,
} from "../../src/metrics";
import { tensor } from "../../src/ndarray";
import { Tensor } from "../../src/ndarray/tensor";
import { __clearSeed, __random, __setSeed } from "../../src/random/random";

const f64 = { dtype: "float64" } as const;
const i32 = { dtype: "int32" } as const;

/** A strided float64 view over `data`, e.g. every second element. */
function view(data: number[], shape: number[], strides: number[], offset = 0): Tensor {
  return Tensor.fromTypedArray({
    data: new Float64Array(data),
    shape,
    dtype: "float64",
    device: "cpu",
    strides,
    offset,
  });
}

function rows(t: Tensor): number[][] {
  return t.toArray() as number[][];
}

describe("v1.5.0 metrics/clustering: label based indices", () => {
  const a = tensor([0, 0, 1, 1, 2, 2, 2, 0, 1, 3, 3, 1], i32);
  const b = tensor([0, 1, 1, 1, 2, 2, 0, 0, 1, 3, 2, 1], i32);

  it("matches scikit-learn on a mixed example", () => {
    // sklearn 1.8: adjusted_mutual_info_score / normalized_mutual_info_score per average_method
    expect(adjustedMutualInfoScore(a, b, "min")).toBeCloseTo(0.44798651544709905, 12);
    expect(adjustedMutualInfoScore(a, b, "geometric")).toBeCloseTo(0.42338357750810124, 12);
    expect(adjustedMutualInfoScore(a, b, "arithmetic")).toBeCloseTo(0.4229643216038148, 12);
    expect(adjustedMutualInfoScore(a, b, "max")).toBeCloseTo(0.40058947937344364, 12);
    expect(normalizedMutualInfoScore(a, b, "min")).toBeCloseTo(0.6570900058135571, 12);
    expect(normalizedMutualInfoScore(a, b, "geometric")).toBeCloseTo(0.6341967566210489, 12);
    expect(normalizedMutualInfoScore(a, b, "arithmetic")).toBeCloseTo(0.6337982027786172, 12);
    expect(normalizedMutualInfoScore(a, b, "max")).toBeCloseTo(0.6121011163617969, 12);
    expect(adjustedRandScore(a, b)).toBeCloseTo(0.4272363150867824, 12);
    expect(fowlkesMallowsScore(a, b)).toBeCloseTo(0.5547001962252291, 12);
    expect(homogeneityScore(a, b)).toBeCloseTo(0.6121011163617969, 12);
    expect(completenessScore(a, b)).toBeCloseTo(0.6570900058135571, 12);
    expect(vMeasureScore(a, b)).toBeCloseTo(0.6337982027786172, 12);
    expect(vMeasureScore(a, b, 2)).toBeCloseTo(0.6413764717725146, 12);
  });

  it("adds randScore and mutualInfoScore (sklearn rand_score, mutual_info_score)", () => {
    expect(randScore(a, b)).toBeCloseTo(0.803030303030303, 12);
    expect(mutualInfoScore(a, b)).toBeCloseTo(0.8312197610323395, 12);
    expect(mutualInfoScore(tensor([0, 0, 1, 1]), tensor([0, 0, 1, 1]))).toBeCloseTo(Math.LN2, 14);
    expect(randScore(tensor([0]), tensor([0]))).toBe(1);
    expect(randScore(tensor([0, 0, 1, 1]), tensor([0, 0, 1, 2]))).toBeCloseTo(5 / 6, 14);
  });

  it("AMI with a single-cluster labeling and 'min'/'geometric' is 0 (was 1)", () => {
    // sklearn: adjusted_mutual_info_score([0,0,0,0], [0,0,1,1], average_method=m) == 0.0
    const one = tensor([0, 0, 0, 0], i32);
    const two = tensor([0, 0, 1, 1], i32);
    expect(adjustedMutualInfoScore(one, two, "min")).toBe(0);
    expect(adjustedMutualInfoScore(one, two, "geometric")).toBe(0);
    expect(adjustedMutualInfoScore(one, two, "arithmetic")).toBe(0);
    expect(adjustedMutualInfoScore(two, one, "min")).toBe(0);
    expect(adjustedMutualInfoScore(one, one)).toBe(1);
  });

  it("AMI of identical partitions is 1 even for two samples (was 0)", () => {
    // sklearn: adjusted_mutual_info_score([0, 1], [0, 1]) == 1.0
    expect(adjustedMutualInfoScore(tensor([0, 1], i32), tensor([0, 1], i32))).toBe(1);
    expect(adjustedMutualInfoScore(tensor([0, 1], i32), tensor([1, 0], i32))).toBe(1);
    expect(adjustedMutualInfoScore(tensor([0, 1, 2, 3], i32), tensor([3, 2, 1, 0], i32))).toBe(1);
  });

  it("AMI groups clusters of equal size without changing the value", () => {
    const x = tensor(
      [
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
        10, 10, 10, 11, 11, 11,
      ],
      i32
    );
    const y = tensor(
      [
        4, 2, 2, 5, 4, 4, 0, 3, 1, 5, 0, 3, 0, 3, 5, 1, 4, 2, 5, 5, 3, 2, 1, 1, 3, 1, 0, 4, 0, 4, 5,
        2, 2, 1, 0, 3,
      ],
      i32
    );
    expect(adjustedMutualInfoScore(x, y)).toBeCloseTo(0.030225752067565666, 12);
    expect(adjustedMutualInfoScore(x, y, "max")).toBeCloseTo(0.023672668574985907, 12);
  });

  it("rejects an unknown averageMethod", () => {
    expect(() => adjustedMutualInfoScore(a, b, "median" as unknown as "min")).toThrow(
      /averageMethod/
    );
    expect(() => normalizedMutualInfoScore(a, b, "median" as unknown as "min")).toThrow(
      /averageMethod/
    );
  });

  it("homogeneity and completeness stay inside [0, 1] for independent labelings", () => {
    const x = tensor([0, 0, 1, 1, 0, 0, 1, 1], i32);
    const y = tensor([0, 1, 0, 1, 0, 1, 0, 1], i32);
    for (const v of [homogeneityScore(x, y), completenessScore(x, y), vMeasureScore(x, y)]) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    expect(homogeneityScore(x, y)).toBeCloseTo(0, 12);
  });

  it("validates labels even for a single sample", () => {
    expect(() => adjustedRandScore(tensor(["a"]), tensor([0], i32))).toThrow(/string/);
    expect(adjustedRandScore(tensor([0], i32), tensor([5], i32))).toBe(1);
  });

  it("reads strided label views", () => {
    const t = view([0, 9, 0, 9, 1, 9, 1, 9], [4], [2]);
    const p = view([5, 5, 7, 7], [4], [1]);
    expect(adjustedRandScore(t, p)).toBe(1);
    expect(normalizedMutualInfoScore(t, p)).toBeCloseTo(1, 14);
  });
});

describe("v1.5.0 metrics/clustering: silhouette and internal indices", () => {
  const X = tensor(
    [
      [0, 0],
      [2, 2],
      [0, 1],
      [2, 3],
      [5, 5],
      [6, 6],
    ],
    f64
  );
  const labels = tensor([0, 0, 1, 1, 2, 2], i32);

  it("matches scikit-learn on a small example", () => {
    // sklearn: silhouette_score / silhouette_samples / davies_bouldin_score / calinski_harabasz_score
    expect(silhouetteScore(X, labels)).toBeCloseTo(0.04483991050903106, 12);
    const samples = Array.from(silhouetteSamples(X, labels).data as Float64Array);
    const expected = [
      -0.18584586550426502, -0.42793859718231575, -0.42793859718231575, -0.18584586550426502,
      0.7174024553896692, 0.7792059330376786,
    ];
    samples.forEach((v, i) => {
      expect(v).toBeCloseTo(expected[i] as number, 12);
    });
    expect(daviesBouldinScore(X, labels)).toBeCloseTo(2.0096528177533353, 12);
    expect(calinskiHarabaszScore(X, labels)).toBeCloseTo(8.222222222222221, 12);
  });

  it("reads transposed (strided) feature matrices", () => {
    const xt = view([0, 2, 0, 2, 5, 6, 0, 2, 1, 3, 5, 6], [6, 2], [1, 6]);
    // columns of the buffer: x = [0,2,0,2,5,6], y = [0,2,1,3,5,6]
    expect(silhouetteScore(xt, labels)).toBeCloseTo(0.04483991050903106, 12);
    expect(daviesBouldinScore(xt, labels)).toBeCloseTo(2.0096528177533353, 12);
    expect(calinskiHarabaszScore(xt, labels)).toBeCloseTo(8.222222222222221, 12);
  });

  it("is invariant to the scale of the data (no overflow or underflow in distances)", () => {
    const reference = silhouetteScore(X, labels);
    for (const scale of [1e200, 1e-200]) {
      const scaled = tensor(
        (X.toArray() as number[][]).map((r) => r.map((v) => v * scale)),
        f64
      );
      expect(silhouetteScore(scaled, labels)).toBeCloseTo(reference, 10);
      expect(daviesBouldinScore(scaled, labels)).toBeCloseTo(2.0096528177533353, 10);
    }
  });

  it("raises a ShapeError when X and labels disagree in silhouetteSamples", () => {
    const short = tensor(
      [
        [0, 0],
        [1, 1],
        [2, 2],
      ],
      f64
    );
    expect(() => silhouetteSamples(short, labels)).toThrow(/labels length/);
    expect(() => silhouetteScore(short, labels)).toThrow(/labels length/);
  });

  it("sampling honors the seed and leaves the global random stream alone", () => {
    const pts: number[][] = [];
    const lab: number[] = [];
    for (let i = 0; i < 60; i++) {
      pts.push([(i % 3) * 10 + Math.sin(i), Math.cos(i * 1.7)]);
      lab.push(i % 3);
    }
    const Xs = tensor(pts, f64);
    const ls = tensor(lab, i32);

    __setSeed(11);
    const expectedNext = __random();
    __setSeed(11);
    const s1 = silhouetteScore(Xs, ls, "euclidean", { sampleSize: 30, randomState: 5 });
    const nextAfter = __random();
    const s2 = silhouetteScore(Xs, ls, "euclidean", { sampleSize: 30, randomState: 5 });
    const s3 = silhouetteScore(Xs, ls, "euclidean", { sampleSize: 30, randomState: 6 });
    __clearSeed();

    expect(nextAfter).toBe(expectedNext);
    expect(s1).toBe(s2);
    expect(s3).not.toBe(s1);
    expect(s1).toBeGreaterThan(0.5);
    // Sampling every sample is the full computation.
    expect(silhouetteScore(Xs, ls, "euclidean", { sampleSize: 60, randomState: 1 })).toBeCloseTo(
      silhouetteScore(Xs, ls),
      12
    );
  });

  it("rejects a non-finite randomState", () => {
    expect(() =>
      silhouetteScore(X, labels, "euclidean", { sampleSize: 4, randomState: Number.NaN })
    ).toThrow(/randomState/);
  });

  it("precomputed diagonal tolerance follows the matrix dtype", () => {
    const base = [
      [0, 1, 10, 11],
      [1, 0, 9, 10],
      [10, 9, 0, 1],
      [11, 10, 1, 0],
    ];
    const lab = tensor([0, 0, 1, 1], i32);
    const expected = silhouetteScore(tensor(base, f64), lab, "precomputed");

    // A float32 diagonal of 1e-7 is rounding noise at float32 precision.
    const noisy32 = tensor(
      base.map((r, i) => r.map((v, j) => (i === j ? 1e-7 : v))),
      { dtype: "float32" }
    );
    expect(silhouetteScore(noisy32, lab, "precomputed")).toBeCloseTo(expected, 5);

    // The same offset in float64 is a genuinely non-zero diagonal.
    const noisy64 = tensor(
      base.map((r, i) => r.map((v, j) => (i === j ? 1e-7 : v))),
      f64
    );
    expect(() => silhouetteScore(noisy64, lab, "precomputed")).toThrow(/diagonal/);
  });

  it("precomputed matrices must be finite and non-negative everywhere", () => {
    const lab = tensor([0, 0, 1, 1], i32);
    const bad = tensor(
      [
        [0, 1, 10, 11],
        [1, 0, 9, -10],
        [10, 9, 0, 1],
        [11, 10, 1, 0],
      ],
      f64
    );
    expect(() => silhouetteScore(bad, lab, "precomputed")).toThrow(/non-negative/);
    const nan = tensor(
      [
        [0, 1, 10, 11],
        [1, 0, 9, Number.NaN],
        [10, 9, 0, 1],
        [11, 10, 1, 0],
      ],
      f64
    );
    expect(() => silhouetteSamples(nan, lab, "precomputed")).toThrow(/finite/);
  });

  it("Davies-Bouldin and Calinski-Harabasz validate X even when there is one cluster", () => {
    const withNaN = tensor(
      [
        [0, 0],
        [Number.NaN, 1],
        [2, 2],
      ],
      f64
    );
    const single = tensor([0, 0, 0], i32);
    expect(() => daviesBouldinScore(withNaN, single)).toThrow(/finite/);
    expect(() => calinskiHarabaszScore(withNaN, single)).toThrow(/finite/);
  });
});

describe("v1.5.0 metrics/extra: input handling", () => {
  it("reads strided views instead of the raw buffer", () => {
    // yTrue = [1, 3, 5], yPred = [2, 4, 6]; the raw buffers hold [1, 2, 3, 4, 5, 6].
    const yt = view([1, 2, 3, 4, 5, 6], [3], [2], 0);
    const yp = view([1, 2, 3, 4, 5, 6], [3], [2], 1);
    // |1-2|/1.5, |3-4|/3.5, |5-6|/5.5 averaged
    expect(smape(yt, yp)).toBeCloseTo((1 / 1.5 + 1 / 3.5 + 1 / 5.5) / 3, 14);
    expect(meanSquaredLogError(yt, yp)).toBeCloseTo(
      ((Math.log(2) - Math.log(3)) ** 2 +
        (Math.log(4) - Math.log(5)) ** 2 +
        (Math.log(6) - Math.log(7)) ** 2) /
        3,
      14
    );
    expect(zeroOneLoss(yt, yp)).toBe(1);
    expect(zeroOneLoss(yt, yt)).toBe(0);
  });

  it("rejects NaN, infinity and string tensors instead of computing garbage", () => {
    const ok = tensor([1, 2, 3], f64);
    const nan = tensor([1, Number.NaN, 3], f64);
    expect(() => smape(nan, ok)).toThrow(/finite/);
    expect(() => meanSquaredLogError(ok, nan)).toThrow(/finite/);
    expect(() => hingeLoss(tensor([-1, 1, 1], f64), nan)).toThrow(/finite/);
    expect(() => meanPoissonDeviance(ok, tensor([1, Infinity, 3], f64))).toThrow(/finite/);
    expect(() => detCurve(tensor([0, 1, 1], f64), nan)).toThrow(/finite/);
    expect(() => smape(tensor(["a", "b", "c"]), ok)).toThrow(/string/);
  });

  it("zeroOneLoss supports string labels and normalize=false", () => {
    const yt = tensor(["cat", "dog", "cat", "bird"]);
    const yp = tensor(["cat", "cat", "cat", "dog"]);
    expect(zeroOneLoss(yt, yp)).toBe(0.5);
    expect(zeroOneLoss(yt, yp, { normalize: false })).toBe(2);
  });
});

describe("v1.5.0 metrics/extra: classification losses", () => {
  it("hingeLoss maps {0, 1} labels onto {-1, +1} like scikit-learn", () => {
    // sklearn hinge_loss([0,1,1,0], [-1,2,0.2,0.3]) == 0.525 (was 0.7)
    expect(hingeLoss(tensor([0, 1, 1, 0], i32), tensor([-1, 2, 0.2, 0.3], f64))).toBeCloseTo(
      0.525,
      14
    );
    expect(hingeLoss(tensor([-1, 1, 1, -1], i32), tensor([-1, 2, 0.2, 0.3], f64))).toBeCloseTo(
      0.525,
      14
    );
    expect(() => hingeLoss(tensor([0, 1, 2], i32), tensor([1, 1, 1], f64))).toThrow(/binary/);
    expect(() => hingeLoss(tensor([3, 3], i32), tensor([1, 1], f64))).toThrow(/single label/);
    expect(hingeLoss(tensor([1, 1], i32), tensor([2, 0.5], f64))).toBeCloseTo(0.25, 14);
  });

  it("brierScoreLoss uses the positive label and validates probabilities", () => {
    // sklearn brier_score_loss([1,2,2,1], [0.1,0.9,0.8,0.2]) == 0.025 (pos_label = 2)
    const y = tensor([1, 2, 2, 1], i32);
    const p = tensor([0.1, 0.9, 0.8, 0.2], f64);
    expect(brierScoreLoss(y, p)).toBeCloseTo(0.025, 14);
    expect(brierScoreLoss(y, p, 2)).toBeCloseTo(0.025, 14);
    expect(brierScoreLoss(y, tensor([0.9, 0.1, 0.2, 0.8], f64), 1)).toBeCloseTo(0.025, 14);
    expect(brierScoreLoss(tensor([-1, 1], i32), tensor([0.2, 0.7], f64))).toBeCloseTo(
      (0.04 + 0.09) / 2,
      14
    );
    expect(() => brierScoreLoss(tensor([0, 1], i32), tensor([1.5, 0.2], f64))).toThrow(/\[0, 1\]/);
    expect(() => brierScoreLoss(tensor([0, 1, 2], i32), tensor([0.1, 0.2, 0.3], f64))).toThrow(
      /binary/
    );
    // sklearn: labels outside {0, 1} and {-1, 1} take the largest label as the positive class,
    // so {-1, 0} treats 0 as positive and a lone label 5 is positive.
    expect(brierScoreLoss(tensor([-1, 0, 0, -1], i32), p)).toBeCloseTo(0.025, 14);
    expect(brierScoreLoss(tensor([5, 5], i32), tensor([0.1, 0.2], f64))).toBeCloseTo(0.725, 14);
    expect(brierScoreLoss(tensor([0, 0], i32), tensor([0.1, 0.2], f64))).toBeCloseTo(0.025, 14);
  });

  it("topKAccuracyScore breaks ties like scikit-learn and clamps the default k", () => {
    // Equal scores: sklearn ranks the larger class index first, so class 0 is never in the top 1.
    const tied = tensor(
      [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ],
      f64
    );
    expect(topKAccuracyScore(tensor([0, 0, 0], i32), tied, 1)).toBe(0);
    expect(topKAccuracyScore(tensor([2, 2, 2], i32), tied, 1)).toBe(1);

    const scores = tensor(
      [
        [0.1, 0.5, 0.4],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
      ],
      f64
    );
    const y = tensor([0, 1, 2], i32);
    expect(topKAccuracyScore(y, scores, 2)).toBeCloseTo(2 / 3, 14);
    // Default k is 5 but only 3 classes exist; previously this threw.
    expect(topKAccuracyScore(y, scores)).toBe(1);
    expect(() => topKAccuracyScore(y, scores, 4)).toThrow(/k must be/);
    expect(() => topKAccuracyScore(y, scores, 1.5)).toThrow(/k must be/);
  });

  it("topKAccuracyScore rejects labels that are not class indices", () => {
    const scores = tensor([[0.2, 0.8]], f64);
    expect(() => topKAccuracyScore(tensor([2], i32), scores, 1)).toThrow(/class indices/);
    expect(() => topKAccuracyScore(tensor([0.5], f64), scores, 1)).toThrow(/class indices/);
  });

  it("topKAccuracyScore reads strided score matrices", () => {
    // scores = [[0.1, 0.9], [0.8, 0.2]] stored column-major
    const s = view([0.1, 0.8, 0.9, 0.2], [2, 2], [1, 2]);
    expect(topKAccuracyScore(tensor([1, 0], i32), s, 1)).toBe(1);
  });
});

describe("v1.5.0 metrics/extra: regression deviances", () => {
  it("Poisson and Gamma deviance stay accurate when predictions are close to targets", () => {
    // mpmath (50 digits): 2*(y*ln(y/p) - y + p) for y=1e8, p=1e8+1 is 9.999999933333334e-9;
    // the direct formula (and scikit-learn) evaluates it to 0.
    const y = tensor([1e8], f64);
    const p = tensor([1e8 + 1], f64);
    expect(meanPoissonDeviance(y, p) / 9.999999933333334e-9).toBeCloseTo(1, 7);
    // gamma: 2*(ln(p/y) + y/p - 1) = 9.999999866666668e-17
    expect(meanGammaDeviance(y, p) / 9.999999866666668e-17).toBeCloseTo(1, 7);

    // y=3, p=3+1e-7: poisson 3.333333259259261e-15, gamma 1.111111061728397e-15
    const y3 = tensor([3], f64);
    const p3 = tensor([3 + 1e-7], f64);
    expect(meanPoissonDeviance(y3, p3) / 3.333333259259261e-15).toBeCloseTo(1, 5);
    expect(meanGammaDeviance(y3, p3) / 1.111111061728397e-15).toBeCloseTo(1, 5);
  });

  it("d2TweedieScore matches scikit-learn for several powers", () => {
    const yt = tensor([1, 2, 3, 4], f64);
    const yp = tensor([1.1, 2.1, 2.9, 4.1], f64);
    const expected: Array<[number, number]> = [
      [0, 0.992],
      [1, 0.9905639949673487],
      [1.5, 0.9890189255417358],
      [2, 0.9867467449974793],
      [3, 0.9794276050936017],
      [-1, 0.9918933333333345],
    ];
    for (const [power, value] of expected) {
      expect(d2TweedieScore(yt, yp, power)).toBeCloseTo(value, 12);
    }
  });

  it("d2TweedieScore enforces the domain of each power", () => {
    const pos = tensor([1, 2, 3], f64);
    expect(() => d2TweedieScore(tensor([-1, 2, 3], f64), pos, 1)).toThrow(/non-negative/);
    expect(() => d2TweedieScore(tensor([-1, 2, 3], f64), pos, 1.5)).toThrow(/non-negative/);
    expect(() => d2TweedieScore(tensor([0, 2, 3], f64), pos, 3)).toThrow(/positive/);
    expect(() => d2TweedieScore(tensor([0, 0, 0], f64), pos, 1)).toThrow(/positive mean/);
    // Negative targets are fine for power <= 0.
    expect(Number.isFinite(d2TweedieScore(tensor([-1, 2, 3], f64), pos, 0))).toBe(true);
    expect(Number.isFinite(d2TweedieScore(tensor([-1, 2, 3], f64), pos, -1))).toBe(true);
  });
});

describe("v1.5.0 metrics/extra: confusion, ranking and DET", () => {
  it("multilabelConfusionMatrix returns labels in the requested order", () => {
    // sklearn multilabel_confusion_matrix(..., labels=[2, 0, 1])
    const result = multilabelConfusionMatrix(
      tensor([0, 1, 2, 1], i32),
      tensor([0, 2, 2, 1], i32),
      [2, 0, 1]
    );
    expect(result).toEqual([
      { label: 2, tn: 2, fp: 1, fn: 0, tp: 1 },
      { label: 0, tn: 3, fp: 0, fn: 0, tp: 1 },
      { label: 1, tn: 2, fp: 0, fn: 1, tp: 1 },
    ]);
    // Labels that never occur still report all negatives.
    expect(multilabelConfusionMatrix(tensor([0, 1], i32), tensor([0, 1], i32), [7])).toEqual([
      { label: 7, tn: 2, fp: 0, fn: 0, tp: 0 },
    ]);
    // Duplicates collapse to one entry.
    expect(
      multilabelConfusionMatrix(tensor([0, 1], i32), tensor([0, 1], i32), [1, 1, 0]).map(
        (r) => r.label
      )
    ).toEqual([1, 0]);
    expect(() =>
      multilabelConfusionMatrix(tensor([0], i32), tensor([0], i32), [Number.NaN])
    ).toThrow(/finite/);
  });

  it("coverageError counts labels tied with the lowest true label (sklearn)", () => {
    // sklearn coverage_error([[1,0,0],[0,1,1]], [[.5,.5,.5],[.2,.2,.2]]) == 3.0 (was 2)
    expect(
      coverageError(
        tensor(
          [
            [1, 0, 0],
            [0, 1, 1],
          ],
          i32
        ),
        tensor(
          [
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],
          ],
          f64
        )
      )
    ).toBe(3);

    // Rows without a true label contribute 0; sklearn gives 2.0 here.
    expect(
      coverageError(
        tensor(
          [
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
          ],
          i32
        ),
        tensor(
          [
            [0.5, 0.5, 0.5],
            [0.2, 0.3, 0.1],
            [0.1, 0.2, 0.3],
          ],
          f64
        )
      )
    ).toBe(2);

    // sklearn coverage_error == 4.333333333333333, label_ranking_loss == 0.6111111111111112
    const Y = tensor(
      [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0],
      ],
      i32
    );
    const S = tensor(
      [
        [0.5, 0.5, 0.2, 0.9, 0.5],
        [0.2, 0.3, 0.1, 0.3, 0.4],
        [0.7, 0.1, 0.7, 0.1, 0.7],
      ],
      f64
    );
    expect(coverageError(Y, S)).toBeCloseTo(4.333333333333333, 14);
    expect(labelRankingLoss(Y, S)).toBeCloseTo(0.6111111111111112, 14);
  });

  it("labelRankingLoss agrees with the pairwise definition (sklearn: 0.5)", () => {
    expect(
      labelRankingLoss(
        tensor(
          [
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
          ],
          i32
        ),
        tensor(
          [
            [0.5, 0.5, 0.5],
            [0.2, 0.3, 0.1],
            [0.1, 0.2, 0.3],
          ],
          f64
        )
      )
    ).toBe(0.5);
  });

  it("ranking losses require a binary indicator matrix", () => {
    const s = tensor([[0.1, 0.2]], f64);
    expect(() => coverageError(tensor([[2, 0]], i32), s)).toThrow(/binary/);
    expect(() => labelRankingLoss(tensor([[0.5, 0]], f64), s)).toThrow(/binary/);
    expect(() => coverageError(tensor([[1, 0]], i32), tensor([[Number.NaN, 0.1]], f64))).toThrow(
      /finite/
    );
  });

  it("detCurve reads strided scores and keeps thresholds in decreasing order", () => {
    const y = tensor([0, 0, 1, 1], i32);
    const scores = view([0.1, 0, 0.4, 0, 0.35, 0, 0.8, 0], [4], [2]);
    const { fpr, fnr, thresholds } = detCurve(y, scores);
    expect(thresholds).toEqual([0.8, 0.4, 0.35, 0.1]);
    expect(fpr).toEqual([0, 0.5, 0.5, 1]);
    expect(fnr).toEqual([0.5, 0.5, 0, 0]);
  });
});

describe("v1.5.0 metrics/pairwise", () => {
  const X = tensor(
    [
      [0, 0],
      [1, 0],
      [3, 4],
    ],
    f64
  );
  const Y = tensor(
    [
      [1, 1],
      [0, 0],
    ],
    f64
  );

  it("supports a second operand Y (sklearn euclidean/manhattan/cosine_distances(X, Y))", () => {
    expect(rows(pairwiseEuclidean(X, Y))).toEqual([
      [Math.SQRT2, 0],
      [1, 1],
      [3.605551275463989, 5],
    ]);
    expect(rows(pairwiseManhattan(X, Y))).toEqual([
      [2, 0],
      [1, 1],
      [5, 7],
    ]);
    const cos = rows(pairwiseCosine(X, Y));
    expect(cos[1]?.[0]).toBeCloseTo(0.29289321881345254, 14);
    expect(cos[2]?.[0]).toBeCloseTo(0.010050506338833531, 14);
    // The zero row of X has no direction: distance 1 to every row of Y, including a zero row.
    expect(cos[0]).toEqual([1, 1]);
    expect(pairwiseEuclidean(X, Y).shape).toEqual([3, 2]);
    expect(() => pairwiseEuclidean(X, tensor([[1, 2, 3]], f64))).toThrow(/same number of features/);
    expect(() => pairwiseCosine(X, tensor([1, 2], f64))).toThrow(/2D/);
  });

  it("cosine distance to a zero row is 1 (was 0), the diagonal stays 0", () => {
    // sklearn cosine_distances([[0,0],[1,0],[0,0]]) == [[0,1,1],[1,0,1],[1,1,0]]
    const D = pairwiseCosine(
      tensor(
        [
          [0, 0],
          [1, 0],
          [0, 0],
        ],
        f64
      )
    );
    expect(rows(D)).toEqual([
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 0],
    ]);
  });

  it("cosine distances are never negative and stay in [0, 2]", () => {
    const D = rows(
      pairwiseCosine(
        tensor(
          [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
          ],
          f64
        )
      )
    );
    for (const r of D)
      for (const v of r) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(2);
      }
    expect(D[0]?.[1]).toBeCloseTo(0, 14);
    expect(D[0]?.[2]).toBeCloseTo(2, 14);
  });

  it("Euclidean distance does not overflow or underflow", () => {
    const big = rows(
      pairwiseEuclidean(
        tensor(
          [
            [0, 0],
            [3e200, 4e200],
          ],
          f64
        )
      )
    );
    expect(big[0]?.[1] as number).toBeGreaterThan(4.999e200);
    expect(big[0]?.[1] as number).toBeLessThan(5.001e200);
    const tiny = rows(
      pairwiseEuclidean(
        tensor(
          [
            [0, 0],
            [3e-200, 4e-200],
          ],
          f64
        )
      )
    );
    expect(tiny[0]?.[1]).toBeCloseTo(5e-200, 210);
    expect(tiny[0]?.[1]).toBeGreaterThan(0);
  });

  it("reads strided inputs and rejects non-finite values", () => {
    const transposed = view([0, 3, 4, 0], [2, 2], [1, 2]); // [[0, 4], [3, 0]]
    expect(rows(pairwiseEuclidean(transposed))).toEqual([
      [0, 5],
      [5, 0],
    ]);
    expect(() =>
      pairwiseEuclidean(
        tensor(
          [
            [0, Number.NaN],
            [1, 2],
          ],
          f64
        )
      )
    ).toThrow(/finite/);
    expect(() => pairwiseManhattan(tensor([["a", "b"]]))).toThrow(/string/);
  });

  it("ndcgScore shares gain among tied scores (sklearn ignore_ties=False)", () => {
    expect(ndcgScore(tensor([3, 2, 1, 0], f64), tensor([1, 1, 1, 1], f64))).toBeCloseTo(
      0.8069136566720543,
      14
    );
    expect(
      ndcgScore(tensor([[0, 1, 0, 2]], f64), tensor([[0.5, 0.5, 0.2, 0.5]], f64), 2)
    ).toBeCloseTo(0.6199062332840657, 14);
    expect(
      ndcgScore(
        tensor(
          [
            [3, 2, 1, 0],
            [1, 0, 0, 3],
          ],
          f64
        ),
        tensor(
          [
            [1, 2, 3, 4],
            [1, 1, 0, 1],
          ],
          f64
        )
      )
    ).toBeCloseTo(0.6981687709625342, 14);
    // The order of tied documents must not matter.
    expect(ndcgScore(tensor([1, 3, 0, 2], f64), tensor([5, 5, 5, 5], f64))).toBeCloseTo(
      ndcgScore(tensor([3, 2, 1, 0], f64), tensor([5, 5, 5, 5], f64)),
      14
    );
  });

  it("ndcgScore validates k and relevance", () => {
    const t = tensor([3, 2, 1, 0], f64);
    expect(() => ndcgScore(t, t, 0)).toThrow(/positive integer/);
    expect(() => ndcgScore(t, t, -2)).toThrow(/positive integer/);
    expect(() => ndcgScore(t, t, 1.5)).toThrow(/positive integer/);
    expect(() => ndcgScore(tensor([-1, 2], f64), tensor([1, 2], f64))).toThrow(/non-negative/);
    expect(ndcgScore(t, t, 100)).toBeCloseTo(1, 14);
  });

  it("reciprocalRank averages over a batch of queries", () => {
    const yTrue = tensor(
      [
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 0],
      ],
      f64
    );
    const yScore = tensor(
      [
        [0.3, 0.2, 0.1],
        [0.1, 0.9, 0.5],
        [1, 2, 3],
      ],
      f64
    );
    // ranks 3, 3 and no relevant item
    expect(reciprocalRank(yTrue, yScore)).toBeCloseTo((1 / 3 + 1 / 3 + 0) / 3, 14);
    // Tied scores keep input order.
    expect(reciprocalRank(tensor([0, 1, 0], f64), tensor([1, 1, 1], f64))).toBeCloseTo(0.5, 14);
    expect(() => reciprocalRank(tensor([1, 0], f64), tensor([[1, 0]], f64))).toThrow(/ndim/);
    expect(() => reciprocalRank(tensor([[1, 0]], f64), tensor([[1, 0, 1]], f64))).toThrow(/shape/);
  });
});

describe("v1.5.0 metrics/regression", () => {
  it("summation error does not grow with the number of samples", () => {
    // 1e6 residuals of exactly 0.1: a plain running sum is off by ~1.3e-12 in the mean.
    const n = 1_000_000;
    const yt = tensor(new Float64Array(n), f64);
    const yp = tensor(new Float64Array(n).fill(0.1), f64);
    expect(Math.abs(mae(yt, yp) - 0.1)).toBeLessThan(2e-16);
    expect(Math.abs(mse(yt, yp) - 0.1 * 0.1)).toBeLessThan(2e-18);
  });

  it("medianAbsoluteError cannot overflow when averaging the two middle errors", () => {
    expect(medianAbsoluteError(tensor([1.7e308, 1.7e308], f64), tensor([0, 0], f64))).toBe(1.7e308);
  });

  it("matches scikit-learn on a regression example", () => {
    // sklearn: r2_score, explained_variance_score, median_absolute_error, max_error
    const yt = tensor([3, -0.5, 2, 7], f64);
    const yp = tensor([2.5, 0, 2, 8], f64);
    expect(r2Score(yt, yp)).toBeCloseTo(0.9486081370449679, 14);
    expect(mse(yt, yp)).toBeCloseTo(0.375, 14);
    expect(rmse(yt, yp)).toBeCloseTo(Math.sqrt(0.375), 14);
    expect(mae(yt, yp)).toBeCloseTo(0.5, 14);
    expect(medianAbsoluteError(yt, yp)).toBeCloseTo(0.5, 14);
    expect(maxError(yt, yp)).toBe(1);
  });

  it("reports non-finite values with one consistent message", () => {
    const ok = tensor([1, 2, 3], f64);
    const bad = tensor([1, Number.NaN, 3], f64);
    for (const metric of [mse, mae, r2Score, medianAbsoluteError, maxError]) {
      expect(() => metric(ok, bad)).toThrow(
        /yPred must contain only finite numbers; found NaN at index 1/
      );
    }
  });

  it("exposes long-name aliases for mse, rmse and mae", () => {
    const yt = tensor([3, -0.5, 2, 7], f64);
    const yp = tensor([2.5, 0, 2, 8], f64);
    expect(meanSquaredError(yt, yp)).toBe(mse(yt, yp));
    expect(rootMeanSquaredError(yt, yp)).toBe(rmse(yt, yp));
    expect(meanAbsoluteError(yt, yp)).toBe(mae(yt, yp));
  });
});

describe("v1.5.0 metrics: review fixes", () => {
  it("sums that overflow stay infinite instead of turning into NaN", () => {
    expect(mse(tensor([1e200, 0], f64), tensor([0, 0], f64))).toBe(Number.POSITIVE_INFINITY);
    expect(mae(tensor([1.7e308, 1.7e308], f64), tensor([0, 0], f64))).toBe(
      Number.POSITIVE_INFINITY
    );
  });

  it("r2Score and explainedVarianceScore treat constant targets exactly", () => {
    // The rounded mean of three 0.1 values is not 0.1, which used to leave a tiny non-zero
    // total sum of squares and an R2 of about -1.7e31.
    const constant = tensor([0.1, 0.1, 0.1], f64);
    expect(r2Score(constant, tensor([0.2, 0.1, 0.1], f64))).toBe(0);
    expect(r2Score(constant, constant)).toBe(1);
    expect(explainedVarianceScore(constant, tensor([0.2, 0.1, 0.1], f64))).toBe(0);
    // A constant offset leaves constant residuals, i.e. zero residual variance.
    const yt = tensor([1, 2, 3, 4], f64);
    expect(explainedVarianceScore(yt, tensor([1.1, 2.1, 3.1, 4.1], f64))).toBe(1);
  });

  it("Calinski-Harabasz is scale invariant for extreme coordinates", () => {
    const labels = tensor([0, 0, 1, 1], i32);
    const base = [
      [0, 0],
      [1, 0],
      [3, 1],
      [4, 1],
    ];
    const reference = calinskiHarabaszScore(tensor(base, f64), labels);
    for (const scale of [1e160, 1e-160]) {
      const scaled = tensor(
        base.map((r) => r.map((v) => v * scale)),
        f64
      );
      expect(calinskiHarabaszScore(scaled, labels)).toBeCloseTo(reference, 10);
    }
  });

  it("Davies-Bouldin is 0 for clusters without scatter that share a centroid", () => {
    const X = tensor(
      [
        [0, 0],
        [0, 0],
        [1, 1],
      ],
      f64
    );
    expect(daviesBouldinScore(X, tensor([0, 1, 2], i32))).toBe(0);
    // Clusters with scatter on top of each other are infinitely hard to separate.
    const overlap = tensor([[-1], [1], [-2], [2]], f64);
    expect(daviesBouldinScore(overlap, tensor([0, 0, 1, 1], i32))).toBe(Number.POSITIVE_INFINITY);
  });

  it("precomputed silhouette sampling matches the euclidean sample for the same seed", () => {
    const pts: number[][] = [];
    const lab: number[] = [];
    for (let i = 0; i < 40; i++) {
      pts.push([(i % 4) * 7 + Math.sin(i * 3.1), Math.cos(i * 1.3)]);
      lab.push(i % 4);
    }
    const Xs = tensor(pts, f64);
    const ls = tensor(lab, i32);
    const options = { sampleSize: 15, randomState: 3 };
    expect(silhouetteScore(pairwiseEuclidean(Xs), ls, "precomputed", options)).toBeCloseTo(
      silhouetteScore(Xs, ls, "euclidean", options),
      12
    );
  });

  it("Tweedie deviance with a non-integer power matches mpmath near the target", () => {
    // mpmath (60 digits): 2 * (y^(2-p)/((1-p)(2-p)) - y mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
    // for y = 7, mu = 7.00007, p = -0.5 is 1.2964224638407905e-8. Two samples, the second
    // exact, make the model deviance equal to this single term; D2 = 1 - dev / dev_null.
    const yTrue = tensor([7, 9], f64);
    const yPred = tensor([7.00007, 9], f64);
    const mean = 8;
    const p = -0.5;
    const dev = (y: number, mu: number) =>
      2 *
      (y ** (2 - p) / ((1 - p) * (2 - p)) -
        (y * mu ** (1 - p)) / (1 - p) +
        mu ** (2 - p) / (2 - p));
    const expected = 1 - 1.2964224638407905e-8 / (dev(7, mean) + dev(9, mean));
    expect(d2TweedieScore(yTrue, yPred, p)).toBeCloseTo(expected, 12);
  });
});
