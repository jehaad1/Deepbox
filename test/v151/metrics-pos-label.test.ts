import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { f1Score, fbetaScore, jaccardScore, precision, recall } from "../../src/metrics";
import { tensor } from "../../src/ndarray";

// Expected values come from scikit-learn 1.9.1 with average="binary", pos_label and
// zero_division=0 (precision_score, recall_score, f1_score, fbeta_score with beta=2,
// jaccard_score).

type Label = number | string;
type Case = {
  case: { yt: Label[]; yp: Label[]; pos: Label | null; sw?: number[] };
  ref: { precision: number; recall: number; f1: number; jaccard: number; fbeta: number };
};

const CASES: Case[] = [
  {
    case: { yt: [-1, 1, 1, -1, 1, 1, -1], yp: [-1, 1, -1, -1, 1, -1, 1], pos: 1 },
    ref: {
      precision: 0.6666666666666666,
      recall: 0.5,
      f1: 0.5714285714285714,
      jaccard: 0.4,
      fbeta: 0.5263157894736842,
    },
  },
  {
    case: { yt: [-1, 1, 1, -1, 1, 1, -1], yp: [-1, 1, -1, -1, 1, -1, 1], pos: -1 },
    ref: {
      precision: 0.5,
      recall: 0.6666666666666666,
      f1: 0.5714285714285714,
      jaccard: 0.4,
      fbeta: 0.625,
    },
  },
  {
    case: { yt: [-1, 1, 1, -1, 1, 1, -1], yp: [-1, 1, -1, -1, 1, -1, 1], pos: null },
    ref: {
      precision: 0.6666666666666666,
      recall: 0.5,
      f1: 0.5714285714285714,
      jaccard: 0.4,
      fbeta: 0.5263157894736842,
    },
  },
  {
    case: { yt: [1, 2, 2, 1, 2, 2], yp: [1, 2, 1, 1, 2, 1], pos: 1 },
    ref: {
      precision: 0.5,
      recall: 1.0,
      f1: 0.6666666666666666,
      jaccard: 0.5,
      fbeta: 0.8333333333333334,
    },
  },
  {
    case: { yt: [1, 2, 2, 1, 2, 2], yp: [1, 2, 1, 1, 2, 1], pos: 2 },
    ref: {
      precision: 1.0,
      recall: 0.5,
      f1: 0.6666666666666666,
      jaccard: 0.5,
      fbeta: 0.5555555555555556,
    },
  },
  {
    case: { yt: [1, 2, 2, 1, 2, 2], yp: [1, 2, 1, 1, 2, 1], pos: 2, sw: [1, 2, 3, 1, 0.5, 2] },
    ref: {
      precision: 1.0,
      recall: 0.3333333333333333,
      f1: 0.5,
      jaccard: 0.3333333333333333,
      fbeta: 0.38461538461538464,
    },
  },
  {
    case: {
      yt: ["spam", "ham", "spam", "spam", "ham", "ham"],
      yp: ["spam", "ham", "ham", "spam", "spam", "ham"],
      pos: "spam",
    },
    ref: {
      precision: 0.6666666666666666,
      recall: 0.6666666666666666,
      f1: 0.6666666666666666,
      jaccard: 0.5,
      fbeta: 0.6666666666666666,
    },
  },
  {
    case: {
      yt: ["spam", "ham", "spam", "spam", "ham", "ham"],
      yp: ["spam", "ham", "ham", "spam", "spam", "ham"],
      pos: "ham",
    },
    ref: {
      precision: 0.6666666666666666,
      recall: 0.6666666666666666,
      f1: 0.6666666666666666,
      jaccard: 0.5,
      fbeta: 0.6666666666666666,
    },
  },
  {
    case: {
      yt: ["spam", "ham", "spam", "spam", "ham", "ham"],
      yp: ["spam", "ham", "ham", "spam", "spam", "ham"],
      pos: "ham",
      sw: [1, 1, 2, 1, 3, 1],
    },
    ref: {
      precision: 0.5,
      recall: 0.4,
      f1: 0.4444444444444444,
      jaccard: 0.2857142857142857,
      fbeta: 0.4166666666666667,
    },
  },
  {
    case: { yt: [0, 0, 0], yp: [0, 0, 0], pos: 1 },
    ref: { precision: 0.0, recall: 0.0, f1: 0.0, jaccard: 0.0, fbeta: 0.0 },
  },
  {
    case: { yt: [1, 1, 1], yp: [1, 1, 0], pos: 1 },
    ref: {
      precision: 1.0,
      recall: 0.6666666666666666,
      f1: 0.8,
      jaccard: 0.6666666666666666,
      fbeta: 0.7142857142857143,
    },
  },
];

function toTensor(values: Label[]) {
  return typeof values[0] === "string" ? tensor(values as string[]) : tensor(values as number[]);
}

describe("posLabel of the binary average matches scikit-learn", () => {
  CASES.forEach(({ case: c, ref }, index) => {
    it(`case ${index}: posLabel=${String(c.pos)} labels=${JSON.stringify([...new Set(c.yt)])}`, () => {
      const yt = toTensor(c.yt);
      const yp = toTensor(c.yp);
      const options = {
        average: "binary" as const,
        zeroDivision: 0,
        ...(c.pos !== null ? { posLabel: c.pos } : {}),
        ...(c.sw !== undefined ? { sampleWeight: c.sw } : {}),
      };
      expect(precision(yt, yp, options)).toBeCloseTo(ref.precision, 12);
      expect(recall(yt, yp, options)).toBeCloseTo(ref.recall, 12);
      expect(f1Score(yt, yp, options)).toBeCloseTo(ref.f1, 12);
      expect(fbetaScore(yt, yp, 2, options)).toBeCloseTo(ref.fbeta, 12);
      expect(jaccardScore(yt, yp, options)).toBeCloseTo(ref.jaccard, 12);
    });
  });

  it("uses the binary average by default for two string labels and asks for posLabel", () => {
    const yt = tensor(["a", "b", "a"]);
    const yp = tensor(["a", "b", "b"]);
    // scikit-learn: pos_label=1 is not a valid label. It should be one of ['a' 'b']
    expect(() => precision(yt, yp)).toThrow(InvalidParameterError);
    expect(() => precision(yt, yp)).toThrow(/posLabel=1 is not a valid label.*\['a' 'b'\]/);
    expect(precision(yt, yp, { posLabel: "b" })).toBeCloseTo(0.5, 12);
    expect(recall(yt, yp, { posLabel: "b" })).toBeCloseTo(1, 12);
  });

  it("rejects a posLabel that is not among two present labels", () => {
    // scikit-learn: pos_label=1 is not a valid label. It should be one of [0 2]
    const yt = tensor([0, 2, 0, 2]);
    const yp = tensor([0, 2, 2, 0]);
    expect(() => precision(yt, yp)).toThrow(
      "posLabel=1 is not a valid label. It should be one of [0 2]"
    );
    expect(precision(yt, yp, { posLabel: 2 })).toBeCloseTo(0.5, 12);
  });

  it("rejects the binary average for more than two labels", () => {
    // scikit-learn: Target is multiclass but average='binary'.
    expect(() => recall(tensor([0, 1, 2]), tensor([0, 1, 2]), "binary")).toThrow(
      /Target is multiclass but average='binary'/
    );
  });

  it("keeps the weighted default for more than two labels and ignores posLabel there", () => {
    const yt = tensor([0, 1, 2, 2]);
    const yp = tensor([0, 2, 2, 2]);
    expect(precision(yt, yp)).toBeCloseTo(precision(yt, yp, { average: "weighted" }), 12);
    expect(precision(yt, yp, { average: "macro", posLabel: 2 })).toBeCloseTo(
      precision(yt, yp, "macro"),
      12
    );
  });

  it("rejects a posLabel of the wrong type", () => {
    expect(() =>
      precision(tensor([0, 1]), tensor([0, 1]), { posLabel: true as unknown as number })
    ).toThrow(InvalidParameterError);
  });
});
