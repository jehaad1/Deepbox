import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { classificationReport } from "../../src/metrics";
import { tensor } from "../../src/ndarray";

// Expected values come from scikit-learn 1.9.1 classification_report(output_dict=True).

type Row = { precision: number; recall: number; "f1-score": number; support: number };
type Ref = Record<string, Row | number>;
type Case = {
  yt: (number | string)[];
  yp: (number | string)[];
  labels?: number[];
  names?: string[];
  zd?: number;
  sw?: number[];
};

const REFERENCE: Record<string, { case: Case; ref: Ref }> = {
  multi: {
    case: { yt: [0, 1, 2, 2, 1, 0, 2, 1, 0, 2], yp: [0, 2, 2, 2, 1, 0, 1, 1, 0, 0] },
    ref: {
      "0": { precision: 0.75, recall: 1.0, "f1-score": 0.8571428571428571, support: 3.0 },
      "1": {
        precision: 0.6666666666666666,
        recall: 0.6666666666666666,
        "f1-score": 0.6666666666666666,
        support: 3.0,
      },
      "2": {
        precision: 0.6666666666666666,
        recall: 0.5,
        "f1-score": 0.5714285714285714,
        support: 4.0,
      },
      accuracy: 0.7,
      "macro avg": {
        precision: 0.6944444444444443,
        recall: 0.7222222222222222,
        "f1-score": 0.6984126984126983,
        support: 10.0,
      },
      "weighted avg": {
        precision: 0.6916666666666667,
        recall: 0.7,
        "f1-score": 0.6857142857142857,
        support: 10.0,
      },
    },
  },
  strings: {
    case: {
      yt: ["cat", "dog", "bird", "bird", "dog", "cat", "bird", "dog", "cat", "bird"],
      yp: ["cat", "bird", "bird", "bird", "dog", "cat", "dog", "dog", "cat", "cat"],
    },
    ref: {
      bird: {
        precision: 0.6666666666666666,
        recall: 0.5,
        "f1-score": 0.5714285714285714,
        support: 4.0,
      },
      cat: { precision: 0.75, recall: 1.0, "f1-score": 0.8571428571428571, support: 3.0 },
      dog: {
        precision: 0.6666666666666666,
        recall: 0.6666666666666666,
        "f1-score": 0.6666666666666666,
        support: 3.0,
      },
      accuracy: 0.7,
      "macro avg": {
        precision: 0.6944444444444443,
        recall: 0.7222222222222222,
        "f1-score": 0.6984126984126983,
        support: 10.0,
      },
      "weighted avg": {
        precision: 0.6916666666666667,
        recall: 0.7,
        "f1-score": 0.6857142857142857,
        support: 10.0,
      },
    },
  },
  signed: {
    case: { yt: [-1, 1, 1, -1, 1, -1, 1], yp: [-1, 1, -1, -1, 1, 1, 1] },
    ref: {
      "-1": {
        precision: 0.6666666666666666,
        recall: 0.6666666666666666,
        "f1-score": 0.6666666666666666,
        support: 3.0,
      },
      "1": { precision: 0.75, recall: 0.75, "f1-score": 0.75, support: 4.0 },
      accuracy: 0.7142857142857143,
      "macro avg": {
        precision: 0.7083333333333333,
        recall: 0.7083333333333333,
        "f1-score": 0.7083333333333333,
        support: 7.0,
      },
      "weighted avg": {
        precision: 0.7142857142857143,
        recall: 0.7142857142857143,
        "f1-score": 0.7142857142857143,
        support: 7.0,
      },
    },
  },
  subset: {
    case: {
      yt: [0, 1, 2, 2, 1, 0, 2, 1, 0, 2],
      yp: [0, 2, 2, 2, 1, 0, 1, 1, 0, 0],
      labels: [0, 2],
    },
    ref: {
      "0": { precision: 0.75, recall: 1.0, "f1-score": 0.8571428571428571, support: 3.0 },
      "2": {
        precision: 0.6666666666666666,
        recall: 0.5,
        "f1-score": 0.5714285714285714,
        support: 4.0,
      },
      "micro avg": {
        precision: 0.7142857142857143,
        recall: 0.7142857142857143,
        "f1-score": 0.7142857142857143,
        support: 7.0,
      },
      "macro avg": {
        precision: 0.7083333333333333,
        recall: 0.75,
        "f1-score": 0.7142857142857142,
        support: 7.0,
      },
      "weighted avg": {
        precision: 0.7023809523809523,
        recall: 0.7142857142857143,
        "f1-score": 0.6938775510204082,
        support: 7.0,
      },
    },
  },
  order: {
    case: {
      yt: [0, 1, 2, 2, 1, 0, 2, 1, 0, 2],
      yp: [0, 2, 2, 2, 1, 0, 1, 1, 0, 0],
      labels: [2, 0, 1],
    },
    ref: {
      "2": {
        precision: 0.6666666666666666,
        recall: 0.5,
        "f1-score": 0.5714285714285714,
        support: 4.0,
      },
      "0": { precision: 0.75, recall: 1.0, "f1-score": 0.8571428571428571, support: 3.0 },
      "1": {
        precision: 0.6666666666666666,
        recall: 0.6666666666666666,
        "f1-score": 0.6666666666666666,
        support: 3.0,
      },
      accuracy: 0.7,
      "macro avg": {
        precision: 0.6944444444444443,
        recall: 0.7222222222222222,
        "f1-score": 0.6984126984126983,
        support: 10.0,
      },
      "weighted avg": {
        precision: 0.6916666666666667,
        recall: 0.7,
        "f1-score": 0.6857142857142857,
        support: 10.0,
      },
    },
  },
  zero1: {
    case: { yt: [0, 1, 2, 2, 1, 0], yp: [0, 1, 1, 1, 1, 0], labels: [0, 1, 2, 3], zd: 1 },
    ref: {
      "0": { precision: 1.0, recall: 1.0, "f1-score": 1.0, support: 2.0 },
      "1": { precision: 0.5, recall: 1.0, "f1-score": 0.6666666666666666, support: 2.0 },
      "2": { precision: 1.0, recall: 0.0, "f1-score": 0.0, support: 2.0 },
      "3": { precision: 1.0, recall: 1.0, "f1-score": 1.0, support: 0.0 },
      accuracy: 0.6666666666666666,
      "macro avg": { precision: 0.875, recall: 0.75, "f1-score": 0.6666666666666666, support: 6.0 },
      "weighted avg": {
        precision: 0.8333333333333334,
        recall: 0.6666666666666666,
        "f1-score": 0.5555555555555555,
        support: 6.0,
      },
    },
  },
  weighted: {
    case: {
      yt: [0, 1, 2, 2, 1, 0, 2, 1, 0, 2],
      yp: [0, 2, 2, 2, 1, 0, 1, 1, 0, 0],
      sw: [1, 2, 3, 1, 2, 1, 0.5, 1, 2, 1],
    },
    ref: {
      "0": { precision: 0.8, recall: 1.0, "f1-score": 0.8888888888888888, support: 4.0 },
      "1": {
        precision: 0.8571428571428571,
        recall: 0.6,
        "f1-score": 0.7058823529411765,
        support: 5.0,
      },
      "2": {
        precision: 0.6666666666666666,
        recall: 0.7272727272727273,
        "f1-score": 0.6956521739130435,
        support: 5.5,
      },
      accuracy: 0.7586206896551724,
      "macro avg": {
        precision: 0.7746031746031745,
        recall: 0.7757575757575758,
        "f1-score": 0.7634744719143697,
        support: 14.5,
      },
      "weighted avg": {
        precision: 0.7691297208538588,
        recall: 0.7586206896551724,
        "f1-score": 0.7524865018471156,
        support: 14.5,
      },
    },
  },
  binary01: {
    case: { yt: [0, 1, 1, 0, 1], yp: [0, 1, 0, 0, 1] },
    ref: {
      "0": { precision: 0.6666666666666666, recall: 1.0, "f1-score": 0.8, support: 2.0 },
      "1": { precision: 1.0, recall: 0.6666666666666666, "f1-score": 0.8, support: 3.0 },
      accuracy: 0.8,
      "macro avg": {
        precision: 0.8333333333333333,
        recall: 0.8333333333333333,
        "f1-score": 0.8,
        support: 5.0,
      },
      "weighted avg": { precision: 0.8666666666666666, recall: 0.8, "f1-score": 0.8, support: 5.0 },
    },
  },
  names: {
    case: {
      yt: [0, 1, 2, 2, 1, 0, 2, 1, 0, 2],
      yp: [0, 2, 2, 2, 1, 0, 1, 1, 0, 0],
      names: ["a", "bb", "ccc"],
    },
    ref: {
      a: { precision: 0.75, recall: 1.0, "f1-score": 0.8571428571428571, support: 3.0 },
      bb: {
        precision: 0.6666666666666666,
        recall: 0.6666666666666666,
        "f1-score": 0.6666666666666666,
        support: 3.0,
      },
      ccc: {
        precision: 0.6666666666666666,
        recall: 0.5,
        "f1-score": 0.5714285714285714,
        support: 4.0,
      },
      accuracy: 0.7,
      "macro avg": {
        precision: 0.6944444444444443,
        recall: 0.7222222222222222,
        "f1-score": 0.6984126984126983,
        support: 10.0,
      },
      "weighted avg": {
        precision: 0.6916666666666667,
        recall: 0.7,
        "f1-score": 0.6857142857142857,
        support: 10.0,
      },
    },
  },
  onlyone: {
    case: { yt: [3, 3, 3], yp: [3, 3, 3] },
    ref: {
      "3": { precision: 1.0, recall: 1.0, "f1-score": 1.0, support: 3.0 },
      accuracy: 1.0,
      "macro avg": { precision: 1.0, recall: 1.0, "f1-score": 1.0, support: 3.0 },
      "weighted avg": { precision: 1.0, recall: 1.0, "f1-score": 1.0, support: 3.0 },
    },
  },
};

const SUMMARY_ROWS: Record<string, string> = {
  accuracy: "Accuracy",
  "micro avg": "Micro Avg",
  "macro avg": "Macro Avg",
  "weighted avg": "Weighted Avg",
};

function parseReport(text: string): Record<string, number[]> {
  const rows: Record<string, number[]> = {};
  const lines = text
    .split("\n")
    .slice(2)
    .filter((l) => l.trim() !== "" && !/^-{5,}$/.test(l));
  for (const line of lines) {
    const match = /^(.*?)\s+(-?[\d.]+|NaN)((?:\s+(?:-?[\d.]+|NaN))*)$/.exec(line.trimEnd());
    if (match === null) throw new Error(`cannot parse report line: ${line}`);
    const numbers = [match[2] as string, ...(match[3] ?? "").trim().split(/\s+/).filter(Boolean)];
    rows[(match[1] as string).trim()] = numbers.map(Number);
  }
  return rows;
}

function toTensor(values: (number | string)[]) {
  return typeof values[0] === "string" ? tensor(values as string[]) : tensor(values as number[]);
}

describe("classificationReport matches scikit-learn", () => {
  for (const [name, { case: c, ref }] of Object.entries(REFERENCE)) {
    it(name, () => {
      const options = {
        digits: 8,
        ...(c.labels !== undefined ? { labels: c.labels } : {}),
        ...(c.names !== undefined ? { targetNames: c.names } : {}),
        ...(c.zd !== undefined ? { zeroDivision: c.zd } : {}),
        ...(c.sw !== undefined ? { sampleWeight: c.sw } : {}),
      };
      const rows = parseReport(classificationReport(toTensor(c.yt), toTensor(c.yp), options));
      for (const [key, value] of Object.entries(ref)) {
        const label = SUMMARY_ROWS[key] ?? key;
        const got = rows[label];
        expect(got, `row ${label}`).toBeDefined();
        if (typeof value === "number") {
          // The accuracy row prints the score in the F1 column, then the support.
          expect((got as number[])[0]).toBeCloseTo(value, 6);
        } else {
          expect((got as number[])[0]).toBeCloseTo(value.precision, 6);
          expect((got as number[])[1]).toBeCloseTo(value.recall, 6);
          expect((got as number[])[2]).toBeCloseTo(value["f1-score"], 6);
          expect((got as number[])[3]).toBeCloseTo(value.support, 6);
        }
      }
    });
  }

  it("keeps the layout for binary labels", () => {
    const text = classificationReport(tensor([0, 1, 1, 0, 1]), tensor([0, 1, 0, 0, 1]));
    expect(text.split("\n")[0]).toBe("Classification Report:");
    expect(text).toContain("Weighted Avg");
    expect(text).toContain("0.6667");
  });

  it("rejects target names whose count differs from the number of classes", () => {
    expect(() =>
      classificationReport(tensor([0, 1, 2]), tensor([0, 1, 2]), { targetNames: ["a", "b"] })
    ).toThrow(InvalidParameterError);
  });

  it("rejects an invalid digits value", () => {
    expect(() => classificationReport(tensor([0, 1]), tensor([0, 1]), { digits: -1 })).toThrow(
      InvalidParameterError
    );
  });
});
