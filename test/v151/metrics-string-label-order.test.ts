import { describe, expect, it } from "vitest";
import { classificationReport, confusionMatrix } from "../../src/metrics";
import { tensor } from "../../src/ndarray";
import { CountVectorizer, LabelEncoder, OneHotEncoder, trainTestSplit } from "../../src/preprocess";

// scikit-learn sorts string labels by Unicode code point (NumPy compares UCS4 strings):
//   np.unique(["z", "￿", "\U0001F600"]) -> ['z', '￿', '\U0001F600']
// JavaScript's < compares UTF-16 code units and would put the astral character before U+FFFF.

const ASTRAL = "\u{1F600}";
const MAX_BMP = "￿";
const ORDER = ["z", MAX_BMP, ASTRAL];

describe("string labels are ordered by code point", () => {
  it("confusionMatrix lists the labels in sklearn order", () => {
    const yTrue = tensor([ASTRAL, MAX_BMP, "z", ASTRAL]);
    const yPred = tensor([ASTRAL, "z", "z", MAX_BMP]);
    // confusion_matrix(yTrue, yPred, labels=sorted) with the order z, U+FFFF, U+1F600:
    // rows true z: [1, 0, 0]; true U+FFFF: [1, 0, 0]; true U+1F600: [0, 1, 1]
    const cm = confusionMatrix(yTrue, yPred);
    expect(cm.toArray()).toEqual([
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 1],
    ]);
    const explicit = confusionMatrix(yTrue, yPred, { labels: ORDER });
    expect(explicit.toArray()).toEqual(cm.toArray());
  });

  it("classificationReport rows follow the same order", () => {
    const report = classificationReport(
      tensor([ASTRAL, MAX_BMP, "z", ASTRAL]),
      tensor([ASTRAL, "z", "z", MAX_BMP])
    );
    const rows = report
      .split("\n")
      .slice(3, 6)
      .map((line) => line.split(" ")[0]);
    expect(rows).toEqual(ORDER);
  });

  it("LabelEncoder and OneHotEncoder use the order", () => {
    const encoder = new LabelEncoder().fit([ASTRAL, MAX_BMP, "z"]);
    expect(encoder.classes?.toArray()).toEqual(ORDER);
    const ohe = new OneHotEncoder().fit([[ASTRAL], [MAX_BMP], ["z"]]);
    expect(ohe.categories?.[0]).toEqual(ORDER);
  });

  it("CountVectorizer sorts its vocabulary by code point", () => {
    const cv = new CountVectorizer({ tokenPattern: /\S+/gu, lowercase: false }).fitText([
      `${ASTRAL}${ASTRAL} ${MAX_BMP}${MAX_BMP} zz`,
    ]);
    expect(cv.getFeatureNames()).toEqual([`zz`, `${MAX_BMP}${MAX_BMP}`, `${ASTRAL}${ASTRAL}`]);
  });

  it("stratified splits keep every class in both parts", () => {
    const X = tensor(Array.from({ length: 12 }, (_, i) => [i]));
    const y = tensor(Array.from({ length: 12 }, (_, i) => ORDER[i % 3] as string));
    const parts = trainTestSplit(X, y, { testSize: 0.25, randomState: 3, stratify: y });
    const yTest = parts[3]?.toArray() as string[];
    expect(new Set(yTest)).toEqual(new Set(ORDER));
  });
});
