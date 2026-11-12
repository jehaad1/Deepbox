import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, NotFittedError, ShapeError } from "../../src/core";
import { LogisticRegression, Pipeline } from "../../src/ml";
import { tensor } from "../../src/ndarray";
import { CountVectorizer, HashingVectorizer, TfidfVectorizer } from "../../src/preprocess";

// Reference: scikit-learn CountVectorizer(ngram_range=(1, 2)).fit(["a bb cc", "bb cc dd"])
// builds the vocabulary ["bb", "bb cc", "cc", "cc dd", "dd"] (single letters are not tokens).

describe("CountVectorizer ngramRange", () => {
  it("does not loop over n-gram sizes larger than the document", () => {
    const cv = new CountVectorizer({ ngramRange: [1, 2 ** 40] });
    const start = performance.now();
    cv.fitText(["aa bb cc", "bb cc dd"]);
    expect(performance.now() - start).toBeLessThan(500);
    expect(cv.vocabulary.get("aa bb cc")).toBeDefined();
    expect(cv.vocabulary.get("bb cc dd")).toBeDefined();
    expect(cv.vocabulary.size).toBe(4 + 3 + 2);
  });

  it("keeps the bigram vocabulary of scikit-learn", () => {
    const cv = new CountVectorizer({ ngramRange: [1, 2] }).fitText(["a bb cc", "bb cc dd"]);
    expect([...cv.vocabulary.keys()].sort()).toEqual(["bb", "bb cc", "cc", "cc dd", "dd"]);
  });

  for (const range of [[0, 1], [2, 1], [1.5, 2], [1, Infinity], [1], "12"]) {
    it(`rejects ngramRange ${JSON.stringify(range)}`, () => {
      expect(
        () => new CountVectorizer({ ngramRange: range as unknown as [number, number] })
      ).toThrow(InvalidParameterError);
    });
  }
});

describe("vectorizers as Pipeline steps", () => {
  const docs = [
    "good great fine movie",
    "great excellent plot",
    "good fine acting",
    "excellent great cast",
    "fine good story",
    "bad awful movie",
    "awful terrible plot",
    "bad poor acting",
    "terrible awful cast",
    "poor bad story",
    "great good film",
    "awful bad film",
  ];
  const labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0];
  const asTensor = () => tensor(docs);
  const asColumn = () => tensor(docs.map((d) => [d]));

  const makers = {
    CountVectorizer: () => new CountVectorizer(),
    TfidfVectorizer: () => new TfidfVectorizer(),
    HashingVectorizer: () => new HashingVectorizer({ nFeatures: 64 }),
  };
  for (const name of Object.keys(makers) as (keyof typeof makers)[]) {
    describe(name, () => {
      const make = makers[name];

      it("fit, transform and fitTransform accept an array or a 1-D string tensor", () => {
        const fromTensor = make().fitTransform(asTensor());
        expect(make().fitTransform(asColumn()).toArray()).toEqual(fromTensor.toArray());
        const fromArray = make().fitTransform(docs);
        const viaText = make().fitTransformText(docs);
        expect(fromTensor.toArray()).toEqual(viaText.toArray());
        expect(fromArray.toArray()).toEqual(viaText.toArray());
        const v = make();
        v.fit(asTensor());
        expect(v.transform(docs.slice(0, 2)).shape[0]).toBe(2);
      });

      it("rejects input that is not a list of strings", () => {
        expect(() => make().fit(tensor([1, 2, 3]))).toThrow(DTypeError);
        expect(() => make().fit(tensor([["a b", "c d"]]))).toThrow(ShapeError);
      });

      it("clone returns an unfitted copy with the same parameters", () => {
        const v = make();
        v.fit(docs);
        const copy = v.clone();
        expect(copy).not.toBe(v);
        expect(copy.getParams()).toEqual(v.getParams());
        if (name !== "HashingVectorizer") {
          expect(() => copy.transform(docs)).toThrow(NotFittedError);
        }
      });

      it("works as the first step of a cloned Pipeline in a manual cross-validation", () => {
        // crossValScore itself rejects string tensors (src/ml/model_selection.ts, takeRows), so the
        // folds are cut here. Pipeline.clone is what crossValScore calls for every fold.
        const pipe = new Pipeline([
          ["vec", make()],
          ["clf", new LogisticRegression()],
        ]);
        const nFolds = 3;
        const accuracies: number[] = [];
        for (let fold = 0; fold < nFolds; fold++) {
          const trainIdx = docs.map((_d, i) => i).filter((i) => i % nFolds !== fold);
          const testIdx = docs.map((_d, i) => i).filter((i) => i % nFolds === fold);
          const cloned = pipe.clone();
          cloned.fit(
            tensor(trainIdx.map((i) => docs[i] as string)),
            tensor(trainIdx.map((i) => labels[i] as number))
          );
          const predicted = cloned
            .predict(tensor(testIdx.map((i) => docs[i] as string)))
            .toArray() as number[];
          const hits = predicted.filter((p, k) => p === labels[testIdx[k] as number]).length;
          accuracies.push(hits / testIdx.length);
        }
        expect(accuracies.length).toBe(nFolds);
        for (const accuracy of accuracies) {
          expect(accuracy).toBeGreaterThanOrEqual(0);
          expect(accuracy).toBeLessThanOrEqual(1);
        }
        // The original pipeline stays unfitted.
        expect(() => pipe.predict(asTensor())).toThrow(NotFittedError);
      });
    });
  }
});
