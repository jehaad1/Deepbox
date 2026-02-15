import { describe, expect, it } from "vitest";
import {
  DataLoader,
  loadBreastCancer,
  loadDigits,
  loadIris,
  makeBlobs,
  makeCircles,
  makeClassification,
  makeMoons,
  makeRegression,
} from "../src/datasets";
import { type Tensor, tensor } from "../src/ndarray";

describe("consumer API: datasets", () => {
  it("loadIris", () => {
    const iris = loadIris();
    expect(iris.data.shape[0]).toBe(150);
    expect(iris.data.shape[1]).toBe(4);
    expect(iris.target.size).toBe(150);
  });

  it("loadDigits", () => {
    const digits = loadDigits();
    expect(digits.data.shape[0]).toBe(1797);
    expect(digits.target.size).toBe(1797);
  });

  it("loadBreastCancer", () => {
    const bc = loadBreastCancer();
    expect(bc.data.shape[0]).toBe(569);
    expect(bc.target.size).toBe(569);
  });

  it("makeBlobs", () => {
    const [X, y] = makeBlobs({ nSamples: 100, nFeatures: 2, centers: 3, randomState: 42 });
    expect(X.shape[0]).toBe(100);
    expect(y.size).toBe(100);
  });

  it("makeClassification", () => {
    const [X, _y] = makeClassification({
      nSamples: 50,
      nFeatures: 4,
      nClasses: 2,
      randomState: 42,
    });
    expect(X.shape[0]).toBe(50);
  });

  it("makeRegression", () => {
    const [X, _y] = makeRegression({ nSamples: 50, nFeatures: 3, randomState: 42 });
    expect(X.shape[0]).toBe(50);
  });

  it("makeMoons", () => {
    const [X, _y] = makeMoons({ nSamples: 80, randomState: 42 });
    expect(X.shape[0]).toBe(80);
  });

  it("makeCircles", () => {
    const [X, _y] = makeCircles({ nSamples: 60, randomState: 42 });
    expect(X.shape[0]).toBe(60);
    expect(X.shape[1]).toBe(2);
  });

  it("DataLoader iteration", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 1, 0, 1]);
    const loader = new DataLoader(X, y, { batchSize: 2, shuffle: false });
    let batchCount = 0;
    for (const batch of loader as Iterable<[Tensor, Tensor]>) {
      expect(batch[0].shape[0]).toBe(2);
      expect(batch[1].size).toBe(2);
      batchCount++;
    }
    expect(batchCount).toBe(2);
  });
});
