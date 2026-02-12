import { describe, expect, it } from "vitest";
import { ShapeError } from "../src/core/errors";
import { DBSCAN } from "../src/ml/clustering/DBSCAN";
import { KMeans } from "../src/ml/clustering/KMeans";
import { PCA } from "../src/ml/decomposition";
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from "../src/ml/ensemble/GradientBoosting";
import { Lasso } from "../src/ml/linear/Lasso";
import { LinearRegression } from "../src/ml/linear/LinearRegression";
import { LogisticRegression } from "../src/ml/linear/LogisticRegression";
import { Ridge } from "../src/ml/linear/Ridge";
import { LinearSVC } from "../src/ml/svm/SVM";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../src/ml/tree/DecisionTree";
import { RandomForestClassifier, RandomForestRegressor } from "../src/ml/tree/RandomForest";
import { tensor } from "../src/ndarray";

describe("ml error branch coverage", () => {
  it("validates KMeans configuration and input shape", () => {
    expect(() => new KMeans({ nClusters: 0 })).toThrow(/nClusters/);
    expect(() => new KMeans({ maxIter: 0 })).toThrow(/maxIter/);
    expect(() => new KMeans({ tol: -1 })).toThrow(/tol/);

    const km = new KMeans({ nClusters: 2 });
    expect(() => km.fit(tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => km.fit(tensor([[1, 2]]))).toThrow(/n_samples/);
  });

  it("validates tree models inputs and setParams", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([0, 1, 0]);
    expect(() => new DecisionTreeClassifier({ maxDepth: 0 })).toThrow(/maxDepth/);
    expect(() => new DecisionTreeRegressor({ minSamplesSplit: 1 })).toThrow(/minSamplesSplit/);
    const clf = new DecisionTreeClassifier();
    expect(() => clf.fit(tensor([1, 2]), tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => clf.fit(X, tensor([[1], [2]]))).toThrow(/1-dimensional/);
    expect(() => clf.fit(X, y)).toThrow(/same number of samples/);
    expect(() => clf.setParams({})).toThrow(/does not support setParams/);

    const reg = new DecisionTreeRegressor();
    expect(() => reg.fit(tensor([1, 2]), tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => reg.fit(X, tensor([[1], [2]]))).toThrow(/1-dimensional/);
    expect(() => reg.fit(X, y)).toThrow(/same number of samples/);
    expect(() => reg.setParams({})).toThrow(/does not support setParams/);
  });

  it("validates random forest inputs and setParams", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => new RandomForestClassifier({ nEstimators: 0 })).toThrow(/nEstimators/);
    expect(() => new RandomForestRegressor({ maxDepth: 0 })).toThrow(/maxDepth/);
    const clf = new RandomForestClassifier();
    expect(() => clf.fit(tensor([1, 2]), tensor([1, 2]))).toThrow(ShapeError);
    expect(() => clf.fit(X, tensor([[1], [2]]))).toThrow(ShapeError);
    clf.fit(X, tensor([0, 1]));
    expect(() => clf.predict(tensor([[1, 2, 3]]))).toThrow(/features/);
    expect(() => clf.setParams({})).toThrow(/does not support setParams/);

    const reg = new RandomForestRegressor();
    expect(() => reg.fit(tensor([1, 2]), tensor([1, 2]))).toThrow(ShapeError);
    expect(() => reg.fit(X, tensor([[1], [2]]))).toThrow(ShapeError);
    reg.fit(X, tensor([1, 2]));
    expect(() => reg.predict(tensor([[1, 2, 3]]))).toThrow(/features/);
    expect(() => reg.setParams({})).toThrow(/does not support setParams/);
  });

  it("validates gradient boosting configuration and inputs", () => {
    expect(() => new GradientBoostingRegressor({ nEstimators: 0 })).toThrow(/nEstimators/);
    expect(() => new GradientBoostingRegressor({ learningRate: 0 })).toThrow(/learningRate/);

    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([0, 1, 0]);
    const gbr = new GradientBoostingRegressor();
    expect(() => gbr.fit(tensor([1, 2]), tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => gbr.fit(X, tensor([[1], [2]]))).toThrow(/1-dimensional/);
    expect(() => gbr.fit(X, y)).toThrow(/same number of samples/);
    expect(() => gbr.setParams({})).toThrow(/does not support setParams/);

    const gbc = new GradientBoostingClassifier();
    const X3 = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    expect(() => gbc.fit(X3, tensor([0, 1, 2]))).toThrow(/exactly 2 classes/);
    expect(() => gbc.setParams({})).toThrow(/does not support setParams/);
  });

  it("validates linear models and SVM inputs", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([1, 2, 3]);
    expect(() => new LinearRegression().fit(tensor([1, 2]), tensor([1, 2]))).toThrow(
      /2-dimensional/
    );
    expect(() => new Ridge().fit(tensor([1, 2]), tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => new Lasso().fit(tensor([1, 2]), tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => new LogisticRegression().fit(tensor([1, 2]), tensor([1, 2]))).toThrow(
      /2-dimensional/
    );
    expect(() => new LinearSVC().fit(X, y)).toThrow(/same number of samples/);
  });

  it("validates clustering preprocessors", () => {
    expect(() => new DBSCAN().fit(tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => new PCA().fit(tensor([1, 2]))).toThrow(/2-dimensional/);
  });
});
