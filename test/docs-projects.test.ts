/**
 * Tests derived from real Deepbox-Docs/src/data/projects.ts code snippets.
 * Covers testable, self-contained portions of each project.
 */
import { describe, expect, it } from "vitest";
import { loadDigits, loadIris } from "../src/datasets";
import { accuracy, confusionMatrix, f1Score, r2Score, silhouetteScore } from "../src/metrics";
import {
  DecisionTreeClassifier,
  GaussianNB,
  KMeans,
  KNeighborsClassifier,
  LogisticRegression,
  PCA,
  RandomForestClassifier,
  Ridge,
} from "../src/ml";
import { dot, tensor } from "../src/ndarray";
import { Dropout, Linear, ReLU, Sequential } from "../src/nn";
import { Figure } from "../src/plot";
import { KFold, StandardScaler, trainTestSplit } from "../src/preprocess";
import { clearSeed, normal, setSeed } from "../src/random";
import { corrcoef, mean, percentile, std } from "../src/stats";

describe("project 01: Financial Portfolio Risk Analysis", () => {
  it("generates correlated returns and computes portfolio stats", () => {
    setSeed(42);
    const nAssets = 5;
    const nDays = 252;
    const expectedReturns = [0.1, 0.08, 0.04, 0.07, 0.05];
    const volatilities = [0.18, 0.22, 0.06, 0.15, 0.2];

    const returns: number[][] = [];
    for (let i = 0; i < nAssets; i++) {
      const dailyReturn = (expectedReturns[i] ?? 0) / nDays;
      const dailyVol = (volatilities[i] ?? 0) / Math.sqrt(nDays);
      const assetReturns = normal(dailyReturn, dailyVol, [nDays]);
      returns.push(Array.from(assetReturns.data as Float32Array));
    }
    const returnMatrix = tensor(returns);
    expect(returnMatrix.shape).toEqual([5, 252]);

    const weights = tensor([0.25, 0.2, 0.3, 0.15, 0.1]);
    const portfolioReturns = dot(weights, returnMatrix);
    expect(portfolioReturns.shape).toEqual([252]);

    const portMean = mean(portfolioReturns);
    const portStd = std(portfolioReturns);
    expect(portMean.size).toBe(1);
    expect(portStd.size).toBe(1);

    const var95 = percentile(portfolioReturns, 5);
    expect(var95.size).toBe(1);

    const corrMatrix = corrcoef(returnMatrix);
    expect(corrMatrix.shape[0]).toBe(corrMatrix.shape[1]);

    clearSeed();
  });

  it("generates efficient frontier visualization", () => {
    const fig = new Figure({ width: 800, height: 600 });
    const ax = fig.addAxes();
    ax.scatter(tensor([0.1, 0.15, 0.2]), tensor([0.05, 0.08, 0.12]));
    ax.setTitle("Efficient Frontier");
    ax.setXLabel("Risk (Volatility)");
    ax.setYLabel("Expected Return");
    const svg = fig.renderSVG();
    expect(svg.svg.length).toBeGreaterThan(0);
  });
});

describe("project 02: Neural Network Image Classifier", () => {
  it("loads digits and builds MLP model", () => {
    const digits = loadDigits();
    expect(digits.data.shape).toEqual([1797, 64]);

    const [X_tr, X_te, ,] = trainTestSplit(digits.data, digits.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(X_tr);
    const X_test = scaler.transform(X_te);

    const model = new Sequential(
      new Linear(64, 64),
      new ReLU(),
      new Dropout(0.2),
      new Linear(64, 32),
      new ReLU(),
      new Dropout(0.2),
      new Linear(32, 10)
    );
    expect([...model.parameters()].length).toBe(6);

    // Forward pass with plain tensor (no training)
    model.eval();
    const out = model.forward(X_test);
    expect(out.shape[0]).toBe(X_te.shape[0]);
    expect(out.shape[1]).toBe(10);
  });

  it("plots training curves", () => {
    const losses = [2.3, 1.5, 0.9, 0.5, 0.3];
    const fig = new Figure({ width: 800, height: 400 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1, 2, 3, 4]), tensor(losses));
    ax.setTitle("Training Loss");
    ax.setXLabel("Epoch");
    ax.setYLabel("Cross-Entropy Loss");
    const svg = fig.renderSVG();
    expect(svg.svg.length).toBeGreaterThan(0);
  });
});

describe("project 03: Customer Churn Prediction", () => {
  it("trains and compares 4 classifiers on Iris (as proxy)", () => {
    const iris = loadIris();
    const [X_tr, X_te, y_tr, y_te] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(X_tr);
    const X_train = scaler.transform(X_tr);
    const X_test = scaler.transform(X_te);

    const models = [
      { name: "Logistic Regression", model: new LogisticRegression() },
      {
        name: "Decision Tree",
        model: new DecisionTreeClassifier({ maxDepth: 8 }),
      },
      {
        name: "Random Forest",
        model: new RandomForestClassifier({ nEstimators: 50 }),
      },
      { name: "KNN (k=5)", model: new KNeighborsClassifier({ nNeighbors: 5 }) },
      { name: "Gaussian NB", model: new GaussianNB() },
    ];

    const results = [];
    for (const { name, model } of models) {
      model.fit(X_train, y_tr);
      const preds = model.predict(X_test);
      results.push({
        name,
        acc: accuracy(y_te, preds),
        f1: f1Score(y_te, preds, "macro") as number,
      });
    }

    results.sort((a, b) => b.f1 - a.f1);
    expect(results.length).toBe(5);
    for (const r of results) {
      expect(r.acc).toBeGreaterThan(0.8);
      expect(r.f1).toBeGreaterThan(0.8);
    }
  });

  it("KFold cross-validation on Iris", () => {
    const iris = loadIris();
    const kf = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });
    let foldCount = 0;
    for (const { trainIndex, testIndex } of kf.split(iris.data)) {
      expect(trainIndex.length + testIndex.length).toBe(150);
      foldCount++;
    }
    expect(foldCount).toBe(5);
  });
});

describe("project 04: Stock Price Forecasting", () => {
  it("generates synthetic prices and computes stats", () => {
    const nDays = 500;
    let price = 100;
    const prices: number[] = [price];
    const returns: number[] = [];

    setSeed(42);
    for (let i = 1; i < nDays; i++) {
      const dailyReturn = 0.0003 + 0.02 * (Math.random() * 2 - 1);
      price *= 1 + dailyReturn;
      prices.push(price);
      returns.push(dailyReturn);
    }
    clearSeed();

    expect(prices.length).toBe(nDays);
    expect(Math.min(...prices)).toBeGreaterThan(0);

    const retTensor = tensor(returns);
    const retMean = mean(retTensor);
    const retStd = std(retTensor);
    expect(retMean.size).toBe(1);
    expect(retStd.size).toBe(1);
  });

  it("Ridge regression on simple data", () => {
    const X = tensor([
      [1, 0.5],
      [2, 1.0],
      [3, 1.5],
      [4, 2.0],
      [5, 2.5],
      [6, 3.0],
      [7, 3.5],
      [8, 4.0],
    ]);
    const y = tensor([1.1, 2.0, 3.1, 4.0, 5.1, 5.9, 7.1, 8.0]);
    const ridge = new Ridge({ alpha: 0.1 });
    ridge.fit(X, y);
    const preds = ridge.predict(X);
    expect(r2Score(y, preds)).toBeGreaterThan(0.99);
  });
});

describe("project 05: Movie Recommendation Engine", () => {
  it("KMeans clustering + PCA + silhouette", () => {
    const iris = loadIris();
    const kmeans = new KMeans({ nClusters: 3, randomState: 42 });
    kmeans.fit(iris.data);
    const labels = kmeans.predict(iris.data);
    const silScore = silhouetteScore(iris.data, labels);
    expect(silScore).toBeGreaterThan(0.3);

    const pca = new PCA({ nComponents: 2 });
    pca.fit(iris.data);
    const projected = pca.transform(iris.data);
    expect(projected.shape).toEqual([150, 2]);
    expect(pca.explainedVarianceRatio.size).toBe(2);
  });

  it("scatter plot of clusters", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.scatter(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 3, 5, 4]), {
      color: "#1f77b4",
    });
    ax.setTitle("User Clusters (PCA)");
    const svg = fig.renderSVG();
    expect(svg.svg.length).toBeGreaterThan(0);
  });
});

describe("project 06: Sentiment Analysis", () => {
  it("LogReg vs GaussianNB on Iris (as proxy for text classification)", () => {
    const iris = loadIris();
    const [X_tr, X_te, y_tr, y_te] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(X_tr);
    const X_train = scaler.transform(X_tr);
    const X_test = scaler.transform(X_te);

    const lr = new LogisticRegression();
    lr.fit(X_train, y_tr);
    const lrPreds = lr.predict(X_test);
    expect(accuracy(y_te, lrPreds)).toBeGreaterThan(0.9);
    expect(f1Score(y_te, lrPreds, "macro")).toBeGreaterThan(0.9);

    const nb = new GaussianNB();
    nb.fit(X_train, y_tr);
    const nbPreds = nb.predict(X_test);
    expect(accuracy(y_te, nbPreds)).toBeGreaterThan(0.8);

    expect(confusionMatrix(y_te, lrPreds).shape).toEqual([3, 3]);
  });
});
