import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  GroupKFold,
  KFold,
  LabelBinarizer,
  LabelEncoder,
  LeaveOneOut,
  LeavePOut,
  MaxAbsScaler,
  MinMaxScaler,
  MultiLabelBinarizer,
  Normalizer,
  OneHotEncoder,
  OrdinalEncoder,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
  StratifiedKFold,
  trainTestSplit,
} from "../src/preprocess";

describe("consumer API: preprocess", () => {
  const X = tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ]);

  describe("scalers", () => {
    it("StandardScaler fit/transform/inverse/fitTransform", () => {
      const s = new StandardScaler();
      s.fit(X);
      const scaled = s.transform(X);
      expect(scaled.shape).toEqual([4, 2]);
      expect(s.inverseTransform(scaled).shape[0]).toBe(4);
      expect(s.fitTransform(X).shape[0]).toBe(4);
    });

    it("MinMaxScaler fit/transform/inverse", () => {
      const s = new MinMaxScaler();
      s.fit(X);
      const scaled = s.transform(X);
      expect(scaled.shape[0]).toBe(4);
      expect(s.inverseTransform(scaled).shape[0]).toBe(4);
    });

    it("RobustScaler fit/transform/inverse", () => {
      const s = new RobustScaler();
      s.fit(X);
      expect(s.transform(X).shape[0]).toBe(4);
      expect(s.inverseTransform(s.transform(X)).shape[0]).toBe(4);
    });

    it("MaxAbsScaler fit/transform/inverse", () => {
      const s = new MaxAbsScaler();
      s.fit(X);
      expect(s.transform(X).shape[0]).toBe(4);
      expect(s.inverseTransform(s.transform(X)).shape[0]).toBe(4);
    });

    it("Normalizer transform", () => {
      const n = new Normalizer();
      expect(n.transform(X).shape[0]).toBe(4);
    });

    it("PowerTransformer fit/transform", () => {
      const data = tensor([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
      ]);
      const pt = new PowerTransformer();
      pt.fit(data);
      expect(pt.transform(data).shape[0]).toBe(4);
    });

    it("QuantileTransformer fit/transform", () => {
      const data = tensor([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50],
      ]);
      const qt = new QuantileTransformer();
      qt.fit(data);
      expect(qt.transform(data).shape[0]).toBe(5);
    });
  });

  describe("encoders", () => {
    it("LabelEncoder with tensor input", () => {
      const le = new LabelEncoder();
      le.fit(tensor(["cat", "dog", "bird", "cat", "dog"]));
      const encoded = le.transform(tensor(["cat", "bird"]));
      expect(encoded.size).toBe(2);
      expect(le.inverseTransform(encoded).size).toBe(2);
    });

    it("LabelEncoder with plain array input", () => {
      const le = new LabelEncoder();
      le.fit(["cat", "dog", "bird"]);
      const encoded = le.transform(["cat", "bird"]);
      expect(encoded.size).toBe(2);
    });

    it("OneHotEncoder", () => {
      const ohe = new OneHotEncoder();
      ohe.fit(tensor([["red"], ["green"], ["blue"], ["red"]]));
      const encoded = ohe.transform(tensor([["red"], ["blue"]]));
      expect(encoded.shape[0]).toBe(2);
    });

    it("OrdinalEncoder", () => {
      const oe = new OrdinalEncoder();
      oe.fit(tensor([["low"], ["med"], ["high"], ["low"]]));
      const encoded = oe.transform(tensor([["low"], ["high"]]));
      expect(encoded.shape[0]).toBe(2);
    });

    it("LabelBinarizer", () => {
      const lb = new LabelBinarizer();
      lb.fit(tensor([0, 1, 2, 0, 1]));
      const encoded = lb.transform(tensor([0, 2]));
      expect(encoded.shape[0]).toBe(2);
      expect(lb.inverseTransform(encoded).size).toBe(2);
    });

    it("MultiLabelBinarizer", () => {
      const mlb = new MultiLabelBinarizer();
      mlb.fit([
        [0, 1],
        [1, 2],
        [0, 2],
      ]);
      const encoded = mlb.transform([[0, 1], [2]]);
      expect(encoded.shape[0]).toBe(2);
    });
  });

  describe("splitting", () => {
    const Xdata = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const ydata = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    it("trainTestSplit", () => {
      const [XTr, XTe, yTr, yTe] = trainTestSplit(Xdata, ydata, {
        testSize: 0.3,
        randomState: 42,
      });
      expect(XTr.shape[0]).toBe(7);
      expect(XTe.shape[0]).toBe(3);
      expect(yTr.size).toBe(7);
      expect(yTe.size).toBe(3);
    });

    it("KFold returns SplitResult objects", () => {
      const kf = new KFold({ nSplits: 3 });
      const splits = kf.split(Xdata);
      expect(splits.length).toBe(3);
      for (const s of splits) {
        expect(s).toHaveProperty("trainIndex");
        expect(s).toHaveProperty("testIndex");
      }
    });

    it("StratifiedKFold", () => {
      const skf = new StratifiedKFold({ nSplits: 2 });
      const splits = skf.split(Xdata, ydata);
      expect(splits.length).toBe(2);
    });

    it("GroupKFold", () => {
      const groups = tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
      const gkf = new GroupKFold({ nSplits: 2 });
      const splits = gkf.split(Xdata, ydata, groups);
      expect(splits.length).toBe(2);
    });

    it("LeaveOneOut", () => {
      const loo = new LeaveOneOut();
      const data = tensor([[1], [2], [3]]);
      expect(loo.split(data).length).toBe(3);
    });

    it("LeavePOut", () => {
      const lpo = new LeavePOut(2);
      const data = tensor([[1], [2], [3]]);
      expect(lpo.split(data).length).toBe(3);
    });
  });
});
