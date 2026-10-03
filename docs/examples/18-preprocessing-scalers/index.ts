/**
 * Example 18: Preprocessing: Scalers
 *
 * Feature scaling puts columns on comparable ranges, which matters for
 * distance-based and gradient-based models. This example applies seven
 * scalers to the same small table that has one large outlier in each of
 * the first and last columns.
 *
 * Fit a scaler on the training data only, then reuse it with transform().
 */

import { type Tensor, tensor } from "deepbox/ndarray";
import {
  MaxAbsScaler,
  MinMaxScaler,
  Normalizer,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
} from "deepbox/preprocess";

console.log("=== Preprocessing: Scalers ===\n");

// First three rows, to keep the printed output short.
const first3 = (t: Tensor) => t.slice({ start: 0, end: 3 });

// Sample data with different scales
const X = tensor([
  [1, 100, 0.01],
  [2, 200, 0.02],
  [3, 300, 0.03],
  [4, 400, 0.04],
  [5, 500, 0.05],
  [100, 50, 0.5],
]);

// ---------------------------------------------------------------------------
// Part 1: StandardScaler: zero mean, unit variance
// ---------------------------------------------------------------------------
console.log("--- Part 1: StandardScaler ---");

const ss = new StandardScaler();
ss.fit(X);
const XStd = ss.transform(X);
console.log("Scaled (first 3 rows):\n", first3(XStd).toString());
console.log("Column means after scaling:", XStd.mean(0).toString());

const XInv = ss.inverseTransform(XStd);
console.log("Inverse transform recovers the first row:", first3(XInv).slice(0).toString());

// ---------------------------------------------------------------------------
// Part 2: MinMaxScaler: scale to [0, 1]
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: MinMaxScaler ---");

const mms = new MinMaxScaler();
mms.fit(X);
const XMinMax = mms.transform(X);
console.log("Scaled (first 3 rows):\n", first3(XMinMax).toString());

// ---------------------------------------------------------------------------
// Part 3: RobustScaler: uses median and IQR (robust to outliers)
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: RobustScaler ---");

const rs = new RobustScaler();
rs.fit(X);
const XRobust = rs.transform(X);
console.log("Scaled (first 3 rows):\n", first3(XRobust).toString());

// ---------------------------------------------------------------------------
// Part 4: MaxAbsScaler: scale by maximum absolute value
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: MaxAbsScaler ---");

const mas = new MaxAbsScaler();
mas.fit(X);
const XMaxAbs = mas.transform(X);
console.log("Scaled (first 3 rows):\n", first3(XMaxAbs).toString());

// ---------------------------------------------------------------------------
// Part 5: Normalizer: normalize each sample (row) to unit norm
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: Normalizer ---");

const norm = new Normalizer();
const XNorm = norm.transform(X);
console.log("Normalized (first 3 rows):\n", first3(XNorm).toString());
console.log("Row norms:", XNorm.square().sum(1).sqrt().toString());

// ---------------------------------------------------------------------------
// Part 6: PowerTransformer: Gaussian-like transformation
// ---------------------------------------------------------------------------
console.log("\n--- Part 6: PowerTransformer ---");

const pt = new PowerTransformer();
pt.fit(X);
const XPower = pt.transform(X);
console.log("Transformed (first 3 rows):\n", first3(XPower).toString());

// Deepbox's PowerTransformer does not standardize by default (scikit-learn does).
// Pass { standardize: true } to get zero mean and unit variance per column.
const ptStd = new PowerTransformer({ standardize: true });
console.log("Column means with standardize: true:", ptStd.fitTransform(X).mean(0).toString());

// ---------------------------------------------------------------------------
// Part 7: QuantileTransformer: map to uniform or normal distribution
// ---------------------------------------------------------------------------
console.log("\n--- Part 7: QuantileTransformer ---");

const qt = new QuantileTransformer();
qt.fit(X);
const XQuantile = qt.transform(X);
console.log("Transformed (first 3 rows):\n", first3(XQuantile).toString());

console.log("\n=== Preprocessing: Scalers Complete ===");
