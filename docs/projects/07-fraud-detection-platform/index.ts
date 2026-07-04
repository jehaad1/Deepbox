/**
 * Fraud Detection Platform
 *
 * A production-style risk scoring workflow that combines supervised fraud
 * classification, probability calibration, anomaly detection, feature
 * inspection, and operational reporting in one Deepbox project.
 */

import { mkdir } from "node:fs/promises";
import { DataFrame, to_datetime } from "deepbox/dataframe";
import { accuracy, f1Score, precision, recall } from "deepbox/metrics";
import {
  CalibratedClassifierCV,
  calibrationCurve,
  IsolationForest,
  LocalOutlierFactor,
  LogisticRegression,
  OneClassSVM,
  permutationImportance,
} from "deepbox/ml";
import { type Tensor, tensor } from "deepbox/ndarray";
import { figure, plotCalibrationCurve, plotFeatureImportance, saveFig } from "deepbox/plot";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

const OUTPUT_DIR = "docs/projects/07-fraud-detection-platform/output";
const RANDOM_SEED = 42;
const NUM_TRANSACTIONS = 1_500;

type TransactionRecord = {
  readonly timestamp: string;
  readonly merchantCategory: string;
  readonly channel: string;
  readonly country: string;
  readonly amount: number;
  readonly hourOfDay: number;
  readonly velocity30m: number;
  readonly geoRisk: number;
  readonly deviceRisk: number;
  readonly merchantRisk: number;
  readonly chargebackHistory: number;
  readonly accountAgeDays: number;
  readonly isFraud: number;
};

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function randomNormal(rng: () => number, mean: number, std: number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

function generateTransactions(n: number, seed: number): TransactionRecord[] {
  const rng = createRng(seed);
  const merchantCategories = ["electronics", "travel", "luxury", "groceries", "gaming"];
  const channels = ["web", "mobile", "call-center"];
  const countries = ["SA", "AE", "EG", "GB", "US"];
  const baseDate = new Date("2026-02-01T00:00:00.000Z").getTime();

  const records: TransactionRecord[] = [];

  for (let i = 0; i < n; i++) {
    const merchantCategory =
      merchantCategories[Math.floor(rng() * merchantCategories.length)] ?? "groceries";
    const channel = channels[Math.floor(rng() * channels.length)] ?? "web";
    const country = countries[Math.floor(rng() * countries.length)] ?? "SA";
    const hourOfDay = Math.floor(rng() * 24);

    let amount = Math.max(8, randomNormal(rng, 180, 120));
    let velocity30m = Math.max(1, Math.round(randomNormal(rng, 2.5, 1.8)));
    let geoRisk = Math.max(0, Math.min(1, rng() * 0.7));
    let deviceRisk = Math.max(0, Math.min(1, rng() * 0.6));
    const merchantRisk =
      merchantCategory === "travel" || merchantCategory === "luxury"
        ? 0.45 + rng() * 0.45
        : 0.05 + rng() * 0.45;
    const chargebackHistory = Math.max(0, Math.round(randomNormal(rng, 0.5, 1.2)));
    let accountAgeDays = Math.max(2, Math.round(randomNormal(rng, 420, 260)));

    let fraudScore = 0.015;
    fraudScore += amount > 650 ? 0.18 : 0;
    fraudScore += hourOfDay < 5 ? 0.12 : 0;
    fraudScore += velocity30m > 5 ? 0.18 : 0;
    fraudScore += geoRisk > 0.65 ? 0.16 : 0;
    fraudScore += deviceRisk > 0.55 ? 0.12 : 0;
    fraudScore += merchantRisk > 0.65 ? 0.09 : 0;
    fraudScore += chargebackHistory > 1 ? 0.14 : 0;
    fraudScore += accountAgeDays < 45 ? 0.2 : 0;

    if (rng() < 0.08) {
      amount *= 2.5;
      velocity30m += 4;
      geoRisk = Math.min(1, geoRisk + 0.35);
      deviceRisk = Math.min(1, deviceRisk + 0.35);
      accountAgeDays = Math.max(2, Math.round(accountAgeDays * 0.12));
      fraudScore += 0.28;
    }

    const isFraud = rng() < Math.min(fraudScore, 0.94) ? 1 : 0;
    const timestamp = new Date(
      baseDate + i * 45 * 60 * 1000 + hourOfDay * 60 * 60 * 1000
    ).toISOString();

    records.push({
      timestamp,
      merchantCategory,
      channel,
      country,
      amount: Number(amount.toFixed(2)),
      hourOfDay,
      velocity30m,
      geoRisk: Number(geoRisk.toFixed(3)),
      deviceRisk: Number(deviceRisk.toFixed(3)),
      merchantRisk: Number(merchantRisk.toFixed(3)),
      chargebackHistory,
      accountAgeDays,
      isFraud,
    });
  }

  return records;
}

function buildFeatureTensor(records: readonly TransactionRecord[]): Tensor {
  return tensor(
    records.map((row) => [
      row.amount,
      row.hourOfDay,
      row.velocity30m,
      row.geoRisk,
      row.deviceRisk,
      row.merchantRisk,
      row.chargebackHistory,
      row.accountAgeDays,
    ])
  );
}

function buildLabelTensor(records: readonly TransactionRecord[]): Tensor {
  return tensor(records.map((row) => row.isFraud));
}

function selectRows(X: Tensor, y: Tensor, label: number): Tensor {
  const rows: number[][] = [];
  for (let i = 0; i < (y.shape[0] ?? 0); i++) {
    if (Number(y.at(i)) !== label) {
      continue;
    }
    rows.push(Array.from({ length: X.shape[1] ?? 0 }, (_, j) => Number(X.at(i, j))));
  }
  return tensor(rows);
}

function positiveProbabilities(probabilities: Tensor): Tensor {
  return tensor(
    Array.from({ length: probabilities.shape[0] ?? 0 }, (_, i) => Number(probabilities.at(i, 1)))
  );
}

function thresholdPredictions(probabilities: Tensor, threshold: number): Tensor {
  return tensor(
    Array.from({ length: probabilities.shape[0] ?? 0 }, (_, i) =>
      Number(probabilities.at(i)) >= threshold ? 1 : 0
    )
  );
}

function anomalyRecall(yTrue: Tensor, predictedLabels: Tensor): number {
  let fraudCount = 0;
  let flaggedFraudCount = 0;

  for (let i = 0; i < (yTrue.shape[0] ?? 0); i++) {
    const actual = Number(yTrue.at(i));
    const predicted = Number(predictedLabels.at(i));
    if (actual === 1) {
      fraudCount++;
      if (predicted === -1) {
        flaggedFraudCount++;
      }
    }
  }

  return fraudCount === 0 ? 0 : flaggedFraudCount / fraudCount;
}

function meanProbabilityByLabel(yTrue: Tensor, probabilities: Tensor, label: number): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < (yTrue.shape[0] ?? 0); i++) {
    if (Number(yTrue.at(i)) !== label) {
      continue;
    }
    sum += Number(probabilities.at(i));
    count++;
  }
  return count === 0 ? 0 : sum / count;
}

console.log("═".repeat(72));
console.log("  FRAUD DETECTION PLATFORM");
console.log("  Deepbox v1.0.0 production example");
console.log("═".repeat(72));

await mkdir(OUTPUT_DIR, { recursive: true });

const records = generateTransactions(NUM_TRANSACTIONS, RANDOM_SEED);
const transactions = new DataFrame({
  timestamp: records.map((row) => row.timestamp),
  merchantCategory: records.map((row) => row.merchantCategory),
  channel: records.map((row) => row.channel),
  country: records.map((row) => row.country),
  amount: records.map((row) => row.amount),
  hourOfDay: records.map((row) => row.hourOfDay),
  velocity30m: records.map((row) => row.velocity30m),
  geoRisk: records.map((row) => row.geoRisk),
  deviceRisk: records.map((row) => row.deviceRisk),
  merchantRisk: records.map((row) => row.merchantRisk),
  chargebackHistory: records.map((row) => row.chargebackHistory),
  accountAgeDays: records.map((row) => row.accountAgeDays),
  isFraud: records.map((row) => row.isFraud),
});

// ============================================================================
// Step 1: Operations reporting
// ============================================================================
console.log("\n📊 STEP 1: Operational Reporting");
console.log("─".repeat(72));

const parsedTimestamps = to_datetime(records.map((row) => row.timestamp));
const nightTransactions = transactions.query("hourOfDay < 5 and amount > 700");
const fraudShare = records.reduce((sum, row) => sum + row.isFraud, 0) / records.length;

console.log(`Transactions generated: ${records.length}`);
console.log(`Fraud rate: ${(fraudShare * 100).toFixed(2)}%`);
console.log(`Parsed timestamps: ${parsedTimestamps.data.length}`);
console.log(`High-value night transactions: ${nightTransactions.shape[0]}`);
console.log("Merchant category means for frauds only:");
console.log(transactions.query("isFraud == 1").groupBy("merchantCategory").mean().toString());

// ============================================================================
// Step 2: Supervised classifier + calibration
// ============================================================================
console.log("\n🛡️  STEP 2: Supervised Fraud Scoring");
console.log("─".repeat(72));

const featureNames = [
  "amount",
  "hourOfDay",
  "velocity30m",
  "geoRisk",
  "deviceRisk",
  "merchantRisk",
  "chargebackHistory",
  "accountAgeDays",
];

const X = buildFeatureTensor(records);
const y = buildLabelTensor(records);

const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
  testSize: 0.25,
  randomState: RANDOM_SEED,
});

const scaler = new StandardScaler();
const XTrainScaled = scaler.fitTransform(XTrain);
const XTestScaled = scaler.transform(XTest);

const baseModel = new LogisticRegression({
  C: 3.5,
  maxIter: 450,
});
baseModel.fit(XTrainScaled, yTrain);

const calibratedModel = new CalibratedClassifierCV({
  estimator: new LogisticRegression({
    C: 3.5,
    maxIter: 450,
  }),
  method: "sigmoid",
  cv: 4,
});
calibratedModel.fit(XTrainScaled, yTrain);

const reviewThreshold = 0.18;
const baseProbabilities = positiveProbabilities(baseModel.predictProba(XTestScaled));
const calibratedProba = positiveProbabilities(calibratedModel.predictProba(XTestScaled));
const basePred = thresholdPredictions(baseProbabilities, reviewThreshold);

console.log(`Decision threshold: ${reviewThreshold.toFixed(2)} (tuned for fraud review queues)`);
console.log(
  `Base LogisticRegression -> accuracy=${(Number(accuracy(yTest, basePred)) * 100).toFixed(2)}%, precision=${Number(precision(yTest, basePred)).toFixed(4)}, recall=${Number(recall(yTest, basePred)).toFixed(4)}, f1=${Number(f1Score(yTest, basePred)).toFixed(4)}`
);
console.log(
  `Calibrated probabilities -> mean(fraud)=${meanProbabilityByLabel(yTest, calibratedProba, 1).toFixed(4)}, mean(clean)=${meanProbabilityByLabel(yTest, calibratedProba, 0).toFixed(4)}`
);

// ============================================================================
// Step 3: Unsupervised anomaly models
// ============================================================================
console.log("\n🕵️  STEP 3: Unsupervised Anomaly Detectors");
console.log("─".repeat(72));

const normalTrain = selectRows(XTrainScaled, yTrain, 0);

const isolationForest = new IsolationForest({
  nEstimators: 120,
  contamination: fraudShare,
  randomState: RANDOM_SEED,
});
isolationForest.fit(normalTrain);

const lof = new LocalOutlierFactor({
  nNeighbors: 12,
  contamination: fraudShare,
});
lof.fit(normalTrain);

const oneClass = new OneClassSVM({
  nu: Math.min(Math.max(fraudShare * 1.2, 0.05), 0.35),
  kernel: "rbf",
  gamma: "scale",
});
oneClass.fit(normalTrain);

const iforestPred = isolationForest.predict(XTestScaled);
const lofPred = lof.predict(XTestScaled);
const oneClassPred = oneClass.predict(XTestScaled);

console.log(
  `IsolationForest fraud recall: ${(anomalyRecall(yTest, iforestPred) * 100).toFixed(2)}%`
);
console.log(`LocalOutlierFactor recall:    ${(anomalyRecall(yTest, lofPred) * 100).toFixed(2)}%`);
console.log(
  `OneClassSVM recall:           ${(anomalyRecall(yTest, oneClassPred) * 100).toFixed(2)}%`
);

// ============================================================================
// Step 4: Inspection + visualization outputs
// ============================================================================
console.log("\n📈 STEP 4: Inspection & Outputs");
console.log("─".repeat(72));

const reliability = calibrationCurve(yTest, calibratedProba, { nBins: 8 });
const importance = permutationImportance(baseModel, XTestScaled, yTest, {
  nRepeats: 6,
  randomState: RANDOM_SEED,
});

const calibrationFigure = figure({ width: 760, height: 520 });
plotCalibrationCurve(tensor(reliability.fractionPositives), tensor(reliability.meanPredicted), {
  color: "#0f766e",
  label: "Fraud model",
});
await saveFig(`${OUTPUT_DIR}/calibration-curve.svg`, {
  figure: calibrationFigure,
  format: "svg",
});

const importanceFigure = figure({ width: 860, height: 520 });
plotFeatureImportance(importance.importancesMean, featureNames, {
  color: "#1d4ed8",
});
await saveFig(`${OUTPUT_DIR}/feature-importance.svg`, {
  figure: importanceFigure,
  format: "svg",
});

const scoreReport = new DataFrame({
  model: ["LogisticRegression", "IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
  primaryMetric: [
    Number(f1Score(yTest, basePred)).toFixed(4),
    anomalyRecall(yTest, iforestPred).toFixed(4),
    anomalyRecall(yTest, lofPred).toFixed(4),
    anomalyRecall(yTest, oneClassPred).toFixed(4),
  ],
  metricName: ["F1", "Fraud recall", "Fraud recall", "Fraud recall"],
});

const reportPath = `${OUTPUT_DIR}/model-report.json`;
await scoreReport.toJson(reportPath);

console.log(`Saved calibration curve: ${OUTPUT_DIR}/calibration-curve.svg`);
console.log(`Saved feature chart:     ${OUTPUT_DIR}/feature-importance.svg`);
console.log(`Saved JSON report:       ${reportPath}`);

console.log("\n✅ Fraud Detection Platform Complete!");
