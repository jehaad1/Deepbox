/**
 * Support Ticket Triage
 *
 * A production-style NLP operations project built with Deepbox text
 * preprocessing, classical ML, model selection, and reporting utilities.
 */

import { mkdir } from "node:fs/promises";
import { DataFrame } from "deepbox/dataframe";
import { accuracy, confusionMatrix, f1Score } from "deepbox/metrics";
import { cross_validate, GridSearchCV, LogisticRegression, MultinomialNB } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { figure, plotConfusionMatrix, saveFig } from "deepbox/plot";
import { CountVectorizer, HashingVectorizer, TfidfVectorizer } from "deepbox/preprocess";

const OUTPUT_DIR = "docs/projects/08-support-ticket-triage/output";
const RANDOM_SEED = 42;
const NUM_TICKETS = 640;

type TicketLabel = 0 | 1 | 2 | 3;

type TicketRecord = {
  readonly ticketId: string;
  readonly channel: string;
  readonly priority: string;
  readonly team: string;
  readonly text: string;
  readonly label: TicketLabel;
};

const LABEL_NAMES = ["billing", "account_access", "outage", "bug_report"];

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function randomChoice<T>(rng: () => number, values: readonly T[]): T {
  return values[Math.floor(rng() * values.length)] ?? values[0]!;
}

function generateTicket(label: TicketLabel, index: number, rng: () => number): TicketRecord {
  const billingPhrases = [
    "invoice charged twice after renewal",
    "refund not reflected on my card statement",
    "subscription amount is different from quote",
    "billing cycle changed without approval",
  ];
  const accessPhrases = [
    "cannot sign in after password reset",
    "mfa code keeps failing on mobile",
    "admin locked out of workspace",
    "sso login redirects to blank page",
  ];
  const outagePhrases = [
    "dashboard returns 503 for all users",
    "api latency is above ten seconds",
    "production sync jobs are stalled",
    "incident bridge opened for regional outage",
  ];
  const bugPhrases = [
    "export csv button freezes the browser",
    "search filters ignore the selected workspace",
    "notification settings save the wrong value",
    "audit log view crashes on pagination",
  ];
  const urgencyPhrases = [
    "urgent customer escalation",
    "sev1 impact to executives",
    "please prioritize this today",
    "blocked for go-live launch",
  ];
  const contexts = [
    "enterprise contract renewal",
    "migration cutover",
    "month-end finance close",
    "on-call support shift",
    "new customer onboarding",
  ];
  const channels = ["email", "chat", "portal"];
  const priorities = ["p1", "p2", "p3"];

  const baseText =
    label === 0
      ? randomChoice(rng, billingPhrases)
      : label === 1
        ? randomChoice(rng, accessPhrases)
        : label === 2
          ? randomChoice(rng, outagePhrases)
          : randomChoice(rng, bugPhrases);

  const urgent = rng() < (label === 2 ? 0.55 : 0.18);
  const priority = urgent ? "p1" : randomChoice(rng, priorities);
  const channel = urgent ? "chat" : randomChoice(rng, channels);
  const team =
    label === 0
      ? "revenue-ops"
      : label === 1
        ? "identity"
        : label === 2
          ? "sre"
          : "product-engineering";

  const text = [
    baseText,
    randomChoice(rng, contexts),
    urgent ? randomChoice(rng, urgencyPhrases) : "customer requested an update by end of day",
  ].join(". ");

  return {
    ticketId: `T-${String(index + 1).padStart(4, "0")}`,
    channel,
    priority,
    team,
    text,
    label,
  };
}

function generateTickets(n: number, seed: number): TicketRecord[] {
  const rng = createRng(seed);
  const records: TicketRecord[] = [];
  const labelWeights: TicketLabel[] = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3];

  for (let i = 0; i < n; i++) {
    const label = randomChoice(rng, labelWeights);
    records.push(generateTicket(label, i, rng));
  }

  return records;
}

function splitArray<T>(values: readonly T[], testFraction: number): [T[], T[]] {
  const splitIndex = Math.floor(values.length * (1 - testFraction));
  return [values.slice(0, splitIndex), values.slice(splitIndex)];
}

console.log("═".repeat(72));
console.log("  SUPPORT TICKET TRIAGE");
console.log("  Deepbox v1.0.0 production example");
console.log("═".repeat(72));

await mkdir(OUTPUT_DIR, { recursive: true });

const tickets = generateTickets(NUM_TICKETS, RANDOM_SEED);
const ticketFrame = new DataFrame({
  ticketId: tickets.map((row) => row.ticketId),
  channel: tickets.map((row) => row.channel),
  priority: tickets.map((row) => row.priority),
  team: tickets.map((row) => row.team),
  text: tickets.map((row) => row.text),
  label: tickets.map((row) => LABEL_NAMES[row.label]),
});

// ============================================================================
// Step 1: Operations view
// ============================================================================
console.log("\n📨 STEP 1: Ticket Operations View");
console.log("─".repeat(72));

const urgentMask = ticketFrame.get("text").str.contains("urgent|sev1|prioritize|blocked");
let urgentCount = 0;
for (const value of urgentMask.data) {
  if (value === true) {
    urgentCount++;
  }
}

console.log(`Tickets generated: ${ticketFrame.shape[0]}`);
console.log(`Urgent/escalated tickets: ${urgentCount}`);
console.log("Average priority mix by team:");
console.log(
  new DataFrame({
    team: tickets.map((row) => row.team),
    priorityCode: tickets.map((row) => (row.priority === "p1" ? 3 : row.priority === "p2" ? 2 : 1)),
  })
    .groupBy("team")
    .mean()
    .toString()
);

// ============================================================================
// Step 2: Train/test text split
// ============================================================================
console.log("\n🧰 STEP 2: Vectorization + Model Selection");
console.log("─".repeat(72));

const texts = tickets.map((row) => row.text);
const labels = tickets.map((row) => row.label);

const [trainTexts, testTexts] = splitArray(texts, 0.2);
const [trainLabels, testLabels] = splitArray(labels, 0.2);

const yTrain = tensor(trainLabels);
const yTest = tensor(testLabels);

const countVectorizer = new CountVectorizer({
  ngramRange: [1, 2],
  maxFeatures: 400,
  minDf: 2,
});
const tfidfVectorizer = new TfidfVectorizer({
  ngramRange: [1, 2],
  maxFeatures: 400,
  minDf: 2,
});
const hashingVectorizer = new HashingVectorizer({
  nFeatures: 256,
  ngramRange: [1, 2],
  alternateSign: false,
});

const XTrainCount = countVectorizer.fitTransformText(trainTexts);
const XTestCount = countVectorizer.transformText(testTexts);
const XTrainTfidf = tfidfVectorizer.fitTransformText(trainTexts);
const XTestTfidf = tfidfVectorizer.transformText(testTexts);
const XTrainHash = hashingVectorizer.fitTransformText(trainTexts);
const XTestHash = hashingVectorizer.transformText(testTexts);

const ticketSearch = new GridSearchCV(
  new LogisticRegression({ maxIter: 300 }),
  {
    C: [0.5, 1.0, 2.0, 4.0],
    maxIter: [250, 350],
  },
  { cv: 4 }
);
ticketSearch.fit(XTrainTfidf, yTrain);

const nbModel = new MultinomialNB();
nbModel.fit(XTrainCount, yTrain);

const hashModel = new LogisticRegression({ C: 1.0, maxIter: 300 });
hashModel.fit(XTrainHash, yTrain);

const bestLogistic = ticketSearch.bestEstimator as LogisticRegression;
const logisticPred = bestLogistic.predict(XTestTfidf);
const nbPred = nbModel.predict(XTestCount);
const hashPred = hashModel.predict(XTestHash);

const comparisonRows = [
  {
    pipeline: "TF-IDF + GridSearchCV(LogReg)",
    accuracy: Number(accuracy(yTest, logisticPred)),
    weightedF1: Number(f1Score(yTest, logisticPred, "weighted")),
    predictions: logisticPred,
  },
  {
    pipeline: "Count + MultinomialNB",
    accuracy: Number(accuracy(yTest, nbPred)),
    weightedF1: Number(f1Score(yTest, nbPred, "weighted")),
    predictions: nbPred,
  },
  {
    pipeline: "Hashing + LogisticRegression",
    accuracy: Number(accuracy(yTest, hashPred)),
    weightedF1: Number(f1Score(yTest, hashPred, "weighted")),
    predictions: hashPred,
  },
];

comparisonRows.sort((left, right) => right.weightedF1 - left.weightedF1);

const bestPipeline = comparisonRows[0]!;

const results = new DataFrame({
  pipeline: comparisonRows.map((row) => row.pipeline),
  accuracy: comparisonRows.map((row) => row.accuracy.toFixed(4)),
  weightedF1: comparisonRows.map((row) => row.weightedF1.toFixed(4)),
});

console.log(`Best logistic params: ${JSON.stringify(ticketSearch.bestParams)}`);
console.log(`Best deployable pipeline: ${bestPipeline.pipeline}`);
console.log(results.toString());

const cv = cross_validate(bestLogistic, XTrainTfidf, yTrain, { cv: 4 });
const cvScores = cv.testScores["score"] ?? [];
console.log(`Cross-validation scores: ${cvScores.map((score) => score.toFixed(4)).join(", ")}`);

// ============================================================================
// Step 3: Confusion matrix and artifacts
// ============================================================================
console.log("\n📈 STEP 3: Reporting Outputs");
console.log("─".repeat(72));

const confusion = confusionMatrix(yTest, bestPipeline.predictions);
const confusionFigure = figure({ width: 720, height: 520 });
plotConfusionMatrix(confusion, LABEL_NAMES, {
  color: "#1d4ed8",
});
await saveFig(`${OUTPUT_DIR}/best-model-confusion-matrix.svg`, {
  figure: confusionFigure,
  format: "svg",
});

await results.toJson(`${OUTPUT_DIR}/model-comparison.json`);

const vocabularyPreview = new DataFrame({
  topTerms: tfidfVectorizer.getFeatureNames().slice(0, 20),
});
await vocabularyPreview.toJson(`${OUTPUT_DIR}/tfidf-vocabulary-preview.json`);

console.log(`Saved confusion matrix: ${OUTPUT_DIR}/best-model-confusion-matrix.svg`);
console.log(`Saved model summary:    ${OUTPUT_DIR}/model-comparison.json`);
console.log(`Saved vocabulary dump:  ${OUTPUT_DIR}/tfidf-vocabulary-preview.json`);

console.log("\n✅ Support Ticket Triage Complete!");
