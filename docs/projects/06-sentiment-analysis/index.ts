/**
 * Sentiment Analysis System
 *
 * Text classification for sentiment analysis using bag-of-words and ML classifiers.
 *
 * Deepbox Modules Used:
 * - deepbox/ml: LogisticRegression, GaussianNB
 * - deepbox/preprocess: StandardScaler, trainTestSplit
 * - deepbox/metrics: Classification metrics
 * - deepbox/dataframe: Data manipulation
 * - deepbox/ndarray: Tensor operations
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { DataFrame } from "deepbox/dataframe";
import { accuracy, confusionMatrix, f1Score, precision, recall } from "deepbox/metrics";
import { GaussianNB, LogisticRegression } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

// ============================================================================
// Configuration
// ============================================================================

const OUTPUT_DIR = "docs/projects/06-sentiment-analysis/output";
const NUM_SAMPLES = 500;
const VOCAB_SIZE = 100;

// ============================================================================
// Data Generation & Text Processing
// ============================================================================

/**
 * Sample positive and negative word lists for sentiment
 */
const POSITIVE_WORDS = [
  "good",
  "great",
  "excellent",
  "amazing",
  "wonderful",
  "fantastic",
  "awesome",
  "love",
  "like",
  "enjoy",
  "happy",
  "pleased",
  "satisfied",
  "recommend",
  "best",
  "perfect",
  "brilliant",
  "outstanding",
  "superb",
  "delightful",
  "impressive",
  "beautiful",
  "incredible",
  "exceptional",
  "remarkable",
  "terrific",
  "fabulous",
];

const NEGATIVE_WORDS = [
  "bad",
  "terrible",
  "awful",
  "horrible",
  "poor",
  "worst",
  "hate",
  "dislike",
  "disappointed",
  "frustrating",
  "annoying",
  "waste",
  "boring",
  "slow",
  "broken",
  "useless",
  "pathetic",
  "dreadful",
  "disgusting",
  "miserable",
  "unpleasant",
  "disappointing",
  "inadequate",
  "inferior",
  "subpar",
  "mediocre",
  "defective",
];

const NEUTRAL_WORDS = [
  "the",
  "a",
  "an",
  "is",
  "was",
  "it",
  "this",
  "that",
  "with",
  "for",
  "on",
  "product",
  "service",
  "item",
  "order",
  "delivery",
  "quality",
  "price",
  "time",
  "experience",
  "customer",
  "support",
  "would",
  "could",
  "should",
  "have",
  "been",
];

/**
 * Generate synthetic review text
 */
function generateReview(isPositive: boolean, seed: number): { text: string; words: string[] } {
  let randomSeed = seed;
  const seededRandom = () => {
    randomSeed = (randomSeed * 1103515245 + 12345) & 0x7fffffff;
    return randomSeed / 0x7fffffff;
  };

  const numWords = Math.floor(seededRandom() * 15) + 10; // 10-25 words
  const words: string[] = [];

  const primaryWords = isPositive ? POSITIVE_WORDS : NEGATIVE_WORDS;
  const secondaryWords = isPositive ? NEGATIVE_WORDS : POSITIVE_WORDS;

  for (let i = 0; i < numWords; i++) {
    const r = seededRandom();
    if (r < 0.3) {
      // 30% sentiment words (mostly matching sentiment)
      if (seededRandom() < 0.85) {
        words.push(primaryWords[Math.floor(seededRandom() * primaryWords.length)]);
      } else {
        words.push(secondaryWords[Math.floor(seededRandom() * secondaryWords.length)]);
      }
    } else {
      // 70% neutral words
      words.push(NEUTRAL_WORDS[Math.floor(seededRandom() * NEUTRAL_WORDS.length)]);
    }
  }

  return { text: words.join(" "), words };
}

/**
 * Build vocabulary from reviews
 */
function buildVocabulary(reviews: string[][], maxSize: number): string[] {
  const wordCounts: Map<string, number> = new Map();

  for (const words of reviews) {
    for (const word of words) {
      wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
    }
  }

  // Sort by frequency and take top words
  const sorted = Array.from(wordCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxSize);

  return sorted.map(([word]) => word);
}

/**
 * Convert text to bag-of-words vector
 */
function textToVector(words: string[], vocabulary: string[]): number[] {
  const vector = new Array(vocabulary.length).fill(0);

  for (const word of words) {
    const idx = vocabulary.indexOf(word);
    if (idx >= 0) {
      vector[idx]++;
    }
  }

  return vector;
}

/**
 * Calculate TF-IDF weights
 */
function calculateTFIDF(documents: number[][], vocabulary: string[]): number[][] {
  const numDocs = documents.length;
  const vocabSize = vocabulary.length;

  // Calculate document frequency for each term
  const df = new Array(vocabSize).fill(0);
  for (const doc of documents) {
    for (let i = 0; i < vocabSize; i++) {
      if (doc[i] > 0) {
        df[i]++;
      }
    }
  }

  // Calculate TF-IDF
  const tfidf: number[][] = [];
  for (const doc of documents) {
    const docLength = doc.reduce((a, b) => a + b, 0);
    const tfidfDoc: number[] = [];

    for (let i = 0; i < vocabSize; i++) {
      const tf = docLength > 0 ? doc[i] / docLength : 0;
      const idf = df[i] > 0 ? Math.log(numDocs / df[i]) : 0;
      tfidfDoc.push(tf * idf);
    }

    tfidf.push(tfidfDoc);
  }

  return tfidf;
}

// ============================================================================
// Main Execution
// ============================================================================

console.log("═".repeat(70));
console.log("  SENTIMENT ANALYSIS SYSTEM");
console.log("  Built with Deepbox - TypeScript Data Science & ML Library");
console.log("═".repeat(70));

// Create output directory
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ============================================================================
// Step 1: Generate Data
// ============================================================================

console.log("\n📊 STEP 1: Generating Sentiment Data");
console.log("─".repeat(70));

const reviews: { text: string; words: string[]; label: number }[] = [];

for (let i = 0; i < NUM_SAMPLES; i++) {
  const isPositive = i < NUM_SAMPLES / 2;
  const { text, words } = generateReview(isPositive, i + 42);
  reviews.push({ text, words, label: isPositive ? 1 : 0 });
}

// Shuffle reviews
for (let i = reviews.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [reviews[i], reviews[j]] = [reviews[j], reviews[i]];
}

const numPositive = reviews.filter((r) => r.label === 1).length;
const numNegative = reviews.filter((r) => r.label === 0).length;

console.log(`\n✓ Generated ${NUM_SAMPLES} reviews`);
console.log(`  Positive: ${numPositive} (${((numPositive / NUM_SAMPLES) * 100).toFixed(1)}%)`);
console.log(`  Negative: ${numNegative} (${((numNegative / NUM_SAMPLES) * 100).toFixed(1)}%)`);

// Sample reviews
console.log("\nSample Reviews:");
console.log(`  [Positive] "${reviews.find((r) => r.label === 1)?.text.slice(0, 60)}..."`);
console.log(`  [Negative] "${reviews.find((r) => r.label === 0)?.text.slice(0, 60)}..."`);

// ============================================================================
// Step 2: Build Vocabulary & Features
// ============================================================================

console.log("\n📝 STEP 2: Building Vocabulary & Features");
console.log("─".repeat(70));

// Build vocabulary
const allWords = reviews.map((r) => r.words);
const vocabulary = buildVocabulary(allWords, VOCAB_SIZE);

console.log(`\n✓ Built vocabulary with ${vocabulary.length} terms`);
console.log(`  Top terms: ${vocabulary.slice(0, 10).join(", ")}...`);

// Convert to bag-of-words
const bowVectors = reviews.map((r) => textToVector(r.words, vocabulary));

// Calculate TF-IDF
const tfidfVectors = calculateTFIDF(bowVectors, vocabulary);

console.log(`✓ Created TF-IDF vectors`);
console.log(`  Vector dimension: ${tfidfVectors[0].length}`);

// Prepare data
const X = tensor(tfidfVectors);
const y = tensor(reviews.map((r) => r.label));

// ============================================================================
// Step 3: Train/Test Split
// ============================================================================

console.log("\n📦 STEP 3: Train/Test Split");
console.log("─".repeat(70));

const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
  testSize: 0.2,
  randomState: 42,
  shuffle: true,
});

console.log(`\n✓ Split data`);
console.log(`  Training: ${XTrain.shape[0]} samples`);
console.log(`  Testing: ${XTest.shape[0]} samples`);

// Scale features
const scaler = new StandardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);

console.log(`✓ Applied StandardScaler`);

// ============================================================================
// Step 4: Model Training
// ============================================================================

console.log("\n🤖 STEP 4: Model Training");
console.log("─".repeat(70));

// Logistic Regression
console.log("\nTraining Logistic Regression...");
const lr = new LogisticRegression({ maxIter: 200, learningRate: 0.1 });
const lrStart = Date.now();
lr.fit(XTrainScaled, yTrain);
const lrTime = Date.now() - lrStart;
console.log(`  ✓ Trained in ${lrTime}ms`);

// Naive Bayes
console.log("Training Gaussian Naive Bayes...");
const nb = new GaussianNB();
const nbStart = Date.now();
nb.fit(XTrainScaled, yTrain);
const nbTime = Date.now() - nbStart;
console.log(`  ✓ Trained in ${nbTime}ms`);

// ============================================================================
// Step 5: Model Evaluation
// ============================================================================

console.log("\n📈 STEP 5: Model Evaluation");
console.log("─".repeat(70));

// Get predictions
const yPredLR = lr.predict(XTestScaled);
const yPredNB = nb.predict(XTestScaled);

// Calculate metrics
const results = [
  {
    name: "Logistic Regression",
    accuracy: accuracy(yTest, yPredLR),
    precision: precision(yTest, yPredLR, "binary"),
    recall: recall(yTest, yPredLR, "binary"),
    f1: f1Score(yTest, yPredLR, "binary"),
    predictions: yPredLR,
    time: lrTime,
  },
  {
    name: "Gaussian Naive Bayes",
    accuracy: accuracy(yTest, yPredNB),
    precision: precision(yTest, yPredNB, "binary"),
    recall: recall(yTest, yPredNB, "binary"),
    f1: f1Score(yTest, yPredNB, "binary"),
    predictions: yPredNB,
    time: nbTime,
  },
];

console.log("\nModel Comparison:\n");
const metricsDF = new DataFrame({
  Model: results.map((r) => r.name),
  "Accuracy (%)": results.map((r) => (Number(r.accuracy) * 100).toFixed(2)),
  "Precision (%)": results.map((r) => (Number(r.precision) * 100).toFixed(2)),
  "Recall (%)": results.map((r) => (Number(r.recall) * 100).toFixed(2)),
  "F1 Score (%)": results.map((r) => (Number(r.f1) * 100).toFixed(2)),
  "Time (ms)": results.map((r) => r.time.toString()),
});
console.log(metricsDF.toString());

// Best model
const bestModel = results.reduce((best, r) => (Number(r.f1) > Number(best.f1) ? r : best));
console.log(`\n🏆 Best Model: ${bestModel.name} (F1: ${(Number(bestModel.f1) * 100).toFixed(2)}%)`);

// ============================================================================
// Step 6: Confusion Matrix
// ============================================================================

console.log("\n📊 STEP 6: Confusion Matrix Analysis");
console.log("─".repeat(70));

const cm = confusionMatrix(yTest, bestModel.predictions);
const cmData = expectNumericTypedArray(cm.data);

console.log(`\nConfusion Matrix (${bestModel.name}):`);
console.log("                  Predicted");
console.log("                  Negative  Positive");
console.log(
  `  Actual Negative    ${String(cmData[0]).padStart(4)}      ${String(cmData[1]).padStart(4)}`
);
console.log(
  `  Actual Positive    ${String(cmData[2]).padStart(4)}      ${String(cmData[3]).padStart(4)}`
);

const tn = cmData[0];
const fp = cmData[1];
const fn = cmData[2];
const tp = cmData[3];

console.log(`\n  True Negatives:  ${tn}`);
console.log(`  False Positives: ${fp}`);
console.log(`  False Negatives: ${fn}`);
console.log(`  True Positives:  ${tp}`);

// ============================================================================
// Step 7: Feature Analysis
// ============================================================================

console.log("\n🔍 STEP 7: Feature Analysis");
console.log("─".repeat(70));

// Analyze most predictive words (simple frequency analysis)
const posWordCounts: Map<string, number> = new Map();
const negWordCounts: Map<string, number> = new Map();

for (const review of reviews) {
  const countMap = review.label === 1 ? posWordCounts : negWordCounts;
  for (const word of review.words) {
    countMap.set(word, (countMap.get(word) || 0) + 1);
  }
}

// Find words with highest positive/negative ratio
const wordRatios: {
  word: string;
  ratio: number;
  posCount: number;
  negCount: number;
}[] = [];
for (const word of vocabulary) {
  const posCount = posWordCounts.get(word) || 0;
  const negCount = negWordCounts.get(word) || 0;
  if (posCount + negCount >= 5) {
    // Minimum frequency
    const ratio = (posCount + 1) / (negCount + 1); // Laplace smoothing
    wordRatios.push({ word, ratio, posCount, negCount });
  }
}

// Most positive words
const mostPositive = wordRatios.sort((a, b) => b.ratio - a.ratio).slice(0, 5);
// Most negative words
const mostNegative = wordRatios.sort((a, b) => a.ratio - b.ratio).slice(0, 5);

console.log("\nMost Positive Words:");
for (const { word, ratio, posCount, negCount } of mostPositive) {
  console.log(`  "${word}": ratio=${ratio.toFixed(2)} (pos=${posCount}, neg=${negCount})`);
}

console.log("\nMost Negative Words:");
for (const { word, ratio, posCount, negCount } of mostNegative) {
  console.log(`  "${word}": ratio=${ratio.toFixed(2)} (pos=${posCount}, neg=${negCount})`);
}

// ============================================================================
// Step 8: Sample Predictions
// ============================================================================

console.log("\n🎯 STEP 8: Sample Predictions");
console.log("─".repeat(70));

// Show some predictions
const testLabels = yTest.data;
const predLabels = bestModel.predictions.data;

console.log("\nSample Test Predictions:\n");
let correct = 0;
let shown = 0;

for (let i = 0; i < Math.min(10, XTest.shape[0]); i++) {
  const actual = testLabels[i] === 1 ? "Positive" : "Negative";
  const predicted = predLabels[i] === 1 ? "Positive" : "Negative";
  const status = testLabels[i] === predLabels[i] ? "✓" : "✗";

  if (shown < 5) {
    console.log(`  ${status} Actual: ${actual.padEnd(10)} Predicted: ${predicted}`);
    shown++;
  }

  if (testLabels[i] === predLabels[i]) correct++;
}

console.log(`\n  Total correct: ${correct}/${XTest.shape[0]}`);

// ============================================================================
// Step 9: Visualizations
// ============================================================================

console.log("\n📊 STEP 9: Generating Visualizations");
console.log("─".repeat(70));

// Model comparison chart
try {
  const fig = new Figure({ width: 800, height: 400 });
  const ax = fig.addAxes();

  const modelIndices = [0, 1];
  const f1Scores = results.map((r) => Number(r.f1) * 100);

  ax.bar(tensor(modelIndices), tensor(f1Scores), { color: "#4CAF50" });
  ax.setTitle("Model Comparison (F1 Score)");
  ax.setXLabel("Model");
  ax.setYLabel("F1 Score (%)");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/model-comparison.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/model-comparison.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate model comparison: ${e}`);
}

// ============================================================================
// Summary
// ============================================================================

console.log(`\n${"═".repeat(70)}`);
console.log("  SENTIMENT ANALYSIS COMPLETE - SUMMARY");
console.log("═".repeat(70));

console.log("\n📌 Key Findings:\n");
console.log("  1. Data Overview:");
console.log(`     • ${NUM_SAMPLES} reviews analyzed`);
console.log(`     • Vocabulary size: ${VOCAB_SIZE} terms`);
console.log(`     • Balanced classes (50/50)`);

console.log("\n  2. Best Model:");
console.log(`     • ${bestModel.name}`);
console.log(`     • Accuracy: ${(Number(bestModel.accuracy) * 100).toFixed(2)}%`);
console.log(`     • F1 Score: ${(Number(bestModel.f1) * 100).toFixed(2)}%`);
console.log(`     • Training time: ${bestModel.time}ms`);

console.log("\n  3. Observations:");
console.log("     • TF-IDF features capture sentiment effectively");
console.log("     • Both models achieve good performance");
console.log("     • Sentiment words are strong predictors");

console.log("\n📁 Output Files:");
console.log(`   • ${OUTPUT_DIR}/model-comparison.svg`);

console.log(`\n${"═".repeat(70)}`);
console.log("  ✅ Sentiment Analysis Complete!");
console.log("═".repeat(70));
