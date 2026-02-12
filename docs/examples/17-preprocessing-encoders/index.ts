/**
 * Example 17: Preprocessing — Encoders
 *
 * Transform categorical and label data into numeric representations
 * using Deepbox's encoding utilities.
 */

import { reshape, tensor } from "deepbox/ndarray";
import {
  LabelBinarizer,
  LabelEncoder,
  MultiLabelBinarizer,
  OneHotEncoder,
  OrdinalEncoder,
} from "deepbox/preprocess";

console.log("=== Preprocessing: Encoders ===\n");

// ---------------------------------------------------------------------------
// Part 1: LabelEncoder — map string labels to integers
// ---------------------------------------------------------------------------
console.log("--- Part 1: LabelEncoder ---");

const le = new LabelEncoder();
le.fit(tensor(["cat", "dog", "bird", "cat", "bird"]));

const encoded = le.transform(tensor(["bird", "cat", "dog"]));
console.log("Encoded:", encoded.toString());

const decoded = le.inverseTransform(encoded);
console.log("Decoded:", decoded.toString());

// ---------------------------------------------------------------------------
// Part 2: OneHotEncoder — one-hot vectors for categorical features
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: OneHotEncoder ---");

const ohe = new OneHotEncoder();
ohe.fit(reshape(tensor(["red", "green", "blue", "red", "blue"]), [5, 1]));

const oneHot = ohe.transform(reshape(tensor(["red", "blue", "green"]), [3, 1]));
console.log("One-hot encoded shape:", oneHot.shape);
console.log("One-hot encoded:\n", oneHot.toString());

// ---------------------------------------------------------------------------
// Part 3: OrdinalEncoder — ordinal integer encoding
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: OrdinalEncoder ---");

const oe = new OrdinalEncoder();
oe.fit(reshape(tensor(["low", "medium", "high", "medium", "low"]), [5, 1]));

const ordinal = oe.transform(reshape(tensor(["low", "high", "medium"]), [3, 1]));
console.log("Ordinal encoded:", ordinal.toString());

const ordinalDecoded = oe.inverseTransform(ordinal);
console.log("Decoded:", ordinalDecoded.toString());

// ---------------------------------------------------------------------------
// Part 4: LabelBinarizer — binary indicator for multi-class labels
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: LabelBinarizer ---");

const lb = new LabelBinarizer();
lb.fit(tensor(["cat", "dog", "bird"]));

const binarized = lb.transform(tensor(["cat", "bird", "dog"]));
console.log("Binarized shape:", binarized.shape);
console.log("Binarized:\n", binarized.toString());

const binarizedDecoded = lb.inverseTransform(binarized);
console.log("Decoded:", binarizedDecoded.toString());

// ---------------------------------------------------------------------------
// Part 5: MultiLabelBinarizer — multi-label binary encoding
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: MultiLabelBinarizer ---");

const mlb = new MultiLabelBinarizer();
mlb.fit([["cat", "dog"], ["bird"], ["cat", "bird", "dog"]]);

const multiEncoded = mlb.transform([["cat", "bird"], ["dog"]]);
console.log("Multi-label encoded shape:", multiEncoded.shape);
console.log("Multi-label encoded:\n", multiEncoded.toString());

console.log("\n=== Preprocessing: Encoders Complete ===");
