/**
 * Example 17: Preprocessing: Encoders
 *
 * Turn categorical and label data into numbers, and back again. Each encoder
 * is fitted on the training values, then transforms and inverse-transforms.
 * Feature encoders (OneHotEncoder, OrdinalEncoder) expect a 2D input with one
 * column per feature, so a 1D list of values is reshaped to [n, 1] first.
 */

import { tensor } from "deepbox/ndarray";
import {
  LabelBinarizer,
  LabelEncoder,
  MultiLabelBinarizer,
  OneHotEncoder,
  OrdinalEncoder,
} from "deepbox/preprocess";

console.log("=== Preprocessing: Encoders ===\n");

// ---------------------------------------------------------------------------
// Part 1: LabelEncoder: map string labels to integers
// ---------------------------------------------------------------------------
console.log("--- Part 1: LabelEncoder ---");

const le = new LabelEncoder();
le.fit(tensor(["cat", "dog", "bird", "cat", "bird"]));

// Classes are sorted alphabetically, so bird = 0, cat = 1, dog = 2.
console.log("Classes:", le.classes?.toString());

const encoded = le.transform(tensor(["bird", "cat", "dog"]));
console.log("Encoded:", encoded.toString());

const decoded = le.inverseTransform(encoded);
console.log("Decoded:", decoded.toString());

// ---------------------------------------------------------------------------
// Part 2: OneHotEncoder: one-hot vectors for categorical features
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: OneHotEncoder ---");

// By default an unseen category throws. { handleUnknown: "ignore" } encodes it as all zeros.
const ohe = new OneHotEncoder();
ohe.fit(tensor(["red", "green", "blue", "red", "blue"]).reshape([5, 1]));

const oneHot = ohe.transform(tensor(["red", "blue", "green"]).reshape([3, 1]));
console.log("One-hot encoded shape:", oneHot.shape);
console.log("One-hot encoded:\n", oneHot.toString());
console.log("Categories per column:", JSON.stringify(ohe.categories));

// ---------------------------------------------------------------------------
// Part 3: OrdinalEncoder: ordinal integer encoding
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: OrdinalEncoder ---");

// With the default categories: "auto", values are numbered in alphabetical order,
// so high = 0, low = 1, medium = 2. That is not the order low < medium < high.
const oe = new OrdinalEncoder();
oe.fit(tensor(["low", "medium", "high", "medium", "low"]).reshape([5, 1]));

const ordinal = oe.transform(tensor(["low", "high", "medium"]).reshape([3, 1]));
console.log("Ordinal encoded (alphabetical):", ordinal.toString());

// Pass the categories in the order you want to get low = 0, medium = 1, high = 2.
const ordered = new OrdinalEncoder({ categories: [["low", "medium", "high"]] });
ordered.fit(tensor(["low", "medium", "high"]).reshape([3, 1]));
console.log(
  "Ordinal encoded (explicit order):",
  ordered.transform(tensor(["low", "high", "medium"]).reshape([3, 1])).toString()
);

const ordinalDecoded = oe.inverseTransform(ordinal);
console.log("Decoded:", ordinalDecoded.toString());

// ---------------------------------------------------------------------------
// Part 4: LabelBinarizer: binary indicator for multi-class labels
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
// Part 5: MultiLabelBinarizer: multi-label binary encoding
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: MultiLabelBinarizer ---");

const mlb = new MultiLabelBinarizer();
mlb.fit([["cat", "dog"], ["bird"], ["cat", "bird", "dog"]]);

const multiEncoded = mlb.transform([["cat", "bird"], ["dog"]]);
console.log("Multi-label encoded shape:", multiEncoded.shape);
console.log("Multi-label encoded:\n", multiEncoded.toString());

console.log("\n=== Preprocessing: Encoders Complete ===");
