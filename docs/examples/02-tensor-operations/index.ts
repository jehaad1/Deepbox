/**
 * Example 02: Tensor Operations
 *
 * Arithmetic, math functions and reductions on tensors, with broadcasting.
 * Every operation is available as a function (add(a, b)) and as a tensor
 * method (a.add(b)). The two forms give the same result. Methods chain, which
 * keeps longer expressions readable.
 */

import { tensor } from "deepbox/ndarray";

console.log("=== Tensor Operations ===\n");

// Basic arithmetic
const a = tensor([1, 2, 3, 4]);
const b = tensor([5, 6, 7, 8]);

console.log("a =", a.toString());
console.log("b =", b.toString());

console.log("\nArithmetic Operations:");
console.log("a + b =", a.add(b).toString());
console.log("a * b =", a.mul(b).toString());
console.log("a - b =", a.sub(b).toString());
console.log("a / b =", a.div(b).toString());

// A JavaScript number is broadcast against every element. It never changes
// the tensor's dtype.
console.log("a * 10 + 1 =", a.mul(10).add(1).toString());

// Broadcasting between shapes: a [2, 1] column against a [3] row gives [2, 3].
const column = tensor([[10], [20]]);
const row = tensor([1, 2, 3]);
console.log("\nBroadcasting [2, 1] + [3]:");
console.log(column.add(row).toString());

// Mathematical functions
const x = tensor([1, 4, 9, 16]);
console.log("\nMathematical Functions:");
console.log("x =", x.toString());
console.log("sqrt(x) =", x.sqrt().toString());
console.log("exp([0, 1, 2]) =", tensor([0, 1, 2]).exp().toString());
console.log("log(x) =", x.log().toString());

// Trigonometric functions
const angles = tensor([0, Math.PI / 4, Math.PI / 2, Math.PI]);
console.log("\nTrigonometric Functions:");
console.log("angles =", angles.toString());
console.log("sin(angles) =", angles.sin().toString());
console.log("cos(angles) =", angles.cos().toString());

// Reduction operations
const matrix = tensor([
  [1, 2, 3],
  [4, 5, 6],
]);

console.log("\nReduction Operations:");
console.log("matrix =");
console.log(matrix.toString());
console.log("sum(matrix) =", matrix.sum().toString());
console.log("mean(matrix) =", matrix.mean().toString());
console.log("max(matrix) =", matrix.max().toString());
console.log("min(matrix) =", matrix.min().toString());

// Axis-wise reductions
console.log("\nAxis-wise Reductions:");
console.log("sum(matrix, axis=0) =", matrix.sum(0).toString());
console.log("sum(matrix, axis=1) =", matrix.sum(1).toString());
console.log("mean(matrix, axis=0) =", matrix.mean(0).toString());

// Reductions return tensors. Use item() to get a plain number.
console.log(`\nmean(matrix) as a number: ${matrix.mean().item()}`);

// Index results are int32 tensors.
console.log("argmax(matrix, axis=1) =", matrix.argmax(1).toString());
