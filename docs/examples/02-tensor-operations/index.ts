/**
 * Example 02: Tensor Operations
 *
 * Explore arithmetic, mathematical, and reduction operations on tensors.
 * Deepbox supports 90+ tensor operations with full broadcasting.
 */

import {
  add,
  cos,
  div,
  exp,
  log,
  max,
  mean,
  min,
  mul,
  sin,
  sqrt,
  sub,
  sum,
  tensor,
} from "deepbox/ndarray";

console.log("=== Tensor Operations ===\n");

// Basic arithmetic
const a = tensor([1, 2, 3, 4]);
const b = tensor([5, 6, 7, 8]);

console.log("a =", a.toString());
console.log("b =", b.toString());

console.log("\nArithmetic Operations:");
console.log("a + b =", add(a, b).toString());
console.log("a * b =", mul(a, b).toString());
console.log("a - b =", sub(a, b).toString());
console.log("a / b =", div(a, b).toString());

// Mathematical functions
const x = tensor([1, 4, 9, 16]);
console.log("\nMathematical Functions:");
console.log("x =", x.toString());
console.log("sqrt(x) =", sqrt(x).toString());
console.log("exp(x) =", exp(tensor([0, 1, 2])).toString());
console.log("log(x) =", log(x).toString());

// Trigonometric functions
const angles = tensor([0, Math.PI / 4, Math.PI / 2, Math.PI]);
console.log("\nTrigonometric Functions:");
console.log("angles =", angles.toString());
console.log("sin(angles) =", sin(angles).toString());
console.log("cos(angles) =", cos(angles).toString());

// Reduction operations
const matrix = tensor([
  [1, 2, 3],
  [4, 5, 6],
]);

console.log("\nReduction Operations:");
console.log("matrix =");
console.log(matrix.toString());
console.log("sum(matrix) =", sum(matrix).toString());
console.log("mean(matrix) =", mean(matrix).toString());
console.log("max(matrix) =", max(matrix).toString());
console.log("min(matrix) =", min(matrix).toString());

// Axis-wise reductions
console.log("\nAxis-wise Reductions:");
console.log("sum(matrix, axis=0) =", sum(matrix, 0).toString());
console.log("sum(matrix, axis=1) =", sum(matrix, 1).toString());
console.log("mean(matrix, axis=0) =", mean(matrix, 0).toString());

console.log("\n✓ Tensor operations complete!");
