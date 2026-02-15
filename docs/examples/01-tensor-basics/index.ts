/**
 * Example 01: Tensor Basics
 *
 * Learn the fundamentals of creating and manipulating tensors (N-dimensional arrays).
 * Tensors are the core data structure in Deepbox, the core data structure in Deepbox.
 */

import { arange, eye, linspace, ones, reshape, tensor, zeros } from "deepbox/ndarray";

console.log("=== Tensor Basics ===\n");

// 1. Creating tensors from JavaScript arrays
const vector = tensor([1, 2, 3, 4, 5]);
console.log("1D Tensor (vector):");
console.log(vector.toString());
console.log(`Shape: [${vector.shape}], Size: ${vector.size}\n`);

// 2. Create a 2D tensor (matrix)
const matrix = tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
console.log("2D Tensor (matrix):");
console.log(matrix.toString());
console.log(`Shape: [${matrix.shape}], Size: ${matrix.size}\n`);

// 3. Create a 3D tensor (higher dimensional)
const tensor3d = tensor([
  [
    [1, 2],
    [3, 4],
  ],
  [
    [5, 6],
    [7, 8],
  ],
]);
console.log("3D Tensor:");
console.log(`Shape: [${tensor3d.shape}], Size: ${tensor3d.size}\n`);

// 4. Special tensor creation functions
const zeroMatrix = zeros([3, 3]);
console.log("3x3 Zero Matrix:");
console.log(`${zeroMatrix.toString()}\n`);

const onesMatrix = ones([2, 4]);
console.log("2x4 Ones Matrix:");
console.log(`${onesMatrix.toString()}\n`);

// 5. Create identity matrix (diagonal of ones)
const identity = eye(4);
console.log("4x4 Identity Matrix:");
console.log(`${identity.toString()}\n`);

// 6. Create range of values with step
const range = arange(0, 10, 2); // start, stop, step
console.log("Range [0, 10) with step 2:");
console.log(`${range.toString()}\n`);

// 7. Create linearly spaced values
const linspaced = linspace(0, 1, 5); // start, stop, num_points
console.log("5 values linearly spaced between 0 and 1:");
console.log(`${linspaced.toString()}\n`);

// 8. Reshape tensors to different dimensions
const flat = tensor([1, 2, 3, 4, 5, 6]);
const reshaped = reshape(flat, [2, 3]);
console.log("Reshaped [6] -> [2, 3]:");
console.log(`${reshaped.toString()}\n`);

console.log("✓ Tensor basics complete!");
