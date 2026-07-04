/**
 * Matrix operations: matrix_power and kron.
 *
 * @module linalg/matrix_ops
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox Linear Algebra}
 */

import { getDevice, getDtype, InvalidParameterError, ShapeError } from "../core";
import { type Tensor, Tensor as TensorClass, tensor, zeros } from "../ndarray";
import { eig } from "./decomposition/index";
import { inv } from "./inverse";

/**
 * Raise a square matrix to an integer power.
 *
 * For positive n: computes A^n via repeated squaring.
 * For n=0: returns the identity matrix.
 * For negative n: computes (A^{-1})^{|n|}.
 *
 * @param A - Square matrix of shape (m, m)
 * @param n - Integer exponent
 * @returns A^n of shape (m, m)
 *
 * @example
 * ```ts
 * import { matrix_power } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const A3 = matrix_power(A, 3); // A * A * A
 * ```
 */
export function matrix_power(A: Tensor, n: number): Tensor {
  if (A.ndim !== 2) {
    throw new ShapeError(`matrix_power requires 2D matrix; got ndim=${A.ndim}`);
  }
  const m = A.shape[0] ?? 0;
  if (m !== (A.shape[1] ?? 0)) {
    throw new ShapeError(`matrix_power requires square matrix; got shape [${A.shape.join(", ")}]`);
  }
  if (!Number.isInteger(n)) {
    throw new InvalidParameterError(`n must be an integer; received ${n}`, "n", n);
  }

  // n = 0 -> identity
  if (n === 0) {
    const data: number[] = [];
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        data.push(i === j ? 1 : 0);
      }
    }
    return tensor(data).reshape([m, m]);
  }

  let base: Tensor;
  let exp: number;

  if (n < 0) {
    base = inv(A);
    exp = -n;
  } else {
    base = A;
    exp = n;
  }

  // n = 1
  if (exp === 1) {
    // Return a copy (stride-aware so transposed/view inputs are handled).
    return tensor(matToArr(base, m).flat()).reshape([m, m]);
  }

  // Exponentiation by squaring
  let result: number[][] | undefined;
  let current = matToArr(base, m);

  while (exp > 0) {
    if (exp % 2 === 1) {
      result = result ? matMul(result, current, m) : current.map((r) => [...r]);
    }
    current = matMul(current, current, m);
    exp = Math.floor(exp / 2);
  }

  const flat: number[] = [];
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      flat.push(result![i]![j]!);
    }
  }
  return tensor(flat).reshape([m, m]);
}

function matToArr(A: Tensor, m: number): number[][] {
  // Honour the tensor's strides/offset so non-contiguous views (e.g. a
  // transposed matrix) are read correctly rather than assuming row-major.
  const s0 = A.strides[0] ?? m;
  const s1 = A.strides[1] ?? 1;
  const arr: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      row.push(Number(A.data[A.offset + i * s0 + j * s1]));
    }
    arr.push(row);
  }
  return arr;
}

function matMul(A: number[][], B: number[][], m: number): number[][] {
  const C: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) {
        sum += A[i]![k]! * B[k]![j]!;
      }
      row.push(sum);
    }
    C.push(row);
  }
  return C;
}

/**
 * Compute the Kronecker product of two matrices.
 *
 * If A is (m, n) and B is (p, q), the result is (m*p, n*q).
 *
 * @param A - First matrix of shape (m, n)
 * @param B - Second matrix of shape (p, q)
 * @returns Kronecker product of shape (m*p, n*q)
 *
 * @example
 * ```ts
 * import { kron } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const B = tensor([[0, 5], [6, 7]]);
 * const K = kron(A, B); // shape [4, 4]
 * ```
 */
export function kron(A: Tensor, B: Tensor): Tensor {
  if (A.ndim !== 2) {
    throw new ShapeError(`kron requires 2D matrices; A has ndim=${A.ndim}`);
  }
  if (B.ndim !== 2) {
    throw new ShapeError(`kron requires 2D matrices; B has ndim=${B.ndim}`);
  }

  const m = A.shape[0] ?? 0;
  const n = A.shape[1] ?? 0;
  const p = B.shape[0] ?? 0;
  const q = B.shape[1] ?? 0;
  const rows = m * p;
  const cols = n * q;

  // Densify both operands once (honouring strides/offset), then run the
  // quadruple loop over monomorphic Float64Arrays into a typed output —
  // per-element Number() reads plus a number[] → tensor() re-validation
  // made this ~20x slower.
  const aS0 = A.strides[0] ?? n;
  const aS1 = A.strides[1] ?? 1;
  const bS0 = B.strides[0] ?? q;
  const bS1 = B.strides[1] ?? 1;
  const aDense = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      aDense[i * n + j] = Number(A.data[A.offset + i * aS0 + j * aS1]);
    }
  }
  const bDense = new Float64Array(p * q);
  for (let k = 0; k < p; k++) {
    for (let l = 0; l < q; l++) {
      bDense[k * q + l] = Number(B.data[B.offset + k * bS0 + l * bS1]);
    }
  }

  const out = new Float64Array(rows * cols);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const aVal = aDense[i * n + j] as number;
      for (let k = 0; k < p; k++) {
        const outBase = (i * p + k) * cols + j * q;
        const bBase = k * q;
        for (let l = 0; l < q; l++) {
          out[outBase + l] = aVal * (bDense[bBase + l] as number);
        }
      }
    }
  }

  return TensorClass.fromTypedArray({
    data: out,
    shape: [rows, cols],
    dtype: "float64",
    device: A.device,
  });
}

/**
 * Construct a block-diagonal matrix from provided square matrices.
 *
 * @param matrices - One or more 2-D tensors to place along the diagonal
 * @returns Block-diagonal matrix
 *
 * @example
 * ```ts
 * import { block_diag } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[1, 2], [3, 4]]);
 * const B = tensor([[5]]);
 * const D = block_diag(A, B); // shape [3, 3]
 * ```
 */
export function block_diag(...matrices: Tensor[]): Tensor {
  if (matrices.length === 0) {
    return zeros([0, 0]);
  }
  let totalRows = 0;
  let totalCols = 0;
  for (const M of matrices) {
    if (M.ndim !== 2) {
      throw new ShapeError(`block_diag requires 2D matrices; got ndim=${M.ndim}`);
    }
    totalRows += M.shape[0] ?? 0;
    totalCols += M.shape[1] ?? 0;
  }

  const data = new Float64Array(totalRows * totalCols); // zero-filled
  let rowOff = 0;
  let colOff = 0;
  for (const M of matrices) {
    const mr = M.shape[0] ?? 0;
    const mc = M.shape[1] ?? 0;
    for (let i = 0; i < mr; i++) {
      for (let j = 0; j < mc; j++) {
        data[(rowOff + i) * totalCols + (colOff + j)] = Number(
          M.data[M.offset + i * (M.strides[0] ?? 0) + j * (M.strides[1] ?? 0)]
        );
      }
    }
    rowOff += mr;
    colOff += mc;
  }

  return TensorClass.fromTypedArray({
    data,
    shape: [totalRows, totalCols],
    dtype: "float64",
    device: matrices[0]?.device ?? "cpu",
  });
}

// ---- Matrix functions via eigendecomposition: expm, logm, sqrtm ----

function assertSquare(A: Tensor, name: string): number {
  if (A.ndim !== 2) {
    throw new ShapeError(`${name} requires a 2-D matrix; got ${A.ndim}-D`);
  }
  const m = A.shape[0] ?? 0;
  if (m !== (A.shape[1] ?? 0)) {
    throw new ShapeError(`${name} requires a square matrix; got shape [${A.shape.join(", ")}]`);
  }
  return m;
}

/**
 * Apply a scalar function to a square matrix via eigendecomposition.
 * Computes f(A) = V diag(f(λ)) V^{-1} using the `eig` decomposition.
 */
function matrixFunction(A: Tensor, fn: (x: number) => number, name: string): Tensor {
  const m = assertSquare(A, name);
  if (m === 0) return zeros([0, 0]);

  // eig returns [eigenvalues: Tensor[m], eigenvectors: Tensor[m,m]]
  // eig already throws for complex eigenvalues
  const [evalsTensor, evecsTensor] = eig(A);

  // Extract eigenvalues (1-D, shape [m])
  const fLambda: number[] = [];
  for (let i = 0; i < m; i++) {
    fLambda.push(fn(Number(evalsTensor.data[evalsTensor.offset + i])));
  }

  // Extract eigenvector matrix V (shape [m, m])
  const V = matToArr(evecsTensor, m);

  // Compute V^{-1}
  const Vinv = inv(evecsTensor);
  const VinvArr = matToArr(Vinv, m);

  // Result = V * diag(fLambda) * V^{-1}
  const result: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) {
        sum += V[i]![k]! * fLambda[k]! * VinvArr[k]![j]!;
      }
      row.push(sum);
    }
    result.push(row);
  }

  return tensor(result.flat()).reshape([m, m]);
}

/**
 * Compute the matrix exponential exp(A).
 *
 * @param A - Square matrix
 * @returns exp(A)
 *
 * @example
 * ```ts
 * import { expm } from 'deepbox/linalg';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const A = tensor([[0, 1], [-1, 0]]);
 * const E = expm(A);
 * ```
 */
export function expm(A: Tensor): Tensor {
  const m = assertSquare(A, "expm");
  if (m === 0) return zeros([0, 0]);

  // Scaling and squaring with a degree-6 Padé approximant. Unlike the
  // eigendecomposition approach, this is well-defined for defective matrices
  // and matrices with complex eigenvalues (e.g. rotation [[0,1],[-1,0]]).
  const a = matToArr(A, m);

  // 1-norm (max column sum) to choose the scaling power
  let norm1 = 0;
  for (let j = 0; j < m; j++) {
    let colSum = 0;
    for (let i = 0; i < m; i++) colSum += Math.abs(a[i]![j]!);
    if (colSum > norm1) norm1 = colSum;
  }

  const s = Math.max(0, Math.ceil(Math.log2(norm1)) + 1);
  const scale = 2 ** s;

  const ident = (): number[][] =>
    Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => (i === j ? 1 : 0)));
  const matmul = (x: number[][], y: number[][]): number[][] => {
    const out = Array.from({ length: m }, () => new Array<number>(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < m; k++) {
        const xik = x[i]![k]!;
        if (xik === 0) continue;
        for (let j = 0; j < m; j++) out[i]![j]! += xik * y[k]![j]!;
      }
    }
    return out;
  };

  // Scaled matrix B = A / 2^s
  const B = a.map((row) => row.map((v) => v / scale));

  // Degree-6 Padé: N = sum c_k B^k, D = sum (-1)^k c_k B^k; expm ≈ D^{-1} N.
  // Coefficients c_k = (2q-k)! q! / ((2q)! k! (q-k)!) for q = 6.
  const cPade = [1, 0.5, 5 / 44, 1 / 66, 1 / 792, 1 / 15840, 1 / 665280];

  const powers: number[][][] = [ident()];
  for (let k = 1; k <= 6; k++) powers.push(matmul(powers[k - 1]!, B));

  const N = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  const D = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  for (let k = 0; k <= 6; k++) {
    const ck = cPade[k]!;
    const sign = k % 2 === 0 ? 1 : -1;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        N[i]![j]! += ck * powers[k]![i]![j]!;
        D[i]![j]! += sign * ck * powers[k]![i]![j]!;
      }
    }
  }

  // R = D^{-1} N
  const Dinv = matToArr(inv(tensor(D.flat()).reshape([m, m])), m);
  let R = matmul(Dinv, N);

  // Square s times
  for (let i = 0; i < s; i++) R = matmul(R, R);

  return tensor(R.flat()).reshape([m, m]);
}

/**
 * Compute the matrix logarithm log(A).
 *
 * Requires all eigenvalues to be real and positive.
 *
 * @param A - Square matrix with positive real eigenvalues
 * @returns log(A)
 */
export function logm(A: Tensor): Tensor {
  return matrixFunction(A, Math.log, "logm");
}

/**
 * Compute the matrix square root sqrt(A).
 *
 * Requires all eigenvalues to be real and non-negative.
 *
 * @param A - Square matrix with non-negative real eigenvalues
 * @returns sqrt(A)
 */
export function sqrtm(A: Tensor): Tensor {
  return matrixFunction(A, Math.sqrt, "sqrtm");
}

// ---- Special matrix constructors ----

/**
 * Create a Hilbert matrix of size n.
 *
 * H[i,j] = 1 / (i + j + 1)
 *
 * @param n - Size of the matrix
 * @returns Hilbert matrix of shape (n, n)
 */
export function hilbert(n: number): Tensor {
  if (!Number.isInteger(n) || n <= 0) {
    throw new InvalidParameterError("hilbert: n must be a positive integer", "n", n);
  }
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push(1 / (i + j + 1));
    }
    data.push(row);
  }
  return tensor(data);
}

/**
 * Create a Toeplitz matrix from first column and optional first row.
 *
 * A Toeplitz matrix has constant diagonals. If only c is given, the
 * result is symmetric. If both c and r are given, c defines the first
 * column and r defines the first row (r[0] is ignored in favor of c[0]).
 *
 * @param c - First column
 * @param r - First row (optional; defaults to c for symmetric Toeplitz)
 * @returns Toeplitz matrix
 */
export function toeplitz(c: number[], r?: number[]): Tensor {
  if (c.length === 0) {
    throw new InvalidParameterError("toeplitz: c must have at least one element", "c", c);
  }
  const row = r ?? c;
  const n = c.length;
  const m = row.length;
  // Fill a typed buffer directly (building nested arrays and re-validating
  // every element made large matrices ~100x slower).
  const dtype = getDtype();
  if (dtype === "float32" || dtype === "float64") {
    const out = dtype === "float32" ? new Float32Array(n * m) : new Float64Array(n * m);
    // Row i is the window buf[n-1-i .. n-1-i+m) of the combined sequence
    // [reversed c[1..], row], so each row is one memcpy instead of a
    // per-element diagonal branch.
    const buf = dtype === "float32" ? new Float32Array(n - 1 + m) : new Float64Array(n - 1 + m);
    for (let i = 1; i < n; i++) buf[n - 1 - i] = c[i] ?? 0;
    for (let j = 0; j < m; j++) buf[n - 1 + j] = (j === 0 ? c[0] : row[j]) ?? 0;
    for (let i = 0; i < n; i++) {
      out.set(buf.subarray(n - 1 - i, n - 1 - i + m), i * m);
    }
    return TensorClass.fromTypedArray({ data: out, shape: [n, m], dtype, device: getDevice() });
  }
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const rowData: number[] = [];
    for (let j = 0; j < m; j++) {
      if (i <= j) {
        rowData.push(row[j - i] ?? 0);
      } else {
        rowData.push(c[i - j] ?? 0);
      }
    }
    data.push(rowData);
  }
  return tensor(data);
}

/**
 * Create a Vandermonde matrix.
 *
 * V[i,j] = x[i]^j (increasing=true) or V[i,j] = x[i]^(N-1-j) (default)
 *
 * @param x - Input vector of length n
 * @param N - Number of columns (default: n)
 * @param increasing - If true, powers increase left to right
 * @returns Vandermonde matrix of shape (n, N)
 */
export function vandermonde(x: number[], N?: number, increasing?: boolean): Tensor {
  if (x.length === 0) {
    throw new InvalidParameterError("vandermonde: x must have at least one element", "x", x);
  }
  const n = x.length;
  const cols = N ?? n;
  if (cols <= 0) {
    throw new InvalidParameterError("vandermonde: N must be positive", "N", cols);
  }
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    const xi = x[i] ?? 0;
    for (let j = 0; j < cols; j++) {
      const power = increasing ? j : cols - 1 - j;
      row.push(xi ** power);
    }
    data.push(row);
  }
  return tensor(data);
}

/**
 * Create a Hadamard matrix of order n.
 *
 * n must be a power of 2. Uses Sylvester's construction.
 *
 * @param n - Order of the matrix (must be power of 2)
 * @returns Hadamard matrix of shape (n, n) with entries +1/-1
 */
export function hadamard(n: number): Tensor {
  if (!Number.isInteger(n) || n <= 0 || (n & (n - 1)) !== 0) {
    throw new InvalidParameterError("hadamard: n must be a positive power of 2", "n", n);
  }
  // Start with H_1 = [[1]]
  let h: number[][] = [[1]];
  let size = 1;
  while (size < n) {
    const newH: number[][] = [];
    for (let i = 0; i < size; i++) {
      const topRow: number[] = [];
      const hRow = h[i]!;
      for (let j = 0; j < size; j++) topRow.push(hRow[j]!);
      for (let j = 0; j < size; j++) topRow.push(hRow[j]!);
      newH.push(topRow);
    }
    for (let i = 0; i < size; i++) {
      const botRow: number[] = [];
      const hRow = h[i]!;
      for (let j = 0; j < size; j++) botRow.push(hRow[j]!);
      for (let j = 0; j < size; j++) botRow.push(-hRow[j]!);
      newH.push(botRow);
    }
    h = newH;
    size *= 2;
  }
  return tensor(h);
}

/**
 * Create a companion matrix from polynomial coefficients.
 *
 * The companion matrix of the polynomial
 *   p(x) = c[0]*x^n + c[1]*x^(n-1) + ... + c[n]
 * is the n×n matrix with ones on the sub-diagonal and
 * -c[1..n]/c[0] in the last column (or first row, depending on convention).
 *
 * @param c - Polynomial coefficients (leading coefficient first), length >= 2
 * @returns Companion matrix of shape (n-1, n-1)
 */
export function companion(c: number[]): Tensor {
  if (c.length < 2) {
    throw new InvalidParameterError("companion: c must have at least 2 elements", "c", c);
  }
  const leading = c[0] ?? 1;
  if (leading === 0) {
    throw new InvalidParameterError(
      "companion: leading coefficient must be non-zero",
      "c[0]",
      leading
    );
  }
  const n = c.length - 1;
  const data: number[][] = [];
  // First row: -c[1..n] / c[0]
  const firstRow: number[] = [];
  for (let j = 0; j < n; j++) {
    firstRow.push(-(c[j + 1] ?? 0) / leading);
  }
  data.push(firstRow);
  // Remaining rows: identity shifted
  for (let i = 1; i < n; i++) {
    const row: number[] = new Array(n).fill(0);
    row[i - 1] = 1;
    data.push(row);
  }
  return tensor(data);
}

/**
 * Create a circulant matrix from a vector.
 *
 * Each row is a cyclic permutation of the input vector.
 *
 * @param c - First column of the circulant matrix
 * @returns Circulant matrix of shape (n, n)
 */
export function circulant(c: number[]): Tensor {
  if (c.length === 0) {
    throw new InvalidParameterError("circulant: c must have at least one element", "c", c);
  }
  const n = c.length;
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push(c[(i - j + n) % n] ?? 0);
    }
    data.push(row);
  }
  return tensor(data);
}

/**
 * Create a Hankel matrix from first column and optional last row.
 *
 * H[i,j] = c[i+j] for i+j < len(c), otherwise from r.
 *
 * @param c - First column
 * @param r - Last row (optional; defaults to zeros)
 * @returns Hankel matrix
 */
export function hankel(c: number[], r?: number[]): Tensor {
  if (c.length === 0) {
    throw new InvalidParameterError("hankel: c must have at least one element", "c", c);
  }
  const n = c.length;
  const m = r ? r.length : n;
  // Build the full sequence of values
  const vals: number[] = [...c];
  if (r) {
    for (let i = 1; i < r.length; i++) {
      vals.push(r[i] ?? 0);
    }
  } else {
    for (let i = 1; i < n; i++) {
      vals.push(0);
    }
  }
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      row.push(vals[i + j] ?? 0);
    }
    data.push(row);
  }
  return tensor(data);
}
