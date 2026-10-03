/**
 * Isomap: Isometric Mapping for nonlinear dimensionality reduction.
 *
 * Computes geodesic distances via a k-nearest-neighbor graph and
 * then applies classical MDS (multidimensional scaling) to embed
 * the data in a lower-dimensional space.
 *
 * @module ml/manifold/Isomap
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, warn } from "../../core";
import { fromDenseMatrix2D } from "../../linalg/_internal";
import { eigh } from "../../linalg/decomposition/eig";
import type { Tensor } from "../../ndarray";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";

/**
 * Pairwise Euclidean distances of the rows of a row-major `n x d` matrix.
 *
 * @returns Symmetric `n x n` matrix with a zero diagonal
 * @internal
 */
export function pairwiseEuclidean(flat: Float64Array, n: number, d: number): Float64Array {
  const dist = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const bi = i * d;
    for (let j = i + 1; j < n; j++) {
      const bj = j * d;
      let sq = 0;
      for (let f = 0; f < d; f++) {
        const diff = (flat[bi + f] as number) - (flat[bj + f] as number);
        sq += diff * diff;
      }
      const v = Math.sqrt(sq);
      dist[i * n + j] = v;
      dist[j * n + i] = v;
    }
  }
  return dist;
}

/**
 * Result of the eigendecomposition used by classical MDS.
 *
 * @internal
 */
export type ClassicalMDSResult = {
  /** Embedding of shape `n x nComponents`, row-major. */
  readonly embedding: Float64Array;
  /** Leading eigenvalues of the double-centred matrix, in descending order. */
  readonly eigenvalues: Float64Array;
  /** Matching unit eigenvectors, `n x nComponents`, row-major. */
  readonly eigenvectors: Float64Array;
};

/**
 * Eigendecomposition behind classical (Torgerson) MDS.
 *
 * Double-centres the squared distances (`B = -1/2 H D^2 H`), takes the
 * `nComponents` largest eigenpairs of `B` with a dense symmetric solver and
 * scales each eigenvector by the square root of its eigenvalue. Components
 * whose eigenvalue is not positive (non-Euclidean input) or is no larger than
 * rounding noise (rank-deficient input) are returned as zeros.
 * The sign of each eigenvector is fixed so that its largest-magnitude entry is
 * positive, which makes the result deterministic.
 *
 * @internal
 */
export function classicalMDSDecompose(
  distMatrix: Float64Array,
  n: number,
  nComponents: number
): ClassicalMDSResult {
  if (!Number.isInteger(nComponents) || nComponents < 1 || nComponents > n) {
    throw new InvalidParameterError(
      `nComponents must be an integer in [1, n_samples=${n}]; received ${nComponents}`,
      "nComponents",
      nComponents
    );
  }
  if (distMatrix.length !== n * n) {
    throw new InvalidParameterError(
      `distance matrix must contain n * n = ${n * n} entries; received ${distMatrix.length}`,
      "distMatrix",
      distMatrix.length
    );
  }

  // Row means of the squared distances (the matrix is symmetric, so the
  // column means are identical).
  const D2 = new Float64Array(n * n);
  const rowMean = new Float64Array(n);
  let grandMean = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      const d = distMatrix[i * n + j] as number;
      const sq = d * d;
      D2[i * n + j] = sq;
      s += sq;
    }
    rowMean[i] = s / n;
    grandMean += s / n;
  }
  grandMean /= n;

  // B is symmetrised explicitly so that rounding noise in the input cannot
  // trip the solver's symmetry check.
  const B = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const bij = -0.5 * (D2[i * n + j] as number);
      const bji = -0.5 * (D2[j * n + i] as number);
      const centred =
        0.5 * (bij + bji) +
        0.5 * ((rowMean[i] as number) + (rowMean[j] as number)) -
        0.5 * grandMean;
      B[i * n + j] = centred;
      B[j * n + i] = centred;
    }
  }

  const [values, vectors] = eigh(fromDenseMatrix2D(n, n, B));
  const vals = values.data as Float64Array;
  const vecs = vectors.data as Float64Array;
  const vOff = vectors.offset;
  const vStride0 = vectors.strides[0] ?? n;
  const vStride1 = vectors.strides[1] ?? 1;

  // Eigenvalues at rounding-noise level (data of lower rank than nComponents) would
  // otherwise become columns of ~1e-8 noise and amplify errors in `transform`.
  let largest = 0;
  for (let i = 0; i < n; i++)
    largest = Math.max(largest, Math.abs(vals[values.offset + i] as number));
  const noiseFloor = n * Number.EPSILON * largest;

  const embedding = new Float64Array(n * nComponents);
  const eigenvalues = new Float64Array(nComponents);
  const eigenvectors = new Float64Array(n * nComponents);
  for (let r = 0; r < nComponents; r++) {
    const col = n - 1 - r; // eigh returns ascending eigenvalues
    const raw = vals[values.offset + col] as number;
    const lambda = raw > noiseFloor ? raw : Math.min(raw, 0);
    eigenvalues[r] = lambda;

    let maxAbs = 0;
    let sign = 1;
    for (let i = 0; i < n; i++) {
      const v = vecs[vOff + i * vStride0 + col * vStride1] as number;
      if (Math.abs(v) > maxAbs) {
        maxAbs = Math.abs(v);
        sign = v < 0 ? -1 : 1;
      }
    }
    const scale = lambda > 0 ? Math.sqrt(lambda) : 0;
    for (let i = 0; i < n; i++) {
      const v = sign * (vecs[vOff + i * vStride0 + col * vStride1] as number);
      eigenvectors[i * nComponents + r] = v;
      embedding[i * nComponents + r] = scale === 0 ? 0 : v * scale;
    }
  }
  return { embedding, eigenvalues, eigenvectors };
}

/**
 * Classical (metric) Multidimensional Scaling.
 *
 * Given a distance matrix, embeds points in `nComponents` dimensions
 * using the eigendecomposition of the double-centred squared-distance
 * matrix.
 *
 * @param distMatrix - Row-major `n x n` matrix of pairwise distances
 * @param n - Number of points
 * @param nComponents - Embedding dimension, between 1 and `n`
 * @returns Embedding of shape `[n, nComponents]`
 * @throws {InvalidParameterError} If `nComponents` is not an integer in `[1, n]` or the matrix size is not `n * n`
 */
export function classicalMDS(distMatrix: Float64Array, n: number, nComponents: number): Tensor {
  const { embedding } = classicalMDSDecompose(distMatrix, n, nComponents);
  return fromDenseMatrix2D(n, nComponents, embedding);
}

/**
 * Binary min-heap of (key, node) pairs used by the Dijkstra search.
 */
class MinHeap {
  private keys: Float64Array;
  private nodes: Int32Array;
  size = 0;

  constructor(capacity: number) {
    this.keys = new Float64Array(Math.max(4, capacity));
    this.nodes = new Int32Array(Math.max(4, capacity));
  }

  clear(): void {
    this.size = 0;
  }

  push(key: number, node: number): void {
    if (this.size === this.keys.length) {
      const keys = new Float64Array(this.size * 2);
      const nodes = new Int32Array(this.size * 2);
      keys.set(this.keys);
      nodes.set(this.nodes);
      this.keys = keys;
      this.nodes = nodes;
    }
    let i = this.size++;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if ((this.keys[parent] as number) <= key) break;
      this.keys[i] = this.keys[parent] as number;
      this.nodes[i] = this.nodes[parent] as number;
      i = parent;
    }
    this.keys[i] = key;
    this.nodes[i] = node;
  }

  /** Remove the smallest entry; read it back from {@link topKey} / {@link topNode} first. */
  get topKey(): number {
    return this.keys[0] as number;
  }

  get topNode(): number {
    return this.nodes[0] as number;
  }

  pop(): void {
    const last = --this.size;
    if (last === 0) return;
    const key = this.keys[last] as number;
    const node = this.nodes[last] as number;
    let i = 0;
    for (;;) {
      const left = 2 * i + 1;
      if (left >= last) break;
      const right = left + 1;
      const child =
        right < last && (this.keys[right] as number) < (this.keys[left] as number) ? right : left;
      if ((this.keys[child] as number) >= key) break;
      this.keys[i] = this.keys[child] as number;
      this.nodes[i] = this.nodes[child] as number;
      i = child;
    }
    this.keys[i] = key;
    this.nodes[i] = node;
  }
}

/**
 * Isomap: nonlinear dimensionality reduction through geodesic distances.
 *
 * 1. Connects every sample to its `nNeighbors` nearest neighbors (the graph is
 *    made symmetric; separate components are joined through their closest points
 *    and a warning is issued).
 * 2. Computes all-pairs shortest-path distances on that graph (Dijkstra).
 * 3. Embeds the geodesic distances with classical MDS.
 *
 * Unlike t-SNE, Isomap can project new samples with `transform`.
 *
 * @example
 * ```ts
 * import { Isomap } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [1, 0.1], [2, 0.3], [3, 0.2], [4, 0.1], [5, 0]]);
 * const iso = new Isomap({ nComponents: 1, nNeighbors: 2 });
 * const embedding = iso.fitTransform(X); // shape [6, 1]
 * const projected = iso.transform(tensor([[2.5, 0.25]]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-manifold | Deepbox Manifold Learning}
 * @see Tenenbaum, de Silva, Langford (2000). "A Global Geometric Framework for Nonlinear Dimensionality Reduction"
 */
export class Isomap {
  private nComponents: number;
  private nNeighbors: number;
  private embedding_?: Tensor;
  private fitted = false;

  // State needed to project new samples.
  private trainX_?: Float64Array;
  private geodesic_?: Float64Array;
  private kernelColMean_?: Float64Array;
  private kernelGrandMean_ = 0;
  private eigenvalues_?: Float64Array;
  private eigenvectors_?: Float64Array;
  private nSamples_ = 0;
  private nFeaturesIn_ = 0;
  // Hyperparameters in effect at fit time; `setParams` may change the live ones afterwards.
  private fitNComponents_ = 0;
  private fitNNeighbors_ = 0;

  /**
   * @param options.nComponents - Dimension of the embedding (default: 2)
   * @param options.nNeighbors - Number of nearest neighbors used to build the graph (default: 5)
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly nNeighbors?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.nNeighbors = options.nNeighbors ?? 5;
    this.validateParams();
  }

  private validateParams(): void {
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1",
        "nComponents",
        this.nComponents
      );
    }
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new InvalidParameterError(
        "nNeighbors must be an integer >= 1",
        "nNeighbors",
        this.nNeighbors
      );
    }
  }

  /**
   * Learn the embedding of the training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @throws {InvalidParameterError} If `nNeighbors >= n_samples` or `nComponents > n_samples`
   */
  fit(X: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;

    if (this.nNeighbors >= n) {
      throw new InvalidParameterError(
        `nNeighbors must be < n_samples (${n})`,
        "nNeighbors",
        this.nNeighbors
      );
    }
    if (this.nComponents > n) {
      throw new InvalidParameterError(
        `nComponents must be <= n_samples (${n})`,
        "nComponents",
        this.nComponents
      );
    }

    const flat = toFloat64View(X);
    const dist = pairwiseEuclidean(flat, n, nF);
    const k = this.nNeighbors;

    // Step 1: symmetric k-NN graph. `edge` marks adjacency; weights come from `dist`.
    const edge = new Uint8Array(n * n);
    const bestD = new Float64Array(k);
    const bestI = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      let count = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const d = dist[i * n + j] as number;
        if (count === k && d >= (bestD[k - 1] as number)) continue;
        // Insert (d, j) into the ascending list; ties keep the lower index first.
        let pos = count < k ? count : k - 1;
        while (pos > 0 && (bestD[pos - 1] as number) > d) {
          bestD[pos] = bestD[pos - 1] as number;
          bestI[pos] = bestI[pos - 1] as number;
          pos--;
        }
        bestD[pos] = d;
        bestI[pos] = j;
        if (count < k) count++;
      }
      for (let t = 0; t < count; t++) {
        const j = bestI[t] as number;
        edge[i * n + j] = 1;
        edge[j * n + i] = 1;
      }
    }

    // A k-NN graph can fall apart into several components (for example two well
    // separated clusters). Join every pair of components through its closest pair
    // of points, as scikit-learn does, so that geodesic distances stay finite.
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x: number): number => {
      let root = x;
      while ((parent[root] as number) !== root) root = parent[root] as number;
      while ((parent[x] as number) !== root) {
        const next = parent[x] as number;
        parent[x] = root;
        x = next;
      }
      return root;
    };
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (edge[i * n + j] === 1) parent[find(i)] = find(j);
      }
    }
    const compId = new Int32Array(n).fill(-1);
    let nComp = 0;
    for (let i = 0; i < n; i++) {
      const root = find(i);
      if (compId[root] === -1) compId[root] = nComp++;
    }
    if (nComp > 1) {
      const bestLink = new Float64Array(nComp * nComp).fill(Infinity);
      const linkI = new Int32Array(nComp * nComp);
      const linkJ = new Int32Array(nComp * nComp);
      for (let i = 0; i < n; i++) {
        const ci = compId[find(i)] as number;
        for (let j = i + 1; j < n; j++) {
          const cj = compId[find(j)] as number;
          if (ci === cj) continue;
          const slot = Math.min(ci, cj) * nComp + Math.max(ci, cj);
          const d = dist[i * n + j] as number;
          if (d < (bestLink[slot] as number)) {
            bestLink[slot] = d;
            linkI[slot] = i;
            linkJ[slot] = j;
          }
        }
      }
      for (let a = 0; a < nComp; a++) {
        for (let b = a + 1; b < nComp; b++) {
          const i = linkI[a * nComp + b] as number;
          const j = linkJ[a * nComp + b] as number;
          edge[i * n + j] = 1;
          edge[j * n + i] = 1;
        }
      }
      warn(
        `Isomap: the ${k}-nearest-neighbor graph has ${nComp} connected components; they were ` +
          "joined through their closest points. Increase nNeighbors to avoid this.",
        "UserWarning",
        "Isomap"
      );
    }

    const adjStart = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) {
      let deg = 0;
      for (let j = 0; j < n; j++) deg += edge[i * n + j] as number;
      adjStart[i + 1] = (adjStart[i] as number) + deg;
    }
    const adjNode = new Int32Array(adjStart[n] as number);
    const adjWeight = new Float64Array(adjStart[n] as number);
    for (let i = 0; i < n; i++) {
      let p = adjStart[i] as number;
      for (let j = 0; j < n; j++) {
        if (edge[i * n + j] === 1) {
          adjNode[p] = j;
          adjWeight[p] = dist[i * n + j] as number;
          p++;
        }
      }
    }

    // Step 2: all-pairs shortest paths, one Dijkstra run per source.
    const geodesic = new Float64Array(n * n).fill(Infinity);
    const heap = new MinHeap(adjNode.length);
    for (let s = 0; s < n; s++) {
      const base = s * n;
      geodesic[base + s] = 0;
      heap.clear();
      heap.push(0, s);
      let reached = 0;
      while (heap.size > 0) {
        const du = heap.topKey;
        const u = heap.topNode;
        heap.pop();
        if (du > (geodesic[base + u] as number)) continue;
        reached++;
        for (let p = adjStart[u] as number; p < (adjStart[u + 1] as number); p++) {
          const v = adjNode[p] as number;
          const nd = du + (adjWeight[p] as number);
          if (nd < (geodesic[base + v] as number)) {
            geodesic[base + v] = nd;
            heap.push(nd, v);
          }
        }
      }
      if (reached < n) {
        // Unreachable after the components were joined; kept so that Infinity can
        // never reach the double centering step (it would turn into NaN).
        throw new DataValidationError("Isomap: the neighborhood graph is disconnected");
      }
    }

    // Step 3: classical MDS on the geodesic distances.
    const mds = classicalMDSDecompose(geodesic, n, this.nComponents);

    // Statistics of the kernel K = -1/2 G^2 that the out-of-sample projection needs.
    const kernelColMean = new Float64Array(n);
    let grand = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const g = geodesic[i * n + j] as number;
        kernelColMean[j] = (kernelColMean[j] as number) - (0.5 * g * g) / n;
      }
    }
    for (let j = 0; j < n; j++) grand += (kernelColMean[j] as number) / n;

    this.trainX_ = Float64Array.from(flat);
    this.geodesic_ = geodesic;
    this.kernelColMean_ = kernelColMean;
    this.kernelGrandMean_ = grand;
    this.eigenvalues_ = mds.eigenvalues;
    this.eigenvectors_ = mds.eigenvectors;
    this.nSamples_ = n;
    this.nFeaturesIn_ = nF;
    this.fitNComponents_ = this.nComponents;
    this.fitNNeighbors_ = k;
    this.embedding_ = fromDenseMatrix2D(n, this.nComponents, mds.embedding);
    this.fitted = true;
    return this;
  }

  /**
   * Fit the model and return the embedding of the training data.
   *
   * @returns Embedding of shape (n_samples, nComponents)
   */
  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.embedding;
  }

  /**
   * Project new samples into the learned embedding.
   *
   * Each sample is connected to its `nNeighbors` nearest training samples, its
   * geodesic distance to every training sample is the shortest path through one
   * of those neighbors, and the distances are mapped with the same kernel
   * centering and eigenvectors as the training data. Transforming the training
   * data reproduces `embedding`.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Embedding of shape (n_samples, nComponents)
   * @throws {NotFittedError} If the model has not been fitted
   */
  transform(X: Tensor): Tensor {
    if (
      !this.fitted ||
      !this.trainX_ ||
      !this.geodesic_ ||
      !this.kernelColMean_ ||
      !this.eigenvalues_ ||
      !this.eigenvectors_
    ) {
      throw new NotFittedError("Isomap must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "Isomap");
    const m = X.shape[0] ?? 0;
    const flat = toFloat64View(X);
    const n = this.nSamples_;
    const d = this.nFeaturesIn_;
    const k = this.fitNNeighbors_;
    const nc = this.fitNComponents_;
    const train = this.trainX_;
    const G = this.geodesic_;
    const colMean = this.kernelColMean_;
    const out = new Float64Array(m * nc);

    const bestD = new Float64Array(k);
    const bestI = new Int32Array(k);
    const kernelRow = new Float64Array(n);
    for (let r = 0; r < m; r++) {
      let count = 0;
      for (let j = 0; j < n; j++) {
        let sq = 0;
        for (let f = 0; f < d; f++) {
          const diff = (flat[r * d + f] as number) - (train[j * d + f] as number);
          sq += diff * diff;
        }
        const dj = Math.sqrt(sq);
        if (count === k && dj >= (bestD[k - 1] as number)) continue;
        let pos = count < k ? count : k - 1;
        while (pos > 0 && (bestD[pos - 1] as number) > dj) {
          bestD[pos] = bestD[pos - 1] as number;
          bestI[pos] = bestI[pos - 1] as number;
          pos--;
        }
        bestD[pos] = dj;
        bestI[pos] = j;
        if (count < k) count++;
      }

      let rowMean = 0;
      for (let j = 0; j < n; j++) {
        let g = Infinity;
        for (let t = 0; t < count; t++) {
          const via = (bestD[t] as number) + (G[(bestI[t] as number) * n + j] as number);
          if (via < g) g = via;
        }
        const kv = -0.5 * g * g;
        kernelRow[j] = kv;
        rowMean += kv / n;
      }
      for (let j = 0; j < n; j++) {
        kernelRow[j] =
          (kernelRow[j] as number) - rowMean - (colMean[j] as number) + this.kernelGrandMean_;
      }
      for (let c = 0; c < nc; c++) {
        const lambda = this.eigenvalues_[c] as number;
        if (!(lambda > 0)) continue; // zero column, as in the training embedding
        let s = 0;
        for (let j = 0; j < n; j++) {
          s += (kernelRow[j] as number) * (this.eigenvectors_[j * nc + c] as number);
        }
        out[r * nc + c] = s / Math.sqrt(lambda);
      }
    }
    return fromDenseMatrix2D(m, nc, out);
  }

  /**
   * Embedding of the training data, shape (n_samples, nComponents).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get embedding(): Tensor {
    if (!this.fitted || !this.embedding_) {
      throw new NotFittedError("Isomap must be fitted before accessing embedding");
    }
    return this.embedding_;
  }

  /**
   * Geodesic (shortest-path) distances between the training samples,
   * shape (n_samples, n_samples).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get distMatrix(): Tensor {
    if (!this.fitted || !this.geodesic_) {
      throw new NotFittedError("Isomap must be fitted before accessing distMatrix");
    }
    return fromDenseMatrix2D(this.nSamples_, this.nSamples_, Float64Array.from(this.geodesic_));
  }

  /**
   * Number of features seen during `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("Isomap must be fitted before accessing nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  getParams(): Record<string, unknown> {
    return { nComponents: this.nComponents, nNeighbors: this.nNeighbors };
  }

  /**
   * Update hyperparameters. The model must be refitted afterwards.
   *
   * @throws {InvalidParameterError} On an unknown or invalid parameter
   */
  setParams(params: Record<string, unknown>): this {
    let nComponents = this.nComponents;
    let nNeighbors = this.nNeighbors;
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          nComponents = value as number;
          break;
        case "nNeighbors":
          nNeighbors = value as number;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    const prev = [this.nComponents, this.nNeighbors] as const;
    this.nComponents = nComponents;
    this.nNeighbors = nNeighbors;
    try {
      this.validateParams();
    } catch (e) {
      this.nComponents = prev[0];
      this.nNeighbors = prev[1];
      throw e;
    }
    return this;
  }
}
