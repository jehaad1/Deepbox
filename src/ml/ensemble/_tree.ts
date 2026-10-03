/**
 * Flat copies of fitted regression trees, shared by the boosting and bagging ensembles.
 *
 * @module ml/ensemble/_tree
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */

import { DeepboxError, NotFittedError } from "../../core";
import type { DecisionTreeRegressor } from "../tree/DecisionTree";

/** Root node of a fitted {@link DecisionTreeRegressor}. */
export type RegressionTreeNode = NonNullable<DecisionTreeRegressor["tree_"]>;

/**
 * A fitted regression tree flattened into typed arrays. Prediction then needs no
 * Tensor allocation, keeps the leaf values in double precision, and exposes the
 * leaf index of every row, which the per-loss leaf updates need. Nodes are numbered in
 * pre-order and leaves in depth-first order, left subtree first.
 *
 * @internal
 */
export type FlatTree = {
  readonly feature: Int32Array;
  readonly threshold: Float64Array;
  readonly left: Int32Array;
  readonly right: Int32Array;
  readonly leafIndex: Int32Array;
  readonly leafValue: Float64Array;
};

/**
 * Flatten a tree without recursion, so trees grown with `maxDepth: Infinity` on data that
 * peels off one sample per level cannot overflow the call stack.
 *
 * @internal
 */
export function flattenTree(root: RegressionTreeNode): FlatTree {
  const feature: number[] = [];
  const threshold: number[] = [];
  const left: number[] = [];
  const right: number[] = [];
  const leafIndex: number[] = [];
  const leafValue: number[] = [];

  type Job = { node: RegressionTreeNode; parent: number; isLeft: boolean };
  const stack: Job[] = [{ node: root, parent: -1, isLeft: true }];
  for (let job = stack.pop(); job !== undefined; job = stack.pop()) {
    const { node, parent, isLeft } = job;
    const id = feature.length;
    if (parent >= 0) (isLeft ? left : right)[parent] = id;
    feature.push(0);
    threshold.push(0);
    left.push(-1);
    right.push(-1);
    leafIndex.push(-1);
    if (node.isLeaf) {
      leafIndex[id] = leafValue.length;
      leafValue.push(node.prediction ?? 0);
      continue;
    }
    if (!node.left || !node.right) {
      throw new DeepboxError("Corrupted tree: internal node is missing a child");
    }
    feature[id] = node.featureIndex ?? 0;
    threshold[id] = node.threshold ?? 0;
    // Pushed right first so the left subtree is numbered first.
    stack.push({ node: node.right, parent: id, isLeft: false });
    stack.push({ node: node.left, parent: id, isLeft: true });
  }
  return {
    feature: Int32Array.from(feature),
    threshold: Float64Array.from(threshold),
    left: Int32Array.from(left),
    right: Int32Array.from(right),
    leafIndex: Int32Array.from(leafIndex),
    leafValue: Float64Array.from(leafValue),
  };
}

/**
 * Write the leaf index of each of the `n` rows of `x` (row-major, `d` columns) into `out`.
 *
 * @internal
 */
export function applyTree(
  f: FlatTree,
  x: Float64Array,
  n: number,
  d: number,
  out: Int32Array
): void {
  const { feature, threshold, left, right, leafIndex } = f;
  for (let i = 0; i < n; i++) {
    const base = i * d;
    let node = 0;
    while ((left[node] as number) !== -1) {
      node =
        (x[base + (feature[node] as number)] as number) <= (threshold[node] as number)
          ? (left[node] as number)
          : (right[node] as number);
    }
    out[i] = leafIndex[node] as number;
  }
}

/**
 * Predictions of a fitted regression tree for the `n` rows of `x` (row-major, `d` columns),
 * in double precision.
 *
 * @internal Shared by the regressors in this folder; not part of the public API.
 */
export function predictRegressionTree(
  tree: DecisionTreeRegressor,
  x: Float64Array,
  n: number,
  d: number
): Float64Array {
  const root = tree.tree_;
  if (!root) throw new NotFittedError("DecisionTreeRegressor must be fitted before prediction");
  const flat = flattenTree(root);
  const leaf = new Int32Array(n);
  applyTree(flat, x, n, d, leaf);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = flat.leafValue[leaf[i] as number] as number;
  return out;
}
