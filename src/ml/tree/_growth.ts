/**
 * Growth controls shared by the tree estimators: sample and class weights, the options that
 * stop or limit growth (`minImpurityDecrease`, `maxLeafNodes`, `ccpAlpha`), best-first growth
 * and minimal cost-complexity pruning.
 *
 * The helper functions are internal; only the option types are exported from `deepbox/ml`.
 *
 * @module ml/tree/_growth
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */

import { DataValidationError, InvalidParameterError, ShapeError } from "../../core";
import type { Tensor } from "../../ndarray";
import { toFloat64View } from "../_validation";
import type { MutableTreeNode, TreeNode } from "./DecisionTree";

/**
 * Class weights of a classification tree: `"balanced"` or a map from class label to weight.
 *
 * `"balanced"` gives class `k` the weight `n_samples / (n_classes * count_k)`. With a map,
 * classes that are not listed keep the weight 1, and every key must be a class of `y`.
 * Weights multiply the `sampleWeight` of `fit`.
 *
 * @example
 * ```ts
 * import { DecisionTreeClassifier } from 'deepbox/ml';
 *
 * new DecisionTreeClassifier({ classWeight: 'balanced' });
 * new DecisionTreeClassifier({ classWeight: { 0: 1, 1: 5 } });
 * ```
 */
export type TreeClassWeight = "balanced" | Readonly<Record<number, number>>;

/**
 * Class weights of a forest: the options of {@link TreeClassWeight} plus
 * `"balanced_subsample"`, which computes the `"balanced"` weights again on the bootstrap sample
 * of every tree.
 */
export type ForestClassWeight = TreeClassWeight | "balanced_subsample";

/**
 * Options that control how far a tree grows. They are accepted by the decision trees, random
 * forests, extremely randomized trees and gradient boosting estimators in `deepbox/ml`.
 *
 * @example
 * ```ts
 * import { DecisionTreeRegressor } from 'deepbox/ml';
 *
 * // At most 8 leaves, grown best-first, then pruned with a cost-complexity alpha of 0.01.
 * const tree = new DecisionTreeRegressor({ maxLeafNodes: 8, ccpAlpha: 0.01 });
 * ```
 */
export type TreeGrowthOptions = {
  /**
   * A node is split only if the split lowers the weighted impurity by at least this value:
   * `(W_node / W_total) * (impurity - W_left / W_node * impurity_left - W_right / W_node *
   * impurity_right) >= minImpurityDecrease`, with `W` the sum of the sample weights (the sample
   * count when no weights are given), as scikit-learn defines `min_impurity_decrease`. The
   * impurity is the Gini index, the base-2 entropy or the variance of the targets. Default 0.
   */
  readonly minImpurityDecrease?: number;
  /**
   * Grow the tree best-first (the split with the largest impurity decrease first) until it has
   * this many leaves. An integer >= 2. Default: no limit, and the tree grows depth first.
   */
  readonly maxLeafNodes?: number;
  /**
   * Strength of minimal cost-complexity pruning: the subtrees with the smallest effective
   * `alpha` are cut away as long as that alpha is not above `ccpAlpha`. A finite number >= 0.
   * Default 0 (no pruning).
   */
  readonly ccpAlpha?: number;
};

/** Smallest relative gain that still counts, `numpy.finfo(float64).eps`. */
const EPSILON = 2.220446049250313e-16;

/** @internal */
export function checkMinImpurityDecrease(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError(
      `minImpurityDecrease must be a finite number >= 0; received ${String(value)}`,
      "minImpurityDecrease",
      value
    );
  }
  return value;
}

/** @internal */
export function checkMaxLeafNodes(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
    throw new InvalidParameterError(
      `maxLeafNodes must be an integer >= 2 or undefined; received ${String(value)}`,
      "maxLeafNodes",
      value
    );
  }
  return value;
}

/** @internal */
export function checkCcpAlpha(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError(
      `ccpAlpha must be a finite number >= 0; received ${String(value)}`,
      "ccpAlpha",
      value
    );
  }
  return value;
}

/** @internal */
export type ClassWeightKind = "balanced" | "balanced_subsample" | "map";

/**
 * Validate a `classWeight` option. Returns `undefined` for "not set" and a defensive copy of a
 * weight map.
 *
 * @internal
 */
export function checkClassWeight(
  value: unknown,
  allowSubsample: boolean
): ForestClassWeight | undefined {
  if (value === undefined || value === null) return undefined;
  if (value === "balanced" || (allowSubsample && value === "balanced_subsample")) return value;
  if (typeof value === "object" && !Array.isArray(value)) {
    const copy: Record<number, number> = {};
    for (const [key, weight] of Object.entries(value as Record<string, unknown>)) {
      if (key.trim() === "" || !Number.isFinite(Number(key))) {
        throw new InvalidParameterError(
          `classWeight keys must be numeric class labels; received '${key}'`,
          "classWeight",
          value
        );
      }
      if (typeof weight !== "number" || !Number.isFinite(weight) || weight < 0) {
        throw new InvalidParameterError(
          `classWeight values must be finite numbers >= 0; received ${String(weight)} for class ${key}`,
          "classWeight",
          value
        );
      }
      copy[Number(key)] = weight;
    }
    return copy;
  }
  throw new InvalidParameterError(
    `classWeight must be "balanced"${allowSubsample ? ', "balanced_subsample"' : ""} or an object mapping class labels to weights; received ${String(value)}`,
    "classWeight",
    value
  );
}

/**
 * Read and validate the `sampleWeight` argument of `fit`. Returns `undefined` when it is not
 * given. The result must not be modified.
 *
 * @throws {ShapeError} If it is not 1-dimensional or its length is not `n`
 * @throws {DataValidationError} If a weight is negative or not finite, or all weights are zero
 *
 * @internal
 */
export function readSampleWeight(
  sampleWeight: Tensor | undefined,
  n: number
): Float64Array | undefined {
  if (sampleWeight === undefined || sampleWeight === null) return undefined;
  if (sampleWeight.ndim !== 1) {
    throw new ShapeError(`sampleWeight must be 1-dimensional; got ndim=${sampleWeight.ndim}`);
  }
  if (sampleWeight.size !== n) {
    throw new ShapeError(
      `sampleWeight must have one entry per sample; got ${sampleWeight.size} for ${n} samples`
    );
  }
  const values = toFloat64View(sampleWeight, "sampleWeight");
  let total = 0;
  for (let i = 0; i < n; i++) {
    const w = values[i] as number;
    if (!Number.isFinite(w) || w < 0) {
      throw new DataValidationError("sampleWeight must contain finite values >= 0");
    }
    total += w;
  }
  if (!(total > 0)) {
    throw new DataValidationError("sampleWeight must contain at least one positive value");
  }
  return values;
}

/**
 * Weight of every class for a `classWeight` setting. `counts[k]` is the (unweighted) number of
 * samples of class `k` among the `n` samples of the training set; `labels[k]` is its label.
 * `"balanced_subsample"` is resolved by the caller (it behaves as `"balanced"` on the sample it
 * is given).
 *
 * @throws {InvalidParameterError} If a key of a weight map is not one of `labels`
 *
 * @internal
 */
export function classWeightPerClass(
  classWeight: ForestClassWeight,
  labels: readonly number[],
  counts: ArrayLike<number>,
  n: number
): Float64Array {
  const k = labels.length;
  const out = new Float64Array(k).fill(1);
  if (classWeight === "balanced" || classWeight === "balanced_subsample") {
    let present = 0;
    for (let c = 0; c < k; c++) if ((counts[c] as number) > 0) present++;
    for (let c = 0; c < k; c++) {
      const count = counts[c] as number;
      out[c] = count > 0 ? n / (present * count) : 1;
    }
    return out;
  }
  const known = new Set(labels);
  for (const key of Object.keys(classWeight)) {
    const label = Number(key);
    if (!known.has(label)) {
      throw new InvalidParameterError(
        `classWeight has a weight for class ${key}, which is not a class of y`,
        "classWeight",
        classWeight
      );
    }
  }
  for (let c = 0; c < k; c++) {
    const weight = (classWeight as Readonly<Record<number, number>>)[labels[c] as number];
    if (weight !== undefined) out[c] = weight;
  }
  return out;
}

/**
 * Per-sample weights: `sampleWeight[i] * classWeight[yCode[i]]`, or `undefined` when neither is
 * given.
 *
 * @internal
 */
export function combineWeights(
  sampleWeight: Float64Array | undefined,
  perClass: Float64Array | undefined,
  yCode: Int32Array
): Float64Array | undefined {
  if (sampleWeight === undefined && perClass === undefined) return undefined;
  const n = yCode.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const base = sampleWeight === undefined ? 1 : (sampleWeight[i] as number);
    out[i] = perClass === undefined ? base : base * (perClass[yCode[i] as number] as number);
  }
  return out;
}

/**
 * Indices of `0..n-1` with a positive weight (all of them without weights). Samples of weight
 * zero take no part in growing a tree, as in scikit-learn.
 *
 * @internal
 */
export function positiveWeightIndices(n: number, weights: Float64Array | undefined): Int32Array {
  if (weights === undefined) {
    const all = new Int32Array(n);
    for (let i = 0; i < n; i++) all[i] = i;
    return all;
  }
  let count = 0;
  for (let i = 0; i < n; i++) if ((weights[i] as number) > 0) count++;
  const out = new Int32Array(count);
  let k = 0;
  for (let i = 0; i < n; i++) if ((weights[i] as number) > 0) out[k++] = i;
  return out;
}

/**
 * Whether a split with weighted impurity decrease `decrease` (`W * impurity` units) is large
 * enough for `minImpurityDecrease`, given the total weight of the training set.
 *
 * @internal
 */
export function splitIsWorthwhile(
  decrease: number,
  totalWeight: number,
  minImpurityDecrease: number
): boolean {
  if (minImpurityDecrease <= 0) return true;
  return decrease / totalWeight + EPSILON >= minImpurityDecrease;
}

// ---------------------------------------------------------------------------
// Best-first growth
// ---------------------------------------------------------------------------

/**
 * Result of expanding one node. For a split node `leaf` gives the leaf the node becomes when it
 * is not expanded (best-first growth) or is pruned (cost-complexity pruning).
 *
 * @internal
 */
export type ExpandResult = {
  node: MutableTreeNode;
  children?: readonly [Int32Array, Int32Array];
  leaf?: () => TreeNode;
};

type FrontierEntry = {
  readonly result: ExpandResult;
  readonly depth: number;
  readonly priority: number;
  readonly order: number;
  readonly attach: (node: TreeNode) => void;
};

/** Max-heap by priority; the entry created first wins a tie. */
class Frontier {
  private readonly items: FrontierEntry[] = [];

  get size(): number {
    return this.items.length;
  }

  private before(a: FrontierEntry, b: FrontierEntry): boolean {
    return a.priority > b.priority || (a.priority === b.priority && a.order < b.order);
  }

  push(entry: FrontierEntry): void {
    const items = this.items;
    let i = items.length;
    items.push(entry);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.before(items[i] as FrontierEntry, items[parent] as FrontierEntry)) break;
      const tmp = items[i] as FrontierEntry;
      items[i] = items[parent] as FrontierEntry;
      items[parent] = tmp;
      i = parent;
    }
  }

  pop(): FrontierEntry | undefined {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (top === undefined || last === undefined) return top;
    if (items.length > 0) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let best = i;
        if (
          l < items.length &&
          this.before(items[l] as FrontierEntry, items[best] as FrontierEntry)
        )
          best = l;
        if (
          r < items.length &&
          this.before(items[r] as FrontierEntry, items[best] as FrontierEntry)
        )
          best = r;
        if (best === i) break;
        const tmp = items[i] as FrontierEntry;
        items[i] = items[best] as FrontierEntry;
        items[best] = tmp;
        i = best;
      }
    }
    return top;
  }
}

/**
 * Best-first growth: the node whose best split lowers the impurity most is expanded first, until
 * the tree has `maxLeafNodes` leaves (scikit-learn's `max_leaf_nodes`). Every node's split is
 * searched when the node is created, left child before right child.
 *
 * @internal
 */
export function growTreeBestFirst(
  rootIndices: Int32Array,
  expand: (indices: Int32Array, depth: number) => ExpandResult,
  maxLeafNodes: number
): TreeNode {
  let root: TreeNode | undefined;
  const frontier = new Frontier();
  let counter = 0;

  const add = (indices: Int32Array, depth: number, attach: (node: TreeNode) => void): void => {
    const result = expand(indices, depth);
    if (result.children === undefined) {
      attach(result.node);
      return;
    }
    attach(result.leaf === undefined ? result.node : result.leaf());
    frontier.push({
      result,
      depth,
      priority: result.node.weightedImpurityDecrease ?? 0,
      order: counter++,
      attach,
    });
  };

  add(rootIndices, 0, (node) => {
    root = node;
  });
  let leaves = 1;
  while (leaves < maxLeafNodes) {
    const entry = frontier.pop();
    if (entry === undefined) break;
    const { node, children } = entry.result;
    if (children === undefined) continue;
    entry.attach(node);
    leaves++;
    add(children[0], entry.depth + 1, (child) => {
      node.left = child;
    });
    add(children[1], entry.depth + 1, (child) => {
      node.right = child;
    });
  }
  return root as unknown as TreeNode;
}

// ---------------------------------------------------------------------------
// Minimal cost-complexity pruning
// ---------------------------------------------------------------------------

/**
 * Minimal cost-complexity pruning (Breiman et al.): while the weakest link, the internal node
 * with the smallest `(R(t) - R(T_t)) / (|leaves(T_t)| - 1)`, has an effective alpha of at most
 * `ccpAlpha`, the subtree below it is replaced by a leaf. `R` is the impurity weighted by the
 * share of the training weight that reaches a node. Needs `impurity`, `weightedNSamples` and,
 * on every split node, `collapsed` (the leaf the node becomes).
 *
 * @internal
 */
export function pruneTree(root: TreeNode, ccpAlpha: number): TreeNode {
  if (root.isLeaf || !(ccpAlpha > 0)) return root;
  const total = root.weightedNSamples ?? 0;
  if (!(total > 0)) return root;

  // Pre-order numbering: a parent always has a smaller id than its children.
  const nodes: TreeNode[] = [];
  const parent: number[] = [];
  const left: number[] = [];
  const right: number[] = [];
  const stack: Array<{ node: TreeNode; parentId: number; isLeft: boolean }> = [
    { node: root, parentId: -1, isLeft: true },
  ];
  for (let job = stack.pop(); job !== undefined; job = stack.pop()) {
    const id = nodes.length;
    nodes.push(job.node);
    parent.push(job.parentId);
    left.push(-1);
    right.push(-1);
    if (job.parentId >= 0) (job.isLeft ? left : right)[job.parentId] = id;
    if (!job.node.isLeaf && job.node.left && job.node.right) {
      stack.push({ node: job.node.right, parentId: id, isLeft: false });
      stack.push({ node: job.node.left, parentId: id, isLeft: true });
    }
  }

  const count = nodes.length;
  const rNode = new Float64Array(count);
  const rBranch = new Float64Array(count);
  const leaves = new Int32Array(count);
  for (let i = count - 1; i >= 0; i--) {
    const node = nodes[i] as TreeNode;
    rNode[i] = ((node.impurity ?? 0) * (node.weightedNSamples ?? 0)) / total;
    if ((left[i] as number) < 0) {
      leaves[i] = 1;
      rBranch[i] = rNode[i] as number;
    } else {
      leaves[i] = (leaves[left[i] as number] as number) + (leaves[right[i] as number] as number);
      rBranch[i] = (rBranch[left[i] as number] as number) + (rBranch[right[i] as number] as number);
    }
  }

  const alphaOf = (i: number): number =>
    ((rNode[i] as number) - (rBranch[i] as number)) / ((leaves[i] as number) - 1);

  // Lazy min-heap of [alpha, id, leavesAtPush]; an entry is stale when the node was pruned or
  // its subtree changed since the push.
  const heap: Array<[number, number, number]> = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const push = (entry: [number, number, number]): void => {
    let i = heap.length;
    heap.push(entry);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i] as [number, number, number], heap[p] as [number, number, number])) break;
      const tmp = heap[i] as [number, number, number];
      heap[i] = heap[p] as [number, number, number];
      heap[p] = tmp;
      i = p;
    }
  };
  const pop = (): [number, number, number] | undefined => {
    const top = heap[0];
    const last = heap.pop();
    if (top === undefined || last === undefined) return top;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let best = i;
        if (
          l < heap.length &&
          less(heap[l] as [number, number, number], heap[best] as [number, number, number])
        )
          best = l;
        if (
          r < heap.length &&
          less(heap[r] as [number, number, number], heap[best] as [number, number, number])
        )
          best = r;
        if (best === i) break;
        const tmp = heap[i] as [number, number, number];
        heap[i] = heap[best] as [number, number, number];
        heap[best] = tmp;
        i = best;
      }
    }
    return top;
  };

  const internal = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    if ((left[i] as number) >= 0) {
      internal[i] = 1;
      push([alphaOf(i), i, leaves[i] as number]);
    }
  }

  let prunedAny = false;
  const collapsedAt = new Uint8Array(count);
  for (let entry = pop(); entry !== undefined; entry = pop()) {
    const [alpha, id, leavesAtPush] = entry;
    if (!internal[id] || leaves[id] !== leavesAtPush) continue;
    if (alpha > ccpAlpha) break;
    // Cut the subtree below `id`.
    const removedLeaves = (leaves[id] as number) - 1;
    const branchGain = (rBranch[id] as number) - (rNode[id] as number);
    const below: number[] = [left[id] as number, right[id] as number];
    for (let id2 = below.pop(); id2 !== undefined; id2 = below.pop()) {
      internal[id2] = 0;
      if ((left[id2] as number) >= 0) below.push(left[id2] as number, right[id2] as number);
    }
    internal[id] = 0;
    collapsedAt[id] = 1;
    prunedAny = true;
    leaves[id] = 1;
    rBranch[id] = rNode[id] as number;
    for (let a = parent[id] as number; a >= 0; a = parent[a] as number) {
      leaves[a] = (leaves[a] as number) - removedLeaves;
      rBranch[a] = (rBranch[a] as number) - branchGain;
      push([alphaOf(a), a, leaves[a] as number]);
    }
  }
  if (!prunedAny) return stripCollapsed(root);

  // Rebuild the tree with the pruned nodes replaced by their leaves.
  const ids = new Map<TreeNode, number>();
  for (let i = 0; i < count; i++) ids.set(nodes[i] as TreeNode, i);
  return rebuild(root, (node) => {
    const id = ids.get(node) as number;
    return collapsedAt[id] === 1 ? (node.collapsed ?? node) : undefined;
  });
}

function stripCollapsed(root: TreeNode): TreeNode {
  return rebuild(root, () => undefined);
}

/**
 * Copy of a tree without the `collapsed` helper leaves; `replaceWith` may return a leaf that
 * stands in for a node (and its whole subtree).
 */
function rebuild(root: TreeNode, replaceWith: (node: TreeNode) => TreeNode | undefined): TreeNode {
  let out: TreeNode | undefined;
  type Job = { source: TreeNode; attach: (copy: TreeNode) => void };
  const stack: Job[] = [
    {
      source: root,
      attach: (copy) => {
        out = copy;
      },
    },
  ];
  for (let job = stack.pop(); job !== undefined; job = stack.pop()) {
    const replacement = replaceWith(job.source);
    if (replacement !== undefined) {
      job.attach(replacement);
      continue;
    }
    const source = job.source;
    if (source.isLeaf || !source.left || !source.right) {
      job.attach(source);
      continue;
    }
    const copy: MutableTreeNode = {
      isLeaf: false,
      ...(source.featureIndex === undefined ? {} : { featureIndex: source.featureIndex }),
      ...(source.threshold === undefined ? {} : { threshold: source.threshold }),
      ...(source.nSamples === undefined ? {} : { nSamples: source.nSamples }),
      ...(source.weightedImpurityDecrease === undefined
        ? {}
        : { weightedImpurityDecrease: source.weightedImpurityDecrease }),
      ...(source.impurity === undefined ? {} : { impurity: source.impurity }),
      ...(source.weightedNSamples === undefined
        ? {}
        : { weightedNSamples: source.weightedNSamples }),
    };
    job.attach(copy as TreeNode);
    stack.push({
      source: source.right,
      attach: (child) => {
        copy.right = child;
      },
    });
    stack.push({
      source: source.left,
      attach: (child) => {
        copy.left = child;
      },
    });
  }
  return out as unknown as TreeNode;
}
