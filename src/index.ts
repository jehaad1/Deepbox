/**
 * Deepbox — TypeScript toolkit for AI & numerical computing
 *
 * A comprehensive framework for tensors, linear algebra, tabular data,
 * machine learning, neural networks, statistics, and related workflows in TypeScript/JavaScript.
 *
 * @example
 * ```ts
 * // Import from specific modules (recommended)
 * import { tensor, zeros, ones } from "deepbox/ndarray";
 * import { DataFrame, Series } from "deepbox/dataframe";
 * import { LinearRegression } from "deepbox/ml";
 *
 * // Or import namespaced modules
 * import * as db from "deepbox";
 * db.ndarray.tensor([1, 2, 3]);
 * ```
 * @see {@link https://deepbox.dev/docs/introduction | Deepbox documentation}
 */

// Re-export modules as namespaces to avoid naming conflicts
import * as core from "./core";
import * as dataframe from "./dataframe";
import * as datasets from "./datasets";
import * as linalg from "./linalg";
import * as metrics from "./metrics";
import * as ml from "./ml";
import * as ndarray from "./ndarray";
import * as nn from "./nn";
import * as optim from "./optim";
import * as plot from "./plot";
import * as preprocess from "./preprocess";
import * as random from "./random";
import * as stats from "./stats";

export {
  core,
  dataframe,
  datasets,
  linalg,
  metrics,
  ml,
  ndarray,
  nn,
  optim,
  plot,
  preprocess,
  random,
  stats,
};
