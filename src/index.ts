/**
 * Deepbox - TypeScript Data Science & ML Library
 *
 * A comprehensive library for numerical computing, data manipulation,
 * and machine learning in TypeScript/JavaScript.
 *
 * @example
 * ```ts
 * // Import from specific modules (recommended)
 * import { tensor, zeros, ones } from "deepbox/ndarray";
 * import { DataFrame, Series } from "deepbox/dataframe";
 * import { LinearRegression } from "deepbox/ml";
 *
 * // Or import namespaced modules
 * import * as ds from "deepbox";
 * ds.ndarray.tensor([1, 2, 3]);
 * ```
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
  ndarray,
  linalg,
  dataframe,
  stats,
  metrics,
  preprocess,
  ml,
  nn,
  optim,
  random,
  plot,
  datasets,
};
