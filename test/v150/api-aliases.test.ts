import { describe, expect, it } from "vitest";
import * as core from "../../src/core";
import * as dataframe from "../../src/dataframe";
import * as linalg from "../../src/linalg";
import * as ml from "../../src/ml";
import * as random from "../../src/random";

// Each pair is [deprecated snake_case name, canonical camelCase name].
const pairs: ReadonlyArray<readonly [Record<string, unknown>, string, string]> = [
  [core, "check_X_y", "checkXY"],
  [core, "check_array", "checkArray"],
  [core, "check_is_fitted", "checkIsFitted"],
  [linalg, "block_diag", "blockDiag"],
  [linalg, "matrix_power", "matrixPower"],
  [linalg, "solve_banded", "solveBanded"],
  [dataframe, "date_range", "dateRange"],
  [dataframe, "to_datetime", "toDatetime"],
  [ml, "cross_val_score", "crossValScore"],
  [ml, "cross_validate", "crossValidate"],
  [ml, "export_text", "exportText"],
  [ml, "get_output", "getOutput"],
  [ml, "reset_output", "resetOutput"],
  [ml, "set_output", "setOutput"],
  [random, "f_distribution", "fDistribution"],
  [random, "gumbel_softmax", "gumbelSoftmax"],
  [random, "multivariate_normal", "multivariateNormal"],
  [random, "negative_binomial", "negativeBinomial"],
  [random, "student_t", "studentT"],
];

describe("1.5.0 camelCase aliases", () => {
  it.each(
    pairs.map(([mod, snake, camel]) => [snake, camel, mod] as const)
  )("%s is the same function as %s", (snake, camel, mod) => {
    expect(typeof mod[camel]).toBe("function");
    expect(mod[camel]).toBe(mod[snake]);
  });
});
