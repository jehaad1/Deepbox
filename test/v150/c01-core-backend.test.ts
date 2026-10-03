import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  availableCores,
  BroadcastError,
  ConvergenceError,
  catchWarnings,
  check_array,
  check_is_fitted,
  check_X_y,
  DataValidationError,
  DeepboxError,
  filterWarnings,
  fromJSON,
  getConfig,
  getSeed,
  IndexError,
  InvalidParameterError,
  Logger,
  load,
  MemoryError,
  NotFittedError,
  NotImplementedError,
  normalizeAxes,
  normalizeAxis,
  resetConfig,
  resetWarnings,
  ShapeError,
  save,
  setConfig,
  setLogHandler,
  setSeed,
  toJSON,
  WorkerPool,
  warn,
} from "../../src/core";
import {
  clearSeed,
  rand,
  getSeed as randomGetSeed,
  setSeed as randomSetSeed,
} from "../../src/random";

describe("c01 errors: cause chaining", () => {
  it("does not define `cause` unless one was supplied", () => {
    const plain = new DeepboxError("x");
    expect("cause" in plain).toBe(false);
    const cause = new Error("root");
    const chained = new DeepboxError("x", { cause });
    expect(chained.cause).toBe(cause);
  });

  it("subclasses accept and forward a cause", () => {
    const cause = new Error("root");
    expect(new ConvergenceError("m", { iterations: 3, cause }).cause).toBe(cause);
    expect(new IndexError("m", { index: 1, cause }).cause).toBe(cause);
    expect(new MemoryError("m", { requestedBytes: 1, cause }).cause).toBe(cause);
    expect(new ShapeError("m", { cause }).cause).toBe(cause);
    expect(new InvalidParameterError("m", "p", 1, { cause }).cause).toBe(cause);
    expect(new NotFittedError("m", "Model", { cause }).cause).toBe(cause);
    expect(new NotImplementedError("m", { cause }).cause).toBe(cause);
    expect(new BroadcastError([2], [3], "ctx", { cause }).cause).toBe(cause);
    expect("cause" in new ConvergenceError("m")).toBe(false);
  });

  it("keeps detail fields next to the cause", () => {
    const err = new ConvergenceError("m", { iterations: 7, tolerance: 1e-6 });
    expect(err.iterations).toBe(7);
    expect(err.tolerance).toBe(1e-6);
    expect(new NotImplementedError().message).toBe("Not implemented");
  });
});

describe("c01 config: seed handling", () => {
  beforeEach(() => {
    resetConfig();
  });
  afterEach(() => {
    resetConfig();
  });

  it("setConfig without `seed` does not restart the random stream", () => {
    randomSetSeed(7);
    rand([1]);
    const expectedSecond = rand([1]).data[0];

    randomSetSeed(7);
    rand([1]);
    setConfig({ defaultDtype: "float32" });
    const second = rand([1]).data[0];
    expect(second).toBe(expectedSecond);
  });

  it("setConfig without `seed` keeps a seed set through the random module", () => {
    randomSetSeed(11);
    setConfig({ defaultDtype: "float64" });
    expect(randomGetSeed()).toBe(11);
    expect(getSeed()).toBe(11);
    expect(getConfig().seed).toBe(11);
  });

  it("getSeed reflects seeds set or cleared through deepbox/random", () => {
    randomSetSeed(5);
    expect(getSeed()).toBe(5);
    clearSeed();
    expect(getSeed()).toBeNull();
    expect(getConfig().seed).toBeNull();
  });

  it("explicit seed in setConfig still seeds and null clears", () => {
    setConfig({ seed: 42 });
    const a = rand([3]).toArray();
    setSeed(42);
    expect(rand([3]).toArray()).toEqual(a);
    setConfig({ seed: null });
    expect(getSeed()).toBeNull();
    expect(randomGetSeed()).toBeUndefined();
  });

  it("reports a clear message for non-numeric seeds", () => {
    // @ts-expect-error - runtime validation
    expect(() => setConfig({ seed: "7" })).toThrow(/safe integer or null/);
    // @ts-expect-error - runtime validation
    expect(() => setSeed("7")).toThrow(/safe integer/);
  });
});

describe("c01 WorkerPool", () => {
  it("applies `initial` exactly once in reduce", async () => {
    const pool = new WorkerPool({ maxWorkers: 3 });
    // sum(1..6) = 21; Array.prototype.reduce gives 31 with initial = 10
    expect(await pool.reduce([1, 2, 3, 4, 5, 6], (a, b) => a + b, 10)).toBe(31);
    expect(await pool.reduce([1, 2, 3, 4, 5, 6], (a, b) => a + b, 10)).toBe(
      [1, 2, 3, 4, 5, 6].reduce((a, b) => a + b, 10)
    );
    expect(await pool.reduce([5], (a, b) => a * b, 2)).toBe(10);
    pool.terminate();
  });

  it("reduce preserves order for non-commutative associative functions", async () => {
    const pool = new WorkerPool({ maxWorkers: 3 });
    const joined = await pool.reduce(["a", "b", "c", "d", "e"], (a, b) => a + b, "");
    expect(joined).toBe("abcde");
    pool.terminate();
  });

  it("passes only the item to map and filter callbacks", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    expect(await pool.map(["10", "10", "10", "10"], parseInt)).toEqual([10, 10, 10, 10]);
    const argCounts: number[] = [];
    const countArgs = (...args: unknown[]): number => {
      argCounts.push(args.length);
      return 0;
    };
    await pool.map([1, 2, 3, 4], countArgs as (item: number) => number);
    expect(argCounts).toEqual([1, 1, 1, 1]);
    const seen: number[] = [];
    const countAndKeep = (...args: unknown[]): boolean => {
      seen.push(args.length);
      return true;
    };
    await pool.filter([1, 2, 3, 4, 5], countAndKeep as (item: number) => boolean);
    expect(seen.every((n) => n === 1)).toBe(true);
    pool.terminate();
  });

  it("rejects invalid maxWorkers instead of looping forever", () => {
    for (const bad of [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => new WorkerPool({ maxWorkers: bad })).toThrow(InvalidParameterError);
    }
    expect(() => new WorkerPool({ taskTimeout: 0 })).toThrow(InvalidParameterError);
    expect(() => new WorkerPool({ taskTimeout: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("keeps order across chunk boundaries and handles empty input", async () => {
    const pool = new WorkerPool({ maxWorkers: 4 });
    const items = Array.from({ length: 11 }, (_, i) => i);
    expect(await pool.map(items, (x) => x * 2)).toEqual(items.map((x) => x * 2));
    expect(await pool.filter(items, (x) => x % 3 === 0)).toEqual([0, 3, 6, 9]);
    const idx: number[] = [];
    await pool.forEach(items, (_v, i) => {
      idx.push(i);
    });
    expect(idx).toEqual(items);
    expect(await pool.filter([], () => true)).toEqual([]);
    await pool.forEach([], () => {
      throw new Error("never");
    });
    pool.terminate();
  });

  it("stops at the first failing chunk", async () => {
    const pool = new WorkerPool({ maxWorkers: 4 });
    const visited: number[] = [];
    await expect(
      pool.map([1, 2, 3, 4, 5, 6, 7, 8], (x) => {
        visited.push(x);
        if (x === 3) throw new Error("boom");
        return x;
      })
    ).rejects.toThrow("boom");
    expect(visited).toEqual([1, 2, 3]);
    expect(pool.status().activeWorkers).toBe(0);
    pool.terminate();
  });

  it("terminated pools reject with a DeepboxError (not NotImplementedError)", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    pool.terminate();
    pool.terminate();
    const error = await pool.map([1], (x) => x).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(DeepboxError);
    expect(error).not.toBeInstanceOf(NotImplementedError);
    expect((error as Error).message).toMatch(/terminated/);
  });

  it("counts completed chunks consistently for small and large inputs", async () => {
    const pool = new WorkerPool({ maxWorkers: 4 });
    await pool.map([1, 2], (x) => x);
    expect(pool.status().completedTasks).toBeGreaterThan(0);
    expect(pool.status().pendingTasks).toBe(0);
    pool.terminate();
  });

  it("availableCores matches the operating system", () => {
    expect(availableCores()).toBe(os.availableParallelism());
    expect(new WorkerPool().status().maxWorkers).toBe(os.availableParallelism());
  });
});

describe("c01 Logger", () => {
  afterEach(() => {
    setLogHandler(undefined);
  });

  it("validates the level passed to log()", () => {
    const log = new Logger(3, "T");
    setLogHandler(() => {});
    expect(() => log.log(4 as 3, "x")).toThrow(InvalidParameterError);
    expect(() => log.log(-1 as 0, "x")).toThrow(InvalidParameterError);
    expect(() => log.log(1.5 as 1, "x")).toThrow(InvalidParameterError);
    expect(() => log.log(0, "x")).not.toThrow();
    expect(log.getEntries()).toHaveLength(0);
  });

  it("requires a string source", () => {
    expect(() => new Logger(1, undefined as unknown as string)).toThrow(InvalidParameterError);
  });

  it("includes the source in entries passed to the handler", () => {
    const seen: (string | undefined)[] = [];
    setLogHandler((e) => seen.push(e.source));
    new Logger(1, "KMeans").info("done");
    expect(seen).toEqual(["KMeans"]);
  });

  it("keeps only the most recent entries, in order, and returns copies", () => {
    const log = new Logger(0, "T");
    for (let i = 0; i < 10_005; i++) log.info(`m${i}`);
    const entries = log.getEntries();
    expect(entries).toHaveLength(10_000);
    expect(entries[0]?.message).toBe("m5");
    expect(entries[9_999]?.message).toBe("m10004");
    log.clear();
    expect(entries).toHaveLength(10_000);
    expect(log.getEntries()).toHaveLength(0);
    log.info("again");
    expect(log.getEntries().map((e) => e.message)).toEqual(["again"]);
  });
});

describe("c01 warnings", () => {
  afterEach(() => {
    resetWarnings();
  });

  it("'error' filters throw a DeepboxError with the category prefix", () => {
    filterWarnings("error", { category: "ConvergenceWarning" });
    let caught: unknown;
    try {
      warn("did not converge", "ConvergenceWarning");
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(DeepboxError);
    expect((caught as Error).message).toBe("[ConvergenceWarning] did not converge");
  });

  it("regex filters with the global flag match on every call", () => {
    filterWarnings("ignore", { message: /noisy/g });
    const out = catchWarnings(() => {
      for (let i = 0; i < 4; i++) warn("noisy message", "UserWarning");
    });
    expect(out).toHaveLength(0);
  });

  it("user regex objects are not mutated", () => {
    const re = /noisy/g;
    filterWarnings("ignore", { message: re });
    catchWarnings(() => warn("noisy", "UserWarning"));
    expect(re.lastIndex).toBe(0);
    expect(re.flags).toBe("g");
  });

  it("string message filters are regular expressions", () => {
    filterWarnings("ignore", { message: "^skip " });
    const out = catchWarnings(() => {
      warn("skip this", "UserWarning");
      warn("keep skip this", "UserWarning");
    });
    expect(out.map((w) => w.message)).toEqual(["keep skip this"]);
  });

  it("rejects bad actions and bad patterns when the filter is added", () => {
    // @ts-expect-error - runtime validation
    expect(() => filterWarnings("silence")).toThrow(InvalidParameterError);
    expect(() => filterWarnings("ignore", { message: "(" })).toThrow(InvalidParameterError);
    const out = catchWarnings(() => warn("still works", "UserWarning"));
    expect(out).toHaveLength(1);
  });

  it("later filters take precedence", () => {
    filterWarnings("ignore", { category: "UserWarning" });
    filterWarnings("always", { category: "UserWarning", message: "keep" });
    const out = catchWarnings(() => {
      warn("drop me", "UserWarning");
      warn("keep me", "UserWarning");
    });
    expect(out.map((w) => w.message)).toEqual(["keep me"]);
  });
});

describe("c01 serialization", () => {
  it("round-trips NaN, Infinity and negative zero exactly", () => {
    const payload = {
      __type: "Tensor" as const,
      data: [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -0, 0, 1.5],
      shape: [6],
      dtype: "float64",
    };
    const restored = fromJSON(toJSON(payload));
    if (restored.__type !== "Tensor") throw new Error("expected tensor");
    const d = restored.data as number[];
    expect(Number.isNaN(d[0])).toBe(true);
    expect(d[1]).toBe(Number.POSITIVE_INFINITY);
    expect(d[2]).toBe(Number.NEGATIVE_INFINITY);
    expect(Object.is(d[3], -0)).toBe(true);
    expect(Object.is(d[4], 0)).toBe(true);
    expect(d[5]).toBe(1.5);
  });

  it("round-trips non-finite numbers inside estimator state", () => {
    const payload = {
      __type: "Estimator" as const,
      className: "X",
      params: { tol: Number.POSITIVE_INFINITY },
      state: { best: Number.NaN, big: 10n ** 20n },
    };
    const restored = fromJSON(toJSON(payload));
    if (restored.__type !== "Estimator") throw new Error("expected estimator");
    expect(restored.params["tol"]).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(restored.state["best"])).toBe(true);
    expect(restored.state["big"]).toBe(10n ** 20n);
  });

  it("reports malformed JSON as DataValidationError", () => {
    expect(() => fromJSON("not json")).toThrow(DataValidationError);
    expect(() => fromJSON('{"__type":"Tensor","data":[1],"shape":[1],"dtype":"x"')).toThrow(
      DataValidationError
    );
  });

  it("rejects structurally invalid payloads", () => {
    expect(() => fromJSON('{"__type":"Tensor"}')).toThrow(/data must be an array/);
    expect(() => fromJSON('{"__type":"Tensor","data":[],"shape":[-1],"dtype":"float32"}')).toThrow(
      /shape must be an array of non-negative integers/
    );
    expect(() => fromJSON('{"__type":"Tensor","data":[],"shape":[0],"dtype":3}')).toThrow(
      /dtype must be a string/
    );
    expect(() => fromJSON('{"__type":"ModuleState","parameters":{},"buffers":3}')).toThrow(
      /buffers must be an object/
    );
    expect(() =>
      fromJSON('{"__type":"ModuleState","parameters":{"w":{"data":[1]}},"buffers":{}}')
    ).toThrow(/parameters\['w'\]/);
    expect(() => fromJSON('{"__type":"Estimator","className":1,"params":{},"state":{}}')).toThrow(
      /className must be a string/
    );
    expect(() => fromJSON("[1,2]")).toThrow(/expected an object/);
  });

  it("rejects data whose length does not match the shape", () => {
    expect(() =>
      fromJSON('{"__type":"Tensor","data":[1,2,3],"shape":[2,2],"dtype":"float32"}')
    ).toThrow(/3 elements but shape \[2, 2\] holds 4/);
    expect(() => toJSON({ __type: "Tensor", data: [1, 2], shape: [3], dtype: "float64" })).toThrow(
      DataValidationError
    );
    // scalar and empty tensors
    expect(() =>
      fromJSON('{"__type":"Tensor","data":[5],"shape":[],"dtype":"float32"}')
    ).not.toThrow();
    expect(() =>
      fromJSON('{"__type":"Tensor","data":[],"shape":[0,3],"dtype":"float32"}')
    ).not.toThrow();
    // complex data may be interleaved, so it is not length-checked
    expect(() =>
      fromJSON('{"__type":"Tensor","data":[1,2,3,4],"shape":[2],"dtype":"complex64"}')
    ).not.toThrow();
  });

  it("toJSON validates its input", () => {
    // @ts-expect-error - runtime validation
    expect(() => toJSON({ foo: 1 })).toThrow(DataValidationError);
    // @ts-expect-error - runtime validation
    expect(() => toJSON(null)).toThrow(DataValidationError);
  });

  it("save/load keep non-finite values and wrap read errors with a cause", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "deepbox-c01-"));
    const file = path.join(dir, "t.json");
    try {
      await save(file, {
        __type: "Tensor",
        data: [Number.NaN, 2],
        shape: [2],
        dtype: "float32",
      });
      const restored = await load(file);
      expect(restored.__type).toBe("Tensor");
      expect(Number.isNaN((restored as unknown as { data: number[] }).data[0])).toBe(true);

      fs.writeFileSync(file, "{broken", "utf-8");
      await expect(load(file)).rejects.toThrow(DataValidationError);

      const err = await load(path.join(dir, "missing.json")).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(DeepboxError);
      expect((err as DeepboxError).cause).toBeInstanceOf(Error);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("c01 validation helpers", () => {
  const fake = (shape: number[], dtype = "float32") => ({ shape, dtype });

  it("check_array uses the shape length for ensureNdim", () => {
    expect(() => check_array(fake([2, 3]), { ensureNdim: 2 })).not.toThrow();
    expect(() => check_array(fake([2, 3]), { ensureNdim: 1 })).toThrow(/Expected 1D array, got 2D/);
  });

  it("check_array treats 0-d inputs as non-empty and flags zero-sized axes", () => {
    expect(() => check_array(fake([]))).not.toThrow();
    expect(() => check_array(fake([0, 3]))).toThrow(/0 samples/);
    expect(() => check_array(fake([3, 0]))).toThrow(/0 features/);
    expect(() => check_array(fake([2, 0, 2]))).toThrow(/axis 1 has size 0/);
    expect(() => check_array(fake([3, 0]), { allowEmpty: true })).not.toThrow();
  });

  it("check_array accepts a list of dtypes", () => {
    expect(() => check_array(fake([2]), { dtype: ["float32", "float64"] })).not.toThrow();
    expect(() => check_array(fake([2], "int32"), { dtype: ["float32", "float64"] })).toThrow(
      /Expected dtype in \[float32, float64\], got 'int32'/
    );
    expect(() => check_array(fake([2], "int32"), { dtype: "float32" })).toThrow(
      /Expected dtype 'float32', got 'int32'/
    );
  });

  it("check_array and check_X_y keep the input type", () => {
    const x = fake([2, 2]);
    const y = fake([2]);
    const out = check_array(x);
    expect(out).toBe(x);
    const [xo, yo] = check_X_y(x, y);
    expect(xo).toBe(x);
    expect(yo).toBe(y);
  });

  it("check_X_y rejects X with zero features and 3-D multi-output y", () => {
    expect(() => check_X_y(fake([3, 0]), fake([3]))).toThrow(/0 features/);
    expect(() => check_X_y(fake([2, 2]), fake([2, 1, 1]), { multiOutput: true })).toThrow(
      /Expected 1D or 2D y, got 3D/
    );
    expect(() => check_X_y(fake([2, 2]), fake([2, 3]), { multiOutput: true })).not.toThrow();
    expect(() => check_X_y(fake([2, 2]), fake([2, 3]))).toThrow(/Expected 1D array/);
  });

  it("check_is_fitted validates its argument and records the model name", () => {
    // @ts-expect-error - runtime validation
    expect(() => check_is_fitted(null)).toThrow(DataValidationError);
    // @ts-expect-error - runtime validation
    expect(() => check_is_fitted(3)).toThrow(DataValidationError);
    class Model {
      coef_: number[] | undefined;
    }
    let caught: unknown;
    try {
      check_is_fitted(new Model() as unknown as Record<string, unknown>);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(NotFittedError);
    expect((caught as NotFittedError).modelName).toBe("Model");
    expect(() => check_is_fitted({ __cache_: 1 })).toThrow(NotFittedError);
    expect(() => check_is_fitted({ coef_: null })).not.toThrow();
  });

  it("normalizeAxis never returns negative zero", () => {
    expect(Object.is(normalizeAxis(-0, 3), 0)).toBe(true);
    expect(Object.is(normalizeAxis(-3, 3), 0)).toBe(true);
    expect(normalizeAxis(-1, 3)).toBe(2);
  });

  it("normalizeAxes accepts readonly arrays and names the duplicate", () => {
    const axes: readonly number[] = [0, 2];
    expect(normalizeAxes(axes, 3)).toEqual([0, 2]);
    expect(normalizeAxes(["rows", -1], 2)).toEqual([0, 1]);
    expect(() => normalizeAxes([0, -2], 2)).toThrow(/duplicate axis/);
    expect(() => normalizeAxes([1, 1], 2)).toThrow(InvalidParameterError);
  });
});
