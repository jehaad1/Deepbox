// NN-21: WorkerPool awaits asynchronous callbacks in map, reduce, filter, forEach, all and
// exec, and enforces taskTimeout with a TaskTimeoutError. Behavior is checked against the
// documented contract (results keep input order; a failed chunk stops later chunks).
import { describe, expect, it } from "vitest";
import { DeepboxError } from "../../src/core";
import { TaskTimeoutError, WorkerPool } from "../../src/core/parallel";

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

function busyWait(ms: number): void {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    // spin
  }
}

describe("NN-21: WorkerPool asynchronous callbacks", () => {
  it("map awaits asynchronous callbacks and keeps the order", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    const out = await pool.map([1, 2, 3, 4, 5], async (x) => {
      await sleep(5 - x);
      return x * 2;
    });
    expect(out).toEqual([2, 4, 6, 8, 10]);
    expect(out[0] instanceof Promise).toBe(false);
  });

  it("filter, forEach, reduce and all await asynchronous callbacks", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    expect(await pool.filter([1, 2, 3, 4], async (x) => x % 2 === 0)).toEqual([2, 4]);
    const seen: number[] = [];
    await pool.forEach([1, 2, 3], async (x, i) => {
      await sleep(1);
      seen.push(x + i);
    });
    expect(seen).toEqual([1, 3, 5]);
    expect(await pool.reduce([1, 2, 3, 4], async (a, b) => a + b, 10)).toBe(20);
    expect(await pool.all([async () => 1, () => 2])).toEqual([1, 2]);
    const r = await pool.exec(async (x: number) => x + 1, 1);
    expect(r.value).toBe(2);
  });

  it("an asynchronous failure rejects the call and later chunks do not start", async () => {
    const pool = new WorkerPool({ maxWorkers: 3 });
    const started: number[] = [];
    await expect(
      pool.map([1, 2, 3, 4, 5, 6], async (x) => {
        started.push(x);
        if (x === 2) throw new Error("boom");
        return x;
      })
    ).rejects.toThrow("boom");
    expect(started).toEqual([1, 2]);
    expect(pool.status().activeWorkers).toBe(0);
  });
});

describe("NN-21: WorkerPool taskTimeout", () => {
  it("rejects an asynchronous task that outlives taskTimeout", async () => {
    const pool = new WorkerPool({ taskTimeout: 20 });
    const t0 = Date.now();
    await expect(pool.map([1], () => sleep(500))).rejects.toThrow(TaskTimeoutError);
    expect(Date.now() - t0).toBeLessThan(400);
    await expect(pool.exec(() => sleep(500), 0)).rejects.toThrow(/taskTimeout/);
    await expect(pool.all([() => sleep(500)])).rejects.toThrow(TaskTimeoutError);
    expect(pool.status().activeWorkers).toBe(0);
  });

  it("rejects a synchronous task that ran longer than taskTimeout", async () => {
    const pool = new WorkerPool({ taskTimeout: 10 });
    await expect(pool.exec(() => busyWait(60), 0)).rejects.toThrow(TaskTimeoutError);
    await expect(pool.map([1, 2], () => busyWait(60))).rejects.toThrow(TaskTimeoutError);
    await expect(pool.filter([1], () => busyWait(60) === undefined)).rejects.toThrow(
      TaskTimeoutError
    );
    await expect(pool.forEach([1], () => busyWait(60))).rejects.toThrow(TaskTimeoutError);
    await expect(
      pool.reduce(
        [1, 2],
        (a, b) => {
          busyWait(60);
          return a + b;
        },
        0
      )
    ).rejects.toThrow(TaskTimeoutError);
    await expect(pool.all([() => busyWait(60)])).rejects.toThrow(TaskTimeoutError);
  });

  it("lets tasks that finish in time through, and the error is a DeepboxError", async () => {
    const pool = new WorkerPool({ taskTimeout: 1000 });
    expect(await pool.map([1, 2, 3], (x) => x + 1)).toEqual([2, 3, 4]);
    expect(
      await pool.map([1], async (x) => {
        await sleep(5);
        return x;
      })
    ).toEqual([1]);
    const error = new TaskTimeoutError("late");
    expect(error).toBeInstanceOf(DeepboxError);
    expect(error.name).toBe("TaskTimeoutError");
  });
});
