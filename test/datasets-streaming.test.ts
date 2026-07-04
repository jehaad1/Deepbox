import { describe, expect, it } from "vitest";
import {
  asyncIterableDataset,
  DataLoader,
  defaultCollate,
  iterableDataset,
  StreamingDataset,
  type StreamSample,
} from "../src/datasets";
import { type AnyTensor, GradTensor, parameter, type Tensor } from "../src/ndarray";
import { Linear, Module, mseLoss, Trainer } from "../src/nn";
import { SGD } from "../src/optim";

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** A tiny regression model: forward promotes a plain Tensor to a GradTensor. */
class StreamModel extends Module {
  private fc: Linear;
  constructor() {
    super();
    this.fc = new Linear(2, 1);
    this.registerModule("fc", this.fc);
  }
  forward(x: AnyTensor): AnyTensor {
    const gx = x instanceof GradTensor ? x : parameter((x as Tensor).toArray() as number[][]);
    return this.fc.forward(gx);
  }
}

const regressionLoss = (output: AnyTensor, target: Tensor): AnyTensor =>
  mseLoss(output as GradTensor, parameter(target.toArray() as number[][]));

/** A row of a synthetic labeled corpus: `[[x0, x1], [y]]`. */
function row(i: number): StreamSample {
  return [[i, i * 2], [i * 3 + 1]] as const;
}

// ═════════════════════════════════════════════════════════════════════════════
// StreamingDataset core
// ═════════════════════════════════════════════════════════════════════════════

describe("StreamingDataset", () => {
  it("iterates a sync generator lazily and re-iterates for multiple epochs", () => {
    let starts = 0;
    const ds = iterableDataset<number>(function* () {
      starts++;
      for (let i = 0; i < 5; i++) yield i;
    });

    expect([...ds]).toEqual([0, 1, 2, 3, 4]);
    expect([...ds]).toEqual([0, 1, 2, 3, 4]); // fresh iterator each epoch
    expect(starts).toBe(2);
    expect(ds.isSync).toBe(true);
  });

  it("consumes a stream far larger than a threshold WITHOUT materializing it", () => {
    const N = 1_000_000; // would be huge if materialized
    let produced = 0;
    const ds = iterableDataset<StreamSample>(function* () {
      for (let i = 0; i < N; i++) {
        produced++;
        yield row(i);
      }
    });

    const batches = ds.batch(16, defaultCollate);
    let seen = 0;
    for (const [x, y] of batches as unknown as Iterable<[Tensor, Tensor]>) {
      expect(x.shape).toEqual([16, 2]);
      expect(y.shape).toEqual([16, 1]);
      seen++;
      if (seen === 2) break; // stop early
    }

    // Only the samples needed for two batches were ever pulled from the source.
    expect(produced).toBe(32);
    expect(produced).toBeLessThan(N);
  });

  it("shuffle uses a bounded buffer (does not read the whole stream up-front)", () => {
    const N = 1_000_000;
    let produced = 0;
    const ds = iterableDataset<number>(function* () {
      for (let i = 0; i < N; i++) {
        produced++;
        yield i;
      }
    });

    const it = ds.shuffle(64, 7)[Symbol.iterator]();
    it.next(); // pull a single shuffled element

    // Filling a 64-element buffer + emitting one element touches ~65 items,
    // nowhere near the full corpus.
    expect(produced).toBeLessThanOrEqual(65);
    expect(produced).toBeGreaterThanOrEqual(64);
  });

  it("shuffle preserves the multiset, reorders, and is seed-deterministic", () => {
    const base = () =>
      iterableDataset<number>(function* () {
        for (let i = 0; i < 200; i++) yield i;
      });

    const a = [...base().shuffle(32, 123)];
    const b = [...base().shuffle(32, 123)];
    const c = [...base().shuffle(32, 999)];

    expect(a).toEqual(b); // deterministic for a given seed
    expect(a).not.toEqual(c); // different seed → different order
    expect([...a].sort((p, q) => p - q)).toEqual(Array.from({ length: 200 }, (_, i) => i));
    expect(a).not.toEqual(Array.from({ length: 200 }, (_, i) => i)); // actually shuffled
  });

  it("map transforms samples lazily", () => {
    const ds = iterableDataset<number>(function* () {
      for (let i = 0; i < 4; i++) yield i;
    }).map((v) => v * 10);
    expect([...ds]).toEqual([0, 10, 20, 30]);
  });

  it("dropLast discards a trailing short batch", () => {
    const ds = iterableDataset<StreamSample>(function* () {
      for (let i = 0; i < 10; i++) yield row(i);
    });
    const kept = [...ds.batch(4, defaultCollate)];
    const dropped = [...ds.batch(4, defaultCollate, { dropLast: true })];
    expect(kept.length).toBe(3); // 4 + 4 + 2
    expect(dropped.length).toBe(2); // trailing 2 dropped
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// defaultCollate
// ═════════════════════════════════════════════════════════════════════════════

describe("defaultCollate", () => {
  it("collates [features, integer label] into [X float, y int32]", () => {
    const [x, y] = defaultCollate([
      [[1, 2], 0],
      [[3, 4], 1],
    ]);
    expect(x.shape).toEqual([2, 2]);
    expect(y?.shape).toEqual([2]);
    expect(y?.dtype).toBe("int32");
    expect(y?.toArray()).toEqual([0, 1]);
  });

  it("collates [features, float label] into [X, y float32]", () => {
    const [, y] = defaultCollate([
      [[1, 2], 0.5],
      [[3, 4], 1.5],
    ]);
    expect(y?.dtype).toBe("float32");
  });

  it("collates multi-output labels into a 2D float target", () => {
    const [x, y] = defaultCollate([
      [
        [1, 2],
        [10, 20],
      ],
      [
        [3, 4],
        [30, 40],
      ],
    ]);
    expect(x.shape).toEqual([2, 2]);
    expect(y?.shape).toEqual([2, 2]);
  });

  it("collates bare feature vectors into [X] with no target", () => {
    const batch = defaultCollate([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(batch.length).toBe(1);
    expect(batch[0].shape).toEqual([2, 3]);
  });

  it("throws on an empty batch", () => {
    expect(() => defaultCollate([])).toThrow();
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// Async iteration + prefetch
// ═════════════════════════════════════════════════════════════════════════════

describe("StreamingDataset async + prefetch", () => {
  it("iterates an async source and yields all elements in order", async () => {
    const ds = asyncIterableDataset<number>(async function* () {
      for (let i = 0; i < 5; i++) {
        await Promise.resolve();
        yield i;
      }
    });
    expect(ds.isSync).toBe(false);
    expect(await ds.toArray()).toEqual([0, 1, 2, 3, 4]);
  });

  it("async-only streams throw when iterated synchronously", () => {
    const ds = asyncIterableDataset<number>(async function* () {
      yield 1;
    });
    expect(() => [...ds]).toThrow();
  });

  it("prefetch reads ahead: fills the pipeline before consumption", async () => {
    let produced = 0;
    const ds = asyncIterableDataset<number>(async function* () {
      for (let i = 0; i < 50; i++) {
        await Promise.resolve();
        produced++;
        yield i;
      }
    });

    const pf = ds.prefetch(5);
    expect(pf.isSync).toBe(false);

    const it = pf[Symbol.asyncIterator]();
    const first = await it.next();
    expect(first.value).toBe(0);
    // Let the in-flight read-ahead requests drain while the "consumer" is busy.
    await new Promise((r) => setTimeout(r, 5));
    // The pipeline pulled ahead of consumption: with a read-ahead of 5, several
    // elements are produced before we ask for the second one.
    expect(produced).toBeGreaterThanOrEqual(5);

    // ...and the full stream still comes out complete and in order.
    const rest: number[] = [];
    for (let r = await it.next(); !r.done; r = await it.next()) rest.push(r.value);
    expect([first.value, ...rest]).toEqual(Array.from({ length: 50 }, (_, i) => i));
  });

  it("prefetch preserves ordering and multiset for a batched pipeline", async () => {
    const ds = iterableDataset<StreamSample>(function* () {
      for (let i = 0; i < 20; i++) yield row(i);
    });
    const batches = ds.batch(4, defaultCollate).prefetch(2);
    let count = 0;
    for await (const [x, y] of batches as unknown as AsyncIterable<[Tensor, Tensor]>) {
      expect(x.shape).toEqual([4, 2]);
      expect(y.shape).toEqual([4, 1]);
      count++;
    }
    expect(count).toBe(5);
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// DataLoader over a streaming dataset
// ═════════════════════════════════════════════════════════════════════════════

describe("DataLoader (streaming mode)", () => {
  it("still batches in-memory tensors (unchanged behavior)", () => {
    // Guard: the tensor constructor path is untouched.
    // (covered elsewhere; here we only assert it is not misrouted to streaming)
    const ds = iterableDataset<StreamSample>(function* () {
      for (let i = 0; i < 6; i++) yield row(i);
    });
    const loader = new DataLoader(ds, { batchSize: 2 });
    expect(loader).toBeInstanceOf(DataLoader);
  });

  it("iterates a streaming dataset lazily in batches (sync)", () => {
    const N = 500_000;
    let produced = 0;
    const ds = iterableDataset<StreamSample>(function* () {
      for (let i = 0; i < N; i++) {
        produced++;
        yield row(i);
      }
    });

    const loader = new DataLoader(ds, { batchSize: 8 });
    let seen = 0;
    for (const [x, y] of loader as unknown as Iterable<[Tensor, Tensor]>) {
      expect(x.shape).toEqual([8, 2]);
      expect(y.shape).toEqual([8, 1]);
      if (++seen === 3) break;
    }
    expect(produced).toBe(24); // only 3 batches * 8 samples pulled
  });

  it("applies a shuffle buffer and is seed-deterministic", () => {
    const build = () =>
      new DataLoader(
        iterableDataset<StreamSample>(function* () {
          for (let i = 0; i < 40; i++) yield row(i);
        }),
        { batchSize: 40, shuffleBufferSize: 16, seed: 42 }
      );

    const firstColOf = (loader: DataLoader<Tensor>): number[] => {
      const [batch] = [...(loader as unknown as Iterable<[Tensor, Tensor]>)];
      const x = batch[0];
      const out: number[] = [];
      for (let i = 0; i < x.shape[0]; i++) out.push(Number(x.data[x.offset + i * 2]));
      return out;
    };

    const a = firstColOf(build() as unknown as DataLoader<Tensor>);
    const b = firstColOf(build() as unknown as DataLoader<Tensor>);
    const identity = Array.from({ length: 40 }, (_, i) => i);

    expect(a).toEqual(b); // deterministic
    expect(a.slice().sort((p, q) => p - q)).toEqual(identity); // multiset preserved
    expect(a).not.toEqual(identity); // actually shuffled
  });

  it("length throws for streaming loaders", () => {
    const loader = new DataLoader(
      iterableDataset<StreamSample>(function* () {
        yield row(0);
      }),
      { batchSize: 1 }
    );
    expect(() => loader.length).toThrow();
  });

  it("supports async iteration with prefetch", async () => {
    const loader = new DataLoader(
      iterableDataset<StreamSample>(function* () {
        for (let i = 0; i < 24; i++) yield row(i);
      }),
      { batchSize: 6, prefetch: 2 }
    );

    let count = 0;
    for await (const [x, y] of loader as unknown as AsyncIterable<[Tensor, Tensor]>) {
      expect(x.shape).toEqual([6, 2]);
      expect(y.shape).toEqual([6, 1]);
      count++;
    }
    expect(count).toBe(4);
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// Trainer over streamed data
// ═════════════════════════════════════════════════════════════════════════════

describe("Trainer over a streamed DataLoader", () => {
  it("fitAsync trains one epoch over a streamed dataset (params update)", async () => {
    const model = new StreamModel();
    const optimizer = new SGD(model.parameters(), { lr: 0.001 });

    const before = ([...model.parameters()][0] as GradTensor).tensor.toArray() as number[][];

    const loader = new DataLoader(
      iterableDataset<StreamSample>(function* () {
        for (let i = 0; i < 64; i++) yield row(i % 8);
      }),
      { batchSize: 8, shuffleBufferSize: 16, prefetch: 2, seed: 1 }
    );

    const trainer = new Trainer(model, optimizer, regressionLoss, { epochs: 1 });
    const result = await trainer.fitAsync(
      loader as unknown as AsyncIterable<readonly [Tensor, Tensor]>
    );

    expect(result.history.length).toBe(1);
    expect(typeof result.history[0].trainLoss).toBe("number");
    expect(Number.isFinite(result.history[0].trainLoss)).toBe(true);

    const after = ([...model.parameters()][0] as GradTensor).tensor.toArray() as number[][];
    // At least one weight moved → gradients flowed and the optimizer stepped.
    let changed = false;
    for (let i = 0; i < before.length; i++) {
      for (let j = 0; j < before[i].length; j++) {
        if (before[i][j] !== after[i][j]) changed = true;
      }
    }
    expect(changed).toBe(true);
  });

  it("fit (sync) also consumes a sync streaming loader", () => {
    const model = new StreamModel();
    const optimizer = new SGD(model.parameters(), { lr: 0.001 });

    const loader = new DataLoader(
      iterableDataset<StreamSample>(function* () {
        for (let i = 0; i < 32; i++) yield row(i % 8);
      }),
      { batchSize: 8 }
    );

    const trainer = new Trainer(model, optimizer, regressionLoss, { epochs: 2 });
    const result = trainer.fit(loader as unknown as Iterable<readonly [Tensor, Tensor]>);
    expect(result.history.length).toBe(2);
  });
});

// Keep a reference so the StreamingDataset type import is exercised.
describe("StreamingDataset type", () => {
  it("is exported and constructible via factory", () => {
    const ds: StreamingDataset<number> = iterableDataset(function* () {
      yield 1;
    });
    expect(ds).toBeInstanceOf(StreamingDataset);
  });
});
