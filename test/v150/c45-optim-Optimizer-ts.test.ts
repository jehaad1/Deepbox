/**
 * Regression tests for src/optim/Optimizer.ts and src/optim/_internal.ts (v1.5.0).
 *
 * The momentum-SGD reference values come from PyTorch (float64):
 * lr=0.1, momentum=0.9, gradient [0.1, 0.2, 0.3] on [1, 2, 3], three steps
 * -> [0.9439, 1.8878, 2.8317].
 */
import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError, ShapeError } from "../../src/core";
import { type GradTensor, parameter, tensor, transpose } from "../../src/ndarray";
import { Adam, LBFGS, Optimizer, SGD } from "../../src/optim";
import { assertHasGradFloat } from "../../src/optim/_internal";

function makeParam(values: number[]): GradTensor {
  return parameter(tensor(values, { dtype: "float64" }));
}

function setGrad(p: GradTensor, values: number[]): void {
  p.setGrad(tensor(values, { dtype: "float64" }));
}

function values(p: GradTensor): number[] {
  return Array.from(p.tensor.data as Float64Array);
}

describe("Optimizer state dict ids", () => {
  it("assigns paramIds in flattened group order, not state insertion order", () => {
    const a = makeParam([1]);
    const b = makeParam([2]);
    const c = makeParam([3]);
    const opt = new SGD([a, b, c], { lr: 0.1, momentum: 0.9 });
    // State only for the second parameter.
    opt.loadStateDict({
      state: [{ paramId: 1, state: { momentumBuffer: new Float64Array([5]) } }],
    });
    const sd = opt.stateDict();
    expect(sd.paramGroups[0]?.paramIds).toEqual([0, 1, 2]);
    expect(sd.state).toHaveLength(1);
    expect(sd.state[0]?.paramId).toBe(1);
    expect(sd.state[0]?.param).toBe(b);

    // The saved dictionary restores into a fresh optimizer with the buffer on `b`.
    const a2 = makeParam([1]);
    const b2 = makeParam([2]);
    const c2 = makeParam([3]);
    const opt2 = new SGD([a2, b2, c2], { lr: 0.1, momentum: 0.9 });
    opt2.loadStateDict(sd);
    setGrad(a2, [0]);
    setGrad(b2, [0]);
    setGrad(c2, [0]);
    opt2.step();
    // Zero gradient, momentum buffer 5 -> buffer 4.5 -> b moves by -0.1 * 4.5.
    expect(values(b2)[0]).toBeCloseTo(2 - 0.45, 12);
    expect(values(a2)[0]).toBe(1);
    expect(values(c2)[0]).toBe(3);
  });
});

describe("Optimizer checkpoint resume", () => {
  it("resumes momentum SGD in a new optimizer over new parameter objects (torch parity)", () => {
    const p1 = makeParam([1, 2, 3]);
    const opt1 = new SGD([p1], { lr: 0.1, momentum: 0.9 });
    for (let i = 0; i < 2; i++) {
      setGrad(p1, [0.1, 0.2, 0.3]);
      opt1.step();
    }
    const checkpoint = opt1.stateDict();

    const p2 = makeParam(values(p1));
    const opt2 = new SGD([p2], { lr: 0.1, momentum: 0.9 });
    opt2.loadStateDict(checkpoint);
    setGrad(p2, [0.1, 0.2, 0.3]);
    opt2.step();

    const expected = [0.9439, 1.8878, 2.8317];
    values(p2).forEach((v, i) => {
      expect(v).toBeCloseTo(expected[i] ?? 0, 12);
    });
  });

  it("stateDict is a snapshot: later steps and lr changes do not alter it", () => {
    const p = makeParam([1, 2, 3]);
    const opt = new SGD([p], { lr: 0.1, momentum: 0.9 });
    setGrad(p, [0.1, 0.2, 0.3]);
    opt.step();
    const checkpoint = opt.stateDict();
    const savedBuffer = Array.from(checkpoint.state[0]?.state.momentumBuffer ?? []);
    expect(savedBuffer).toEqual([0.1, 0.2, 0.3]);

    setGrad(p, [1, 1, 1]);
    opt.step();
    const g0 = opt.paramGroups[0];
    if (g0) g0.options.lr = 0.5;

    expect(Array.from(checkpoint.state[0]?.state.momentumBuffer ?? [])).toEqual(savedBuffer);
    expect(checkpoint.paramGroups[0]?.options.lr).toBe(0.1);
  });

  it("loading does not alias the dictionary buffers into the optimizer", () => {
    const p = makeParam([1, 2, 3]);
    const opt = new SGD([p], { lr: 0.1, momentum: 0.9 });
    setGrad(p, [0.1, 0.2, 0.3]);
    opt.step();
    const checkpoint = opt.stateDict();

    const q = makeParam([1, 2, 3]);
    const other = new SGD([q], { lr: 0.1, momentum: 0.9 });
    other.loadStateDict(checkpoint);
    setGrad(q, [1, 1, 1]);
    other.step();
    // The dictionary still holds the buffer it was saved with.
    expect(Array.from(checkpoint.state[0]?.state.momentumBuffer ?? [])).toEqual([0.1, 0.2, 0.3]);
  });

  it("a failed load leaves groups and state untouched", () => {
    const p = makeParam([1, 2]);
    const opt = new SGD([p], { lr: 0.1, momentum: 0.9 });
    setGrad(p, [1, 1]);
    opt.step();
    const before = opt.stateDict();

    const bad = {
      paramGroups: [{ paramIds: [0], options: { lr: 0.9, momentum: 0.9 } }],
      state: [{ paramId: 99, state: {} }],
    };
    expect(() => opt.loadStateDict(bad)).toThrow(DataValidationError);
    const after = opt.stateDict();
    expect(after.paramGroups[0]?.options.lr).toBe(0.1);
    expect(after.state).toEqual(before.state);
  });

  it("rejects duplicate state entries and mismatching ids/params", () => {
    const a = makeParam([1]);
    const b = makeParam([2]);
    const opt = new SGD([a, b], { lr: 0.1 });
    const entry = { paramId: 0, state: {} };
    expect(() => opt.loadStateDict({ state: [entry, entry] })).toThrow(DataValidationError);
    expect(() => opt.loadStateDict({ state: [{ paramId: 0, param: b, state: {} }] })).toThrow(
      /does not match provided param/
    );
    expect(() =>
      opt.loadStateDict({
        paramGroups: [{ paramIds: [1, 0], params: [a, b], options: { lr: 0.1 } }],
      })
    ).toThrow(DataValidationError);
  });

  it("rejects arrays and non-objects where records are expected", () => {
    const opt = new SGD([makeParam([1])], { lr: 0.1 });
    expect(() => opt.loadStateDict({ paramGroups: [{ paramIds: [0], options: [] }] })).toThrow(
      DataValidationError
    );
    expect(() => opt.loadStateDict({ state: [{ paramId: 0, state: [] }] })).toThrow(
      DataValidationError
    );
    expect(() => opt.loadStateDict(null as unknown as Record<string, unknown>)).toThrow(
      DataValidationError
    );
  });

  it("requires params-only groups to cover every parameter without duplicates", () => {
    const a = makeParam([1]);
    const b = makeParam([2]);
    const opt = new SGD(
      [
        { params: [a], lr: 0.1 },
        { params: [b], lr: 0.2 },
      ],
      { lr: 0.5 }
    );
    expect(() =>
      opt.loadStateDict({
        paramGroups: [
          { params: [a], options: { lr: 0.1 } },
          { params: [a], options: { lr: 0.2 } },
        ],
      })
    ).toThrow(/Duplicate paramId/);
    expect(() =>
      opt.loadStateDict({
        paramGroups: [
          { params: [a], options: { lr: 0.1 } },
          { params: [makeParam([9])], options: { lr: 0.2 } },
        ],
      })
    ).toThrow(/not part of this optimizer/);
  });
});

describe("Optimizer construction", () => {
  it("keeps a parameter repeated inside one group once (tied weights)", () => {
    const w = makeParam([1]);
    const opt = new SGD([w, w], { lr: 0.1 });
    expect(opt.paramGroups[0]?.params).toHaveLength(1);
    setGrad(w, [1]);
    opt.step();
    expect(values(w)[0]).toBeCloseTo(0.9, 12);
  });

  it("rejects a parameter that belongs to two groups", () => {
    const w = makeParam([1]);
    expect(() => new SGD([{ params: [w] }, { params: [w], lr: 0.5 }], { lr: 0.1 })).toThrow(
      InvalidParameterError
    );
    const opt = new SGD([w], { lr: 0.1 });
    expect(() => opt.addParamGroup({ params: [w] })).toThrow(InvalidParameterError);
  });

  it("rejects non-parameter entries and malformed groups", () => {
    expect(() => new SGD([tensor([1, 2]) as unknown as GradTensor], { lr: 0.1 })).toThrow(
      InvalidParameterError
    );
    const w = makeParam([1]);
    const mixed = [{ params: [w] }, w] as unknown as ConstructorParameters<typeof SGD>[0];
    expect(() => new SGD(mixed, { lr: 0.1 })).toThrow(InvalidParameterError);
    const noIter = [{ params: 5 }] as unknown as ConstructorParameters<typeof SGD>[0];
    expect(() => new SGD(noIter, { lr: 0.1 })).toThrow(InvalidParameterError);
  });

  it("does not let an undefined group option erase the default", () => {
    const w = makeParam([1]);
    const opt = new SGD([{ params: [w], lr: undefined } as unknown as { params: GradTensor[] }], {
      lr: 0.1,
    });
    expect(opt.paramGroups[0]?.options.lr).toBe(0.1);
    setGrad(w, [1]);
    opt.step();
    expect(values(w)[0]).toBeCloseTo(0.9, 12);
  });

  it("checks group option types and NaN", () => {
    const w = makeParam([1]);
    const wrongType = [{ params: [w], lr: "0.1" }] as unknown as ConstructorParameters<
      typeof SGD
    >[0];
    expect(() => new SGD(wrongType, { lr: 0.1 })).toThrow(/expected number, got string/);
    expect(() => new SGD([{ params: [w], lr: Number.NaN }], { lr: 0.1 })).toThrow(
      InvalidParameterError
    );
  });

  it("copies array-valued options and defaults per group", () => {
    type Opts = { lr: number; betas: number[] };
    class Probe extends Optimizer<Opts, Record<string, never>> {
      step(): undefined {
        return undefined;
      }
      protected isState(state: Record<string, unknown>): state is Record<string, never> {
        return Object.keys(state).length === 0;
      }
    }
    const a = makeParam([1]);
    const b = makeParam([2]);
    const shared = [0.5, 0.6];
    const defaults: Opts = { lr: 0.1, betas: [0.9, 0.999] };
    const opt = new Probe([{ params: [a] }, { params: [b], betas: shared }], defaults);
    const g0 = opt.paramGroups[0]?.options;
    const g1 = opt.paramGroups[1]?.options;
    g0?.betas.push(1);
    expect(defaults.betas).toEqual([0.9, 0.999]);
    expect(g1?.betas).toEqual([0.5, 0.6]);
    expect(g1?.betas).not.toBe(shared);
    const sd = opt.stateDict();
    expect(sd.paramGroups[1]?.options.betas).not.toBe(g1?.betas);
  });

  it("accepts a string or null option when the default is null (LBFGS lineSearchFn)", () => {
    const w = makeParam([1]);
    const opt = new LBFGS([{ params: [w], lineSearchFn: "strong_wolfe" }], { lr: 1 });
    expect(opt.paramGroups[0]?.options.lineSearchFn).toBe("strong_wolfe");
    const fresh = new LBFGS([makeParam([1])], { lr: 1 });
    expect(() => fresh.loadStateDict(opt.stateDict())).not.toThrow();
    expect(fresh.paramGroups[0]?.options.lineSearchFn).toBe("strong_wolfe");
  });
});

describe("assertHasGradFloat", () => {
  it("rejects a gradient with the same size but a different shape", () => {
    const p = parameter(tensor([[1, 2, 3]], { dtype: "float64" }));
    p.setGrad(tensor([[1, 2, 3]], { dtype: "float64" }));
    // Bypass setGrad's shape check the way autograd internals can.
    Reflect.set(p, "_grad", tensor([[1], [2], [3]], { dtype: "float64" }));
    expect(() => assertHasGradFloat(p, "Test")).toThrow(ShapeError);
  });

  it("rejects a strided (transposed) parameter instead of updating the wrong elements", () => {
    const base = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" }
    );
    const p = parameter(transpose(base));
    p.setGrad(
      tensor(
        [
          [0, 0],
          [0, 0],
          [0, 0],
        ],
        { dtype: "float64" }
      )
    );
    expect(() => assertHasGradFloat(p, "Test")).toThrow(/contiguous/);
  });

  it("reads a strided gradient in logical order", () => {
    const p = parameter(
      tensor(
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
        { dtype: "float64" }
      )
    );
    const g = transpose(
      tensor(
        [
          [10, 20, 30],
          [40, 50, 60],
        ],
        { dtype: "float64" }
      )
    );
    p.setGrad(g);
    const info = assertHasGradFloat(p, "Test");
    expect(Array.from(info.grad.subarray(info.gradOffset, info.gradOffset + 6))).toEqual([
      10, 40, 20, 50, 30, 60,
    ]);
  });

  it("SGD updates with a transposed gradient in logical order", () => {
    const p = parameter(
      tensor(
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
        { dtype: "float64" }
      )
    );
    p.setGrad(
      transpose(
        tensor(
          [
            [10, 20, 30],
            [40, 50, 60],
          ],
          { dtype: "float64" }
        )
      )
    );
    new SGD([p], { lr: 1 }).step();
    expect(values(p)).toEqual([1 - 10, 2 - 40, 3 - 20, 4 - 50, 5 - 30, 6 - 60]);
  });

  it("states the valid range in non-negative and finite messages", () => {
    expect(() => new Adam([makeParam([1])], { lr: -1 })).toThrow(/must be >= 0/);
    expect(() => new Adam([makeParam([1])], { lr: Number.POSITIVE_INFINITY })).toThrow(
      /Invalid learning rate/
    );
  });
});
