/**
 * WebGPU backend integration tests. They run the real WGSL kernels via the
 * `webgpu` (Dawn) dev dependency. The whole suite skips itself when Dawn
 * is not installed or no GPU adapter is available, so CI without a GPU
 * stays green while machines with one verify the kernels end to end.
 */

import { describe, expect, it } from "vitest";
import { registerBackend, WebGpuBackend } from "../src/core";
import {
  add,
  div,
  dot,
  exp,
  GradTensor,
  gelu,
  max,
  mean,
  min,
  mul,
  parameter,
  pow,
  relu,
  sub,
  sum,
  tensor,
  transpose,
  where,
} from "../src/ndarray";
import { softmax as gradSoftmax } from "../src/ndarray/autograd/index";

async function tryCreateBackend(): Promise<WebGpuBackend | null> {
  try {
    const dawn = await import("webgpu");
    Object.assign(globalThis, dawn.globals);
    const backend = new WebGpuBackend({ gpu: dawn.create([]) });
    await backend.init();
    if (!backend.info().available) return null;
    return backend;
  } catch {
    return null;
  }
}

const backend = await tryCreateBackend();
if (backend) registerBackend("webgpu", backend);

describe.skipIf(!backend)("WebGPU WGSL kernels (real GPU)", () => {
  const dev = { device: "webgpu" as const };
  const flat = (t: unknown): number[] => [t].flat(Infinity) as number[];
  const cpu = async (t: { cpu(): Promise<{ toArray(): unknown }> }) =>
    flat((await t.cpu()).toArray());

  it("element-wise kernels with broadcast and strided views", async () => {
    const a = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      dev
    );
    const row = tensor([[10, 20, 30]], dev);
    expect(await cpu(add(a, row))).toEqual([11, 22, 33, 14, 25, 36]);
    expect(await cpu(sub(a, row))).toEqual([-9, -18, -27, -6, -15, -24]);
    expect(await cpu(mul(a, a))).toEqual([1, 4, 9, 16, 25, 36]);
    expect(await cpu(div(a, a))).toEqual([1, 1, 1, 1, 1, 1]);
    const tr = transpose(a);
    expect(await cpu(add(tr, tr))).toEqual([2, 8, 4, 10, 6, 12]);
  });

  it("pow handles negative bases with integral exponents", async () => {
    const base = tensor([-2, -3, 4], dev);
    const e = tensor([3, 2, 0.5], dev);
    const out = await cpu(pow(base, e));
    expect(out[0]).toBeCloseTo(-8, 5);
    expect(out[1]).toBeCloseTo(9, 5);
    expect(out[2]).toBeCloseTo(2, 5);
  });

  it("unary kernels", async () => {
    const t = tensor([-1.5, 0, 2], dev);
    expect(await cpu(relu(t))).toEqual([0, 0, 2]);
    const e = await cpu(exp(tensor([0, 1], dev)));
    expect(e[0]).toBeCloseTo(1, 5);
    expect(e[1]).toBeCloseTo(Math.E, 5);
  });

  it("matmul matches the CPU implementation on a 64x64 case", async () => {
    const N = 64;
    const rows = (f: (i: number) => number) =>
      Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => f(r * N + c)));
    const cpuA = tensor(rows((i) => Math.sin(i)));
    const cpuB = tensor(rows((i) => Math.cos(i * 0.5)));
    const ref = flat(dot(cpuA, cpuB).toArray());
    const got = await cpu(dot(await cpuA.to("webgpu"), await cpuB.to("webgpu")));
    for (let i = 0; i < ref.length; i++) {
      expect(Math.abs((got[i] ?? 0) - (ref[i] ?? 0))).toBeLessThan(1e-3);
    }
  });

  it("multi-pass reductions over 100k elements", async () => {
    const data = Array.from({ length: 100_000 }, (_, i) => (i % 7) - 3);
    const t = tensor(data, dev);
    const refSum = data.reduce((a, b) => a + b, 0);
    expect((await cpu(sum(t)))[0]).toBeCloseTo(refSum, 0);
    expect((await cpu(mean(t)))[0]).toBeCloseTo(refSum / data.length, 4);
    expect(await cpu(max(t))).toEqual([3]);
    expect(await cpu(min(t))).toEqual([-3]);
  });

  it("axis reductions (sum/mean/max/min) match the CPU result", async () => {
    const nested = [
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ],
      [
        [-1, -2, -3, -4],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
      ],
    ]; // [2,3,4]
    const g = tensor(nested, dev);
    const c = tensor(nested);
    for (const axis of [0, 1, 2]) {
      expect(await cpu(sum(g, axis))).toEqual(flat(sum(c, axis).toArray()));
      expect(await cpu(max(g, axis))).toEqual(flat(max(c, axis).toArray()));
      expect(await cpu(min(g, axis))).toEqual(flat(min(c, axis).toArray()));
      const gm = await cpu(mean(g, axis));
      const cm = flat(mean(c, axis).toArray());
      for (let i = 0; i < cm.length; i++) expect(gm[i]).toBeCloseTo(cm[i] ?? 0, 4);
    }
    // keepdims
    const kd = sum(g, 1, true);
    expect(kd.shape).toEqual([2, 1, 4]);
    expect(await cpu(kd)).toEqual(flat(sum(c, 1, true).toArray()));
  });

  it("batched matmul (attention-shaped) matches the CPU result", async () => {
    // [batch=2, heads=2, seq=3, dk=4] @ [.,.,4,3] -> [2,2,3,3]
    const mk = (n: number, f: (i: number) => number) => Array.from({ length: n }, (_, i) => f(i));
    const shape4 = (data: number[], s: number[]) => {
      const nest = (arr: number[], dims: number[]): unknown => {
        if (dims.length === 1) return arr;
        const [d, ...rest] = dims;
        const step = rest.reduce((a, b) => a * b, 1);
        return Array.from({ length: d ?? 0 }, (_, i) =>
          nest(arr.slice(i * step, (i + 1) * step), rest)
        );
      };
      return nest(data, s);
    };
    const qd = mk(2 * 2 * 3 * 4, (i) => Math.sin(i * 0.3));
    const kd = mk(2 * 2 * 3 * 4, (i) => Math.cos(i * 0.2));
    const q = tensor(shape4(qd, [2, 2, 3, 4]) as number[], dev);
    const k = tensor(shape4(kd, [2, 2, 3, 4]) as number[], dev);
    const qc = tensor(shape4(qd, [2, 2, 3, 4]) as number[]);
    const kc = tensor(shape4(kd, [2, 2, 3, 4]) as number[]);
    const scores = dot(q, transpose(k, [0, 1, 3, 2]));
    const scoresC = dot(qc, transpose(kc, [0, 1, 3, 2]));
    expect(scores.shape).toEqual([2, 2, 3, 3]);
    const got = await cpu(scores);
    const ref = flat(scoresC.toArray());
    for (let i = 0; i < ref.length; i++) expect(got[i]).toBeCloseTo(ref[i] ?? 0, 3);
  });

  it("gelu and where run on device and match the CPU result", async () => {
    const xs = [-2, -0.5, 0, 0.5, 2, 3];
    const gg = await cpu(gelu(tensor(xs, dev)));
    const gc = flat(gelu(tensor(xs)).toArray());
    for (let i = 0; i < gc.length; i++) expect(gg[i]).toBeCloseTo(gc[i] ?? 0, 5);

    const cond = tensor([1, 0, 1, 0], dev);
    const a = tensor([10, 20, 30, 40], dev);
    const b = tensor([-1, -2, -3, -4], dev);
    expect(await cpu(where(cond, a, b))).toEqual([10, -2, 30, -4]);
  });

  it("full MLP training step runs on the GPU with gradients matching the CPU", async () => {
    // matmul -> relu -> matmul -> softmax -> cross-entropy -> backward, entirely
    // on the GPU (forward + axis-reduction backward + broadcast-sum), verified
    // against the CPU autograd. This is the core transformer/MLP training loop.
    const W1data = [
      [0.1, -0.2, 0.3],
      [0.4, 0.5, -0.6],
    ];
    const W2data = [
      [0.2, -0.1],
      [0.3, 0.4],
      [-0.5, 0.6],
    ];
    const Xdata = [
      [1, -1],
      [0.5, 2],
      [-1, 0.3],
      [2, 1],
    ];
    const Ydata = [
      [1, 0],
      [0, 1],
      [1, 0],
      [0, 1],
    ];
    const run = async (onDevice: boolean) => {
      const mv = async (d: number[][]) => (onDevice ? await tensor(d).to("webgpu") : tensor(d));
      const W1 = parameter(await mv(W1data));
      const W2 = parameter(await mv(W2data));
      const X = GradTensor.fromTensor(await mv(Xdata), { requiresGrad: false });
      const Y = GradTensor.fromTensor(await mv(Ydata), { requiresGrad: false });
      const probs = gradSoftmax(X.matmul(W1).relu().matmul(W2), -1);
      const loss = Y.mul(probs.log()).sum().neg();
      loss.backward();
      return {
        gW1: flat(
          ((onDevice ? await W1.grad!.cpu() : W1.grad!) as { toArray(): unknown }).toArray()
        ),
        gW2: flat(
          ((onDevice ? await W2.grad!.cpu() : W2.grad!) as { toArray(): unknown }).toArray()
        ),
      };
    };
    const ref = await run(false);
    const got = await run(true);
    for (let i = 0; i < ref.gW1.length; i++) expect(got.gW1[i]).toBeCloseTo(ref.gW1[i] ?? 0, 3);
    for (let i = 0; i < ref.gW2.length; i++) expect(got.gW2[i]).toBeCloseTo(ref.gW2[i] ?? 0, 3);
  });

  it("softmax composes on device (max/sub/exp/sum/div) and matches the CPU result", async () => {
    // Manual softmax over the last axis, the exact composition the autograd
    // softmax uses, proving axis reductions + elementwise chain on device.
    const rows = [
      [1, 2, 3],
      [1, 1, 1],
      [-1, 0, 4],
    ];
    const softmaxLast = (t: ReturnType<typeof tensor>): ReturnType<typeof tensor> => {
      const m = max(t, 1, true);
      const e = exp(sub(t, m));
      return div(e, sum(e, 1, true));
    };
    const g = tensor(rows, dev);
    const c = tensor(rows);
    const got = await cpu(softmaxLast(g));
    const ref = flat(softmaxLast(c).toArray());
    for (let i = 0; i < ref.length; i++) expect(got[i]).toBeCloseTo(ref[i] ?? 0, 5);
    // rows sum to 1
    for (let r = 0; r < 3; r++) {
      expect((got[r * 3] ?? 0) + (got[r * 3 + 1] ?? 0) + (got[r * 3 + 2] ?? 0)).toBeCloseTo(1, 5);
    }
  });

  // ─── Half precision (float16 / bfloat16) ─────────────────────────────────
  //
  // float16 executes as true on-device half (WGSL `enable f16;`,
  // `array<f16>` storage); bfloat16 stores on-device as float32 rounded to
  // bfloat16 at upload and re-rounded at download. Both are compared against
  // the float32/CPU reference within half-precision tolerance.

  const f16ok = backend ? backend.supportsF16() : false;

  for (const dtype of ["float16", "bfloat16"] as const) {
    // Skip float16 hardware tests if the adapter lacks `shader-f16`; bfloat16
    // works regardless (it computes in float32 on-device).
    const runs = dtype === "bfloat16" || f16ok;
    // Relative tolerance: f16 ~1e-2..1e-3, bf16 ~1e-2 (7 mantissa bits).
    const rel = dtype === "bfloat16" ? 3e-2 : 1e-2;
    const closeAll = (got: number[], ref: number[]): void => {
      expect(got.length).toBe(ref.length);
      for (let i = 0; i < ref.length; i++) {
        const r = ref[i] ?? 0;
        expect(Math.abs((got[i] ?? 0) - r)).toBeLessThanOrEqual(Math.abs(r) * rel + 1e-2);
      }
    };

    it.skipIf(!runs)(`${dtype}: add/mul run on device and match float32`, async () => {
      const half = { device: "webgpu" as const, dtype };
      const aRows = [
        [1.5, 2.25, 3],
        [4, 5.5, 6.25],
      ];
      const bRows = [
        [0.5, 1.25, 2],
        [3, 4.5, 5.25],
      ];
      const a = tensor(aRows, half);
      const b = tensor(bRows, half);
      const aC = tensor(aRows);
      const bC = tensor(bRows);
      closeAll(await cpu(add(a, b)), flat(add(aC, bC).toArray()));
      closeAll(await cpu(mul(a, b)), flat(mul(aC, bC).toArray()));
    });

    it.skipIf(!runs)(`${dtype}: 32x32 matmul matches the CPU result`, async () => {
      const N = 32;
      const rows = (f: (i: number, j: number) => number): number[][] =>
        Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => f(i, j)));
      // Small magnitudes keep the accumulation inside f16/bf16 range.
      const aRows = rows((i, j) => ((i + j) % 5) * 0.25);
      const bRows = rows((i, j) => ((i * 2 + j) % 3) * 0.5);
      const half = { device: "webgpu" as const, dtype };
      const got = await cpu(dot(tensor(aRows, half), tensor(bRows, half)));
      const ref = flat(dot(tensor(aRows), tensor(bRows)).toArray());
      closeAll(got, ref);
    });

    it.skipIf(!runs)(`${dtype}: sum reduction matches the CPU result`, async () => {
      const vals = Array.from({ length: 300 }, (_, i) => ((i % 7) - 3) * 0.5);
      const half = { device: "webgpu" as const, dtype };
      const got = await cpu(sum(tensor(vals, half)));
      const ref = flat(sum(tensor(vals)).toArray());
      closeAll(got, ref);
    });

    it.skipIf(!runs)(`${dtype}: round-trips its dtype through cpu()`, async () => {
      const half = { device: "webgpu" as const, dtype };
      const t = tensor([1, 2, 3, 4], half);
      expect(t.dtype).toBe(dtype);
      expect(t.deviceBuffer?.dtype).toBe(dtype);
      const back = await t.cpu();
      expect(back.dtype).toBe(dtype);
      closeAll(flat(back.toArray()), [1, 2, 3, 4]);
    });
  }

  it.skipIf(!f16ok)("rejects mixed float16/float32 device ops with a clear error", () => {
    const a = tensor([1, 2, 3], { device: "webgpu", dtype: "float16" });
    const b = tensor([1, 2, 3], { device: "webgpu" });
    expect(() => add(a, b)).toThrow(/dtype/i);
  });
});
