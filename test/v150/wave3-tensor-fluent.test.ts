import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import {
  add,
  argsort,
  clip,
  cumsum,
  GradTensor,
  mul,
  parameter,
  Tensor,
  tensor,
} from "../../src/ndarray";

// Reference values computed with numpy 2.4 / torch 2.12 (float32 inputs).
const REF = JSON.parse(
  '{"A":[[0.004000000189989805,0.8960000276565552,-0.8220000267028809,-2.671999931335449],[-1.3639999628067017,-2.9749999046325684,0.18000000715255737,4.020999908447266],[-1.4769999980926514,-1.8609999418258667,1.4700000286102295,1.0709999799728394]],"B":[[0.10499999672174454,-0.9300000071525574],[-0.028999999165534973,0.6949999928474426],[-1.343999981880188,-0.4580000042915344],[-1.9010000228881836,-1.2899999618530273]],"C":[[2.3420000076293945,0.7350000143051147,1.7669999599456787,0.7710000276565552],[0.6570000052452087,0.6869999766349792,3.0169999599456787,1.0390000343322754],[0.5490000247955322,0.6129999756813049,2.0299999713897705,0.9779999852180481]],"sum0":[-2.8369998931884766,-3.940000057220459,0.828000009059906,2.4200000762939453],"sum1k":[[-2.5939998626708984],[-0.1380000114440918],[-0.7969998121261597]],"sumall":-3.5289993286132812,"mean1":[-0.6484999656677246,-0.03450000286102295,-0.19924995303153992],"max0":[0.004000000189989805,0.8960000276565552,1.4700000286102295,4.020999908447266],"min1":[-2.671999931335449,-2.9749999046325684,-1.8609999418258667],"prod1":[0.007871841080486774,2.9370269775390625,4.327466011047363],"std0":[0.6730984449386597,1.6270861625671387,0.9381641745567322,2.7387912273406982],"var1_ddof1":[2.3119635581970215,8.969066619873047,2.9313297271728516],"argmax1":[1.0,3.0,2.0],"argmin0":[2.0,1.0,0.0,0.0],"argmaxAll":7,"cumsum1":[[0.004000000189989805,0.9000000357627869,0.078000009059906,-2.5939998626708984],[-1.3639999628067017,-4.3389997482299805,-4.158999919891357,-0.1380000114440918],[-1.4769999980926514,-3.3379998207092285,-1.867999792098999,-0.7969998121261597]],"cumsumAll":[0.004000000189989805,0.9000000357627869,0.078000009059906,-2.5939998626708984,-3.9579997062683105,-6.932999610900879,-6.752999782562256,-2.7319998741149902,-4.2089996337890625,-6.069999694824219,-4.59999942779541,-3.5289993286132812],"softmax1":[[0.2533801198005676,0.618248701095581,0.11092904210090637,0.01744217239320278],[0.004464424680918455,0.0008914911886677146,0.02090817503631115,0.9737358689308167],[0.0298406220972538,0.02032538689672947,0.568425714969635,0.38140836358070374]],"softmax0":[[0.6747520565986633,0.9222374558448792,0.0734298899769783,0.0011765215313062072],[0.17180246114730835,0.019217144697904587,0.2000027745962143,0.9491455554962158],[0.15344548225402832,0.05854541435837746,0.7265673279762268,0.0496780090034008]],"logsoftmax1":[[-1.3728644847869873,-0.4808644652366638,-2.198864459991455,-4.048864364624023],[-5.411614894866943,-7.0226149559021,-3.867614984512329,-0.026615185663104057],[-3.5118846893310547,-3.8958845138549805,-0.5648847222328186,-0.9638847708702087]],"sigmoid":[[0.5009999871253967,0.7101268172264099,0.3053392767906189,0.0646459311246872],[0.2035909742116928,0.048568155616521835,0.5448789000511169,0.9823809862136841],[0.18588098883628845,0.13458654284477234,0.8130573630332947,0.7447870373725891]],"tanh":[[0.003999978769570589,0.7143446207046509,-0.6761569976806641,-0.9904919862747192],[-0.8773175477981567,-0.9948018789291382,0.17808087170124054,0.9993568658828735],[-0.9009044170379639,-0.9527711272239685,0.8995774388313293,0.789837658405304]],"relu":[[0.004000000189989805,0.8960000276565552,0.0,0.0],[0.0,0.0,0.18000000715255737,4.020999908447266],[0.0,0.0,1.4700000286102295,1.0709999799728394]],"exp":[[1.0040080547332764,2.449784517288208,0.4395516514778137,0.06911386549472809],[0.25563618540763855,0.051047440618276596,1.197217345237732,55.75682830810547],[0.2283216267824173,0.15551704168319702,4.3492350578308105,2.9182963371276855]],"sinA":[[0.003999989479780197,0.7808342576026917,-0.7325088381767273,-0.4525231420993805],[-0.9786937236785889,-0.16582323610782623,0.17902958393096924,-0.770361065864563],[-0.9956043362617493,-0.958185613155365,0.9949243664741516,0.8776801824569702]],"cosA":[[0.9999920129776001,0.6247382760047913,0.6807575225830078,-0.8917526602745056],[0.20532557368278503,-0.9861555099487305,0.9838436841964722,-0.6376078724861145],[0.09365885704755783,-0.2861473262310028,0.10062570124864578,0.4792468249797821]],"sqrtC":[[1.5303593873977661,0.8573214411735535,1.3292855024337769,0.8780660629272461],[0.8105553388595581,0.8288546204566956,1.7369513511657715,1.0193134546279907],[0.7409453392028809,0.7829431295394897,1.4247807264328003,0.9889388084411621]],"logC":[[0.8510052561759949,-0.30788475275039673,0.5692831873893738,-0.2600668668746948],[-0.42007124423980713,-0.37542101740837097,1.1042629480361938,0.03825874626636505],[-0.5996568202972412,-0.48939037322998047,0.7080357670783997,-0.0222456231713295]],"square":[[1.6000001778593287e-05,0.8028160333633423,0.6756840348243713,7.139583587646484],[1.860495924949646,8.850624084472656,0.03240000084042549,16.168439865112305],[2.1815290451049805,3.463320732116699,2.160900115966797,1.1470409631729126]],"matmul":[[6.158675670623779,4.442355632781982],[-7.942785739898682,-6.068634510040283],[-4.112767219543457,-1.9746348857879639]],"divC":[[0.0017079419922083616,1.2190476655960083,-0.46519526839256287,-3.4656288623809814],[-2.076103448867798,-4.3304219245910645,0.05966191738843918,3.8700671195983887],[-2.6903460025787354,-3.035889148712158,0.7241379618644714,1.0950920581817627]],"powC":[[3.584101915359497,0.6301312446594238,2.3488473892211914,0.676988959312439],[0.5325348973274231,0.5694230794906616,5.240382194519043,1.0590667724609375],[0.406779021024704,0.4799441397190094,2.8923046588897705,0.9671821594238281]],"sub":[[-2.3380000591278076,0.16100001335144043,-2.5889999866485596,-3.442999839782715],[-2.0209999084472656,-3.6619999408721924,-2.8369998931884766,2.9819998741149902],[-2.0260000228881836,-2.4739999771118164,-0.559999942779541,0.09299999475479126]],"clip":[[0.004000000189989805,0.8960000276565552,-0.8220000267028809,-1.0],[-1.0,-1.0,0.18000000715255737,2.0],[-1.0,-1.0,1.4700000286102295,1.0709999799728394]],"sort1d":[[0.8960000276565552,0.004000000189989805,-0.8220000267028809,-2.671999931335449],[4.020999908447266,0.18000000715255737,-1.3639999628067017,-2.9749999046325684],[1.4700000286102295,1.0709999799728394,-1.4769999980926514,-1.8609999418258667]],"argsort0":[[2.0,1.0,0.0,0.0],[1.0,2.0,1.0,2.0],[0.0,0.0,2.0,1.0]],"flip1":[[-2.671999931335449,-0.8220000267028809,0.8960000276565552,0.004000000189989805],[4.020999908447266,0.18000000715255737,-2.9749999046325684,-1.3639999628067017],[1.0709999799728394,1.4700000286102295,-1.8609999418258667,-1.4769999980926514]],"flipAll":[[1.0709999799728394,1.4700000286102295,-1.8609999418258667,-1.4769999980926514],[4.020999908447266,0.18000000715255737,-2.9749999046325684,-1.3639999628067017],[-2.671999931335449,-0.8220000267028809,0.8960000276565552,0.004000000189989805]],"gt":[[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,0.0,1.0]],"le1":[[1.0,1.0,1.0,1.0],[1.0,1.0,1.0,0.0],[1.0,1.0,0.0,0.0]],"round":[0.0,2.0,2.0,-0.0,-2.0,3.0,1235.0],"round2":[2.68,1.0,-3.14,0.12],"ceil":[[1.0,1.0,-0.0,-2.0],[-1.0,-2.0,1.0,5.0],[-1.0,-1.0,2.0,2.0]],"floor":[[0.0,0.0,-1.0,-3.0],[-2.0,-3.0,0.0,4.0],[-2.0,-2.0,1.0,1.0]],"powScalar":[[8.393966674804688,0.46314647793769836,4.150413513183594,0.5219585299491882],[0.34987542033195496,0.3911936581134796,15.810233116149902,1.1003704071044922],[0.22332169115543365,0.29420575499534607,5.871378421783447,0.9459041357040405]]}'
) as Record<string, unknown>;

function ref2(name: string): number[][] {
  return REF[name] as number[][];
}
function ref1(name: string): number[] {
  return REF[name] as number[];
}
function flat(t: Tensor): number[] {
  const value = t.toArray() as number | number[] | number[][];
  return Array.isArray(value) ? (value as number[][]).flat(5) : [value];
}
function expectClose(t: Tensor, expected: unknown, tol = 1e-4): void {
  const e = (Array.isArray(expected) ? (expected as unknown[]).flat(5) : [expected]) as number[];
  const got = flat(t.ndim === 0 ? t.reshape([1]) : t);
  expect(got.length).toBe(e.length);
  for (let i = 0; i < e.length; i++) {
    const want = e[i] as number;
    const have = got[i] as number;
    expect(Math.abs(have - want)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(want)));
  }
}

const A = tensor(ref2("A"));
const B = tensor(ref2("B"));
const C = tensor(ref2("C"));

describe("Tensor fluent arithmetic", () => {
  it("add/sub/mul/div/pow with tensors match numpy", () => {
    expectClose(A.sub(C), ref2("sub"));
    expectClose(A.div(C), ref2("divC"));
    expectClose(C.pow(tensor(1.5)), ref2("powC"));
    expectClose(A.add(C), add(A, C).toArray());
    expectClose(A.mul(C), mul(A, C).toArray());
  });

  it("number operands match the scalar rules and numpy", () => {
    expectClose(C.pow(2.5), ref2("powScalar"));
    expectClose(
      A.add(1.5),
      (ref2("A") as number[][]).map((r) => r.map((v) => v + 1.5))
    );
    expectClose(
      A.mul(-2),
      (ref2("A") as number[][]).map((r) => r.map((v) => v * -2))
    );
    expectClose(
      A.sub(0.5),
      (ref2("A") as number[][]).map((r) => r.map((v) => v - 0.5))
    );
    expectClose(
      A.div(4),
      (ref2("A") as number[][]).map((r) => r.map((v) => v / 4))
    );
  });

  it("broadcasts like the functional ops", () => {
    const row = tensor([1, 2, 3, 4]);
    expect(A.add(row).shape).toEqual([3, 4]);
    expect(() => A.add(tensor([1, 2, 3]))).toThrow(ShapeError);
  });

  it("integer dtype rules match PyTorch", () => {
    const i = tensor([1, 2, 3], { dtype: "int32" });
    expect(i.add(2).dtype).toBe("int32");
    expect(i.add(2.5).dtype).toBe("float32");
    expect(i.sub(1).dtype).toBe("int32");
    expect(i.sub(1).toArray()).toEqual([0, 1, 2]);
    expect(i.mul(3).dtype).toBe("int32");
    expect(i.div(2).dtype).toBe("float32");
    expect(i.div(2).toArray()).toEqual([0.5, 1, 1.5]);
    expect(i.pow(2).dtype).toBe("int32");
    expect(i.pow(2).toArray()).toEqual([1, 4, 9]);
    expect(tensor([true, false], { dtype: "bool" }).add(1).dtype).toBe("int32");
    expect(tensor([1, 2], { dtype: "uint8" }).sub(1).dtype).toBe("uint8");
  });

  it("int64 tensors take BigInt-safe scalars", () => {
    const t = tensor([1, 2, 3], { dtype: "int64" });
    expect(t.sub(1).dtype).toBe("int64");
    expect(t.sub(1).toArray()).toEqual([0n, 1n, 2n]);
    expect(t.gt(1).toArray()).toEqual([0, 1, 1]);
  });

  it("float64 and half dtypes keep their dtype with number operands", () => {
    expect(tensor([1, 2], { dtype: "float64" }).div(3).dtype).toBe("float64");
    expect(tensor([1, 2], { dtype: "float16" }).sub(1).dtype).toBe("float16");
    expect(tensor([1, 2], { dtype: "bfloat16" }).pow(2).dtype).toBe("bfloat16");
    expect(tensor([0.1], { dtype: "float64" }).sub(0.1).item()).toBe(0);
  });

  it("subtracting a number keeps signed zeros, NaN and infinities", () => {
    const y = tensor([0, -0, Number.NaN, 1], { dtype: "float64" });
    const z = y.sub(0).toArray() as number[];
    expect(Object.is(z[0], 0)).toBe(true);
    expect(Object.is(z[1], -0)).toBe(true);
    expect(Number.isNaN(z[2])).toBe(true);
    expect(y.sub(Number.NEGATIVE_INFINITY).toArray()).toEqual([
      Infinity,
      Infinity,
      Number.NaN,
      Infinity,
    ]);
    expect(tensor([1.5, 2.5], { dtype: "float32" }).sub(0.5).toArray()).toEqual([1, 2]);
  });

  it("rejects bad operands", () => {
    expect(() => A.sub("x" as unknown as number)).toThrow(InvalidParameterError);
    const s = tensor(["a", "b"]);
    expect(() => s.add(1)).toThrow(DTypeError);
    expect(() => s.sub(1)).toThrow(DTypeError);
  });

  it("neg/abs/square/sqrt/exp/log", () => {
    expectClose(
      A.neg(),
      ref2("A").map((r) => r.map((v) => -v))
    );
    expectClose(
      A.abs(),
      ref2("A").map((r) => r.map(Math.abs))
    );
    expectClose(A.square(), ref2("square"));
    expectClose(C.sqrt(), ref2("sqrtC"));
    expectClose(C.log(), ref2("logC"));
    expectClose(A.exp(), ref2("exp"));
    expect(Number.isNaN(tensor([-1]).sqrt().item())).toBe(true);
    expect(tensor([0]).log().item()).toBe(Number.NEGATIVE_INFINITY);
  });

  it("sin/cos/tanh/sigmoid/relu match numpy and torch", () => {
    expectClose(A.sin(), ref2("sinA"));
    expectClose(A.cos(), ref2("cosA"));
    expectClose(A.tanh(), ref2("tanh"));
    expectClose(A.sigmoid(), ref2("sigmoid"));
    expectClose(A.relu(), ref2("relu"));
  });

  it("softmax and logSoftmax match torch, with negative axes", () => {
    expectClose(A.softmax(), ref2("softmax1"));
    expectClose(A.softmax(-1), ref2("softmax1"));
    expectClose(A.softmax(1), ref2("softmax1"));
    expectClose(A.softmax(0), ref2("softmax0"));
    expectClose(A.softmax(-2), ref2("softmax0"));
    expectClose(A.logSoftmax(1), ref2("logsoftmax1"));
    expectClose(A.logSoftmax(), ref2("logsoftmax1"));
  });

  it("clip with both, one or no bound", () => {
    expectClose(A.clip(-1, 2), ref2("clip"));
    expect(tensor([-5, 5]).clip(undefined, 1).toArray()).toEqual([-5, 1]);
    expect(tensor([-5, 5]).clip(0).toArray()).toEqual([0, 5]);
    expect(tensor([-5, 5]).clip().toArray()).toEqual([-5, 5]);
    expect(() => tensor([1]).clip(2, 1)).toThrow();
    expect(clip(A, -1, 2).toArray()).toEqual(A.clip(-1, 2).toArray());
  });

  it("round, floor, ceil match numpy (half to even)", () => {
    const r = tensor([0.5, 1.5, 2.5, -0.5, -1.5, 2.675, 1234.5678], { dtype: "float64" });
    expectClose(r.round(), ref1("round"));
    const d = tensor([2.675, 1.005, -3.14259, 0.1234], { dtype: "float64" });
    expectClose(d.round(2), ref1("round2"));
    expectClose(A.floor(), ref2("floor"));
    expectClose(A.ceil(), ref2("ceil"));
    expect(() => A.round(1.5)).toThrow(InvalidParameterError);
  });
});

describe("Tensor fluent reductions", () => {
  it("sum with axis, negative axis, keepdims and axis lists", () => {
    expectClose(A.sum(0), ref1("sum0"));
    expectClose(A.sum(-2), ref1("sum0"));
    expectClose(A.sum(), REF["sumall"]);
    const k = A.sum(1, true);
    expect(k.shape).toEqual([3, 1]);
    expectClose(k, ref2("sum1k"));
    expect(A.sum([0, 1]).item()).toBeCloseTo(REF["sumall"] as number, 4);
    expect(A.sum([0, 1], true).shape).toEqual([1, 1]);
  });

  it("mean/max/min/prod/std/var match numpy", () => {
    expectClose(A.mean(1), ref1("mean1"));
    expectClose(A.max(0), ref1("max0"));
    expectClose(A.min(1), ref1("min1"));
    expectClose(A.prod(1), ref1("prod1"));
    expectClose(A.std(0), ref1("std0"));
    expectClose(A.var(1, false, 1), ref1("var1_ddof1"));
    expect(A.var().item()).toBeCloseTo(
      (() => {
        const v = ref2("A").flat();
        const m = v.reduce((a, b) => a + b, 0) / v.length;
        return v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length;
      })(),
      3
    );
  });

  it("argmax/argmin/cumsum match numpy", () => {
    expectClose(A.argmax(1), ref1("argmax1"));
    expectClose(A.argmin(0), ref1("argmin0"));
    expect(A.argmax().item()).toBe(REF["argmaxAll"]);
    expect(A.argmax(1, true).shape).toEqual([3, 1]);
    expectClose(A.cumsum(1), ref2("cumsum1"));
    expectClose(A.cumsum(), ref1("cumsumAll"));
    expectClose(A.cumsum(-1), ref2("cumsum1"));
    expect(cumsum(A, 0).toArray()).toEqual(A.cumsum(0).toArray());
  });

  it("any and all, with axes", () => {
    const t = tensor([
      [0, 1],
      [0, 0],
    ]);
    expect(t.any().item()).toBe(1);
    expect(t.all().item()).toBe(0);
    expect(t.any(1).toArray()).toEqual([1, 0]);
    expect(t.all(0).toArray()).toEqual([0, 0]);
    expect(t.any(-1, true).shape).toEqual([2, 1]);
  });

  it("empty and 0-d tensors behave like the functional ops", () => {
    expect(tensor([], { dtype: "float32" }).sum().item()).toBe(0);
    expect(tensor(5).sum().item()).toBe(5);
    expect(tensor(5).mean().item()).toBe(5);
    expect(tensor([]).prod().item()).toBe(1);
    expect(Number.isNaN(tensor([]).mean().item())).toBe(true);
  });

  it("NaN propagates through max, min and sum", () => {
    const t = tensor([1, Number.NaN, 3]);
    expect(Number.isNaN(t.max().item())).toBe(true);
    expect(Number.isNaN(t.min().item())).toBe(true);
    expect(Number.isNaN(t.sum().item())).toBe(true);
  });
});

describe("Tensor fluent linear algebra and shape", () => {
  it("matmul and dot", () => {
    expectClose(A.matmul(B), ref2("matmul"));
    expect(
      tensor([1, 2, 3])
        .dot(tensor([4, 5, 6]))
        .item()
    ).toBe(32);
    expectClose(A.dot(B), ref2("matmul"));
    expect(() => A.matmul(A)).toThrow(ShapeError);
  });

  it("transpose, T and views", () => {
    expect(A.transpose().shape).toEqual([4, 3]);
    expect(A.T.shape).toEqual([4, 3]);
    expect(A.T.T.toArray()).toEqual(A.toArray());
    const cube = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    expect(cube.transpose([2, 0, 1]).shape).toEqual([2, 2, 2]);
    expect(cube.T.shape).toEqual([2, 2, 2]);
    expect(tensor([1, 2, 3]).T.shape).toEqual([3]);
    expect(tensor(4).T.shape).toEqual([]);
    expect(A.T.matmul(A).shape).toEqual([4, 4]);
  });

  it("works on non-contiguous views", () => {
    const v = A.T; // strided view of A, shape [4, 3]
    const rows = ref2("A");
    const colSums = [0, 1, 2, 3].map((j) => rows.reduce((acc, r) => acc + (r[j] as number), 0));
    expectClose(v.sum(1), colSums);
    expectClose(
      v.add(1),
      [0, 1, 2, 3].map((j) => rows.map((r) => (r[j] as number) + 1))
    );
    expectClose(
      v.mul(2).T,
      rows.map((r) => r.map((x) => x * 2))
    );
    expectClose(
      v.max(0),
      rows.map((r) => Math.max(...r))
    );
    expectClose(v.softmax(0), A.softmax(1).T.toArray());
    // v.cumsum(0)[j][i] is the running sum of row i of A up to column j.
    const running = [0, 1, 2, 3].map((j) =>
      rows.map((r) => r.slice(0, j + 1).reduce((acc, x) => acc + x, 0))
    );
    expectClose(v.cumsum(0), running);
    expectClose(v.sort(0), A.sort(1).T.toArray());
  });

  it("squeeze and unsqueeze with negative axes", () => {
    const t = tensor([[1, 2, 3]]);
    expect(t.squeeze().shape).toEqual([3]);
    expect(t.squeeze(0).shape).toEqual([3]);
    expect(t.squeeze(-2).shape).toEqual([3]);
    expect(t.squeeze([0]).shape).toEqual([3]);
    expect(t.unsqueeze(0).shape).toEqual([1, 1, 3]);
    expect(t.unsqueeze(-1).shape).toEqual([1, 3, 1]);
    expect(tensor(1).unsqueeze(0).shape).toEqual([1]);
    expect(() => t.unsqueeze(5)).toThrow();
  });

  it("clone copies the storage", () => {
    const t = tensor([1, 2, 3]);
    const c = t.clone();
    c.fill(9);
    expect(t.toArray()).toEqual([1, 2, 3]);
    expect(c.toArray()).toEqual([9, 9, 9]);
    const strided = A.T.clone();
    expect(strided.shape).toEqual([4, 3]);
    expect(strided.toArray()).toEqual(A.T.toArray());
  });

  it("flip, sort and argsort match numpy", () => {
    expectClose(A.flip(1), ref2("flip1"));
    expectClose(A.flip(), ref2("flipAll"));
    expectClose(A.flip([0, 1]), ref2("flipAll"));
    expectClose(A.flip(-1), ref2("flip1"));
    expectClose(A.sort(1, true), ref2("sort1d"));
    expectClose(A.argsort(0), ref2("argsort0"));
    expect(tensor([3, 1, 2]).sort().toArray()).toEqual([1, 2, 3]);
    expect(tensor([3, 1, 2]).argsort(-1, true).toArray()).toEqual([0, 2, 1]);
    expect(argsort(A, 0).toArray()).toEqual(A.argsort(0).toArray());
    expect(flat(tensor([3, Number.NaN, 1]).sort()).map(String)).toEqual(["1", "3", "NaN"]);
  });
});

describe("Tensor fluent comparisons", () => {
  it("eq/ne/gt/ge/lt/le with tensors and numbers", () => {
    expectClose(A.gt(C), ref2("gt"));
    expectClose(A.le(1), ref2("le1"));
    const t = tensor([1, 2, 3]);
    expect(t.eq(2).toArray()).toEqual([0, 1, 0]);
    expect(t.ne(2).toArray()).toEqual([1, 0, 1]);
    expect(t.gt(2).toArray()).toEqual([0, 0, 1]);
    expect(t.ge(2).toArray()).toEqual([0, 1, 1]);
    expect(t.lt(2).toArray()).toEqual([1, 0, 0]);
    expect(t.le(2).toArray()).toEqual([1, 1, 0]);
    expect(t.eq(tensor([1, 5, 3])).toArray()).toEqual([1, 0, 1]);
    expect(t.eq(2).dtype).toBe("bool");
  });

  it("integer tensors compare exactly with fractional and large numbers", () => {
    const i = tensor([1, 2], { dtype: "int32" });
    expect(i.gt(1.5).toArray()).toEqual([0, 1]);
    expect(i.lt(2 ** 40).toArray()).toEqual([1, 1]);
    const u = tensor([1, 255], { dtype: "uint8" });
    expect(u.lt(300).toArray()).toEqual([1, 1]);
    expect(u.eq(255).toArray()).toEqual([0, 1]);
  });

  it("NaN compares false except for ne", () => {
    const t = tensor([Number.NaN, 1]);
    expect(t.eq(Number.NaN).toArray()).toEqual([0, 0]);
    expect(t.ne(Number.NaN).toArray()).toEqual([1, 1]);
    expect(t.gt(0).toArray()).toEqual([0, 1]);
    expect(t.isnan().toArray()).toEqual([1, 0]);
    expect(tensor([1, 2], { dtype: "int32" }).isnan().toArray()).toEqual([0, 0]);
  });
});

describe("Tensor.fill", () => {
  it("fills in place and returns the tensor", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(t.fill(7)).toBe(t);
    expect(t.toArray()).toEqual([
      [7, 7],
      [7, 7],
    ]);
  });

  it("converts by dtype", () => {
    expect(tensor([1, 2], { dtype: "int32" }).fill(2.9).toArray()).toEqual([2, 2]);
    expect(tensor([1, 2], { dtype: "bool" }).fill(5).toArray()).toEqual([1, 1]);
    expect(tensor([1, 2], { dtype: "bool" }).fill(0).toArray()).toEqual([0, 0]);
    expect(tensor([1, 2], { dtype: "bool" }).fill(true).toArray()).toEqual([1, 1]);
    expect(tensor([1, 2], { dtype: "bool" }).fill(5).dtype).toBe("bool");
    expect(tensor([1, 2], { dtype: "int64" }).fill(5).toArray()).toEqual([5n, 5n]);
    expect(tensor([1, 2], { dtype: "int64" }).fill(7n).toArray()).toEqual([7n, 7n]);
    expect(tensor([1, 2], { dtype: "uint8" }).fill(257).toArray()).toEqual([1, 1]);
    expect(tensor([1, 2]).fill(0.1).toArray()).toEqual([Math.fround(0.1), Math.fround(0.1)]);
    expect(tensor(["a", "b"]).fill("z").toArray()).toEqual(["z", "z"]);
    expect(tensor([1], { dtype: "float16" }).fill(1.0001).item()).toBe(1);
    expect(tensor([1], { dtype: "bfloat16" }).fill(1.001).item()).toBe(1);
  });

  it("writes through strided views and leaves other cells alone", () => {
    const base = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    base.T.fill(0); // transposed view shares the buffer
    expect(base.toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);

    const buf = new Float32Array([1, 2, 3, 4, 5, 6]);
    const col = Tensor.fromTypedArray({
      data: buf,
      shape: [2],
      dtype: "float32",
      device: "cpu",
      strides: [3],
      offset: 1,
    });
    col.fill(9);
    expect(Array.from(buf)).toEqual([1, 9, 3, 4, 9, 6]);

    const matrixView = Tensor.fromTypedArray({
      data: new Int32Array([1, 2, 3, 4, 5, 6, 7, 8]),
      shape: [2, 2],
      dtype: "int32",
      device: "cpu",
      strides: [4, 2],
      offset: 0,
    });
    matrixView.fill(0);
    expect(Array.from(matrixView.data as Int32Array)).toEqual([0, 2, 0, 4, 0, 6, 0, 8]);
  });

  it("handles empty and 0-d tensors", () => {
    expect(tensor([], { dtype: "float32" }).fill(1).size).toBe(0);
    expect(tensor(3).fill(4).item()).toBe(4);
  });

  it("rejects mismatched value types", () => {
    expect(() => tensor([1, 2]).fill("a")).toThrow(DTypeError);
    expect(() => tensor(["a"]).fill(1)).toThrow(DTypeError);
    expect(() => tensor([1], { dtype: "int64" }).fill(Number.NaN)).toThrow(DTypeError);
  });
});

describe("Tensor and GradTensor surface", () => {
  const names = [
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "square",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "relu",
    "softmax",
    "logSoftmax",
    "sum",
    "mean",
    "max",
    "min",
    "prod",
    "std",
    "var",
    "argmax",
    "argmin",
    "cumsum",
    "matmul",
    "dot",
    "transpose",
    "squeeze",
    "unsqueeze",
    "clone",
    "fill",
    "clip",
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "isnan",
    "any",
    "all",
    "round",
    "floor",
    "ceil",
    "sort",
    "argsort",
    "flip",
  ] as const;

  it("exposes every fluent method on Tensor", () => {
    const t = tensor([1, 2]);
    for (const n of names)
      expect(typeof (t as unknown as Record<string, unknown>)[n]).toBe("function");
    expect(t.T).toBeInstanceOf(Tensor);
  });

  it("overlapping GradTensor methods agree with Tensor values", () => {
    const x = tensor([
      [1, -2],
      [3, 4],
    ]);
    const g = parameter([
      [1, -2],
      [3, 4],
    ]);
    expect(g.sum(0).tensor.toArray()).toEqual(x.sum(0).toArray());
    expect(g.mean(1).tensor.toArray()).toEqual(x.mean(1).toArray());
    expect(g.softmax(0).tensor.toArray()).toEqual(x.softmax(0).toArray());
    expect(g.relu().tensor.toArray()).toEqual(x.relu().toArray());
    expect(g.clip(-1, 2).tensor.toArray()).toEqual(x.clip(-1, 2).toArray());
    expect(g.transpose().tensor.toArray()).toEqual(x.transpose().toArray());
    expect(g.T.tensor.toArray()).toEqual(x.T.toArray());
    expect(g.var(0, false, 1).tensor.toArray()).toEqual(x.var(0, false, 1).toArray());
    expect(g).toBeInstanceOf(GradTensor);
  });
});
