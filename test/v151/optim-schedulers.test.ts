// NN-11 and NN-12: MultiStepLR accepts duplicate milestones and OneCycleLR / CyclicLR cycle the
// momentum (or beta1) by default. Reference sequences come from torch.optim.lr_scheduler
// (torch 2.15): MultiStepLR, OneCycleLR and CyclicLR over SGD and Adam.
import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { parameter, tensor } from "../../src/ndarray";
import { Adam, CyclicLR, MultiStepLR, OneCycleLR, SGD } from "../../src/optim";

type Row = { lr: number; momentum: number | null; beta1: number | null };
const REF = JSON.parse(
  `{"multi":[1.0,1.0,0.01,0.01,0.01,0.001,0.001,0.001],"onecycle_sgd":[{"lr":0.04,"momentum":0.95,"beta1":null},{"lr":0.52,"momentum":0.9,"beta1":null},{"lr":1.0,"momentum":0.85,"beta1":null},{"lr":0.9504846320135,"momentum":0.8549515566049,"beta1":null},{"lr":0.8117456539498,"momentum":0.8688255099071,"beta1":null},{"lr":0.6112620219363,"momentum":0.8888739533022,"beta1":null},{"lr":0.3887419780637,"momentum":0.9111260466978,"beta1":null},{"lr":0.1882583460502,"momentum":0.9311744900929,"beta1":null},{"lr":0.04951936798653,"momentum":0.9450484433951,"beta1":null},{"lr":4e-06,"momentum":0.95,"beta1":null}],"onecycle_sgd_params":[{"lr":0.04,"momentum":0.95,"beta1":null},{"lr":0.52,"momentum":0.825,"beta1":null},{"lr":1.0,"momentum":0.7,"beta1":null},{"lr":0.9504846320135,"momentum":0.7123788915122,"beta1":null},{"lr":0.8117456539498,"momentum":0.7470637747677,"beta1":null},{"lr":0.6112620219363,"momentum":0.7971848832555,"beta1":null},{"lr":0.3887419780637,"momentum":0.8528151167445,"beta1":null},{"lr":0.1882583460502,"momentum":0.9029362252323,"beta1":null},{"lr":0.04951936798653,"momentum":0.9376211084878,"beta1":null},{"lr":4e-06,"momentum":0.95,"beta1":null}],"onecycle_nomom":[{"lr":0.04,"momentum":0.9,"beta1":null},{"lr":0.52,"momentum":0.9,"beta1":null},{"lr":1.0,"momentum":0.9,"beta1":null},{"lr":0.9504846320135,"momentum":0.9,"beta1":null},{"lr":0.8117456539498,"momentum":0.9,"beta1":null},{"lr":0.6112620219363,"momentum":0.9,"beta1":null},{"lr":0.3887419780637,"momentum":0.9,"beta1":null},{"lr":0.1882583460502,"momentum":0.9,"beta1":null},{"lr":0.04951936798653,"momentum":0.9,"beta1":null},{"lr":4e-06,"momentum":0.9,"beta1":null}],"onecycle_adam":[{"lr":0.04,"momentum":null,"beta1":0.95},{"lr":0.52,"momentum":null,"beta1":0.9},{"lr":1.0,"momentum":null,"beta1":0.85},{"lr":0.8571434285714,"momentum":null,"beta1":0.8642857142857},{"lr":0.7142868571429,"momentum":null,"beta1":0.8785714285714},{"lr":0.5714302857143,"momentum":null,"beta1":0.8928571428571},{"lr":0.4285737142857,"momentum":null,"beta1":0.9071428571429},{"lr":0.2857171428571,"momentum":null,"beta1":0.9214285714286},{"lr":0.1428605714286,"momentum":null,"beta1":0.9357142857143},{"lr":4.000000000004e-06,"momentum":null,"beta1":0.95}],"onecycle_sgd_nomomentum":[{"lr":0.04,"momentum":0.95,"beta1":null},{"lr":0.52,"momentum":0.9,"beta1":null},{"lr":1.0,"momentum":0.85,"beta1":null},{"lr":0.9504846320135,"momentum":0.8549515566049,"beta1":null},{"lr":0.8117456539498,"momentum":0.8688255099071,"beta1":null},{"lr":0.6112620219363,"momentum":0.8888739533022,"beta1":null},{"lr":0.3887419780637,"momentum":0.9111260466978,"beta1":null},{"lr":0.1882583460502,"momentum":0.9311744900929,"beta1":null},{"lr":0.04951936798653,"momentum":0.9450484433951,"beta1":null},{"lr":4e-06,"momentum":0.95,"beta1":null}],"cyclic_sgd":[{"lr":0.1,"momentum":0.9,"beta1":null},{"lr":0.4,"momentum":0.8666666666667,"beta1":null},{"lr":0.7,"momentum":0.8333333333333,"beta1":null},{"lr":1.0,"momentum":0.8,"beta1":null},{"lr":0.55,"momentum":0.85,"beta1":null},{"lr":0.1,"momentum":0.9,"beta1":null},{"lr":0.4,"momentum":0.8666666666667,"beta1":null},{"lr":0.7,"momentum":0.8333333333333,"beta1":null},{"lr":1.0,"momentum":0.8,"beta1":null},{"lr":0.55,"momentum":0.85,"beta1":null},{"lr":0.1,"momentum":0.9,"beta1":null},{"lr":0.4,"momentum":0.8666666666667,"beta1":null}],"cyclic_sgd_params":[{"lr":0.1,"momentum":0.95,"beta1":null},{"lr":0.4,"momentum":0.8666666666667,"beta1":null},{"lr":0.7,"momentum":0.7833333333333,"beta1":null},{"lr":1.0,"momentum":0.7,"beta1":null},{"lr":0.7,"momentum":0.7833333333333,"beta1":null},{"lr":0.4,"momentum":0.8666666666667,"beta1":null},{"lr":0.1,"momentum":0.95,"beta1":null},{"lr":0.25,"momentum":0.9083333333333,"beta1":null},{"lr":0.4,"momentum":0.8666666666667,"beta1":null},{"lr":0.55,"momentum":0.825,"beta1":null},{"lr":0.4,"momentum":0.8666666666667,"beta1":null},{"lr":0.25,"momentum":0.9083333333333,"beta1":null}],"cyclic_nomom":[{"lr":0.1,"momentum":0.9,"beta1":null},{"lr":0.4,"momentum":0.9,"beta1":null},{"lr":0.7,"momentum":0.9,"beta1":null},{"lr":1.0,"momentum":0.9,"beta1":null},{"lr":0.7,"momentum":0.9,"beta1":null},{"lr":0.4,"momentum":0.9,"beta1":null},{"lr":0.1,"momentum":0.9,"beta1":null},{"lr":0.4,"momentum":0.9,"beta1":null}],"cyclic_sgd_nomomentum_err":null,"onecycle_sgd_nomomentum_err":null}`
) as {
  multi: number[];
  onecycle_sgd: Row[];
  onecycle_sgd_params: Row[];
  onecycle_nomom: Row[];
  onecycle_adam: Row[];
  onecycle_sgd_nomomentum: Row[];
  cyclic_sgd: Row[];
  cyclic_sgd_params: Row[];
  cyclic_nomom: Row[];
};

const param = (): ReturnType<typeof parameter>[] => [parameter(tensor([0]))];

type Group = { options: Record<string, unknown> };

function trace(
  optimizer: SGD | Adam,
  scheduler: { step(): void },
  steps: number,
  key: "momentum" | "beta1"
): Array<{ lr: number; value: number }> {
  const rows: Array<{ lr: number; value: number }> = [];
  for (let i = 0; i < steps; i++) {
    const group = optimizer.paramGroups[0] as unknown as Group;
    rows.push({ lr: group.options["lr"] as number, value: group.options[key] as number });
    scheduler.step();
  }
  return rows;
}

function expectRows(
  actual: Array<{ lr: number; value: number }>,
  expected: Row[],
  key: "momentum" | "beta1"
): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((row, i) => {
    expect(actual[i]?.lr).toBeCloseTo(row.lr, 10);
    const want = key === "momentum" ? row.momentum : row.beta1;
    expect(actual[i]?.value).toBeCloseTo(want as number, 10);
  });
}

describe("NN-11: MultiStepLR duplicate milestones", () => {
  it("applies gamma once per occurrence, like torch", () => {
    const opt = new SGD(param(), { lr: 1 });
    const sched = new MultiStepLR(opt, { milestones: [5, 2, 2], gamma: 0.1 });
    const lrs: number[] = [];
    for (let i = 0; i < 8; i++) {
      lrs.push((opt.paramGroups[0] as unknown as Group).options["lr"] as number);
      sched.step();
    }
    lrs.forEach((lr, i) => {
      expect(lr).toBeCloseTo(REF.multi[i] as number, 12);
    });
  });

  it("still rejects negative, fractional and empty milestones", () => {
    const opt = new SGD(param(), { lr: 1 });
    expect(() => new MultiStepLR(opt, { milestones: [] })).toThrow(InvalidParameterError);
    expect(() => new MultiStepLR(opt, { milestones: [-1] })).toThrow(InvalidParameterError);
    expect(() => new MultiStepLR(opt, { milestones: [1.5] })).toThrow(InvalidParameterError);
  });
});

describe("NN-12: OneCycleLR momentum cycling", () => {
  it("cycles SGD momentum by default", () => {
    const opt = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    const sched = new OneCycleLR(opt, { maxLr: 1, totalSteps: 10 });
    expectRows(trace(opt, sched, 10, "momentum"), REF.onecycle_sgd, "momentum");
  });

  it("honors baseMomentum and maxMomentum", () => {
    const opt = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    const sched = new OneCycleLR(opt, {
      maxLr: 1,
      totalSteps: 10,
      baseMomentum: 0.7,
      maxMomentum: 0.95,
    });
    expectRows(trace(opt, sched, 10, "momentum"), REF.onecycle_sgd_params, "momentum");
  });

  it("leaves the momentum alone with cycleMomentum: false", () => {
    const opt = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    const sched = new OneCycleLR(opt, { maxLr: 1, totalSteps: 10, cycleMomentum: false });
    expectRows(trace(opt, sched, 10, "momentum"), REF.onecycle_nomom, "momentum");
  });

  it("cycles beta1 of Adam", () => {
    const opt = new Adam(param(), { lr: 0.1 });
    const sched = new OneCycleLR(opt, {
      maxLr: 1,
      totalSteps: 10,
      annealStrategy: "linear",
    });
    expectRows(trace(opt, sched, 10, "beta1"), REF.onecycle_adam, "beta1");
  });

  it("switches momentum on for SGD without momentum, as torch does", () => {
    const opt = new SGD(param(), { lr: 0.1 });
    const sched = new OneCycleLR(opt, { maxLr: 1, totalSteps: 10 });
    expectRows(trace(opt, sched, 10, "momentum"), REF.onecycle_sgd_nomomentum, "momentum");
  });

  it("works with a plain parameter group that has no momentum unless asked", () => {
    const bare = { paramGroups: [{ params: [], lr: 0.1 }] };
    expect(() => new OneCycleLR(bare, { maxLr: 1, totalSteps: 10 })).not.toThrow();
    expect(() => new OneCycleLR(bare, { maxLr: 1, totalSteps: 10, cycleMomentum: true })).toThrow(
      InvalidParameterError
    );
  });

  it("validates the momentum bounds", () => {
    const opt = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    expect(() => new OneCycleLR(opt, { maxLr: 1, totalSteps: 10, baseMomentum: -0.1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new OneCycleLR(opt, { maxLr: 1, totalSteps: 10, maxMomentum: 2 })).toThrow(
      InvalidParameterError
    );
  });

  it("restores the momentum from a saved scheduler state", () => {
    const opt = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    const sched = new OneCycleLR(opt, { maxLr: 1, totalSteps: 10 });
    for (let i = 0; i < 4; i++) sched.step();
    const state = sched.stateDict();
    const after = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    const restored = new OneCycleLR(after, { maxLr: 1, totalSteps: 10 });
    restored.loadStateDict(state);
    expect((after.paramGroups[0] as unknown as Group).options["momentum"]).toBeCloseTo(
      (opt.paramGroups[0] as unknown as Group).options["momentum"] as number,
      12
    );
  });
});

describe("NN-12: CyclicLR momentum cycling", () => {
  it("cycles SGD momentum by default", () => {
    const opt = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    const sched = new CyclicLR(opt, { baseLr: 0.1, maxLr: 1, stepSizeUp: 3, stepSizeDown: 2 });
    expectRows(trace(opt, sched, 12, "momentum"), REF.cyclic_sgd, "momentum");
  });

  it("honors baseMomentum, maxMomentum and the triangular2 mode", () => {
    const opt = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    const sched = new CyclicLR(opt, {
      baseLr: 0.1,
      maxLr: 1,
      stepSizeUp: 3,
      baseMomentum: 0.7,
      maxMomentum: 0.95,
      mode: "triangular2",
    });
    expectRows(trace(opt, sched, 12, "momentum"), REF.cyclic_sgd_params, "momentum");
  });

  it("leaves the momentum alone with cycleMomentum: false", () => {
    const opt = new SGD(param(), { lr: 0.1, momentum: 0.9 });
    const sched = new CyclicLR(opt, {
      baseLr: 0.1,
      maxLr: 1,
      stepSizeUp: 3,
      cycleMomentum: false,
    });
    expectRows(trace(opt, sched, 8, "momentum"), REF.cyclic_nomom, "momentum");
  });

  it("works with a plain parameter group that has no momentum unless asked", () => {
    const bare = { paramGroups: [{ params: [], lr: 0.1 }] };
    expect(() => new CyclicLR(bare, { baseLr: 0.1, maxLr: 1 })).not.toThrow();
    expect(() => new CyclicLR(bare, { baseLr: 0.1, maxLr: 1, cycleMomentum: true })).toThrow(
      InvalidParameterError
    );
  });
});
