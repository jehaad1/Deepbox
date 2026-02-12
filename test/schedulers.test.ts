import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../src/core";
import { parameter, tensor } from "../src/ndarray";
import { SGD } from "../src/optim";
import {
  CosineAnnealingLR,
  ExponentialLR,
  LinearLR,
  MultiStepLR,
  OneCycleLR,
  ReduceLROnPlateau,
  StepLR,
  WarmupLR,
} from "../src/optim/schedulers";

type MockOptimizer = {
  paramGroups: Array<{ lr: number; params: unknown[] }>;
};

// Create a mock optimizer that exposes paramGroups publicly for scheduler testing
function createMockOptimizer(lr: number = 0.1): MockOptimizer {
  const params: unknown[] = [];
  return {
    paramGroups: [{ lr, params }],
  };
}

describe("Learning Rate Schedulers", () => {
  describe("StepLR", () => {
    it("should decay lr by gamma every stepSize epochs", () => {
      const optimizer = createMockOptimizer(0.1);
      const scheduler = new StepLR(optimizer, { stepSize: 2, gamma: 0.5 });

      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);

      scheduler.step(); // epoch 0
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);

      scheduler.step(); // epoch 1
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);

      scheduler.step(); // epoch 2
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.05);

      scheduler.step(); // epoch 3
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.05);

      scheduler.step(); // epoch 4
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.025);
    });

    it("should use default gamma of 0.1", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new StepLR(optimizer, { stepSize: 1 });

      scheduler.step();
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step();
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);
    });
  });

  describe("ExponentialLR", () => {
    it("should decay lr exponentially each epoch", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new ExponentialLR(optimizer, { gamma: 0.9 });

      scheduler.step(); // epoch 0
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(); // epoch 1
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.9);

      scheduler.step(); // epoch 2
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.81);

      scheduler.step(); // epoch 3
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.729);
    });
  });

  describe("CosineAnnealingLR", () => {
    it("should follow cosine schedule", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new CosineAnnealingLR(optimizer, {
        T_max: 4,
        etaMin: 0,
      });

      scheduler.step(); // epoch 0
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(); // epoch 1
      // cos(pi * 1 / 4) = sqrt(2)/2 ≈ 0.707
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.8535533905932738);

      scheduler.step(); // epoch 2
      // cos(pi * 2 / 4) = 0
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.5);

      scheduler.step(); // epoch 3
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.14644660940672627);

      scheduler.step(); // epoch 4
      // cos(pi) = -1, so (1 + (-1))/2 = 0
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0);
    });

    it("should respect etaMin", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new CosineAnnealingLR(optimizer, {
        T_max: 2,
        etaMin: 0.1,
      });

      scheduler.step();
      scheduler.step();
      scheduler.step(); // epoch 2 = T_max

      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);
    });
  });

  describe("MultiStepLR", () => {
    it("should decay at milestones", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new MultiStepLR(optimizer, {
        milestones: [2, 5],
        gamma: 0.1,
      });

      scheduler.step(); // epoch 0
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(); // epoch 1
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(); // epoch 2 - first milestone
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);

      scheduler.step(); // epoch 3
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);

      scheduler.step(); // epoch 4
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);

      scheduler.step(); // epoch 5 - second milestone
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.01);
    });
  });

  describe("LinearLR", () => {
    it("should linearly interpolate between start and end factors", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new LinearLR(optimizer, {
        startFactor: 0.1,
        endFactor: 1.0,
        totalIters: 4,
      });

      scheduler.step(); // epoch 0
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);

      scheduler.step(); // epoch 1
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.325);

      scheduler.step(); // epoch 2
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.55);

      scheduler.step(); // epoch 3
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.775);

      scheduler.step(); // epoch 4 = totalIters
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(); // epoch 5 > totalIters (stays at end)
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);
    });
  });

  describe("ReduceLROnPlateau", () => {
    it("should reduce lr when metric stops improving (min mode)", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new ReduceLROnPlateau(optimizer, {
        mode: "min",
        factor: 0.5,
        patience: 2,
      });

      // Improving
      scheduler.step(1.0);
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(0.9);
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      // Not improving for patience epochs
      scheduler.step(0.95);
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(0.95);
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(0.95); // patience exceeded
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.5);
    });

    it("should reduce lr when metric stops improving (max mode)", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new ReduceLROnPlateau(optimizer, {
        mode: "max",
        factor: 0.1,
        patience: 1,
      });

      scheduler.step(0.5);
      scheduler.step(0.6); // improving
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(0.55); // not improving
      scheduler.step(0.55); // patience exceeded
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.1);
    });

    it("should respect minLr", () => {
      const optimizer = createMockOptimizer(0.1);
      const scheduler = new ReduceLROnPlateau(optimizer, {
        factor: 0.1,
        patience: 0,
        minLr: 0.05,
      });

      scheduler.step(1.0);
      scheduler.step(1.0); // trigger reduction
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.05); // clamped to minLr
    });
  });

  describe("WarmupLR", () => {
    it("should linearly warmup then use base scheduler", () => {
      const optimizer = createMockOptimizer(1.0);
      const baseScheduler = new StepLR(optimizer, { stepSize: 2, gamma: 0.5 });
      const scheduler = new WarmupLR(optimizer, baseScheduler, {
        warmupEpochs: 3,
      });

      scheduler.step(); // epoch 0: warmup 1/3
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1 / 3);

      scheduler.step(); // epoch 1: warmup 2/3
      expect(scheduler.getLastLr()[0]).toBeCloseTo(2 / 3);

      scheduler.step(); // epoch 2: warmup 3/3 = 1.0
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      // After warmup, delegates to base scheduler
      scheduler.step(); // epoch 3
      scheduler.step(); // epoch 4
    });

    it("should work without after scheduler", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new WarmupLR(optimizer, null, { warmupEpochs: 2 });

      scheduler.step();
      expect(scheduler.getLastLr()[0]).toBeCloseTo(0.5);

      scheduler.step();
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);

      scheduler.step(); // past warmup, stays at base lr
      expect(scheduler.getLastLr()[0]).toBeCloseTo(1.0);
    });
  });

  describe("OneCycleLR", () => {
    it("should increase then decrease lr", () => {
      const optimizer = createMockOptimizer(0.1);
      const scheduler = new OneCycleLR(optimizer, {
        maxLr: 1.0,
        totalSteps: 10,
        pctStart: 0.3, // 3 steps up, 7 steps down
        divFactor: 10, // start at 0.1
        finalDivFactor: 100, // end at 0.01
      });

      // Initial lr = maxLr / divFactor = 0.1
      scheduler.step(); // step 0
      const lr0 = scheduler.getLastLr()[0] ?? 0;
      expect(lr0).toBeCloseTo(0.1);

      // Step 1, 2 should increase
      scheduler.step();
      scheduler.step();
      const lr2 = scheduler.getLastLr()[0] ?? 0;
      expect(lr2).toBeGreaterThan(lr0);

      // Continue stepping until we pass the peak
      scheduler.step(); // step 3 - should be at or past peak
      scheduler.step();
      scheduler.step();
      scheduler.step();
      scheduler.step();
      scheduler.step();
      scheduler.step(); // step 9

      // Final lr should be close to maxLr / finalDivFactor = 0.01
      const lrFinal = scheduler.getLastLr()[0] ?? 0;
      expect(lrFinal).toBeLessThan(0.1);
    });
  });

  describe("LRScheduler base", () => {
    it("should track epoch number", () => {
      const optimizer = createMockOptimizer(1.0);
      const scheduler = new StepLR(optimizer, { stepSize: 5 });

      expect(scheduler.epoch).toBe(-1);
      scheduler.step();
      expect(scheduler.epoch).toBe(0);
      scheduler.step();
      expect(scheduler.epoch).toBe(1);
    });
  });

  describe("Integration", () => {
    it("should update learning rates on deepbox optimizers", () => {
      const p = parameter(tensor([1], { dtype: "float32" }));
      const optimizer = new SGD([p], { lr: 0.1 });
      const scheduler = new StepLR(optimizer, { stepSize: 1, gamma: 0.1 });

      scheduler.step(); // epoch 0
      expect(optimizer.getLearningRate()).toBeCloseTo(0.1);

      scheduler.step(); // epoch 1
      expect(optimizer.getLearningRate()).toBeCloseTo(0.01);
    });
  });

  describe("validation", () => {
    it("validates scheduler constructor arguments", () => {
      const optimizer = createMockOptimizer(0.1);

      expect(() => new StepLR(optimizer, { stepSize: 0 })).toThrow(InvalidParameterError);
      expect(() => new ExponentialLR(optimizer, { gamma: 0 })).toThrow(InvalidParameterError);
      expect(() => new CosineAnnealingLR(optimizer, { T_max: 0 })).toThrow(InvalidParameterError);
      expect(() => new MultiStepLR(optimizer, { milestones: [] })).toThrow(InvalidParameterError);
      expect(() => new LinearLR(optimizer, { totalIters: 0 })).toThrow(InvalidParameterError);
      expect(() => new WarmupLR(optimizer, null, { warmupEpochs: 0 })).toThrow(
        InvalidParameterError
      );
      expect(
        () =>
          new OneCycleLR(optimizer, {
            maxLr: 1.0,
            totalSteps: 10,
            pctStart: 1,
          })
      ).toThrow(InvalidParameterError);
      expect(() => new ReduceLROnPlateau(optimizer, { factor: 1.2 })).toThrow(
        InvalidParameterError
      );
    });
  });
});
