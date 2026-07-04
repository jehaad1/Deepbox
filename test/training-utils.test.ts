import { describe, expect, it } from "vitest";
import { EarlyStopping, GradientAccumulator, Linear, ModelCheckpoint, Sequential } from "../src/nn";

describe("EarlyStopping", () => {
  describe("constructor", () => {
    it("creates with default options", () => {
      const es = new EarlyStopping();
      expect(es.isStopped).toBe(false);
      expect(es.best).toBe(Infinity);
      expect(es.waitCount).toBe(0);
    });

    it("creates with custom options", () => {
      const es = new EarlyStopping({ patience: 5, minDelta: 0.01, mode: "max" });
      expect(es.best).toBe(-Infinity);
    });

    it("validates patience", () => {
      expect(() => new EarlyStopping({ patience: 0 })).toThrow();
      expect(() => new EarlyStopping({ patience: -1 })).toThrow();
      expect(() => new EarlyStopping({ patience: 1.5 })).toThrow();
    });

    it("validates minDelta", () => {
      expect(() => new EarlyStopping({ minDelta: -0.1 })).toThrow();
      expect(() => new EarlyStopping({ minDelta: NaN })).toThrow();
    });
  });

  describe("step (min mode)", () => {
    it("does not stop while improving", () => {
      const es = new EarlyStopping({ patience: 3 });
      expect(es.step(10)).toBe(false);
      expect(es.step(9)).toBe(false);
      expect(es.step(8)).toBe(false);
      expect(es.isStopped).toBe(false);
    });

    it("stops after patience epochs without improvement", () => {
      const es = new EarlyStopping({ patience: 3 });
      es.step(5); // best = 5
      es.step(6); // no improvement, wait=1
      es.step(6); // no improvement, wait=2
      expect(es.step(6)).toBe(true); // wait=3 >= patience
      expect(es.isStopped).toBe(true);
    });

    it("resets counter on improvement", () => {
      const es = new EarlyStopping({ patience: 3 });
      es.step(10);
      es.step(11); // wait=1
      es.step(9); // improvement, wait=0
      expect(es.waitCount).toBe(0);
      es.step(10); // wait=1
      es.step(10); // wait=2
      expect(es.step(10)).toBe(true); // wait=3
    });

    it("respects minDelta", () => {
      const es = new EarlyStopping({ patience: 2, minDelta: 1.0 });
      es.step(10);
      es.step(9.5); // only 0.5 better, not enough
      expect(es.waitCount).toBe(1);
      es.step(8.5); // 1.5 better than best (10), counts
      expect(es.waitCount).toBe(0);
    });

    it("tracks bestEpochNum", () => {
      const es = new EarlyStopping({ patience: 5 });
      es.step(10);
      es.step(8);
      es.step(9);
      expect(es.bestEpochNum).toBe(2);
    });
  });

  describe("step (max mode)", () => {
    it("stops when metric stops increasing", () => {
      const es = new EarlyStopping({ patience: 2, mode: "max" });
      es.step(0.8);
      es.step(0.9);
      es.step(0.85); // wait=1
      expect(es.step(0.88)).toBe(true); // wait=2
      expect(es.best).toBe(0.9);
    });
  });

  describe("reset", () => {
    it("resets all state", () => {
      const es = new EarlyStopping({ patience: 2 });
      es.step(5);
      es.step(6);
      es.step(7);
      es.reset();
      expect(es.isStopped).toBe(false);
      expect(es.best).toBe(Infinity);
      expect(es.waitCount).toBe(0);
      expect(es.bestEpochNum).toBe(0);
    });
  });
});

describe("GradientAccumulator", () => {
  describe("constructor", () => {
    it("creates with valid steps", () => {
      const ga = new GradientAccumulator(4);
      expect(ga.steps).toBe(4);
      expect(ga.accumulated).toBe(0);
    });

    it("validates accumSteps", () => {
      expect(() => new GradientAccumulator(0)).toThrow();
      expect(() => new GradientAccumulator(-1)).toThrow();
      expect(() => new GradientAccumulator(1.5)).toThrow();
    });
  });

  describe("step", () => {
    it("returns true every accumSteps steps", () => {
      const ga = new GradientAccumulator(3);
      expect(ga.step()).toBe(false); // 1
      expect(ga.step()).toBe(false); // 2
      expect(ga.step()).toBe(true); // 3 → trigger
      expect(ga.step()).toBe(false); // 1
      expect(ga.step()).toBe(false); // 2
      expect(ga.step()).toBe(true); // 3 → trigger
    });

    it("accumSteps=1 triggers every step", () => {
      const ga = new GradientAccumulator(1);
      expect(ga.step()).toBe(true);
      expect(ga.step()).toBe(true);
      expect(ga.step()).toBe(true);
    });

    it("tracks accumulated count", () => {
      const ga = new GradientAccumulator(4);
      ga.step();
      expect(ga.accumulated).toBe(1);
      ga.step();
      expect(ga.accumulated).toBe(2);
      ga.step();
      expect(ga.accumulated).toBe(3);
      ga.step(); // triggers, resets to 0
      expect(ga.accumulated).toBe(0);
    });

    it("tracks totalProcessed", () => {
      const ga = new GradientAccumulator(3);
      for (let i = 0; i < 7; i++) ga.step();
      expect(ga.totalProcessed).toBe(7);
    });

    it("tracks optimizerSteps", () => {
      const ga = new GradientAccumulator(3);
      for (let i = 0; i < 9; i++) ga.step();
      expect(ga.optimizerSteps).toBe(3);
    });
  });

  describe("scaleFactor", () => {
    it("equals accumSteps", () => {
      const ga = new GradientAccumulator(8);
      expect(ga.scaleFactor).toBe(8);
    });
  });

  describe("reset", () => {
    it("resets all state", () => {
      const ga = new GradientAccumulator(4);
      ga.step();
      ga.step();
      ga.reset();
      expect(ga.accumulated).toBe(0);
      expect(ga.totalProcessed).toBe(0);
      expect(ga.optimizerSteps).toBe(0);
    });
  });
});

describe("ModelCheckpoint", () => {
  function makeModel(): Sequential {
    return new Sequential(new Linear(2, 1));
  }

  describe("constructor", () => {
    it("creates with default options (min mode)", () => {
      const cp = new ModelCheckpoint();
      expect(cp.best).toBe(Infinity);
      expect(cp.hasSavedState).toBe(false);
    });

    it("creates with max mode", () => {
      const cp = new ModelCheckpoint({ mode: "max" });
      expect(cp.best).toBe(-Infinity);
    });
  });

  describe("step", () => {
    it("saves state on first call", () => {
      const cp = new ModelCheckpoint();
      const model = makeModel();
      expect(cp.step(model, 5.0)).toBe(true);
      expect(cp.hasSavedState).toBe(true);
      expect(cp.best).toBe(5.0);
      expect(cp.bestEpochNum).toBe(1);
    });

    it("saves state when metric improves (min)", () => {
      const cp = new ModelCheckpoint();
      const model = makeModel();
      cp.step(model, 5.0);
      expect(cp.step(model, 4.0)).toBe(true);
      expect(cp.best).toBe(4.0);
      expect(cp.bestEpochNum).toBe(2);
    });

    it("does not save when metric worsens (min)", () => {
      const cp = new ModelCheckpoint();
      const model = makeModel();
      cp.step(model, 5.0);
      expect(cp.step(model, 6.0)).toBe(false);
      expect(cp.best).toBe(5.0);
    });

    it("saves state when metric improves (max)", () => {
      const cp = new ModelCheckpoint({ mode: "max" });
      const model = makeModel();
      cp.step(model, 0.8);
      expect(cp.step(model, 0.9)).toBe(true);
      expect(cp.best).toBe(0.9);
    });
  });

  describe("restore", () => {
    it("restores saved model state", () => {
      const cp = new ModelCheckpoint();
      const model = makeModel();
      cp.step(model, 5.0);
      // Model state was saved. We can restore it.
      expect(() => cp.restore(model)).not.toThrow();
    });

    it("throws if no checkpoint saved", () => {
      const cp = new ModelCheckpoint();
      const model = makeModel();
      expect(() => cp.restore(model)).toThrow();
    });
  });

  describe("reset", () => {
    it("clears all state", () => {
      const cp = new ModelCheckpoint();
      const model = makeModel();
      cp.step(model, 5.0);
      cp.reset();
      expect(cp.hasSavedState).toBe(false);
      expect(cp.best).toBe(Infinity);
      expect(cp.bestEpochNum).toBe(0);
    });
  });
});
