import { afterEach, describe, expect, it } from "vitest";
import type { LogEntry, VerboseLevel } from "../src/core";
import { getLogHandler, Logger, setLogHandler } from "../src/core";

describe("Logger", () => {
  afterEach(() => {
    setLogHandler(undefined);
  });

  describe("constructor", () => {
    it("creates with valid verbosity levels", () => {
      for (const v of [0, 1, 2, 3] as VerboseLevel[]) {
        const log = new Logger(v, "Test");
        expect(log.level).toBe(v);
      }
    });

    it("throws on invalid verbosity level", () => {
      expect(() => new Logger(-1 as VerboseLevel, "Test")).toThrow();
      expect(() => new Logger(4 as VerboseLevel, "Test")).toThrow();
      expect(() => new Logger(1.5 as VerboseLevel, "Test")).toThrow();
    });
  });

  describe("logging at verbose=0 (silent)", () => {
    it("does not emit any messages", () => {
      const entries: LogEntry[] = [];
      setLogHandler((e) => entries.push(e));
      const log = new Logger(0, "Silent");
      log.info("should not appear");
      log.debug("should not appear");
      log.trace("should not appear");
      expect(entries).toHaveLength(0);
    });

    it("still records entries internally", () => {
      const log = new Logger(0, "Silent");
      log.info("recorded");
      expect(log.getEntries()).toHaveLength(1);
    });
  });

  describe("logging at verbose=1 (summary)", () => {
    it("emits info messages", () => {
      const entries: LogEntry[] = [];
      setLogHandler((e) => entries.push(e));
      const log = new Logger(1, "Summary");
      log.info("converged");
      expect(entries).toHaveLength(1);
      expect(entries[0]?.message).toBe("converged");
    });

    it("does not emit debug or trace", () => {
      const entries: LogEntry[] = [];
      setLogHandler((e) => entries.push(e));
      const log = new Logger(1, "Summary");
      log.debug("iteration 5");
      log.trace("internal detail");
      expect(entries).toHaveLength(0);
    });
  });

  describe("logging at verbose=2 (progress)", () => {
    it("emits info and debug messages", () => {
      const entries: LogEntry[] = [];
      setLogHandler((e) => entries.push(e));
      const log = new Logger(2, "Progress");
      log.info("converged");
      log.debug("iteration 5");
      log.trace("should not appear");
      expect(entries).toHaveLength(2);
    });
  });

  describe("logging at verbose=3 (trace)", () => {
    it("emits all messages", () => {
      const entries: LogEntry[] = [];
      setLogHandler((e) => entries.push(e));
      const log = new Logger(3, "Trace");
      log.info("converged");
      log.debug("iteration 5");
      log.trace("internal detail");
      expect(entries).toHaveLength(3);
    });
  });

  describe("getEntries", () => {
    it("records all entries regardless of verbosity", () => {
      const log = new Logger(0, "Test");
      log.info("a");
      log.debug("b");
      log.trace("c");
      const entries = log.getEntries();
      expect(entries).toHaveLength(3);
      expect(entries[0]?.level).toBe(1);
      expect(entries[1]?.level).toBe(2);
      expect(entries[2]?.level).toBe(3);
    });

    it("entries have timestamps", () => {
      const log = new Logger(0, "Test");
      const before = Date.now();
      log.info("test");
      const after = Date.now();
      const entry = log.getEntries()[0];
      expect(entry).toBeDefined();
      expect(entry!.timestamp).toBeGreaterThanOrEqual(before);
      expect(entry!.timestamp).toBeLessThanOrEqual(after);
    });
  });

  describe("clear", () => {
    it("clears all entries", () => {
      const log = new Logger(0, "Test");
      log.info("a");
      log.info("b");
      expect(log.getEntries()).toHaveLength(2);
      log.clear();
      expect(log.getEntries()).toHaveLength(0);
    });
  });

  describe("setLogHandler / getLogHandler", () => {
    it("custom handler receives entries", () => {
      const entries: LogEntry[] = [];
      setLogHandler((e) => entries.push(e));
      expect(getLogHandler()).toBeDefined();
      const log = new Logger(3, "Custom");
      log.info("test");
      expect(entries).toHaveLength(1);
    });

    it("reset handler to undefined", () => {
      setLogHandler(() => {});
      setLogHandler(undefined);
      expect(getLogHandler()).toBeUndefined();
    });
  });
});
