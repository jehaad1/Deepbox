// NN-22: save() writes atomically: the payload goes to a temporary file in the same directory
// and is renamed over the target, so a failed write never leaves a truncated target file.
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DeepboxError, load, type SerializedTensor, save } from "../../src/core";

const payload: SerializedTensor = {
  __type: "Tensor",
  data: [1, 2, 3],
  shape: [3],
  dtype: "float32",
};

describe("NN-22: atomic save", () => {
  let dir: string;

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "deepbox-save-"));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("writes through a temporary file in the same directory and renames it", async () => {
    const target = path.join(dir, "model.json");
    const write = vi.spyOn(fs.promises, "writeFile");
    const rename = vi.spyOn(fs.promises, "rename");
    await save(target, payload);
    const written = String(write.mock.calls[0]?.[0]);
    expect(written).not.toBe(target);
    expect(path.dirname(written)).toBe(dir);
    expect(rename).toHaveBeenCalledWith(written, target);
    expect(fs.readdirSync(dir)).toEqual(["model.json"]);
    expect(await load(target)).toEqual(payload);
  });

  it("keeps the old file when the write fails", async () => {
    const target = path.join(dir, "model.json");
    await save(target, payload);
    const before = fs.readFileSync(target, "utf-8");
    vi.spyOn(fs.promises, "writeFile").mockRejectedValueOnce(new Error("disk full"));
    await expect(save(target, { ...payload, data: [9, 9, 9] })).rejects.toThrow(DeepboxError);
    expect(fs.readFileSync(target, "utf-8")).toBe(before);
    expect(fs.readdirSync(dir)).toEqual(["model.json"]);
  });

  it("removes the temporary file when the rename fails", async () => {
    const target = path.join(dir, "model.json");
    fs.mkdirSync(target);
    await expect(save(target, payload)).rejects.toThrow(/Failed to save/);
    expect(fs.readdirSync(dir)).toEqual(["model.json"]);
    expect(fs.statSync(target).isDirectory()).toBe(true);
  });

  it("overwrites an existing file", async () => {
    const target = path.join(dir, "model.json");
    await save(target, payload);
    await save(target, { ...payload, data: [4, 5, 6] });
    expect(await load(target)).toEqual({ ...payload, data: [4, 5, 6] });
    expect(fs.readdirSync(dir)).toEqual(["model.json"]);
  });
});
