import { describe, expect, it } from "bun:test";
import { assertOnlySupportedFiles } from "../../../src/ai/shared/messages.js";

describe("message helpers", () => {
  it("rejects unsupported file parts", () => {
    expect(() =>
      assertOnlySupportedFiles([{ type: "file", data: "abc", mediaType: "application/pdf" }], false, "test"),
    ).toThrow("does not support input file type");
  });
});
