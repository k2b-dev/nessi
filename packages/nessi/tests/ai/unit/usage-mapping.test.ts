import { describe, expect, it } from "bun:test";
import { makeUsage } from "../../../src/ai/shared/usage.js";

describe("usage helpers", () => {
  it("builds usage totals", () => {
    expect(makeUsage(2, 3)).toEqual({ input: 2, output: 3, total: 5 });
  });

});
