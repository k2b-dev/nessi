import { describe, expect, it } from "bun:test";
import { resolveReasoning, withExtraBody } from "../../../src/ai/shared/request-options.js";

describe("resolveReasoning", () => {
  it("prefers the request effort over the provider default and the deprecated flag", () => {
    expect(resolveReasoning({ messages: [], reasoningEffort: "high", disableReasoning: true }, { reasoningEffort: "low" }))
      .toEqual({ effort: "high", legacyDisable: false });
  });

  it("treats disableReasoning as a request setting that beats the provider default", () => {
    expect(resolveReasoning({ messages: [], disableReasoning: true }, { reasoningEffort: "high" }))
      .toEqual({ legacyDisable: true });
  });

  it("falls back to the provider default and otherwise sends nothing", () => {
    expect(resolveReasoning({ messages: [] }, { reasoningEffort: "medium" })).toEqual({ effort: "medium", legacyDisable: false });
    expect(resolveReasoning({ messages: [] })).toEqual({ effort: undefined, legacyDisable: false });
  });

  it("passes unknown effort strings through unchanged", () => {
    expect(resolveReasoning({ messages: [], reasoningEffort: "ultra-2027" }).effort).toBe("ultra-2027");
  });
});

describe("withExtraBody", () => {
  it("returns the body unchanged without extras", () => {
    const body = { model: "m" };
    expect(withExtraBody(body, { messages: [] })).toBe(body);
  });

  it("merges plain objects deeply, request over provider, and replaces other values", () => {
    const body = { model: "m", generationConfig: { temperature: 1, stopSequences: ["a"] }, tools: [1] };
    const merged = withExtraBody(
      body,
      { messages: [], extraBody: { generationConfig: { thinkingConfig: { thinkingLevel: "low" } }, tools: [2] } },
      { extraBody: { generationConfig: { temperature: 0, thinkingConfig: { includeThoughts: true } }, top_k: 5 } },
    );

    expect(merged).toEqual({
      model: "m",
      generationConfig: { temperature: 0, stopSequences: ["a"], thinkingConfig: { includeThoughts: true, thinkingLevel: "low" } },
      tools: [2],
      top_k: 5,
    });
    expect(body.generationConfig).toEqual({ temperature: 1, stopSequences: ["a"] });
  });

  it("replaces values that are not plain objects instead of merging into them", () => {
    const date = new Date("2026-01-01T00:00:00Z");
    expect(withExtraBody({ metadata: { stale: true } }, { messages: [], extraBody: { metadata: date } }).metadata).toBe(date);
    expect(withExtraBody({ metadata: date }, { messages: [], extraBody: { metadata: { fresh: true } } }).metadata)
      .toEqual({ fresh: true });
  });

  it("ignores __proto__ keys", () => {
    const merged = withExtraBody({}, { messages: [], extraBody: JSON.parse('{"__proto__": {"polluted": true}}') });
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    expect(Object.getPrototypeOf(merged)).toBe(Object.prototype);
  });
});
