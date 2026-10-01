import { describe, it, expect } from "bun:test";
import { isContextOverflow, parseOverflowRatio } from "../../../src/ai/shared/errors.js";

describe("isContextOverflow", () => {
  const overflow = (status: number, msg: string) => isContextOverflow(status, msg);

  it("detects common context overflow messages", () => {
    expect(overflow(400, "context length exceeded")).toBe(true);
    expect(overflow(400, "input is too long")).toBe(true);
    expect(overflow(400, "maximum context length is 32768")).toBe(true);
    expect(overflow(400, "max tokens exceeded")).toBe(true);
    expect(overflow(400, "context window exceeded")).toBe(true);
    expect(overflow(400, "token limit reached")).toBe(true);
    expect(overflow(400, "prompt is too long")).toBe(true);
    expect(overflow(400, "reduce the length of the input")).toBe(true);
  });

  it("detects provider-specific overflow messages", () => {
    expect(overflow(400, "prompt is too long: 210000 tokens > 200000 maximum")).toBe(true);
    expect(overflow(400, "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).")).toBe(true);
    expect(overflow(400, "Prompt contains 140000 tokens and 0 draft tokens, too large for model with 131072 maximum context length")).toBe(true);
    expect(overflow(400, "the request exceeds the available context size, try increasing it")).toBe(true);
    expect(overflow(400, "Invalid request (code: context_length_exceeded)")).toBe(true);
  });

  it("detects on status 413 and 422", () => {
    expect(overflow(413, "context too long")).toBe(true);
    expect(overflow(422, "token limit exceeded")).toBe(true);
  });

  it("rejects non-overflow status codes", () => {
    expect(overflow(200, "context too long")).toBe(false);
    expect(overflow(401, "context too long")).toBe(false);
    expect(overflow(500, "context too long")).toBe(false);
  });

  it("rejects unrelated 400 errors", () => {
    expect(overflow(400, "invalid api key")).toBe(false);
    expect(overflow(400, "model not found")).toBe(false);
    expect(overflow(400, "max_tokens: 100000 > 64000, which is the maximum allowed number of output tokens for claude-sonnet")).toBe(false);
    expect(overflow(400, "Invalid 'tools[0].function.description': string too long. Expected a string with maximum length 1024")).toBe(false);
    expect(overflow(400, "temperature: input should be less than or equal to the maximum of 2")).toBe(false);
    expect(overflow(400, "monthly limit exceeded")).toBe(false);
  });
});

describe("parseOverflowRatio", () => {
  it("parses vLLM-style error message", () => {
    const msg =
      "This model's maximum context length is 32768 tokens. However, you requested 0 output tokens and your prompt contains at least 32769 input tokens, for a total of at least 32769 tokens.";
    const ratio = parseOverflowRatio(msg);
    expect(ratio).toBeDefined();
    expect(ratio!).toBeCloseTo(32769 / 32768, 2);
  });

  it("parses OpenAI-style error message", () => {
    const msg =
      "This model's maximum context length is 4096 tokens. However, your messages resulted in 5120 tokens.";
    const ratio = parseOverflowRatio(msg);
    expect(ratio).toBeDefined();
    expect(ratio!).toBeCloseTo(5120 / 4096, 2);
  });

  it("parses Anthropic-style error message", () => {
    const ratio = parseOverflowRatio("prompt is too long: 210,000 tokens > 200000 maximum");
    expect(ratio).toBeCloseTo(210000 / 200000, 4);
  });

  it("returns undefined for non-overflow messages", () => {
    expect(parseOverflowRatio("invalid api key")).toBeUndefined();
    expect(parseOverflowRatio("model not found")).toBeUndefined();
    expect(parseOverflowRatio("")).toBeUndefined();
  });

  it("returns undefined when only max is present", () => {
    expect(parseOverflowRatio("maximum context length is 32768 tokens")).toBeUndefined();
  });
});
