import { describe, expect, it } from "bun:test";
import { complete } from "../../../src/ai/complete.js";
import { normalizeProviderStream } from "../../../src/ai/shared/tool-stream-normalizer.js";
import type { GenerateRequest, Provider, RawStreamEvent } from "../../../src/ai/types.js";

async function* rawSource(events: RawStreamEvent[]) {
  for (const event of events) yield event;
}

describe("complete", () => {
  it("aggregates text, thinking, tool calls and usage", async () => {
    const provider: Provider = {
      name: "mock",
      family: "openai-compatible",
      model: "mock",
      contextWindow: 1000,
      capabilities: { streaming: true, tools: true, images: true, thinking: true, usage: true },
      stream() {
        return normalizeProviderStream(rawSource([
          { type: "thinking", delta: "hmm" },
          { type: "text", delta: "hello" },
          { type: "tool_start", callId: "c1", name: "search" },
          { type: "tool_delta", callId: "c1", argsDelta: "{\"q\":\"x\"}" },
          { type: "tool_call", callId: "c1", name: "search", args: { q: "x" } },
          { type: "usage", usage: { input: 1, output: 2, total: 3 } },
        ]));
      },
    };

    const result = await complete(provider, { messages: [] });
    expect(result.finishReason).toBe("tool_use");
    expect(result.usage?.total).toBe(3);
    expect(result.message.content.map((block) => block.type)).toEqual(["thinking", "text", "tool_call"]);
  });

  it("prefers the provider-reported finish reason when available", async () => {
    const provider: Provider = {
      name: "mock",
      family: "openai-compatible",
      model: "mock",
      contextWindow: 1000,
      capabilities: { streaming: true, tools: true, images: true, thinking: true, usage: true },
      stream() {
        return normalizeProviderStream(rawSource([
          { type: "text", delta: "partial" },
          {
            type: "usage",
            usage: { input: 1, output: 2, total: 3 },
            finishReason: "max_tokens",
          },
        ]));
      },
    };

    const result = await complete(provider, { messages: [] });
    expect(result.finishReason).toBe("max_tokens");
  });

  it("ignores empty text and thinking deltas", async () => {
    const provider: Provider = {
      name: "mock",
      family: "openai-compatible",
      model: "mock",
      contextWindow: 1000,
      capabilities: { streaming: true, tools: true, images: true, thinking: true, usage: true },
      stream() {
        return normalizeProviderStream(rawSource([
          { type: "thinking", delta: "" },
          { type: "text", delta: "" },
          { type: "usage", usage: { input: 1, output: 0, total: 1 } },
        ]));
      },
    };

    const result = await complete(provider, { messages: [] });
    expect(result.message.content).toEqual([]);
  });
});
