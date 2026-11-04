import { afterEach, describe, expect, it } from "bun:test";
import { complete } from "../../../src/ai/index.js";
import { endpoint } from "../helpers/endpoints.js";
import { completionStream, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("provider follow-ups", () => {
  it("does not repeat fallback tool call ids across responses", async () => {
    stubFetch(async () => completionStream({
      choices: [{ index: 0, message: { content: null, tool_calls: [{ function: { name: "lookup", arguments: "{}" } }] }, finish_reason: "tool_calls" }],
    }));
    const provider = endpoint("vllm", { model: "llama3" });

    const first = await complete(provider, { messages: [] });
    const second = await complete(provider, { messages: [] });
    const idOf = (message: typeof first.message) => message.content.find((block) => block.type === "tool_call")?.id;

    expect(idOf(first.message)).toMatch(/^vllm-[A-Za-z0-9]{8}-0$/);
    expect(idOf(first.message)).not.toBe(idOf(second.message));
  });

  it("maps provider-side stops with tool calls to error", async () => {
    stubFetch(async () => completionStream({
      choices: [{ index: 0, message: { content: null, tool_calls: [{ id: "c1", function: { name: "x", arguments: "{}" } }] }, finish_reason: "error" }],
    }));
    expect((await complete(endpoint("openrouter", { model: "a/b" }), { messages: [] })).finishReason).toBe("error");
  });

  it("keeps a Mistral length cut when tool calls are still pending", async () => {
    const sse = [
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"x","arguments":"{}"}}]},"finish_reason":null}]}',
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"model_length"}]}',
      "data: [DONE]",
    ].join("\n\n") + "\n\n";
    stubFetch(async () => textResponse(sse, "text/event-stream"));

    expect((await complete(endpoint("mistral", { model: "m" }), { messages: [] })).finishReason).toBe("max_tokens");
  });
});
