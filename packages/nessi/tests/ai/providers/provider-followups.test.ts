import { afterEach, describe, expect, it } from "bun:test";
import { anthropic, completeFromStream, gemini, mistral, ollama, openrouter } from "../../../src/ai/index.js";
import { jsonResponse, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("provider follow-ups", () => {
  it("reports Gemini safety and malformed-call stops as errors", async () => {
    const finish = async (finishReason: string) => {
      stubFetch(async () => jsonResponse({ candidates: [{ content: { role: "model", parts: [{ text: "x" }] }, finishReason }] }));
      return (await gemini("gemini-3-flash", { apiKey: "k" }).complete({ messages: [] })).finishReason;
    };

    expect(await finish("STOP")).toBe("stop");
    expect(await finish("MAX_TOKENS")).toBe("max_tokens");
    expect(await finish("SAFETY")).toBe("error");
    expect(await finish("MALFORMED_FUNCTION_CALL")).toBe("error");
  });

  it("sends an explicit Ollama contextWindow as num_ctx", async () => {
    const bodies: Array<{ options?: Record<string, unknown> }> = [];
    stubFetch(async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return jsonResponse({ message: { role: "assistant", content: "ok" }, done: true });
    });

    await ollama("llama3").complete({ messages: [] });
    await ollama("llama3", { contextWindow: 32_768 }).complete({ messages: [] });

    expect(bodies[0]!.options).toBeUndefined();
    expect(bodies[1]!.options).toEqual({ num_ctx: 32_768 });
  });

  it("does not repeat fallback tool call ids across responses", async () => {
    stubFetch(async () => jsonResponse({
      message: { role: "assistant", content: "", tool_calls: [{ function: { name: "lookup", arguments: {} } }] },
      done: true,
    }));
    const provider = ollama("llama3");

    const first = await provider.complete({ messages: [] });
    const second = await provider.complete({ messages: [] });
    const idOf = (message: typeof first.message) => message.content.find((block) => block.type === "tool_call")?.id;

    expect(idOf(first.message)).toMatch(/^ollama-[A-Za-z0-9]{8}-0$/);
    expect(idOf(first.message)).not.toBe(idOf(second.message));
  });

  it("gives Anthropic tool calls without an id a fallback id in complete()", async () => {
    stubFetch(async () => jsonResponse({ content: [{ type: "tool_use", name: "lookup", input: {} }], stop_reason: "tool_use" }));

    const { message } = await anthropic("claude", { apiKey: "k" }).complete({ messages: [] });

    expect(message.content[0]).toMatchObject({ type: "tool_call", name: "lookup", id: expect.stringMatching(/^anthropic-[A-Za-z0-9]{8}-0$/) });
  });

  it("maps provider-side stops with tool calls to error across adapters", async () => {
    stubFetch(async () => jsonResponse({
      choices: [{ index: 0, message: { content: null, tool_calls: [{ id: "c1", function: { name: "x", arguments: "{}" } }] }, finish_reason: "error" }],
    }));
    expect((await openrouter("a/b", { apiKey: "k" }).complete({ messages: [] })).finishReason).toBe("error");

    stubFetch(async () => jsonResponse({ content: [{ type: "tool_use", id: "t1", name: "x", input: {} }], stop_reason: "refusal" }));
    expect((await anthropic("claude", { apiKey: "k" }).complete({ messages: [] })).finishReason).toBe("error");
  });

  it("keeps a Mistral length cut when tool calls are still pending", async () => {
    const sse = [
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"x","arguments":"{}"}}]},"finish_reason":null}]}',
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"model_length"}]}',
      "data: [DONE]",
    ].join("\n\n") + "\n\n";
    stubFetch(async () => textResponse(sse, "text/event-stream"));

    expect((await completeFromStream(mistral("m", { apiKey: "k" }), { messages: [] })).finishReason).toBe("max_tokens");
  });
});
