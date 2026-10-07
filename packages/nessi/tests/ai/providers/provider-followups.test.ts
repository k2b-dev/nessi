import { afterEach, describe, expect, it } from "bun:test";
import { gemini, ollama } from "../../../src/ai/index.js";
import { jsonResponse, stubFetch } from "../helpers/fixtures.js";

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
});
