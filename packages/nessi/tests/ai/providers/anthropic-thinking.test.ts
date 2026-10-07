import { afterEach, describe, expect, it } from "bun:test";
import { anthropic, completeFromStream } from "../../../src/ai/index.js";
import type { AssistantMessage, GenerateRequest, Message } from "../../../src/ai/index.js";
import { z } from "zod";
import { defineTool, memoryStore, nessi } from "../../../src/index.js";
import { fixtureText, jsonResponse, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const thinkingTurn: AssistantMessage = {
  role: "assistant",
  provider: "anthropic",
  content: [
    { type: "thinking", thinking: "Let me check the weather.", signature: "sig-A" },
    { type: "thinking", thinking: "", redacted: "enc-B" },
    { type: "thinking", thinking: "", signature: "sig-C" },
    { type: "text", text: "Checking." },
    { type: "tool_call", id: "tu_1", name: "weather", args: { city: "Ulm" } },
  ],
};

/** Captures the request body of one complete() call. */
const sentBody = async (request: Partial<GenerateRequest>, options: Parameters<typeof anthropic>[1] = {}) => {
  let body: Record<string, unknown> = {};
  stubFetch(async (_input, init) => {
    body = JSON.parse(String(init?.body));
    return jsonResponse({ content: [{ type: "text", text: "ok" }], stop_reason: "end_turn" });
  });
  await anthropic("claude", { apiKey: "k", ...options }).complete({ messages: [], ...request });
  return body;
};

describe("anthropic thinking", () => {
  it("streams thinking, signatures and redacted reasoning in order", async () => {
    stubFetch(async () => textResponse(await fixtureText("../fixtures/anthropic/thinking.sse"), "text/event-stream"));
    const provider = anthropic("claude", { apiKey: "k" });

    const { message, finishReason } = await completeFromStream(provider, { messages: [] });

    expect(finishReason).toBe("tool_use");
    expect(message.provider).toBe("anthropic");
    expect(message.content).toEqual(thinkingTurn.content);
  });

  it("parses thinking from non-streaming responses", async () => {
    stubFetch(async () => jsonResponse({
      content: [
        { type: "thinking", thinking: "Hmm.", signature: "sig-1" },
        { type: "redacted_thinking", data: "enc-2" },
        { type: "text", text: "Answer." },
      ],
      stop_reason: "end_turn",
      usage: { input_tokens: 3, output_tokens: 4 },
    }));

    const { message } = await anthropic("claude", { apiKey: "k" }).complete({ messages: [] });

    expect(message.provider).toBe("anthropic");
    expect(message.content).toEqual([
      { type: "thinking", thinking: "Hmm.", signature: "sig-1" },
      { type: "thinking", thinking: "", redacted: "enc-2" },
      { type: "text", text: "Answer." },
    ]);
  });

  it("sends its own thinking back unchanged and in order", async () => {
    const messages: Message[] = [
      { role: "user", content: [{ type: "text", text: "Weather in Ulm?" }] },
      thinkingTurn,
      { role: "tool_result", callId: "tu_1", name: "weather", result: "sunny" },
    ];

    const body = await sentBody({ messages });

    expect((body.messages as Array<{ role: string; content: unknown }>)[1]).toEqual({
      role: "assistant",
      content: [
        { type: "thinking", thinking: "Let me check the weather.", signature: "sig-A" },
        { type: "redacted_thinking", data: "enc-B" },
        { type: "thinking", thinking: "", signature: "sig-C" },
        { type: "text", text: "Checking." },
        { type: "tool_use", id: "tu_1", name: "weather", input: { city: "Ulm" } },
      ],
    });
  });

  it("drops reasoning from other providers and unsigned thinking", async () => {
    const foreign: AssistantMessage = {
      ...thinkingTurn,
      provider: "mistral",
      content: [{ type: "thinking", thinking: "x", signature: "mistral-sig" }, { type: "text", text: "Hi" }],
    };
    const unsigned: AssistantMessage = { role: "assistant", provider: "anthropic", content: [{ type: "thinking", thinking: "y" }, { type: "text", text: "" }, { type: "text", text: "Yo" }] };

    const body = await sentBody({ messages: [{ role: "user", content: ["a"] }, foreign, { role: "user", content: ["b"] }, unsigned] });

    const contents = (body.messages as Array<{ role: string; content: unknown }>).map((message) => message.content);
    expect(contents[1]).toEqual([{ type: "text", text: "Hi" }]);
    expect(contents[3]).toEqual([{ type: "text", text: "Yo" }]);
  });

  it("maps reasoningEffort to adaptive thinking and effort", async () => {
    expect(await sentBody({})).not.toHaveProperty("thinking");
    expect((await sentBody({})).max_tokens).toBe(8192);

    const high = await sentBody({ reasoningEffort: "high", responseFormat: { type: "json_schema", name: "x", schema: { type: "object" } } });
    expect(high.thinking).toEqual({ type: "adaptive" });
    expect(high.output_config).toEqual({ format: { type: "json_schema", schema: { type: "object" } }, effort: "high" });

    const none = await sentBody({ reasoningEffort: "none" }, { reasoningEffort: "max" });
    expect(none.thinking).toEqual({ type: "disabled" });
    expect(none.output_config).toBeUndefined();

    const legacy = await sentBody({ disableReasoning: true });
    expect(legacy.thinking).toBeUndefined();

    const budget = await sentBody({ extraBody: { thinking: { type: "enabled", budget_tokens: 2048 } } });
    expect(budget.thinking).toEqual({ type: "enabled", budget_tokens: 2048 });
  });

  it("keeps signed thinking across a tool loop", async () => {
    const bodies: Array<{ messages: Array<{ role: string; content: Array<{ type: string }> }> }> = [];
    const finalAnswer = [
      'event: content_block_delta\ndata: {"index":0,"delta":{"type":"text_delta","text":"Sunny in Ulm."}}\n\n',
      'event: message_delta\ndata: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
      'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ].join("");
    stubFetch(async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      const sse = bodies.length === 1 ? await fixtureText("../fixtures/anthropic/thinking.sse") : finalAnswer;
      return textResponse(sse, "text/event-stream");
    });
    const weather = defineTool({
      name: "weather",
      description: "Weather",
      inputSchema: z.object({ city: z.string() }),
    }).server(async () => "sunny");

    const store = memoryStore();
    for await (const _event of nessi({
      provider: anthropic("claude", { apiKey: "k" }),
      systemPrompt: "sys",
      store,
      input: "Weather in Ulm?",
      tools: [weather],
      reasoningEffort: "high",
    })) { /* drain */ }

    expect(bodies).toHaveLength(2);
    expect(bodies[1]!.messages[1]!.content.map((block) => block.type))
      .toEqual(["thinking", "redacted_thinking", "thinking", "text", "tool_use"]);
    expect(bodies[1]!.messages[2]!.content.map((block) => block.type)).toEqual(["tool_result"]);
    const stored = (await store.load()).map((entry) => entry.message);
    expect(stored[1]).toMatchObject({ role: "assistant", provider: "anthropic" });
  });
});
