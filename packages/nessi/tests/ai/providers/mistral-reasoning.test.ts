import { afterEach, describe, expect, it } from "bun:test";
import { completeFromStream } from "../../../src/ai/index.js";
import { endpoint } from "../helpers/endpoints.js";
import type { AssistantMessage, GenerateRequest } from "../../../src/ai/index.js";
import { fixtureText, jsonResponse, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const sentBody = async (request: Partial<GenerateRequest>, options: Parameters<typeof endpoint>[1] = {}) => {
  let body: Record<string, unknown> = {};
  stubFetch(async (_input, init) => {
    body = JSON.parse(String(init?.body));
    return jsonResponse({ choices: [{ index: 0, message: { content: "ok" }, finish_reason: "stop" }] });
  });
  await endpoint("mistral", { model: "mistral-medium", ...options }).complete({ messages: [], ...request });
  return body;
};

describe("Mistral reasoning through openAICompatible", () => {
  it("streams thinking with its signature before the answer", async () => {
    stubFetch(async () => textResponse(await fixtureText("../fixtures/mistral/reasoning-signed.sse"), "text/event-stream"));

    const { message } = await completeFromStream(endpoint("mistral", { model: "mistral-medium" }), { messages: [] });

    expect(message.provider).toBe("mistral");
    expect(message.content).toEqual([
      { type: "thinking", thinking: "Check units.", signature: "m-sig" },
      { type: "text", text: "42 km" },
    ]);
  });

  it("parses thinking chunks from non-streaming responses in order", async () => {
    stubFetch(async () => jsonResponse({
      choices: [{
        index: 0,
        message: {
          content: [
            { type: "thinking", thinking: [{ type: "text", text: "Hmm." }], signature: "s" },
            { type: "text", text: "Yes." },
          ],
        },
        finish_reason: "stop",
      }],
    }));

    const { message } = await endpoint("mistral", { model: "mistral-medium" }).complete({ messages: [] });

    expect(message.content).toEqual([{ type: "thinking", thinking: "Hmm.", signature: "s" }, { type: "text", text: "Yes." }]);
  });

  it("replays its own reasoning as chunks and keeps other assistant messages as text", async () => {
    const own: AssistantMessage = {
      role: "assistant",
      provider: "mistral",
      content: [{ type: "thinking", thinking: "Check units.", signature: "m-sig" }, { type: "text", text: "42 km" }],
    };
    const foreign: AssistantMessage = {
      role: "assistant",
      provider: "anthropic",
      content: [{ type: "thinking", thinking: "x", signature: "a-sig" }, { type: "text", text: "Hi" }],
    };

    const body = await sentBody({ messages: [{ role: "user", content: ["a"] }, own, { role: "user", content: ["b"] }, foreign] });

    const messages = body.messages as Array<Record<string, unknown>>;
    expect(messages[1]).toEqual({
      role: "assistant",
      content: [
        { type: "thinking", thinking: [{ type: "text", text: "Check units." }], signature: "m-sig" },
        { type: "text", text: "42 km" },
      ],
    });
    expect(messages[3]).toEqual({ role: "assistant", content: "Hi" });
  });

  it("sends reasoning_effort unchanged", async () => {
    expect(await sentBody({})).not.toHaveProperty("reasoning_effort");
    expect((await sentBody({}, { reasoningEffort: "high" })).reasoning_effort).toBe("high");
    expect((await sentBody({ reasoningEffort: "none" }, { reasoningEffort: "high" })).reasoning_effort).toBe("none");
  });
});

describe("Mistral signed reasoning in a tool loop", () => {
  it("keeps the signature through storage and replays it with mapped tool ids", async () => {
    const { nessi } = await import("../../../src/nessi.js");
    const { memoryStore } = await import("../../../src/stores.js");
    const { defineTool } = await import("../../../src/tools.js");
    const { z } = await import("zod");
    const lookup = defineTool({ name: "lookup", description: "Look up", inputSchema: z.object({ q: z.string() }) })
      .server(async () => "found");
    const sse = (...records: unknown[]) => records.map((record) => `data: ${typeof record === "string" ? record : JSON.stringify(record)}\n\n`).join("");
    const bodies: Array<Record<string, unknown>> = [];
    stubFetch(async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return textResponse(bodies.length === 1
        ? sse(
          { choices: [{ index: 0, delta: { content: [{ type: "thinking", thinking: [{ type: "text", text: "Need a lookup." }], signature: "m-sig" }] }, finish_reason: null }] },
          { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "provider-long-id", function: { name: "lookup", arguments: '{"q":"x"}' } }] }, finish_reason: null }] },
          { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
          "[DONE]",
        )
        : sse({ choices: [{ index: 0, delta: { content: "Done." }, finish_reason: "stop" }] }, "[DONE]"), "text/event-stream");
    });

    const store = memoryStore();
    for await (const _event of nessi({ provider: endpoint("mistral", { model: "mistral-medium" }), store, systemPrompt: "s", input: "Find x", tools: [lookup] })) {
      // drain
    }

    const stored = (await store.load()).find(({ message }) => message.role === "assistant")?.message;
    expect(stored).toMatchObject({ provider: "mistral", content: expect.arrayContaining([{ type: "thinking", thinking: "Need a lookup.", signature: "m-sig" }]) });
    const replay = bodies[1]!.messages as Array<Record<string, unknown>>;
    const assistant = replay.find((message) => message.role === "assistant")!;
    expect(assistant.content).toEqual([{ type: "thinking", thinking: [{ type: "text", text: "Need a lookup." }], signature: "m-sig" }]);
    const [toolCall] = assistant.tool_calls as Array<{ id: string }>;
    expect(toolCall!.id).toMatch(/^[A-Za-z0-9]{9}$/);
    expect(replay.find((message) => message.role === "tool")).toMatchObject({ tool_call_id: toolCall!.id, name: "lookup", content: "found" });
  });
});
