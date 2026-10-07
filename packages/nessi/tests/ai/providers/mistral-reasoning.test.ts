import { afterEach, describe, expect, it } from "bun:test";
import { completeFromStream, mistral } from "../../../src/ai/index.js";
import type { AssistantMessage, GenerateRequest } from "../../../src/ai/index.js";
import { fixtureText, jsonResponse, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const sentBody = async (request: Partial<GenerateRequest>, options: Parameters<typeof mistral>[1] = {}) => {
  let body: Record<string, unknown> = {};
  stubFetch(async (_input, init) => {
    body = JSON.parse(String(init?.body));
    return jsonResponse({ choices: [{ index: 0, message: { content: "ok" }, finish_reason: "stop" }] });
  });
  await mistral("mistral-medium", { apiKey: "k", ...options }).complete({ messages: [], ...request });
  return body;
};

describe("mistral reasoning", () => {
  it("streams thinking with its signature before the answer", async () => {
    stubFetch(async () => textResponse(await fixtureText("../fixtures/mistral/reasoning-signed.sse"), "text/event-stream"));

    const { message } = await completeFromStream(mistral("mistral-medium", { apiKey: "k" }), { messages: [] });

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

    const { message } = await mistral("mistral-medium", { apiKey: "k" }).complete({ messages: [] });

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
    expect(await sentBody({ disableReasoning: true })).not.toHaveProperty("reasoning_effort");
  });
});
