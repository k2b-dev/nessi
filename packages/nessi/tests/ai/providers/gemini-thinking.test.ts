import { afterEach, describe, expect, it } from "bun:test";
import { completeFromStream, gemini } from "../../../src/ai/index.js";
import type { AssistantMessage, GenerateRequest, Message } from "../../../src/ai/index.js";
import { fixtureText, jsonResponse, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

type GeminiBody = {
  contents: Array<{ role: string; parts: Array<Record<string, unknown>> }>;
  generationConfig?: Record<string, unknown>;
};

const sentBody = async (request: Partial<GenerateRequest>, options: Parameters<typeof gemini>[1] = {}) => {
  let body = {} as GeminiBody;
  stubFetch(async (_input, init) => {
    body = JSON.parse(String(init?.body));
    return jsonResponse({ candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }] });
  });
  await gemini("gemini-3-flash", { apiKey: "k", ...options }).complete({ messages: [], ...request });
  return body;
};

describe("gemini thinking", () => {
  it("streams thought parts as thinking and keeps function call signatures", async () => {
    stubFetch(async () => textResponse(await fixtureText("../fixtures/gemini/thinking.sse"), "text/event-stream"));

    const { message } = await completeFromStream(gemini("gemini-3-flash", { apiKey: "k" }), { messages: [] });

    expect(message.provider).toBe("gemini");
    expect(message.content.map(({ ...block }) => ("id" in block ? { ...block, id: "id" } : block))).toEqual([
      { type: "thinking", thinking: "Need the weather." },
      { type: "tool_call", id: "id", name: "weather", args: { city: "Ulm" }, signature: "fc-sig-1" },
      { type: "tool_call", id: "id", name: "weather", args: { city: "Bonn" } },
    ]);
  });

  it("attaches a trailing empty-text signature to the answer", async () => {
    stubFetch(async () => textResponse(await fixtureText("../fixtures/gemini/text-signature.sse"), "text/event-stream"));

    const { message } = await completeFromStream(gemini("gemini-3-flash", { apiKey: "k" }), { messages: [] });

    expect(message.content).toEqual([{ type: "text", text: "Sunny in Ulm.", signature: "text-sig" }]);
  });

  it("parses thoughts and signatures from non-streaming responses", async () => {
    stubFetch(async () => jsonResponse({
      candidates: [{
        content: { role: "model", parts: [{ text: "Hmm.", thought: true }, { text: "Answer.", thoughtSignature: "t-sig" }] },
        finishReason: "STOP",
      }],
    }));

    const { message } = await gemini("gemini-3-flash", { apiKey: "k" }).complete({ messages: [] });

    expect(message.content).toEqual([
      { type: "thinking", thinking: "Hmm." },
      { type: "text", text: "Answer.", signature: "t-sig" },
    ]);
  });

  it("returns signatures in their parts and skips validation for foreign calls", async () => {
    const own: AssistantMessage = {
      role: "assistant",
      provider: "gemini",
      content: [
        { type: "thinking", thinking: "Need the weather.", signature: "thought-sig" },
        { type: "tool_call", id: "a", name: "weather", args: { city: "Ulm" }, signature: "fc-sig-1" },
        { type: "tool_call", id: "b", name: "weather", args: { city: "Bonn" } },
      ],
    };
    const foreign: AssistantMessage = {
      role: "assistant",
      provider: "anthropic",
      content: [
        { type: "text", text: "Checking.", signature: "anthropic-ignored" },
        { type: "tool_call", id: "c", name: "weather", args: { city: "Rom" } },
        { type: "tool_call", id: "d", name: "weather", args: { city: "Oslo" } },
      ],
    };
    const messages: Message[] = [
      { role: "user", content: ["Weather?"] },
      own,
      { role: "tool_result", callId: "a", name: "weather", result: "sunny" },
      { role: "tool_result", callId: "b", name: "weather", result: "rain" },
      foreign,
      { role: "tool_result", callId: "c", name: "weather", result: "warm" },
      { role: "tool_result", callId: "d", name: "weather", result: "cold" },
    ];

    const body = await sentBody({ messages });

    expect(body.contents.map((content) => content.role)).toEqual(["user", "model", "user", "model", "user"]);
    expect(body.contents[1]!.parts).toEqual([
      { text: "Need the weather.", thought: true, thoughtSignature: "thought-sig" },
      { functionCall: { name: "weather", args: { city: "Ulm" } }, thoughtSignature: "fc-sig-1" },
      { functionCall: { name: "weather", args: { city: "Bonn" } } },
    ]);
    expect(body.contents[3]!.parts).toEqual([
      { text: "Checking." },
      { functionCall: { name: "weather", args: { city: "Rom" } }, thoughtSignature: "skip_thought_signature_validator" },
      { functionCall: { name: "weather", args: { city: "Oslo" } } },
    ]);
  });

  it("maps reasoningEffort to thinkingConfig", async () => {
    expect((await sentBody({})).generationConfig).toBeUndefined();
    expect((await sentBody({}, { reasoningEffort: "low" })).generationConfig).toEqual({ thinkingConfig: { thinkingLevel: "low" } });
    expect((await sentBody({ reasoningEffort: "none" }, { reasoningEffort: "high" })).generationConfig)
      .toEqual({ thinkingConfig: { thinkingBudget: 0 } });
    expect((await sentBody({ disableReasoning: true })).generationConfig).toEqual({ thinkingConfig: { thinkingBudget: 0 } });
    expect((await sentBody({ disableReasoning: true, reasoningEffort: "high" })).generationConfig)
      .toEqual({ thinkingConfig: { thinkingLevel: "high" } });
    expect((await sentBody({
      reasoningEffort: "high",
      extraBody: { generationConfig: { thinkingConfig: { includeThoughts: true } } },
    })).generationConfig).toEqual({ thinkingConfig: { thinkingLevel: "high", includeThoughts: true } });
  });
});
