import { afterEach, describe, expect, it } from "bun:test";
import { anthropic, gemini, mistral, ollama, openAICompatible, ProviderError } from "../../../src/ai/index.js";
import type { Provider, StreamEvent } from "../../../src/ai/types.js";
import { jsonResponse, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const request = { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "Hi" }] }] };

const sse = (...records: Array<[string | undefined, unknown]>) => records
  .map(([event, data]) => `${event ? `event: ${event}\n` : ""}data: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`)
  .join("");
const ndjson = (...records: unknown[]) => records.map((record) => (typeof record === "string" ? record : JSON.stringify(record))).join("\n") + "\n";

type Case = {
  name: string;
  create: () => Provider;
  /** A stream with `bad` between a text record and a normal end. */
  stream: (bad: string | Record<string, unknown>) => Response;
  /** An in-band error record this provider sends. */
  envelope: Record<string, unknown> | string;
};

const chatStream = (bad: string | Record<string, unknown>) => textResponse(sse(
  [undefined, { choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: null }] }],
  [undefined, bad],
  [undefined, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }],
  [undefined, "[DONE]"],
), "text/event-stream");

const cases: Case[] = [
  {
    name: "openai-compatible",
    create: () => openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }),
    stream: chatStream,
    envelope: { error: { message: "rate limited", code: 429 } },
  },
  {
    name: "mistral",
    create: () => mistral("test", { apiKey: "test", baseURL: "https://example.test" }),
    stream: chatStream,
    envelope: { object: "error", message: "Service unavailable", type: "service_unavailable" },
  },
  {
    name: "gemini",
    create: () => gemini("test", { apiKey: "test", baseURL: "https://example.test" }),
    stream: (bad) => textResponse(sse(
      [undefined, { candidates: [{ content: { parts: [{ text: "Hello" }] } }] }],
      [undefined, bad],
      [undefined, { candidates: [{ content: { parts: [{ text: "!" }] }, finishReason: "STOP" }] }],
    ), "text/event-stream"),
    envelope: { error: { code: 503, message: "The model is overloaded.", status: "UNAVAILABLE" } },
  },
  {
    name: "anthropic",
    create: () => anthropic("test", { apiKey: "test", baseURL: "https://example.test" }),
    stream: (bad) => textResponse(sse(
      ["message_start", { type: "message_start", message: { usage: { input_tokens: 1, output_tokens: 0 } } }],
      ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
      ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } }],
      // Without an event name, as some proxies forward it.
      [undefined, bad],
      ["content_block_stop", { type: "content_block_stop", index: 0 }],
      ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }],
      ["message_stop", { type: "message_stop" }],
    ), "text/event-stream"),
    envelope: { type: "error", error: { type: "overloaded_error", message: "Overloaded" } },
  },
  {
    name: "ollama",
    create: () => ollama("test", { baseURL: "https://example.test" }),
    stream: (bad) => textResponse(ndjson(
      { message: { content: "Hello" }, done: false },
      bad,
      { done: true, done_reason: "stop" },
    ), "application/x-ndjson"),
    envelope: { error: "model runner has unexpectedly stopped" },
  },
];

const issuesOf = async (provider: Provider) => {
  const events: StreamEvent[] = [];
  for await (const event of provider.stream(request)) events.push(event);
  return events.flatMap((event) => (event.type === "issue" ? [event.issue] : []));
};

describe("stream records", () => {
  for (const { name, create, stream, envelope } of cases) {
    it(`${name} fails a stream with a malformed record instead of returning the rest`, async () => {
      for (const bad of ['{"choices":[{"delta":{"content":"lost', "[1,2]"]) {
        stubFetch(() => stream(bad));
        expect(await issuesOf(create())).toEqual([expect.objectContaining({
          kind: "provider_error",
          retryable: true,
          message: expect.stringContaining("malformed stream record"),
        })]);
      }
    });

    it(`${name} reports an in-band error record even when a normal end follows`, async () => {
      stubFetch(() => stream(envelope));
      expect(await issuesOf(create())).toEqual([expect.objectContaining({ kind: "provider_error", retryable: true })]);
    });

    it(`${name} complete() rejects an HTTP 200 error body`, async () => {
      stubFetch(() => jsonResponse({ error: { message: "upstream failed", code: 500 } }));
      const error = await create().complete(request).catch((caught: unknown) => caught);
      expect(error).toBeInstanceOf(ProviderError);
      expect(error).toMatchObject({ retryable: true, status: 500, message: expect.stringContaining("upstream failed") });
    });
  }

  it("skips empty SSE data records", async () => {
    stubFetch(() => textResponse(sse(
      [undefined, { choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: null }] }],
      [undefined, " "],
      [undefined, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }],
      [undefined, "[DONE]"],
    ), "text/event-stream"));
    expect(await issuesOf(cases[0]!.create())).toEqual([]);
  });

  it("classifies a non-retryable in-band status", async () => {
    stubFetch(() => chatStream({ error: { message: "Invalid API key", code: 401 } }));
    expect(await issuesOf(cases[0]!.create())).toEqual([expect.objectContaining({ kind: "provider_error", retryable: false, status: 401 })]);
  });
});

describe("success-shaped bodies without a result", () => {
  for (const [name, create, body] of [
    ["openai-compatible", cases[0]!.create, {}],
    ["mistral", cases[1]!.create, { choices: [] }],
    ["gemini", cases[2]!.create, { candidates: [] }],
    ["anthropic", cases[3]!.create, { id: "msg_1" }],
  ] as const) {
    it(`${name} complete() rejects a body without a result`, async () => {
      stubFetch(() => jsonResponse(body));
      await expect(create().complete(request)).rejects.toBeInstanceOf(ProviderError);
    });
  }

  it("gemini complete() reports a blocked prompt like the stream does", async () => {
    stubFetch(() => jsonResponse({ promptFeedback: { blockReason: "SAFETY" } }));
    const error = await cases[2]!.create().complete(request).catch((caught: unknown) => caught);
    expect(error).toMatchObject({ retryable: false, message: expect.stringContaining("SAFETY") });
  });
});

describe("OpenAI refusals", () => {
  it("complete() returns the refusal as text that ends with a provider-side stop", async () => {
    stubFetch(() => jsonResponse({
      choices: [{ index: 0, message: { role: "assistant", content: null, refusal: "I can't help with that." }, finish_reason: "stop" }],
    }));
    const result = await cases[0]!.create().complete(request);
    expect(result.finishReason).toBe("error");
    expect(result.message.content).toEqual([{ type: "text", text: "I can't help with that." }]);
  });

  it("stream() ends a streamed refusal with a provider-side stop", async () => {
    stubFetch(() => textResponse(sse(
      [undefined, { choices: [{ index: 0, delta: { refusal: "I can't" }, finish_reason: null }] }],
      [undefined, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }],
      [undefined, "[DONE]"],
    ), "text/event-stream"));
    const events: StreamEvent[] = [];
    for await (const event of cases[0]!.create().stream(request)) events.push(event);
    expect(events.at(-1)).toMatchObject({ type: "usage", finishReason: "error" });
    expect(events).toContainEqual(expect.objectContaining({ type: "block_end", block: { type: "text", text: "I can't" } }));
  });
});
