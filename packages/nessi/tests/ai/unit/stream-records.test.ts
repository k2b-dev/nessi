import { afterEach, describe, expect, it } from "bun:test";
import { complete, ProviderError } from "../../../src/ai/index.js";
import { endpoint } from "../helpers/endpoints.js";
import type { Provider, StreamEvent } from "../../../src/ai/types.js";
import { completionStream, jsonResponse, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const request = { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "Hi" }] }] };

const sse = (...records: Array<[string | undefined, unknown]>) => records
  .map(([event, data]) => `${event ? `event: ${event}\n` : ""}data: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`)
  .join("");

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

// Each endpoint variant with an in-band error record in the shape that endpoint (or a gateway in
// front of it) sends.
const cases: Case[] = [
  {
    name: "openai",
    create: () => endpoint("openai", { baseURL: "https://example.test" }),
    stream: chatStream,
    envelope: { error: { message: "rate limited", code: 429 } },
  },
  {
    name: "mistral",
    create: () => endpoint("mistral", { baseURL: "https://example.test" }),
    stream: chatStream,
    envelope: { object: "error", message: "Service unavailable", type: "service_unavailable" },
  },
  {
    name: "openrouter",
    create: () => endpoint("openrouter", { baseURL: "https://example.test" }),
    stream: chatStream,
    envelope: { error: { code: 503, message: "The model is overloaded.", status: "UNAVAILABLE" } },
  },
  {
    name: "vllm",
    create: () => endpoint("vllm", { baseURL: "https://example.test" }),
    stream: chatStream,
    envelope: { error: "model runner has unexpectedly stopped" },
  },
  {
    name: "vllm behind a proxy",
    create: () => endpoint("vllm", { baseURL: "https://example.test" }),
    stream: chatStream,
    envelope: { type: "error", error: { type: "overloaded_error", message: "Overloaded" } },
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

    it(`${name} complete() rejects an HTTP 200 JSON error body sent instead of a stream`, async () => {
      stubFetch(() => jsonResponse({ error: { message: "upstream failed", code: 500 } }));
      const error = await complete(create(), request).catch((caught: unknown) => caught);
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

describe("answers without content", () => {
  it("keeps empty answers and content-free stops", async () => {
    stubFetch(() => completionStream({ choices: [{ index: 0, message: { role: "assistant", content: "" }, finish_reason: "stop" }] }));
    expect((await complete(cases[0]!.create(), request)).finishReason).toBe("stop");
    // A filter stop may come without a message.
    stubFetch(() => completionStream({ choices: [{ index: 0, finish_reason: "content_filter" }] }));
    expect((await complete(cases[2]!.create(), request)).finishReason).toBe("error");
  });
});

describe("OpenAI refusals", () => {
  it("complete() returns the refusal as text that ends with a provider-side stop", async () => {
    stubFetch(() => completionStream({
      choices: [{ index: 0, message: { role: "assistant", content: null, refusal: "I can't help with that." }, finish_reason: "stop" }],
    }));
    const result = await complete(cases[0]!.create(), request);
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
