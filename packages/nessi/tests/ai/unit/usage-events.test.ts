import { afterEach, describe, expect, it } from "bun:test";
import { complete, openAICompatible } from "../../../src/ai/index.js";
import type { RawStreamEvent, StreamEvent } from "../../../src/ai/types.js";
import { normalizeProviderStream } from "../../../src/ai/shared/tool-stream-normalizer.js";
import { mergeUsage } from "../../../src/aggregates.js";
import { completionStream, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const request = { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "Hi" }] }] };
const sse = (...records: Array<[string | undefined, unknown]>) => records
  .map(([event, data]) => `${event ? `event: ${event}\n` : ""}data: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`)
  .join("");

const collect = async (events: AsyncIterable<StreamEvent>) => {
  const all: StreamEvent[] = [];
  for await (const event of events) all.push(event);
  return all;
};

const textBlocks = (events: StreamEvent[]) =>
  events.flatMap((event) => (event.type === "block_end" && event.block.type === "text" ? [event.block.text] : []));

describe("interim usage events", () => {
  it("do not split the open text block", async () => {
    const raw = async function* (): AsyncIterable<RawStreamEvent> {
      yield { type: "text", delta: "Hel" };
      yield { type: "usage", usage: { input: 3, output: 1, total: 4 } };
      yield { type: "text", delta: "lo" };
      yield { type: "usage", usage: { input: 3, output: 2, total: 5 }, finishReason: "stop" };
    };
    expect(textBlocks(await collect(normalizeProviderStream(raw())))).toEqual(["Hello"]);
  });

  it("keep a signed thinking block whole and close it on the final usage", async () => {
    const raw = async function* (): AsyncIterable<RawStreamEvent> {
      yield { type: "thinking", delta: "Let me " };
      yield { type: "usage", usage: { input: 3, output: 1, total: 4 } };
      yield { type: "thinking", delta: "think.", signature: "sig" };
      yield { type: "usage", usage: { input: 3, output: 2, total: 5 }, finishReason: "stop" };
    };
    const blocks = (await collect(normalizeProviderStream(raw())))
      .flatMap((event) => (event.type === "block_end" ? [event.block] : []));
    expect(blocks).toEqual([{ type: "thinking", thinking: "Let me think.", signature: "sig" }]);
  });

  it("keep one text block when a gateway sends usage with every chunk", async () => {
    const usage = (output: number) => ({ prompt_tokens: 3, completion_tokens: output, total_tokens: 3 + output });
    stubFetch(() => textResponse(sse(
      [undefined, { choices: [{ index: 0, delta: { content: "Hel" }, finish_reason: null }], usage: usage(1) }],
      [undefined, { choices: [{ index: 0, delta: { content: "lo" }, finish_reason: null }], usage: usage(2) }],
      [undefined, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: usage(2) }],
      [undefined, "[DONE]"],
    ), "text/event-stream"));
    const events = await collect(openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }).stream(request));
    expect(textBlocks(events)).toEqual(["Hello"]);
  });
});

describe("cache usage", () => {
  const openai = () => openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" });
  const cachedUsage = { prompt_tokens: 100, completion_tokens: 5, total_tokens: 105, prompt_tokens_details: { cached_tokens: 80 } };

  it("reads OpenAI cached prompt tokens as a part of input", async () => {
    stubFetch(() => completionStream({ choices: [{ index: 0, message: { content: "ok" }, finish_reason: "stop" }], usage: cachedUsage }));
    expect((await complete(openai(), request)).usage).toEqual({ input: 100, output: 5, total: 105, cacheRead: 80 });

    stubFetch(() => textResponse(sse(
      [undefined, { choices: [{ index: 0, delta: { content: "ok" }, finish_reason: "stop" }] }],
      [undefined, { choices: [], usage: cachedUsage }],
      [undefined, "[DONE]"],
    ), "text/event-stream"));
    const events = await collect(openai().stream(request));
    expect(events.at(-1)).toMatchObject({ type: "usage", usage: { input: 100, cacheRead: 80, total: 105 } });
  });


  it("adds cache reads across turns", () => {
    expect(mergeUsage({ input: 10, output: 1, total: 11 }, { input: 5, output: 1, total: 6, cacheRead: 3 }))
      .toEqual({ input: 15, output: 2, total: 17, cacheRead: 3 });
  });
});

describe("reasoning usage", () => {
  it("reads reasoning tokens as a part of output and adds them across turns", async () => {
    stubFetch(() => textResponse([
      `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "ok" }, finish_reason: "stop" }] })}`,
      `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 40, total_tokens: 50, completion_tokens_details: { reasoning_tokens: 32 } } })}`,
      "data: [DONE]",
    ].join("\n\n") + "\n\n", "text/event-stream"));
    const events = await collect(openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }).stream(request));
    expect(events.at(-1)).toMatchObject({ type: "usage", usage: { input: 10, output: 40, total: 50, reasoning: 32 } });
    expect(mergeUsage({ input: 1, output: 40, total: 41, reasoning: 32 }, { input: 1, output: 5, total: 6 }))
      .toEqual({ input: 2, output: 45, total: 47, reasoning: 32 });
  });
});
