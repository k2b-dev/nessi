import { afterEach, describe, expect, it } from "bun:test";
import { gemini } from "../../../src/ai/index.js";
import { fixtureJson, fixtureText, jsonResponse, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("gemini provider", () => {
  it("maps complete responses and function calls", async () => {
    globalThis.fetch = (async () => jsonResponse(await fixtureJson("../fixtures/gemini/complete.json"))) as typeof fetch;

    const provider = gemini("gemini-2.0-flash", { apiKey: "x" });
    const result = await provider.complete({ messages: [] });
    expect(result.message.content.some((block) => block.type === "tool_call")).toBe(true);
    expect(result.usage?.total).toBe(3);
  });

  it("streams text and function calls", async () => {
    globalThis.fetch = (async () =>
      textResponse(await fixtureText("../fixtures/gemini/stream.sse"), "text/event-stream")) as typeof fetch;

    const provider = gemini("gemini-2.0-flash", { apiKey: "x" });
    const events = [];
    for await (const event of provider.stream({ messages: [] })) events.push(event);

    expect(events.some((event) => event.type === "block_end" && event.block.type === "text")).toBe(true);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(true);
  });

  it("sends temperature 0 explicitly", async () => {
    let capturedBody: any;
    globalThis.fetch = (async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/gemini/complete.json"));
    }) as typeof fetch;

    const provider = gemini("gemini-2.0-flash", { apiKey: "x", temperature: 0.8 });
    await provider.complete({ messages: [], temperature: 0 });

    expect(capturedBody.generationConfig.temperature).toBe(0);
  });

  it("maps responseFormat to response schema generation config", async () => {
    let capturedBody: any;
    globalThis.fetch = (async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/gemini/complete.json"));
    }) as typeof fetch;

    const schema = { type: "object", properties: { title: { type: "string" } }, required: ["title"] };
    const provider = gemini("gemini-2.0-flash", { apiKey: "x" });
    await provider.complete({
      messages: [],
      responseFormat: { type: "json_schema", name: "card", schema },
    });

    expect(capturedBody.generationConfig.responseMimeType).toBe("application/json");
    expect(capturedBody.generationConfig.responseJsonSchema).toEqual(schema);
    expect(capturedBody.generationConfig.responseSchema).toBeUndefined();
  });
  it("wraps non-object tool results and groups parallel responses into one content", async () => {
    let capturedBody: any;
    globalThis.fetch = (async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/gemini/complete.json"));
    }) as typeof fetch;

    await gemini("gemini-2.0-flash", { apiKey: "x" }).complete({
      messages: [
        { role: "user", content: ["hi"] },
        {
          role: "assistant",
          content: [
            { type: "tool_call", id: "a", name: "search", args: {} },
            { type: "tool_call", id: "b", name: "lookup", args: {} },
            { type: "tool_call", id: "c", name: "fetch", args: {} },
          ],
        },
        { role: "tool_result", callId: "a", name: "search", result: "the answer is 42" },
        { role: "tool_result", callId: "b", name: "lookup", result: { ok: true } },
        { role: "tool_result", callId: "c", name: "fetch", result: "not found", isError: true },
        { role: "user", content: ["thanks"] },
      ],
    });

    expect(capturedBody.contents.slice(2)).toEqual([
      {
        role: "user",
        parts: [
          { functionResponse: { name: "search", response: { output: "the answer is 42" } } },
          { functionResponse: { name: "lookup", response: { ok: true } } },
          { functionResponse: { name: "fetch", response: { error: "not found" } } },
        ],
      },
      { role: "user", parts: [{ text: "thanks" }] },
    ]);
  });

  it("keeps streamed text in one block and reports usage with thinking tokens once", async () => {
    globalThis.fetch = (async () =>
      textResponse(await fixtureText("../fixtures/gemini/multi-chunk.sse"), "text/event-stream")) as typeof fetch;

    const events = [];
    for await (const event of gemini("gemini-2.5-flash", { apiKey: "x" }).stream({ messages: [] })) events.push(event);

    expect(events.filter((event) => event.type === "block_end")).toEqual([
      { type: "block_end", blockId: "block-0", index: 0, block: { type: "text", text: "Hello" } },
    ]);
    expect(events.filter((event) => event.type === "usage")).toEqual([
      { type: "usage", usage: { input: 5, output: 12, total: 17, creditsUsed: 0 }, finishReason: "stop" },
    ]);
  });

  it("reports a stream without finish reason as a provider error", async () => {
    globalThis.fetch = (async () =>
      textResponse(await fixtureText("../fixtures/gemini/truncated.sse"), "text/event-stream")) as typeof fetch;

    const events = [];
    for await (const event of gemini("gemini-2.5-flash", { apiKey: "x" }).stream({ messages: [] })) events.push(event);

    expect(events.at(-1)).toMatchObject({ type: "issue", issue: { kind: "provider_error", retryable: true } });
  });

  it("reports a blocked prompt as a non-retryable error and keeps its usage", async () => {
    globalThis.fetch = (async () =>
      textResponse(await fixtureText("../fixtures/gemini/blocked.sse"), "text/event-stream")) as typeof fetch;

    const events = [];
    for await (const event of gemini("gemini-2.5-flash", { apiKey: "x" }).stream({ messages: [] })) events.push(event);

    expect(events).toContainEqual(expect.objectContaining({ type: "usage", usage: expect.objectContaining({ input: 7 }) }));
    expect(events.at(-1)).toMatchObject({
      type: "issue",
      issue: { kind: "provider_error", message: "gemini blocked the prompt (SAFETY).", retryable: false },
    });
  });
});
