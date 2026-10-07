import { afterEach, describe, expect, it } from "bun:test";
import { ollama } from "../../../src/ai/index.js";
import { fixtureJson, fixtureText, jsonResponse, textResponse, stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("ollama provider", () => {
  it("supports complete", async () => {
    stubFetch(async () => jsonResponse(await fixtureJson("../fixtures/ollama/complete.json")));

    const provider = ollama("llama3.1");
    const result = await provider.complete({ messages: [] });
    expect(result.message.content[0]).toEqual({ type: "text", text: "hello" });
    expect(result.usage?.total).toBe(5);
  });

  it("streams text and tool calls", async () => {
    stubFetch(async () =>
      textResponse(await fixtureText("../fixtures/ollama/stream.ndjson"), "application/x-ndjson"));

    const provider = ollama("llama3.1");
    const events = [];
    for await (const event of provider.stream({ messages: [] })) events.push(event);

    expect(events.filter((event) => event.type === "block_end" && event.block.type === "text")).toHaveLength(1);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(true);
    expect(events.some((event) => event.type === "usage")).toBe(true);
  });

  it("sends temperature 0 explicitly", async () => {
    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/ollama/complete.json"));
    });

    const provider = ollama("llama3.1", { temperature: 0.8 });
    await provider.complete({ messages: [], temperature: 0 });

    expect(capturedBody.options).toEqual({ temperature: 0 });
  });

  it("maps responseFormat to native format schema", async () => {
    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/ollama/complete.json"));
    });

    const schema = { type: "object", properties: { ok: { type: "boolean" } }, required: ["ok"] };
    const provider = ollama("llama3.1");
    await provider.complete({
      messages: [],
      responseFormat: { type: "json_schema", name: "result", schema },
    });

    expect(capturedBody.format).toEqual(schema);
  });
  const streamFixture = async (fixture: string) => {
    stubFetch(async () => textResponse(await fixtureText(fixture), "application/x-ndjson"));
    const events = [];
    for await (const event of ollama("llama3.1").stream({ messages: [] })) events.push(event);
    return events;
  };

  it("maps a mid-stream error line to a provider error", async () => {
    const events = await streamFixture("../fixtures/ollama/stream-error.ndjson");

    expect(events.at(-1)).toMatchObject({
      type: "issue",
      issue: { kind: "provider_error", message: "ollama stream error: model runner has unexpectedly stopped" },
    });
  });

  it("reports a stream without a done line as a provider error", async () => {
    const events = await streamFixture("../fixtures/ollama/truncated.ndjson");

    expect(events.at(-1)).toMatchObject({ type: "issue", issue: { kind: "provider_error", retryable: true } });
  });

  it("maps done_reason length to max_tokens", async () => {
    const events = await streamFixture("../fixtures/ollama/length.ndjson");

    expect(events.at(-1)).toMatchObject({ type: "usage", finishReason: "max_tokens" });
  });

  it("passes maxOutputTokens as num_predict", async () => {
    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/ollama/complete.json"));
    });

    await ollama("llama3.1").complete({ messages: [], maxOutputTokens: 64 });

    expect(capturedBody.options).toEqual({ num_predict: 64 });
  });

  it("streams thinking before the answer", async () => {
    stubFetch(async () => textResponse(await fixtureText("../fixtures/ollama/thinking.ndjson"), "application/x-ndjson"));

    const events = [];
    for await (const event of ollama("gpt-oss").stream({ messages: [] })) events.push(event);
    const blocks = events.flatMap((event) => (event.type === "block_end" ? [event.block] : []));

    expect(blocks).toEqual([{ type: "thinking", thinking: "Let me think." }, { type: "text", text: "Answer." }]);
  });

  it("maps reasoningEffort to think", async () => {
    const bodies: Record<string, unknown>[] = [];
    stubFetch(async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return jsonResponse(await fixtureJson("../fixtures/ollama/complete.json"));
    });

    await ollama("gpt-oss").complete({ messages: [] });
    await ollama("gpt-oss", { reasoningEffort: "high" }).complete({ messages: [] });
    await ollama("gpt-oss", { reasoningEffort: "high" }).complete({ messages: [], reasoningEffort: "none" });
    await ollama("qwen3").complete({ messages: [], extraBody: { think: true } });

    expect(bodies.map((body) => body.think)).toEqual([undefined, "high", false, true]);
  });
});
