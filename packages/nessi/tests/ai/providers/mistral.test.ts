import { afterEach, describe, expect, it } from "bun:test";
import { mistral } from "../../../src/ai/index.js";
import { fixtureJson, fixtureText, jsonResponse, textResponse, stubFetch } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("mistral provider", () => {
  it("supports complete through the mistral preset", async () => {
    stubFetch(async () => jsonResponse(await fixtureJson("../fixtures/mistral/complete.json")));

    const provider = mistral("mistral-small-latest");
    const result = await provider.complete({ messages: [] });
    expect(result.finishReason).toBe("tool_use");
    expect(result.message.content.some((block) => block.type === "tool_call")).toBe(true);
  });

  it("sends temperature 0 explicitly", async () => {
    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/mistral/complete.json"));
    });

    const provider = mistral("mistral-small-latest", { temperature: 0.8 });
    await provider.complete({ messages: [], temperature: 0 });

    expect(capturedBody.temperature).toBe(0);
  });

  it("maps responseFormat to json_schema response_format", async () => {
    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/mistral/complete.json"));
    });

    const schema = { type: "object", properties: { title: { type: "string" } }, required: ["title"] };
    const provider = mistral("mistral-small-latest");
    await provider.complete({
      messages: [],
      responseFormat: { type: "json_schema", name: "card", schema },
    });

    expect(capturedBody.response_format).toEqual({
      type: "json_schema",
      json_schema: {
        name: "card",
        schema,
        strict: true,
      },
    });
  });
  const streamFixture = async (fixture: string) => {
    stubFetch(async () => textResponse(await fixtureText(fixture), "text/event-stream"));
    const events = [];
    for await (const event of mistral("magistral-medium-latest").stream({ messages: [] })) events.push(event);
    return events;
  };

  it("maps Magistral content chunks to thinking and text blocks", async () => {
    const events = await streamFixture("../fixtures/mistral/magistral-stream.sse");

    expect(events.flatMap((event) => event.type === "block_end" ? [event.block] : [])).toEqual([
      { type: "thinking", thinking: "let me think" },
      { type: "text", text: "answer" },
    ]);
    expect(events.at(-1)).toMatchObject({ type: "usage", finishReason: "stop" });
  });

  it("reports a stream that ends without finish reason or [DONE] as a provider error", async () => {
    const events = await streamFixture("../fixtures/mistral/truncated.sse");

    expect(events.at(-1)).toMatchObject({ type: "issue", issue: { kind: "provider_error", retryable: true } });
  });
});
