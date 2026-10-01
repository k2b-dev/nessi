import { afterEach, describe, expect, it } from "bun:test";
import { anthropic } from "../../../src/ai/index.js";
import { fixtureJson, fixtureText, jsonResponse, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("anthropic provider", () => {
  it("batches consecutive tool results into a single user message and preserves temperature 0", async () => {
    let capturedBody: any;
    globalThis.fetch = (async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse({
        id: "msg_123",
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
      });
    }) as typeof fetch;

    const provider = anthropic("claude-sonnet", { temperature: 0.7 });
    await provider.complete({
      temperature: 0,
      messages: [
        { role: "assistant", content: [{ type: "tool_call", id: "call-1", name: "search", args: { q: "one" } }] },
        { role: "tool_result", callId: "call-1", name: "search", result: { ok: 1 } },
        { role: "tool_result", callId: "call-2", name: "lookup", result: { ok: 2 } },
      ],
    });

    expect(capturedBody.temperature).toBe(0);
    expect(capturedBody.messages).toEqual([
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "call-1", name: "search", input: { q: "one" } }],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "call-1", content: "{\"ok\":1}" },
          { type: "tool_result", tool_use_id: "call-2", content: "{\"ok\":2}" },
        ],
      },
    ]);
  });

  it("maps complete tool_use blocks", async () => {
    globalThis.fetch = (async () => jsonResponse(await fixtureJson("../fixtures/anthropic/complete.json"))) as typeof fetch;

    const provider = anthropic("claude-sonnet");
    const result = await provider.complete({ messages: [] });
    expect(result.finishReason).toBe("tool_use");
    expect(result.message.content.some((block) => block.type === "tool_call")).toBe(true);
  });

  it("streams text and tool input deltas", async () => {
    globalThis.fetch = (async () =>
      textResponse(await fixtureText("../fixtures/anthropic/stream.sse"), "text/event-stream")) as typeof fetch;

    const provider = anthropic("claude-sonnet");
    const events = [];
    for await (const event of provider.stream({ messages: [] })) events.push(event);

    expect(events.some((event) => event.type === "block_delta")).toBe(true);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(true);
    expect(events.some((event) => event.type === "usage")).toBe(true);
  });

  it("merges streaming usage chunks instead of overwriting prior input tokens", async () => {
    globalThis.fetch = (async () =>
      textResponse(
        [
          "event: message_start",
          'data: {"message":{"usage":{"input_tokens":11}}}',
          "",
          "event: message_delta",
          'data: {"usage":{"output_tokens":7},"delta":{"stop_reason":"end_turn"}}',
          "",
        ].join("\n"),
        "text/event-stream",
      )) as typeof fetch;

    const provider = anthropic("claude-sonnet");
    const events = [];
    for await (const event of provider.stream({ messages: [] })) events.push(event);

    const usageEvent = events.find((event) => event.type === "usage");
    expect(usageEvent && usageEvent.type === "usage" ? usageEvent.finishReason : undefined).toBe("stop");
    expect(usageEvent && usageEvent.type === "usage" ? usageEvent.usage.input : undefined).toBe(11);
    expect(usageEvent && usageEvent.type === "usage" ? usageEvent.usage.output : undefined).toBe(7);
    expect(usageEvent && usageEvent.type === "usage" ? usageEvent.usage.total : undefined).toBe(18);
  });

  it("maps responseFormat to output_config json_schema", async () => {
    let capturedBody: any;
    globalThis.fetch = (async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse({
        id: "msg_123",
        content: [{ type: "text", text: "{\"ok\":true}" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
      });
    }) as typeof fetch;

    const schema = { type: "object", properties: { ok: { type: "boolean" } }, required: ["ok"] };
    const provider = anthropic("claude-sonnet");
    await provider.complete({
      messages: [],
      responseFormat: { type: "json_schema", name: "result", schema },
    });

    expect(capturedBody.output_config).toEqual({
      format: { type: "json_schema", schema },
    });
  });
  it("maps a mid-stream error event to a retryable provider error", async () => {
    globalThis.fetch = (async () =>
      textResponse(await fixtureText("../fixtures/anthropic/stream-error.sse"), "text/event-stream")) as typeof fetch;

    const events = [];
    for await (const event of anthropic("claude-sonnet").stream({ messages: [] })) events.push(event);

    expect(events.at(-1)).toMatchObject({
      type: "issue",
      issue: { kind: "provider_error", retryable: true, message: "anthropic stream error (overloaded_error): Overloaded" },
    });
    expect(events.some((event) => event.type === "usage")).toBe(false);
  });

  it("reports a stream that ends without stop reason or message_stop as a provider error", async () => {
    globalThis.fetch = (async () =>
      textResponse(await fixtureText("../fixtures/anthropic/truncated.sse"), "text/event-stream")) as typeof fetch;

    const events = [];
    for await (const event of anthropic("claude-sonnet").stream({ messages: [] })) events.push(event);

    expect(events.at(-1)).toMatchObject({ type: "issue", issue: { kind: "provider_error", retryable: true } });
  });
});
