import { afterEach, describe, expect, it } from "bun:test";
import { completeFromStream, openAICompatible, openrouter, vllm } from "../../../src/ai/index.js";
import { expectProviderContract } from "../contracts/provider-contract.js";
import { fixtureJson, fixtureText, jsonResponse, textResponse, stubFetch } from "../helpers/fixtures.js";
import type { OpenAICompat } from "../../../src/ai/types.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("openAICompatible provider", () => {
  const reasoningCases: Array<{
    name: string;
    thinkingFormat?: OpenAICompat["thinkingFormat"];
    delta: Record<string, unknown>;
    expected: string;
  }> = [
    { name: "default reasoning_content", delta: { reasoning_content: "A" }, expected: "A" },
    { name: "text reasoning_content", thinkingFormat: "text", delta: { reasoning_content: "A" }, expected: "A" },
    { name: "text reasoning", thinkingFormat: "text", delta: { reasoning: "A" }, expected: "A" },
    { name: "default reasoning", delta: { reasoning: "A" }, expected: "A" },
    { name: "default details", delta: { reasoning_details: [{ text: "A" }, { summary: "B" }] }, expected: "AB" },
    { name: "details text and summary", thinkingFormat: "reasoning_details", delta: { reasoning_details: [{ text: "A", summary: "duplicate" }, { summary: "B" }] }, expected: "AB" },
    { name: "default deduplication", delta: { reasoning: "A", reasoning_content: "A", reasoning_details: [{ text: "A" }] }, expected: "A" },
    { name: "text precedence", thinkingFormat: "text", delta: { reasoning: "A", reasoning_content: "B", reasoning_details: [{ text: "C" }] }, expected: "A" },
    { name: "details precedence", thinkingFormat: "reasoning_details", delta: { reasoning: "B", reasoning_content: "C", reasoning_details: [{ text: "A" }] }, expected: "A" },
    { name: "default precedence", delta: { reasoning: "B", reasoning_content: "C", reasoning_details: [{ text: "A" }] }, expected: "A" },
    { name: "empty reasoning fallback", thinkingFormat: "text", delta: { reasoning: "", reasoning_content: "A" }, expected: "A" },
    { name: "null reasoning fallback", delta: { reasoning: null, reasoning_content: "A" }, expected: "A" },
    { name: "empty details fallback", thinkingFormat: "reasoning_details", delta: { reasoning_details: [], reasoning_content: "A" }, expected: "A" },
    { name: "non-text details fallback", thinkingFormat: "reasoning_details", delta: { reasoning_details: [{ type: "reasoning.encrypted" }], reasoning: "A" }, expected: "A" },
    { name: "text details fallback", thinkingFormat: "text", delta: { reasoning_details: [{ summary: "A" }] }, expected: "A" },
    { name: "none opt-out", thinkingFormat: "none", delta: { reasoning: "A", reasoning_content: "A", reasoning_details: [{ text: "A" }] }, expected: "" },
    { name: "empty fields", delta: { reasoning: "", reasoning_content: null, reasoning_details: [] }, expected: "" },
  ];

  for (const testCase of reasoningCases) {
    it(`streams thinking with ${testCase.name} before text and tool calls`, async () => {
      const provider = openAICompatible({
        name: "synthetic",
        model: "test",
        baseURL: "https://example.com/v1",
        compat: { thinkingFormat: testCase.thinkingFormat },
      });
      const frames = [
        { delta: testCase.delta, finish_reason: null },
        { delta: testCase.delta, finish_reason: null },
        { delta: { content: "answer" }, finish_reason: null },
        { delta: { tool_calls: [{ index: 0, id: "call_test", function: { name: "lookup", arguments: '{"q":' } }] }, finish_reason: null },
        { delta: { tool_calls: [{ index: 0, function: { arguments: '"test"}' } }] }, finish_reason: "tool_calls" },
      ];
      stubFetch(async () => textResponse(
        frames.map((frame) => `data: ${JSON.stringify({ choices: [{ index: 0, ...frame }] })}\n\n`).join("") + "data: [DONE]\n\n",
        "text/event-stream",
      ));

      const events = [];
      for await (const event of provider.stream({ messages: [] })) events.push(event);
      const textIndex = testCase.expected ? 1 : 0;
      const toolIndex = textIndex + 1;

      expect(events.filter((event) => event.type !== "usage")).toEqual([
        ...(testCase.expected ? [
          { type: "block_start", blockId: "block-0", index: 0, kind: "thinking" },
          { type: "block_delta", blockId: "block-0", delta: testCase.expected },
          { type: "block_delta", blockId: "block-0", delta: testCase.expected },
          { type: "block_end", blockId: "block-0", index: 0, block: { type: "thinking", thinking: testCase.expected.repeat(2) } },
        ] as const : []),
        { type: "block_start", blockId: `block-${textIndex}`, index: textIndex, kind: "text" },
        { type: "block_delta", blockId: `block-${textIndex}`, delta: "answer" },
        { type: "block_end", blockId: `block-${textIndex}`, index: textIndex, block: { type: "text", text: "answer" } },
        { type: "block_start", blockId: `block-${toolIndex}`, index: toolIndex, kind: "tool_call", callId: "call_test", name: "lookup" },
        { type: "block_delta", blockId: `block-${toolIndex}`, delta: '{"q":"test"}' },
        { type: "block_end", blockId: `block-${toolIndex}`, index: toolIndex, block: { type: "tool_call", id: "call_test", name: "lookup", args: { q: "test" } } },
      ]);
      expect(events.at(-1)).toMatchObject({ type: "usage", finishReason: "tool_use" });
    });
  }

  it("emits thinking before later SSE frames arrive and transitions directly to a tool call", async () => {
    const source = Promise.withResolvers<ReadableStreamDefaultController<Uint8Array>>();
    const body = new ReadableStream<Uint8Array>({ start: (controller) => source.resolve(controller) });
    const controller = await source.promise;
    const encoder = new TextEncoder();
    stubFetch(async () => new Response(body, {
      headers: { "Content-Type": "text/event-stream" },
    }));
    const provider = openAICompatible({ name: "synthetic", model: "test", baseURL: "https://example.com/v1" });
    const stream = provider.stream({ messages: [] })[Symbol.asyncIterator]();

    controller.enqueue(encoder.encode('data: {"choices":[{"index":0,"delta":{"reasoning_content":"A"},"finish_reason":null}]}\n\n'));
    expect((await stream.next()).value).toEqual({ type: "block_start", blockId: "block-0", index: 0, kind: "thinking" });
    expect((await stream.next()).value).toEqual({ type: "block_delta", blockId: "block-0", delta: "A" });

    controller.enqueue(encoder.encode('data: {"choices":[{"index":0,"delta":{"reasoning_content":"B","tool_calls":[{"index":0,"id":"call_test","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n'));
    controller.close();
    const remaining = [];
    for (let next = await stream.next(); !next.done; next = await stream.next()) remaining.push(next.value);
    expect(remaining.filter((event) => event.type !== "usage")).toEqual([
      { type: "block_delta", blockId: "block-0", delta: "B" },
      { type: "block_end", blockId: "block-0", index: 0, block: { type: "thinking", thinking: "AB" } },
      { type: "block_start", blockId: "block-1", index: 1, kind: "tool_call", callId: "call_test", name: "lookup" },
      { type: "block_delta", blockId: "block-1", delta: "{}" },
      { type: "block_end", blockId: "block-1", index: 1, block: { type: "tool_call", id: "call_test", name: "lookup", args: {} } },
    ]);
  });

  it("supports complete and stream contract for simple text", async () => {
    const provider = openAICompatible({
      name: "custom",
      model: "gpt-test",
      baseURL: "https://example.com/v1",
      compat: { supportsUsageInStreaming: true, thinkingFormat: "none" },
    });

    let call = 0;
    stubFetch(async () => {
      call++;
      if (call === 1) return jsonResponse(await fixtureJson("../fixtures/openai/complete.json"));
      return textResponse(await fixtureText("../fixtures/openai/stream.sse"), "text/event-stream");
    });

    await expectProviderContract(provider, { messages: [] });
  });

  it("flushes streamed tool calls at stream end and normalizes strict ids", async () => {
    const provider = openAICompatible({
      name: "strict",
      model: "mistral-small-latest",
      baseURL: "https://example.com/v1",
      compat: {
        toolCallIdPolicy: "strict9",
        supportsUsageInStreaming: true,
        thinkingFormat: "none",
      },
    });

    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return textResponse(await fixtureText("../fixtures/openai/strict-tool-stream.sse"), "text/event-stream");
    });

    const messages = [
      { role: "user" as const, content: [{ type: "text" as const, text: "find" }] },
      {
        role: "assistant" as const,
        content: [{ type: "tool_call" as const, id: "call_abc123456789", name: "search", args: { q: "hello" } }],
      },
      { role: "tool_result" as const, callId: "call_abc123456789", name: "search", result: { ok: true } },
    ];

    const events = [];
    for await (const event of provider.stream({ messages })) events.push(event);

    expect(events.find((event) => event.type === "block_end" && event.block.type === "tool_call")).toBeDefined();
    const assistantMessage = capturedBody.messages.find((message: any) => message.role === "assistant");
    expect(/^[A-Za-z0-9]{9}$/.test(assistantMessage.tool_calls[0].id)).toBe(true);
  });

  it("classifies vLLM-style text during partial tool calls as malformed", async () => {
    const provider = openAICompatible({
      name: "vllm",
      model: "qwen-test",
      baseURL: "https://example.com/v1",
      compat: {
        supportsUsageInStreaming: true,
        thinkingFormat: "none",
        maxTokensField: "max_tokens",
      },
    });

    stubFetch(async () =>
      textResponse(await fixtureText("../fixtures/openai/vllm-malformed-tool-text.sse"), "text/event-stream"));

    const events = [];
    for await (const event of provider.stream({ messages: [] })) events.push(event);

    expect(events.some((event) => event.type === "block_start" && event.kind === "tool_call")).toBe(false);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(false);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "text")).toBe(false);

    const issue = events.find((event) => event.type === "issue") as any;
    expect(issue.issue.reason).toBe("text_during_tool_call");
    expect(issue.issue.callId).toBe("call_card");
    expect(issue.issue.name).toBe("card");
    expect(issue.issue.textDelta).toBe("</invoke>");
  });

  it("maps openrouter reasoning details to thinking events", async () => {
    const provider = openrouter("openai/gpt-4.1-mini", { apiKey: "x", baseURL: "https://openrouter.ai/api/v1" });
    stubFetch(async () =>
      textResponse(await fixtureText("../fixtures/openrouter/reasoning.sse"), "text/event-stream"));

    const events = [];
    for await (const event of provider.stream({ messages: [] })) events.push(event);

    expect(events.some((event) => event.type === "block_end" && event.block.type === "thinking")).toBe(true);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "text")).toBe(true);
  });

  it("sends temperature 0 explicitly", async () => {
    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/openai/complete.json"));
    });

    const provider = openAICompatible({
      name: "custom",
      model: "gpt-test",
      baseURL: "https://example.com/v1",
      temperature: 0.7,
      compat: { supportsUsageInStreaming: true, thinkingFormat: "none" },
    });

    await provider.complete({ messages: [], temperature: 0 });
    expect(capturedBody.temperature).toBe(0);
  });

  it("maps responseFormat to OpenAI json_schema response_format", async () => {
    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/openai/complete.json"));
    });

    const provider = openAICompatible({
      name: "custom",
      model: "gpt-test",
      baseURL: "https://example.com/v1",
      compat: { supportsUsageInStreaming: true, thinkingFormat: "none" },
    });

    await provider.complete({
      messages: [],
      responseFormat: {
        type: "json_schema",
        name: "card",
        schema: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
      },
    });

    expect(capturedBody.response_format).toEqual({
      type: "json_schema",
      json_schema: {
        name: "card",
        schema: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
        strict: true,
      },
    });
  });

  it("maps vLLM responseFormat to structured_outputs", async () => {
    let capturedBody: any;
    stubFetch(async (_input, init) => {
      capturedBody = JSON.parse(String(init?.body ?? "{}"));
      return jsonResponse(await fixtureJson("../fixtures/openai/complete.json"));
    });

    const provider = vllm("qwen-test", { apiKey: "x", baseURL: "https://example.com/v1" });
    const schema = { type: "object", properties: { ok: { type: "boolean" } }, required: ["ok"] };
    await provider.complete({
      messages: [],
      responseFormat: { type: "json_schema", name: "result", schema },
    });

    expect(capturedBody.structured_outputs).toEqual({ json: schema });
    expect(capturedBody.response_format).toBeUndefined();
  });

  it("does not advertise native structured output for generic compatible providers by default", () => {
    const provider = openAICompatible({
      name: "custom",
      model: "gpt-test",
      baseURL: "https://example.com/v1",
      compat: { supportsUsageInStreaming: true, thinkingFormat: "none" },
    });

    expect(provider.capabilities.structuredOutput).toBe(false);
  });
  const streamFixture = async (fixture: string) => {
    stubFetch(async () => textResponse(await fixtureText(fixture), "text/event-stream"));
    const provider = openAICompatible({ name: "custom", model: "gpt-test", baseURL: "https://example.com/v1" });
    const events = [];
    for await (const event of provider.stream({ messages: [] })) events.push(event);
    return events;
  };

  it("reports a stream that ends without finish reason or [DONE] as a retryable provider error", async () => {
    const events = await streamFixture("../fixtures/openai/truncated.sse");

    expect(events.at(-1)).toMatchObject({ type: "issue", issue: { kind: "provider_error", retryable: true } });
    expect(events.some((event) => event.type === "usage" && event.finishReason)).toBe(false);

    stubFetch(async () =>
      textResponse(await fixtureText("../fixtures/openai/truncated.sse"), "text/event-stream"));
    const provider = openAICompatible({ name: "custom", model: "gpt-test", baseURL: "https://example.com/v1" });
    await expect(completeFromStream(provider, { messages: [] })).rejects.toThrow("stream ended unexpectedly");
  });

  it("reports an in-stream error chunk with its message", async () => {
    const events = await streamFixture("../fixtures/openai/stream-error.sse");

    expect(events.at(-1)).toMatchObject({
      type: "issue",
      issue: { kind: "provider_error", message: "custom stream error: Upstream provider overloaded", retryable: true },
    });
  });

  it("treats [DONE] without a finish reason as a normal stop", async () => {
    const events = await streamFixture("../fixtures/openai/done-without-finish.sse");

    expect(events.some((event) => event.type === "issue")).toBe(false);
    expect(events.at(-1)).toMatchObject({ type: "usage", finishReason: "stop" });
  });

  it("keeps max_tokens when a later chunk has a null finish reason", async () => {
    const events = await streamFixture("../fixtures/openai/length-then-usage.sse");

    expect(events.at(-1)).toMatchObject({ type: "usage", finishReason: "max_tokens", usage: { input: 3, output: 4 } });
  });

  it("treats a new tool call id on a reused index as a separate tool call", async () => {
    const events = await streamFixture("../fixtures/openai/same-index-parallel-tools.sse");

    expect(events.some((event) => event.type === "issue")).toBe(false);
    expect(events.flatMap((event) => event.type === "block_end" && event.block.type === "tool_call" ? [event.block] : [])).toEqual([
      { type: "tool_call", id: "call_1", name: "search", args: { q: "a" } },
      { type: "tool_call", id: "call_2", name: "lookup", args: { id: 2 } },
    ]);
    expect(events.at(-1)).toMatchObject({ type: "usage", finishReason: "tool_use" });
  });
});
