import { afterEach, describe, expect, it } from "bun:test";
import { openai, openAICompatible, openrouter, vllm } from "../../../src/ai/index.js";
import type { GenerateRequest, Provider } from "../../../src/ai/index.js";
import { stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const completion = { choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }] };
const streamBody = `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "ok" }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`;

/** Captures the JSON body of one complete() and one stream() call. */
const sentBodies = async (provider: Provider, request: Partial<GenerateRequest> = {}) => {
  const bodies: Record<string, unknown>[] = [];
  stubFetch(async (_input, init) => {
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    bodies.push(body);
    return body.stream
      ? textResponse(streamBody, "text/event-stream")
      : new Response(JSON.stringify(completion), { headers: { "Content-Type": "application/json" } });
  });
  await provider.complete({ messages: [], ...request });
  for await (const _event of provider.stream({ messages: [], ...request })) { /* drain */ }
  return bodies;
};

const custom = (options: { reasoningEffort?: string; extraBody?: Record<string, unknown> } = {}) =>
  openAICompatible({ name: "custom", model: "m", baseURL: "https://example.com/v1", ...options });

describe("reasoning effort for OpenAI-compatible providers", () => {
  it("sends nothing by default", async () => {
    for (const body of await sentBodies(custom())) {
      expect(body.reasoning_effort).toBeUndefined();
      expect(body.reasoning).toBeUndefined();
    }
  });

  it("sends the provider default and lets the request override it", async () => {
    for (const body of await sentBodies(custom({ reasoningEffort: "medium" }))) expect(body.reasoning_effort).toBe("medium");
    for (const body of await sentBodies(custom({ reasoningEffort: "medium" }), { reasoningEffort: "none" })) {
      expect(body.reasoning_effort).toBe("none");
    }
  });

  it("passes unknown levels through unchanged", async () => {
    for (const body of await sentBodies(openai("gpt-x", { apiKey: "k" }), { reasoningEffort: "xhigh" })) {
      expect(body.reasoning_effort).toBe("xhigh");
    }
  });

  it("keeps the deprecated disableReasoning mapping", async () => {
    for (const body of await sentBodies(vllm("m", { reasoningEffort: "high" }), { disableReasoning: true })) {
      expect(body.reasoning_effort).toBe("low");
    }
  });

  it("merges extraBody from provider and request over the generated body", async () => {
    const provider = custom({ extraBody: { chat_template_kwargs: { enable_thinking: false }, top_k: 20 } });
    const bodies = await sentBodies(provider, { temperature: 0.2, extraBody: { chat_template_kwargs: { foo: 1 }, temperature: 0.7 } });

    for (const body of bodies) {
      expect(body.chat_template_kwargs).toEqual({ enable_thinking: false, foo: 1 });
      expect(body.top_k).toBe(20);
      expect(body.temperature).toBe(0.7);
      expect(body.model).toBe("m");
    }
  });

  it("uses OpenRouter's reasoning object", async () => {
    for (const body of await sentBodies(openrouter("anthropic/claude", { apiKey: "k", reasoningEffort: "high" }))) {
      expect(body.reasoning).toEqual({ effort: "high" });
      expect(body.reasoning_effort).toBeUndefined();
    }
  });
});

describe("OpenRouter reasoning_details round-trip", () => {
  it("collects streamed reasoning items and sends them back unchanged", async () => {
    const { completeFromStream } = await import("../../../src/ai/index.js");
    const { fixtureText } = await import("../helpers/fixtures.js");
    const bodies: Array<Record<string, unknown>> = [];
    stubFetch(async (_input, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return textResponse(await fixtureText("../fixtures/openai/openrouter-reasoning-details.sse"), "text/event-stream");
    });
    const provider = openrouter("anthropic/claude", { apiKey: "k" });

    const { message } = await completeFromStream(provider, { messages: [] });

    expect(message.provider).toBe("openrouter");
    // Without answer text, the items attach after the tool call so they cannot interrupt it.
    expect(message.content).toEqual([
      { type: "thinking", thinking: "Need weather." },
      { type: "tool_call", id: "call_1", name: "weather", args: { city: "Ulm" } },
      {
        type: "thinking",
        thinking: "",
        details: [
          { type: "reasoning.text", text: "Need weather.", signature: "sig-1", index: 0, format: "anthropic-claude-v1" },
          { type: "reasoning.encrypted", data: "enc-2", index: 1, format: "anthropic-claude-v1" },
        ],
      },
    ]);

    await completeFromStream(provider, {
      messages: [
        { role: "user", content: ["Weather?"] },
        message,
        { role: "tool_result", callId: "call_1", name: "weather", result: "sunny" },
      ],
    });
    const replayed = (bodies[1]!.messages as Array<Record<string, unknown>>)[1]!;
    expect(replayed.reasoning_details).toEqual([
      { type: "reasoning.text", text: "Need weather.", signature: "sig-1", index: 0, format: "anthropic-claude-v1" },
      { type: "reasoning.encrypted", data: "enc-2", index: 1, format: "anthropic-claude-v1" },
    ]);
  });

  it("keeps a tool call valid when reasoning metadata arrives after it started", async () => {
    const { completeFromStream } = await import("../../../src/ai/index.js");
    const { fixtureText } = await import("../helpers/fixtures.js");
    stubFetch(async () => textResponse(await fixtureText("../fixtures/openai/openrouter-late-signature.sse"), "text/event-stream"));
    const events = [];
    for await (const event of openrouter("anthropic/claude", { apiKey: "k" }).stream({ messages: [] })) events.push(event);

    expect(events.some((event) => event.type === "issue")).toBe(false);

    stubFetch(async () => textResponse(await fixtureText("../fixtures/openai/openrouter-late-signature.sse"), "text/event-stream"));
    const { message } = await completeFromStream(openrouter("anthropic/claude", { apiKey: "k" }), { messages: [] });
    const calls = message.content.filter((block) => block.type === "tool_call");
    const details = message.content.flatMap((block) => (block.type === "thinking" ? block.details ?? [] : []));
    expect(calls).toEqual([{ type: "tool_call", id: "call_1", name: "weather", args: { city: "Ulm" } }]);
    expect(details).toEqual([{ type: "reasoning.text", text: "Plan.", signature: "late-sig", index: 0, format: "f1" }]);
  });

  it("does not send reasoning items to another provider", async () => {
    const bodies = await sentBodies(openai("gpt-x", { apiKey: "k" }), {
      messages: [
        { role: "user", content: ["Hi"] },
        {
          role: "assistant",
          provider: "openrouter",
          content: [{ type: "thinking", thinking: "x", details: [{ type: "reasoning.text", text: "x" }] }, { type: "text", text: "Hello" }],
        },
      ],
    });
    for (const body of bodies) {
      expect((body.messages as Array<Record<string, unknown>>)[1]).toEqual({ role: "assistant", content: "Hello" });
    }
  });

  it("parses reasoning items from non-streaming responses", async () => {
    stubFetch(async () => new Response(JSON.stringify({
      choices: [{
        index: 0,
        message: { role: "assistant", content: "Hi", reasoning_details: [{ type: "reasoning.summary", summary: "Greeting.", index: 0 }] },
        finish_reason: "stop",
      }],
    }), { headers: { "Content-Type": "application/json" } }));

    const { message } = await openrouter("x/y", { apiKey: "k" }).complete({ messages: [] });

    expect(message.content).toEqual([
      { type: "thinking", thinking: "Greeting.", details: [{ type: "reasoning.summary", summary: "Greeting.", index: 0 }] },
      { type: "text", text: "Hi" },
    ]);
  });
});
