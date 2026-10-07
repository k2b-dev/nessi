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
