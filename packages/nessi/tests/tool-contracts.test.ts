import { afterEach, describe, expect, it } from "bun:test";
import { z } from "zod";
import { mistral, ollama, openAICompatible } from "../src/ai/index.js";
import type { Provider, StreamEvent } from "../src/ai/types.js";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import { defineTool } from "../src/tools.js";
import type { OutboundEvent, ProviderRequest } from "../src/types.js";
import { jsonResponse, stubFetch, textResponse } from "./ai/helpers/fixtures.js";
import { mockProviderMultiTurn } from "./mock-provider.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const request = { messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "Hi" }] }] };

async function collect<T>(events: AsyncIterable<T>): Promise<T[]> {
  const all: T[] = [];
  for await (const event of events) all.push(event);
  return all;
}

const chatCompletion = (args: unknown) => ({
  choices: [{
    message: { content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "wipe", arguments: args } }] },
    finish_reason: "tool_calls",
  }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
});

const nativeProviders: Array<{ name: string; create: () => Provider; payload: (args: unknown) => unknown }> = [
  {
    name: "openai-compatible",
    create: () => openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }),
    payload: chatCompletion,
  },
  { name: "mistral", create: () => mistral("test", { apiKey: "test", baseURL: "https://example.test" }), payload: chatCompletion },
  {
    name: "ollama",
    create: () => ollama("test", { baseURL: "https://example.test" }),
    payload: (args) => ({ message: { content: "", tool_calls: [{ function: { name: "wipe", arguments: args } }] }, done: true, done_reason: "stop" }),
  },
];

describe("tool arguments", () => {
  for (const { name, create, payload } of nativeProviders) {
    for (const args of ['{"target":', "[]", '"all"', 42]) {
      it(`${name} completion rejects malformed arguments ${JSON.stringify(args)} instead of returning {}`, async () => {
        stubFetch(() => jsonResponse(payload(args)));
        await expect(create().complete(request)).rejects.toThrow('malformed JSON arguments for "wipe"');
      });
    }

    it(`${name} completion keeps valid and absent arguments`, async () => {
      stubFetch(() => jsonResponse(payload(name === "ollama" ? { target: "one" } : '{"target":"one"}')));
      expect((await create().complete(request)).message.content).toContainEqual(expect.objectContaining({ args: { target: "one" } }));
      stubFetch(() => jsonResponse(payload(undefined)));
      expect((await create().complete(request)).message.content).toContainEqual(expect.objectContaining({ args: {} }));
    });
  }

  it("ollama stream reports non-object arguments as a malformed call", async () => {
    stubFetch(() => textResponse(
      `${JSON.stringify({ message: { content: "", tool_calls: [{ function: { name: "wipe", arguments: '{"target":' } }] }, done: false })}\n`
        + `${JSON.stringify({ done: true, done_reason: "stop" })}\n`,
      "application/x-ndjson",
    ));
    const events = await collect<StreamEvent>(ollama("test", { baseURL: "https://example.test" }).stream(request));
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(false);
    expect(events).toContainEqual(expect.objectContaining({
      type: "issue",
      issue: expect.objectContaining({ kind: "malformed_tool_call", reason: "invalid_tool_arguments", name: "wipe" }),
    }));
  });

  it("openai-compatible stream reports invalid argument JSON as a malformed call", async () => {
    const chunk = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;
    stubFetch(() => textResponse(
      chunk({ choices: [{ delta: { tool_calls: [{ index: 0, id: "call_1", function: { name: "wipe", arguments: '{"target":' } }] } }] })
        + chunk({ choices: [{ delta: {}, finish_reason: "tool_calls" }] })
        + "data: [DONE]\n\n",
      "text/event-stream",
    ));
    const events = await collect<StreamEvent>(openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }).stream(request));
    expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(false);
    expect(events.some((event) => event.type === "issue" && event.issue.kind === "malformed_tool_call")).toBe(true);
  });

  it("never runs a tool with its input defaults when the model sent malformed arguments", async () => {
    const executed: unknown[] = [];
    const wipe = defineTool({ name: "wipe", description: "Wipe", inputSchema: z.object({ target: z.string().default("all") }) })
      .server(async (input) => { executed.push(input); return "done"; });
    let turn = 0;
    stubFetch(() => {
      const call = turn++ === 0
        ? { message: { content: "", tool_calls: [{ function: { name: "wipe", arguments: "not json" } }] }, done: false }
        : { message: { content: "Stopped." }, done: false };
      return textResponse(`${JSON.stringify(call)}\n${JSON.stringify({ done: true, done_reason: "stop" })}\n`, "application/x-ndjson");
    });
    await collect<OutboundEvent>(nessi({
      provider: ollama("test", { baseURL: "https://example.test" }),
      store: memoryStore(),
      systemPrompt: "test",
      input: "Wipe one",
      tools: [wipe],
      maxTurns: 3,
    }));
    expect(executed).toEqual([]);
  });
});

describe("tool output schemas", () => {
  const output = z.object({ value: z.string().trim(), defaulted: z.number().default(7) });
  const expected = { value: "x", defaulted: 7 };

  const loopWith = (tool: Parameters<typeof nessi>[0]["tools"]) => {
    const requests: ProviderRequest[] = [];
    const store = memoryStore();
    const provider = mockProviderMultiTurn((providerRequest, index) => {
      requests.push(providerRequest);
      return index === 0
        ? [
          { type: "tool_call", callId: "c1", name: "lookup", args: {} },
          { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
        ]
        : [{ type: "text", delta: "Done." }, { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" }];
    });
    const loop = nessi({ provider, store, systemPrompt: "test", input: "Look up", tools: tool });
    return { loop, requests, store };
  };

  const expectParsedEverywhere = async (
    events: OutboundEvent[],
    requests: ProviderRequest[],
    store: ReturnType<typeof memoryStore>,
  ) => {
    expect(events.find((event) => event.type === "tool_execution_end")).toMatchObject({ result: expected });
    const stored = (await store.load()).find(({ message }) => message.role === "tool_result")?.message;
    expect(stored).toMatchObject({ result: expected });
    const sent = JSON.stringify(requests[1]?.messages);
    expect(sent).toContain('"defaulted":7');
    expect(sent).not.toContain("secret");
    expect(sent).not.toContain("  x  ");
  };

  it("server tools pass on the parsed output", async () => {
    const lookup = defineTool({ name: "lookup", description: "Look up", inputSchema: z.object({}), outputSchema: output })
      // Typed server output still carries extra fields and untransformed values at runtime.
      .server(async () => {
        const raw = { value: "  x  ", defaulted: 7, secret: "internal" };
        return raw;
      });
    const { loop, requests, store } = loopWith([lookup]);
    await expectParsedEverywhere(await collect<OutboundEvent>(loop), requests, store);
  });

  it("client tools pass on the parsed output", async () => {
    const lookup = defineTool({ name: "lookup", description: "Look up", inputSchema: z.object({}), outputSchema: output })
      .client(() => expected);
    const { loop, requests, store } = loopWith([lookup]);
    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "client_tool") {
        loop.push({ type: "tool_result", callId: event.callId, result: { value: "  x  ", secret: "internal" } });
      }
    }
    await expectParsedEverywhere(events, requests, store);
  });
});
