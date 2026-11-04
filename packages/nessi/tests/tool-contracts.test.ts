import { afterEach, describe, expect, it } from "bun:test";
import { z } from "zod";
import { anthropic, gemini, mistral, ollama, openAICompatible } from "../src/ai/index.js";
import type { Provider, StreamEvent } from "../src/ai/types.js";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import { defineTool } from "../src/tools.js";
import type { OutboundEvent, ProviderRequest, Tool } from "../src/types.js";
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

const sse = (...records: Array<[string | undefined, unknown]>) => records
  .map(([event, data]) => `${event ? `event: ${event}\n` : ""}data: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`)
  .join("");
const ndjson = (...records: unknown[]) => records.map((record) => `${JSON.stringify(record)}\n`).join("");

const chatCompletion = (args: unknown) => ({
  choices: [{
    message: { content: null, tool_calls: [{ id: "call_00001", type: "function", function: { name: "wipe", arguments: args } }] },
    finish_reason: "tool_calls",
  }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
});
const chatStream = (args: unknown) => sse(
  [undefined, { choices: [{ delta: { tool_calls: [{ index: 0, id: "call_00001", function: { name: "wipe", arguments: args } }] } }] }],
  [undefined, { choices: [{ delta: {}, finish_reason: "tool_calls" }] }],
  [undefined, "[DONE]"],
);
const chatAnswer = () => sse(
  [undefined, { choices: [{ delta: { content: "Stopped." } }] }],
  [undefined, { choices: [{ delta: {}, finish_reason: "stop" }] }],
  [undefined, "[DONE]"],
);
const anthropicStream = (block: unknown, stopReason: string) => sse(
  ["message_start", { type: "message_start", message: { usage: { input_tokens: 1, output_tokens: 0 } } }],
  ["content_block_start", { type: "content_block_start", index: 0, content_block: block }],
  ["content_block_stop", { type: "content_block_stop", index: 0 }],
  ["message_delta", { type: "message_delta", delta: { stop_reason: stopReason }, usage: { output_tokens: 1 } }],
  ["message_stop", { type: "message_stop" }],
);

type ArgumentsCase = {
  name: string;
  create: () => Provider;
  /** How this provider's wire format carries valid arguments. */
  valid: unknown;
  native: (args: unknown) => unknown;
  stream: (args: unknown) => Response;
  answer: () => Response;
};

const cases: ArgumentsCase[] = [
  {
    name: "openai-compatible",
    create: () => openAICompatible({ name: "test", model: "test", baseURL: "https://example.test" }),
    valid: '{"target":"one"}',
    native: chatCompletion,
    stream: (args) => textResponse(chatStream(args), "text/event-stream"),
    answer: () => textResponse(chatAnswer(), "text/event-stream"),
  },
  {
    name: "mistral",
    create: () => mistral("test", { apiKey: "test", baseURL: "https://example.test" }),
    valid: '{"target":"one"}',
    native: chatCompletion,
    stream: (args) => textResponse(chatStream(args), "text/event-stream"),
    answer: () => textResponse(chatAnswer(), "text/event-stream"),
  },
  {
    name: "anthropic",
    create: () => anthropic("test", { apiKey: "test", baseURL: "https://example.test" }),
    valid: { target: "one" },
    native: (input) => ({ content: [{ type: "tool_use", id: "toolu_1", name: "wipe", input }], stop_reason: "tool_use", usage: { input_tokens: 1, output_tokens: 1 } }),
    stream: (input) => textResponse(anthropicStream({ type: "tool_use", id: "toolu_1", name: "wipe", input }, "tool_use"), "text/event-stream"),
    answer: () => textResponse(sse(
      ["message_start", { type: "message_start", message: { usage: { input_tokens: 1, output_tokens: 0 } } }],
      ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
      ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Stopped." } }],
      ["content_block_stop", { type: "content_block_stop", index: 0 }],
      ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }],
      ["message_stop", { type: "message_stop" }],
    ), "text/event-stream"),
  },
  {
    name: "gemini",
    create: () => gemini("test", { apiKey: "test", baseURL: "https://example.test" }),
    valid: { target: "one" },
    native: (args) => ({ candidates: [{ content: { parts: [{ functionCall: { name: "wipe", args } }] }, finishReason: "STOP" }] }),
    stream: (args) => textResponse(sse(
      [undefined, { candidates: [{ content: { parts: [{ functionCall: { name: "wipe", args } }] }, finishReason: "STOP" }] }],
    ), "text/event-stream"),
    answer: () => textResponse(sse(
      [undefined, { candidates: [{ content: { parts: [{ text: "Stopped." }] }, finishReason: "STOP" }] }],
    ), "text/event-stream"),
  },
  {
    name: "ollama",
    create: () => ollama("test", { baseURL: "https://example.test" }),
    valid: { target: "one" },
    native: (args) => ({ message: { content: "", tool_calls: [{ function: { name: "wipe", arguments: args } }] }, done: true, done_reason: "stop" }),
    stream: (args) => textResponse(ndjson(
      { message: { content: "", tool_calls: [{ function: { name: "wipe", arguments: args } }] }, done: false },
      { done: true, done_reason: "stop" },
    ), "application/x-ndjson"),
    answer: () => textResponse(ndjson({ message: { content: "Stopped." }, done: false }, { done: true, done_reason: "stop" }), "application/x-ndjson"),
  },
];

/** Supplied arguments that are not a JSON object; none may run a tool with its input defaults. */
const malformedArguments: unknown[] = ['{"target":', "[]", '"all"', 42, [], ["all"]];

describe("tool arguments", () => {
  for (const { name, create, valid, native, stream, answer } of cases) {
    for (const args of malformedArguments) {
      it(`${name} completion rejects ${JSON.stringify(args)} instead of returning {}`, async () => {
        stubFetch(() => jsonResponse(native(args)));
        await expect(create().complete(request)).rejects.toThrow('malformed JSON arguments for "wipe"');
      });

      it(`${name} stream reports ${JSON.stringify(args)} as a malformed call`, async () => {
        stubFetch(() => stream(args));
        const events = await collect<StreamEvent>(create().stream(request));
        expect(events.some((event) => event.type === "block_end" && event.block.type === "tool_call")).toBe(false);
        expect(events.some((event) => event.type === "issue" && event.issue.kind === "malformed_tool_call")).toBe(true);
      });
    }

    it(`${name} keeps valid and absent arguments`, async () => {
      const toolArgs = async (response: () => Response, mode: "complete" | "stream") => {
        stubFetch(response);
        const content = mode === "complete"
          ? (await create().complete(request)).message.content
          : (await collect<StreamEvent>(create().stream(request))).flatMap((event) => event.type === "block_end" ? [event.block] : []);
        return content.flatMap((block) => block.type === "tool_call" ? [block.args] : []);
      };
      expect(await toolArgs(() => jsonResponse(native(valid)), "complete")).toEqual([{ target: "one" }]);
      expect(await toolArgs(() => stream(valid), "stream")).toEqual([{ target: "one" }]);
      expect(await toolArgs(() => jsonResponse(native(undefined)), "complete")).toEqual([{}]);
      expect(await toolArgs(() => stream(undefined), "stream")).toEqual([{}]);
    });

    it(`${name} never runs a tool with its input defaults after malformed arguments`, async () => {
      const executed: unknown[] = [];
      const wipe = defineTool({ name: "wipe", description: "Wipe", inputSchema: z.object({ target: z.string().default("all") }) })
        .server(async (input) => { executed.push(input); return "done"; });
      for (const args of malformedArguments) {
        let turn = 0;
        stubFetch(() => (turn++ === 0 ? stream(args) : answer()));
        await collect<OutboundEvent>(nessi({
          provider: create(),
          store: memoryStore(),
          systemPrompt: "test",
          input: "Wipe one",
          tools: [wipe],
          maxTurns: 3,
        }));
      }
      expect(executed).toEqual([]);

      // Control: the same loop runs the tool once the arguments are valid.
      let turn = 0;
      stubFetch(() => (turn++ === 0 ? stream(valid) : answer()));
      await collect<OutboundEvent>(nessi({ provider: create(), store: memoryStore(), systemPrompt: "test", input: "Wipe one", tools: [wipe], maxTurns: 3 }));
      expect(executed).toEqual([{ target: "one" }]);
    });
  }
});

describe("tool output schemas", () => {
  const output = z.object({ value: z.string().trim(), defaulted: z.number().default(7) });
  const expected = { value: "x", defaulted: 7 };

  const run = async (tool: Tool, clientResult?: unknown) => {
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
    const loop = nessi({ provider, store, systemPrompt: "test", input: "Look up", tools: [tool] });
    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_action_request" && event.kind === "client_tool") {
        loop.push({ type: "tool_result", callId: event.callId, result: clientResult });
      }
    }
    const end = events.find((event) => event.type === "tool_execution_end");
    expect(end?.type === "tool_execution_end" ? end.result : undefined).toEqual(expected);
    const stored = (await store.load()).find(({ message }) => message.role === "tool_result")?.message;
    expect(stored?.role === "tool_result" ? stored.result : undefined).toEqual(expected);
    const sent = JSON.stringify(requests[1]?.messages);
    expect(sent).toContain('"defaulted":7');
    expect(sent).not.toContain("internal");
    expect(sent).not.toContain("  x  ");
  };

  it("server tools pass on the parsed output, also to the historical projection", async () => {
    const projected: unknown[] = [];
    const lookup = defineTool({
      name: "lookup",
      description: "Look up",
      inputSchema: z.object({}),
      outputSchema: output,
      toHistoricalResult: ({ output: value }) => { projected.push(value); return undefined; },
    })
      // Typed server output still carries extra fields and untransformed values at runtime.
      .server(async () => {
        const raw = { value: "  x  ", defaulted: 7, secret: "internal" };
        return raw;
      });
    await run(lookup);
    expect(projected).toEqual([expected]);
  });

  it("client tools pass on the parsed output", async () => {
    const lookup = defineTool({ name: "lookup", description: "Look up", inputSchema: z.object({}), outputSchema: output })
      .client(() => expected);
    await run(lookup, { value: "  x  ", secret: "internal" });
  });
});
