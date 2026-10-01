import { describe, it, expect } from "bun:test";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { defineTool } from "../src/tools.js";
import { memoryStore } from "../src/stores.js";
import { mockProvider } from "./mock-provider.js";
import type { OutboundEvent, Provider } from "../src/types.js";

const collect = async (loop: ReturnType<typeof nessi>) => {
  const events: OutboundEvent[] = [];
  for await (const event of loop) events.push(event);
  return events;
};

const toolCallTurn = (name: string, args: unknown = {}) => [
  { type: "tool_start" as const, callId: "c1", name },
  { type: "tool_call" as const, callId: "c1", name, args },
  { type: "usage" as const, usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" as const },
];

describe("nessi loop lifecycle", () => {
  it("aborts promptly while a server tool ignores its signal", async () => {
    const stubborn = defineTool({
      name: "stubborn",
      description: "Never finishes",
      inputSchema: z.object({}),
    }).server(() => new Promise(() => {}));

    const loop = nessi({
      provider: mockProvider(toolCallTurn("stubborn")),
      store: memoryStore(),
      systemPrompt: "sys",
      input: "go",
      tools: [stubborn],
    });

    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_execution_start") setTimeout(() => loop.abort(), 5);
    }

    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "aborted" });
  });

  it("limits nested approval waits to the tool's own timeout", async () => {
    const slow = defineTool({
      name: "slow",
      description: "Asks for approval after some work",
      inputSchema: z.object({}),
      timeoutMs: 60,
    }).server(async (_input, ctx) => {
      await new Promise((resolve) => setTimeout(resolve, 40));
      return ctx.requestApproval("continue?");
    });

    const startedAt = Date.now();
    const events = await collect(nessi({
      provider: mockProvider(toolCallTurn("slow")),
      store: memoryStore(),
      systemPrompt: "sys",
      input: "go",
      tools: [slow],
      maxTurns: 1,
    }));

    const end = events.find((event) => event.type === "tool_execution_end");
    expect(end).toMatchObject({ isError: true, result: 'Tool "slow" timed out after 60ms.' });
    expect(Date.now() - startedAt).toBeLessThan(100);
  });

  it("does not count the previous exchange when starting from a trailing user message", async () => {
    const store = memoryStore();
    const reply = [
      { type: "text" as const, delta: "Hi." },
      { type: "usage" as const, usage: { input: 10, output: 5, total: 15 } },
    ];
    await collect(nessi({ provider: mockProvider(reply), store, systemPrompt: "sys", input: "one" }));
    await store.append({ role: "user", content: [{ type: "text", text: "two" }] });

    const events = await collect(nessi({ provider: mockProvider(reply), store, systemPrompt: "sys" }));
    const end = events.at(-1) as Extract<OutboundEvent, { type: "loop_end" }>;

    expect(end.aggregate.assistantMessageCount).toBe(1);
    expect(end.aggregate.usage).toMatchObject({ input: 10, output: 5, total: 15 });
  });

  it("closes the provider stream when a coalesced consumer stops early", async () => {
    let closed = false;
    const provider: Provider = {
      ...mockProvider([]),
      async *stream() {
        try {
          yield { type: "block_start", blockId: "b1", kind: "text", index: 0 };
          for (let i = 0; i < 100; i++) {
            yield { type: "block_delta", blockId: "b1", kind: "text", delta: "x" };
            await new Promise((resolve) => setTimeout(resolve, 1));
          }
        } finally {
          closed = true;
        }
      },
    };

    const loop = nessi({ provider, store: memoryStore(), systemPrompt: "sys", input: "go", coalesce: { ms: 5 } });
    for await (const event of loop) {
      if (event.type === "block_delta") break;
    }
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(closed).toBe(true);
  });

  it("aborts running tools when the consumer stops iterating", async () => {
    let toolSignal: AbortSignal | undefined;
    const waiting = defineTool({
      name: "waiting",
      description: "Waits for abort",
      inputSchema: z.object({}),
    }).server(async (_input, ctx) => {
      toolSignal = ctx.signal;
      await ctx.requestApproval("continue?");
    });

    const loop = nessi({
      provider: mockProvider(toolCallTurn("waiting")),
      store: memoryStore(),
      systemPrompt: "sys",
      input: "go",
      tools: [waiting],
    });
    for await (const event of loop) {
      if (event.type === "tool_action_request") break;
    }

    expect(toolSignal?.aborted).toBe(true);
  });

  it("continues without compaction when routine compaction fails", async () => {
    const events = await collect(nessi({
      provider: mockProvider([
        { type: "text", delta: "Done." },
        { type: "usage", usage: { input: 1, output: 1, total: 2 } },
      ]),
      store: memoryStore(),
      systemPrompt: "sys",
      input: "go",
      compact: () => Promise.reject(new Error("summary model down")),
    }));

    expect(events).toContainEqual(expect.objectContaining({
      type: "issue",
      issue: expect.objectContaining({ message: "Compaction failed: summary model down" }),
    }));
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });

  it("does not append input when the signal is already aborted", async () => {
    const store = memoryStore();
    const controller = new AbortController();
    controller.abort();

    const events = await collect(nessi({
      provider: mockProvider([]),
      store,
      systemPrompt: "sys",
      input: "go",
      signal: controller.signal,
    }));

    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "aborted" });
    expect(await store.load()).toHaveLength(0);
  });
});
