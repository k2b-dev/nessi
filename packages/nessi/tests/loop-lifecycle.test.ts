import { describe, it, expect } from "bun:test";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { defineTool } from "../src/tools.js";
import { memoryStore } from "../src/stores.js";
import { mockProvider, mockProviderMultiTurn } from "./mock-provider.js";
import type { OutboundEvent, Provider } from "../src/types.js";

const collect = async (loop: ReturnType<typeof nessi>) => {
  const events: OutboundEvent[] = [];
  for await (const event of loop) events.push(event);
  return events;
};

const toolCallTurn = (name: string, args: Record<string, unknown> = {}): Parameters<typeof mockProvider>[0] => [
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

  it("persists the interrupted turn before a coalesced consumer's early exit completes", async () => {
    const store = memoryStore();
    const provider: Provider = {
      ...mockProvider([]),
      async *stream(request) {
        yield { type: "block_start", blockId: "b1", kind: "text", index: 0 };
        for (let i = 0; i < 100; i++) {
          if (request.signal?.aborted) throw new Error("aborted");
          yield { type: "block_delta", blockId: "b1", kind: "text", delta: "x" };
          await new Promise((resolve) => setTimeout(resolve, 1));
        }
      },
    };

    const loop = nessi({ provider, store, systemPrompt: "sys", input: "go", coalesce: { ms: 5 } });
    for await (const event of loop) {
      if (event.type === "block_delta") break;
    }
    await store.append({ role: "user", content: [{ type: "text", text: "next" }] });
    await new Promise((resolve) => setTimeout(resolve, 20));

    const roles = (await store.load()).map((entry) => entry.message.role);
    expect(roles.at(-1)).toBe("user");
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

  it("hands the loop's reasoning settings to compaction", async () => {
    let seen: { reasoningEffort?: string; extraBody?: Record<string, unknown> } = {};
    await collect(nessi({
      provider: mockProvider([
        { type: "text", delta: "Done." },
        { type: "usage", usage: { input: 1, output: 1, total: 2 } },
      ]),
      store: memoryStore(),
      systemPrompt: "sys",
      input: "go",
      reasoningEffort: "low",
      extraBody: { top_k: 3 },
      compact: (ctx) => {
        seen = { reasoningEffort: ctx.reasoningEffort, extraBody: ctx.extraBody };
        return null;
      },
    }));

    expect(seen).toEqual({ reasoningEffort: "low", extraBody: { top_k: 3 } });
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

  it("closes a failed turn, keeps its partial answer and counts its usage", async () => {
    const store = memoryStore();
    let deducted = 0;
    const events = await collect(nessi({
      provider: mockProvider([
        { type: "text", delta: "Half an ans" },
        { type: "usage", usage: { input: 10, output: 4, total: 14, creditsUsed: 2 } },
        { type: "error", error: "connection reset", retryable: true },
      ]),
      store,
      systemPrompt: "sys",
      input: "go",
      creditStore: { remaining: async () => 100, deduct: async (credits) => { deducted += credits; } },
    }));

    expect(events.map((event) => event.type).filter((type) => type.startsWith("turn_") || type === "loop_end"))
      .toEqual(["turn_start", "turn_end", "loop_end"]);
    const turnEnd = events.find((event) => event.type === "turn_end") as Extract<OutboundEvent, { type: "turn_end" }>;
    expect(turnEnd.message).toMatchObject({ stopReason: "error", content: [{ type: "text", text: "Half an ans" }] });
    const end = events.at(-1) as Extract<OutboundEvent, { type: "loop_end" }>;
    expect(end).toMatchObject({ reason: "error", aggregate: { usage: { input: 10, output: 4, total: 14 } } });
    expect(deducted).toBe(2);
    const stored = (await store.load()).map((entry) => entry.message);
    expect(stored.at(-1)).toMatchObject({ role: "assistant", stopReason: "error" });
  });

  it("closes the overflowing attempt before a compaction retry and does not store it", async () => {
    const store = memoryStore();
    const provider = mockProviderMultiTurn((_request, callIndex) => callIndex === 0
      ? [{ type: "error", error: "context too long", retryable: false, contextOverflow: true }]
      : [{ type: "text", delta: "Done." }, { type: "usage", usage: { input: 1, output: 1, total: 2 } }]);

    const events = await collect(nessi({
      provider,
      store,
      systemPrompt: "sys",
      input: "go",
      compact: () => Promise.resolve(),
    }));

    const turnEvents = events.filter((event) => event.type === "turn_start" || event.type === "turn_end");
    expect(turnEvents.map((event) => event.type)).toEqual(["turn_start", "turn_end", "turn_start", "turn_end"]);
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
    const assistants = (await store.load()).filter((entry) => entry.message.role === "assistant");
    expect(assistants).toHaveLength(1);
  });

  it("closes a resumed turn when it is aborted during a pending call", async () => {
    const store = memoryStore();
    await store.append({ role: "user", content: [{ type: "text", text: "go" }] });
    await store.append({
      role: "assistant",
      content: [
        { type: "tool_call", id: "c1", name: "waiting", args: {} },
        { type: "tool_call", id: "c2", name: "waiting", args: {} },
      ],
    });
    const waiting = defineTool({ name: "waiting", description: "Waits", inputSchema: z.object({}) })
      .server(() => new Promise(() => {}));

    const loop = nessi({ provider: mockProvider([]), store, systemPrompt: "sys", tools: [waiting] });
    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_execution_start") setTimeout(() => loop.abort(), 5);
    }

    expect(events.map((event) => event.type).filter((type) => type.startsWith("turn_") || type === "loop_end"))
      .toEqual(["turn_start", "turn_end", "loop_end"]);
    expect(events.at(-1)).toMatchObject({ reason: "aborted" });
  });

  it("does not run tools when the provider stopped the answer with an error", async () => {
    let executed = false;
    const lookup = defineTool({ name: "lookup", description: "Lookup", inputSchema: z.object({}) })
      .server(async () => { executed = true; return "x"; });

    const events = await collect(nessi({
      provider: mockProvider([
        { type: "tool_start", callId: "c1", name: "lookup" },
        { type: "tool_call", callId: "c1", name: "lookup", args: {} },
        { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "error" },
      ]),
      store: memoryStore(),
      systemPrompt: "sys",
      input: "go",
      tools: [lookup],
    }));

    expect(executed).toBe(false);
    expect(events.map((event) => event.type).filter((type) => type === "issue" || type.startsWith("turn_") || type === "loop_end"))
      .toEqual(["turn_start", "issue", "turn_end", "loop_end"]);
    expect(events.at(-1)).toMatchObject({ reason: "error" });
  });

  it("closes a normal turn when it is aborted while a tool runs", async () => {
    const waiting = defineTool({ name: "waiting", description: "Waits", inputSchema: z.object({}) })
      .server(() => new Promise(() => {}));
    const loop = nessi({
      provider: mockProvider(toolCallTurn("waiting")),
      store: memoryStore(),
      systemPrompt: "sys",
      input: "go",
      tools: [waiting],
    });
    const events: OutboundEvent[] = [];
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_execution_start") setTimeout(() => loop.abort(), 5);
    }

    expect(events.map((event) => event.type).filter((type) => type.startsWith("turn_") || type === "loop_end"))
      .toEqual(["turn_start", "turn_end", "loop_end"]);
    expect(events.at(-1)).toMatchObject({ reason: "aborted", aggregate: { assistantMessageCount: 1 } });
  });

  it("still closes a failed turn when storing its partial message fails", async () => {
    const store = memoryStore();
    let appends = 0;
    const failingStore = {
      load: () => store.load(),
      append: async (...args: Parameters<typeof store.append>) => {
        appends++;
        if (appends > 1) throw new Error("disk full");
        await store.append(...args);
      },
    };

    const events = await collect(nessi({
      provider: mockProvider([
        { type: "text", delta: "partial" },
        { type: "usage", usage: { input: 3, output: 2, total: 5 } },
        { type: "error", error: "boom", retryable: true },
      ]),
      store: failingStore,
      systemPrompt: "sys",
      input: "go",
    }));

    expect(events.some((event) => event.type === "issue"
      && event.issue.message.includes("Storing the partial assistant message failed: disk full"))).toBe(true);
    expect(events.map((event) => event.type).filter((type) => type.startsWith("turn_") || type === "loop_end"))
      .toEqual(["turn_start", "turn_end", "loop_end"]);
    expect(events.at(-1)).toMatchObject({ reason: "error", aggregate: { usage: { total: 5 } } });
  });
});
