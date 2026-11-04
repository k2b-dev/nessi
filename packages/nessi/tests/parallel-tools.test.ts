import { describe, expect, it } from "bun:test";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import { defineTool } from "../src/tools.js";
import type { OutboundEvent, SessionStore, Tool, ToolCallBlock, ToolContext } from "../src/types.js";
import type { RawStreamEvent } from "../src/ai/types.js";
import { mockProviderMultiTurn } from "./mock-provider.js";

const done: RawStreamEvent[] = [
  { type: "text", delta: "Done." },
  { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
];

/** A provider that sends `calls` in its first response and then finishes. */
const callsThenDone = (calls: Array<{ id: string; name: string; args?: Record<string, unknown> }>) =>
  mockProviderMultiTurn((_request, index) => index === 0
    ? [
      ...calls.map((call): RawStreamEvent => ({ type: "tool_call", callId: call.id, name: call.name, args: call.args ?? {} })),
      { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
    ]
    : done);

const sleep = (ms: number, signal?: AbortSignal) => new Promise<void>((resolve, reject) => {
  const timer = setTimeout(resolve, ms);
  signal?.addEventListener("abort", () => {
    clearTimeout(timer);
    reject(signal.reason);
  }, { once: true });
});

/** Tools that record when they run; `ms` sets how long the named call takes. */
const tracker = () => {
  let active = 0;
  const log: string[] = [];
  const state = { maxActive: 0, log };
  const tool = (name: string, options: { concurrency?: "parallel" | "serial"; needsApproval?: boolean; timeoutMs?: number } = {}) =>
    defineTool({ name, description: name, inputSchema: z.object({ ms: z.number().default(10), id: z.string() }), ...options })
      .server(async ({ ms, id }, ctx: ToolContext) => {
        active++;
        state.maxActive = Math.max(state.maxActive, active);
        log.push(`start ${id}`);
        try {
          await sleep(ms, ctx.signal);
          return id;
        } finally {
          active--;
          log.push(`end ${id}`);
        }
      });
  return { state, tool };
};

const collect = async (loop: ReturnType<typeof nessi>) => {
  const events: OutboundEvent[] = [];
  for await (const event of loop) events.push(event);
  return events;
};

const storedResults = async (store: SessionStore) =>
  (await store.load()).flatMap(({ message }) => (message.role === "tool_result" ? [message] : []));

describe("parallel tool calls", () => {
  it("runs neighbouring parallel calls together and stores their results in call order", async () => {
    const { state, tool } = tracker();
    const read = tool("read", { concurrency: "parallel" });
    const store = memoryStore();
    const events = await collect(nessi({
      provider: callsThenDone([
        { id: "a", name: "read", args: { id: "a", ms: 40 } },
        { id: "b", name: "read", args: { id: "b", ms: 5 } },
        { id: "c", name: "read", args: { id: "c", ms: 20 } },
      ]),
      store,
      systemPrompt: "s",
      input: "go",
      tools: [read],
    }));

    expect(state.maxActive).toBe(3);
    const stored = (await store.load()).find(({ message }) => message.role === "assistant")?.message;
    expect(stored?.role === "assistant" ? stored.content.map((block) => block.type === "tool_call" && block.parallel) : []).toEqual([true, true, true]);
    // b finishes first, yet the history follows the call order.
    expect(state.log.indexOf("end b")).toBeLessThan(state.log.indexOf("end a"));
    expect((await storedResults(store)).map((result) => [result.callId, result.result])).toEqual([["a", "a"], ["b", "b"], ["c", "c"]]);
    // Every end event follows its stored result, so it also comes in call order here.
    const ends = events.filter((event) => event.type === "tool_execution_end");
    expect(ends.map((event) => event.type === "tool_execution_end" && event.callId)).toEqual(["a", "b", "c"]);
    expect(events.filter((event) => event.type === "tool_execution_start")).toHaveLength(3);
    const end = events.at(-1);
    expect(end?.type === "loop_end" ? end.aggregate.toolCallCount : undefined).toBe(3);
    expect(end?.type === "loop_end" ? end.aggregate.usage : undefined).toMatchObject({ input: 2, output: 2, total: 4 });
  });

  it("runs serial tools one at a time by default", async () => {
    const { state, tool } = tracker();
    const write = tool("write");
    await collect(nessi({
      provider: callsThenDone([{ id: "a", name: "write", args: { id: "a" } }, { id: "b", name: "write", args: { id: "b" } }]),
      store: memoryStore(),
      systemPrompt: "s",
      input: "go",
      tools: [write],
    }));
    expect(state.maxActive).toBe(1);
    expect(state.log).toEqual(["start a", "end a", "start b", "end b"]);
  });

  it("keeps a serial call between parallel groups", async () => {
    const { state, tool } = tracker();
    const store = memoryStore();
    await collect(nessi({
      provider: callsThenDone([
        { id: "p1", name: "read", args: { id: "p1" } },
        { id: "p2", name: "read", args: { id: "p2" } },
        { id: "s3", name: "write", args: { id: "s3" } },
        { id: "p4", name: "read", args: { id: "p4" } },
        { id: "p5", name: "read", args: { id: "p5" } },
      ]),
      store,
      systemPrompt: "s",
      input: "go",
      tools: [tool("read", { concurrency: "parallel" }), tool("write")],
    }));
    const at = (entry: string) => state.log.indexOf(entry);
    expect(at("start s3")).toBeGreaterThan(Math.max(at("end p1"), at("end p2")));
    expect(Math.min(at("start p4"), at("start p5"))).toBeGreaterThan(at("end s3"));
    expect(at("start p2")).toBeLessThan(at("end p1"));
    expect((await storedResults(store)).map((result) => result.callId)).toEqual(["p1", "p2", "s3", "p4", "p5"]);
  });

  it("runs at most maxParallelToolCalls calls at once", async () => {
    const { state, tool } = tracker();
    const calls = ["a", "b", "c", "d", "e"].map((id) => ({ id, name: "read", args: { id } }));
    const store = memoryStore();
    await collect(nessi({
      provider: callsThenDone(calls),
      store,
      systemPrompt: "s",
      input: "go",
      tools: [tool("read", { concurrency: "parallel" })],
      maxParallelToolCalls: 2,
    }));
    expect(state.maxActive).toBe(2);
    expect((await storedResults(store)).map((result) => result.callId)).toEqual(["a", "b", "c", "d", "e"]);
  });

  it("rejects a limit that is not a positive integer", () => {
    for (const maxParallelToolCalls of [0, -1, 1.5, Number.NaN]) {
      expect(() => nessi({ provider: callsThenDone([]), store: memoryStore(), systemPrompt: "s", maxParallelToolCalls }))
        .toThrow("maxParallelToolCalls");
    }
  });

  it("aborts every running call and stores the results that finished, in order", async () => {
    const signals: AbortSignal[] = [];
    const read = defineTool({ name: "read", description: "Read", inputSchema: z.object({ ms: z.number() }), concurrency: "parallel" })
      .server(async ({ ms }, ctx) => {
        signals.push(ctx.signal);
        await sleep(ms, ctx.signal);
        return "ok";
      });
    const store = memoryStore();
    const loop = nessi({
      provider: callsThenDone([
        { id: "quick", name: "read", args: { ms: 1 } },
        { id: "slow-1", name: "read", args: { ms: 10_000 } },
        { id: "slow-2", name: "read", args: { ms: 10_000 } },
      ]),
      store,
      systemPrompt: "s",
      input: "go",
      tools: [read],
    });
    const events: OutboundEvent[] = [];
    const started = Date.now();
    for await (const event of loop) {
      events.push(event);
      if (event.type === "tool_execution_end" && event.callId === "quick") loop.abort();
    }
    expect(Date.now() - started).toBeLessThan(2_000);
    expect(signals).toHaveLength(3);
    expect(signals.every((each) => each.aborted)).toBe(true);
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "aborted" });
    expect((await storedResults(store)).map((result) => result.callId)).toEqual(["quick"]);
  });

  it("keeps the timeout of each call", async () => {
    const read = defineTool({ name: "read", description: "Read", inputSchema: z.object({ hang: z.boolean() }), concurrency: "parallel", timeoutMs: 30 })
      .server(async ({ hang }) => (hang ? new Promise<never>(() => {}) : "ok"));
    const store = memoryStore();
    await collect(nessi({
      provider: callsThenDone([{ id: "stuck", name: "read", args: { hang: true } }, { id: "fine", name: "read", args: { hang: false } }]),
      store,
      systemPrompt: "s",
      input: "go",
      tools: [read],
    }));
    expect((await storedResults(store)).map((result) => [result.callId, result.reason])).toEqual([["stuck", "timeout"], ["fine", undefined]]);
  });

  it("runs a tool that needs approval alone, even when it is marked parallel", async () => {
    const { state, tool } = tracker();
    const guarded = tool("guarded", { concurrency: "parallel", needsApproval: true });
    const loop = nessi({
      provider: callsThenDone([
        { id: "r1", name: "read", args: { id: "r1" } },
        { id: "g", name: "guarded", args: { id: "g" } },
        { id: "r2", name: "read", args: { id: "r2" } },
      ]),
      store: memoryStore(),
      systemPrompt: "s",
      input: "go",
      tools: [tool("read", { concurrency: "parallel" }), guarded],
    });
    for await (const event of loop) {
      if (event.type === "tool_action_request") loop.push({ type: "approval_response", callId: event.callId, approved: true });
    }
    expect(state.maxActive).toBe(1);
    expect(state.log).toEqual(["start r1", "end r1", "start g", "end g", "start r2", "end r2"]);
  });

  it("asks one nested question at a time", async () => {
    const asked: string[] = [];
    const confirm = defineTool({ name: "confirm", description: "Confirm", inputSchema: z.object({}), concurrency: "parallel" })
      .server(async (_input, ctx) => ((await ctx.requestApproval(`ok ${ctx.callId}?`)) ? "yes" : "no"));
    const loop = nessi({
      provider: callsThenDone([{ id: "a", name: "confirm" }, { id: "b", name: "confirm" }]),
      store: memoryStore(),
      systemPrompt: "s",
      input: "go",
      tools: [confirm],
    });
    let open = 0;
    let maxOpen = 0;
    const answers: Promise<void>[] = [];
    for await (const event of loop) {
      if (event.type !== "tool_action_request") continue;
      asked.push(event.callId);
      maxOpen = Math.max(maxOpen, ++open);
      // Answer later, so a second question could show up while this one is open.
      answers.push(sleep(20).then(() => {
        open--;
        loop.push({ type: "approval_response", callId: event.callId, approved: true });
      }));
    }
    await Promise.all(answers);
    expect(asked).toEqual(["a-approval-0", "b-approval-0"]);
    expect(maxOpen).toBe(1);
  });

  it("still repairs malformed calls next to parallel ones", async () => {
    const { tool } = tracker();
    const store = memoryStore();
    const provider = mockProviderMultiTurn((_request, index) => index === 0
      ? [
        { type: "tool_call", callId: "a", name: "read", args: { id: "a" } },
        { type: "tool_call", callId: "b", name: "read", args: { id: "b" } },
        { type: "tool_start", callId: "bad", name: "read" },
        { type: "tool_delta", callId: "bad", argsDelta: '{"id":' },
        { type: "tool_call", callId: "bad", name: "read", args: {} },
        { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
      ]
      : done);
    const events = await collect(nessi({ provider, store, systemPrompt: "s", input: "go", tools: [tool("read", { concurrency: "parallel" })] }));
    const results = await storedResults(store);
    expect(results.map((result) => result.reason ?? result.callId)).toEqual(["malformed_tool_call", "a", "b"]);
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });

  it("does not run again any call of the parallel group that was running when the run stopped", async () => {
    const ran: string[] = [];
    const read = defineTool({ name: "read", description: "Read", inputSchema: z.object({ id: z.string() }), concurrency: "parallel" })
      .server(async ({ id }) => { ran.push(id); return id; });
    const write = defineTool({ name: "write", description: "Write", inputSchema: z.object({ id: z.string() }) })
      .server(async ({ id }) => { ran.push(id); return id; });
    const store = memoryStore();
    await store.append({ role: "user", content: [{ type: "text", text: "go" }] });
    await store.append({
      role: "assistant",
      stopReason: "tool_use",
      content: [
        ...["a", "b", "c"].map((id): ToolCallBlock => ({ type: "tool_call", id, name: "read", args: { id }, parallel: true })),
        { type: "tool_call", id: "w", name: "write", args: { id: "w" } },
      ],
    });
    await store.append({ role: "tool_result", callId: "a", name: "read", result: "a" });
    // The group comes from the stored marks, so it holds even after read became serial.
    const tools: Tool[] = [{ ...read, def: { ...read.def, concurrency: "serial" } }, write];
    await collect(nessi({ provider: mockProviderMultiTurn(() => done), store, systemPrompt: "s", tools }));
    // b and c ran with a; w waited for the group and never started.
    expect(ran).toEqual(["w"]);
    expect((await storedResults(store)).map((result) => [result.callId, result.reason])).toEqual([
      ["a", undefined], ["b", "not_returned"], ["c", "not_returned"], ["w", undefined],
    ]);
  });

  it("ends the group's other calls when storing a result fails, instead of waiting for them", async () => {
    const base = memoryStore();
    const store: SessionStore = {
      load: () => base.load(),
      append: async (message, opts) => {
        if (message.role === "tool_result" && message.callId === "a") throw new Error("disk full");
        await base.append(message, opts);
      },
    };
    const quick = defineTool({ name: "quick", description: "Quick", inputSchema: z.object({}), concurrency: "parallel" }).server(async () => "ok");
    // Waits for an answer that never comes and has no timeout.
    const ask = defineTool({ name: "ask", description: "Ask", inputSchema: z.object({}), concurrency: "parallel", timeoutMs: false })
      .server(async (_input, ctx) => ctx.requestApproval("ok?"));
    const started = Date.now();
    const events = await collect(nessi({
      provider: callsThenDone([{ id: "a", name: "quick" }, { id: "b", name: "ask" }]),
      store,
      systemPrompt: "s",
      input: "go",
      tools: [quick, ask],
    }));
    expect(Date.now() - started).toBeLessThan(2_000);
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "error" });
  });

  it("does not change the history after the consumer stops early", async () => {
    const ask = defineTool({ name: "ask", description: "Ask", inputSchema: z.object({}), concurrency: "parallel" })
      .server(async (_input, ctx) => ctx.requestApproval("ok?"));
    const quick = defineTool({ name: "quick", description: "Quick", inputSchema: z.object({}), concurrency: "parallel" }).server(async () => "ok");
    const store = memoryStore();
    const loop = nessi({
      provider: callsThenDone([{ id: "a", name: "ask" }, { id: "b", name: "quick" }]),
      store,
      systemPrompt: "s",
      input: "go",
      tools: [ask, quick],
    });
    for await (const event of loop) if (event.type === "tool_action_request") break;
    await sleep(30);
    // b finished but waited behind a; it stays without a result, which a resume answers as not_returned.
    expect(await storedResults(store)).toEqual([]);
  });

  it("keeps call order when an earlier call fails before storing anything", async () => {
    const store = memoryStore();
    const read = defineTool({
      name: "read",
      description: "Read",
      inputSchema: z.object({ ms: z.number() }),
      concurrency: "parallel",
      outputSchema: z.string(),
    }).server(async ({ ms }, ctx) => {
      await sleep(ms, ctx.signal);
      return ms === 5 ? (1 as unknown as string) : "ok";
    });
    await collect(nessi({
      provider: callsThenDone([
        { id: "a", name: "read", args: { ms: 40 } },
        { id: "b", name: "read", args: { ms: 5 } },
        { id: "c", name: "read", args: { ms: 1 } },
      ]),
      store,
      systemPrompt: "s",
      input: "go",
      tools: [read],
    }));
    expect((await storedResults(store)).map((result) => [result.callId, result.reason])).toEqual([
      ["a", undefined], ["b", "output_validation_failed"], ["c", undefined],
    ]);
  });
});
