import { describe, expect, it } from "bun:test";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import { defineTool } from "../src/tools.js";
import type { OutboundEvent } from "../src/types.js";
import { mockProviderMultiTurn } from "./mock-provider.js";

async function bounded<T>(promise: Promise<T>, ms = 1_000): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => { timeout = setTimeout(() => reject(new Error(`still waiting after ${ms}ms`)), ms); }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

const provider = () => mockProviderMultiTurn((_request, index) => index === 0
  ? [
    { type: "tool_call", callId: "c1", name: "report", args: {} },
    { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
  ]
  : [{ type: "text", delta: "Done." }, { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" }]);

/** A tool that succeeds, with a historical projection that never settles. */
const stuckReport = (timeoutMs?: number) => defineTool({
  name: "report",
  description: "Report",
  inputSchema: z.object({}),
  ...(timeoutMs ? { timeoutMs } : {}),
  toHistoricalResult: () => new Promise<never>(() => {}),
}).server(async () => ({ rows: 3 }));

const run = async (loop: ReturnType<typeof nessi>, onEvent?: (event: OutboundEvent) => void) => {
  const events: OutboundEvent[] = [];
  for await (const event of loop) {
    events.push(event);
    onEvent?.(event);
  }
  return events;
};

describe("historical result projection", () => {
  it("is bounded by the tool deadline and keeps the full result", async () => {
    const store = memoryStore();
    const events = await bounded(run(nessi({ provider: provider(), store, systemPrompt: "test", input: "Report", tools: [stuckReport(20)] })));

    expect(events.find((event) => event.type === "tool_execution_end")).toMatchObject({ result: { rows: 3 } });
    expect(events.find((event) => event.type === "tool_execution_end")).not.toHaveProperty("isError", true);
    expect(events).toContainEqual(expect.objectContaining({
      type: "issue",
      issue: expect.objectContaining({ kind: "tool_historical_result_error", message: expect.stringContaining("deadline") }),
    }));
    const stored = (await store.load()).find(({ message }) => message.role === "tool_result")?.message;
    expect(stored).toMatchObject({ result: { rows: 3 } });
    expect(stored).not.toHaveProperty("historicalResult");
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "stop" });
  });

  it("ends promptly when the loop is aborted during the projection", async () => {
    const store = memoryStore();
    let loop: ReturnType<typeof nessi> | undefined;
    let executions = 0;
    const report = defineTool({
      name: "report",
      description: "Report",
      inputSchema: z.object({}),
      toHistoricalResult: () => {
        loop?.abort();
        return new Promise<never>(() => {});
      },
    }).server(async () => {
      executions++;
      return { rows: 3 };
    });
    loop = nessi({ provider: provider(), store, systemPrompt: "test", input: "Report", tools: [report] });
    const events = await bounded(run(loop));
    expect(executions).toBe(1);
    expect(events).toContainEqual(expect.objectContaining({
      type: "issue",
      issue: expect.objectContaining({ kind: "tool_historical_result_error", message: expect.stringContaining("loop was aborted") }),
    }));
    expect(events.at(-1)).toMatchObject({ type: "loop_end", reason: "aborted" });
    const stored = (await store.load()).find(({ message }) => message.role === "tool_result")?.message;
    expect(stored).toMatchObject({ result: { rows: 3 } });
  });

  it("bounds a client tool projection by the rest of its timeout", async () => {
    const store = memoryStore();
    const report = defineTool({
      name: "report",
      description: "Report",
      inputSchema: z.object({}),
      timeoutMs: 30,
      toHistoricalResult: () => new Promise<never>(() => {}),
    }).client(() => ({ rows: 3 }));
    const loop = nessi({ provider: provider(), store, systemPrompt: "test", input: "Report", tools: [report] });
    const events = await bounded(run(loop, (event) => {
      if (event.type === "tool_action_request") loop.push({ type: "tool_result", callId: event.callId, result: { rows: 3 } });
    }));
    expect(events).toContainEqual(expect.objectContaining({
      type: "issue",
      issue: expect.objectContaining({ kind: "tool_historical_result_error", message: expect.stringContaining("deadline") }),
    }));
    expect((await store.load()).find(({ message }) => message.role === "tool_result")?.message).toMatchObject({ result: { rows: 3 } });
  });

  it("skips the projection when the loop is already aborted", async () => {
    let loop: ReturnType<typeof nessi> | undefined;
    let projected = false;
    const report = defineTool({
      name: "report",
      description: "Report",
      inputSchema: z.object({}),
      toHistoricalResult: () => {
        projected = true;
        throw new Error("must not run");
      },
    }).server(async () => {
      loop?.abort();
      return { rows: 3 };
    });
    loop = nessi({ provider: provider(), store: memoryStore(), systemPrompt: "test", input: "Report", tools: [report] });
    await bounded(run(loop));
    expect(projected).toBe(false);
  });

  it("still stores a projection that finishes in time", async () => {
    const store = memoryStore();
    const report = defineTool({
      name: "report",
      description: "Report",
      inputSchema: z.object({}),
      timeoutMs: 1_000,
      toHistoricalResult: async ({ output }) => ({ rowCount: output.rows }),
    }).server(async () => ({ rows: 3 }));
    await bounded(run(nessi({ provider: provider(), store, systemPrompt: "test", input: "Report", tools: [report] })));
    const stored = (await store.load()).find(({ message }) => message.role === "tool_result")?.message;
    expect(stored).toMatchObject({ result: { rows: 3 }, historicalResult: { value: { rowCount: 3 } } });
  });
});
