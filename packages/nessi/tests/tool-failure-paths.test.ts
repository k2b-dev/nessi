import { describe, expect, it } from "bun:test";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import { defineTool } from "../src/tools.js";
import type { OutboundEvent, Tool } from "../src/types.js";
import { mockProviderMultiTurn } from "./mock-provider.js";

const provider = () => mockProviderMultiTurn((_request, index) => index === 0
  ? [
    { type: "tool_call", callId: "c1", name: "work", args: {} },
    { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "tool_use" },
  ]
  : [{ type: "text", delta: "Done." }, { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" }]);

/** Runs one tool call to its failure and checks the shared failure contract. */
const expectTimedOutCall = async (tool: Tool) => {
  const store = memoryStore();
  const events: OutboundEvent[] = [];
  for await (const event of nessi({ provider: provider(), store, systemPrompt: "s", input: "go", tools: [tool], maxTurns: 2 })) {
    events.push(event);
  }
  const issueIndex = events.findIndex((event) => event.type === "issue" && event.issue.kind === "timeout");
  const endIndex = events.findIndex((event) => event.type === "tool_execution_end");
  expect(issueIndex).toBeGreaterThan(-1);
  expect(endIndex).toBeGreaterThan(issueIndex);
  const end = events[endIndex];
  expect(end).toMatchObject({ isError: true, result: 'Tool "work" timed out after 20ms.' });
  const stored = (await store.load()).find(({ message }) => message.role === "tool_result")?.message;
  expect(stored).toMatchObject({ isError: true, result: 'Tool "work" timed out after 20ms.', reason: "timeout" });
  const loopEnd = events.at(-1);
  expect(loopEnd?.type === "loop_end" ? loopEnd.aggregate.turns[0]?.toolCalls[0] : undefined)
    .toMatchObject({ isError: true, result: 'Tool "work" timed out after 20ms.' });
  return events;
};

describe("tool failure paths", () => {
  it("times out an unanswered approval without running the tool", async () => {
    let ran = false;
    const tool = defineTool({ name: "work", description: "Work", inputSchema: z.object({}), needsApproval: true, timeoutMs: 20 })
      .server(async () => { ran = true; return "ok"; });
    const events = await expectTimedOutCall(tool);
    expect(ran).toBe(false);
    expect(events.some((event) => event.type === "tool_action_request" && event.kind === "approval")).toBe(true);
  });

  it("times out a server tool that never settles", async () => {
    const tool = defineTool({ name: "work", description: "Work", inputSchema: z.object({}), timeoutMs: 20 })
      .server(() => new Promise<never>(() => {}));
    await expectTimedOutCall(tool);
  });

  it("times out an unanswered nested client tool request", async () => {
    const tool = defineTool({ name: "work", description: "Work", inputSchema: z.object({}), timeoutMs: 20 })
      .server(async (_input, ctx) => ctx.requestClientTool("pick", {}));
    const events = await expectTimedOutCall(tool);
    expect(events.some((event) => event.type === "tool_action_request" && event.kind === "client_tool")).toBe(true);
  });

  it("marks each error result nessi writes with a fixed reason", async () => {
    const cases: Array<{ tool?: Tool; reason: string }> = [
      { reason: "unknown_tool" },
      { tool: defineTool({ name: "work", description: "Work", inputSchema: z.object({ path: z.string() }) }).server(async () => "ok"), reason: "input_validation_failed" },
      { tool: defineTool({ name: "work", description: "Work", inputSchema: z.object({}) }).server(async () => { throw new Error("disk full"); }), reason: "execution_failed" },
      { tool: defineTool({ name: "work", description: "Work", inputSchema: z.object({}), outputSchema: z.string() }).server(async () => 1 as unknown as string), reason: "output_validation_failed" },
    ];
    for (const { tool, reason } of cases) {
      const store = memoryStore();
      for await (const _ of nessi({ provider: provider(), store, systemPrompt: "s", input: "go", tools: tool ? [tool] : [], maxTurns: 2 }));
      const stored = (await store.load()).find(({ message }) => message.role === "tool_result")?.message;
      expect(stored).toMatchObject({ isError: true, reason });
    }
  });
});
