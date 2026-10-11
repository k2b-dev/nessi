import { describe, expect, it } from "bun:test";
import { z } from "zod";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import { defineTool } from "../src/tools.js";
import type { OutboundEvent, ProviderRequest } from "../src/types.js";
import type { RawStreamEvent } from "../src/ai/types.js";
import { mockProviderMultiTurn } from "./mock-provider.js";

const malformed = (): RawStreamEvent[] => [
  { type: "text", delta: "Now I will echo the text." },
  { type: "tool_start", callId: "bad", name: "echo" },
  { type: "tool_delta", callId: "bad", argsDelta: '{"text":"unfinished' },
  { type: "tool_call", callId: "bad", name: "echo", args: {} },
  { type: "usage", usage: { input: 10, output: 3, total: 13 }, finishReason: "tool_use" },
];

const answer: RawStreamEvent[] = [
  { type: "text", delta: "Done." },
  { type: "usage", usage: { input: 10, output: 3, total: 13 }, finishReason: "stop" },
];

async function collect(loop: ReturnType<typeof nessi>): Promise<OutboundEvent[]> {
  const events: OutboundEvent[] = [];
  for await (const event of loop) events.push(event);
  return events;
}

describe("malformed tool recovery", () => {
  it("lets the model correct its arguments without executing the malformed call", async () => {
    const requests: ProviderRequest[] = [];
    const executed: string[] = [];
    const echo = defineTool({ name: "echo", description: "Echo", inputSchema: z.object({ text: z.string() }) })
      .server(async ({ text }) => { executed.push(text); return text; });
    const provider = mockProviderMultiTurn((request, index) => {
      requests.push(request);
      if (index === 0) return malformed();
      if (index === 1) return [
        { type: "tool_call", callId: "good", name: "echo", args: { text: "smaller" } },
        { type: "usage", usage: { input: 10, output: 3, total: 13 }, finishReason: "tool_use" },
      ];
      return answer;
    });
    const store = memoryStore();
    const events = await collect(nessi({ provider, store, systemPrompt: "test", input: "Echo", tools: [echo] }));

    expect(requests).toHaveLength(3);
    expect(requests[1]?.messages.at(-1)).toMatchObject({ role: "user" });
    const feedback = requests[1]?.messages.at(-1);
    expect(JSON.stringify(feedback)).toContain('echo');
    expect(JSON.stringify(feedback)).toContain("malformed JSON");
    expect(JSON.stringify(feedback)).toContain("smaller");
    expect(JSON.stringify(feedback)).not.toContain("unfinished");
    expect(executed).toEqual(["smaller"]);
    expect(events.filter((event) => event.type === "tool_execution_start")).toHaveLength(1);
    expect(events.some((event) => event.type === "block_start" && event.callId === "bad")).toBe(false);
    expect((await store.load()).some(({ message }) => message.role === "tool_result" && message.callId === "bad")).toBe(false);
    expect(events.filter((event) => event.type === "turn_end")).toHaveLength(3);
    const done = events.find((event) => event.type === "loop_end");
    expect(done?.reason).toBe("stop");
    expect(done?.aggregate).toMatchObject({ assistantMessageCount: 3, toolMalformedCount: 1, toolCallCount: 1, usage: { input: 30, output: 9, total: 39 } });
  });

  it("bounds consecutive malformed retries even without maxTurns", async () => {
    let requests = 0;
    const provider = mockProviderMultiTurn(() => { requests++; return malformed(); });
    const events = await collect(nessi({ provider, store: memoryStore(), systemPrompt: "test", input: "Echo" }));

    expect(requests).toBe(3);
    expect(events.filter((event) => event.type === "turn_start")).toHaveLength(3);
    expect(events.filter((event) => event.type === "turn_end")).toHaveLength(3);
    expect(events.find((event) => event.type === "loop_end")?.reason).toBe("error");
    expect(events.some((event) => event.type === "issue" && event.issue.kind === "runtime_error" && event.issue.message.includes("malformed"))).toBe(true);
  });

  it("counts a malformed response against maxTurns", async () => {
    let requests = 0;
    const provider = mockProviderMultiTurn(() => { requests++; return malformed(); });
    const events = await collect(nessi({ provider, store: memoryStore(), systemPrompt: "test", input: "Echo", maxTurns: 1 }));
    expect(requests).toBe(1);
    expect(events.find((event) => event.type === "loop_end")?.reason).toBe("max_turns");
  });

  it("keeps valid parallel calls and returns their results before the correction", async () => {
    const requests: ProviderRequest[] = [];
    let executed = 0;
    const echo = defineTool({ name: "echo", description: "Echo", inputSchema: z.object({ text: z.string() }) })
      .server(async ({ text }) => { executed++; return text; });
    const provider = mockProviderMultiTurn((request, index) => {
      requests.push(request);
      return index === 0 ? [
        { type: "tool_call", callId: "good", name: "echo", args: { text: "valid" } },
        ...malformed(),
      ] : answer;
    });
    const events = await collect(nessi({ provider, store: memoryStore(), systemPrompt: "test", input: "Echo", tools: [echo] }));
    expect(executed).toBe(1);
    expect(requests[1]?.messages.at(-2)).toMatchObject({ role: "tool_result", callId: "good", result: "valid" });
    expect(requests[1]?.messages.at(-1)?.role).toBe("user");
    expect(events.find((event) => event.type === "loop_end")?.reason).toBe("stop");
  });

  it("resets the recovery budget after a well-formed turn", async () => {
    let requests = 0;
    const echo = defineTool({ name: "echo", description: "Echo", inputSchema: z.object({ text: z.string() }) })
      .server(async ({ text }) => text);
    const provider = mockProviderMultiTurn((_request, index) => {
      requests++;
      if (index === 5) return answer;
      if (index % 2 === 0) return malformed();
      return [{ type: "tool_call", callId: `good-${index}`, name: "echo", args: { text: "ok" } }];
    });
    const events = await collect(nessi({ provider, store: memoryStore(), systemPrompt: "test", input: "Echo", tools: [echo] }));
    expect(requests).toBe(6);
    expect(events.find((event) => event.type === "loop_end")?.reason).toBe("stop");
  });

  it("recovers even if the malformed stream has no usable call id or name", async () => {
    const requests: ProviderRequest[] = [];
    const provider = mockProviderMultiTurn((request, index) => {
      requests.push(request);
      return index === 0 ? [{ type: "tool_error", reason: "missing_tool_name", message: "Missing tool name." }] : answer;
    });
    const events = await collect(nessi({ provider, store: memoryStore(), systemPrompt: "test", input: "Echo" }));
    expect(requests).toHaveLength(2);
    expect(JSON.stringify(requests[1]?.messages.at(-1))).toContain("Missing tool name");
    expect(events.find((event) => event.type === "loop_end")?.reason).toBe("stop");
  });

  for (const stopped of ["error", "cancelled"] as const) {
    it(`does not recover ${stopped} responses`, async () => {
      let requests = 0;
      const store = memoryStore();
      const provider = mockProviderMultiTurn(() => {
        requests++;
        return stopped === "error" ? [
          ...malformed(),
          { type: "usage", usage: { input: 10, output: 3, total: 13 }, finishReason: "error" },
        ] : [
          { type: "tool_start", callId: "pending", name: "echo" },
          { type: "usage", usage: { input: 10, output: 3, total: 13 }, finishReason: "stop" },
        ];
      });
      const events = await collect(nessi({ provider, store, systemPrompt: "test", input: "Echo" }));
      expect(requests).toBe(1);
      expect(events.find((event) => event.type === "loop_end")?.reason).toBe(stopped === "error" ? "error" : "stop");
      expect((await store.load()).filter(({ message }) => message.role === "user")).toHaveLength(1);
    });
  }

  it("does not retry after the caller aborts a malformed stream", async () => {
    const caller = new AbortController();
    const store = memoryStore();
    let requests = 0;
    const provider = mockProviderMultiTurn(() => { requests++; return malformed(); });
    const events: OutboundEvent[] = [];
    for await (const event of nessi({ provider, store, systemPrompt: "test", input: "Echo", signal: caller.signal })) {
      events.push(event);
      if (event.type === "issue" && event.issue.kind === "malformed_tool_call") caller.abort();
    }
    expect(requests).toBe(1);
    expect(events.find((event) => event.type === "loop_end")?.reason).toBe("aborted");
    expect((await store.load()).filter(({ message }) => message.role === "user")).toHaveLength(1);
  });
});
