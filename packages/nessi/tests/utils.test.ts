import { describe, it, expect } from "bun:test";
import {
  closeUnansweredToolCalls,
  estimateTokens,
  newMalformedCallId,
  NOT_RETURNED_TOOL_RESULT,
  NOT_RUN_TOOL_RESULT,
  projectHistoricalToolResults,
  zeroUsage,
  toErrorMessage,
  truncateMiddle,
  truncateToolResults,
} from "../src/utils.js";
import type { Message } from "../src/types.js";

describe("estimateTokens", () => {
  it("estimates roughly 1 token per 4 chars", () => {
    const messages: Message[] = [
      { role: "user", content: [{ type: "text", text: "a".repeat(400) }] },
    ];
    const tokens = estimateTokens(messages);
    expect(tokens).toBeGreaterThan(90);
    expect(tokens).toBeLessThan(200);
  });

  it("returns higher estimate for more messages", () => {
    const small: Message[] = [{ role: "user", content: [{ type: "text", text: "hi" }] }];
    const large: Message[] = [
      { role: "user", content: [{ type: "text", text: "hello world ".repeat(100) }] },
      { role: "assistant", content: [{ type: "text", text: "response ".repeat(100) }] },
    ];
    expect(estimateTokens(large)).toBeGreaterThan(estimateTokens(small));
  });

  it("handles empty message list", () => {
    expect(estimateTokens([])).toBe(1);
  });
});

describe("truncateMiddle", () => {
  it("returns short text unchanged", () => {
    expect(truncateMiddle("hello", 100)).toBe("hello");
  });

  it("truncates long text keeping first and last half", () => {
    const text = "a".repeat(50) + "b".repeat(50);
    const result = truncateMiddle(text, 20);
    expect(result).toContain("a".repeat(10));
    expect(result).toContain("b".repeat(10));
    expect(result).toContain("80 characters omitted");
  });

  it("returns text unchanged at exact limit", () => {
    const text = "x".repeat(100);
    expect(truncateMiddle(text, 100)).toBe(text);
  });
});

describe("truncateToolResults", () => {
  it("truncates long string tool results", () => {
    const messages: Message[] = [
      { role: "tool_result", callId: "c1", name: "test", result: "x".repeat(5000) },
    ];
    const result = truncateToolResults(messages, 100);
    const tool = result[0];
    expect(tool.role).toBe("tool_result");
    if (tool.role === "tool_result") {
      expect(typeof tool.result).toBe("string");
      expect((tool.result as string).length).toBeLessThan(200);
      expect((tool.result as string)).toContain("characters omitted");
    }
  });

  it("truncates object tool results via JSON.stringify", () => {
    const messages: Message[] = [
      { role: "tool_result", callId: "c1", name: "test", result: { data: "x".repeat(5000) } },
    ];
    const result = truncateToolResults(messages, 100);
    const tool = result[0];
    if (tool.role === "tool_result") {
      expect(typeof tool.result).toBe("string");
      expect((tool.result as string)).toContain("characters omitted");
    }
  });

  it("does not truncate short tool results", () => {
    const messages: Message[] = [
      { role: "tool_result", callId: "c1", name: "test", result: "short" },
    ];
    const result = truncateToolResults(messages, 100);
    expect(result[0]).toBe(messages[0]); // same reference — untouched
  });

  it("leaves user and assistant messages untouched", () => {
    const messages: Message[] = [
      { role: "user", content: [{ type: "text", text: "x".repeat(5000) }] },
      { role: "assistant", content: [{ type: "text", text: "y".repeat(5000) }] },
    ];
    const result = truncateToolResults(messages, 100);
    expect(result[0]).toBe(messages[0]);
    expect(result[1]).toBe(messages[1]);
  });

  it("preserves thinking blocks in assistant messages", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "answer" },
          { type: "thinking", thinking: "let me think..." },
        ],
      },
    ];
    const result = truncateToolResults(messages, 100);
    expect(result[0]).toBe(messages[0]); // untouched
    if (result[0].role === "assistant") {
      expect(result[0].content).toHaveLength(2);
      expect(result[0].content[1].type).toBe("thinking");
    }
  });
});

describe("projectHistoricalToolResults", () => {
  const fullResult = { output: "full" };
  const historicalResult = { output: "historical" };
  const messages: Message[] = [{
    role: "tool_result",
    callId: "c1",
    name: "shell",
    result: fullResult,
    historicalResult: { originLoopId: "origin", value: historicalResult },
  }];

  it("keeps full results in their originating loop", () => {
    const projected = projectHistoricalToolResults(messages, "origin");
    expect(projected[0]).not.toBe(messages[0]);
    expect(projected[0]?.role === "tool_result" ? projected[0].result : undefined).toBe(fullResult);
    expect(projected[0]?.role === "tool_result" ? projected[0].historicalResult : undefined).toBeUndefined();
  });

  it("projects historical results in later loops without mutating stored messages", () => {
    const projected = projectHistoricalToolResults(messages, "later");
    expect(projected[0]).not.toBe(messages[0]);
    expect(projected[0]?.role === "tool_result" ? projected[0].result : undefined).toEqual(historicalResult);
    expect(projected[0]?.role === "tool_result" ? projected[0].historicalResult : undefined).toBeUndefined();
    expect(messages[0]?.role === "tool_result" ? messages[0].result : undefined).toBe(fullResult);
  });

  it("keeps legacy tool results without a historical representation", () => {
    const legacy: Message[] = [{ role: "tool_result", callId: "c1", name: "shell", result: fullResult }];
    expect(projectHistoricalToolResults(legacy, "later")[0]).toBe(legacy[0]);
  });
});

describe("zeroUsage", () => {
  it("returns zero usage", () => {
    const usage = zeroUsage();
    expect(usage.input).toBe(0);
    expect(usage.output).toBe(0);
    expect(usage.total).toBe(0);
  });
});

describe("toErrorMessage", () => {
  it("extracts message from Error", () => {
    expect(toErrorMessage(new Error("boom"))).toBe("boom");
  });

  it("stringifies non-Error values", () => {
    expect(toErrorMessage("oops")).toBe("oops");
    expect(toErrorMessage(42)).toBe("42");
  });
});

describe("closeUnansweredToolCalls", () => {
  const assistant = (ids: string[], stopReason?: "tool_use" | "error" | "interrupted"): Message => ({
    role: "assistant",
    content: ids.map((id) => ({ type: "tool_call" as const, id, name: "echo", args: {} })),
    ...(stopReason ? { stopReason } : {}),
  });
  const user = (text: string): Message => ({ role: "user", content: [{ type: "text", text }] });

  it("answers an open call as not returned, next to its call", () => {
    const messages: Message[] = [
      user("go"),
      assistant(["a", "b"], "tool_use"),
      { role: "tool_result", callId: "a", name: "echo", result: "ok" },
      user("next"),
    ];

    const closed = closeUnansweredToolCalls(messages);

    expect(closed.map((message) => message.role)).toEqual(["user", "assistant", "tool_result", "tool_result", "user"]);
    expect(closed).toContainEqual({ role: "tool_result", callId: "b", name: "echo", result: NOT_RETURNED_TOOL_RESULT, isError: true, reason: "not_returned" });
    expect(messages).toHaveLength(4);
  });

  for (const stopReason of ["error", "interrupted"] as const) {
    it(`answers the calls of a response that ended with "${stopReason}" as not run`, () => {
      const closed = closeUnansweredToolCalls([user("go"), assistant(["a"], stopReason), user("next")]);
      expect(closed[2]).toEqual({ role: "tool_result", callId: "a", name: "echo", result: NOT_RUN_TOOL_RESULT, isError: true, reason: "not_run" });
    });
  }

  it("answers a stored malformed call as malformed", () => {
    const id = newMalformedCallId();
    const closed = closeUnansweredToolCalls([user("go"), assistant([id], "tool_use")]);
    expect(closed[2]).toMatchObject({ role: "tool_result", callId: id, isError: true, reason: "malformed_tool_call" });
  });

  it("moves a result stored after a later response up to its call", () => {
    const late: Message = { role: "tool_result", callId: "a", name: "echo", result: "late" };
    const messages: Message[] = [user("go"), assistant(["a"], "tool_use"), { role: "assistant", content: [{ type: "text", text: "meanwhile" }] }, late];
    expect(closeUnansweredToolCalls(messages)).toEqual([user("go"), assistant(["a"], "tool_use"), late, messages[2]!]);
  });

  it("does not take a result across a user message or a later call with the same ID", () => {
    const reused: Message[] = [user("go"), assistant(["a"], "tool_use"), user("again"), assistant(["a"], "tool_use"), { role: "tool_result", callId: "a", name: "echo", result: "second" }];
    const closed = closeUnansweredToolCalls(reused);
    expect(closed[2]).toMatchObject({ callId: "a", reason: "not_returned" });
    expect(closed.slice(3)).toEqual(reused.slice(2));
  });

  it("keeps looking for the result of a call whose ID a later response did not reuse", () => {
    const resultB: Message = { role: "tool_result", callId: "b", name: "echo", result: "b" };
    const resultA: Message = { role: "tool_result", callId: "a", name: "echo", result: "a" };
    const closed = closeUnansweredToolCalls([user("go"), assistant(["a", "b"], "tool_use"), assistant(["a"], "tool_use"), resultB, resultA]);
    expect(closed).toEqual([
      user("go"),
      assistant(["a", "b"], "tool_use"),
      expect.objectContaining({ callId: "a", reason: "not_returned" }),
      resultB,
      assistant(["a"], "tool_use"),
      resultA,
    ]);
  });

  it("leaves fully answered histories unchanged", () => {
    const messages: Message[] = [
      assistant(["a"]),
      { role: "tool_result", callId: "a", name: "echo", result: "ok" },
    ];
    expect(closeUnansweredToolCalls(messages)).toBe(messages);
  });
});

describe("options removed in 1.0", () => {
  it("fail instead of silently changing a run", async () => {
    const { nessi } = await import("../src/index.js");
    const { mockProvider } = await import("./mock-provider.js");
    const { memoryStore } = await import("../src/stores.js");
    const { z } = await import("zod");
    const legacy: Record<string, unknown> = { creditStore: { remaining: async () => 1, deduct: async () => {} } };
    expect(() => nessi({ provider: mockProvider([]), store: memoryStore(), systemPrompt: "s", ...legacy }))
      .toThrow('the "creditStore" option was removed');
    const structuredLegacy: Record<string, unknown> = { tools: [] };
    await expect(nessi.structured({ provider: mockProvider([]), input: "x", output: z.object({}), ...structuredLegacy }))
      .rejects.toThrow("use nessi() for tool work");
    // Pure correlation options that no longer do anything stay harmless.
    const tagged: Record<string, unknown> = { agentId: "worker" };
    await expect(nessi.structured({ provider: mockProvider([{ type: "text", delta: "{}" }, { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" }]), input: "x", output: z.object({}), ...tagged }))
      .resolves.toMatchObject({ output: {} });
  });
});
