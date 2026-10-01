import { describe, it, expect } from "bun:test";
import {
  closeUnansweredToolCalls,
  estimateTokens,
  INTERRUPTED_TOOL_RESULT,
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
  const assistant = (ids: string[]): Message => ({
    role: "assistant",
    content: ids.map((id) => ({ type: "tool_call" as const, id, name: "echo", args: {} })),
  });

  it("adds error results for unanswered calls after the existing results", () => {
    const messages: Message[] = [
      { role: "user", content: [{ type: "text", text: "go" }] },
      assistant(["a", "b"]),
      { role: "tool_result", callId: "a", name: "echo", result: "ok" },
      { role: "user", content: [{ type: "text", text: "next" }] },
    ];

    const closed = closeUnansweredToolCalls(messages);

    expect(closed.map((message) => message.role)).toEqual(["user", "assistant", "tool_result", "tool_result", "user"]);
    expect(closed[3]).toEqual({ role: "tool_result", callId: "b", name: "echo", result: INTERRUPTED_TOOL_RESULT, isError: true });
    expect(messages).toHaveLength(4);
  });

  it("leaves fully answered histories unchanged", () => {
    const messages: Message[] = [
      assistant(["a"]),
      { role: "tool_result", callId: "a", name: "echo", result: "ok" },
    ];
    expect(closeUnansweredToolCalls(messages)).toEqual(messages);
  });
});
