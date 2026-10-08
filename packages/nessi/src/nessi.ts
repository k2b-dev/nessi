// ============================================================================
// nessi - Core Loop
// ============================================================================

import type {
  AssistantContentBlock,
  AssistantMessage,
  CoalesceOptions,
  DoneReason,
  InboundEvent,
  LoopIssueAggregate,
  LoopTimingAggregate,
  LoopToolCallAggregate,
  LoopToolIssueAggregate,
  LoopTurnAggregate,
  Message,
  NessiIssue,
  NessiLoop,
  NessiOptions,
  OutboundEvent,
  StoreEntry,
  Tool,
  ToolCallBlock,
  ToolContext,
  ToolExecutionIssue,
  ToolResultMessage,
  Usage,
  UserMessage,
} from "./types.js";
import { aggregateFromTurns, buildLoopTiming, cloneUsage } from "./aggregates.js";
import { appendAssistantContentBlock, buildAssistantMessageFromContent } from "./ai/shared/messages.js";
import { toolToSpec } from "./tools.js";
import {
  closeUnansweredToolCalls,
  createLoopId,
  projectHistoricalToolResults,
  toErrorMessage,
  truncateToolResults,
  zeroUsage,
} from "./utils.js";

// ----------------------------------------------------------------------------
// Inbound event channel
// ----------------------------------------------------------------------------

type Channel<T> = {
  push(value: T): void;
  pull(signal?: AbortSignal): Promise<T>;
  drain(): T[];
}

type ToolSnapshot = {
  providerTools: ReturnType<typeof toolToSpec>[];
  toolMap: ReadonlyMap<string, Tool>;
}

const createToolSnapshot = (value: unknown): ToolSnapshot => {
  if (!Array.isArray(value)) throw new Error("Tool resolver must return an array");

  const tools = [...value] as Tool[];
  const approvalClientTool = tools.find((tool) => tool.kind === "client" && tool.def.needsApproval);
  if (approvalClientTool) {
    throw new Error(`Tool "${approvalClientTool.def.name}": needsApproval is only supported for server tools.`);
  }
  const names = tools.map((tool) => tool.def.name);
  if (new Set(names).size !== names.length) {
    const duplicate = names.find((name, index) => names.indexOf(name) !== index);
    throw new Error(`Duplicate tool name: ${duplicate}`);
  }

  return {
    providerTools: tools.map(toolToSpec),
    toolMap: new Map(tools.map((tool) => [tool.def.name, tool])),
  };
}

class PullCancelledError extends Error {
  constructor() {
    super("channel pull cancelled");
    this.name = "PullCancelledError";
  }
}

class ToolExecutionFailure extends Error {
  readonly issue: ToolExecutionIssue;

  constructor(issue: ToolExecutionIssue) {
    super(issue.message);
    this.name = "ToolExecutionFailure";
    this.issue = issue;
  }
}

const createChannel = <T>(): Channel<T> => {
  const queue: T[] = [];
  const waiters: Array<{
    resolve(value: T): void;
    reject(error: unknown): void;
    cleanup?: () => void;
  }> = [];

  return {
    push(value: T) {
      const waiter = waiters.shift();
      if (waiter) {
        waiter.cleanup?.();
        waiter.resolve(value);
      }
      else queue.push(value);
    },
    pull(signal?: AbortSignal) {
      const queued = queue.shift();
      if (queued !== undefined) return Promise.resolve(queued);
      if (signal?.aborted) return Promise.reject(new PullCancelledError());
      return new Promise((resolve, reject) => {
        const waiter = { resolve, reject } as {
          resolve(value: T): void;
          reject(error: unknown): void;
          cleanup?: () => void;
        };
        const cancel = () => {
          const index = waiters.indexOf(waiter);
          if (index >= 0) waiters.splice(index, 1);
          waiter.cleanup?.();
          reject(new PullCancelledError());
        };
        if (signal) {
          waiter.cleanup = () => signal.removeEventListener("abort", cancel);
          signal.addEventListener("abort", cancel, { once: true });
        }
        waiters.push(waiter);
      });
    },
    drain() {
      return queue.splice(0, queue.length);
    },
  };
}

// ----------------------------------------------------------------------------
// Input normalization
// ----------------------------------------------------------------------------

const normalizeInput = (input: NonNullable<NessiOptions["input"]>): UserMessage => {
  if (typeof input === "string") return { role: "user", content: [{ type: "text", text: input }] };
  return {
    role: "user",
    content: input.map((part) => (typeof part === "string" ? { type: "text" as const, text: part } : part)),
  };
}

// ----------------------------------------------------------------------------
// Debug and issue helpers
// ----------------------------------------------------------------------------

const formatDebugJson = (value: unknown, maxLength = 2400) => {
  try {
    const text = JSON.stringify(value, null, 2) ?? String(value);
    if (text.length <= maxLength) return text;
    return `${text.slice(0, maxLength)}\n... truncated`;
  } catch {
    return String(value);
  }
}

const formatToolValidationError = (tool: Tool, args: unknown, error: { issues: unknown[] }) => {
  const issues = error.issues.length > 0
    ? error.issues
        .map((rawIssue, index) => {
          const issue = rawIssue as {
            path?: unknown;
            code?: unknown;
            message?: unknown;
            expected?: unknown;
            input?: unknown;
          };
          const path = Array.isArray(issue.path) && issue.path.length > 0 ? issue.path.join(".") : "(root)";
          const code = typeof issue.code === "string" ? issue.code : "unknown";
          const message = typeof issue.message === "string" ? issue.message : "Validation failed";
          const expected = typeof issue.expected === "string" ? `, expected ${issue.expected}` : "";
          const received = Object.prototype.hasOwnProperty.call(issue, "input")
            ? `, received ${formatDebugJson(issue.input, 200).replace(/\s+/g, " ")}`
            : "";
          return `${index + 1}. ${path}: ${message} [${code}${expected}${received}]`;
        })
        .join("\n")
    : "No detailed issues reported.";

  return [
    `Validation error for tool "${tool.def.name}"`,
    "",
    "Issues:",
    issues,
    "",
    "Received args:",
    formatDebugJson(args),
    "",
    "Expected input schema:",
    formatDebugJson(toolToSpec(tool).inputSchema),
  ].join("\n");
}

const createTurnId = (loopId: string, turnIndex: number, suffix = "turn") => `${loopId}:${suffix}:${turnIndex}`;

const isToolStreamIssue = (issue: NessiIssue): issue is LoopToolIssueAggregate =>
  issue.kind === "malformed_tool_call" || issue.kind === "cancelled_tool_call";

const toolExecutionIssue = (
  reason: ToolExecutionIssue["reason"],
  message: string,
  call: { id: string; name: string },
): ToolExecutionIssue => ({
  kind: "tool_execution_error",
  reason,
  message,
  retryable: false,
  callId: call.id,
  name: call.name,
});

const toolTimeoutIssue = (call: { id: string; name: string }, timeoutMs: number): NessiIssue => ({
  kind: "timeout",
  scope: "tool",
  message: `Tool "${call.name}" timed out after ${timeoutMs}ms.`,
  retryable: false,
  callId: call.id,
  name: call.name,
});

const runtimeIssue = (error: unknown): NessiIssue => ({
  kind: "runtime_error",
  message: toErrorMessage(error),
  retryable: false,
});

const issueToToolResult = (issue: NessiIssue) => issue.message;

const timeoutMsFor = (tool: Tool) => {
  const timeoutMs = tool.def.timeoutMs;
  return typeof timeoutMs === "number" && timeoutMs > 0 ? timeoutMs : undefined;
}

const withTimeout = async <T>(
  run: (signal?: AbortSignal) => Promise<T>,
  timeoutMs: number | undefined,
): Promise<{ ok: true; value: T } | { ok: false }> => {
  if (!timeoutMs) return { ok: true, value: await run() };
  const timeoutController = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;
  try {
    timeout = setTimeout(() => {
      timedOut = true;
      timeoutController.abort();
    }, timeoutMs);
    return { ok: true, value: await run(timeoutController.signal) };
  } catch (error) {
    if (timedOut && error instanceof PullCancelledError) return { ok: false };
    throw error;
  } finally {
    if (timeout) clearTimeout(timeout);
    timeoutController.abort();
  }
}

class LoopAbortedError extends Error {
  constructor() {
    super("nessi loop aborted");
    this.name = "LoopAbortedError";
  }
}

const linkedAbortSignal = (signals: Array<AbortSignal | undefined>) => {
  const activeSignals = signals.filter((signal): signal is AbortSignal => Boolean(signal));
  const controller = new AbortController();
  const listeners: Array<() => void> = [];
  for (const signal of activeSignals) {
    const abort = () => controller.abort(signal.reason);
    if (signal.aborted) {
      abort();
      continue;
    }
    signal.addEventListener("abort", abort, { once: true });
    listeners.push(() => signal.removeEventListener("abort", abort));
  }
  return {
    signal: controller.signal,
    cleanup() {
      for (const remove of listeners) remove();
    },
  };
};

// ----------------------------------------------------------------------------
// Timing
// ----------------------------------------------------------------------------

type LoopTimingAccumulator = {
  loopStartedAt?: number;
  generationMs: number;
  toolExecutionMs: number;
  actionWaitMs: number;
};

const nowMs = () => Date.now();

const elapsedSince = (startedAt: number) => Math.max(0, nowMs() - startedAt);

const snapshotTiming = (
  timing: LoopTimingAccumulator,
  usage: Usage | undefined,
): LoopTimingAggregate =>
  buildLoopTiming({
    wallMs: timing.loopStartedAt === undefined ? 0 : elapsedSince(timing.loopStartedAt),
    generationMs: timing.generationMs,
    toolExecutionMs: timing.toolExecutionMs,
    actionWaitMs: timing.actionWaitMs,
  }, usage);

// ----------------------------------------------------------------------------
// Aggregate reconstruction
// ----------------------------------------------------------------------------

const aggregateTurnsFromEntries = (entries: StoreEntry[]): LoopTurnAggregate[] => {
  const messages = entries.filter((entry) => entry.kind === "message").map((entry) => entry.message);
  const lastAssistantIdx = messages.findLastIndex((message) => message.role === "assistant");
  if (lastAssistantIdx < 0) return [];

  let start = 0;
  for (let i = lastAssistantIdx - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") {
      start = i + 1;
      break;
    }
  }

  const turns: LoopTurnAggregate[] = [];
  for (let i = start; i < messages.length; i++) {
    const message = messages[i];
    if (message?.role !== "assistant") continue;

    const toolCalls = message.content
      .filter((block): block is ToolCallBlock => block.type === "tool_call")
      .map((block): LoopToolCallAggregate => ({
        callId: block.id,
        name: block.name,
        args: block.args,
      }));
    const byId = new Map(toolCalls.map((toolCall) => [toolCall.callId, toolCall]));

    for (let j = i + 1; j < messages.length; j++) {
      const next = messages[j];
      if (!next || next.role === "assistant" || next.role === "user") break;
      const toolCall = byId.get(next.callId);
      if (toolCall) {
        toolCall.result = next.result;
        toolCall.isError = next.isError;
      }
    }

    turns.push({
      message,
      usage: cloneUsage(message.usage),
      stopReason: message.stopReason,
      toolCalls,
    });
  }

  return turns;
}

// ----------------------------------------------------------------------------
// Delta coalescing
// ----------------------------------------------------------------------------

type BlockDeltaOutbound = Extract<OutboundEvent, { type: "block_delta" }>;

const canMergeDelta = (left: BlockDeltaOutbound, right: BlockDeltaOutbound) =>
  left.agentId === right.agentId
  && left.loopId === right.loopId
  && left.turnId === right.turnId
  && left.blockId === right.blockId;

const coalesceOutboundEvents = async function* (
  source: AsyncIterable<OutboundEvent>,
  options: CoalesceOptions,
): AsyncGenerator<OutboundEvent> {
  const maxChars = typeof options.maxChars === "number" && options.maxChars > 0 ? options.maxChars : undefined;
  const ms = typeof options.ms === "number" && options.ms > 0 ? options.ms : undefined;
  if (!maxChars && !ms) {
    yield* source;
    return;
  }

  const iterator = source[Symbol.asyncIterator]();
  let next = iterator.next();
  let buffer: BlockDeltaOutbound | undefined;
  let timer: Promise<{ type: "timer"; seq: number }> | undefined;
  let timerSeq = 0;

  const clearTimer = () => {
    timer = undefined;
    timerSeq++;
  };

  const startTimer = () => {
    if (!ms || timer) return;
    const seq = ++timerSeq;
    timer = new Promise((resolve) => setTimeout(() => resolve({ type: "timer", seq }), ms));
  };

  const flush = function* (): Generator<OutboundEvent> {
    if (!buffer) return;
    const event = buffer;
    buffer = undefined;
    clearTimer();
    yield event;
  };

  let sourceDone = false;
  try {
    while (true) {
      const raced = await (timer
        ? Promise.race([
            next.then((result) => ({ type: "event" as const, result })),
            timer,
          ])
        : next.then((result) => ({ type: "event" as const, result })));

      if (raced.type === "timer") {
        if (raced.seq !== timerSeq) continue;
        yield* flush();
        continue;
      }

      const { result } = raced;
      if (result.done) {
        sourceDone = true;
        yield* flush();
        return;
      }
      next = iterator.next();
      const event = result.value;

      if (event.type !== "block_delta") {
        yield* flush();
        yield event;
        continue;
      }

      if (!buffer) {
        buffer = event;
        startTimer();
      } else if (canMergeDelta(buffer, event)) {
        buffer = { ...buffer, delta: buffer.delta + event.delta };
      } else {
        yield* flush();
        buffer = event;
        startTimer();
      }

      if (maxChars && buffer.delta.length >= maxChars) {
        yield* flush();
      }
    }
  } finally {
    // Close the source when the consumer stops early. The loop is aborted first, so a source
    // request still in flight settles promptly; awaiting it keeps any history write it makes
    // (e.g. the interrupted assistant message) ahead of whatever the consumer does next.
    if (!sourceDone) await Promise.resolve(iterator.return?.()).catch(() => {});
  }
}

// ----------------------------------------------------------------------------
// nessi()
// ----------------------------------------------------------------------------

export const nessi = (options: NessiOptions): NessiLoop => {
  const {
    agentId = "main",
    loopId: requestedLoopId,
    input,
    provider,
    systemPrompt,
    tools: toolSource = [],
    store,
    creditStore,
    compact,
    steering,
    maxTurns = Infinity,
    temperature,
    maxOutputTokens,
    disableReasoning,
    reasoningEffort,
    extraBody,
    coalesce,
    maxToolResultChars,
    signal: externalSignal,
  } = options;

  const channel = createChannel<InboundEvent>();
  const deferredInbound: InboundEvent[] = [];
  const steerQueue: string[] = [];
  const subscribers: Array<(event: OutboundEvent) => void> = [];
  const abortController = new AbortController();
  let lastUsage: Usage = zeroUsage();
  const loopTurns: LoopTurnAggregate[] = [];
  const loopIssues: LoopIssueAggregate[] = [];
  const loopId = requestedLoopId?.trim() ? requestedLoopId : createLoopId();
  const timing: LoopTimingAccumulator = {
    generationMs: 0,
    toolExecutionMs: 0,
    actionWaitMs: 0,
  };
  const measureGeneration = async <T>(run: () => Promise<T>): Promise<T> => {
    const startedAt = nowMs();
    try {
      return await run();
    } finally {
      timing.generationMs += elapsedSince(startedAt);
    }
  };
  const waitForActionResponse = async <T>(startedAt: number | undefined, run: () => Promise<T>): Promise<T> => {
    try {
      return await run();
    } finally {
      if (startedAt !== undefined) timing.actionWaitMs += elapsedSince(startedAt);
    }
  };
  const recordToolExecution = (startedAt: number, actionWaitMsAtStart: number) => {
    const elapsedMs = elapsedSince(startedAt);
    const nestedActionWaitMs = Math.max(0, timing.actionWaitMs - actionWaitMsAtStart);
    timing.toolExecutionMs += Math.max(0, elapsedMs - nestedActionWaitMs);
  };
  const snapshotAggregate = () => {
    const aggregate = aggregateFromTurns(loopTurns, loopIssues);
    return { ...aggregate, timing: snapshotTiming(timing, aggregate.usage) };
  };

  const loopEndEvent = (reason: DoneReason): Extract<OutboundEvent, { type: "loop_end" }> => ({
    type: "loop_end",
    agentId,
    loopId,
    reason,
    aggregate: snapshotAggregate(),
  });

  const recordIssue = (
    issue: NessiIssue,
    turn?: { issues: LoopIssueAggregate[]; toolIssues: LoopToolIssueAggregate[] },
  ) => {
    loopIssues.push({ ...issue });
    if (turn) {
      turn.issues.push({ ...issue });
      if (isToolStreamIssue(issue)) turn.toolIssues.push({ ...issue });
    }
  };

  const issueEvent = (
    issue: NessiIssue,
    turn?: { turnId: string; turnIndex: number },
  ): Extract<OutboundEvent, { type: "issue" }> => ({
    type: "issue",
    agentId,
    loopId,
    issue,
    ...(turn ? { turnId: turn.turnId, turnIndex: turn.turnIndex } : {}),
  });

  const recordAssistantTurn = (
    message: AssistantMessage,
    usage: Usage | undefined,
    toolCalls: LoopToolCallAggregate[],
    issues: LoopIssueAggregate[] = [],
    toolIssues: LoopToolIssueAggregate[] = [],
  ) => {
    const turn: LoopTurnAggregate = {
      message,
      usage: cloneUsage(usage),
      stopReason: message.stopReason,
      toolCalls,
      ...(issues.length > 0 ? { issues: issues.map((issue) => ({ ...issue })) } : {}),
      ...(toolIssues.length > 0 ? { toolIssues: toolIssues.map((issue) => ({ ...issue })) } : {}),
    };
    loopTurns.push(turn);
  };

  const hasBufferedInbound = (match: (event: InboundEvent) => boolean): boolean => {
    deferredInbound.push(...channel.drain());
    return deferredInbound.some(match);
  };

  const pullMatching = async <T extends InboundEvent>(
    match: (event: InboundEvent) => event is T,
    localSignal?: AbortSignal,
  ): Promise<T> => {
    while (true) {
      const bufferedIdx = deferredInbound.findIndex(match);
      if (bufferedIdx >= 0) return deferredInbound.splice(bufferedIdx, 1)[0] as T;
      if (abortController.signal.aborted) throw new LoopAbortedError();
      const linked = linkedAbortSignal([abortController.signal, localSignal]);
      let inbound: InboundEvent;
      try {
        inbound = await channel.pull(linked.signal);
      } catch (error) {
        if (error instanceof PullCancelledError && abortController.signal.aborted) {
          throw new LoopAbortedError();
        }
        throw error;
      } finally {
        linked.cleanup();
      }
      if (match(inbound)) return inbound;
      deferredInbound.push(inbound);
    }
  };

  if (externalSignal) {
    if (externalSignal.aborted) abortController.abort();
    else externalSignal.addEventListener("abort", () => abortController.abort(), { once: true });
  }

  const signal = abortController.signal;

  const toolResolver = typeof toolSource === "function" ? toolSource : undefined;
  const staticToolSnapshot = toolResolver ? undefined : createToolSnapshot(toolSource);
  const resolveToolSnapshot = async (): Promise<ToolSnapshot> =>
    staticToolSnapshot ?? createToolSnapshot(await toolResolver!());

  async function* applyPendingSteering(): AsyncGenerator<OutboundEvent, boolean> {
    const pending = steerQueue.splice(0);
    const supplied = await steering?.({ agentId, loopId, signal });
    if (typeof supplied === "string") pending.push(supplied);
    else if (supplied) pending.push(...supplied);
    pending.push(...steerQueue.splice(0));

    let applied = false;
    for (const text of pending) {
      if (!text.trim()) continue;
      const steerMessage: UserMessage = { role: "user", content: [{ type: "text", text }] };
      await store.append(steerMessage);
      applied = true;
      yield { type: "steer_applied", agentId, loopId, message: text };
    }
    return applied;
  }

  const appendToolResult = async (callId: string, name: string, result: unknown, isError = false) => {
    const msg: ToolResultMessage = { role: "tool_result", callId, name, result, isError };
    await store.append(msg);
    return msg;
  };

  const appendSuccessfulToolResult = async (
    tool: Tool,
    call: ToolCallBlock,
    input: unknown,
    output: unknown,
  ): Promise<NessiIssue | undefined> => {
    let historicalResult: ToolResultMessage["historicalResult"];
    try {
      const value = await tool.def.toHistoricalResult?.({ input, output, callId: call.id });
      if (value !== undefined) historicalResult = { originLoopId: loopId, value };
    } catch (error) {
      const issue: NessiIssue = {
        kind: "tool_historical_result_error",
        message: `Historical result projection failed for tool "${call.name}": ${toErrorMessage(error)}`,
        retryable: false,
        callId: call.id,
        name: call.name,
      };
      await store.append({ role: "tool_result", callId: call.id, name: call.name, result: output, isError: false });
      return issue;
    }

    await store.append({
      role: "tool_result",
      callId: call.id,
      name: call.name,
      result: output,
      ...(historicalResult ? { historicalResult } : {}),
      isError: false,
    });
    return undefined;
  };

  type TurnContext = { turnId: string; turnIndex: number };
  type UpdateAggregateToolCall = (callId: string, patch: Partial<LoopToolCallAggregate>) => void;

  async function* failToolCall(
    tc: ToolCallBlock,
    turnCtx: TurnContext,
    updateAggregateToolCall: UpdateAggregateToolCall,
    issue: NessiIssue,
    turnIssues?: { issues: LoopIssueAggregate[]; toolIssues: LoopToolIssueAggregate[] },
  ): AsyncGenerator<OutboundEvent> {
    const result = issueToToolResult(issue);
    await appendToolResult(tc.id, tc.name, result, true);
    updateAggregateToolCall(tc.id, { result, isError: true });
    recordIssue(issue, turnIssues);
    yield issueEvent(issue, turnCtx);
    yield {
      type: "tool_execution_end",
      agentId,
      loopId,
      ...turnCtx,
      callId: tc.id,
      name: tc.name,
      result,
      isError: true,
    };
  }

  async function* executeToolCall(
    tc: ToolCallBlock,
    toolSnapshot: ToolSnapshot,
    turnCtx: TurnContext,
    updateAggregateToolCall: UpdateAggregateToolCall,
    turnIssues?: { issues: LoopIssueAggregate[]; toolIssues: LoopToolIssueAggregate[] },
  ): AsyncGenerator<OutboundEvent> {
    const eventFields = { agentId, loopId, ...turnCtx };
    yield { type: "tool_execution_start", ...eventFields, callId: tc.id, name: tc.name, args: tc.args };

    const tool = toolSnapshot.toolMap.get(tc.name);
    if (!tool) {
      yield* failToolCall(
        tc,
        turnCtx,
        updateAggregateToolCall,
        toolExecutionIssue("unknown_tool", `Unknown tool: ${tc.name}`, tc),
        turnIssues,
      );
      return;
    }

    const inputResult = tool.def.inputSchema.safeParse(tc.args);
    if (!inputResult.success) {
      yield* failToolCall(
        tc,
        turnCtx,
        updateAggregateToolCall,
        toolExecutionIssue("input_validation_failed", formatToolValidationError(tool, tc.args, inputResult.error), tc),
        turnIssues,
      );
      return;
    }

    const validatedInput = inputResult.data;
    updateAggregateToolCall(tc.id, { args: validatedInput });

    const timeoutMs = timeoutMsFor(tool);
    const timeoutIssue = () => toolTimeoutIssue(tc, timeoutMs ?? 0);

    if (tool.kind === "client") {
      const matchesToolResult = (event: InboundEvent): event is Extract<InboundEvent, { type: "tool_result" }> =>
        event.type === "tool_result" && event.callId === tc.id;
      let actionWaitStartedAt: number | undefined;
      if (!hasBufferedInbound(matchesToolResult)) {
        actionWaitStartedAt = nowMs();
        yield {
          type: "tool_action_request",
          ...eventFields,
          kind: "client_tool",
          callId: tc.id,
          name: tc.name,
          args: validatedInput,
        };
      }

      const pulled = await waitForActionResponse(actionWaitStartedAt, () =>
        withTimeout((signal) => pullMatching(matchesToolResult, signal), timeoutMs),
      );
      if (!pulled.ok) {
        const issue = timeoutIssue();
        const result = issueToToolResult(issue);
        await appendToolResult(tc.id, tc.name, result, true);
        updateAggregateToolCall(tc.id, { result, isError: true });
        recordIssue(issue, turnIssues);
        yield issueEvent(issue, turnCtx);
        yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result, isError: true };
        return;
      }

      const output = pulled.value.result;
      if (tool.def.outputSchema) {
        const outputResult = tool.def.outputSchema.safeParse(output);
        if (!outputResult.success) {
          const issue = toolExecutionIssue(
            "output_validation_failed",
            `Output validation error for tool "${tc.name}": ${outputResult.error.message}`,
            tc,
          );
          const result = issueToToolResult(issue);
          await appendToolResult(tc.id, tc.name, result, true);
          updateAggregateToolCall(tc.id, { result, isError: true });
          recordIssue(issue, turnIssues);
          yield issueEvent(issue, turnCtx);
          yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result, isError: true };
          return;
        }
      }

      const historicalIssue = await appendSuccessfulToolResult(tool, tc, validatedInput, output);
      updateAggregateToolCall(tc.id, { result: output });
      if (historicalIssue) {
        recordIssue(historicalIssue, turnIssues);
        yield issueEvent(historicalIssue, turnCtx);
      }
      yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result: output };
      return;
    }

    if (tool.def.needsApproval) {
      const matchesApproval = (event: InboundEvent): event is Extract<InboundEvent, { type: "approval_response" }> =>
        event.type === "approval_response" && event.callId === tc.id;
      let actionWaitStartedAt: number | undefined;
      if (!hasBufferedInbound(matchesApproval)) {
        actionWaitStartedAt = nowMs();
        yield {
          type: "tool_action_request",
          ...eventFields,
          kind: "approval",
          callId: tc.id,
          name: tc.name,
          args: validatedInput,
        };
      }
      const pulled = await waitForActionResponse(actionWaitStartedAt, () =>
        withTimeout((signal) => pullMatching(matchesApproval, signal), timeoutMs),
      );
      if (!pulled.ok) {
        const issue = timeoutIssue();
        const result = issueToToolResult(issue);
        await appendToolResult(tc.id, tc.name, result, true);
        updateAggregateToolCall(tc.id, { result, isError: true });
        recordIssue(issue, turnIssues);
        yield issueEvent(issue, turnCtx);
        yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result, isError: true };
        return;
      }
      if (!pulled.value.approved) {
        const issue = toolExecutionIssue("approval_denied", "User denied this action", tc);
        const result = issueToToolResult(issue);
        await appendToolResult(tc.id, tc.name, result, true);
        updateAggregateToolCall(tc.id, { result, isError: true });
        recordIssue(issue, turnIssues);
        yield issueEvent(issue, turnCtx);
        yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result, isError: true };
        return;
      }
    }

    const toolAbort = new AbortController();
    const abortTool = () => toolAbort.abort();
    if (signal.aborted) abortTool();
    else signal.addEventListener("abort", abortTool, { once: true });
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    let resolveLoopAborted: (() => void) | undefined;
    const loopAborted = new Promise<{ kind: "aborted" }>((resolve) => {
      resolveLoopAborted = () => resolve({ kind: "aborted" });
      if (signal.aborted) resolveLoopAborted();
      else signal.addEventListener("abort", resolveLoopAborted, { once: true });
    });
    // Nested approval/client waits share the tool's deadline: toolAbort fires on timeout or loop abort.
    const pullWithinToolDeadline = async <T extends InboundEvent>(
      match: (event: InboundEvent) => event is T,
    ): Promise<T | undefined> => {
      try {
        return await pullMatching(match, toolAbort.signal);
      } catch (error) {
        if (error instanceof PullCancelledError) return undefined;
        throw error;
      }
    };
    const toolStartedAt = nowMs();
    const actionWaitMsAtToolStart = timing.actionWaitMs;
    let toolExecutionRecorded = false;
    const finishToolExecution = () => {
      if (toolExecutionRecorded) return;
      toolExecutionRecorded = true;
      recordToolExecution(toolStartedAt, actionWaitMsAtToolStart);
    };

    try {
      const approvalQueue: Array<{ id: string; message: string; resolve: (approved: boolean) => void }> = [];
      const clientToolQueue: Array<{
        id: string;
        name: string;
        args: unknown;
        resolve: (result: unknown) => void;
        reject: (error: unknown) => void;
      }> = [];
      let queueNotify: (() => void) | null = null;
      let approvalCounter = 0;
      let clientToolCounter = 0;

      const ctx: ToolContext = {
        callId: tc.id,
        signal: toolAbort.signal,
        requestApproval(message: string) {
          return new Promise((resolve) => {
            const id = `${tc.id}-approval-${approvalCounter++}`;
            approvalQueue.push({ id, message, resolve });
            queueNotify?.();
          });
        },
        requestClientTool<T = unknown>(name: string, args: unknown) {
          return new Promise<T>((resolve, reject) => {
            const id = `${tc.id}-client-${clientToolCounter++}`;
            clientToolQueue.push({
              id,
              name,
              args,
              resolve: resolve as (result: unknown) => void,
              reject,
            });
            queueNotify?.();
          });
        },
      };

      // Settle once and capture rejections so a tool that finishes after a timeout or abort
      // cannot cause an unhandled rejection.
      const toolOutcome = Promise.resolve()
        .then(() => tool.execute(validatedInput, ctx))
        .then(
          (value) => ({ kind: "done" as const, result: value as unknown }),
          (error: unknown) => ({ kind: "failed" as const, error }),
        );
      let timeout: Promise<{ kind: "timeout" }> | undefined;
      if (timeoutMs) {
        timeout = new Promise((resolve) => {
          timeoutHandle = setTimeout(() => {
            toolAbort.abort();
            resolve({ kind: "timeout" });
          }, timeoutMs);
        });
      }

      let result: unknown;
      let done = false;
      while (!done) {
        const waitForQueue = new Promise<{ kind: "queue" }>((resolve) => {
          if (approvalQueue.length > 0 || clientToolQueue.length > 0) resolve({ kind: "queue" });
          else queueNotify = () => resolve({ kind: "queue" });
        });
        const settled = await Promise.race([
          toolOutcome,
          waitForQueue,
          loopAborted,
          ...(timeout ? [timeout] : []),
        ]);

        if (settled.kind === "aborted") throw new LoopAbortedError();
        if (settled.kind === "failed") throw settled.error;

        if (settled.kind === "timeout") {
          const issue = timeoutIssue();
          const timeoutResult = issueToToolResult(issue);
          await appendToolResult(tc.id, tc.name, timeoutResult, true);
          updateAggregateToolCall(tc.id, { result: timeoutResult, isError: true });
          recordIssue(issue, turnIssues);
          yield issueEvent(issue, turnCtx);
          finishToolExecution();
          yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result: timeoutResult, isError: true };
          return;
        }

        if (settled.kind === "done") {
          result = settled.result;
          done = true;
          continue;
        }

        while (approvalQueue.length > 0) {
          const req = approvalQueue.shift()!;
          const matchesCustomApproval = (event: InboundEvent): event is Extract<InboundEvent, { type: "approval_response" }> =>
            event.type === "approval_response" && event.callId === req.id;
          let actionWaitStartedAt: number | undefined;
          if (!hasBufferedInbound(matchesCustomApproval)) {
            actionWaitStartedAt = nowMs();
            yield {
              type: "tool_action_request",
              ...eventFields,
              kind: "custom_approval",
              callId: req.id,
              name: tc.name,
              args: validatedInput,
              message: req.message,
            };
          }
          const response = await waitForActionResponse(actionWaitStartedAt, () =>
            pullWithinToolDeadline(matchesCustomApproval),
          );
          if (!response) {
            const issue = timeoutIssue();
            const timeoutResult = issueToToolResult(issue);
            await appendToolResult(tc.id, tc.name, timeoutResult, true);
            updateAggregateToolCall(tc.id, { result: timeoutResult, isError: true });
            recordIssue(issue, turnIssues);
            yield issueEvent(issue, turnCtx);
            finishToolExecution();
            yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result: timeoutResult, isError: true };
            return;
          }
          req.resolve(response.approved);
        }

        while (clientToolQueue.length > 0) {
          const req = clientToolQueue.shift()!;
          const bridgeTool = toolSnapshot.toolMap.get(req.name);
          let requestArgs = req.args;
          if (bridgeTool) {
            if (bridgeTool.kind !== "client") {
              req.reject(new ToolExecutionFailure(toolExecutionIssue(
                "unknown_tool",
                `Client tool "${req.name}" is not registered as a client tool.`,
                { id: req.id, name: req.name },
              )));
              continue;
            }
            const inputResult = bridgeTool.def.inputSchema.safeParse(req.args);
            if (!inputResult.success) {
              req.reject(new ToolExecutionFailure(toolExecutionIssue(
                "input_validation_failed",
                formatToolValidationError(bridgeTool, req.args, inputResult.error),
                { id: req.id, name: req.name },
              )));
              continue;
            }
            requestArgs = inputResult.data;
          }

          const matchesClientResult = (event: InboundEvent): event is Extract<InboundEvent, { type: "tool_result" }> =>
            event.type === "tool_result" && event.callId === req.id;
          let actionWaitStartedAt: number | undefined;
          if (!hasBufferedInbound(matchesClientResult)) {
            actionWaitStartedAt = nowMs();
            yield {
              type: "tool_action_request",
              ...eventFields,
              kind: "client_tool",
              callId: req.id,
              name: req.name,
              args: requestArgs,
            };
          }
          const response = await waitForActionResponse(actionWaitStartedAt, () =>
            pullWithinToolDeadline(matchesClientResult),
          );
          if (!response) {
            const issue = timeoutIssue();
            const timeoutResult = issueToToolResult(issue);
            await appendToolResult(tc.id, tc.name, timeoutResult, true);
            updateAggregateToolCall(tc.id, { result: timeoutResult, isError: true });
            recordIssue(issue, turnIssues);
            yield issueEvent(issue, turnCtx);
            finishToolExecution();
            yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result: timeoutResult, isError: true };
            return;
          }
          let output = response.result;
          if (bridgeTool?.kind === "client" && bridgeTool.def.outputSchema) {
            const outputResult = bridgeTool.def.outputSchema.safeParse(output);
            if (!outputResult.success) {
              req.reject(new ToolExecutionFailure(toolExecutionIssue(
                "output_validation_failed",
                `Output validation error for client tool "${req.name}": ${outputResult.error.message}`,
                { id: req.id, name: req.name },
              )));
              continue;
            }
            output = outputResult.data;
          }
          req.resolve(output);
        }
        queueNotify = null;
      }

      if (tool.def.outputSchema) {
        const outputResult = tool.def.outputSchema.safeParse(result);
        if (!outputResult.success) {
          const issue = toolExecutionIssue(
            "output_validation_failed",
            `Output validation error for tool "${tc.name}": ${outputResult.error.message}`,
            tc,
          );
          const output = issueToToolResult(issue);
          await appendToolResult(tc.id, tc.name, output, true);
          updateAggregateToolCall(tc.id, { result: output, isError: true });
          recordIssue(issue, turnIssues);
          yield issueEvent(issue, turnCtx);
          finishToolExecution();
          yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result: output, isError: true };
          return;
        }
      }

      const historicalIssue = await appendSuccessfulToolResult(tool, tc, validatedInput, result);
      updateAggregateToolCall(tc.id, { result });
      if (historicalIssue) {
        recordIssue(historicalIssue, turnIssues);
        yield issueEvent(historicalIssue, turnCtx);
      }
      finishToolExecution();
      yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result };
    } catch (error) {
      if (error instanceof LoopAbortedError || signal.aborted) throw error;
      const issue = error instanceof ToolExecutionFailure
        ? error.issue
        : toolExecutionIssue("execution_failed", toErrorMessage(error), tc);
      const result = issueToToolResult(issue);
      await appendToolResult(tc.id, tc.name, result, true);
      updateAggregateToolCall(tc.id, { result, isError: true });
      recordIssue(issue, turnIssues);
      yield issueEvent(issue, turnCtx);
      finishToolExecution();
      yield { type: "tool_execution_end", ...eventFields, callId: tc.id, name: tc.name, result, isError: true };
    } finally {
      finishToolExecution();
      if (timeoutHandle) clearTimeout(timeoutHandle);
      signal.removeEventListener("abort", abortTool);
      if (resolveLoopAborted) signal.removeEventListener("abort", resolveLoopAborted);
    }
  }

  const noopAggregateUpdate: UpdateAggregateToolCall = () => {};

  async function* runCompaction(operation: Promise<void>): AsyncGenerator<OutboundEvent> {
    yield { type: "compaction_start", agentId, loopId };
    try {
      await operation;
    } finally {
      yield { type: "compaction_end", agentId, loopId };
    }
  }

  async function* resumePendingToolCalls(turnCtx: TurnContext): AsyncGenerator<OutboundEvent> {
    const entries = await store.load();

    let lastAssistantIdx = -1;
    for (let i = entries.length - 1; i >= 0; i--) {
      const role = entries[i]!.message.role;
      if (role === "user") return;
      if (role === "assistant") {
        lastAssistantIdx = i;
        break;
      }
    }
    if (lastAssistantIdx < 0) return;

    // Only a trailing assistant exchange continues in this loop; a trailing user message starts a fresh one.
    loopTurns.splice(0, loopTurns.length, ...aggregateTurnsFromEntries(entries));

    const entry = entries[lastAssistantIdx]!;
    if (entry.kind === "summary") return;
    const assistantMessage = entry.message as AssistantMessage;
    // Calls from a response the provider stopped, or that the user aborted while it was still
    // being generated, are never executed. They reach the provider as interrupted calls instead.
    const stopped = assistantMessage.stopReason;
    if (stopped === "error" || stopped === "interrupted" || stopped === "aborted") return;
    const toolCallBlocks = assistantMessage.content.filter((block): block is ToolCallBlock => block.type === "tool_call");
    if (toolCallBlocks.length === 0) return;

    const resolvedCallIds = new Set<string>();
    for (let i = lastAssistantIdx + 1; i < entries.length; i++) {
      const message = entries[i]!.message;
      if (message.role === "tool_result") resolvedCallIds.add(message.callId);
    }

    const pending = toolCallBlocks.filter((block) => !resolvedCallIds.has(block.id));
    if (pending.length === 0) return;

    const toolSnapshot = await resolveToolSnapshot();

    const aggregateTurn = loopTurns.findLast((turn) => turn.message === assistantMessage);
    const aggregateToolCallMap = new Map((aggregateTurn?.toolCalls ?? []).map((toolCall) => [toolCall.callId, toolCall]));
    const updateAggregateToolCall = aggregateTurn
      ? (callId: string, patch: Partial<LoopToolCallAggregate>) => {
          const aggregateToolCall = aggregateToolCallMap.get(callId);
          if (aggregateToolCall) Object.assign(aggregateToolCall, patch);
        }
      : noopAggregateUpdate;

    const turnIssues = { issues: [] as LoopIssueAggregate[], toolIssues: [] as LoopToolIssueAggregate[] };
    const syncResumedIssues = () => {
      if (!aggregateTurn || turnIssues.issues.length === 0) return;
      aggregateTurn.issues = [...(aggregateTurn.issues ?? []), ...turnIssues.issues.map((issue) => ({ ...issue }))];
      aggregateTurn.toolIssues = [
        ...(aggregateTurn.toolIssues ?? []),
        ...turnIssues.toolIssues.map((issue) => ({ ...issue })),
      ];
    };
    yield { type: "turn_start", agentId, loopId, ...turnCtx, resumed: true };
    try {
      for (const tc of pending) {
        if (signal.aborted) break;
        yield* executeToolCall(tc, toolSnapshot, turnCtx, updateAggregateToolCall, turnIssues);
      }
    } catch (error) {
      // An abort while a resumed call runs still closes the turn.
      if (signal.aborted) {
        syncResumedIssues();
        yield { type: "turn_end", agentId, loopId, ...turnCtx, message: assistantMessage };
      }
      throw error;
    }
    syncResumedIssues();
    yield { type: "turn_end", agentId, loopId, ...turnCtx, message: assistantMessage };
  }

  async function* run(): AsyncGenerator<OutboundEvent> {
    let providerTurn = 0;
    let eventTurnIndex = 0;
    let compactionRetried = false;
    const prepareProviderMessages = (sourceEntries: StoreEntry[]): Message[] => {
      const rawMessages = sourceEntries.map((entry) => entry.message);
      const projectedMessages = closeUnansweredToolCalls(projectHistoricalToolResults(rawMessages, loopId));
      return typeof maxToolResultChars === "number"
        ? truncateToolResults(projectedMessages, maxToolResultChars)
        : projectedMessages;
    };

    timing.loopStartedAt = nowMs();
    yield { type: "loop_start", agentId, loopId };

    try {
      if (signal.aborted) {
        yield loopEndEvent("aborted");
        return;
      }

      if (input !== undefined) {
        await store.append(normalizeInput(input));
      } else {
        const resumeCtx = { turnId: createTurnId(loopId, eventTurnIndex, "resume"), turnIndex: eventTurnIndex };
        let emittedResumeTurn = false;
        for await (const event of resumePendingToolCalls(resumeCtx)) {
          emittedResumeTurn = true;
          yield event;
        }
        if (emittedResumeTurn) eventTurnIndex++;
        if (signal.aborted) {
          yield loopEndEvent("aborted");
          return;
        }
      }

      while (true) {
        if (signal.aborted) {
          yield loopEndEvent("aborted");
          return;
        }

        if (creditStore) {
          const remaining = await creditStore.remaining();
          if (remaining <= 0) {
            yield loopEndEvent("no_credits");
            return;
          }
        }

        const steeringApplied = yield* applyPendingSteering();
        if (steeringApplied) {
          providerTurn = 0;
          compactionRetried = false;
        }

        if (providerTurn >= maxTurns) {
          yield loopEndEvent("max_turns");
          return;
        }

        const toolSnapshot = await resolveToolSnapshot();
        const providerTools = toolSnapshot.providerTools;

        let entries = await store.load();
        let messages = prepareProviderMessages(entries);
        const contextWindow = provider.contextWindow;
        const computeFillRatio = (providerMessages: Message[]) => {
          if (typeof contextWindow !== "number" || contextWindow <= 0) return undefined;
          const estimatedTokens = Math.ceil(JSON.stringify({ systemPrompt, messages: providerMessages, tools: providerTools }).length / 4);
          const tokens = lastUsage.input > 0 ? Math.max(lastUsage.input, estimatedTokens) : estimatedTokens;
          return tokens / contextWindow;
        };

        if (compact && !compactionRetried) {
          const fillRatio = computeFillRatio(messages);
          const shouldForce = typeof fillRatio === "number" && fillRatio >= 0.85;
          const compaction = compact({
            entries,
            store,
            provider,
            usage: lastUsage,
            force: shouldForce,
            fillRatio,
            signal,
            reasoningEffort,
            extraBody,
          });
          if (compaction) {
            // Routine compaction is best effort: report a failure and continue with the current history.
            try {
              yield* runCompaction(compaction);
            } catch (error) {
              if (signal.aborted) throw error;
              const issue: NessiIssue = {
                kind: "runtime_error",
                message: `Compaction failed: ${toErrorMessage(error)}`,
                retryable: false,
              };
              recordIssue(issue);
              yield issueEvent(issue);
            }
            entries = await store.load();
            messages = prepareProviderMessages(entries);
          }
        }

        const turnCtx = { turnId: createTurnId(loopId, eventTurnIndex), turnIndex: eventTurnIndex };
        eventTurnIndex++;
        yield { type: "turn_start", agentId, loopId, ...turnCtx };

        let turnUsage: Usage = zeroUsage();
        let turnUsageReported = false;
        let stopReason: AssistantMessage["stopReason"] = "stop";
        const assistantBlocks: AssistantContentBlock[] = [];
        const openBlocks = new Map<string, { index: number; kind: "text" | "thinking"; text: string }>();
        const toolCalls: ToolCallBlock[] = [];
        const turnIssues = { issues: [] as LoopIssueAggregate[], toolIssues: [] as LoopToolIssueAggregate[] };
        let hadContextOverflow = false;
        let overflowRatio: number | undefined;
        let providerFailure: NessiIssue | undefined;

        const eventFields = { agentId, loopId, ...turnCtx };
        const makePartialMessage = (reason: AssistantMessage["stopReason"]): AssistantMessage => {
          const content = [...assistantBlocks];
          const pendingBlocks = [...openBlocks.entries()]
            .sort((left, right) => left[1].index - right[1].index)
            .map(([, block]): AssistantContentBlock =>
              block.kind === "thinking"
                ? { type: "thinking", thinking: block.text }
                : { type: "text", text: block.text },
            );
          for (const block of pendingBlocks) appendAssistantContentBlock(content, block);
          return buildAssistantMessageFromContent(provider.model, content, turnUsage, reason, provider.name);
        };

        /**
         * Ends a turn that did not complete normally: emits turn_end, counts usage the provider
         * already reported (aggregate and credits) and optionally keeps partial content in history.
         */
        const closeUnfinishedTurn = async function* (
          reason: "interrupted" | "error",
          persist: boolean,
        ): AsyncGenerator<OutboundEvent> {
          const message = makePartialMessage(reason);
          const usage = turnUsageReported ? turnUsage : undefined;
          // Storage or credit failures are reported but must not keep the turn open.
          const reportFailure = function* (what: string, error: unknown): Generator<OutboundEvent> {
            const issue = runtimeIssue(new Error(`${what}: ${toErrorMessage(error)}`));
            recordIssue(issue, turnIssues);
            yield issueEvent(issue, turnCtx);
          };
          if (persist && message.content.length > 0) {
            try {
              await store.append(message);
            } catch (error) {
              yield* reportFailure("Storing the partial assistant message failed", error);
            }
          }
          if (creditStore && usage?.creditsUsed && usage.creditsUsed > 0) {
            try {
              await creditStore.deduct(usage.creditsUsed);
            } catch (error) {
              yield* reportFailure("Deducting credits failed", error);
            }
          }
          if (message.content.length > 0 || usage) {
            recordAssistantTurn(
              message,
              usage,
              toolCalls.map((toolCall) => ({ callId: toolCall.id, name: toolCall.name, args: toolCall.args })),
              turnIssues.issues,
              turnIssues.toolIssues,
            );
          }
          yield { type: "turn_end", agentId, loopId, ...turnCtx, message };
        };

        try {
          const providerIterator = provider.stream({
            systemPrompt,
            messages,
            tools: providerTools,
            temperature,
            maxOutputTokens,
            disableReasoning,
            reasoningEffort,
            extraBody,
            signal,
          })[Symbol.asyncIterator]();
          try {
            streamLoop: while (true) {
              const next = await measureGeneration(() => providerIterator.next());
              if (next.done) break;
              const event = next.value;
              if (signal.aborted) break;

              switch (event.type) {
                case "block_start":
                  if (event.kind === "text" || event.kind === "thinking") {
                    openBlocks.set(event.blockId, { index: event.index, kind: event.kind, text: "" });
                  }
                  yield { ...event, ...eventFields };
                  break;

                case "block_delta": {
                  const open = openBlocks.get(event.blockId);
                  if (open) open.text += event.delta;
                  yield { ...event, ...eventFields };
                  break;
                }

                case "block_end":
                  openBlocks.delete(event.blockId);
                  appendAssistantContentBlock(assistantBlocks, event.block);
                  if (event.block.type === "tool_call") {
                    toolCalls.push(event.block);
                    stopReason = "tool_use";
                  }
                  yield { ...event, ...eventFields };
                  break;

                case "usage":
                  turnUsage = event.usage;
                  turnUsageReported = true;
                  stopReason = event.finishReason ?? stopReason;
                  yield { ...event, ...eventFields };
                  break;

                case "issue":
                  recordIssue(event.issue, turnIssues);
                  yield issueEvent(event.issue, turnCtx);
                  if (event.issue.kind === "provider_error" && event.issue.contextOverflow) {
                    hadContextOverflow = true;
                    overflowRatio = event.issue.overflowRatio;
                    break;
                  }
                  if (
                    event.issue.kind === "provider_error"
                    || (event.issue.kind === "timeout" && event.issue.scope !== "tool")
                    || event.issue.kind === "runtime_error"
                  ) {
                    providerFailure = event.issue;
                    break streamLoop;
                  }
                  break;

                default: {
                  const unsupported = event as { type?: unknown };
                  const issue: NessiIssue = {
                    kind: "runtime_error",
                    message: `Unsupported provider event type: ${String(unsupported.type)}`,
                    retryable: false,
                  };
                  recordIssue(issue, turnIssues);
                  yield issueEvent(issue, turnCtx);
                  providerFailure = issue;
                  break streamLoop;
                }
              }
            }
          } finally {
            await providerIterator.return?.();
          }
        } catch (error) {
          if (signal.aborted) {
            yield* closeUnfinishedTurn("interrupted", true);
            yield loopEndEvent("aborted");
            return;
          }
          const issue = runtimeIssue(error);
          recordIssue(issue, turnIssues);
          yield issueEvent(issue, turnCtx);
          yield* closeUnfinishedTurn("error", true);
          yield loopEndEvent("error");
          return;
        }

        if (hadContextOverflow) {
          // The attempt is not kept in history: compaction retries it, or the loop ends.
          yield* closeUnfinishedTurn("error", false);
          if (compact && !compactionRetried) {
            const estimatedFillRatio = computeFillRatio(messages);
            const fillRatio = typeof overflowRatio === "number"
              ? Math.max(estimatedFillRatio ?? 0, overflowRatio)
              : estimatedFillRatio;
            const compaction = compact({
              entries,
              store,
              provider,
              usage: lastUsage,
              force: true,
              fillRatio,
              signal,
              reasoningEffort,
              extraBody,
            });
            if (compaction) {
              yield* runCompaction(compaction);
              compactionRetried = true;
              continue;
            }
          }
          yield loopEndEvent("context_overflow");
          return;
        }

        if (providerFailure) {
          yield* closeUnfinishedTurn("error", true);
          yield loopEndEvent("error");
          return;
        }

        if (signal.aborted) {
          yield* closeUnfinishedTurn("interrupted", true);
          yield loopEndEvent("aborted");
          return;
        }

        const assistantMessage = buildAssistantMessageFromContent(
          provider.model,
          assistantBlocks,
          turnUsage,
          stopReason,
          provider.name,
        );
        lastUsage = turnUsage;
        const aggregateToolCalls: LoopToolCallAggregate[] = toolCalls.map((toolCall) => ({
          callId: toolCall.id,
          name: toolCall.name,
          args: toolCall.args,
        }));
        // Counted before storing, so a storage failure cannot lose usage the provider billed.
        recordAssistantTurn(
          assistantMessage,
          turnUsageReported ? turnUsage : undefined,
          aggregateToolCalls,
          turnIssues.issues,
          turnIssues.toolIssues,
        );
        // Storing and charging are attempted independently; the first failure then ends the loop.
        let storeFailed = false;
        let storeError: unknown;
        try {
          await store.append(assistantMessage);
        } catch (error) {
          storeFailed = true;
          storeError = error;
        }
        if (creditStore && turnUsage.creditsUsed && turnUsage.creditsUsed > 0) {
          try {
            await creditStore.deduct(turnUsage.creditsUsed);
          } catch (error) {
            if (!storeFailed) throw error;
          }
        }
        if (storeFailed) throw storeError;

        // The provider stopped the answer itself (content filter, safety, malformed call): its
        // tool calls are not executed and the loop ends with an error.
        if (stopReason === "error") {
          const issue: NessiIssue = {
            kind: "provider_error",
            message: `${provider.name} stopped the response early (finish reason "error").`,
            retryable: false,
          };
          recordIssue(issue, turnIssues);
          const recordedTurn = loopTurns[loopTurns.length - 1];
          if (recordedTurn?.message === assistantMessage) recordedTurn.issues = turnIssues.issues.map((item) => ({ ...item }));
          yield issueEvent(issue, turnCtx);
          yield { type: "turn_end", agentId, loopId, ...turnCtx, message: assistantMessage };
          yield loopEndEvent("error");
          return;
        }

        if (toolCalls.length === 0) {
          yield { type: "turn_end", agentId, loopId, ...turnCtx, message: assistantMessage };
          const lateSteeringApplied = yield* applyPendingSteering();
          if (lateSteeringApplied) {
            providerTurn = 0;
            compactionRetried = false;
            continue;
          }
          yield loopEndEvent("stop");
          return;
        }

        const aggregateToolCallMap = new Map(aggregateToolCalls.map((toolCall) => [toolCall.callId, toolCall]));
        const updateAggregateToolCall = (callId: string, patch: Partial<LoopToolCallAggregate>) => {
          const aggregateToolCall = aggregateToolCallMap.get(callId);
          if (aggregateToolCall) Object.assign(aggregateToolCall, patch);
        };

        let terminalToolCompleted = false;
        try {
          for (const tc of toolCalls) {
            yield* executeToolCall(tc, toolSnapshot, turnCtx, updateAggregateToolCall, turnIssues);
            const aggregateToolCall = aggregateToolCallMap.get(tc.id);
            const isTerminalTool = Boolean(
              (toolSnapshot.toolMap.get(tc.name)?.def as { terminal?: boolean } | undefined)?.terminal,
            );
            if (isTerminalTool && aggregateToolCall && !aggregateToolCall.isError) {
              terminalToolCompleted = true;
              break;
            }
          }
        } catch (error) {
          // The turn is already stored and counted; an abort during its tools only closes it.
          if (signal.aborted) {
            const recordedTurn = loopTurns[loopTurns.length - 1];
            if (recordedTurn?.message === assistantMessage) {
              recordedTurn.issues = turnIssues.issues.map((issue) => ({ ...issue }));
              recordedTurn.toolIssues = turnIssues.toolIssues.map((issue) => ({ ...issue }));
            }
            yield { type: "turn_end", agentId, loopId, ...turnCtx, message: assistantMessage };
          }
          throw error;
        }

        const recordedTurn = loopTurns[loopTurns.length - 1];
        if (recordedTurn?.message === assistantMessage) {
          recordedTurn.issues = turnIssues.issues.map((issue) => ({ ...issue }));
          recordedTurn.toolIssues = turnIssues.toolIssues.map((issue) => ({ ...issue }));
        }

        yield { type: "turn_end", agentId, loopId, ...turnCtx, message: assistantMessage };
        if (terminalToolCompleted) {
          const lateSteeringApplied = yield* applyPendingSteering();
          if (lateSteeringApplied) {
            providerTurn = 0;
            compactionRetried = false;
            continue;
          }
          yield loopEndEvent("stop");
          return;
        }
        providerTurn++;
        compactionRetried = false;
      }

    } catch (error) {
      if (signal.aborted) {
        yield loopEndEvent("aborted");
        return;
      }
      const issue = runtimeIssue(error);
      recordIssue(issue);
      yield issueEvent(issue);
      yield loopEndEvent("error");
    }
  }

  /**
   * Safety net for the turn_end guarantee: if an unexpected error ends the loop while a turn is
   * open, the turn is closed with an empty message before loop_end.
   */
  async function* closeOpenTurns(source: AsyncGenerator<OutboundEvent>): AsyncGenerator<OutboundEvent> {
    let openTurn: { turnId: string; turnIndex: number } | undefined;
    for await (const event of source) {
      if (event.type === "turn_start") openTurn = { turnId: event.turnId, turnIndex: event.turnIndex };
      else if (event.type === "turn_end") openTurn = undefined;
      else if (event.type === "loop_end" && openTurn) {
        const stopReason = event.reason === "aborted" ? "interrupted" : "error";
        const message = buildAssistantMessageFromContent(provider.model, [], undefined, stopReason, provider.name);
        yield { type: "turn_end", agentId, loopId, ...openTurn, message };
        openTurn = undefined;
      }
      yield event;
    }
  }

  const eventSource = coalesce ? coalesceOutboundEvents(closeOpenTurns(run()), coalesce) : closeOpenTurns(run());
  const generator = eventSource[Symbol.asyncIterator]();

  const loop: NessiLoop = {
    [Symbol.asyncIterator]() {
      return {
        async next() {
          const result = await generator.next();
          if (!result.done && result.value) {
            for (const listener of subscribers) listener(result.value);
          }
          return result;
        },
        async return(value?: OutboundEvent) {
          // Stopping iteration early ends the loop: cancel running tools and provider requests.
          abortController.abort();
          return generator.return(value as OutboundEvent);
        },
        async throw(error?: unknown) {
          return generator.throw(error);
        },
      };
    },
    subscribe(listener: (event: OutboundEvent) => void) {
      subscribers.push(listener);
      return () => {
        const idx = subscribers.indexOf(listener);
        if (idx >= 0) subscribers.splice(idx, 1);
      };
    },
    push(event: InboundEvent) {
      channel.push(event);
    },
    steer(message: string) {
      if (message.trim()) steerQueue.push(message);
    },
    abort() {
      abortController.abort();
    },
  };

  return loop;
}
