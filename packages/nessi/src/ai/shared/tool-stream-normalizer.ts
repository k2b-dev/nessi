import { safeJsonParse, stringifyJson } from "./json.js";
import type {
  AssistantBlockKind,
  AssistantContentBlock,
  NessiIssue,
  RawStreamEvent,
  StreamEvent,
  ToolCallBlock,
  ToolStreamIssue,
  ToolStreamIssueReason,
} from "../types.js";

type PendingToolCall = {
  callId: string;
  name: string;
  argsText: string;
  argsDeltas: string[];
};

type NormalizerOptions = {
  suppressTextAfterMalformedTool?: boolean;
};

type OpenBlock = {
  blockId: string;
  index: number;
  kind: "text" | "thinking";
  text: string;
  signature?: string;
  redacted?: string;
  details?: Record<string, unknown>[];
};

type BlockData = { signature?: string; redacted?: string; details?: Record<string, unknown>[] };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const parseArgsText = (argsText: string) => {
  const trimmed = argsText.trim();
  if (!trimmed) return { ok: true as const };
  return isRecord(safeJsonParse(trimmed))
    ? { ok: true as const }
    : { ok: false as const };
};

const issueMessage = (reason: ToolStreamIssueReason, tool: PendingToolCall | undefined) => {
  const label = tool?.name ? `"${tool.name}"` : "unknown tool";
  switch (reason) {
    case "text_during_tool_call":
      return `Received text while a tool call for ${label} was still open.`;
    case "thinking_during_tool_call":
      return `Received thinking text while a tool call for ${label} was still open.`;
    case "tool_delta_without_start":
      return "Received a tool argument delta before a matching tool_start event.";
    case "missing_tool_name":
      return "Received a tool call without a usable tool name.";
    case "invalid_tool_arguments":
      return `Received malformed JSON arguments for ${label}.`;
    case "stream_ended_before_tool_call":
      return `The stream ended before a pending tool call for ${label} became executable.`;
    case "provider_error_before_tool_call":
      return `The provider errored before a pending tool call for ${label} became executable.`;
  }
};

function toolIssue(
  kind: ToolStreamIssue["kind"],
  pending: PendingToolCall | undefined,
  reason: ToolStreamIssueReason,
  extra?: { textDelta?: string },
): ToolStreamIssue {
  return {
    kind,
    callId: pending?.callId,
    name: pending?.name,
    reason,
    message: issueMessage(reason, pending),
    argsText: pending?.argsText,
    textDelta: extra?.textDelta,
  };
}

const blockFromOpen = (block: OpenBlock): AssistantContentBlock => {
  const signature = block.signature !== undefined ? { signature: block.signature } : {};
  if (block.kind === "text") return { type: "text", text: block.text, ...signature };
  return {
    type: "thinking",
    thinking: block.text,
    ...signature,
    ...(block.redacted !== undefined ? { redacted: block.redacted } : {}),
    ...(block.details !== undefined ? { details: block.details } : {}),
  };
};

const isProviderTimeoutError = (error: unknown): error is { scope: "provider_first_byte" | "provider_idle"; message: string } =>
  Boolean(error)
  && typeof error === "object"
  && ((error as { scope?: unknown }).scope === "provider_first_byte" || (error as { scope?: unknown }).scope === "provider_idle")
  && typeof (error as { message?: unknown }).message === "string";

export async function* normalizeProviderStream(
  events: AsyncIterable<RawStreamEvent>,
  options: NormalizerOptions = {},
): AsyncIterable<StreamEvent> {
  const pending = new Map<string, PendingToolCall>();
  const malformedCallIds = new Set<string>();
  let openBlock: OpenBlock | undefined;
  let nextBlockIndex = 0;
  let emittedToolCallCount = 0;
  let issueCount = 0;
  let suppressMalformedTextSpan = false;
  let pendingThinkingWhitespace = "";

  const emitIssue = function* (value: NessiIssue): Generator<StreamEvent> {
    issueCount++;
    yield { type: "issue", issue: value };
  };

  const closeOpenBlock = function* (): Generator<StreamEvent> {
    if (!openBlock) return;
    const closing = openBlock;
    openBlock = undefined;
    yield {
      type: "block_end",
      blockId: closing.blockId,
      index: closing.index,
      block: blockFromOpen(closing),
    };
  };

  const appendTextBlock = function* (
    kind: "text" | "thinking",
    delta: string,
    data: BlockData = {},
  ): Generator<StreamEvent> {
    const hasData = data.signature !== undefined || data.redacted !== undefined || data.details !== undefined;
    if (!hasData && delta.length === 0) return;

    if (openBlock?.kind !== kind) {
      yield* closeOpenBlock();
      // Leading whitespace alone does not open a block. Thinking keeps it for the block that
      // follows, because signed reasoning must be returned byte for byte.
      if (!hasData && delta.trim().length === 0) {
        if (kind === "thinking") pendingThinkingWhitespace += delta;
        return;
      }
      const index = nextBlockIndex++;
      openBlock = { blockId: `block-${index}`, index, kind, text: "" };
      yield { type: "block_start", blockId: openBlock.blockId, index, kind };
      if (kind === "thinking") delta = pendingThinkingWhitespace + delta;
      pendingThinkingWhitespace = "";
    }

    if (delta.length > 0) {
      openBlock.text += delta;
      yield { type: "block_delta", blockId: openBlock.blockId, delta };
    }
    if (data.signature !== undefined) openBlock.signature = data.signature;
    if (data.redacted !== undefined) openBlock.redacted = data.redacted;
    if (data.details !== undefined) openBlock.details = data.details;
    // A signature completes a thinking block; following reasoning belongs to a new one.
    if (kind === "thinking" && hasData) yield* closeOpenBlock();
  };

  const emitToolCallBlock = function* (toolCall: ToolCallBlock): Generator<StreamEvent> {
    yield* closeOpenBlock();
    pendingThinkingWhitespace = "";
    const index = nextBlockIndex++;
    const blockId = `block-${index}`;
    yield {
      type: "block_start",
      blockId,
      index,
      kind: "tool_call" as AssistantBlockKind,
      callId: toolCall.id,
      name: toolCall.name,
    };
    yield { type: "block_delta", blockId, delta: stringifyJson(toolCall.args) };
    yield { type: "block_end", blockId, index, block: toolCall };
  };

  const malformedPending = function* (
    reason: "text_during_tool_call" | "thinking_during_tool_call",
    textDelta: string,
  ): Generator<StreamEvent> {
    yield* closeOpenBlock();
    for (const tool of pending.values()) {
      malformedCallIds.add(tool.callId);
      yield* emitIssue(toolIssue("malformed_tool_call", tool, reason, { textDelta }));
    }
    pending.clear();
    if (options.suppressTextAfterMalformedTool) suppressMalformedTextSpan = true;
  };

  const cancelPending = function* (reason: "stream_ended_before_tool_call" | "provider_error_before_tool_call") {
    if (pending.size === 0) return;
    yield* closeOpenBlock();
    for (const tool of pending.values()) {
      yield* emitIssue(toolIssue("cancelled_tool_call", tool, reason));
    }
    pending.clear();
  };

  try {
    for await (const event of events) {
      switch (event.type) {
        case "text":
          if (pending.size > 0) {
            yield* malformedPending("text_during_tool_call", event.delta);
            if (options.suppressTextAfterMalformedTool) break;
          }
          if (suppressMalformedTextSpan) break;
          yield* appendTextBlock("text", event.delta, { signature: event.signature });
          break;

        case "thinking":
          if (pending.size > 0) {
            yield* malformedPending("thinking_during_tool_call", event.delta);
            if (options.suppressTextAfterMalformedTool) break;
          }
          if (suppressMalformedTextSpan) break;
          yield* appendTextBlock("thinking", event.delta, {
            signature: event.signature,
            redacted: event.redacted,
            details: event.details,
          });
          break;

        case "tool_start":
          suppressMalformedTextSpan = false;
          yield* closeOpenBlock();
          pending.set(event.callId, { callId: event.callId, name: event.name, argsText: "", argsDeltas: [] });
          break;

        case "tool_delta": {
          suppressMalformedTextSpan = false;
          if (malformedCallIds.has(event.callId)) break;
          const tool = pending.get(event.callId);
          if (!tool) {
            yield* emitIssue({
              kind: "malformed_tool_call",
              reason: "tool_delta_without_start",
              message: issueMessage("tool_delta_without_start", undefined),
              callId: event.callId,
              argsText: event.argsDelta,
            });
            break;
          }
          tool.argsText += event.argsDelta;
          tool.argsDeltas.push(event.argsDelta);
          break;
        }

        case "tool_call": {
          suppressMalformedTextSpan = false;
          const tool = pending.get(event.callId);
          if (malformedCallIds.has(event.callId)) {
            malformedCallIds.delete(event.callId);
            pending.delete(event.callId);
            break;
          }

          if (!event.name.trim()) {
            malformedCallIds.add(event.callId);
            pending.delete(event.callId);
            yield* emitIssue(toolIssue(
              "malformed_tool_call",
              tool ?? { callId: event.callId, name: event.name, argsText: "", argsDeltas: [] },
              "missing_tool_name",
            ));
            break;
          }

          if (tool && !parseArgsText(tool.argsText).ok) {
            malformedCallIds.add(event.callId);
            pending.delete(event.callId);
            yield* emitIssue(toolIssue("malformed_tool_call", tool, "invalid_tool_arguments"));
            break;
          }

          pending.delete(event.callId);
          emittedToolCallCount++;
          yield* emitToolCallBlock({
            type: "tool_call",
            id: event.callId,
            name: event.name,
            args: event.args,
            ...(event.signature !== undefined ? { signature: event.signature } : {}),
          });
          break;
        }

        case "tool_error":
        case "tool_cancel": {
          suppressMalformedTextSpan = false;
          if (event.callId) pending.delete(event.callId);
          yield* closeOpenBlock();
          yield* emitIssue({
            kind: event.type === "tool_error" ? "malformed_tool_call" : "cancelled_tool_call",
            reason: event.reason,
            message: event.message,
            callId: event.callId,
            name: event.name,
            argsText: event.argsText,
            textDelta: event.textDelta,
          });
          break;
        }

        case "usage":
          suppressMalformedTextSpan = false;
          if (event.finishReason && pending.size > 0) {
            yield* cancelPending("stream_ended_before_tool_call");
          }
          yield* closeOpenBlock();
          yield {
            ...event,
            finishReason: event.finishReason === "tool_use" && emittedToolCallCount === 0 && issueCount > 0
              ? "stop"
              : event.finishReason,
          };
          break;

        case "error":
          suppressMalformedTextSpan = false;
          yield* cancelPending("provider_error_before_tool_call");
          yield* closeOpenBlock();
          yield* emitIssue({
            kind: "provider_error",
            message: event.error,
            retryable: event.retryable,
            contextOverflow: event.contextOverflow,
            overflowRatio: event.overflowRatio,
          });
          break;

        case "timeout":
          suppressMalformedTextSpan = false;
          yield* cancelPending("provider_error_before_tool_call");
          yield* closeOpenBlock();
          yield* emitIssue({
            kind: "timeout",
            scope: event.scope,
            message: event.message,
            retryable: event.retryable,
          });
          break;
      }
    }
  } catch (error) {
    yield* cancelPending("provider_error_before_tool_call");
    yield* closeOpenBlock();
    if (isProviderTimeoutError(error)) {
      yield* emitIssue({
        kind: "timeout",
        scope: error.scope,
        message: error.message,
        retryable: true,
      });
      return;
    }
    yield* emitIssue({
      kind: "provider_error",
      message: error instanceof Error ? error.message : String(error),
      retryable: true,
    });
    return;
  }

  yield* cancelPending("stream_ended_before_tool_call");
  yield* closeOpenBlock();
}
