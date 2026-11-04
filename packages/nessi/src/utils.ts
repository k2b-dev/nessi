// ============================================================================
// nessi – Shared Utilities
// ============================================================================

import type { Message, ToolCallBlock, ToolResultMessage, Usage } from "./types.js";

export const zeroUsage = (): Usage => ({ input: 0, output: 0, total: 0 })

export const toErrorMessage = (err: unknown) =>
  err instanceof Error ? err.message : String(err)

export const createLoopId = () =>
  globalThis.crypto?.randomUUID?.() ?? `loop-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

/**
 * Tokens one file part (an image) is assumed to cost. Providers price images by pixels, not by
 * their base64 size; about 1,600 tokens is Anthropic's documented upper range for one image.
 */
export const FILE_PART_TOKEN_ESTIMATE = 1_600;

/**
 * Rough token estimate: ~4 characters per token for text and JSON, a fixed allowance per file
 * part of a user message. Those base64 bodies are not counted as text; anything else, tool
 * results included, is. `extra` (system prompt, tool specs) counts as text. Slightly
 * overestimates text due to JSON syntax.
 */
export const estimateContextTokens = (messages: Message[], extra?: Record<string, unknown>): number => {
  let files = 0;
  const withoutFileBodies = messages.map((message) => {
    if (message.role !== "user" || !message.content.some((part) => typeof part !== "string" && part.type === "file")) return message;
    return {
      ...message,
      content: message.content.map((part) => {
        if (typeof part === "string" || part.type !== "file") return part;
        files++;
        return { ...part, data: "" };
      }),
    };
  });
  const text = JSON.stringify(extra === undefined ? withoutFileBodies : { ...extra, messages: withoutFileBodies });
  return Math.ceil(text.length / 4) + files * FILE_PART_TOKEN_ESTIMATE;
};

/** Rough token estimate of messages; see `estimateContextTokens()`. */
export const estimateTokens = (messages: Message[]): number => estimateContextTokens(messages);

/** Truncate text keeping first half and last half with omission notice. */
export const truncateMiddle = (text: string, maxChars: number): string => {
  if (text.length <= maxChars) return text;
  const half = Math.floor(maxChars / 2);
  const omitted = text.length - 2 * half;
  return `${text.slice(0, half)}\n[... ${omitted} characters omitted ...]\n${text.slice(-half)}`;
}

const stringifyResult = (value: unknown): string => {
  if (typeof value === "string") return value;
  try { return JSON.stringify(value); }
  catch { return String(value); }
}

/**
 * Truncate oversized tool results in a message list.
 * Returns shallow copies — the store is untouched.
 */
export const truncateToolResults = (messages: Message[], maxChars: number): Message[] =>
  messages.map((msg) => {
    if (msg.role !== "tool_result") return msg;
    const text = stringifyResult(msg.result);
    if (text.length <= maxChars) return msg;
    return { ...msg, result: truncateMiddle(text, maxChars) };
  })

/** What the model reads for a call of a response that ended before its calls could run. */
export const NOT_RUN_TOOL_RESULT = "No result: the response ended before this call could run, so it did not run.";

/** What the model reads for a call that may have run but has no stored result. */
export const NOT_RETURNED_TOOL_RESULT =
  "No result: the turn ended before this call returned, so it may or may not have run. Check its effect before you repeat it.";

const MALFORMED_CALL_ID = /^malformed_[0-9a-f]{24}$/;

/**
 * nessi stores a call the provider sent malformed under this kind of ID, so the call can be paired
 * with its error result and is never executed.
 */
export const newMalformedCallId = (): string =>
  `malformed_${globalThis.crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`;

export const isMalformedCallId = (id: string): boolean => MALFORMED_CALL_ID.test(id);

const excerpt = (text: string, maxChars: number) => (text.length <= maxChars ? text : `${text.slice(0, maxChars - 1)}…`);

/** A name every provider accepts in history; a missing or unusable one becomes `unknown_tool`. */
export const malformedCallName = (name: string | undefined): string =>
  name && /^[A-Za-z0-9_-]{1,64}$/.test(name) ? name : "unknown_tool";

const MALFORMED_REPAIR_REQUEST =
  "Re-send the call with its exact tool name and complete, valid JSON arguments. The arguments may have been cut off; use smaller ones, for example one file per call. Do not repeat calls that already succeeded.";

/** The error result of a malformed call. It never repeats the arguments, which can be very large. */
export const malformedCallResult = (issueMessage?: string): string =>
  issueMessage ? `${excerpt(issueMessage, 512)} ${MALFORMED_REPAIR_REQUEST}` : `This tool call could not be used. ${MALFORMED_REPAIR_REQUEST}`;

const STOPPED_BEFORE_TOOLS = new Set<string | undefined>(["error", "interrupted", "aborted"]);

/** The error result nessi sends for a call without a stored result. */
const openCallResult = (call: ToolCallBlock, stopReason: string | undefined): ToolResultMessage => {
  const reason = isMalformedCallId(call.id) ? "malformed_tool_call" : STOPPED_BEFORE_TOOLS.has(stopReason) ? "not_run" : "not_returned";
  const result = reason === "malformed_tool_call" ? malformedCallResult() : reason === "not_run" ? NOT_RUN_TOOL_RESULT : NOT_RETURNED_TOOL_RESULT;
  return { role: "tool_result", callId: call.id, name: call.name, result, isError: true, reason };
};

/**
 * Where each call of the assistant message at `index` has its result: the first result for it
 * before the next user message, or before a later assistant message with a call of the same ID,
 * since some providers reuse call IDs from turn to turn.
 */
const resultsOf = (messages: Message[], index: number, ids: ReadonlySet<string>): Map<string, number> => {
  const found = new Map<string, number>();
  for (let next = index + 1; next < messages.length && found.size < ids.size; next++) {
    const later = messages[next]!;
    if (later.role === "user") break;
    if (later.role === "assistant" && later.content.some((block) => block.type === "tool_call" && ids.has(block.id))) break;
    if (later.role === "tool_result" && ids.has(later.callId) && !found.has(later.callId)) found.set(later.callId, next);
  }
  return found;
};

/**
 * Gives every tool call its result right after the message that made it, as providers require. A
 * call without a result gets an error result with a `reason`; a result stored after a later
 * assistant message moves up to its call. Returns a copy; the store is untouched.
 */
export const closeUnansweredToolCalls = (messages: Message[]): Message[] => {
  let out: Message[] | null = null;
  const moved = new Set<number>();
  for (let index = 0; index < messages.length; index++) {
    if (moved.has(index)) continue;
    const message = messages[index]!;
    out?.push(message);
    if (message.role !== "assistant") continue;
    const calls = message.content.filter((block): block is ToolCallBlock => block.type === "tool_call");
    if (calls.length === 0) continue;
    const results = resultsOf(messages, index, new Set(calls.map((call) => call.id)));
    const late = (at: number) => messages.slice(index + 1, at).some((later) => later.role === "assistant");
    const misplaced = calls.filter((call) => {
      const at = results.get(call.id);
      return at === undefined || late(at);
    });
    if (misplaced.length === 0) continue;
    out ??= messages.slice(0, index + 1);
    for (const call of misplaced) {
      const at = results.get(call.id);
      if (at === undefined) {
        out.push(openCallResult(call, message.stopReason));
      } else {
        out.push(messages[at]!);
        moved.add(at);
      }
    }
  }
  return out ?? messages;
};

/** Select persisted historical tool results for provider calls outside their originating loop. */
export const projectHistoricalToolResults = (messages: Message[], loopId: string): Message[] =>
  messages.map((message) => {
    if (message.role !== "tool_result" || !message.historicalResult) return message;
    const { historicalResult, ...providerMessage } = message;
    if (historicalResult.originLoopId === loopId) return providerMessage;
    return { ...providerMessage, result: historicalResult.value };
  })

/**
 * Aborts `controller` with `signal`'s reason when `signal` aborts. Returns a cleanup that detaches
 * the listener, so a long-lived caller signal does not collect one listener per finished run.
 */
export const followAbortSignal = (signal: AbortSignal | undefined, controller: AbortController): (() => void) => {
  if (!signal) return () => {};
  if (signal.aborted) {
    controller.abort(signal.reason);
    return () => {};
  }
  const abort = () => controller.abort(signal.reason);
  signal.addEventListener("abort", abort, { once: true });
  return () => signal.removeEventListener("abort", abort);
};

/**
 * Fails on options that @k2b/nessi 1.0 removed and that changed behavior, so a caller that still
 * passes one (for example from JavaScript or a spread object) does not get a silently different run.
 */
export const rejectRemovedOptions = (options: object, removed: Record<string, string>, api: string) => {
  for (const [name, hint] of Object.entries(removed)) {
    if (Object.hasOwn(options, name) && Reflect.get(options, name) !== undefined) {
      throw new TypeError(`${api}: the "${name}" option was removed in @k2b/nessi 1.0. ${hint}`);
    }
  }
};
