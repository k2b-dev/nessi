import { fallbackToolCallPrefix } from "../shared/tool-call-ids.js";
import { buildRequest, invalidResponseError, normalizeConnectionError, normalizeHttpError, providerError, serializeRequest, streamEndedError } from "../shared/errors.js";
import { fetchCompleteResponse } from "../shared/complete-response.js";
import { assertOnlySupportedFiles, buildAssistantMessage } from "../shared/messages.js";
import { parseNDJSON } from "../shared/ndjson.js";
import { resolveReasoning, withExtraBody } from "../shared/request-options.js";
import { malformedToolArgumentsError, parseToolArguments, safeJsonParse, stringifyJson } from "../shared/json.js";
import { normalizeProviderStream } from "../shared/tool-stream-normalizer.js";
import { toOllamaTools } from "../shared/tools.js";
import { applyCredits, makeUsage } from "../shared/usage.js";
import type {
  GenerateRequest,
  GenerateResult,
  Message,
  Provider,
  ProviderRequestDefaults,
  ProviderTimeouts,
  RawStreamEvent,
  StreamEvent,
  ToolCallBlock,
} from "../types.js";

type OllamaMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  images?: string[];
  tool_call_id?: string;
  name?: string;
  tool_calls?: Array<{ function: { name: string; arguments: Record<string, unknown> } }>;
};

type OllamaResponse = {
  model?: string;
  message?: {
    role?: string;
    content?: string;
    thinking?: string;
    /** Usually an object; validated before a call becomes executable. */
    tool_calls?: Array<{ function: { name: string; arguments?: unknown } }>;
  };
  done: boolean;
  done_reason?: string;
  error?: string;
  prompt_eval_count?: number;
  eval_count?: number;
};

export type OllamaOptions = ProviderRequestDefaults & {
  baseURL?: string;
  contextWindow?: number;
  temperature?: number;
  creditsPerInputToken?: number;
  creditsPerOutputToken?: number;
  timeouts?: ProviderTimeouts;
};

const convertMessages = (messages: Message[], systemPrompt: string | undefined) => {
  const out: OllamaMessage[] = [];
  if (systemPrompt) out.push({ role: "system", content: systemPrompt });

  for (const message of messages) {
    if (message.role === "user") {
      assertOnlySupportedFiles(message.content, true, "ollama");
      let text = "";
      const images: string[] = [];
      for (const part of message.content) {
        if (typeof part === "string") text += part;
        else if (part.type === "text") text += part.text;
        else images.push(part.data);
      }
      const next: OllamaMessage = { role: "user", content: text };
      if (images.length > 0) next.images = images;
      out.push(next);
    } else if (message.role === "assistant") {
      let text = "";
      const toolCalls: OllamaMessage["tool_calls"] = [];
      for (const block of message.content) {
        if (block.type === "text") text += block.text;
        else if (block.type === "tool_call") {
          toolCalls.push({ function: { name: block.name, arguments: block.args } });
        }
      }
      const next: OllamaMessage = { role: "assistant", content: text };
      if (toolCalls.length > 0) next.tool_calls = toolCalls;
      out.push(next);
    } else {
      out.push({
        role: "tool",
        name: message.name,
        tool_call_id: message.callId,
        content: stringifyJson({
          tool_call_id: message.callId,
          name: message.name,
          result: message.result,
        }),
      });
    }
  }

  return out;
};

const usageFromResponse = (response: OllamaResponse, options?: OllamaOptions) =>
  applyCredits(
    makeUsage(response.prompt_eval_count ?? 0, response.eval_count ?? 0),
    options?.creditsPerInputToken,
    options?.creditsPerOutputToken,
  );

const toolCallsFromResponse = (response: OllamaResponse, idPrefix = fallbackToolCallPrefix("ollama")): ToolCallBlock[] =>
  (response.message?.tool_calls ?? []).map((toolCall, index) => {
    const { name } = toolCall.function;
    const args = parseToolArguments(toolCall.function.arguments);
    if (!args) throw malformedToolArgumentsError("ollama", name);
    return { type: "tool_call", id: `${idPrefix}-${index}`, name, args };
  });

const generationOptions = (request: GenerateRequest, options?: OllamaOptions) => {
  const result: Record<string, unknown> = {};
  const temperature = request.temperature ?? options?.temperature;
  if (temperature !== undefined) result.temperature = temperature;
  if (request.maxOutputTokens !== undefined) result.num_predict = request.maxOutputTokens;
  // Ollama's own default context depends on the server's memory; an explicit contextWindow is
  // sent so Ollama and Nessi's compaction agree on the same size.
  if (options?.contextWindow !== undefined) result.num_ctx = options.contextWindow;
  return Object.keys(result).length > 0 ? result : undefined;
};

/**
 * Ollama's `think` takes `false` or a model-defined level such as "high". "none" turns it off;
 * other levels pass through. Boolean-only models need `extraBody: { think: true }`.
 */
const applyThink = (body: Record<string, unknown>, request: GenerateRequest, options?: OllamaOptions) => {
  const { effort } = resolveReasoning(request, options);
  if (effort !== undefined) body.think = effort === "none" ? false : effort;
};

const finishReasonFrom = (response: OllamaResponse, hasTools: boolean) => {
  if (response.done_reason === "length") return "max_tokens" as const;
  return hasTools ? "tool_use" as const : "stop" as const;
};

export const ollama = (model: string, options?: OllamaOptions): Provider => {
  const baseURL = (options?.baseURL ?? "http://localhost:11434").replace(/\/+$/, "");
  const contextWindow = options?.contextWindow ?? 128_000;

  return {
    name: "ollama",
    family: "ollama",
    model,
    contextWindow,
    capabilities: {
      streaming: true,
      tools: true,
      images: true,
      thinking: true,
      usage: true,
      structuredOutput: true,
    },

    async complete(request: GenerateRequest): Promise<GenerateResult> {
      const body: Record<string, unknown> = {
        model,
        messages: buildRequest("ollama", () => convertMessages(request.messages, request.systemPrompt)),
        stream: false,
      };
      if (request.tools?.length) body.tools = toOllamaTools(request.tools);
      if (request.responseFormat) body.format = request.responseFormat.schema;
      const generation = generationOptions(request, options);
      if (generation) body.options = generation;
      applyThink(body, request, options);

      const response = await fetchCompleteResponse(`${baseURL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: serializeRequest("ollama", withExtraBody(body, request, options)),
        signal: request.signal,
      }, "ollama", options?.timeouts);

      if (!response.ok) {
        const normalized = await normalizeHttpError("ollama", response);
        throw providerError(normalized);
      }

      const payload = safeJsonParse<OllamaResponse>(await response.text());
      if (!payload) throw invalidResponseError("ollama");
      const usage = usageFromResponse(payload, options);
      const toolCalls = toolCallsFromResponse(payload);
      const finishReason = finishReasonFrom(payload, toolCalls.length > 0);

      return {
        message: buildAssistantMessage(
          model,
          payload.message?.content ?? "",
          payload.message?.thinking ?? "",
          toolCalls,
          usage,
          finishReason,
          "ollama",
        ),
        usage,
        finishReason,
        providerMeta: { model },
      };
    },

    stream(request: GenerateRequest): AsyncIterable<StreamEvent> {
      const raw = async function* (): AsyncIterable<RawStreamEvent> {
      const body: Record<string, unknown> = {
        model,
        messages: buildRequest("ollama", () => convertMessages(request.messages, request.systemPrompt)),
        stream: true,
      };
      if (request.tools?.length) body.tools = toOllamaTools(request.tools);
      if (request.responseFormat) body.format = request.responseFormat.schema;
      const generation = generationOptions(request, options);
      if (generation) body.options = generation;
      applyThink(body, request, options);

      let response: Response;
      const serializedBody = serializeRequest("ollama", withExtraBody(body, request, options));
      const controller = new AbortController();
      const abortExternal = () => controller.abort(request.signal?.reason);
      const cleanupExternalAbort = () => {
        if (request.signal) request.signal.removeEventListener("abort", abortExternal);
      };
      if (request.signal) {
        if (request.signal.aborted) controller.abort(request.signal.reason);
        else request.signal.addEventListener("abort", abortExternal, { once: true });
      }
      let firstByteTimeout: ReturnType<typeof setTimeout> | undefined;
      const firstByteDeadline = options?.timeouts?.firstByteMs && options.timeouts.firstByteMs > 0
        ? Date.now() + options.timeouts.firstByteMs
        : undefined;
      try {
        response = await Promise.race([
          fetch(`${baseURL}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: serializedBody,
            redirect: "error",
            signal: controller.signal,
          }),
          new Promise<never>((_, reject) => {
            if (!options?.timeouts?.firstByteMs || options.timeouts.firstByteMs <= 0) return;
            firstByteTimeout = setTimeout(() => {
              // Reject first: the aborted fetch would otherwise win the race as a connection error.
              const timeout = {
                scope: "provider_first_byte",
                message: `ollama first byte timeout after ${options.timeouts!.firstByteMs}ms.`,
              };
              reject(timeout);
              controller.abort(timeout);
            }, options.timeouts.firstByteMs);
          }),
        ]);
      } catch (error) {
        cleanupExternalAbort();
        if (error && typeof error === "object" && (error as { scope?: unknown }).scope === "provider_first_byte") {
          yield {
            type: "timeout",
            scope: "provider_first_byte",
            message: String((error as { message?: unknown }).message ?? "ollama first byte timeout"),
            retryable: true,
          };
          return;
        }
        yield { type: "error", ...normalizeConnectionError("ollama", error) };
        return;
      } finally {
        if (firstByteTimeout) clearTimeout(firstByteTimeout);
      }

      if (!response.ok) {
        const normalized = await normalizeHttpError("ollama", response);
        cleanupExternalAbort();
        yield { type: "error", ...normalized };
        return;
      }

      const reader = response.body?.getReader() as ReadableStreamDefaultReader<Uint8Array> | undefined;
      if (!reader) {
        cleanupExternalAbort();
        yield { type: "error", error: "ollama response body missing", retryable: false };
        return;
      }

      let toolCounter = 0;
      const idPrefix = fallbackToolCallPrefix("ollama");
      let sawDone = false;
      const streamTimeouts = options?.timeouts ? { ...options.timeouts } : undefined;
      if (firstByteDeadline && streamTimeouts) {
        streamTimeouts.firstByteMs = Math.max(1, firstByteDeadline - Date.now());
      }

      try {
        for await (const chunk of parseNDJSON<OllamaResponse>(reader, streamTimeouts)) {
          if (chunk.error) {
            yield { type: "error", error: `ollama stream error: ${chunk.error}`, retryable: true };
            return;
          }
          if (chunk.message?.thinking) yield { type: "thinking", delta: chunk.message.thinking };
          if (chunk.message?.content) yield { type: "text", delta: chunk.message.content };
          for (const toolCall of chunk.message?.tool_calls ?? []) {
            const callId = `${idPrefix}-${toolCounter++}`;
            const { name } = toolCall.function;
            yield { type: "tool_start", callId, name };
            const args = parseToolArguments(toolCall.function.arguments);
            if (!args) {
              yield {
                type: "tool_error",
                callId,
                name,
                reason: "invalid_tool_arguments",
                message: `Received malformed JSON arguments for "${name}".`,
                argsText: stringifyJson(toolCall.function.arguments),
              };
              continue;
            }
            yield { type: "tool_call", callId, name, args };
          }
          if (chunk.done) {
            sawDone = true;
            yield {
              type: "usage",
              usage: usageFromResponse(chunk, options),
              finishReason: finishReasonFrom(chunk, toolCounter > 0),
            };
          }
        }
        if (!sawDone) yield streamEndedError("ollama");
      } finally {
        cleanupExternalAbort();
        controller.abort();
        await reader.cancel().catch(() => {});
      }
      };
      return normalizeProviderStream(raw(), { suppressTextAfterMalformedTool: true });
    },
  };
};
