import { formatConnectionError, normalizeHttpError, streamEndedError } from "../shared/errors.js";
import { assertOnlySupportedFiles, buildAssistantMessage } from "../shared/messages.js";
import { parseNDJSON } from "../shared/ndjson.js";
import { ensureRecord, safeJsonParse, stringifyJson } from "../shared/json.js";
import { normalizeProviderStream } from "../shared/tool-stream-normalizer.js";
import { toOllamaTools } from "../shared/tools.js";
import { applyCredits, makeUsage } from "../shared/usage.js";
import type {
  GenerateRequest,
  GenerateResult,
  Message,
  Provider,
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
    tool_calls?: Array<{ function: { name: string; arguments: Record<string, unknown> } }>;
  };
  done: boolean;
  done_reason?: string;
  error?: string;
  prompt_eval_count?: number;
  eval_count?: number;
};

export type OllamaOptions = {
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

const toolCallsFromResponse = (response: OllamaResponse): ToolCallBlock[] =>
  (response.message?.tool_calls ?? []).map((toolCall, index) => ({
    type: "tool_call",
    id: `ollama-${index}`,
    name: toolCall.function.name,
    args: toolCall.function.arguments,
  }));

const generationOptions = (request: GenerateRequest, options?: OllamaOptions) => {
  const result: Record<string, unknown> = {};
  const temperature = request.temperature ?? options?.temperature;
  if (temperature !== undefined) result.temperature = temperature;
  if (request.maxOutputTokens !== undefined) result.num_predict = request.maxOutputTokens;
  return Object.keys(result).length > 0 ? result : undefined;
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
      thinking: false,
      usage: true,
      structuredOutput: true,
    },

    async complete(request: GenerateRequest): Promise<GenerateResult> {
      const body: Record<string, unknown> = {
        model,
        messages: convertMessages(request.messages, request.systemPrompt),
        stream: false,
      };
      if (request.tools?.length) body.tools = toOllamaTools(request.tools);
      if (request.responseFormat) body.format = request.responseFormat.schema;
      const generation = generationOptions(request, options);
      if (generation) body.options = generation;

      const response = await fetch(`${baseURL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: request.signal,
      }).catch((error: unknown) => {
        throw new Error(formatConnectionError("ollama", error));
      });

      if (!response.ok) {
        const normalized = await normalizeHttpError("ollama", response);
        throw new Error(normalized.error);
      }

      const payload = safeJsonParse<OllamaResponse>(await response.text());
      if (!payload) throw new Error("ollama returned invalid JSON.");
      const usage = usageFromResponse(payload, options);
      const toolCalls = toolCallsFromResponse(payload);
      const finishReason = finishReasonFrom(payload, toolCalls.length > 0);

      return {
        message: buildAssistantMessage(model, payload.message?.content ?? "", "", toolCalls, usage, finishReason),
        usage,
        finishReason,
        providerMeta: { model },
      };
    },

    stream(request: GenerateRequest): AsyncIterable<StreamEvent> {
      const raw = async function* (): AsyncIterable<RawStreamEvent> {
      const body: Record<string, unknown> = {
        model,
        messages: convertMessages(request.messages, request.systemPrompt),
        stream: true,
      };
      if (request.tools?.length) body.tools = toOllamaTools(request.tools);
      if (request.responseFormat) body.format = request.responseFormat.schema;
      const generation = generationOptions(request, options);
      if (generation) body.options = generation;

      let response: Response;
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
            body: JSON.stringify(body),
            signal: controller.signal,
          }),
          new Promise<never>((_, reject) => {
            if (!options?.timeouts?.firstByteMs || options.timeouts.firstByteMs <= 0) return;
            firstByteTimeout = setTimeout(() => {
              controller.abort();
              reject({
                scope: "provider_first_byte",
                message: `ollama first byte timeout after ${options.timeouts!.firstByteMs}ms.`,
              });
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
        yield {
          type: "error",
          error: formatConnectionError("ollama", error),
          retryable: true,
        };
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
          if (chunk.message?.content) yield { type: "text", delta: chunk.message.content };
          for (const toolCall of chunk.message?.tool_calls ?? []) {
            const callId = `ollama-${toolCounter++}`;
            yield { type: "tool_start", callId, name: toolCall.function.name };
            yield {
              type: "tool_call",
              callId,
              name: toolCall.function.name,
              args: ensureRecord(toolCall.function.arguments),
            };
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
