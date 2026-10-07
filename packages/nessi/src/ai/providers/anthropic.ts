import { formatConnectionError, normalizeHttpError, streamEndedError } from "../shared/errors.js";
import { assertOnlySupportedFiles, buildAssistantMessage } from "../shared/messages.js";
import { ensureRecord, safeJsonParse, stringifyJson } from "../shared/json.js";
import { resolveReasoning, withExtraBody } from "../shared/request-options.js";
import { openSSEStream } from "../shared/stream-helpers.js";
import { normalizeProviderStream } from "../shared/tool-stream-normalizer.js";
import { toAnthropicTools } from "../shared/tools.js";
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

type AnthropicBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: { type: "base64"; media_type: string; data: string } }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

type AnthropicMessage = {
  role: "user" | "assistant";
  content: AnthropicBlock[];
};

type AnthropicResponse = {
  id?: string;
  model?: string;
  content?: Array<
    | { type: "text"; text?: string }
    | { type: "tool_use"; id: string; name: string; input?: Record<string, unknown> }
  >;
  stop_reason?: string | null;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
};

type AnthropicStreamEvent = {
  message?: {
    id?: string;
    model?: string;
    usage?: { input_tokens?: number; output_tokens?: number };
  };
  usage?: { input_tokens?: number; output_tokens?: number };
  content_block?: {
    type?: string;
    id?: string;
    name?: string;
    input?: Record<string, unknown>;
  };
  index?: number;
  delta?: {
    type?: string;
    text?: string;
    partial_json?: string;
    stop_reason?: string | null;
  };
  error?: { type?: string; message?: string };
};

// Mid-stream error types worth retrying; see https://docs.anthropic.com/en/api/errors
const RETRYABLE_STREAM_ERRORS = new Set(["overloaded_error", "rate_limit_error", "api_error", "timeout_error"]);

export type AnthropicOptions = ProviderRequestDefaults & {
  apiKey?: string;
  baseURL?: string;
  apiVersion?: string;
  contextWindow?: number;
  temperature?: number;
  maxOutputTokens?: number;
  creditsPerInputToken?: number;
  creditsPerOutputToken?: number;
  timeouts?: ProviderTimeouts;
};

const mapFinishReason = (reason: string | null | undefined, hasTools: boolean) => {
  if (reason === "tool_use") return "tool_use" as const;
  if (reason === "max_tokens") return "max_tokens" as const;
  if (hasTools) return "tool_use" as const;
  return "stop" as const;
};

const pushMessage = (messages: AnthropicMessage[], next: AnthropicMessage) => {
  const last = messages[messages.length - 1];
  if (last?.role === next.role) {
    last.content.push(...next.content);
    return;
  }
  messages.push(next);
};

const convertMessages = (messages: Message[]) => {
  const out: AnthropicMessage[] = [];
  for (const message of messages) {
    if (message.role === "user") {
      assertOnlySupportedFiles(message.content, true, "anthropic");
      const content: AnthropicBlock[] = [];
      for (const part of message.content) {
        if (typeof part === "string") content.push({ type: "text", text: part });
        else if (part.type === "text") content.push({ type: "text", text: part.text });
        else content.push({
          type: "image",
          source: { type: "base64", media_type: part.mediaType, data: part.data },
        });
      }
      pushMessage(out, { role: "user", content });
      continue;
    }

    if (message.role === "assistant") {
      const content: AnthropicBlock[] = [];
      for (const block of message.content) {
        if (block.type === "text") content.push({ type: "text", text: block.text });
        else if (block.type === "tool_call") {
          content.push({
            type: "tool_use",
            id: block.id,
            name: block.name,
            input: block.args,
          });
        }
      }
      pushMessage(out, { role: "assistant", content });
      continue;
    }

    pushMessage(out, {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: message.callId,
        content: stringifyJson(message.result),
        is_error: message.isError,
      }],
    });
  }
  return out;
};

const resolveTemperature = (request: GenerateRequest, options?: AnthropicOptions) =>
  request.temperature ?? options?.temperature;

const usageFromValue = (
  usage: { input_tokens?: number; output_tokens?: number } | undefined,
  options?: AnthropicOptions,
) =>
  applyCredits(
    makeUsage(usage?.input_tokens ?? 0, usage?.output_tokens ?? 0),
    options?.creditsPerInputToken,
    options?.creditsPerOutputToken,
  );

const mergeUsage = (
  current: ReturnType<typeof makeUsage>,
  usage: { input_tokens?: number; output_tokens?: number } | undefined,
  options?: AnthropicOptions,
) => {
  if (!usage) return current;
  return applyCredits(
    makeUsage(
      usage.input_tokens ?? current.input,
      usage.output_tokens ?? current.output,
    ),
    options?.creditsPerInputToken,
    options?.creditsPerOutputToken,
  );
};

const applyResponseFormat = (body: Record<string, unknown>, request: GenerateRequest) => {
  if (!request.responseFormat) return;
  body.output_config = {
    format: {
      type: "json_schema",
      schema: request.responseFormat.schema,
    },
  };
};

export const anthropic = (model: string, options?: AnthropicOptions): Provider => {
  const baseURL = (options?.baseURL ?? "https://api.anthropic.com").replace(/\/+$/, "");
  const apiVersion = options?.apiVersion ?? "2023-06-01";
  const maxOutputTokens = options?.maxOutputTokens ?? 1024;

  return {
    name: "anthropic",
    family: "anthropic",
    model,
    contextWindow: options?.contextWindow ?? 200_000,
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
        system: request.systemPrompt,
        messages: convertMessages(request.messages),
        max_tokens: request.maxOutputTokens ?? maxOutputTokens,
      };
      if (request.tools?.length) body.tools = toAnthropicTools(request.tools);
      applyResponseFormat(body, request);
      const temperature = resolveTemperature(request, options);
      if (temperature !== undefined) body.temperature = temperature;

      const response = await fetch(`${baseURL}/v1/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": options?.apiKey ?? globalThis.process?.env?.ANTHROPIC_API_KEY ?? "",
          "anthropic-version": apiVersion,
        },
        body: JSON.stringify(withExtraBody(body, request, options)),
        signal: request.signal,
      }).catch((error: unknown) => {
        throw new Error(formatConnectionError("anthropic", error));
      });

      if (!response.ok) {
        const normalized = await normalizeHttpError("anthropic", response);
        throw new Error(normalized.error);
      }

      const payload = safeJsonParse<AnthropicResponse>(await response.text());
      if (!payload) throw new Error("anthropic returned invalid JSON.");
      const text = (payload.content ?? [])
        .filter((block): block is { type: "text"; text?: string } => block.type === "text")
        .map((block) => block.text ?? "")
        .join("");
      const toolCalls: ToolCallBlock[] = (payload.content ?? [])
        .filter((block): block is { type: "tool_use"; id: string; name: string; input?: Record<string, unknown> } => block.type === "tool_use")
        .map((block) => ({
          type: "tool_call",
          id: block.id,
          name: block.name,
          args: block.input ?? {},
        }));
      const usage = usageFromValue(payload.usage, options);
      const finishReason = mapFinishReason(payload.stop_reason, toolCalls.length > 0);

      return {
        message: buildAssistantMessage(model, text, "", toolCalls, usage, finishReason),
        usage,
        finishReason,
        providerMeta: { model, requestId: payload.id },
      };
    },

    stream(request: GenerateRequest): AsyncIterable<StreamEvent> {
      const raw = async function* (): AsyncIterable<RawStreamEvent> {
      const body: Record<string, unknown> = {
        model,
        system: request.systemPrompt,
        messages: convertMessages(request.messages),
        max_tokens: request.maxOutputTokens ?? maxOutputTokens,
        stream: true,
      };
      if (request.tools?.length) body.tools = toAnthropicTools(request.tools);
      applyResponseFormat(body, request);
      const temperature = resolveTemperature(request, options);
      if (temperature !== undefined) body.temperature = temperature;

      const result = await openSSEStream(
        `${baseURL}/v1/messages`,
        {
          "Content-Type": "application/json",
          "x-api-key": options?.apiKey ?? globalThis.process?.env?.ANTHROPIC_API_KEY ?? "",
          "anthropic-version": apiVersion,
        },
        withExtraBody(body, request, options),
        "anthropic",
        request.signal,
        undefined,
        options?.timeouts,
      );

      if (!result.ok) {
        yield result.error;
        return;
      }

      const toolBuffers = new Map<number, { callId: string; name: string; argsBuffer: string }>();
      let latestUsage = makeUsage();
      let latestFinishReason: GenerateResult["finishReason"] | undefined;
      let syntheticIndex = 0;
      let sawToolCall = false;
      let sawMessageStop = false;

      for await (const event of result.events) {
        if (event.data === "[DONE]") break;
        const payload = safeJsonParse<AnthropicStreamEvent>(event.data);
        if (!payload) continue;

        if (event.event === "error") {
          const type = payload.error?.type ?? "unknown_error";
          yield {
            type: "error",
            error: `anthropic stream error (${type}): ${payload.error?.message ?? "unknown error"}`,
            retryable: RETRYABLE_STREAM_ERRORS.has(type),
          };
          return;
        }
        if (event.event === "message_stop") sawMessageStop = true;

        if (event.event === "message_start" && payload.message?.usage) {
          latestUsage = mergeUsage(latestUsage, payload.message.usage, options);
        }

        if (event.event === "content_block_start" && payload.content_block?.type === "tool_use") {
          const index = typeof payload.index === "number" ? payload.index : syntheticIndex++;
          const startInput = payload.content_block.input;
          const argsBuffer = startInput && Object.keys(startInput).length > 0 ? JSON.stringify(startInput) : "";
          toolBuffers.set(index, {
            callId: payload.content_block.id ?? `anthropic-${index}`,
            name: payload.content_block.name ?? "",
            argsBuffer,
          });
          sawToolCall = true;
          yield {
            type: "tool_start",
            callId: payload.content_block.id ?? `anthropic-${index}`,
            name: payload.content_block.name ?? "",
          };
          if (argsBuffer) {
            yield {
              type: "tool_delta",
              callId: payload.content_block.id ?? `anthropic-${index}`,
              argsDelta: argsBuffer,
            };
          }
        }

        if (event.event === "content_block_delta") {
          if (payload.delta?.type === "text_delta" && payload.delta.text) {
            yield { type: "text", delta: payload.delta.text };
          } else if (payload.delta?.type === "input_json_delta") {
            if (typeof payload.index !== "number") continue;
            const index = payload.index;
            const existing = toolBuffers.get(index);
            if (existing && payload.delta.partial_json) {
              existing.argsBuffer += payload.delta.partial_json;
              yield { type: "tool_delta", callId: existing.callId, argsDelta: payload.delta.partial_json };
            }
          }
        }

        if (event.event === "content_block_stop") {
          if (typeof payload.index !== "number") continue;
          const index = payload.index;
          const existing = toolBuffers.get(index);
          if (existing) {
            yield {
              type: "tool_call",
              callId: existing.callId,
              name: existing.name,
              args: ensureRecord(safeJsonParse(existing.argsBuffer || "{}")),
            };
            toolBuffers.delete(index);
          }
        }

        if (event.event === "message_delta" && payload.usage) {
          latestUsage = mergeUsage(latestUsage, payload.usage, options);
        }
        if (event.event === "message_delta" && payload.delta?.stop_reason) {
          latestFinishReason = mapFinishReason(payload.delta.stop_reason, sawToolCall || toolBuffers.size > 0);
        }
      }

      if (!latestFinishReason && !sawMessageStop) {
        yield streamEndedError("anthropic");
        return;
      }

      if (latestUsage.total > 0 || latestFinishReason) {
        yield { type: "usage", usage: latestUsage, finishReason: latestFinishReason };
      }
      };
      return normalizeProviderStream(raw(), { suppressTextAfterMalformedTool: true });
    },
  };
};
