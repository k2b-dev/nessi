import { formatConnectionError, isRetryableStatus, normalizeHttpError, streamEndedError } from "../shared/errors.js";
import { assertOnlySupportedFiles, buildAssistantMessage } from "../shared/messages.js";
import { ensureRecord, safeJsonParse, stringifyJson } from "../shared/json.js";
import { resolveReasoning, withExtraBody } from "../shared/request-options.js";
import { openSSEStream } from "../shared/stream-helpers.js";
import { normalizeProviderStream } from "../shared/tool-stream-normalizer.js";
import { createStrictToolCallIdFactory } from "../shared/tool-call-ids.js";
import { toOpenAITools } from "../shared/tools.js";
import { applyCredits, makeUsage } from "../shared/usage.js";
import type {
  AssistantStopReason,
  GenerateRequest,
  GenerateResult,
  Message,
  OpenAICompatibleConfig,
  Provider,
  RawStreamEvent,
  StreamEvent,
  ToolCallBlock,
  Usage,
} from "../types.js";

type OAIMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | OAIContentPart[] | null;
  tool_calls?: OAIToolCall[];
  tool_call_id?: string;
  name?: string;
};

type OAIContentPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } };

type OAIToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

type SSEChunk = {
  id?: string;
  /** Some gateways (e.g. OpenRouter) report failures after the response started as an error chunk. */
  error?: { message?: string; code?: number | string };
  choices?: Array<{
    index: number;
    delta: {
      content?: string | null;
      reasoning?: string | null;
      reasoning_content?: string | null;
      reasoning_details?: Array<{
        type?: string;
        text?: string;
        summary?: string;
      }>;
      tool_calls?: Array<{
        index: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    message?: {
      role?: string;
      content?: string | null;
      tool_calls?: Array<{
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
};

type OpenAIStreamDelta = NonNullable<NonNullable<SSEChunk["choices"]>[number]>["delta"];

const normalizeToolCallIds = (mode: OpenAICompatibleConfig["compat"]) =>
  mode?.toolCallIdPolicy === "strict9";

const mapFinishReason = (reason: string | null | undefined, hasTools: boolean): AssistantStopReason => {
  if (reason === "tool_calls") return "tool_use";
  if (reason === "length") return "max_tokens";
  if (reason === "content_filter") return "error";
  if (hasTools) return "tool_use";
  return "stop";
};

const responseFormatName = (name: string | undefined) => {
  const safe = (name ?? "structured_output").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  return safe || "structured_output";
};

const applyResponseFormat = (
  body: Record<string, unknown>,
  request: GenerateRequest,
  config: OpenAICompatibleConfig,
) => {
  if (!request.responseFormat) return;
  const mode = config.compat?.structuredOutput ?? "response_format";
  if (mode === false) return;
  if (mode === "vllm_structured_outputs") {
    body.structured_outputs = { json: request.responseFormat.schema };
    return;
  }
  body.response_format = {
    type: "json_schema",
    json_schema: {
      name: responseFormatName(request.responseFormat.name),
      schema: request.responseFormat.schema,
      strict: true,
    },
  };
};

const convertMessages = (messages: Message[], systemPrompt: string | undefined, config: OpenAICompatibleConfig) => {
  const result: OAIMessage[] = [];
  const strictIds = normalizeToolCallIds(config.compat);
  const makeStrictId = createStrictToolCallIdFactory();
  const pendingToolIds = new Map<string, string[]>();

  if (systemPrompt) result.push({ role: "system", content: systemPrompt });

  for (const message of messages) {
    if (message.role === "user") {
      assertOnlySupportedFiles(message.content, true, config.name);
      const parts: OAIContentPart[] = [];
      for (const part of message.content) {
        if (typeof part === "string") parts.push({ type: "text", text: part });
        else if (part.type === "text") parts.push({ type: "text", text: part.text });
        else {
          parts.push({
            type: "image_url",
            image_url: { url: `data:${part.mediaType};base64,${part.data}` },
          });
        }
      }
      if (parts.length === 1 && parts[0]?.type === "text") result.push({ role: "user", content: parts[0].text });
      else result.push({ role: "user", content: parts });
      continue;
    }

    if (message.role === "assistant") {
      let text = "";
      const toolCalls: OAIToolCall[] = [];
      for (const block of message.content) {
        if (block.type === "text") text += block.text;
        else if (block.type === "tool_call") {
          const mappedId = strictIds ? makeStrictId(block.id) : block.id;
          if (strictIds) {
            const queue = pendingToolIds.get(block.id) ?? [];
            queue.push(mappedId);
            pendingToolIds.set(block.id, queue);
          }
          toolCalls.push({
            id: mappedId,
            type: "function",
            function: { name: block.name, arguments: stringifyJson(block.args) },
          });
        }
      }
      const out: OAIMessage = { role: "assistant", content: text || null };
      if (toolCalls.length > 0) out.tool_calls = toolCalls;
      result.push(out);
      continue;
    }

    let toolCallId = message.callId;
    if (strictIds) {
      const queue = pendingToolIds.get(message.callId);
      const mapped = queue?.shift();
      if (!mapped) continue;
      toolCallId = mapped;
      if (queue && queue.length === 0) pendingToolIds.delete(message.callId);
    }
    const toolMessage: OAIMessage = {
      role: "tool",
      tool_call_id: toolCallId,
      content: stringifyJson(message.result),
    };
    if (config.compat?.requiresToolResultName) toolMessage.name = message.name;
    result.push(toolMessage);
  }

  return result;
};

const usageFromChunk = (chunk: SSEChunk, config: OpenAICompatibleConfig): Usage | undefined => {
  if (!chunk.usage) return undefined;
  return applyCredits(
    makeUsage(chunk.usage.prompt_tokens ?? 0, chunk.usage.completion_tokens ?? 0),
    config.creditsPerInputToken,
    config.creditsPerOutputToken,
  );
};

const thinkingFromDelta = (delta: OpenAIStreamDelta, config: OpenAICompatibleConfig) => {
  if (config.compat?.thinkingFormat === "none") return "";
  const text = delta.reasoning || delta.reasoning_content || "";
  const details = (delta.reasoning_details ?? [])
    .map((detail) => detail.text ?? detail.summary ?? "")
    .join("");
  // Select one representation per delta; providers may send the same text in multiple fields.
  return config.compat?.thinkingFormat === "text" ? text || details : details || text;
};

const parseCompletionResponse = async (response: Response, config: OpenAICompatibleConfig): Promise<GenerateResult> => {
  const payload = safeJsonParse<SSEChunk>(await response.text());
  if (!payload) throw new Error(`${config.name} returned invalid JSON.`);

  const choice = payload.choices?.[0];
  const message = choice?.message;
  const content = message?.content ?? "";
  const toolCalls: ToolCallBlock[] = (message?.tool_calls ?? []).map((call, index) => ({
    type: "tool_call",
    id: call.id ?? `${config.name}-${index}`,
    name: call.function?.name ?? "",
    args: ensureRecord(safeJsonParse(call.function?.arguments ?? "{}")),
  }));

  const usage = usageFromChunk(payload, config);
  const finishReason = mapFinishReason(choice?.finish_reason, toolCalls.length > 0);

  return {
    message: buildAssistantMessage(config.model, content ?? "", "", toolCalls, usage, finishReason),
    usage,
    finishReason,
    providerMeta: {
      model: config.model,
      requestId: response.headers.get("x-request-id") ?? response.headers.get("request-id") ?? undefined,
    },
  };
};

/**
 * Sends the reasoning effort in the configured wire format. The deprecated `disableReasoning`
 * keeps its original `reasoning_effort: "low"` mapping.
 */
const applyReasoning = (body: Record<string, unknown>, request: GenerateRequest, config: OpenAICompatibleConfig) => {
  const reasoning = resolveReasoning(request, config);
  if (reasoning.legacyDisable) {
    body.reasoning_effort = "low";
    return;
  }
  if (reasoning.effort === undefined) return;
  if (config.compat?.reasoningFormat === "openrouter") body.reasoning = { effort: reasoning.effort };
  else body.reasoning_effort = reasoning.effort;
};

export const openAICompatible = (config: OpenAICompatibleConfig): Provider => {
  const baseURL = config.baseURL.replace(/\/+$/, "");
  const contextWindow = config.contextWindow ?? 128_000;
  const resolveTemperature = (request: GenerateRequest) => request.temperature ?? config.temperature;

  const provider: Provider = {
    name: config.name,
    family: "openai-compatible",
    model: config.model,
    contextWindow,
    capabilities: {
      streaming: true,
      tools: true,
      images: true,
      thinking: config.compat?.thinkingFormat !== "none",
      usage: true,
      structuredOutput: config.compat?.structuredOutput !== undefined && config.compat.structuredOutput !== false,
    },

    async complete(request: GenerateRequest): Promise<GenerateResult> {
      const messages = convertMessages(request.messages, request.systemPrompt, config);
      const tools = request.tools?.length ? toOpenAITools(request.tools) : undefined;
      const body: Record<string, unknown> = {
        model: config.model,
        messages,
        stream: false,
      };
      if (tools) body.tools = tools;
      applyResponseFormat(body, request, config);
      if (request.maxOutputTokens !== undefined) {
        body[config.compat?.maxTokensField ?? "max_completion_tokens"] = request.maxOutputTokens;
      }
      applyReasoning(body, request, config);
      const temperature = resolveTemperature(request);
      if (temperature !== undefined) body.temperature = temperature;

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...config.headers,
      };
      if (config.apiKey) headers.Authorization = `Bearer ${config.apiKey}`;

      const response = await fetch(`${baseURL}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify(withExtraBody(body, request, config)),
        signal: request.signal,
      }).catch((error: unknown) => {
        throw new Error(formatConnectionError(config.name, error));
      });

      if (!response.ok) {
        const normalized = await normalizeHttpError(config.name, response);
        throw new Error(normalized.error);
      }

      return parseCompletionResponse(response, config);
    },

    stream(request: GenerateRequest): AsyncIterable<StreamEvent> {
      const raw = async function* (): AsyncIterable<RawStreamEvent> {
      const messages = convertMessages(request.messages, request.systemPrompt, config);
      const tools = request.tools?.length ? toOpenAITools(request.tools) : undefined;

      const body: Record<string, unknown> = {
        model: config.model,
        messages,
        stream: true,
      };
      if (config.compat?.supportsUsageInStreaming !== false) {
        body.stream_options = { include_usage: true };
      }
      if (tools) body.tools = tools;
      applyResponseFormat(body, request, config);
      const temperature = resolveTemperature(request);
      if (temperature !== undefined) body.temperature = temperature;
      if (request.maxOutputTokens !== undefined) {
        body[config.compat?.maxTokensField ?? "max_completion_tokens"] = request.maxOutputTokens;
      }
      applyReasoning(body, request, config);

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...config.headers,
      };
      if (config.apiKey) headers.Authorization = `Bearer ${config.apiKey}`;

      const result = await openSSEStream(
        `${baseURL}/chat/completions`,
        headers,
        withExtraBody(body, request, config),
        config.name,
        request.signal,
        contextWindow,
        config.timeouts,
      );

      if (!result.ok) {
        yield result.error;
        return;
      }

      type ToolBuffer = { callId: string; name: string; argsBuffer: string; started: boolean };
      const toolBuffers = new Map<number, ToolBuffer>();
      const pendingToolCalls: ToolBuffer[] = [];
      let latestUsage: Usage | undefined;
      let latestFinishReason: AssistantStopReason | undefined;
      let sawDone = false;
      const startToolCall = function* (buffer: ToolBuffer) {
        if (buffer.started || !buffer.name.trim()) return;
        buffer.started = true;
        yield { type: "tool_start" as const, callId: buffer.callId, name: buffer.name };
        if (buffer.argsBuffer) yield { type: "tool_delta" as const, callId: buffer.callId, argsDelta: buffer.argsBuffer };
      };
      const flushToolCalls = function* (): Generator<RawStreamEvent> {
        for (const buffer of pendingToolCalls) {
          yield* startToolCall(buffer);
          yield {
            type: "tool_call",
            callId: buffer.callId,
            name: buffer.name,
            args: ensureRecord(safeJsonParse(buffer.argsBuffer || "{}")),
          };
        }
        toolBuffers.clear();
        pendingToolCalls.length = 0;
      };

      for await (const event of result.events) {
        if (event.data === "[DONE]") {
          sawDone = true;
          break;
        }
        const chunk = safeJsonParse<SSEChunk>(event.data);
        if (!chunk) continue;
        if (chunk.error) {
          const code = chunk.error.code;
          yield {
            type: "error",
            error: `${config.name} stream error: ${chunk.error.message ?? "unknown error"}`,
            retryable: typeof code === "number" ? isRetryableStatus(code) : true,
          };
          return;
        }

        const choice = chunk.choices?.[0];
        const usage = usageFromChunk(chunk, config);
        if (!choice) {
          if (usage) {
            latestUsage = usage;
            yield { type: "usage", usage };
          }
          continue;
        }
        const delta = choice.delta;

        if (delta.content) yield { type: "text", delta: delta.content };
        const thinking = thinkingFromDelta(delta, config);
        if (thinking) yield { type: "thinking", delta: thinking };

        if (delta.tool_calls) {
          for (const toolCall of delta.tool_calls) {
            const current = toolBuffers.get(toolCall.index);
            // Some servers reuse one index for parallel calls; a new id starts a new call.
            const existing = current && (!toolCall.id || toolCall.id === current.callId) ? current : undefined;
            if (!existing) {
              const callId = toolCall.id ?? `${config.name}-${toolCall.index}`;
              const name = toolCall.function?.name ?? "";
              const argsDelta = toolCall.function?.arguments ?? "";
              const buffer = {
                callId,
                name,
                argsBuffer: argsDelta,
                started: false,
              };
              toolBuffers.set(toolCall.index, buffer);
              pendingToolCalls.push(buffer);
              yield* startToolCall(buffer);
            } else {
              if (toolCall.function?.name) existing.name = toolCall.function.name;
              const argsDelta = toolCall.function?.arguments ?? "";
              if (argsDelta) existing.argsBuffer += argsDelta;
              const wasStarted = existing.started;
              yield* startToolCall(existing);
              if (wasStarted && argsDelta) {
                yield { type: "tool_delta", callId: existing.callId, argsDelta };
              }
            }
          }
        }

        if (choice.finish_reason === "tool_calls") {
          yield* flushToolCalls();
        }
        if (choice.finish_reason) latestFinishReason = mapFinishReason(choice.finish_reason, false);
        if (usage) {
          latestUsage = usage;
          yield { type: "usage", usage };
        }
      }

      if (!latestFinishReason && !sawDone) {
        yield streamEndedError(config.name);
        return;
      }

      if (pendingToolCalls.length > 0) {
        latestFinishReason = "tool_use";
        yield* flushToolCalls();
      }

      yield {
        type: "usage",
        usage: latestUsage ?? makeUsage(),
        finishReason: latestFinishReason ?? "stop",
      };
      };
      return normalizeProviderStream(raw(), { suppressTextAfterMalformedTool: true });
    },
  };

  return provider;
};
