import { buildRequest, errorEnvelope, invalidResponseError, malformedRecordError, normalizeHttpError, ProviderError, providerError, serializeRequest, streamEndedError } from "../shared/errors.js";
import { fetchCompleteResponse } from "../shared/complete-response.js";
import { appendAssistantContentBlock, assertOnlySupportedFiles, buildAssistantMessageFromContent } from "../shared/messages.js";
import { ensureRecord, isRecord, malformedToolArgumentsError, parseJsonRecord, parseToolArguments, safeJsonParse, stringifyJson, toolArgumentsText } from "../shared/json.js";
import { resolveReasoning, withExtraBody } from "../shared/request-options.js";
import { openSSEStream } from "../shared/stream-helpers.js";
import { normalizeProviderStream } from "../shared/tool-stream-normalizer.js";
import { createStrictToolCallIdFactory, fallbackToolCallPrefix } from "../shared/tool-call-ids.js";
import { toOpenAITools } from "../shared/tools.js";
import { makeUsage } from "../shared/usage.js";
import type {
  AssistantContentBlock,
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
  reasoning_details?: Record<string, unknown>[];
  tool_calls?: OAIToolCall[];
  tool_call_id?: string;
  name?: string;
};

type OAIContentPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } } | OAIThinkingChunk;

/** Typed reasoning chunk some endpoints (Mistral Magistral) use inside `content`. */
type OAIThinkingChunk = { type: "thinking"; thinking: Array<{ type: "text"; text: string }> | string; signature?: string };

/** `content` as a string, or as typed text and reasoning chunks. */
type OAIResponseContent = string | null | Array<{ type: "text"; text: string } | OAIThinkingChunk | { type: string }>;

const chunkThinking = (chunk: OAIThinkingChunk) =>
  typeof chunk.thinking === "string" ? chunk.thinking : chunk.thinking.map((part) => part.text ?? "").join("");

/** Content blocks in response order; reasoning chunks keep their signature, or are dropped without `withThinking`. */
const contentBlocks = (content: OAIResponseContent | undefined, withThinking: boolean): AssistantContentBlock[] => {
  const blocks: AssistantContentBlock[] = [];
  if (typeof content === "string") {
    if (content) appendAssistantContentBlock(blocks, { type: "text", text: content });
    return blocks;
  }
  for (const chunk of content ?? []) {
    if (chunk.type === "text" && "text" in chunk) appendAssistantContentBlock(blocks, { type: "text", text: chunk.text });
    else if (chunk.type === "thinking" && "thinking" in chunk && withThinking) {
      appendAssistantContentBlock(blocks, {
        type: "thinking",
        thinking: chunkThinking(chunk),
        ...(chunk.signature !== undefined ? { signature: chunk.signature } : {}),
      });
    }
  }
  return blocks;
};

type OAIToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

/** OpenRouter-style reasoning item: reasoning.text, reasoning.summary or reasoning.encrypted. */
type ReasoningDetail = Record<string, unknown> & { type?: string; text?: string; summary?: string; index?: number };

const readableReasoning = (detail: ReasoningDetail) =>
  (typeof detail.text === "string" ? detail.text : "") || (typeof detail.summary === "string" ? detail.summary : "");

type SSEChunk = {
  id?: string;
  /** Some gateways (e.g. OpenRouter) report failures after the response started as an error chunk. */
  error?: { message?: string; code?: number | string };
  choices?: Array<{
    index: number;
    delta: {
      content?: OAIResponseContent;
      refusal?: string | null;
      reasoning?: string | null;
      reasoning_content?: string | null;
      reasoning_details?: ReasoningDetail[];
      tool_calls?: Array<{
        index: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    message?: {
      role?: string;
      content?: OAIResponseContent;
      refusal?: string | null;
      reasoning?: string | null;
      reasoning_content?: string | null;
      reasoning_details?: ReasoningDetail[];
      tool_calls?: Array<{
        id?: string;
        type?: string;
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    /** Includes `prompt_tokens_details.cached_tokens`. */
    prompt_tokens?: number;
    prompt_tokens_details?: { cached_tokens?: number } | null;
    completion_tokens?: number;
    total_tokens?: number;
  };
};

type OpenAIStreamDelta = NonNullable<NonNullable<SSEChunk["choices"]>[number]>["delta"];

const normalizeToolCallIds = (mode: OpenAICompatibleConfig["compat"]) =>
  mode?.toolCallIdPolicy === "strict9";

const mapFinishReason = (reason: string | null | undefined, hasTools: boolean): AssistantStopReason => {
  if (reason === "tool_calls") return "tool_use";
  if (reason === "length" || reason === "model_length") return "max_tokens";
  // The provider stopped the answer itself (filter or gateway error); its tool calls must not run.
  if (reason === "content_filter" || reason === "error") return "error";
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
      // `content_chunks` endpoints want their own reasoning back as typed chunks next to the text.
      const ownChunks = config.compat?.thinkingFormat === "content_chunks"
        && message.provider === config.name
        && message.content.some((block) => block.type === "thinking");
      const chunks: OAIContentPart[] = [];
      let text = "";
      const toolCalls: OAIToolCall[] = [];
      for (const block of message.content) {
        if (block.type === "thinking" && ownChunks) {
          chunks.push({
            type: "thinking",
            thinking: [{ type: "text", text: block.thinking }],
            ...(block.signature !== undefined ? { signature: block.signature } : {}),
          });
        } else if (block.type === "text") {
          text += block.text;
          if (ownChunks && block.text) chunks.push({ type: "text", text: block.text });
        } else if (block.type === "tool_call") {
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
      const out: OAIMessage = { role: "assistant", content: ownChunks ? chunks : text || null };
      // Reasoning items go back unchanged, and only to the provider that produced them.
      const details = message.provider === config.name
        ? message.content.flatMap((block) => (block.type === "thinking" && block.details ? block.details : []))
        : [];
      if (details.length > 0) out.reasoning_details = details;
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

const usageFromChunk = (chunk: SSEChunk): Usage | undefined => {
  if (!chunk.usage) return undefined;
  const usage = makeUsage(chunk.usage.prompt_tokens ?? 0, chunk.usage.completion_tokens ?? 0);
  const cacheRead = chunk.usage.prompt_tokens_details?.cached_tokens;
  if (typeof cacheRead === "number") usage.cacheRead = cacheRead;
  return usage;
};

const thinkingFromDelta = (delta: OpenAIStreamDelta, config: OpenAICompatibleConfig) => {
  if (config.compat?.thinkingFormat === "none") return "";
  const text = delta.reasoning || delta.reasoning_content || "";
  const details = (delta.reasoning_details ?? []).map(readableReasoning).join("");
  // Select one representation per delta; providers may send the same text in multiple fields.
  return config.compat?.thinkingFormat === "text" ? text || details : details || text;
};

const parseCompletionResponse = async (response: Response, config: OpenAICompatibleConfig): Promise<GenerateResult> => {
  const payload = parseJsonRecord<SSEChunk>(await response.text());
  if (!payload) throw invalidResponseError(config.name);
  // Some gateways answer HTTP 200 with an error body.
  const envelope = errorEnvelope(config.name, payload);
  if (envelope) throw providerError(envelope);
  const choice = payload.choices?.[0];
  // An empty message is a valid answer; a choice with neither a message nor a finish reason is not.
  if (!choice || (!isRecord(choice.message) && !choice.finish_reason)) {
    throw new ProviderError(`${config.name} returned no choices.`, { kind: "provider_error", retryable: true });
  }
  const message = choice?.message;
  const withThinking = config.compat?.thinkingFormat !== "none";
  const answer = contentBlocks(message?.content, withThinking);
  const thinking: AssistantContentBlock[] = [];
  // Typed reasoning chunks carry the signature; the plain fields would only repeat their text.
  if (withThinking && !answer.some((block) => block.type === "thinking")) {
    if (message?.reasoning_details?.length && config.compat?.thinkingFormat !== "text") {
      const details = message.reasoning_details;
      thinking.push({ type: "thinking", thinking: details.map(readableReasoning).join(""), details });
    } else {
      const text = message?.reasoning || message?.reasoning_content || "";
      if (text) thinking.push({ type: "thinking", thinking: text });
    }
  }
  const idPrefix = fallbackToolCallPrefix(config.name);
  const toolCalls: ToolCallBlock[] = (message?.tool_calls ?? []).map((call, index) => {
    const name = call.function?.name ?? "";
    const args = parseToolArguments(call.function?.arguments);
    if (!args) throw malformedToolArgumentsError(config.name, name);
    return { type: "tool_call", id: call.id ?? `${idPrefix}-${index}`, name, args };
  });

  const usage = usageFromChunk(payload);
  // A refusal is a provider-side stop: it is shown as text, and tool calls in it must not run.
  const refusal = message?.refusal || "";
  const finishReason = refusal ? "error" : mapFinishReason(choice?.finish_reason, toolCalls.length > 0);
  if (refusal) appendAssistantContentBlock(answer, { type: "text", text: refusal });

  return {
    message: buildAssistantMessageFromContent(
      config.model,
      [...thinking, ...answer, ...toolCalls],
      usage,
      finishReason,
      config.name,
    ),
    usage,
    finishReason,
    providerMeta: {
      model: config.model,
      requestId: response.headers.get("x-request-id") ?? response.headers.get("request-id") ?? undefined,
    },
  };
};

/** Sends the reasoning effort in the configured wire format. */
const applyReasoning = (body: Record<string, unknown>, request: GenerateRequest, config: OpenAICompatibleConfig) => {
  const effort = resolveReasoning(request, config);
  if (effort === undefined) return;
  if (config.compat?.reasoningFormat === "openrouter") body.reasoning = { effort };
  else body.reasoning_effort = effort;
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
      ...(config.compat?.structuredOutput === "vllm_structured_outputs" ? { strictStructuredOutput: false } : {}),
    },

    async complete(request: GenerateRequest): Promise<GenerateResult> {
      const messages = buildRequest(config.name, () => convertMessages(request.messages, request.systemPrompt, config));
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

      const response = await fetchCompleteResponse(`${baseURL}/chat/completions`, {
        method: "POST",
        headers,
        body: serializeRequest(config.name, withExtraBody(body, request, config)),
        signal: request.signal,
      }, config.name, config.timeouts);

      if (!response.ok) {
        const normalized = await normalizeHttpError(config.name, response);
        throw providerError(normalized);
      }

      return parseCompletionResponse(response, config);
    },

    stream(request: GenerateRequest): AsyncIterable<StreamEvent> {
      const raw = async function* (): AsyncIterable<RawStreamEvent> {
      const messages = buildRequest(config.name, () => convertMessages(request.messages, request.systemPrompt, config));
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
      const idPrefix = fallbackToolCallPrefix(config.name);
      let sawDone = false;
      // Reasoning items arrive in pieces keyed by `index`, sometimes with the signature after the
      // answer started; they are completed over the whole stream and attached once at the end.
      const reasoningItems = new Map<number, ReasoningDetail>();
      const collectReasoningDetail = (detail: ReasoningDetail, position: number) => {
        const index = typeof detail.index === "number" ? detail.index : position;
        const current = reasoningItems.get(index);
        if (!current) {
          reasoningItems.set(index, { ...detail });
          return;
        }
        for (const [key, value] of Object.entries(detail)) {
          const previous = current[key];
          if ((key === "text" || key === "summary" || key === "data") && typeof value === "string" && typeof previous === "string") {
            current[key] = previous + value;
          } else if (value !== undefined && value !== null && value !== "") {
            current[key] = value;
          }
        }
      };
      let deferredThinking = "";
      let refused = false;
      const flushDeferredThinking = function* (): Generator<RawStreamEvent> {
        if (!deferredThinking) return;
        const delta = deferredThinking;
        deferredThinking = "";
        yield { type: "thinking", delta };
      };
      const flushReasoningDetails = function* (): Generator<RawStreamEvent> {
        if (reasoningItems.size === 0) return;
        const details = [...reasoningItems.entries()].sort(([left], [right]) => left - right).map(([, item]) => item);
        reasoningItems.clear();
        yield { type: "thinking", delta: "", details };
      };
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
        if (!event.data.trim()) continue;
        const chunk = parseJsonRecord<SSEChunk>(event.data);
        if (!chunk) {
          yield malformedRecordError(config.name);
          return;
        }
        const envelope = errorEnvelope(`${config.name} stream`, chunk);
        if (envelope) {
          yield { type: "error", ...envelope };
          return;
        }

        const choice = chunk.choices?.[0];
        const usage = usageFromChunk(chunk);
        if (!choice) {
          if (usage) {
            latestUsage = usage;
            yield { type: "usage", usage };
          }
          continue;
        }
        const delta = choice.delta;

        // Typed reasoning chunks carry the signature; the plain fields would only repeat their text.
        const chunkedThinking = Array.isArray(delta.content) && delta.content.some((chunk) => chunk.type === "thinking");
        const thinking = chunkedThinking ? "" : thinkingFromDelta(delta, config);
        // Reasoning that arrives while a tool call is still streaming waits until the call is complete.
        if (thinking && pendingToolCalls.length > 0) deferredThinking += thinking;
        else if (thinking) yield { type: "thinking", delta: thinking };
        if (config.compat?.thinkingFormat !== "none" && config.compat?.thinkingFormat !== "text") {
          (delta.reasoning_details ?? []).forEach(collectReasoningDetail);
        }
        if (typeof delta.content === "string") {
          if (delta.content) yield { type: "text", delta: delta.content };
        } else {
          for (const chunk of delta.content ?? []) {
            if (chunk.type === "text" && "text" in chunk) {
              if (chunk.text) yield { type: "text", delta: chunk.text };
            } else if (chunk.type === "thinking" && "thinking" in chunk && config.compat?.thinkingFormat !== "none") {
              const text = chunkThinking(chunk);
              if (text) yield { type: "thinking", delta: text };
              if (chunk.signature) yield { type: "thinking", delta: "", signature: chunk.signature };
            }
          }
        }
        if (delta.refusal) {
          refused = true;
          yield { type: "text", delta: delta.refusal };
        }

        if (delta.tool_calls) {
          for (const toolCall of delta.tool_calls) {
            const current = toolBuffers.get(toolCall.index);
            // Some servers reuse one index for parallel calls; a new id starts a new call.
            const existing = current && (!toolCall.id || toolCall.id === current.callId) ? current : undefined;
            if (!existing) {
              const callId = toolCall.id ?? `${idPrefix}-${toolCall.index}`;
              const name = toolCall.function?.name ?? "";
              const argsDelta = toolArgumentsText(toolCall.function?.arguments);
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
              const argsDelta = toolArgumentsText(toolCall.function?.arguments);
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
          yield* flushDeferredThinking();
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

      if (refused) latestFinishReason = "error";
      if (pendingToolCalls.length > 0) {
        // A provider-side stop or a length cut stays visible even with pending calls.
        if (latestFinishReason !== "error" && latestFinishReason !== "max_tokens") latestFinishReason = "tool_use";
        yield* flushToolCalls();
        yield* flushDeferredThinking();
      }
      yield* flushReasoningDetails();

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
