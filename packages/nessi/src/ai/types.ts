export type InputFilePart = { type: "file"; data: string; mediaType: string };
export type ContentPart = string | { type: "text"; text: string } | InputFilePart;

// `signature` and `redacted` are opaque provider data. The provider that produced a block
// (see `AssistantMessage.provider`) needs them back unchanged in later requests.

export type TextBlock = {
  type: "text";
  text: string;
  /** Opaque provider signature for this text; sent back unchanged. */
  signature?: string;
};

export type ThinkingBlock = {
  type: "thinking";
  /** Readable reasoning; empty when the provider only returns a signature or encrypted data. */
  thinking: string;
  /** Opaque signature that verifies this reasoning (Mistral `thinking` chunks); sent back unchanged. */
  signature?: string;
  /** Encrypted reasoning without readable text; sent back unchanged. */
  redacted?: string;
  /** Original OpenRouter-style `reasoning_details` items behind this block, sent back as they are. */
  details?: Record<string, unknown>[];
};

export type ToolCallBlock = {
  type: "tool_call";
  id: string;
  name: string;
  args: Record<string, unknown>;
  /** Opaque provider signature for this call; sent back unchanged. */
  signature?: string;
};

export type AssistantContentBlock = TextBlock | ThinkingBlock | ToolCallBlock;

export type AssistantBlockKind = AssistantContentBlock["type"];

export type UserMessage = {
  role: "user";
  content: ContentPart[];
};

export type AssistantStopReason = "stop" | "tool_use" | "max_tokens" | "aborted" | "interrupted" | "error";

export type AssistantMessage = {
  role: "assistant";
  content: AssistantContentBlock[];
  model?: string;
  /** Name of the provider that produced the message. Providers only send signatures back to themselves. */
  provider?: string;
  usage?: Usage;
  stopReason?: AssistantStopReason;
};

export type HistoricalToolResult = {
  originLoopId: string;
  value: unknown;
};

export type ToolResultMessage = {
  role: "tool_result";
  callId: string;
  name: string;
  result: unknown;
  historicalResult?: HistoricalToolResult;
  isError?: boolean;
  /** Set on every error result nessi writes itself; code can rely on it, unlike the `result` text. */
  reason?: ToolResultReason;
};

/**
 * Why nessi answered a call with an error. `malformed_tool_call`: the provider sent a call nessi
 * could not use. `not_run`: the response ended or was cut off before its calls could run.
 * `not_returned`: the call may have run, but no result was stored. The rest name the failure of a
 * call that ran.
 */
export type ToolResultReason = ToolExecutionIssue["reason"] | "timeout" | "malformed_tool_call";

export type Message = UserMessage | AssistantMessage | ToolResultMessage;

export type Usage = {
  /** Every prompt token, including the ones read from or written to a provider cache. */
  input: number;
  output: number;
  /** Part of `input` the provider read from its prompt cache. */
  cacheRead?: number;
  /** Part of `output` the model spent on reasoning. */
  reasoning?: number;
  total: number;
};

export type JsonSchemaObject = Record<string, unknown>;

export type ResponseFormat = {
  type: "json_schema";
  name?: string;
  schema: JsonSchemaObject;
};

export type ToolStreamIssueKind = "malformed_tool_call" | "cancelled_tool_call";

export type ToolStreamIssueReason =
  | "text_during_tool_call"
  | "thinking_during_tool_call"
  | "tool_delta_without_start"
  | "missing_tool_name"
  | "invalid_tool_arguments"
  | "stream_ended_before_tool_call"
  | "provider_error_before_tool_call";

export type ToolStreamIssue = {
  kind: ToolStreamIssueKind;
  reason: ToolStreamIssueReason;
  message: string;
  callId?: string;
  name?: string;
  argsText?: string;
  textDelta?: string;
};

export type ProviderIssue = {
  kind: "provider_error";
  message: string;
  retryable: boolean;
  /** HTTP status of a rejected request. */
  status?: number;
  /** Provider error code from the body, or the runtime's connection error code. */
  code?: string;
  /** nessi could not build the request (unsupported input, unserializable values) and sent nothing. */
  notSent?: true;
  /** How long the provider asked the caller to wait (`retry-after-ms` or `retry-after`). */
  retryAfterMs?: number;
  contextOverflow?: boolean;
  overflowRatio?: number;
};

export type TimeoutIssue = {
  kind: "timeout";
  scope: "provider_first_byte" | "provider_idle" | "tool";
  message: string;
  retryable: boolean;
  callId?: string;
  name?: string;
};

export type ToolExecutionIssue = {
  kind: "tool_execution_error";
  reason:
    | "unknown_tool"
    | "input_validation_failed"
    | "output_validation_failed"
    | "execution_failed"
    | "approval_denied"
    | "not_run"
    | "not_returned";
  message: string;
  retryable: boolean;
  callId: string;
  name: string;
};

export type ToolHistoricalResultIssue = {
  kind: "tool_historical_result_error";
  message: string;
  retryable: false;
  callId: string;
  name: string;
};

export type RuntimeIssue = {
  kind: "runtime_error";
  message: string;
  retryable: boolean;
};

export type NessiIssue =
  | ToolStreamIssue
  | ProviderIssue
  | TimeoutIssue
  | ToolExecutionIssue
  | ToolHistoricalResultIssue
  | RuntimeIssue;

export type ToolSpec = {
  name: string;
  description: string;
  inputSchema: unknown;
};

export type ProviderFamily = "openai-compatible";

export type ProviderCapabilities = {
  streaming: boolean;
  tools: boolean;
  images: boolean;
  thinking: boolean;
  usage: boolean;
  structuredOutput?: boolean;
  /**
   * Native structured output follows strict JSON Schema rules: every object property is required.
   * Defaults to `true`; guided decoding such as vLLM's `structured_outputs` sets `false`.
   */
  strictStructuredOutput?: boolean;
};

/**
 * How much the model should reason. Passed to the provider unchanged, so new levels
 * work without a Nessi update; "none" turns reasoning off. Support depends on the model.
 */
export type ReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | (string & {});

/** Per-provider defaults that a request can override. */
export type ProviderRequestDefaults = {
  /** Default reasoning effort; `GenerateRequest.reasoningEffort` wins. Unset leaves the model default. */
  reasoningEffort?: ReasoningEffort;
  /**
   * Extra fields for the provider's request body, merged over what Nessi sends. Plain objects
   * merge deeply, everything else replaces. Use it for parameters Nessi does not model.
   */
  extraBody?: Record<string, unknown>;
};

export type GenerateRequest = {
  systemPrompt?: string;
  messages: Message[];
  tools?: ToolSpec[];
  responseFormat?: ResponseFormat;
  signal?: AbortSignal;
  temperature?: number;
  maxOutputTokens?: number;
  /** Reasoning effort for this call; overrides the provider's `reasoningEffort`. */
  reasoningEffort?: ReasoningEffort;
  /** Extra request body fields for this call, merged over the provider's `extraBody`. */
  extraBody?: Record<string, unknown>;
  /**
   * Called once when a request nessi sent settles: answered, rejected, failed, timed out, or
   * aborted. Observation only; an error thrown here does not change the request.
   */
  onTransport?: (report: TransportReport) => void;
};

/** What the transport saw of one request; times are `Date.now()` milliseconds. */
export type TransportReport = {
  startedAt: number;
  /** Response headers arrived; missing when no response came. */
  headersAt?: number;
  /** The first body chunk arrived. */
  firstByteAt?: number;
  status?: number;
  /** How long the provider asked the caller to wait (`retry-after-ms` or `retry-after`). */
  retryAfterMs?: number;
  /** The runtime's code when the connection failed before a response (`ECONNREFUSED`, …). */
  connectionCode?: string;
};

export type GenerateResult = {
  message: AssistantMessage;
  usage?: Usage;
  finishReason: AssistantStopReason;
  providerMeta?: {
    requestId?: string;
    model?: string;
  };
};

export type BlockStartEvent = {
  type: "block_start";
  blockId: string;
  index: number;
  kind: AssistantBlockKind;
  callId?: string;
  name?: string;
};

export type BlockDeltaEvent = {
  type: "block_delta";
  blockId: string;
  delta: string;
};

export type BlockEndEvent = {
  type: "block_end";
  blockId: string;
  index: number;
  block: AssistantContentBlock;
};

export type StreamEvent =
  | BlockStartEvent
  | BlockDeltaEvent
  | BlockEndEvent
  | { type: "issue"; issue: NessiIssue }
  | { type: "usage"; usage: Usage; finishReason?: AssistantStopReason };

export type RawStreamEvent =
  /** `signature` attaches to the current text block. */
  | { type: "text"; delta: string; signature?: string }
  /**
   * `signature`, `redacted` or `details` complete the current thinking block (or form one when none is open);
   * the next thinking delta starts a new block.
   */
  | { type: "thinking"; delta: string; signature?: string; redacted?: string; details?: Record<string, unknown>[] }
  | { type: "tool_start"; callId: string; name: string }
  | { type: "tool_delta"; callId: string; argsDelta: string }
  | { type: "tool_call"; callId: string; name: string; args: Record<string, unknown>; signature?: string }
  | ({ type: "tool_error" } & Omit<ToolStreamIssue, "kind">)
  | ({ type: "tool_cancel" } & Omit<ToolStreamIssue, "kind">)
  | { type: "usage"; usage: Usage; finishReason?: AssistantStopReason }
  | { type: "timeout"; scope: "provider_first_byte" | "provider_idle"; message: string; retryable: boolean }
  | {
    type: "error";
    error: string;
    retryable: boolean;
    status?: number;
    code?: string;
    retryAfterMs?: number;
    contextOverflow?: boolean;
    overflowRatio?: number;
  };

export type ProviderTimeouts = {
  /** Request start through the first response-body chunk, which may precede the first token. */
  firstByteMs?: number;
  /** Each gap between body chunks after the first. */
  idleMs?: number;
};

export type Provider = {
  name: string;
  family: ProviderFamily;
  model: string;
  contextWindow?: number;
  capabilities: ProviderCapabilities;
  /** The only request path. `complete(provider, request)` collects a stream into one result. */
  stream(request: GenerateRequest): AsyncIterable<StreamEvent>;
};

export type OpenAICompat = {
  toolCallIdPolicy?: "passthrough" | "strict9";
  supportsUsageInStreaming?: boolean;
  requiresToolResultName?: boolean;
  requiresAssistantAfterToolResult?: boolean;
  /**
   * Defaults to details, then reasoning/reasoning_content. "text" prefers the text fields; "none" disables
   * thinking. "content_chunks" replays the endpoint's own reasoning as typed `thinking` chunks in
   * `content` (Mistral). Typed `text` and `thinking` chunks in responses are read in every mode.
   */
  thinkingFormat?: "none" | "reasoning_details" | "text" | "content_chunks";
  maxTokensField?: "max_tokens" | "max_completion_tokens";
  /** How `reasoningEffort` is sent: OpenAI's `reasoning_effort` (default) or OpenRouter's `reasoning: { effort }`. */
  reasoningFormat?: "reasoning_effort" | "openrouter";
  structuredOutput?: "response_format" | "vllm_structured_outputs" | false;
};

export type OpenAICompatibleConfig = ProviderRequestDefaults & {
  name: string;
  model: string;
  baseURL: string;
  apiKey?: string;
  contextWindow?: number;
  compat?: OpenAICompat;
  timeouts?: ProviderTimeouts;
  temperature?: number;
  headers?: Record<string, string>;
};
