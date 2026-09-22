export type InputFilePart = { type: "file"; data: string; mediaType: string };
export type ContentPart = string | { type: "text"; text: string } | InputFilePart;

export type TextBlock = {
  type: "text";
  text: string;
};

export type ThinkingBlock = {
  type: "thinking";
  thinking: string;
};

export type ToolCallBlock = {
  type: "tool_call";
  id: string;
  name: string;
  args: Record<string, unknown>;
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
};

export type Message = UserMessage | AssistantMessage | ToolResultMessage;

export type Usage = {
  input: number;
  output: number;
  cacheRead?: number;
  total: number;
  creditsUsed?: number;
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
    | "approval_denied";
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

export type ProviderFamily =
  | "openai-compatible"
  | "ollama"
  | "anthropic"
  | "mistral"
  | "gemini";

export type ProviderCapabilities = {
  streaming: boolean;
  tools: boolean;
  images: boolean;
  thinking: boolean;
  usage: boolean;
  structuredOutput?: boolean;
};

export type GenerateRequest = {
  systemPrompt?: string;
  messages: Message[];
  tools?: ToolSpec[];
  responseFormat?: ResponseFormat;
  signal?: AbortSignal;
  temperature?: number;
  maxOutputTokens?: number;
  /**
   * Ask the provider to skip (or minimize) internal reasoning for this call.
   * Useful for simple generative tasks where reasoning tokens would otherwise
   * consume the entire output budget. Provider-specific mapping:
   * - openai-compatible: sets `reasoning_effort: "low"`
   * - gemini: sets `thinkingConfig.thinkingBudget: 0`
   * - anthropic/mistral/ollama/vllm: no-op (reasoning is opt-in or absent)
   */
  disableReasoning?: boolean;
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
  | { type: "text"; delta: string }
  | { type: "thinking"; delta: string }
  | { type: "tool_start"; callId: string; name: string }
  | { type: "tool_delta"; callId: string; argsDelta: string }
  | { type: "tool_call"; callId: string; name: string; args: Record<string, unknown> }
  | ({ type: "tool_error" } & Omit<ToolStreamIssue, "kind">)
  | ({ type: "tool_cancel" } & Omit<ToolStreamIssue, "kind">)
  | { type: "usage"; usage: Usage; finishReason?: AssistantStopReason }
  | { type: "timeout"; scope: "provider_first_byte" | "provider_idle"; message: string; retryable: boolean }
  | { type: "error"; error: string; retryable: boolean; contextOverflow?: boolean; overflowRatio?: number };

export type ProviderTimeouts = {
  firstByteMs?: number;
  idleMs?: number;
};

export type Provider = {
  name: string;
  family: ProviderFamily;
  model: string;
  contextWindow?: number;
  capabilities: ProviderCapabilities;
  stream(request: GenerateRequest): AsyncIterable<StreamEvent>;
  complete(request: GenerateRequest): Promise<GenerateResult>;
};

export type OpenAICompat = {
  toolCallIdPolicy?: "passthrough" | "strict9";
  supportsUsageInStreaming?: boolean;
  requiresToolResultName?: boolean;
  requiresAssistantAfterToolResult?: boolean;
  /** Defaults to details, then reasoning/reasoning_content. "text" prefers the text fields; "none" disables thinking. */
  thinkingFormat?: "none" | "reasoning_details" | "text";
  maxTokensField?: "max_tokens" | "max_completion_tokens";
  structuredOutput?: "response_format" | "vllm_structured_outputs" | false;
};

export type OpenAICompatibleConfig = {
  name: string;
  model: string;
  baseURL: string;
  apiKey?: string;
  contextWindow?: number;
  compat?: OpenAICompat;
  timeouts?: ProviderTimeouts;
  temperature?: number;
  creditsPerInputToken?: number;
  creditsPerOutputToken?: number;
  headers?: Record<string, string>;
};
