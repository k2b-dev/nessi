// ============================================================================
// nessi – Types
// ============================================================================

import type { z } from "zod";
import type {
  AssistantStopReason,
  BlockDeltaEvent,
  BlockEndEvent,
  BlockStartEvent,
  AssistantMessage,
  ContentPart,
  GenerateRequest,
  GenerateResult,
  HistoricalToolResult,
  Message,
  NessiIssue,
  Provider,
  ReasoningEffort,
  StreamEvent,
  ToolExecutionIssue,
  ToolHistoricalResultIssue,
  ToolResultMessage,
  ToolStreamIssue,
  Usage,
  UserMessage,
} from "./ai/index.js";
export type {
  AssistantContentBlock,
  AssistantBlockKind,
  AssistantMessage,
  AssistantStopReason,
  BlockDeltaEvent,
  BlockEndEvent,
  BlockStartEvent,
  ContentPart,
  JsonSchemaObject,
  HistoricalToolResult,
  Message,
  NessiIssue,
  Provider,
  ProviderIssue,
  ReasoningEffort,
  ResponseFormat,
  RuntimeIssue,
  TextBlock,
  ThinkingBlock,
  ToolCallBlock,
  ToolExecutionIssue,
  ToolHistoricalResultIssue,
  ToolResultMessage,
  ToolStreamIssue,
  ToolStreamIssueKind,
  ToolStreamIssueReason,
  ToolSpec,
  TimeoutIssue,
  Usage,
  UserMessage,
} from "./ai/index.js";

export type ProviderEvent = StreamEvent;
export type ProviderRequest = GenerateRequest;

// ----------------------------------------------------------------------------
// 1. Content
// ----------------------------------------------------------------------------

export type Input = string | ContentPart[];

// ----------------------------------------------------------------------------
// 2. Events – Bidirektional
// ----------------------------------------------------------------------------

type LoopEventFields = {
  agentId: string;
  loopId: string;
}

type TurnEventFields = LoopEventFields & {
  turnId: string;
  turnIndex: number;
}

export type ToolActionKind = "approval" | "client_tool" | "custom_approval";

export type OutboundEvent =
  | (LoopEventFields & { type: "loop_start" })
  | (TurnEventFields & { type: "turn_start"; resumed?: boolean })
  | (TurnEventFields & BlockStartEvent)
  | (TurnEventFields & BlockDeltaEvent)
  | (TurnEventFields & BlockEndEvent)
  | (TurnEventFields & { type: "usage"; usage: Usage; finishReason?: AssistantStopReason })
  | (LoopEventFields & { type: "issue"; issue: NessiIssue; turnId?: string; turnIndex?: number })
  | (TurnEventFields & { type: "tool_execution_start"; callId: string; name: string; args: unknown })
  | (TurnEventFields & {
      type: "tool_action_request";
      kind: ToolActionKind;
      callId: string;
      name: string;
      args: unknown;
      message?: string;
    })
  | (TurnEventFields & { type: "tool_execution_end"; callId: string; name: string; result: unknown; isError?: boolean })
  | (TurnEventFields & { type: "turn_end"; message: AssistantMessage })
  | (LoopEventFields & { type: "steer_applied"; message: string })
  | (LoopEventFields & { type: "compaction_start" })
  | (LoopEventFields & { type: "compaction_end" })
  | (LoopEventFields & { type: "loop_end"; reason: DoneReason; aggregate: LoopAggregate });

export type InboundEvent =
  | { type: "approval_response"; callId: string; approved: boolean }
  | { type: "tool_result"; callId: string; result: unknown };

export type DoneReason = "stop" | "no_credits" | "max_turns" | "context_overflow" | "error" | "aborted";

export type LoopToolCallAggregate = {
  callId: string;
  name: string;
  args: unknown;
  result?: unknown;
  isError?: boolean;
};

export type LoopToolIssueAggregate = ToolStreamIssue;
export type LoopIssueAggregate = NessiIssue;

export type LoopTurnAggregate = {
  message: AssistantMessage;
  usage?: Usage;
  stopReason?: AssistantMessage["stopReason"];
  toolCalls: LoopToolCallAggregate[];
  toolIssues?: LoopToolIssueAggregate[];
  issues?: LoopIssueAggregate[];
};

export type LoopTimingAggregate = {
  wallMs: number;
  totalElapsedMs: number;
  generationMs: number;
  toolExecutionMs: number;
  actionWaitMs: number;
  outputTokensPerSecond?: number;
};

export type LoopAggregate = {
  turns: LoopTurnAggregate[];
  usage?: Usage;
  timing?: LoopTimingAggregate;
  issueCount: number;
  issues: LoopIssueAggregate[];
  toolCallCount: number;
  toolErrorCount: number;
  toolIssueCount: number;
  toolMalformedCount: number;
  toolCancelledCount: number;
  toolIssues: LoopToolIssueAggregate[];
  assistantMessageCount: number;
};

// ----------------------------------------------------------------------------
// 3. Tools
// ----------------------------------------------------------------------------

export type HistoricalToolResultContext<
  TInput extends z.ZodType = z.ZodType,
  TOutput extends z.ZodType = z.ZodType,
> = {
  input: z.infer<TInput>;
  output: z.infer<TOutput>;
  callId: string;
};

export type ToolDefinition<TInput extends z.ZodType = z.ZodType, TOutput extends z.ZodType = z.ZodType> = {
  name: string;
  description: string;
  inputSchema: TInput;
  outputSchema?: TOutput;
  needsApproval?: boolean;
  timeoutMs?: number | false;
  /** Derive a smaller representation that later loops send to the provider instead of the full result. */
  toHistoricalResult?: (context: HistoricalToolResultContext<TInput, TOutput>) => unknown | Promise<unknown>;

  server(execute: (input: z.infer<TInput>, ctx: ToolContext) => Promise<z.infer<TOutput>>): ServerTool<TInput, TOutput>;

  client(
    execute: (input: z.infer<TInput>) => z.infer<TOutput> | Promise<z.infer<TOutput>>,
  ): ClientTool<TInput, TOutput>;
}

export type ServerTool<TInput extends z.ZodType = z.ZodType, TOutput extends z.ZodType = z.ZodType> = {
  readonly kind: "server";
  readonly def: ToolDefinition<TInput, TOutput>;
  execute(input: z.infer<TInput>, ctx: ToolContext): Promise<z.infer<TOutput>>;
}

export type ClientTool<TInput extends z.ZodType = z.ZodType, TOutput extends z.ZodType = z.ZodType> = {
  readonly kind: "client";
  readonly def: ToolDefinition<TInput, TOutput>;
  execute(input: z.infer<TInput>): z.infer<TOutput> | Promise<z.infer<TOutput>>;
}

export type Tool = ServerTool | ClientTool;

export type ToolResolver = () => Tool[] | Promise<Tool[]>;

export type ToolContext = {
  /** Provider-assigned ID of the tool call currently being executed. */
  callId?: string;
  signal: AbortSignal;
  /** Request user approval mid-execution. Returns true if approved, false if denied. */
  requestApproval(message: string): Promise<boolean>;
  /** Request a client-side tool execution mid-execution. */
  requestClientTool<T = unknown>(name: string, args: unknown): Promise<T>;
}

// ----------------------------------------------------------------------------
// 4. nessi()
// ----------------------------------------------------------------------------

export type NessiOptions = {
  agentId?: string;
  /** Correlates every outbound event emitted by one logical nessi() loop. Generated when omitted. */
  loopId?: string;
  /**
   * User input for this loop. When omitted, the loop runs directly over the existing
   * store history: a trailing user message acts as the prompt, and unresolved tool_call
   * blocks on the trailing assistant message are resumed (executed or re-requested)
   * before the next provider turn. Push matching approval_response / tool_result events
   * before iterating to seed a resumed loop.
   */
  input?: Input;
  provider: Provider;
  systemPrompt: string;
  /** Static tools or a resolver evaluated once before every provider turn. */
  tools?: Tool[] | ToolResolver;
  store: SessionStore;
  creditStore?: CreditStore;
  compact?: CompactFn;
  /** Supplies pending steering messages at safe loop boundaries. */
  steering?: SteeringFn;
  maxTurns?: number;
  temperature?: number;
  maxOutputTokens?: number;
  /** Reasoning effort passed to the provider unchanged; "none" turns reasoning off. Overrides the provider default. */
  reasoningEffort?: ReasoningEffort;
  /** Extra request body fields for every provider call, merged over the provider's `extraBody`. */
  extraBody?: Record<string, unknown>;
  /** @deprecated Use `reasoningEffort` instead. Ignored when `reasoningEffort` is set. */
  disableReasoning?: boolean;
  coalesce?: CoalesceOptions;
  /** Max chars for tool results in the context sent to the provider. Longer results are truncated. */
  maxToolResultChars?: number;
  signal?: AbortSignal;
}

export type SteeringContext = {
  agentId: string;
  loopId: string;
  signal: AbortSignal;
}

export type SteeringFn = (
  context: SteeringContext,
) => string | readonly string[] | undefined | Promise<string | readonly string[] | undefined>;

export type CoalesceOptions = {
  ms?: number;
  maxChars?: number;
}

export type NessiLoop = {
  [Symbol.asyncIterator](): AsyncIterator<OutboundEvent>;
  subscribe(listener: (event: OutboundEvent) => void): () => void;
  push(event: InboundEvent): void;
  steer(message: string): void;
  abort(): void;
}

export type StructuredInput = Input | UserMessage;

export type StructuredMode = "native" | "fallback" | "repair" | "tool_loop";

export type StructuredMeta = {
  mode: StructuredMode;
  repaired: boolean;
  attempts: number;
  usedResponseFormat: boolean;
};

export type StructuredToolResolver = () => ServerTool[] | Promise<ServerTool[]>;

export type StructuredOptions<TOutput extends z.ZodType = z.ZodType> = {
  agentId?: string;
  /** Correlates the internal structured task. Generated when omitted. */
  loopId?: string;
  provider: Provider;
  systemPrompt?: string;
  input: StructuredInput;
  output: TOutput;
  outputName?: string;
  /** Static server tools or a resolver evaluated once before every provider turn. */
  tools?: ServerTool[] | StructuredToolResolver;
  maxTurns?: number;
  temperature?: number;
  maxOutputTokens?: number;
  /** Reasoning effort passed to the provider unchanged; "none" turns reasoning off. Overrides the provider default. */
  reasoningEffort?: ReasoningEffort;
  /** Extra request body fields for every provider call, merged over the provider's `extraBody`. */
  extraBody?: Record<string, unknown>;
  /** @deprecated Use `reasoningEffort` instead. Ignored when `reasoningEffort` is set. */
  disableReasoning?: boolean;
  signal?: AbortSignal;
  onEvent?: (event: OutboundEvent) => void;
}

export type StructuredResult<TOutput> = {
  output: TOutput;
  message: AssistantMessage;
  aggregate: LoopAggregate;
  reason: DoneReason;
  loopId: string;
  usage?: Usage;
  providerMeta?: GenerateResult["providerMeta"];
  structuredMeta: StructuredMeta;
}

// ----------------------------------------------------------------------------
// 5. SessionStore
// ----------------------------------------------------------------------------

export type StoreEntry = {
  seq: number;
  kind: "message" | "summary";
  message: Message;
}

export type SessionStore = {
  load(): Promise<StoreEntry[]>;
  append(message: Message, opts?: { seq?: number; kind?: "message" | "summary" }): Promise<void>;
}

// ----------------------------------------------------------------------------
// 6. Compaction
// ----------------------------------------------------------------------------

export type CompactFn = (ctx: CompactContext) => null | Promise<void>;

export type CompactContext = {
  entries: StoreEntry[];
  store: SessionStore;
  provider: Provider;
  usage: Usage;
  force: boolean;
  /** Estimated fill ratio (estimatedTokens / contextWindow). Only set when contextWindow is known. */
  fillRatio?: number;
  /** Aborted when the surrounding loop or compact() run is aborted. Pass it to provider calls. */
  signal?: AbortSignal;
  /** The loop's `reasoningEffort`, to forward to provider calls if the compaction should use it. */
  reasoningEffort?: ReasoningEffort;
  /** The loop's `extraBody`, to forward to provider calls if the compaction should use it. */
  extraBody?: Record<string, unknown>;
}

export type CompactOptions = {
  agentId?: string;
  /** Correlates every event emitted by one standalone compact() run. Generated when omitted. */
  loopId?: string;
  store: SessionStore;
  provider: Provider;
  compact: CompactFn;
  usage?: Usage;
  force?: boolean;
  signal?: AbortSignal;
}

export type CompactResult = {
  applied: boolean;
  entriesBefore: number;
  entriesAfter: number;
  forced: boolean;
}

export type CompactDoneReason = "stop" | "error" | "aborted";

export type CompactEvent =
  | (LoopEventFields & { type: "loop_start" })
  | (LoopEventFields & { type: "compaction_start" })
  | (LoopEventFields & { type: "compaction_end" })
  | (LoopEventFields & { type: "issue"; issue: NessiIssue })
  | (LoopEventFields & { type: "loop_end"; reason: CompactDoneReason; result: CompactResult });

export type CompactLoop = {
  [Symbol.asyncIterator](): AsyncIterator<CompactEvent>;
  subscribe(listener: (event: CompactEvent) => void): () => void;
  abort(): void;
}

// ----------------------------------------------------------------------------
// 7. CreditStore
// ----------------------------------------------------------------------------

export type CreditStore = {
  remaining(): Promise<number>;
  deduct(credits: number): Promise<void>;
}
