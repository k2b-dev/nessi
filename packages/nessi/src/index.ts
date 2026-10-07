// ============================================================================
// nessi – Public API
// ============================================================================

import { nessi as createNessiLoop } from "./nessi.js";
import { structured, StructuredOutputError } from "./structured.js";

export const nessi = Object.assign(createNessiLoop, { structured });
export { structured, StructuredOutputError };
export { compact } from "./compact.js";
export { defineTool, toolToJsonSchema, toolToSpec } from "./tools.js";
export { memoryStore } from "./stores.js";
export { estimateTokens, truncateMiddle, truncateToolResults } from "./utils.js";
export { cloneLoopAggregate, cloneUsage, mergeLoopAggregates, mergeUsage } from "./aggregates.js";

export type {
  // Core
  NessiOptions,
  NessiLoop,
  CoalesceOptions,
  SteeringContext,
  SteeringFn,
  StructuredInput,
  StructuredMeta,
  StructuredMode,
  StructuredOptions,
  StructuredResult,
  StructuredToolResolver,
  // Content
  ContentPart,
  JsonSchemaObject,
  Input,
  // Events
  OutboundEvent,
  InboundEvent,
  DoneReason,
  ToolActionKind,
  NessiIssue,
  LoopAggregate,
  LoopTimingAggregate,
  LoopTurnAggregate,
  LoopToolCallAggregate,
  LoopToolIssueAggregate,
  // Messages
  Message,
  UserMessage,
  AssistantMessage,
  AssistantStopReason,
  HistoricalToolResult,
  ToolResultMessage,
  AssistantContentBlock,
  TextBlock,
  ThinkingBlock,
  ToolCallBlock,
  ToolStreamIssue,
  ToolStreamIssueKind,
  ToolStreamIssueReason,
  ToolHistoricalResultIssue,
  Usage,
  // Tools
  ToolDefinition,
  HistoricalToolResultContext,
  ServerTool,
  ClientTool,
  Tool,
  ToolResolver,
  ToolContext,
  // Provider
  Provider,
  ProviderRequest,
  ProviderEvent,
  ReasoningEffort,
  ResponseFormat,
  // Store
  StoreEntry,
  SessionStore,
  // Compaction
  CompactFn,
  CompactContext,
  CompactOptions,
  CompactResult,
  CompactDoneReason,
  CompactEvent,
  CompactLoop,
  // Credits
  CreditStore,
} from "./types.js";
