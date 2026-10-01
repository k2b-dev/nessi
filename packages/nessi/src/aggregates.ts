// ============================================================================
// nessi - Loop aggregate helpers
// ============================================================================

import type {
  LoopAggregate,
  LoopIssueAggregate,
  LoopTimingAggregate,
  LoopToolCallAggregate,
  LoopToolIssueAggregate,
  LoopTurnAggregate,
  Usage,
} from "./types.js";

export const cloneUsage = (usage: Usage | undefined): Usage | undefined =>
  usage ? { ...usage } : undefined;

export const mergeUsage = (left: Usage | undefined, right: Usage | undefined): Usage | undefined => {
  if (!right) return cloneUsage(left);
  const next: Usage = {
    input: (left?.input ?? 0) + right.input,
    output: (left?.output ?? 0) + right.output,
    total: (left?.total ?? 0) + right.total,
  };
  if (left?.cacheRead !== undefined || right.cacheRead !== undefined) {
    next.cacheRead = (left?.cacheRead ?? 0) + (right.cacheRead ?? 0);
  }
  if (left?.creditsUsed !== undefined || right.creditsUsed !== undefined) {
    next.creditsUsed = (left?.creditsUsed ?? 0) + (right.creditsUsed ?? 0);
  }
  return next;
}

const cloneToolCall = (toolCall: LoopToolCallAggregate): LoopToolCallAggregate => ({ ...toolCall });

const cloneToolIssue = (toolIssue: LoopToolIssueAggregate): LoopToolIssueAggregate => ({ ...toolIssue });

const cloneIssue = (issue: LoopIssueAggregate): LoopIssueAggregate => ({ ...issue });

const cloneTiming = (timing: LoopTimingAggregate | undefined): LoopTimingAggregate | undefined =>
  timing
    ? {
        ...timing,
        totalElapsedMs: timing.totalElapsedMs ?? timing.generationMs + timing.toolExecutionMs,
      }
    : undefined;

const outputTokensPerSecond = (usage: Usage | undefined, generationMs: number): number | undefined => {
  if (!usage || usage.output <= 0 || generationMs <= 0) return undefined;
  return usage.output / (generationMs / 1000);
}

export const buildLoopTiming = (
  timing: Pick<LoopTimingAggregate, "wallMs" | "generationMs" | "toolExecutionMs" | "actionWaitMs">,
  usage: Usage | undefined,
): LoopTimingAggregate => {
  const snapshot: LoopTimingAggregate = {
    ...timing,
    totalElapsedMs: timing.generationMs + timing.toolExecutionMs,
  };
  const throughput = outputTokensPerSecond(usage, timing.generationMs);
  if (throughput !== undefined) snapshot.outputTokensPerSecond = throughput;
  return snapshot;
}

const mergeTiming = (
  left: LoopTimingAggregate | undefined,
  right: LoopTimingAggregate | undefined,
  usage: Usage | undefined,
): LoopTimingAggregate | undefined => {
  if (!left && !right) return undefined;
  const generationMs = (left?.generationMs ?? 0) + (right?.generationMs ?? 0);
  return buildLoopTiming({
    wallMs: (left?.wallMs ?? 0) + (right?.wallMs ?? 0),
    generationMs,
    toolExecutionMs: (left?.toolExecutionMs ?? 0) + (right?.toolExecutionMs ?? 0),
    actionWaitMs: (left?.actionWaitMs ?? 0) + (right?.actionWaitMs ?? 0),
  }, usage);
}

const toolIssuesFromAggregate = (aggregate: LoopAggregate): LoopToolIssueAggregate[] =>
  (aggregate.toolIssues ?? aggregate.turns.flatMap((turn) => turn.toolIssues ?? [])).map(cloneToolIssue);

const issuesFromAggregate = (aggregate: LoopAggregate): LoopIssueAggregate[] =>
  (aggregate.issues ?? aggregate.toolIssues ?? aggregate.turns.flatMap((turn) => turn.issues ?? turn.toolIssues ?? []))
    .map(cloneIssue);

const cloneTurn = (turn: LoopTurnAggregate): LoopTurnAggregate => ({
  ...turn,
  usage: cloneUsage(turn.usage),
  toolCalls: turn.toolCalls.map(cloneToolCall),
  ...(turn.toolIssues ? { toolIssues: turn.toolIssues.map(cloneToolIssue) } : {}),
  ...(turn.issues ? { issues: turn.issues.map(cloneIssue) } : {}),
});

export const aggregateFromTurns = (
  turns: LoopTurnAggregate[],
  loopIssues: LoopIssueAggregate[] = [],
): LoopAggregate => {
  const clonedTurns = turns.map(cloneTurn);
  const issues = loopIssues.length > 0
    ? loopIssues.map(cloneIssue)
    : clonedTurns.flatMap((turn) => turn.issues ?? turn.toolIssues ?? []);
  const toolIssues = issues.filter((issue): issue is LoopToolIssueAggregate =>
    issue.kind === "malformed_tool_call" || issue.kind === "cancelled_tool_call",
  );
  const usage = clonedTurns.reduce((mergedUsage, turn) => mergeUsage(mergedUsage, turn.usage), undefined as Usage | undefined);
  return {
    turns: clonedTurns,
    usage,
    issueCount: issues.length,
    issues: issues.map(cloneIssue),
    toolCallCount: clonedTurns.reduce((count, turn) => count + turn.toolCalls.length, 0),
    toolErrorCount: clonedTurns.reduce(
      (count, turn) => count + turn.toolCalls.filter((toolCall) => toolCall.isError).length,
      0,
    ),
    toolIssueCount: toolIssues.length,
    toolMalformedCount: toolIssues.filter((issue) => issue.kind === "malformed_tool_call").length,
    toolCancelledCount: toolIssues.filter((issue) => issue.kind === "cancelled_tool_call").length,
    toolIssues: toolIssues.map(cloneToolIssue),
    assistantMessageCount: clonedTurns.length,
  };
}

export const cloneLoopAggregate = (aggregate: LoopAggregate): LoopAggregate => {
  const turns = aggregate.turns.map(cloneTurn);
  const issues = issuesFromAggregate(aggregate);
  const toolIssues = toolIssuesFromAggregate(aggregate);
  return {
    turns,
    usage: cloneUsage(aggregate.usage),
    ...(aggregate.timing ? { timing: cloneTiming(aggregate.timing) } : {}),
    issueCount: aggregate.issueCount ?? issues.length,
    issues,
    toolCallCount: aggregate.toolCallCount,
    toolErrorCount: aggregate.toolErrorCount,
    toolIssueCount: aggregate.toolIssueCount ?? toolIssues.length,
    toolMalformedCount: aggregate.toolMalformedCount ?? toolIssues.filter((issue) => issue.kind === "malformed_tool_call").length,
    toolCancelledCount: aggregate.toolCancelledCount ?? toolIssues.filter((issue) => issue.kind === "cancelled_tool_call").length,
    toolIssues,
    assistantMessageCount: aggregate.assistantMessageCount,
  };
}

export const mergeLoopAggregates = (
  left: LoopAggregate | undefined,
  right: LoopAggregate | undefined,
): LoopAggregate | undefined => {
  if (!left) return right ? cloneLoopAggregate(right) : undefined;
  if (!right) return cloneLoopAggregate(left);
  const merged = aggregateFromTurns(
    [...left.turns, ...right.turns],
    [...issuesFromAggregate(left), ...issuesFromAggregate(right)],
  );
  const timing = mergeTiming(left.timing, right.timing, merged.usage);
  return timing ? { ...merged, timing } : merged;
}
