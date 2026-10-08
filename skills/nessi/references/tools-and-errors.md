# Tools and Errors

`@k2b/nessi/ai` normalizes tool-call data, but it does not execute tools. The application owns the tool registry, validation, execution, and follow-up call.

Use `@k2b/nessi` if the user wants a ready-made agent loop with tool execution, approvals, store handling, and compaction.

For the root `nessi()` agent loop, keep using streaming events for live UI and
debug output:

- `loopId` is present on every outbound event from one logical loop.
- `turn_end` reports each internal provider turn.
- `block_end` with `block.type === "tool_call"` reports final executable provider tool calls.
- `tool_execution_start` / `tool_execution_end` report Nessi's tool execution attempts.
- `tool_action_request` asks the app for approval, client-side tool output, or a custom approval inside a server tool. `needsApproval` is for server tools only; client tools run in the app, which asks for confirmation itself, and `defineTool().client()` rejects `needsApproval: true`.
- `issue` reports provider errors, timeouts, malformed/cancelled tool streams, and tool execution failures.
- `loop_end.aggregate` reports the complete logical loop after all internal turns.

Use `loop_end.aggregate` when the app needs one user-visible response group,
aggregate usage, loop timing, loop-level tool counts, persisted tool execution
errors, or malformed/cancelled tool-stream metadata:

```ts
const loop = nessi({
  loopId: responseGroupId,
  provider,
  systemPrompt,
  input,
  store,
  tools,
});

for await (const event of loop) {
  if (event.type === "loop_end") {
    const { aggregate } = event;
    saveResponseGroup({
      loopId: event.loopId,
      reason: event.reason,
      usage: aggregate.usage,
      timing: aggregate.timing,
      turns: aggregate.turns,
      issueCount: aggregate.issueCount,
      issues: aggregate.issues,
      toolCallCount: aggregate.toolCallCount,
      toolErrorCount: aggregate.toolErrorCount,
      toolIssueCount: aggregate.toolIssueCount,
      toolMalformedCount: aggregate.toolMalformedCount,
      toolCancelledCount: aggregate.toolCancelledCount,
      toolIssues: aggregate.toolIssues,
    });
  }
}
```

## Define tool specs

Tool specs are provider-facing JSON schemas:

```ts
const tools = [
  {
    name: "search_docs",
    description: "Search internal docs for a query.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string" },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
];
```

## One-shot tool-call flow

```ts
import type { Message } from "@k2b/nessi/ai";

const messages: Message[] = [
  {
    role: "user" as const,
    content: [{ type: "text" as const, text: "Find the docs for streaming." }],
  },
];

const first = await provider.complete({ messages, tools });

const toolCalls = first.message.content.filter((block) => block.type === "tool_call");

if (toolCalls.length > 0) {
  messages.push(first.message);
}

for (const call of toolCalls) {
  const result = await searchDocs(call.args);
  messages.push({
    role: "tool_result" as const,
    callId: call.id,
    name: call.name,
    result,
  });
}

const final = await provider.complete({ messages, tools });
```

If there can be multiple tool calls, append the assistant message once, then append all matching tool results.

## Streaming tool-call flow

During provider-only streaming, collect final `tool_call` content blocks from
`block_end`. If an `issue` appears with `kind` set to `malformed_tool_call` or
`cancelled_tool_call`, the pending provider start never became executable.

```ts
const toolCalls = [];
const toolIssues = [];

for await (const event of provider.stream({ messages, tools })) {
  if (event.type === "block_end" && event.block.type === "tool_call") {
    toolCalls.push(event.block);
  }
  if (
    event.type === "issue"
    && (event.issue.kind === "malformed_tool_call" || event.issue.kind === "cancelled_tool_call")
  ) {
    toolIssues.push(event.issue);
  }
}
```

For root `nessi()` loops, wait for `tool_action_request` before asking the app
to approve an action or execute a client-side tool:

```ts
for await (const event of loop) {
  if (event.type === "tool_action_request" && event.kind === "client_tool") {
    const result = await runClientTool(event.name, event.args);
    loop.push({ type: "tool_result", callId: event.callId, result });
  }
}
```

Top-level client tools validate pushed results against their `outputSchema`.
Nested `ctx.requestClientTool()` calls also validate args and output when the
requested client tool name is registered in `tools`.

## Dynamic root-loop tools

Use a resolver when the active root-loop tools depend on application state:

```ts
const loop = nessi({
  provider,
  systemPrompt,
  input,
  store,
  tools: () => permissions.allowedToolsFor(userId),
});
```

The resolver may be synchronous or asynchronous. Nessi evaluates it once per
provider turn, rejects duplicate names, and uses one copied snapshot for both
provider schemas and all execution from that response. If a tool enables or
removes another tool, the change appears on the next provider turn. Pending
calls restored from history resolve an additional current snapshot before
execution.

Resolvers should read application state without mutating it. The application
continues to own persistence, discovery, authorization, and cleanup. A resolver
failure becomes a `runtime_error` and ends the loop without calling the
provider.

Server tools can correlate their work with the provider call:

```ts
const audit = defineTool({
  name: "audit",
  description: "Write an audit entry",
  inputSchema: z.object({ action: z.string() }),
}).server(async ({ action }, ctx) => {
  await writeAudit({ action, toolCallId: ctx.callId });
  return { recorded: true };
});
```

`nessi.structured()` accepts a resolver with the same per-turn snapshot
semantics, limited to server tools without approval:

```ts
const result = await nessi.structured({
  provider,
  input: "Resolve the report.",
  output: reportSchema,
  tools: async () => registry.activeServerTools(),
});
```

Providing a resolver always selects `tool_loop`, including when the current
snapshot is empty. Nessi adds `submit_result` to every snapshot and rejects
client tools, approval tools, and a user tool with that reserved name. A static
empty tool array continues to use direct native/fallback structured output.

## Historical tool results

Use `toHistoricalResult` for tools whose raw output is useful during execution
but unnecessarily large in future loops:

```ts
const inspectFile = defineTool({
  name: "inspect_file",
  description: "Inspect a source file.",
  inputSchema: z.object({ path: z.string() }),
  outputSchema: z.object({ content: z.string(), language: z.string() }),
  toHistoricalResult: ({ input, output, callId }) => ({
    path: input.path,
    language: output.language,
    excerpt: output.content.slice(0, 800),
    callId,
  }),
}).server(inspectFile);
```

Nessi computes this value once after output validation and stores it alongside
the full result. The originating `loopId` always sends the full result to the
provider, including a resumed loop. A later loop ID sends the historical value
without changing the stored message, emitted events, or loop aggregate.

Return `undefined` to omit the historical value for a particular result. If the
callback throws, Nessi emits `tool_historical_result_error`, persists the full
successful output, and continues. Tools without the callback and legacy stored
messages are unchanged. If configured, `maxToolResultChars` truncates the
selected full or historical value afterward.

## Error handling

Stream problems are normalized `issue` events:

```ts
for await (const event of provider.stream({ messages })) {
  if (event.type === "issue") {
    if (event.issue.kind === "provider_error" && event.issue.contextOverflow) {
      // Summarize, truncate, or ask the user to reduce input.
    }
    if ("retryable" in event.issue && event.issue.retryable) {
      // Apply bounded retry with backoff in application code.
    }
    throw new Error(event.issue.message);
  }
}
```

Malformed tool streams are not provider connection errors. Handle them separately
if the UI wants to show model/tool-call diagnostics:

```ts
if (
  event.type === "issue"
  && (event.issue.kind === "malformed_tool_call" || event.issue.kind === "cancelled_tool_call")
) {
  console.warn(event.issue.reason, event.issue.callId, event.issue.message);
}
```

`complete()` throws on connection errors, non-OK HTTP responses, and invalid provider JSON.

## Context overflow

Context overflow can appear as:

- `event.type === "issue"` with `issue.kind === "provider_error"` and `contextOverflow: true` in streaming
- a thrown error from `complete()`

Recommended application behavior:

1. Keep recent user-visible messages.
2. Drop or summarize large tool results.
3. Retry once after reducing context.
4. If the app needs automatic compaction, use `@k2b/nessi`.

## Browser safety

Do not call hosted providers directly from browser code with secret API keys. Put `@k2b/nessi/ai` behind a backend route:

```ts
// server route
const provider = openrouter("openai/gpt-4.1-mini", {
  apiKey: process.env.OPENROUTER_API_KEY,
});
```

The browser can receive text chunks or server-sent events from your own backend without seeing provider credentials.
