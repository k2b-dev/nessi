# Messages and Streaming

`@k2b/nessi/ai` uses one message model across providers. Adapters translate this model to each provider's native wire format.

## Message model

```ts
type Message = UserMessage | AssistantMessage | ToolResultMessage;
```

User messages contain content parts:

```ts
const userMessage = {
  role: "user" as const,
  content: [
    { type: "text" as const, text: "Describe this image." },
    {
      type: "file" as const,
      mediaType: "image/png",
      data: base64Png,
    },
  ],
};
```

Assistant messages contain normalized blocks:

```ts
const assistantMessage = {
  role: "assistant" as const,
  content: [
    { type: "text" as const, text: "I can help with that." },
    { type: "thinking" as const, thinking: "Reasoning if the provider exposes it.", signature: "opaque" },
    { type: "tool_call" as const, id: "call-1", name: "search", args: { q: "@k2b/nessi/ai" } },
  ],
};
```

`signature`, `redacted` and `details` (on thinking blocks; `signature` also on
text and tool calls) are opaque provider data, and `message.provider` names the provider
that produced the message. Keep them when storing history: providers need them
back unchanged, and each provider only sends its own data back.

Tool result messages return application-computed tool outputs to the next provider call:

```ts
const toolResult = {
  role: "tool_result" as const,
  callId: "call-1",
  name: "search",
  result: { hits: [] },
};
```

Root `nessi()` loops may persist an additional representation derived by the
tool for later loops:

```ts
const storedToolResult = {
  role: "tool_result" as const,
  callId: "call-1",
  name: "search",
  result: fullSearchResult,
  historicalResult: {
    originLoopId: "loop-123",
    value: compactSearchResult,
  },
};
```

This metadata is persisted, not manually supplied to provider-only calls.
Nessi substitutes `result` only in the provider-facing message list of a later
loop; the stored full result remains authoritative.

## Completion response

`complete()` returns a final assistant message, usage, finish reason, and provider metadata.

```ts
const result = await provider.complete({ messages });

if (result.finishReason === "tool_use") {
  // Inspect result.message.content for tool_call blocks.
}
```

Extract final text:

```ts
const text = result.message.content
  .filter((block) => block.type === "text")
  .map((block) => block.text)
  .join("");
```

## Stream events

Always switch on `event.type`:

```ts
const textBlocks = new Set<string>();
const thinkingBlocks = new Set<string>();

for await (const event of provider.stream({ messages })) {
  switch (event.type) {
    case "block_start":
      if (event.kind === "text") textBlocks.add(event.blockId);
      if (event.kind === "thinking") thinkingBlocks.add(event.blockId);
      break;
    case "block_delta":
      if (textBlocks.has(event.blockId)) process.stdout.write(event.delta);
      if (thinkingBlocks.has(event.blockId)) {
        console.error(event.delta);
      }
      break;
    case "block_end":
      if (event.block.type === "tool_call") {
        await handleToolCall(event.block);
      }
      textBlocks.delete(event.blockId);
      thinkingBlocks.delete(event.blockId);
      break;
    case "issue":
      if (event.issue.kind === "provider_error" || event.issue.kind === "timeout") {
        throw new Error(event.issue.message);
      }
      console.warn(event.issue.kind, event.issue.message);
      break;
    case "usage":
      console.error("usage", event.usage, event.finishReason);
      break;
  }
}
```

Event meanings:

- `block_start`: a normalized assistant content block started. `kind` is `text`, `thinking`, or `tool_call`.
- `block_delta`: incremental text for the current block. Track `block_start.kind` by `blockId` before printing deltas.
- `block_end`: final structured assistant content block. Executable provider tool calls appear here as `block.type === "tool_call"`.
- `issue`: structured provider, timeout, malformed/cancelled tool-stream, tool-execution, or runtime problem.
- `usage`: token usage, sometimes with final `finishReason`.

Every `turn_start` gets a matching `turn_end`, including failed or aborted turns (partial message with `stopReason` `"error"` or `"interrupted"`; an abort during tool execution keeps the stored message). Usage reported before a failure still counts in the aggregate and credits. A provider-side stop (`finishReason: "error"`, such as a safety filter) skips that turn's tool calls, also on later resumes, and ends the loop with `"error"`.

Root `nessi()` loops forward provider block events and add `loopId`, `turnId`, and `turnIndex`:

```ts
for await (const event of loop) {
  switch (event.type) {
    case "loop_start":
    case "turn_start":
      break;
    case "block_delta":
      // Same block event shape as provider.stream(), with loop and turn metadata.
      break;
    case "tool_execution_start":
      console.error("tool attempt", event.name, event.callId);
      break;
    case "tool_action_request":
      // Respond with loop.push({ type: "approval_response" | "tool_result", ... }).
      break;
    case "tool_execution_end":
      console.error("tool finished", event.name, event.isError);
      break;
    case "loop_end":
      console.error(event.aggregate.usage);
      console.error(event.aggregate.timing);
      break;
  }
}
```

## Tool-stream invariants

Apps should execute tools only from final `tool_call` content blocks:

```ts
if (event.type === "block_end" && event.block.type === "tool_call") {
  await executeTool(event.block.name, event.block.args);
}
```

Provider adapters do not expose half-open tool starts through the public stream.
If text, thinking text, invalid JSON, a provider error, or stream end interrupts
a pending tool call, the public event is `issue` with `issue.kind` set to
`malformed_tool_call` or `cancelled_tool_call`.

Root `nessi()` loops only execute tool calls that reached a final `tool_call`
content block. Tool runtime events are separate:

- `tool_execution_start`: Nessi started handling a final tool call. It is paired with `tool_execution_end`, including validation failures.
- `tool_action_request`: the app must answer an approval or client-side tool request with `loop.push()`.
- `tool_execution_end`: Nessi produced a tool result or an error result.
- `steer_applied`: Nessi persisted a local or callback-supplied steering message and will include it in the next provider call.

Malformed/cancelled provider tool streams are reported in `issue` events and in
`loop_end.aggregate.toolIssues`.

## Standalone compaction loop

The exported `compact()` helper uses the same loop-style event pattern for UI
and persistence code. Every event carries `agentId` and `loopId`.
After `compaction_start`, Nessi emits `compaction_end` even when the compaction
operation fails; the failure is reported as `issue` before the final `loop_end`.

```ts
for await (const event of compact({ store, provider, compact: compactFn })) {
  switch (event.type) {
    case "loop_start":
      break;
    case "compaction_start":
      break;
    case "compaction_end":
      break;
    case "issue":
      console.error(event.issue.kind, event.issue.message);
      break;
    case "loop_end":
      console.log(event.loopId, event.reason, event.result);
      break;
  }
}
```

## Usage accounting

Usage has normalized token counts:

```ts
type Usage = {
  input: number;
  output: number;
  total: number;
  cacheRead?: number;
  creditsUsed?: number;
};
```

If the provider was configured with `creditsPerInputToken` and `creditsPerOutputToken`, `creditsUsed` is attached when usage is available.

Loop aggregates also include timing:

```ts
type LoopTimingAggregate = {
  wallMs: number;
  totalElapsedMs: number; // generationMs + toolExecutionMs
  generationMs: number;
  toolExecutionMs: number;
  actionWaitMs: number;
  outputTokensPerSecond?: number;
};
```

Use `generationMs` and `outputTokensPerSecond` for model throughput. Use
`actionWaitMs` separately for approval/client-tool waits; those waits are not
part of `totalElapsedMs`.

## Abort and output budget

Pass an `AbortSignal` to cancel a request:

```ts
const controller = new AbortController();

const result = await provider.complete({
  messages,
  signal: controller.signal,
  maxOutputTokens: 256,
});
```

Aborting a root `nessi()` loop with `loop.abort()`, its `signal`, or by leaving
the `for await` early cancels the provider request and the `ctx.signal` of a
running tool. The loop then ends with `loop_end` reason `aborted`. Unfinished
tool calls stay unanswered in the store. A later loop without `input` resumes
calls of a completed answer, for example one waiting for approval, but never
calls from an answer that was aborted while being generated
(`stopReason: "interrupted"`). Those, and any call before new `input`, reach the
provider as interrupted error results. A `compact` function receives the loop signal as `ctx.signal`.

Use `reasoningEffort` to control reasoning: `"none"` for simple calls where reasoning would spend the output budget, or a level such as `"low"` or `"high"`. The value passes through unchanged and works on provider defaults, single calls and root loops; a call value wins. `extraBody` adds request fields Nessi does not model. `disableReasoning` is deprecated. Root `nessi()` accepts the same generation controls:

```ts
const loop = nessi({
  provider,
  systemPrompt,
  input,
  store,
  temperature: 0,
  maxOutputTokens: 512,
  reasoningEffort: "none",
});
```
