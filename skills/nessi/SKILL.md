---
name: nessi
description: "Build applications with the @k2b/nessi TypeScript library. Use for creating or modifying a CLI, backend endpoint, service, prototype, provider switcher, streaming UI adapter, tool-calling workflow, structured-output task, agent loop, or AI integration using @k2b/nessi. Trigger for questions about root nessi(), nessi.structured(), @k2b/nessi/ai complete(), stream(), provider setup, audio transcription, speech-to-text, Scaleway, responseFormat, OpenAI/OpenRouter/vLLM/Ollama/Anthropic/Mistral/Gemini, message formats, multimodal input, canonical block streaming events, issue events for malformed tool streams, dynamic per-turn tool resolution, tool call IDs, loopId correlation, loop_end.aggregate metadata, loop-level stats, local or durable steering, historical tool results, context growth, tool execution events, usage accounting, generation options, provider timeouts, API keys, local models, context-overflow handling, or when choosing whether the provider-only /ai layer is enough versus the root APIs."
---

# @k2b/nessi Consumer Skill

Use this skill to help someone build software on top of `@k2b/nessi`. Keep the focus on the public consumer API. Do not drift into repository-maintainer work unless the user explicitly asks to change nessi itself.

`@k2b/nessi` replaces the deprecated `@valentinkolb/nessi` package. The API and
subpaths are unchanged; migrate dependencies and imports by replacing the
scope.

## First move

1. Identify the application shape: CLI, backend route, browser adapter, service job, test fixture, or agent prototype.
2. Identify the provider family: hosted API, OpenRouter aggregation, local Ollama/vLLM, or a custom OpenAI-compatible endpoint.
3. Decide whether the user needs only the provider layer or the full agent loop:
   - Use `@k2b/nessi/ai` for provider calls, streaming, message normalization, tool-call extraction, and usage data.
   - For audio transcription, use `openAICompatibleTranscription()` from `/ai` and read `references/providers.md`. It returns a separate `TranscriptionProvider` with `transcribe()`.
   - Use `nessi.structured()` from the package root when the user wants a schema-valid typed task result, optionally with bounded server tools.
   - Use the `@k2b/nessi` root exports when the user wants an agent loop that executes tools, stores conversation history, handles approvals, or compacts context.
4. Produce working TypeScript that matches the current public API:
   - Root agent APIs come from `@k2b/nessi`.
   - Chat provider constructors come from `@k2b/nessi/ai`, then expose `complete(request)` and `stream(request)`. The transcription constructor takes `(model, options)` with a required `baseURL`.

## Reference routing

Read only the references needed for the task:

| Need | Read |
| --- | --- |
| Minimal setup, imports, complete() examples | `references/quickstart.md` |
| Choosing and configuring providers, audio transcription and Scaleway | `references/providers.md` |
| Message shapes, files, stream events, usage | `references/messages-and-streaming.md` |
| Tool-call flow, errors, retries, context overflow | `references/tools-and-errors.md` |

## Implementation preferences

- Prefer small examples that can be pasted into a real TypeScript/Bun project.
- Keep provider configuration explicit: model string first, options second.
- Use environment variables for API keys and avoid hardcoding secrets.
- Stream by iterating events and switching on `event.type`.
- For root `nessi()` loops, pass an application-level `loopId` when the app already has a request/response group id; otherwise use the generated `event.loopId` that appears on every outbound event.
- For root `nessi()` loops, pass `temperature`, `maxOutputTokens`, and `disableReasoning` at the top level when the app has a default generation policy for the whole loop.
- For root `nessi()` loops, use `event.type === "loop_end"` plus `event.aggregate` for one logical response group, aggregate usage, timing, loop-level stats, assistant turn count, tool calls, tool results, validation/execution errors, and malformed/cancelled tool-stream issues across multi-turn tool loops.
- Use `loop.steer(message)` when the caller owns the active loop. Use the optional `steering` callback when input arrives through another process or worker; return one message, an ordered array, or `undefined` from application-owned persistence.
- Treat `steer_applied` as the common confirmation for local and callback-supplied steering. The callback is checked only at safe loop boundaries, including before provider calls and before normal completion; it does not interrupt an in-flight provider request or tool.
- Keep queue claims, delivery guarantees, and persistence schemas inside the application. Nessi only decides when returned steering messages can affect execution.
- Treat `event.aggregate.timing.totalElapsedMs` as generation plus active tool execution time; it intentionally excludes `actionWaitMs` for approval/client-tool waits. Use `timing.generationMs` for output-token throughput via `timing.outputTokensPerSecond`.
- For `nessi.structured()`, require a Zod `output` schema and return `result.output`; use `result.structuredMeta` and `result.aggregate` for diagnostics.
- For `nessi.structured()` with tools, only pass server tools that do not need approval. Use full `nessi()` for client tools, approvals, or custom interactive tool bridges.
- `nessi.structured()` also accepts a server-tool resolver. A resolver always selects `tool_loop`, even for an empty snapshot; Nessi validates every snapshot and appends its internal `submit_result` tool. A static empty array retains direct native/fallback mode.
- For provider-only structured output, pass `responseFormat` to `provider.complete()` only when the user explicitly needs the low-level provider request. Prefer `nessi.structured()` for consumer code that needs validated typed output.
- For standalone `compact()` loops, use the same `loopId` grouping pattern and handle `loop_start`, `compaction_start`, `compaction_end`, `issue`, and `loop_end`.
- Treat `issue.kind === "malformed_tool_call"` and `issue.kind === "cancelled_tool_call"` as pre-execution stream issues. They mean no executable tool call exists for that pending provider start.
- Treat `tool_execution_start` / `tool_execution_end` as Nessi's tool-runtime attempt boundary. `tool_action_request` is the event that asks the app for approval or client-side tool output.
- Pass `tools` as an array for a fixed toolset or as `() => Tool[] | Promise<Tool[]>` when application state controls availability. Nessi resolves one copied snapshot per provider turn and uses it for both advertised schemas and execution; changes become visible on the next turn. Keep persistence, discovery, authorization, and cleanup in the application.
- Use `ctx.callId` inside server tools when execution or audit records must correlate with the provider tool call. The field is optional in the public type for compatibility but populated by normal and resumed Nessi execution.
- Use a tool's optional `toHistoricalResult({ input, output, callId })` when full output is needed in the current loop but a smaller tool-specific representation is sufficient in later loops. Nessi persists both values and derives the historical value only once.
- Keep the same `loopId` when resuming an originating loop so provider calls continue receiving full tool results. Different loop IDs receive persisted historical values; legacy results without one remain full.
- Treat `tool_historical_result_error` as non-fatal: the full successful result remains persisted and the loop continues. `maxToolResultChars` is applied after historical selection as a final safety boundary.
- Treat tool calls as data the application must handle; `@k2b/nessi/ai` does not execute tools.
- Surface unsupported file input and provider error behavior instead of hiding it.
- If the user asks for browser code, keep API keys server-side and expose a backend route.

## Current public surface

Root agent APIs:

```ts
import { nessi, defineTool, memoryStore } from "@k2b/nessi";
```

Loop aggregate helpers:

```ts
import { mergeUsage, cloneLoopAggregate, mergeLoopAggregates } from "@k2b/nessi";
```

Provider constructors:

```ts
import {
  anthropic,
  gemini,
  mistral,
  ollama,
  openai,
  openAICompatible,
  openrouter,
  vllm,
} from "@k2b/nessi/ai";
```

Focused provider imports are also available:

```ts
import { openrouter } from "@k2b/nessi/ai/providers/openrouter";
```

Common request shape:

```ts
const result = await provider.complete({
  systemPrompt: "Be concise.",
  messages: [
    { role: "user", content: [{ type: "text", text: "Summarize this." }] },
  ],
  temperature: 0,
  maxOutputTokens: 512,
});
```

Common structured-output shape:

```ts
import { nessi } from "@k2b/nessi";
import { openrouter } from "@k2b/nessi/ai";
import { z } from "zod";

const result = await nessi.structured({
  provider: openrouter("openai/gpt-4.1-mini", {
    apiKey: process.env.OPENROUTER_API_KEY,
  }),
  input: "Extract a task from: Ship onboarding by Friday.",
  outputName: "task",
  output: z.object({
    title: z.string(),
    due: z.string().nullable(),
  }),
  temperature: 0,
});

console.log(result.output);
console.log(result.structuredMeta);
```

Common stream shape:

```ts
const textBlocks = new Set<string>();

for await (const event of provider.stream({ messages })) {
  if (event.type === "block_start" && event.kind === "text") textBlocks.add(event.blockId);
  if (event.type === "block_delta" && textBlocks.has(event.blockId)) process.stdout.write(event.delta);
  if (event.type === "issue") console.error(event.issue.kind, event.issue.message);
}
```

Common root-loop handling:

```ts
const loop = nessi({
  loopId: requestId,
  provider,
  systemPrompt,
  input,
  store,
  tools,
  temperature: 0,
  maxOutputTokens: 512,
  disableReasoning: true,
});

const textBlocks = new Set<string>();

for await (const event of loop) {
  if (event.type === "block_start" && event.kind === "text") textBlocks.add(event.blockId);
  if (event.type === "block_delta" && textBlocks.has(event.blockId)) process.stdout.write(event.delta);
  if (event.type === "issue") console.error(event.issue.kind, event.issue.message);
  if (event.type === "loop_end") {
    console.log(event.loopId);
    console.log(event.reason);
    console.log(event.aggregate.usage);
    console.log(event.aggregate.timing);
    console.log(event.aggregate.toolErrorCount);
    console.log(event.aggregate.toolMalformedCount);
  }
}
```

## Output style

When answering user requests, include:

1. The shortest viable implementation.
2. Any provider-specific setup or environment variables.
3. How to handle streaming, tool calls, and errors if relevant.
4. One verification step, such as a Bun command or a small mocked test.

Avoid large frameworks unless the user asks for one. If the requested task is ambiguous, state the assumption and pick the simplest useful shape.
