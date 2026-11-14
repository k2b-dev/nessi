# @k2b/nessi

Minimal agent loop and provider adapters for TypeScript.

Use the package root for the managed `nessi()` loop with tools, storage, and loop metadata. Use `@k2b/nessi/ai` when an app only needs provider calls through one normalized message and stream API.

`@k2b/nessi` replaces the deprecated `@valentinkolb/nessi` package. Existing
APIs and subpaths are unchanged; replace the package scope in dependencies and
imports.

## Quick start

```bash
bun add @k2b/nessi
```

```ts
import { nessi, defineTool, memoryStore } from "@k2b/nessi";
import { complete, openAICompatible } from "@k2b/nessi/ai";
import { z } from "zod";

const weather = defineTool({
  name: "weather",
  description: "Return a fake weather response",
  inputSchema: z.object({ city: z.string() }),
}).server(async ({ city }) => {
  return { city, forecast: "sunny" };
});

const loop = nessi({
  loopId: crypto.randomUUID(),
  provider: openAICompatible({
    name: "ollama",
    model: "llama3.1",
    baseURL: "http://localhost:11434/v1",
  }),
  systemPrompt: "You are concise.",
  input: "How is the weather in Berlin?",
  store: memoryStore(),
  tools: [weather],
  temperature: 0,
  maxOutputTokens: 512,
});

const textBlocks = new Set<string>();

for await (const event of loop) {
  if (event.type === "block_start" && event.kind === "text") {
    textBlocks.add(event.blockId);
  }
  if (event.type === "block_delta" && textBlocks.has(event.blockId)) {
    process.stdout.write(event.delta);
  }
  if (event.type === "issue") {
    console.error(event.issue.kind, event.issue.message);
  }
  if (event.type === "loop_end") {
    console.log(event.loopId);
    console.log(event.aggregate.usage);
    console.log(event.aggregate.timing);
  }
}
```

Every outbound event from one `nessi()` run carries the same `loopId`. Pass your own `loopId` to align events with a persisted request or UI response group, or let Nessi generate one when omitted.

`turn_end` reports each internal provider turn and closes every `turn_start`,
also when a turn fails or is aborted. A turn that fails or is aborted while the
model is still answering carries the partial message with `stopReason`
`"error"` or `"interrupted"`; an abort during tool execution carries the stored
message. Usage the provider already reported counts in the aggregate. Partial content is stored, except for a context
overflow, which is retried after compaction or ends the loop. When the provider
itself stops an answer (`finishReason: "error"`, e.g. a content or safety
filter), its tool calls are not executed, not even by a later resume, and the
loop ends with `"error"`. The final `loop_end` event includes `aggregate`, which groups assistant turns, executable tool calls, tool results, validation/execution errors, malformed or cancelled tool streams, summed usage, and timing for the complete logical loop. `aggregate.timing.totalElapsedMs` is model generation plus active tool execution; approval/client-tool waits are tracked separately as `aggregate.timing.actionWaitMs`. Helper exports such as `mergeUsage()`, `cloneLoopAggregate()`, and `mergeLoopAggregates()` are available from `@k2b/nessi`.

## Malformed tool calls

When a provider emits malformed tool arguments, `nessi()` reports a
`malformed_tool_call` issue and gives the model another turn to retry with
smaller, valid JSON arguments. The stored response gets the malformed call as a
`tool_call` with an ID that starts with `malformed_` and empty arguments, and the
call gets an error result with `reason: "malformed_tool_call"` that explains
what went wrong. The malformed call is never executed, not even by a resume.
Valid calls from the same response still run.

Every error result nessi writes itself carries a fixed `reason`, so code can
tell them apart without comparing text: `malformed_tool_call`, `unknown_tool`,
`input_validation_failed`, `output_validation_failed`, `execution_failed`,
`approval_denied`, `timeout`, and, for calls without a stored result,
`not_run` or `not_returned`.

A call can stay without a result, for example after an abort or a crash. The
store keeps it that way, and every provider request answers it with an error
result: `not_run` when its response ended before calls could run (`stopReason`
`"error"`, `"interrupted"` or `"max_tokens"`), otherwise `not_returned`, which
tells the model the call may have run. A result stored after a later message,
such as one a client pushed late, moves up to its call in the request.

A response cut off at the output token limit (`stopReason: "max_tokens"`) may
lack calls or text it meant to send, so none of its calls runs. Each gets an
error result with `reason: "not_run"` that asks for a shorter response, and the
model gets another turn, as for malformed calls.

A loop resumed without `input` runs the calls of the last response that have no
result, one at a time. Calls run in order and store their results in call
order, so only the first of them, and the rest of its
[parallel group](#parallel-tool-calls), can have been running when the run
stopped. Such a call of a server tool without `needsApproval` may already have
had its effect: nessi answers it with `reason: "not_returned"` instead of
running it again, unless the tool is defined with `replay: "safe"` (running it
twice does no harm, for example a read). The calls after them never started and
run normally. A call that needs approval runs only after the approval the
resumed loop receives; client tools get their request again.

Recovery allows two retries after consecutive turns with malformed or cut-off
calls, subject to `maxTurns`; a resumed loop counts the turns its history
already spent. Another turn resets this limit. If both retries fail, the loop
reports an issue and ends with `"error"`. Cancelled calls and responses stopped
by the provider do not trigger recovery.

## Parallel tool calls

A response often asks for several independent reads at once. Define such tools
with `concurrency: "parallel"`, and neighbouring calls to them run at the same
time:

```ts
const readFile = defineTool({
  name: "read_file",
  description: "Read a file",
  inputSchema: z.object({ path: z.string() }),
  concurrency: "parallel",
  replay: "safe",
}).server(async ({ path }) => fs.readFile(path, "utf8"));
```

- Tools run alone (`"serial"`) unless they are marked. Tools with
  `needsApproval` and client tools always run alone. Mark only tools that do
  not depend on each other's effects.
- Consecutive parallel calls of one response form a group; a serial call waits
  until the group before it has finished, and the next group waits for it.
- `maxParallelToolCalls` (default 4) bounds how many calls of a group run at
  once. In sampled Cloud chats, 98 % of the responses with tool calls had at
  most three.
- Results are stored in call order. A call stores its result, and sends its
  `tool_execution_end`, once every earlier call of its group has finished.
- Each call keeps its own `timeoutMs`. An abort cancels the `ctx.signal` of
  every running call.
- Nested `ctx.requestApproval()` and `ctx.requestClientTool()` requests of
  parallel calls reach the app one at a time: the next `tool_action_request`
  comes after the previous one was answered.
- `aggregate.timing.toolExecutionMs` counts a group's wall time once, without
  the time spent waiting for the app.
- The stored calls of a group carry `parallel: true`, so a resumed loop knows
  which calls may have been running, also when a tool's `concurrency` changed
  in between. Providers never see the mark.
- If a call fails in a way that ends the loop (for example a store error), the
  other calls of its group are cancelled. After an early stop, a call that
  finished but still waited for an earlier one stores nothing; a resume answers
  it as `not_returned`.

## Dynamic tools

Pass a resolver when the active tools depend on application state:

```ts
const loop = nessi({
  provider,
  systemPrompt,
  input,
  store,
  tools: async () => toolRegistry.activeFor(userId),
});
```

Nessi resolves dynamic tools before every provider turn. One copied and
validated snapshot supplies both the provider schemas and all tool execution
for that turn. Changes made during tool execution therefore become visible on
the following provider turn, while calls already emitted by the provider remain
executable from their original snapshot. Duplicate names and resolver failures
end the loop with a `runtime_error` issue.

Pending tool calls restored from history resolve one fresh snapshot before
execution because an in-memory snapshot cannot survive a restart. Keep the
resolver side-effect free; persistence, discovery, authorization, and cleanup
remain application-owned.

Server tools receive the provider tool call ID through `ctx.callId`:

```ts
const inspect = defineTool({
  name: "inspect",
  description: "Inspect an item",
  inputSchema: z.object({ id: z.string() }),
}).server(async ({ id }, ctx) => {
  return inspectItem(id, { callId: ctx.callId, signal: ctx.signal });
});
```

## Historical tool results

Verbose tool output can remain fully persisted without being sent to the model
in every later loop. A tool may derive a compact historical representation once
after its output passes validation:

```ts
const shell = defineTool({
  name: "shell",
  description: "Run a shell command.",
  inputSchema: z.object({ command: z.string() }),
  outputSchema: z.object({
    exitCode: z.number(),
    stdout: z.string(),
    changedFiles: z.array(z.string()),
  }),
  toHistoricalResult: ({ output }) => ({
    exitCode: output.exitCode,
    changedFiles: output.changedFiles,
    excerpt: output.stdout.slice(0, 500),
  }),
}).server(runShell);
```

Nessi persists the full `result` and the optional `historicalResult` together.
Provider calls in the originating loop, including resumes with the same
`loopId`, receive the full result. Calls from a different loop receive the
historical value instead. Stored messages, events, and loop aggregates remain
full and inspectable. Returning `undefined` skips the historical representation.

Derivation shares the rest of the tool's `timeoutMs`, and Nessi stops waiting
for it when the loop is aborted; the callback gets no signal and is not
cancelled. If it throws or does not finish in time, Nessi stores the full
successful result, emits a non-fatal `tool_historical_result_error` issue, and
continues the loop; the tool does not run again. Legacy messages
without `historicalResult` remain unchanged. `maxToolResultChars`, when set,
runs after this selection as the final context-size boundary.

## Steering

Use `loop.steer()` when the process handling new input owns the running loop:

```ts
loop.steer("Skip deployment and only prepare the migration.");
```

For loops running in another worker or process, provide a `steering` callback
that reads pending messages from application-owned persistence:

```ts
const loop = nessi({
  provider,
  systemPrompt,
  store,
  input,
  tools,
  steering: ({ loopId, signal }) => steeringQueue.takePending(loopId, { signal }),
});
```

The callback may return one message, an ordered array, or `undefined`. Nessi
checks it before provider calls and before a normal loop completion. Applied
messages are persisted as user messages and emit the same `steer_applied`
event as `loop.steer()`. The application owns persistence, claiming, and
delivery semantics; Nessi only controls when steering can affect the loop.

## Structured output

Use `nessi.structured()` when an app wants a schema-valid typed result instead
of a streamed chat response:

```ts
import { nessi } from "@k2b/nessi";
import { complete, openAICompatible } from "@k2b/nessi/ai";
import { z } from "zod";

const result = await nessi.structured({
  provider: openAICompatible({
    name: "openrouter",
    model: "openai/gpt-4.1-mini",
    baseURL: "https://openrouter.ai/api/v1",
    apiKey: process.env.OPENROUTER_API_KEY,
  }),
  input: "Extract a task card for: Ship the onboarding flow by Friday.",
  outputName: "task_card",
  output: z.object({
    title: z.string(),
    due: z.string().nullable(),
    priority: z.enum(["low", "medium", "high"]),
  }),
  temperature: 0,
});

console.log(result.output.title);
console.log(result.structuredMeta);
console.log(result.aggregate.usage);
```

For providers and schemas that are safe for native structured output, Nessi
passes a provider-specific `responseFormat`. Otherwise it falls back to schema
instructions and one repair attempt. Nullable fields and unions below the root
stay native. Strict providers fall back for `.optional()` properties and open
objects; vLLM guided decoding (`capabilities.strictStructuredOutput: false`)
accepts them. `input` can be a string, content parts, or
a full user message, including image file parts when the provider supports
images.

Failures throw `StructuredOutputError` with a `code`: `invalid_output`,
`truncated` (the response hit the output token limit), `provider_stopped` (the
provider ended it with an error, a refusal, or an interruption), `aborted`,
or `loop_failed`. Output from a response that did not end normally is never
accepted or repaired. Failures carry `details.aggregate` with the usage and
timing of the responses that completed.

`nessi.structured()` makes one provider call (plus at most one repair) and
takes no tools; use the full `nessi()` loop for tool work.

## Provider-only usage

```ts
import { complete, openAICompatible } from "@k2b/nessi/ai";

const provider = openAICompatible({
  name: "openrouter",
  model: "openai/gpt-4.1-mini",
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
});

const result = await complete(provider, {
  systemPrompt: "Be concise.",
  messages: [
    {
      role: "user",
      content: [{ type: "text", text: "Summarize this package." }],
    },
  ],
});

console.log(result.message.content);
```

Provider streams use the same block events as the root loop:

```ts
const textBlocks = new Set<string>();

for await (const event of provider.stream({ messages })) {
  if (event.type === "block_start" && event.kind === "text") {
    textBlocks.add(event.blockId);
  }
  if (event.type === "block_delta" && textBlocks.has(event.blockId)) {
    process.stdout.write(event.delta);
  }
  if (event.type === "block_end" && event.block.type === "tool_call") {
    console.log("tool call", event.block.name, event.block.args);
  }
  if (event.type === "issue") {
    console.error(event.issue.kind, event.issue.message);
  }
}
```

To show tool arguments while the model writes them, opt in with
`provider.stream({ messages, toolArgsPreview: true })` or pass
`toolArgsPreview: true` to root `nessi()`. Both emit the display-only event:

```ts
type ToolArgsDeltaEvent = {
  type: "tool_args_delta";
  callId: string;
  name: string;
  delta: string;
};
```

`ToolArgsDeltaEvent` is exported from `@k2b/nessi/ai` and `@k2b/nessi`.
Root events also carry `agentId`, `loopId`, `turnId`, and `turnIndex`.
Fragments arrive in provider order and may contain incomplete or invalid JSON.
They never open a block, change block indices, or become stored messages or
tool-call aggregates. Coalescing leaves these fragments unchanged. A preview
may stop without a final call and be followed by a `malformed_tool_call` or
`cancelled_tool_call` issue. Never execute from a preview; only a final
`tool_call` block is executable. Previews are off by default, and the existing
block and issue sequence stays unchanged. `complete()` ignores previews;
`nessi.structured()` does not enable them.

## Provider timeouts

Set `timeouts` in the provider options to bound waits. Unset or non-positive
values disable the corresponding timeout. Every request is a stream, so both
limits apply to `stream()` and `complete()` alike:

| Option | Bounds |
| --- | --- |
| `firstByteMs` | Request start through the first response-body chunk, which may precede the first token |
| `idleMs` | Each gap between body chunks after the first; also the read of an HTTP error body |

A slow answer that keeps sending chunks never times out; a stalled one does. A
timeout rejects with a `ProviderError` carrying `kind: "timeout"`, `scope:
"provider_first_byte"` or `"provider_idle"`, and `retryable: true`. Caller
cancellation preserves the signal's original reason. For an overall deadline,
pass a request `signal` (for example `AbortSignal.timeout(ms)`).

## Provider errors

`complete()` throws a `ProviderError` (from
`@k2b/nessi/ai`) when a provider request fails. It carries `kind`
(`"provider_error"`, `"timeout"`, or `"invalid_request"` for a request nessi
could not build and never sent), `retryable`, and, when known, `status`,
`code`, `retryAfterMs`, `contextOverflow`, `overflowRatio`, and the timeout
`scope`. Streaming `provider_error` issues carry the same `status`, `code`, and
`retryAfterMs`. Nessi never retries; a caller abort throws the signal's reason.

### Transport report

Pass `onTransport` on a request to observe the transport without patching
`fetch`. It is called once when a request nessi sent settles (answered,
rejected, failed, timed out, or aborted) and before `complete()` throws:

```ts
await complete(provider, {
  messages,
  onTransport: ({ startedAt, headersAt, firstByteAt, status, retryAfterMs, connectionCode }) => {
    metrics.record({ headersMs: headersAt && headersAt - startedAt, firstByteMs: firstByteAt && firstByteAt - startedAt, status });
  },
});
```

`headersAt` is missing when no response came; `connectionCode` names a
connection that failed before one (`ECONNREFUSED`). An error thrown in the
hook does not change the request.

## Reasoning effort

Set `reasoningEffort` to control how much a reasoning model thinks. Set it on
the provider as a default, or per call on `complete()`, `stream()`, `nessi()` or
`nessi.structured()`. A call value wins over the provider default, and nothing
is sent when neither is set, so the model keeps its own default.

```ts
const provider = openAICompatible({ name: "openai", model: "gpt-6.1-sol", baseURL: "https://api.openai.com/v1", reasoningEffort: "medium" });

const loop = nessi({ provider, systemPrompt, input, store, reasoningEffort: "high" });
const quick = await complete(provider, { messages, reasoningEffort: "none" });
```

The value is passed through unchanged, so new levels work without a Nessi
update. `"none"` turns reasoning off. Common levels are `"none"`, `"minimal"`,
`"low"`, `"medium"`, `"high"`, `"xhigh"` and `"max"`. Which levels a model
accepts is up to the model; unsupported values come back as provider errors.

`openAICompatible()` sends `reasoning_effort`; with
`compat.reasoningFormat: "openrouter"` it sends `reasoning: { effort }`.

Recent vLLM versions also pass the level to the chat template and turn off
thinking for templates such as Qwen's when it is `"none"`. On older versions,
use `extraBody: { chat_template_kwargs: { enable_thinking: false } }`.

The OpenAI-compatible adapter also reads an initial `<think>…</think>` span and
`<|open|>think<|sep|>…<|close|>think<|sep|>` markers across deltas.
Markers are removed before text and thinking blocks are emitted, including
incomplete control syntax at the end of a stream. Kimi K3 can omit the opening
marker: its ambiguous initial content is held until the closing marker,
structured reasoning, a tool call, or stream end identifies the span. Without
any marker it remains text. This can delay the first text for Kimi K3 endpoints
that send neither an opener nor structured reasoning. Regular `reasoning_content`
deltas continue streaming immediately, as does text with `reasoningEffort: "none"`.
An ambiguous Kimi prefix interrupted by an error or output limit stays thinking. `compat.thinkingFormat: "none"` hides
in-band thinking as well.

Thinking appears as `thinking` blocks. Some endpoints sign their reasoning or
return it encrypted and require it back unchanged in later requests, especially
during tool loops. Nessi keeps that data on the blocks (`signature` for
Mistral's typed `thinking` chunks, `details` for OpenRouter's
`reasoning_details`) and records the producing provider's `name` in
`message.provider`; it sends reasoning back only to the provider with that
name, as typed chunks when `compat.thinkingFormat` is `"content_chunks"`.
With `compat.toolCallSignatures: "google"`, a Gemini tool call keeps its
thought signature as `signature` and goes back with it; a call from another
provider gets Gemini's placeholder that skips the check.
Store assistant messages as they are, including these fields, and keep history
append-only.

### Extra request fields

`extraBody` adds fields to the provider's request body for parameters Nessi does
not model. Set it on the provider or per call; call values win. Plain objects
merge deeply into what Nessi sends, other values replace it:

```ts
const provider = openAICompatible({
  name: "vllm",
  model: "Qwen/Qwen3-32B",
  baseURL: "http://localhost:8000/v1",
  extraBody: { chat_template_kwargs: { enable_thinking: false } },
});
```

## Audio transcription

Use `openAICompatibleTranscription()` to upload an audio file to a service that
implements the OpenAI-compatible `/audio/transcriptions` endpoint. Configure
the service URL, API key and transcription model explicitly:

```ts
import { complete, openAICompatibleTranscription } from "@k2b/nessi/ai";

const speech = openAICompatibleTranscription("whisper-large-v3", {
  baseURL: "https://api.scaleway.ai/v1",
  apiKey: process.env.SCW_SECRET_KEY,
});

const result = await speech.transcribe({
  file: Bun.file("./aufnahme.mp3"),
  filename: "aufnahme.mp3",
  language: "de",
  signal: AbortSignal.timeout(120_000),
});

console.log(result.text);
```

For OpenAI, set `baseURL: "https://api.openai.com/v1"`, use your OpenAI key and
a supported transcription model such as `whisper-1`. Local compatible services
can omit `apiKey`. No environment variable is read automatically. Optional
`headers` support gateways; `apiKey` overrides their Authorization header.

`file` accepts a `Blob`, `File` or `Bun.file()`. Use `filename` to supply an
extension for an unnamed Blob or override the filename. Automatically derived
filenames omit local directories. Omit `language`
for automatic detection. Optional `prompt` supplies vocabulary or context
when supported by the model. File formats, size limits and optional parameter
support depend on the service; Nessi does not convert or split audio.

The result is `{ text: string }`, including an empty string for an empty
transcript. HTTP, connection and malformed-response errors reject the promise.
Pass `signal` for cancellation or a timeout; cancellation preserves the signal's
reason. There are no automatic retries or default timeout.

Transcription uses its own `TranscriptionProvider` contract with `name`, `model`
and `transcribe(request)`. Custom adapters can implement that interface for other
protocols. It is separate from the chat provider passed to `nessi()`; pass the
resulting text into the agent when needed. Timestamps and speaker
identification are not exposed; for live audio see below.

The example follows [Scaleway's audio API documentation](https://www.scaleway.com/en/docs/generative-apis/how-to/query-audio-models/).
Keep API keys on the server when integrating a browser application.

### Live transcription

`vllmRealtimeTranscription()` transcribes audio while it is still being recorded.
It streams audio to vLLM's `/v1/realtime` WebSocket and yields text as the model
recognizes it, for example with `mistralai/Voxtral-Mini-4B-Realtime-2602`:

```ts
import { complete, vllmRealtimeTranscription } from "@k2b/nessi/ai";

const speech = vllmRealtimeTranscription("voxtral-realtime", {
  baseURL: "https://vllm.example.com/v1",
  apiKey: process.env.VLLM_API_KEY,
});

let text = "";
for await (const event of speech.stream({ audio: microphoneChunks, signal })) {
  if (event.type === "delta") text += event.text;
  if (event.type === "done") console.log(event.text, event.audioMs, event.usage);
}
```

`audio` is an async iterable of mono 16-bit little-endian PCM chunks
(`Int16Array` or `Uint8Array`) at 16 kHz (`speech.sampleRate`). Send chunks as
they are recorded; the transcript is finished when the iterable ends.
`delta` events carry new text to append, and the final `done` event carries the
complete transcript, the duration of the sent audio and token usage. Nessi does
not resample audio. In a browser, an `AudioContext` created with
`{ sampleRate: 16000 }` records at the right rate.

The adapter uses the standard global `WebSocket`, so it runs in browsers, Bun,
Deno and Node 22 or later. Sending `apiKey` or `headers` requires a WebSocket that accepts
headers, which Bun does and browsers do not. Pass `webSocket: (url, headers) =>
...` to supply your own WebSocket, for example from the `ws` package. Keep API
keys on the server and relay browser audio through it.

Errors, a closed connection and a rejected handshake reject the iteration;
cancellation through `signal` uses the signal's reason and closes the connection
immediately. Stopping the iteration early closes it too. Nessi asks the audio
iterable to stop, but cannot interrupt a pending read, so stop the microphone with
the same signal. There are no automatic retries or reconnects.

## Decision models

Decision models such as Cloudflare's Clef or Kev answer
typed questions about a state with a probability for every option, in one forward
pass and without generating text. They are fast and bounded, which makes them a
good fit for routing, triage and checks before an LLM acts. Nessi speaks the
System One API (`POST /v1/systemone`) that these models share:

```ts
import { complete, systemOneDecision } from "@k2b/nessi/ai";

const router = systemOneDecision("clef-flash", {
  baseURL: "https://decide.example.com/v1",
  apiKey: process.env.DECISION_API_KEY,
});

const { answers } = await router.decide({
  state: "Checkout has been failing for every customer for the last hour.",
  questions: {
    urgent: { type: "noul", instructions: "Is this support request urgent?" },
    team: {
      type: "choice",
      instructions: "Which team should handle this request?",
      criteria: { billing: "Payments and refunds", technical: "Outages and errors", sales: "Plans" },
    },
    severity: { type: "score", instructions: "How severe is the impact?", criteria: ["None", "Minor", "Major", "Critical"] },
  },
});

answers.urgent.probability; // 0.91, `answers.urgent.value` is true at 0.5 or more
answers.team.choice;        // typed as "billing" | "technical" | "sales"
answers.team.confidence;    // probability of the chosen option
answers.severity.level;     // most likely level index, `score` is the weighted level
```

Question types:

- `noul`: yes or no; optional `criteria: { true, false }` describe both answers.
  The answer has `probability` (of yes) and `value`.
- `choice`: `criteria` maps option IDs to descriptions (or `null`). The answer
  has `choice`, `confidence` and `probabilities` per option.
- `score`: `criteria` lists ordered levels, lowest first. The answer has the
  weighted `score`, the most likely `level`, `confidence` and `probabilities`
  per level index.

Option IDs are inferred from inline questions. For questions stored in a
variable, keep the literal types with `satisfies DecisionQuestions` or
`as const`. `state` accepts text or JSON-serializable data. Optional `images`
(`{ data: base64, mediaType }`) are sent as data URLs; only multimodal models
such as Clef accept them.

The result contains `model`, the typed `answers`, `usage` and `raw`, the
unchanged provider response for fields Nessi does not map. Nessi checks that
every question has a valid answer and otherwise rejects; it does not enforce
provider limits such as the number of questions or options, which differ
between models. HTTP, connection and malformed-response errors reject the
promise, `signal` cancels with its reason, and there are no automatic retries.
Decision calls have no timeout of their own; pass one, for example
`signal: AbortSignal.timeout(5_000)`. `onTransport` reports the request once it
settles, with the same report as for model calls (see
[Transport report](#transport-report)): status, Retry-After, timings, or the
connection code.

Every provider checks its request headers before sending. An API key or header
value with a line break or another character HTTP does not allow rejects with an
error that names the header but never contains its value.

For Clef on Cloudflare Workers AI, use the preset:

```ts
import { complete, cloudflareDecision } from "@k2b/nessi/ai";

const router = cloudflareDecision("clef-flash", {
  accountId: process.env.CLOUDFLARE_ACCOUNT_ID!,
  apiToken: process.env.CLOUDFLARE_API_TOKEN!,
});
```

A decision model does not replace the agent loop; it decides before the loop
acts. For example, pick the tools for a turn and fall back to all tools when
the model is unsure:

```ts
const { answers } = await router.decide({
  state: userMessage,
  questions: {
    area: { type: "choice", instructions: "What is the request about?", criteria: { calendar: null, files: null, other: null } },
  },
});

const tools = answers.area.confidence >= 0.7 ? toolsByArea[answers.area.choice] : allTools;
const loop = nessi({ provider, store, systemPrompt, input: userMessage, tools });
```

## Focused provider imports

```ts
import { openAICompatible } from "@k2b/nessi/ai/providers/openai-compatible";
import { openAICompatibleTranscription } from "@k2b/nessi/ai/providers/openai-compatible-transcription";
import { systemOneDecision } from "@k2b/nessi/ai/providers/systemone-decision";
```

## Features

- Turn-based agent loop with canonical block streaming events
- Stable `loopId` correlation across all events from one agent loop
- `loop_start`, `turn_start`, `turn_end`, and `loop_end.aggregate` for logical response grouping
- Local `loop.steer()` and optional `steering` callbacks for steering at safe loop boundaries
- Loop timing metadata for wall time, generation time, active tool time, action wait time, and output tokens/second
- `nessi.structured()` for typed schema-valid task results
- Server tools and client tools
- Tool approval flow and explicit `tool_action_request` events
- Tool execution start/end events with per-tool `timeoutMs`
- Opt-in parallel tool calls with a bounded limit and results in call order
- Optional per-tool historical result representations for bounded future context
- Structured `issue` events for provider errors, timeouts, malformed tool streams, and tool execution failures
- Pluggable session store
- Optional history compaction
- Standalone `compact()` loop with `loop_start`, `compaction_start`, `compaction_end`, `issue`, and `loop_end` events
- Provider adapters with shared `complete()` and `stream()` APIs
- Audio transcription through configurable OpenAI-compatible services
- One chat adapter, `openAICompatible()`, with explicit options for OpenAI, OpenRouter, Mistral, vLLM, and other Chat Completions endpoints (see the skill's `providers.md`)

## Package layout

```txt
@k2b/nessi
  Agent loop, structured task helper, tools, stores, compaction, shared types

@k2b/nessi/ai
  Provider factories, provider types, complete(), stream(), responseFormat, transcribe()

@k2b/nessi/ai/providers/*
  Focused provider entrypoints
```
