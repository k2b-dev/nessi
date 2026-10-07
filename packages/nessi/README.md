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
import { ollama } from "@k2b/nessi/ai";
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
  provider: ollama("llama3.1", {
    baseURL: "http://localhost:11434",
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

`turn_end` reports each internal provider turn. The final `loop_end` event includes `aggregate`, which groups assistant turns, executable tool calls, tool results, validation/execution errors, malformed or cancelled tool streams, summed usage, and timing for the complete logical loop. `aggregate.timing.totalElapsedMs` is model generation plus active tool execution; approval/client-tool waits are tracked separately as `aggregate.timing.actionWaitMs`. Helper exports such as `mergeUsage()`, `cloneLoopAggregate()`, and `mergeLoopAggregates()` are available from `@k2b/nessi`.

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

`nessi.structured()` accepts the same pattern for server tools:

```ts
const result = await nessi.structured({
  provider,
  input: "Resolve the current task.",
  output: taskSchema,
  tools: () => toolRegistry.activeServerTools(),
});
```

A structured tool resolver always selects `tool_loop` mode, even when a
snapshot is empty, because later turns may add tools. Nessi appends its internal
`submit_result` tool to every snapshot. Client tools, approval tools, and a
user-defined `submit_result` remain unsupported. A static empty array keeps the
direct native/fallback structured-output path.

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

If derivation throws, Nessi stores the full successful result, emits a non-fatal
`tool_historical_result_error` issue, and continues the loop. Legacy messages
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
import { openrouter } from "@k2b/nessi/ai";
import { z } from "zod";

const result = await nessi.structured({
  provider: openrouter("openai/gpt-4.1-mini", {
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
instructions and one repair attempt. `input` can be a string, content parts, or
a full user message, including image file parts when the provider supports
images.

`nessi.structured()` can use server tools for bounded task work. It adds an
internal `submit_result` tool and returns after that tool receives a valid
schema value. Client tools, approval tools, and interactive tool bridges remain
the job of the full `nessi()` loop.

## Provider-only usage

```ts
import { openrouter } from "@k2b/nessi/ai";

const provider = openrouter("openai/gpt-4.1-mini", {
  apiKey: process.env.OPENROUTER_API_KEY,
});

const result = await provider.complete({
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

## Reasoning effort

Set `reasoningEffort` to control how much a reasoning model thinks. Set it on
the provider as a default, or per call on `complete()`, `stream()`, `nessi()` or
`nessi.structured()`. A call value wins over the provider default, and nothing
is sent when neither is set, so the model keeps its own default.

```ts
const provider = openai("gpt-6.1-sol", { reasoningEffort: "medium" });

const loop = nessi({ provider, systemPrompt, input, store, reasoningEffort: "high" });
const quick = await provider.complete({ messages, reasoningEffort: "none" });
```

The value is passed through unchanged, so new levels work without a Nessi
update. `"none"` turns reasoning off. Common levels are `"none"`, `"minimal"`,
`"low"`, `"medium"`, `"high"`, `"xhigh"` and `"max"`. Which levels a model
accepts is up to the model; unsupported values come back as provider errors.

| Provider | Sent as |
|---|---|
| `openai`, `vllm`, `openAICompatible()` | `reasoning_effort` |
| `openrouter` | `reasoning: { effort }` |
| `ollama` | `think`; `"none"` sends `false` |

On vLLM, `"none"` also turns off thinking for templates such as Qwen's. Ollama
models that only accept `think: true` or `false` need `extraBody: { think: true }`
to turn thinking on.

`disableReasoning` is deprecated. It keeps its original behavior: OpenAI-compatible
providers send `reasoning_effort: "low"`, Gemini sets a zero thinking budget and
other providers ignore it. It is ignored when the call sets `reasoningEffort`.

### Extra request fields

`extraBody` adds fields to the provider's request body for parameters Nessi does
not model. Set it on the provider or per call; call values win. Plain objects
merge deeply into what Nessi sends, other values replace it:

```ts
const provider = vllm("Qwen/Qwen3-32B", {
  baseURL: "http://localhost:8000/v1",
  extraBody: { chat_template_kwargs: { enable_thinking: false } },
});
```

## Audio transcription

Use `openAICompatibleTranscription()` to upload an audio file to a service that
implements the OpenAI-compatible `/audio/transcriptions` endpoint. Configure
the service URL, API key and transcription model explicitly:

```ts
import { openAICompatibleTranscription } from "@k2b/nessi/ai";

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
import { vllmRealtimeTranscription } from "@k2b/nessi/ai";

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
import { systemOneDecision } from "@k2b/nessi/ai";

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

For Clef on Cloudflare Workers AI, use the preset:

```ts
import { cloudflareDecision } from "@k2b/nessi/ai";

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
import { anthropic } from "@k2b/nessi/ai/providers/anthropic";
import { openai } from "@k2b/nessi/ai/providers/openai";
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
- Optional per-tool historical result representations for bounded future context
- Structured `issue` events for provider errors, timeouts, malformed tool streams, and tool execution failures
- Pluggable session store
- Optional history compaction
- Standalone `compact()` loop with `loop_start`, `compaction_start`, `compaction_end`, `issue`, and `loop_end` events
- Optional token-credit budgeting
- Provider adapters with shared `complete()` and `stream()` APIs
- Audio transcription through configurable OpenAI-compatible services
- Native adapters for OpenAI, OpenRouter, vLLM, Ollama, Anthropic, Mistral, and Gemini

## Package layout

```txt
@k2b/nessi
  Agent loop, structured task helper, tools, stores, compaction, shared types

@k2b/nessi/ai
  Provider factories, provider types, complete(), stream(), responseFormat, transcribe()

@k2b/nessi/ai/providers/*
  Focused provider entrypoints
```
