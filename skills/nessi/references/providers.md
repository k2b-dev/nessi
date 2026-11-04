# Provider Setup

nessi has one chat adapter, `openAICompatible()`, for any endpoint that speaks
OpenAI Chat Completions. Explicit `compat` options cover the differences between
endpoints. It returns a `Provider`:

```ts
const provider = openAICompatible({ name, model, baseURL, apiKey, compat });
await provider.complete(request);
for await (const event of provider.stream(request)) {}
```

## Audio transcription

Use `openAICompatibleTranscription(model, options)` from `@k2b/nessi/ai` for
speech-to-text file uploads. It returns a separate `TranscriptionProvider` with
`name`, `model` and `transcribe(request)`, not a chat `Provider`.

```ts
import { openAICompatibleTranscription } from "@k2b/nessi/ai";

const speech = openAICompatibleTranscription("whisper-large-v3", {
  baseURL: "https://api.scaleway.ai/v1",
  apiKey: process.env.SCW_SECRET_KEY,
});
const { text } = await speech.transcribe({
  file: Bun.file("./aufnahme.mp3"),
  filename: "aufnahme.mp3",
  language: "de",
  signal: AbortSignal.timeout(120_000),
});
```

- `baseURL` is required. The adapter appends `/audio/transcriptions` and uploads
  multipart form data requesting a JSON result. It reads no environment variables.
- Change URL, key and model for other compatible services. For OpenAI use
  `https://api.openai.com/v1` and a transcription model such as `whisper-1`.
- `apiKey` is optional for local services. `headers` supports gateway headers;
  an explicit API key overrides Authorization. Fetch supplies the multipart boundary.
- `file` accepts `Blob`, `File` and `Bun.file()`. Supply `filename` for unnamed
  Blobs that need an extension; otherwise the basename is preserved without local directories.
- `language` is optional (for example `de`); omission allows detection.
  `prompt` is an optional context hint subject to model support.
- Results contain only `{ text: string }`. Errors reject; cancellation uses the
  signal reason. There is no default timeout or automatic retry.
- Do not promise timestamps, speaker identification, transcoding or automatic
  file splitting. Formats, limits and optional fields depend on the service.
- For other protocols, implement the exported `TranscriptionProvider` interface.
  Pass transcript text into the agent; do not pass this provider to `nessi()`.

## Live transcription

Use `vllmRealtimeTranscription(model, options)` from `@k2b/nessi/ai` when text
should appear while the user speaks. It talks to vLLM's `/v1/realtime`
WebSocket, for example with `mistralai/Voxtral-Mini-4B-Realtime-2602`.

```ts
import { vllmRealtimeTranscription } from "@k2b/nessi/ai";

const speech = vllmRealtimeTranscription("voxtral-realtime", {
  baseURL: "https://vllm.example.com/v1",
  apiKey: process.env.VLLM_API_KEY,
});

for await (const event of speech.stream({ audio: pcm16Chunks, signal })) {
  if (event.type === "delta") appendToDraft(event.text);
  if (event.type === "done") saveTranscript(event.text, event.audioMs);
}
```

- `audio` is an async iterable of mono PCM16 little-endian chunks
  (`Int16Array` or `Uint8Array`) at 16 kHz (`speech.sampleRate`).
  Nessi does not resample; a browser `AudioContext({ sampleRate: 16000 })` does.
- The transcript ends when the audio iterable ends. `delta` carries new text,
  `done` the full text, `audioMs` and token `usage`.
- The default is the global `WebSocket`. `apiKey`/`headers` need a runtime whose
  WebSocket accepts headers (Bun). Otherwise pass `webSocket: (url, headers) => ...`.
  Keep keys on the server and relay browser audio through it.
- Errors, early close and rejected handshakes reject; `signal` cancels with its
  reason and closes the connection at once; breaking out of the loop closes it
  too. Stop the microphone with the same signal. No retries.
- A cleanup or correction pass over the final text is application logic: run it
  with `complete()` or `nessi.structured()` on a chat provider.

## Decision models

Use `systemOneDecision(model, { baseURL, apiKey? })` from `@k2b/nessi/ai` for
decision models behind the System One API (`POST {baseURL}/systemone`), e.g.
self-hosted Clef, Kev or jev-local, or hosted Jev. Use
`cloudflareDecision(model, { accountId, apiToken })` for Clef on Workers AI.
Both return a `DecisionProvider` with `decide()`, not a chat `Provider`.

```ts
const { answers, usage, raw } = await router.decide({
  state: ticketText,
  questions: {
    urgent: { type: "noul", instructions: "Is this urgent?" },
    team: { type: "choice", instructions: "Which team?", criteria: { billing: "Payments", technical: "Outages" } },
    severity: { type: "score", instructions: "How severe?", criteria: ["None", "Minor", "Major"] },
  },
  signal,
});
if (answers.team.confidence > 0.8) route(answers.team.choice); // "billing" | "technical"
```

- Answers: `noul` has `probability` and `value`; `choice` has `choice`,
  `confidence`, `probabilities`; `score` has `score`, `level`, `confidence`,
  `probabilities` (array by level). `raw` is the unchanged provider response.
- Option IDs are typed from inline questions; for variables use
  `satisfies DecisionQuestions` or `as const`.
- Use decisions before or around the agent loop (pick tools, triage, guard);
  there is no loop-level decision hook. Choose thresholds per use case.
- Invalid or missing answers, HTTP and connection errors reject; no retries.
  Limits (questions, options, levels, images, languages) depend on the model.

## Structured output support

Most consumers should use root `nessi.structured()` for typed structured
results. It validates the result with Zod, uses native provider structured
output where available, and falls back to schema instructions plus one repair
attempt.

Nullable fields and unions (`anyOf`) below the root work natively; a union at
the root, `oneOf`, `allOf`, and `not` fall back. Nessi treats every adapter
except vLLM as strict: objects must be closed and every property required, so
`.optional()`, `z.looseObject()`, and `z.record()` fall back. Use `.nullable()`
for fields that may be missing. vLLM guided decoding sets
`capabilities.strictStructuredOutput: false` and accepts optional properties and
open objects natively.

The provider layer also exposes a low-level `responseFormat` request option for
apps that intentionally want a single provider call:

```ts
const result = await provider.complete({
  messages: [{ role: "user", content: [{ type: "text", text: "Extract a card." }] }],
  responseFormat: {
    type: "json_schema",
    name: "card",
    schema: {
      type: "object",
      properties: { title: { type: "string" } },
      required: ["title"],
    },
  },
});
```

Provider mappings:

- OpenAI/OpenRouter/OpenAI-compatible: `response_format.json_schema`
- vLLM: `structured_outputs.json`
- Ollama: top-level `format` schema
- Anthropic: `output_config.format`
- Mistral: `response_format.json_schema`
- Gemini: `generationConfig.responseMimeType` plus `responseJsonSchema`

## Reasoning effort and extra request fields

- `reasoningEffort` works on the provider options and per call (`complete`,
  `stream`, `nessi()`, `nessi.structured()`); the call wins, and nothing is
  sent when unset. Values pass through unchanged; `"none"` is off. Do not add
  model lists: unsupported levels come back as provider errors.
- Wire format: `reasoning_effort` by default; `compat.reasoningFormat:
  "openrouter"` sends `reasoning: { effort }`.
- Signed or encrypted reasoning stays on blocks (`signature`, OpenRouter
  `details`) and `message.provider` records the producer (`name`). Persist
  assistant messages unchanged and append-only so the endpoint gets them back
  intact.
- `extraBody` (provider and call) merges extra fields into the request body;
  plain objects merge deeply. Use it for parameters nessi does not model, e.g.
  vLLM `chat_template_kwargs: { enable_thinking: false }`.
- `disableReasoning` is deprecated and keeps its old mapping; prefer
  `reasoningEffort: "none"`.

## Chat endpoints

```ts
import { openAICompatible } from "@k2b/nessi/ai";

const provider = openAICompatible({
  name: "mistral",
  model: "mistral-medium-latest",
  baseURL: "https://api.mistral.ai/v1",
  apiKey: process.env.MISTRAL_API_KEY,
  compat: {
    toolCallIdPolicy: "strict9",
    requiresToolResultName: true,
    thinkingFormat: "content_chunks",
    supportsUsageInStreaming: false,
    maxTokensField: "max_tokens",
    structuredOutput: "response_format",
  },
});
```

`name` labels errors and `message.provider`; reasoning is replayed only to the
provider with the same `name`.

| Endpoint | `baseURL` | `compat` |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `maxTokensField: "max_completion_tokens"`, `structuredOutput: "response_format"`, `thinkingFormat: "none"` |
| OpenRouter | `https://openrouter.ai/api/v1` | `thinkingFormat: "reasoning_details"`, `reasoningFormat: "openrouter"`, `maxTokensField: "max_tokens"`, `structuredOutput: "response_format"`; optional `HTTP-Referer` and `X-Title` via `headers` |
| Mistral | `https://api.mistral.ai/v1` | as in the example: 9-character tool call IDs, tool results with `name`, reasoning as typed `thinking` chunks, no `stream_options` |
| vLLM, Scaleway | e.g. `http://localhost:8000/v1` | `structuredOutput: "vllm_structured_outputs"` (guided decoding, not strict), `maxTokensField: "max_tokens"`, `thinkingFormat: "text"` |

Other Chat Completions endpoints work the same way, for example Ollama's
`/v1`. Responses may carry `content` as a string or as typed `text` and
`thinking` chunks; reasoning arrives from `reasoning_details`, `reasoning`, or
`reasoning_content`, one representation per frame. Consume it through
`block_start` (`kind: "thinking"`), `block_delta`, and `block_end`; set
`thinkingFormat: "none"` to hide it. Do not log reasoning content.

The other options:

- `contextWindow` when the app wants overflow heuristics.
- `creditsPerInputToken` and `creditsPerOutputToken` for cost accounting.
- `headers` for extra request headers.
- `timeouts` as below.

**Migrating from 0.19:** the `openai()`, `openrouter()`, `vllm()`, `mistral()`,
`anthropic()`, `gemini()`, and `ollama()` factories are gone. Use
`openAICompatible()` with the settings above; `normalizeToolCallIds: "strict9"`
becomes `compat.toolCallIdPolicy: "strict9"`. Anthropic, Gemini, and Ollama
offer OpenAI-compatible endpoints; their native features (Anthropic or Gemini
signed thinking, Ollama `num_ctx`) are not available there.

## Provider timeouts

`openAICompatible()` accepts `timeouts` in its options. Unset or non-positive
values disable the corresponding timeout.

- `stream()` and `completeFromStream()` use `firstByteMs` from request start
  through the first response-body chunk, which may precede the first token.
  Subsequent body-read gaps use `idleMs`. These calls do not use `totalMs`.
- Native `complete()` uses `totalMs` from request start through the entire
  response body. This includes generation before response headers arrive.
  `idleMs` bounds each body read after headers, including the first read and
  HTTP error bodies. Each chunk resets the idle timer, while the total
  deadline keeps running. Native calls do not use `firstByteMs`.

When upgrading from 0.17.1, add `totalMs` to retain a deadline for native
`complete()` calls. Without it, waiting for native response headers has no
provider timeout. A native timeout rejects with a `ProviderError` carrying
`kind: "timeout"`, `scope: "provider_total"` or `"provider_idle"`, and
`retryable: true`. Pass a request `signal` for caller-controlled cancellation;
its reason is preserved. These options apply to chat providers, not the
separate transcription or decision APIs.

## Provider selection heuristics

- One adapter for every endpoint: pick the endpoint, then copy its `compat` row.
- Prefer the endpoint's own base URL over a gateway unless the app needs routing.
- Use `structuredOutput: "vllm_structured_outputs"` only for guided-decoding servers.
