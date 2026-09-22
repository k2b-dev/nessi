# Provider Setup

Chat provider constructors return a `Provider` with the same consumer-facing interface:

```ts
const provider = providerFactory(model, options);
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
- Do not promise timestamps, speaker identification, streaming, transcoding or
  automatic file splitting. Formats, limits and optional fields depend on the service.
- For other protocols, implement the exported `TranscriptionProvider` interface.
  Pass transcript text into the agent; do not pass this provider to `nessi()`.

## Structured output support

Most consumers should use root `nessi.structured()` for typed structured
results. It validates the result with Zod, uses native provider structured
output where available, and falls back to schema instructions plus one repair
attempt.

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

## Hosted OpenAI

```ts
import { openai } from "@k2b/nessi/ai";

const provider = openai("gpt-4.1-mini", {
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
});
```

Useful options:

- `baseURL` for compatible gateways
- `contextWindow` when the app wants overflow heuristics
- `creditsPerInputToken` and `creditsPerOutputToken` for cost accounting
- `normalizeToolCallIds: "strict9"` for providers that need short alphanumeric tool IDs
- `timeouts.firstByteMs` and `timeouts.idleMs` for streaming timeout policy

## OpenRouter

```ts
import { openrouter } from "@k2b/nessi/ai";

const provider = openrouter("openai/gpt-4.1-mini", {
  apiKey: process.env.OPENROUTER_API_KEY,
  referer: "https://example.com",
  title: "Example App",
});
```

OpenRouter is a good default when the app needs model routing or easy model swapping. Reasoning-capable models can appear as normalized `thinking` blocks in stream events.

## vLLM and custom OpenAI-compatible endpoints

```ts
import { vllm, openAICompatible } from "@k2b/nessi/ai";

const localVllm = vllm("meta-llama/Llama-3.1-8B-Instruct", {
  baseURL: "http://localhost:8000/v1",
});

const custom = openAICompatible({
  name: "internal-gateway",
  model: "company-model",
  baseURL: "https://ai.example.com/v1",
  apiKey: process.env.INTERNAL_AI_KEY,
  compat: {
    supportsUsageInStreaming: true,
    maxTokensField: "max_tokens",
  },
});
```

Use `openAICompatible` when a gateway follows Chat Completions semantics closely enough but needs explicit compatibility flags.

Since `@k2b/nessi` 0.12.1, generic OpenAI-compatible streams recognize
`delta.reasoning_content` without an extra compatibility setting. The default
priority is readable `reasoning_details`, then `reasoning`, then
`reasoning_content`. `thinkingFormat: "text"` prefers `reasoning`, then
`reasoning_content`, with readable details as a fallback. Only one representation
is emitted per frame, so parallel fields do not duplicate thinking text.

Consume these through the existing `block_start` (`kind: "thinking"`),
`block_delta`, and `block_end` events; see [Messages and streaming](messages-and-streaming.md).
To hide thinking, explicitly set `thinkingFormat: "none"`. The `vllm()` preset
still uses this opt-out; use `openAICompatible()` when you want thinking from a
vLLM endpoint. Do not log reasoning content or raise timeouts to compensate for
missing visible progress.

Use `timeouts` for vLLM/OpenAI-compatible streams that may stall or emit malformed partial tool calls:

```ts
const localVllm = vllm("Qwen/Qwen3-32B", {
  baseURL: "http://localhost:8000/v1",
  timeouts: {
    firstByteMs: 30_000,
    idleMs: 15_000,
  },
});
```

## Ollama

```ts
import { ollama } from "@k2b/nessi/ai";

const provider = ollama("llama3.1", {
  baseURL: "http://localhost:11434",
  temperature: 0.2,
});
```

Ollama is useful for local development and offline workflows. It streams NDJSON internally, but consumers still receive normalized `StreamEvent` values. It supports the same `timeouts.firstByteMs` and `timeouts.idleMs` streaming controls.

## Anthropic

```ts
import { anthropic } from "@k2b/nessi/ai";

const provider = anthropic("claude-sonnet", {
  apiKey: process.env.ANTHROPIC_API_KEY,
  maxOutputTokens: 1024,
});
```

Anthropic uses native content blocks for tool use. Consumers still receive normalized assistant content and `block_end` events for final `tool_call` blocks.

## Mistral

```ts
import { mistral } from "@k2b/nessi/ai";

const provider = mistral("mistral-small-latest", {
  apiKey: process.env.MISTRAL_API_KEY,
  normalizeToolCallIds: "strict9",
});
```

Mistral looks OpenAI-like but has enough tool-call differences to use its native adapter. Keep tool-call IDs short if the backend requires it.

## Gemini

```ts
import { gemini } from "@k2b/nessi/ai";

const provider = gemini("gemini-2.0-flash", {
  apiKey: process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY,
  maxOutputTokens: 1024,
});
```

Gemini supports native multimodal input and function calls. `disableReasoning: true` maps to a zero thinking budget.

## Provider selection heuristics

- Choose `ollama` for local-only prototypes.
- Choose `openrouter` for model choice and hosted routing.
- Choose `openai` for direct OpenAI billing and behavior.
- Choose `anthropic`, `gemini`, or `mistral` when the user explicitly wants that vendor's native API behavior.
- Choose `vllm` for local or self-hosted OpenAI-compatible serving.
- Choose `openAICompatible` for a custom gateway where explicit compatibility flags matter.
