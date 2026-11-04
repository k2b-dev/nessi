# @k2b/nessi Quickstart

`@k2b/nessi` exposes an agent loop at the package root and a provider adapter layer under `@k2b/nessi/ai`.

Use the root package when the app needs a managed agent loop with tools, storage, approvals, or compaction. Use the `/ai` subpath when the app only needs provider calls.

The provider layer gives applications a common `Provider` interface with:

- `complete(request)` for one-shot generation
- `stream(request)` for streaming generation

The provider layer does not run an agent loop, persist messages, or execute tools.
For schema-valid typed task results, prefer `nessi.structured()` from the root
package over hand-parsing `provider.complete()` text.

## Install and import

In a consuming project, install the package once:

```bash
bun add @k2b/nessi
```

Import providers from the `/ai` subpath:

```ts
import { openrouter } from "@k2b/nessi/ai";
```

Import agent-loop helpers from the package root:

```ts
import { nessi, defineTool, memoryStore } from "@k2b/nessi";
```

Loop aggregate helpers are also available from the root package:

```ts
import { mergeUsage, cloneLoopAggregate, mergeLoopAggregates } from "@k2b/nessi";
```

Provider constructors use this shape:

```ts
const provider = openrouter("openai/gpt-4.1-mini", {
  apiKey: process.env.OPENROUTER_API_KEY,
});
```

## Agent loop with a tool

```ts
import { nessi, defineTool, memoryStore } from "@k2b/nessi";
import { openrouter } from "@k2b/nessi/ai";
import { z } from "zod";

const searchDocs = defineTool({
  name: "search_docs",
  description: "Search internal documentation.",
  inputSchema: z.object({ query: z.string() }),
}).server(async ({ query }) => {
  return { results: [`Result for ${query}`] };
});

const loop = nessi({
  loopId: crypto.randomUUID(),
  provider: openrouter("openai/gpt-4.1-mini", {
    apiKey: process.env.OPENROUTER_API_KEY,
  }),
  systemPrompt: "Answer from the available tools when useful.",
  input: "Find the docs for streaming events.",
  store: memoryStore(),
  tools: [searchDocs],
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
    console.error("loop id", event.loopId);
    console.error("finish", event.reason);
    console.error("loop usage", event.aggregate.usage);
    console.error("loop timing", event.aggregate.timing);
  }
}
```

Every outbound event from one root `nessi()` run carries the same `loopId`.
Pass an application request or response-group id when you have one, or use the
generated `event.loopId` if omitted.

`turn_end` is emitted for each internal provider turn. A single user request
with tools can have multiple internal turns. Use the final `loop_end.aggregate`
for one logical response group, aggregate usage, timing, assistant-turn count,
tool-call count, tool-error count, and structured issues:

```ts
if (event.type === "loop_end") {
  const aggregate = event.aggregate;
  console.log(aggregate.assistantMessageCount);
  console.log(aggregate.toolCallCount, aggregate.toolErrorCount);
  console.log(aggregate.toolMalformedCount, aggregate.toolCancelledCount);
  console.log(aggregate.toolIssues);
  console.log(aggregate.issues);
  console.log(aggregate.usage);
  console.log(aggregate.timing);
}
```

`aggregate.timing.totalElapsedMs` is `generationMs + toolExecutionMs`; it
excludes `actionWaitMs` for approval/client-tool waits. `outputTokensPerSecond`
is based on output tokens divided by `generationMs`, not wall time.

If your app persists chat UI state, persist the `loop_end.aggregate` payload with
`event.loopId` on your response group. Nessi owns the generic loop semantics;
your app still owns database IDs and storage schema.

## Steer a running loop

Call `loop.steer()` when the current process owns the loop. If user input can
arrive through another request, process, or worker, provide a callback that
reads pending messages from your application-owned storage:

```ts
const loop = nessi({
  loopId: requestId,
  provider,
  systemPrompt,
  store,
  input,
  tools,
  steering: ({ loopId, signal }) => steeringQueue.takePending(loopId, { signal }),
});

loop.steer("Also check the rollback plan.");

for await (const event of loop) {
  if (event.type === "steer_applied") {
    console.log("Applied steering:", event.message);
  }
}
```

The callback may return a string, an ordered array of strings, or `undefined`.
Nessi checks for steering before provider calls and once more before normal
completion, so input arriving during the final response can continue the loop.
It does not poll while generation or tools are running. The application owns
how pending messages are persisted and consumed.

## Structured output task

Use `nessi.structured()` when the app needs a typed object validated by Zod:

```ts
import { nessi } from "@k2b/nessi";
import { openrouter } from "@k2b/nessi/ai";
import { z } from "zod";

const result = await nessi.structured({
  provider: openrouter("openai/gpt-4.1-mini", {
    apiKey: process.env.OPENROUTER_API_KEY,
  }),
  input: "Extract a task from: Ship the onboarding flow by Friday.",
  outputName: "task",
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

`input` may be a string, content parts, or a full user message. Content parts
can include image file parts when the selected provider supports images.

When the provider and schema are safe for native structured output, Nessi passes
`responseFormat` to the adapter. Otherwise it adds schema instructions and
performs one repair attempt if the first response is invalid.

Failures throw `StructuredOutputError` with a `code`: `invalid_output`,
`truncated` (the response hit the output token limit), `provider_stopped` (the
provider ended it with an error, a refusal, or an interruption), `aborted`,
`loop_failed`, `max_turns`, or `unsupported_tool`. A response that did not end
normally is never accepted or repaired. `details.aggregate` carries the usage
of every attempt that ran.

`nessi.structured()` may use server tools:

```ts
const lookupCustomer = defineTool({
  name: "lookup_customer",
  description: "Find a customer by email.",
  inputSchema: z.object({ email: z.string().email() }),
}).server(async ({ email }) => {
  return { email, tier: "pro" };
});

const result = await nessi.structured({
  provider,
  input: "Find ada@example.com and return the account summary.",
  output: z.object({
    email: z.string().email(),
    tier: z.string(),
  }),
  tools: [lookupCustomer],
  maxTurns: 6,
});
```

Only pass server tools that do not need approval. Use the full `nessi()` loop
for client tools, approvals, UI actions, or custom interactive tool bridges.

## One-shot completion

```ts
import { openrouter } from "@k2b/nessi/ai";

const provider = openrouter("openai/gpt-4.1-mini", {
  apiKey: process.env.OPENROUTER_API_KEY,
  temperature: 0.2,
});

const result = await provider.complete({
  systemPrompt: "You write compact technical summaries.",
  messages: [
    {
      role: "user",
      content: [{ type: "text", text: "Explain what SSE is in one paragraph." }],
    },
  ],
  maxOutputTokens: 300,
});

const text = result.message.content
  .filter((block) => block.type === "text")
  .map((block) => block.text)
  .join("");

console.log(text);
console.log(result.usage);
```

## Streaming text

```ts
import { ollama } from "@k2b/nessi/ai";

const provider = ollama("llama3.1", {
  baseURL: "http://localhost:11434",
  temperature: 0,
});

const messages = [
  {
    role: "user" as const,
    content: [{ type: "text" as const, text: "Write a haiku about Bun." }],
  },
];

const textBlocks = new Set<string>();

for await (const event of provider.stream({ messages })) {
  switch (event.type) {
    case "block_start":
      if (event.kind === "text") textBlocks.add(event.blockId);
      break;
    case "block_delta":
      if (textBlocks.has(event.blockId)) process.stdout.write(event.delta);
      break;
    case "usage":
      console.error("\nusage", event.usage);
      break;
    case "issue":
      console.error(event.issue.kind, event.issue.message);
      break;
  }
}
```

## Minimal provider switcher

```ts
import { ollama, openrouter, type Provider } from "@k2b/nessi/ai";

function createProvider(name: "local" | "hosted"): Provider {
  if (name === "local") {
    return ollama("llama3.1", { baseURL: "http://localhost:11434" });
  }

  return openrouter("openai/gpt-4.1-mini", {
    apiKey: process.env.OPENROUTER_API_KEY,
  });
}
```

## Common mistakes

- Do not call provider constructors with an object containing `model`; use `(model, options?)`.
- Do not put hosted provider API keys in browser code.
- Do not expect `@k2b/nessi/ai` to execute tools; it emits normalized tool-call data.
- Do not assume every provider supports every content type, tool behavior, or thinking event.
