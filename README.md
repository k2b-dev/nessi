<p align="center">
  <img src="./assets/logo.svg" alt="nessi" width="96" />
</p>

# nessi

Minimal agent stack: provider adapters and an event-driven loop, published as
`@k2b/nessi`.

`@k2b/nessi` replaces the deprecated `@valentinkolb/nessi` package; migrate
dependencies and imports by replacing the scope.

## Packages

### `@k2b/nessi`

Agent loop at the package root, provider layer under `/ai`.

For audio files, `/ai` also exports `openAICompatibleTranscription()` with a
separate `transcribe()` API. Configure the endpoint and model for Scaleway,
OpenAI or another compatible service. For live audio, `vllmRealtimeTranscription()`
streams text while the user speaks. See [audio transcription](packages/nessi/README.md#audio-transcription).

For fast typed routing and checks, `systemOneDecision()` and `cloudflareDecision()`
query decision models such as Clef through the System One API. See
[decision models](packages/nessi/README.md#decision-models).

```ts
import { nessi, memoryStore } from "@k2b/nessi";
import { openrouter } from "@k2b/nessi/ai";

const provider = openrouter("openai/gpt-4.1-mini", {
  apiKey: process.env.OPENROUTER_API_KEY,
});

const loop = nessi({
  provider,
  systemPrompt: "You are concise.",
  input: "Summarize this repo.",
  store: memoryStore(),
});

for await (const event of loop) {
  if (event.type === "loop_end") {
    console.log(event.aggregate.usage);
    console.log(event.aggregate.timing);
    console.log(event.aggregate.toolIssues);
  }
}
```

## Development

Install dependencies once:

```bash
bun install
```

Useful commands from the repository root:

```bash
bun run typecheck
bun run test
bun run build:nessi
```

## Repository Layout

```txt
packages/
  nessi/       Published package: agent loop plus /ai provider API
```

## Skill

This repo also ships a `nessi` skill that teaches coding agents how to build with `@k2b/nessi`.

```bash
bunx skills add https://github.com/k2b-dev/nessi
```

## Notes

The browser reference client `nessi-ui` was removed after `@k2b/nessi` 0.19.0;
the tag `nessi-v0.19.0` keeps its last state.
