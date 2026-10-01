<p align="center">
  <img src="./packages/nessi-ui/public/logo.svg" alt="nessi" width="96" />
</p>

# nessi

Minimal agent stack for provider adapters, an event-driven loop, and a browser UI.

The monorepo is intentionally split into two focused packages:

- `@k2b/nessi` for the published agent loop and provider API
- `nessi-ui` for the browser client

`@k2b/nessi` replaces the deprecated `@valentinkolb/nessi` package. The public
API and subpaths are unchanged; migrate dependencies and imports by replacing
the scope. Future UI container releases use `ghcr.io/k2b-dev/nessi-ui`.

## Packages

### `@k2b/nessi`

Agent loop at the package root, provider layer under `/ai`.

For audio files, `/ai` also exports `openAICompatibleTranscription()` with a
separate `transcribe()` API. Configure the endpoint and model for Scaleway,
OpenAI or another compatible service. For live audio, `vllmRealtimeTranscription()`
streams text while the user speaks. See [audio transcription](packages/nessi/README.md#audio-transcription).

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

### `nessi-ui`

Browser-first reference client built on `@k2b/nessi`.

```bash
bun install
bun --filter nessi-ui dev
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
bun run dev
bun run build
```

## Repository Layout

```txt
packages/
  nessi/       Published package: agent loop plus /ai provider API
  nessi-ui/    Browser UI, settings, local persistence, Docker setup
```

## Skill

This repo also ships a `nessi` skill that teaches coding agents how to build with `@k2b/nessi`.

```bash
bunx skills add https://github.com/k2b-dev/nessi
```

## Notes

- `@k2b/nessi` is the reusable library package.
- `nessi-ui` is the reference application.
- The UI Docker build lives at [`packages/nessi-ui/Dockerfile`](./packages/nessi-ui/Dockerfile).
