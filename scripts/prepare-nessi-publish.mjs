import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const packageRoot = resolve(repoRoot, "packages/nessi");
const distRoot = resolve(packageRoot, "dist");

const sourcePackage = JSON.parse(
  await readFile(resolve(packageRoot, "package.json"), "utf8"),
);

const providerExports = [
  "openai-compatible-transcription",
  "openai-compatible",
  "openai",
  "openrouter",
  "vllm",
  "ollama",
  "anthropic",
  "mistral",
  "gemini",
];

const packageJson = {
  name: sourcePackage.name,
  version: sourcePackage.version,
  description: sourcePackage.description,
  main: "index.js",
  module: "index.js",
  types: "index.d.ts",
  type: "module",
  sideEffects: false,
  exports: {
    ".": {
      import: "./index.js",
      types: "./index.d.ts",
    },
    "./ai": {
      import: "./ai/index.js",
      types: "./ai/index.d.ts",
    },
    ...Object.fromEntries(
      providerExports.map((name) => [
        `./ai/providers/${name}`,
        {
          import: `./ai/providers/${name}.js`,
          types: `./ai/providers/${name}.d.ts`,
        },
      ]),
    ),
  },
  files: ["**/*"],
  keywords: [
    "ai",
    "llm",
    "agent",
    "tools",
    "streaming",
    "openai",
    "openrouter",
    "ollama",
    "anthropic",
    "gemini",
    "mistral",
  ],
  author: "Valentin Kolb",
  license: "MIT",
  repository: {
    type: "git",
    url: "git+https://github.com/k2b-dev/nessi.git",
  },
  publishConfig: sourcePackage.publishConfig ?? {
    access: "public",
  },
  dependencies: sourcePackage.dependencies,
};

await mkdir(distRoot, { recursive: true });
await writeFile(
  resolve(distRoot, "package.json"),
  `${JSON.stringify(packageJson, null, 2)}\n`,
);
await copyFile(resolve(packageRoot, "README.md"), resolve(distRoot, "README.md"));
await copyFile(resolve(repoRoot, "LICENSE"), resolve(distRoot, "LICENSE"));
