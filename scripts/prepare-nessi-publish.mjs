import { access, copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const packageRoot = resolve(repoRoot, "packages/nessi");
const distRoot = resolve(packageRoot, "dist");

const sourcePackage = JSON.parse(
  await readFile(resolve(packageRoot, "package.json"), "utf8"),
);

// The published subpaths are the source manifest's, so a removed module cannot stay exported.
const providerExports = Object.keys(sourcePackage.exports ?? {})
  .filter((subpath) => subpath.startsWith("./ai/providers/"))
  .map((subpath) => subpath.slice("./ai/providers/".length));

for (const name of providerExports) {
  for (const extension of [".js", ".d.ts"]) {
    const target = resolve(distRoot, "ai/providers", `${name}${extension}`);
    await access(target).catch(() => {
      throw new Error(`Exported subpath ./ai/providers/${name} has no build output at ${target}.`);
    });
  }
}

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
    "openai-compatible",
    "openrouter",
    "mistral",
    "vllm",
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
