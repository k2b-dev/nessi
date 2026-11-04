import { describe, expect, it } from "bun:test";
import { compact } from "../src/compact.js";
import { nessi } from "../src/nessi.js";
import { memoryStore } from "../src/stores.js";
import type { ProviderRequest } from "../src/types.js";
import { mockProvider } from "./mock-provider.js";

/** A caller signal that counts its attached abort listeners. */
const countedSignal = () => {
  const controller = new AbortController();
  const { signal } = controller;
  const listeners = new Set<unknown>();
  const add = signal.addEventListener.bind(signal);
  const remove = signal.removeEventListener.bind(signal);
  signal.addEventListener = (...args: Parameters<typeof add>) => {
    if (args[0] === "abort") listeners.add(args[1]);
    add(...args);
  };
  signal.removeEventListener = (...args: Parameters<typeof remove>) => {
    if (args[0] === "abort") listeners.delete(args[1]);
    remove(...args);
  };
  return { controller, signal, active: () => listeners.size };
};

const answer = () => mockProvider([
  { type: "text", delta: "Done." },
  { type: "usage", usage: { input: 1, output: 1, total: 2 }, finishReason: "stop" },
]);

describe("caller abort signals", () => {
  it("keep no listener from finished nessi loops", async () => {
    const caller = countedSignal();
    for (let run = 0; run < 25; run++) {
      for await (const _event of nessi({ provider: answer(), store: memoryStore(), systemPrompt: "test", input: "Hi", signal: caller.signal })) {
        // drain
      }
    }
    expect(caller.active()).toBe(0);
  });

  it("keep no listener from loops the caller stops iterating", async () => {
    const caller = countedSignal();
    for (let run = 0; run < 5; run++) {
      for await (const event of nessi({ provider: answer(), store: memoryStore(), systemPrompt: "test", input: "Hi", signal: caller.signal })) {
        if (event.type === "turn_start") break;
      }
    }
    expect(caller.active()).toBe(0);
  });

  it("keep no listener from finished compactions", async () => {
    const caller = countedSignal();
    for (let run = 0; run < 10; run++) {
      for await (const _event of compact({ store: memoryStore(), provider: answer(), compact: () => null, signal: caller.signal })) {
        // drain
      }
    }
    expect(caller.active()).toBe(0);
  });

  it("pass the caller's abort reason to the provider request", async () => {
    const caller = countedSignal();
    const reason = new Error("caller stopped");
    let request: ProviderRequest | undefined;
    const provider = mockProvider([{ type: "text", delta: "Hi" }], {
      onRequest: (providerRequest) => {
        request = providerRequest;
        caller.controller.abort(reason);
      },
    });
    for await (const _event of nessi({ provider, store: memoryStore(), systemPrompt: "test", input: "Hi", signal: caller.signal })) {
      // drain
    }
    expect(request?.signal?.reason).toBe(reason);
  });
});
