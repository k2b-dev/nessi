import { describe, it, expect } from "bun:test";
import { compact } from "../src/compact.js";
import { memoryStore } from "../src/stores.js";
import { mockProvider } from "./mock-provider.js";
import type { CompactEvent, Message } from "../src/types.js";

const provider = mockProvider([]);

function userMessage(text: string): Message {
  return { role: "user", content: [{ type: "text", text }] };
}

async function collectEvents(loop: ReturnType<typeof compact>): Promise<CompactEvent[]> {
  const events: CompactEvent[] = [];
  for await (const event of loop) {
    events.push(event);
  }
  return events;
}

describe("compact loop", () => {
  it("returns loop_end(stop) when compact function skips", async () => {
    const store = memoryStore();
    await store.append(userMessage("a"));
    await store.append(userMessage("b"));

    const events = await collectEvents(compact({
      store,
      provider,
      compact: () => null,
    }));

    expect(events.map((event) => event.type)).toEqual(["loop_start", "loop_end"]);
    expect(new Set(events.map((event) => event.loopId)).size).toBe(1);
    const done = events[1];
    expect(done.type).toBe("loop_end");
    if (done.type !== "loop_end") return;
    expect(done.reason).toBe("stop");
    expect(done.result.applied).toBe(false);
    expect(done.result.entriesBefore).toBe(2);
    expect(done.result.entriesAfter).toBe(2);
  });

  it("emits compaction start/end and loop_end on success", async () => {
    const store = memoryStore();
    await store.append(userMessage("1"));
    await store.append(userMessage("2"));
    await store.append(userMessage("3"));

    const events = await collectEvents(compact({
      store,
      provider,
      compact: (ctx) => {
        return ctx.store.append(userMessage("summary"), { seq: 2, kind: "summary" });
      },
    }));

    expect(events.map((e) => e.type)).toEqual(["loop_start", "compaction_start", "compaction_end", "loop_end"]);
    expect(new Set(events.map((event) => event.loopId)).size).toBe(1);
    const done = events[3];
    if (done.type !== "loop_end") return;
    expect(done.reason).toBe("stop");
    expect(done.result.applied).toBe(true);
    expect(done.result.entriesBefore).toBe(3);
    expect(done.result.entriesAfter).toBe(2);
  });

  it("emits issue and loop_end(error) when compact throws", async () => {
    const store = memoryStore();
    await store.append(userMessage("x"));

    const events = await collectEvents(compact({
      store,
      provider,
      compact: () => {
        throw new Error("boom");
      },
    }));

    expect(events.map((e) => e.type)).toEqual(["loop_start", "issue", "loop_end"]);
    expect(new Set(events.map((event) => event.loopId)).size).toBe(1);
    const err = events[1];
    expect(err.type).toBe("issue");
    if (err.type !== "issue") return;
    expect(err.issue.kind).toBe("runtime_error");
    expect(err.issue.message).toContain("boom");

    const done = events[2];
    if (done.type !== "loop_end") return;
    expect(done.reason).toBe("error");
    expect(done.result.applied).toBe(false);
  });

  it("emits compaction_end before issue when compaction rejects asynchronously", async () => {
    const store = memoryStore();
    await store.append(userMessage("x"));

    const events = await collectEvents(compact({
      store,
      provider,
      compact: async () => {
        await Promise.resolve();
        throw new Error("async boom");
      },
    }));

    expect(events.map((event) => event.type)).toEqual([
      "loop_start",
      "compaction_start",
      "compaction_end",
      "issue",
      "loop_end",
    ]);
    const issue = events[3];
    expect(issue.type).toBe("issue");
    if (issue.type !== "issue") return;
    expect(issue.issue.message).toContain("async boom");
  });

  it("returns loop_end(aborted) when signal is already aborted", async () => {
    const store = memoryStore();
    await store.append(userMessage("x"));

    const controller = new AbortController();
    controller.abort();

    const events = await collectEvents(compact({
      store,
      provider,
      compact: () => null,
      signal: controller.signal,
    }));

    expect(events.map((event) => event.type)).toEqual(["loop_start", "loop_end"]);
    const done = events[1];
    if (done.type !== "loop_end") return;
    expect(done.reason).toBe("aborted");
    expect(done.result.applied).toBe(false);
  });

  it("uses a provided loopId on every event", async () => {
    const store = memoryStore();
    await store.append(userMessage("x"));

    const events = await collectEvents(compact({
      loopId: "compact-1",
      store,
      provider,
      compact: () => null,
    }));

    expect(events.map((event) => event.loopId)).toEqual(["compact-1", "compact-1"]);
  });

});
