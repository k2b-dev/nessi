import type {
  CompactEvent,
  CompactLoop,
  CompactOptions,
  CompactResult,
} from "./types.js";
import { createLoopId, zeroUsage, toErrorMessage } from "./utils.js";

/**
 * Run compaction as a loop-style operation so consumers can iterate or subscribe to events.
 */
export const compact = (options: CompactOptions): CompactLoop => {
  const {
    agentId = "main",
    loopId: requestedLoopId,
    store,
    provider,
    compact: compactFn,
    usage = zeroUsage(),
    force = true,
    signal: externalSignal,
  } = options;

  const subscribers: Array<(event: CompactEvent) => void> = [];
  const abortController = new AbortController();
  const loopId = requestedLoopId?.trim() ? requestedLoopId : createLoopId();
  const eventFields = { agentId, loopId };

  if (externalSignal) {
    if (externalSignal.aborted) abortController.abort();
    else externalSignal.addEventListener("abort", () => abortController.abort(), { once: true });
  }

  const signal = abortController.signal;

  const mkResult = (applied: boolean, entriesBefore: number, entriesAfter: number): CompactResult => ({
    applied,
    entriesBefore,
    entriesAfter,
    forced: force,
  })

  async function* run(): AsyncGenerator<CompactEvent> {
    let entriesBefore = 0;
    yield { type: "loop_start", ...eventFields };

    try {
      const entries = await store.load();
      entriesBefore = entries.length;

      if (signal.aborted) {
        yield { type: "loop_end", ...eventFields, reason: "aborted", result: mkResult(false, entriesBefore, entriesBefore) };
        return;
      }

      const operation = compactFn({
        entries,
        store,
        provider,
        usage,
        force,
        signal,
      });

      if (!operation) {
        yield { type: "loop_end", ...eventFields, reason: "stop", result: mkResult(false, entriesBefore, entriesBefore) };
        return;
      }

      yield { type: "compaction_start", ...eventFields };
      try {
        await operation;
      } finally {
        yield { type: "compaction_end", ...eventFields };
      }

      const entriesAfter = (await store.load()).length;
      yield {
        type: "loop_end",
        ...eventFields,
        reason: signal.aborted ? "aborted" : "stop",
        result: mkResult(true, entriesBefore, entriesAfter),
      };
    } catch (err) {
      const message = toErrorMessage(err);
      const entriesAfter = await store.load().then(e => e.length).catch(() => entriesBefore);
      yield {
        type: "issue",
        ...eventFields,
        issue: { kind: "runtime_error", message, retryable: false },
      };
      yield {
        type: "loop_end",
        ...eventFields,
        reason: signal.aborted ? "aborted" : "error",
        result: mkResult(false, entriesBefore, entriesAfter),
      };
    }
  }

  const generator = run();

  return {
    [Symbol.asyncIterator]() {
      return {
        async next() {
          const result = await generator.next();
          if (!result.done && result.value) {
            for (const subscriber of subscribers) subscriber(result.value);
          }
          return result;
        },
        async return(value?: CompactEvent) {
          return generator.return(value as CompactEvent);
        },
        async throw(err?: unknown) {
          return generator.throw(err);
        },
      };
    },
    subscribe(listener: (event: CompactEvent) => void) {
      subscribers.push(listener);
      return () => {
        const idx = subscribers.indexOf(listener);
        if (idx >= 0) subscribers.splice(idx, 1);
      };
    },
    abort() {
      abortController.abort();
    },
  };
}
