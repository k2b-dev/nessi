import type {
  CompactEvent,
  CompactLoop,
  CompactOptions,
  CompactResult,
} from "./types.js";
import { createLoopId, followAbortSignal, zeroUsage, toErrorMessage } from "./utils.js";

/**
 * Run compaction as a loop-style operation whose events consumers iterate.
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

  const abortController = new AbortController();
  const loopId = requestedLoopId?.trim() ? requestedLoopId : createLoopId();
  const eventFields = { agentId, loopId };

  // Attached on the first next(), so a run that never starts leaves no listener behind.
  let detachExternal: (() => void) | undefined;
  let detached = false;
  const attachExternalSignal = () => {
    if (!detached) detachExternal ??= followAbortSignal(externalSignal, abortController);
  };
  const detachExternalSignal = () => {
    detached = true;
    detachExternal?.();
  };

  const signal = abortController.signal;

  const mkResult = (applied: boolean, entriesBefore: number, entriesAfter: number): CompactResult => ({
    applied,
    entriesBefore,
    entriesAfter,
    forced: force,
  })

  async function* run(): AsyncGenerator<CompactEvent> {
    try {
      yield* compactOnce();
    } finally {
      detachExternalSignal();
    }
  }

  async function* compactOnce(): AsyncGenerator<CompactEvent> {
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
          attachExternalSignal();
          return generator.next();
        },
        async return(value?: CompactEvent) {
          detachExternalSignal();
          return generator.return(value as CompactEvent);
        },
        async throw(err?: unknown) {
          detachExternalSignal();
          return generator.throw(err);
        },
      };
    },
    abort() {
      abortController.abort();
    },
  };
}
