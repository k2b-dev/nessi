import type { ByteStreamReader } from "./sse.js";

export const parseNDJSON = async function* <T>(
  reader: ByteStreamReader,
  timeouts: { firstByteMs?: number; idleMs?: number } = {},
): AsyncGenerator<T> {
  const decoder = new TextDecoder();
  let buffer = "";
  let readCount = 0;

  const readWithTimeout = async () => {
    const scope = readCount === 0 ? "provider_first_byte" : "provider_idle";
    const timeoutMs = readCount === 0 ? timeouts.firstByteMs : timeouts.idleMs;
    if (!timeoutMs || timeoutMs <= 0) return reader.read();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        reader.read(),
        new Promise<never>((_, reject) => {
          timeout = setTimeout(() => {
            const error = {
              scope,
              message: `NDJSON stream ${scope === "provider_first_byte" ? "first byte" : "idle"} timeout after ${timeoutMs}ms.`,
            };
            void reader.cancel?.(error).catch(() => {});
            reject(error);
          }, timeoutMs);
        }),
      ]);
    } finally {
      if (timeout) clearTimeout(timeout);
    }
  };

  while (true) {
    const { done, value } = await readWithTimeout();
    readCount++;
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        yield JSON.parse(trimmed) as T;
      } catch {
        // silently skip malformed records
      }
    }
  }

  const trimmed = buffer.trim();
  if (trimmed) {
    try {
      yield JSON.parse(trimmed) as T;
    } catch {
      // silently skip trailing malformed records
    }
  }
};
