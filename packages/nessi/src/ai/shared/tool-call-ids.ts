const ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

const hash32 = (input: string) => {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

const encodeBase62 = (seed: number, length: number) => {
  let n = seed >>> 0;
  let out = "";
  for (let i = 0; i < length; i++) {
    if (n === 0) n = hash32(`${seed}:${i}:${out.length}`);
    out += ALNUM[n % ALNUM.length];
    n = Math.floor(n / ALNUM.length);
  }
  return out;
};

const createStrictToolCallIdFactory = () => {
  const used = new Set<string>();
  let seq = 0;

  return (seed: string) => {
    let attempt = 0;
    while (attempt < 50_000) {
      const candidate = encodeBase62(hash32(`${seed}:${seq}:${attempt}`), 9);
      if (!used.has(candidate)) {
        used.add(candidate);
        seq++;
        return candidate;
      }
      attempt++;
    }
    throw new Error("Failed to generate unique strict tool call id");
  };
};

/**
 * Prefix for tool call IDs that a provider did not supply. It is unique per response, so
 * fallback IDs such as `ollama-<prefix>-0` never repeat across turns of a conversation.
 */
const fallbackToolCallPrefix = (label: string) => {
  const random = globalThis.crypto?.randomUUID?.().replace(/-/g, "")
    ?? `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
  return `${label}-${random.slice(0, 8)}`;
};

export { ALNUM, hash32, encodeBase62, createStrictToolCallIdFactory, fallbackToolCallPrefix };
