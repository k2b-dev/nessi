import type { GenerateRequest, ProviderRequestDefaults, ReasoningEffort } from "../types.js";

/** The effort to send: the request's, else the provider default, else undefined (model default). */
export const resolveReasoning = (request: GenerateRequest, defaults?: ProviderRequestDefaults): ReasoningEffort | undefined =>
  request.reasoningEffort ?? defaults?.reasoningEffort;

/** Only object literals (or null-prototype objects) merge; dates, class instances and arrays replace. */
const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (typeof value !== "object" || value === null) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
};

const mergeInto = (target: Record<string, unknown>, extra: Record<string, unknown>) => {
  for (const [key, value] of Object.entries(extra)) {
    if (key === "__proto__") continue;
    const current = target[key];
    target[key] = isPlainObject(current) && isPlainObject(value) ? mergeInto({ ...current }, value) : value;
  }
  return target;
};

/** Applies provider and request `extraBody` over a request body. Plain objects merge deeply. */
export const withExtraBody = (
  body: Record<string, unknown>,
  request: GenerateRequest,
  defaults?: ProviderRequestDefaults,
): Record<string, unknown> => {
  if (!defaults?.extraBody && !request.extraBody) return body;
  const merged = mergeInto({ ...body }, defaults?.extraBody ?? {});
  return mergeInto(merged, request.extraBody ?? {});
};
