import type { DecisionProvider } from "../decision.js";
import { createDecisionProvider, decisionErrorMessage } from "./systemone-decision.js";

export type CloudflareDecisionOptions = {
  accountId: string;
  apiToken: string;
  /** Override for AI Gateway or tests. Defaults to the Cloudflare API. */
  baseURL?: string;
  name?: string;
};

/** Clef decision models on Cloudflare Workers AI (`clef` or `clef-flash`). */
export const cloudflareDecision = (
  model: "clef" | "clef-flash" | (string & {}),
  options: CloudflareDecisionOptions,
): DecisionProvider => {
  const name = options.name ?? "cloudflare";
  const baseURL = (options.baseURL ?? "https://api.cloudflare.com/client/v4").replace(/\/+$/, "");
  return createDecisionProvider({
    name,
    model,
    url: `${baseURL}/accounts/${encodeURIComponent(options.accountId)}/ai/run/@cf/cloudflare/${model}`,
    headers: { Authorization: `Bearer ${options.apiToken}` },
    // Workers AI wraps the System One response as { result, success, errors }.
    unwrap: (body) => {
      if (typeof body !== "object" || body === null || !("result" in body)) return body;
      if ("success" in body && body.success === false) {
        throw new Error(`${name} decision failed: ${decisionErrorMessage(body) ?? "unknown error"}`);
      }
      return body.result;
    },
  });
};
