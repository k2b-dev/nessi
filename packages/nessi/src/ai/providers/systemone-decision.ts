import type {
  ChoiceQuestion,
  DecisionAnswer,
  DecisionAnswers,
  DecisionProvider,
  DecisionQuestion,
  DecisionQuestions,
  DecisionRequest,
  DecisionResult,
  ScoreQuestion,
  SystemOneResponse,
} from "../decision.js";
import { assertHeaderValues, formatConnectionError, parseRetryAfterMs, readErrorBody, runtimeErrorCode } from "../shared/errors.js";
import type { TransportReport } from "../types.js";

export type SystemOneDecisionOptions = {
  /** Base URL including `/v1`; `/systemone` is appended. */
  baseURL: string;
  apiKey?: string;
  headers?: Record<string, string>;
  name?: string;
};

type DecisionEndpoint = {
  name: string;
  model: string;
  url: string;
  headers: Record<string, string>;
  /** Extracts the System One response from a provider-specific envelope. */
  unwrap?: (body: unknown) => unknown;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isProbability = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;

/** Finds a readable message in the common error body shapes (OpenAI, Cloudflare, FastAPI). */
export const decisionErrorMessage = (body: unknown): string | undefined => {
  if (!isRecord(body)) return undefined;
  const firstError = Array.isArray(body.errors) ? body.errors[0] : undefined;
  if (isRecord(firstError) && typeof firstError.message === "string") return firstError.message;
  if (isRecord(body.error) && typeof body.error.message === "string") return body.error.message;
  for (const key of ["error", "detail", "message"] as const) {
    if (typeof body[key] === "string") return body[key];
  }
  return undefined;
};

const probabilityOf = (probabilities: unknown, key: string) =>
  isRecord(probabilities) && isProbability(probabilities[key]) ? probabilities[key] : 0;

const choiceAnswer = (question: ChoiceQuestion, raw: Record<string, unknown>) => {
  const options = Object.keys(question.criteria);
  if (typeof raw.choice !== "string" || !options.includes(raw.choice)) return undefined;
  const probabilities = Object.fromEntries(options.map((option) => [option, probabilityOf(raw.probabilities, option)]));
  return {
    type: "choice" as const,
    choice: raw.choice,
    confidence: isProbability(raw.confidence) ? raw.confidence : probabilities[raw.choice]!,
    probabilities,
  };
};

const scoreAnswer = (question: ScoreQuestion, raw: Record<string, unknown>) => {
  if (!isRecord(raw.probabilities) && typeof raw.score !== "number") return undefined;
  const probabilities = question.criteria.map((_, level) => probabilityOf(raw.probabilities, String(level)));
  const expected = probabilities.reduce((sum, probability, level) => sum + level * probability, 0);
  const score = typeof raw.score === "number" && Number.isFinite(raw.score) ? raw.score : expected;
  if (score < 0 || score > question.criteria.length - 1) return undefined;
  const best = Math.max(...probabilities);
  const level = best > 0 ? probabilities.indexOf(best) : Math.round(score);
  return {
    type: "score" as const,
    score,
    level,
    confidence: isProbability(raw.confidence) ? raw.confidence : probabilities[level]!,
    probabilities,
  };
};

const toAnswer = (question: DecisionQuestion, raw: unknown) => {
  if (!isRecord(raw) || raw.type !== question.type) return undefined;
  if (question.type === "noul") {
    return isProbability(raw.noul) ? { type: "noul" as const, probability: raw.noul, value: raw.noul >= 0.5 } : undefined;
  }
  return question.type === "choice" ? choiceAnswer(question, raw) : scoreAnswer(question, raw);
};

/**
 * Reads a whole response body and notes when its first byte arrived. An abort cancels the read, also
 * when the body ignores the request's signal, and rejects with the caller's reason.
 */
const readBody = async (response: Response, onFirstByte: () => void, signal?: AbortSignal): Promise<string> => {
  const reader = response.body?.getReader();
  if (!reader) return "";
  const cancel = () => void reader.cancel(signal?.reason).catch(() => {});
  signal?.addEventListener("abort", cancel, { once: true });
  const decoder = new TextDecoder();
  let text = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value.byteLength > 0) onFirstByte();
      text += decoder.decode(value, { stream: true });
    }
  } finally {
    signal?.removeEventListener("abort", cancel);
  }
  signal?.throwIfAborted();
  return text + decoder.decode();
};

/** Shared System One client; presets differ only in URL, headers and response envelope. */
export const createDecisionProvider = (endpoint: DecisionEndpoint): DecisionProvider => ({
  name: endpoint.name,
  model: endpoint.model,
  async decide<const TQuestions extends DecisionQuestions>(
    { state, questions, images, signal, onTransport }: DecisionRequest<TQuestions>,
  ): Promise<DecisionResult<TQuestions>> {
    signal?.throwIfAborted();
    const { name } = endpoint;
    const headers = { ...endpoint.headers, "Content-Type": "application/json" };
    assertHeaderValues(name, headers);
    // Built first: a request that cannot be serialized is never sent and reports nothing.
    const body = JSON.stringify({
      model: endpoint.model,
      state,
      questions,
      ...(images?.length ? { images: images.map((image) => `data:${image.mediaType};base64,${image.data}`) } : {}),
    });
    // One report per request that left, with the same fields as for model calls. It reaches the hook
    // once the request settles, before the result or the error reaches the caller.
    const report: TransportReport = { startedAt: Date.now() };
    const noteFirstByte = () => {
      report.firstByteAt ??= Date.now();
    };
    try {
      const response = await fetch(endpoint.url, {
        method: "POST",
        headers,
        body,
        redirect: "error",
        signal,
      }).catch((error: unknown) => {
        const code = signal?.aborted ? undefined : runtimeErrorCode(error);
        if (code !== undefined) report.connectionCode = code;
        signal?.throwIfAborted();
        throw new Error(formatConnectionError(name, error));
      });
      report.headersAt = Date.now();
      report.status = response.status;
      const retryAfterMs = parseRetryAfterMs(response.headers);
      if (retryAfterMs !== undefined) report.retryAfterMs = retryAfterMs;

      const text = response.ok
        ? await readBody(response, noteFirstByte, signal)
        : await readErrorBody(response, { signal, onFirstByte: noteFirstByte });
      signal?.throwIfAborted();
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        if (!response.ok) throw new Error(`${name} ${response.status}: ${text || response.statusText}`);
        throw new Error(`${name} returned invalid decision JSON.`);
      }
      if (!response.ok) {
        throw new Error(`${name} ${response.status}: ${decisionErrorMessage(parsed) ?? (text || response.statusText)}`);
      }

      const raw = endpoint.unwrap ? endpoint.unwrap(parsed) : parsed;
      if (!isRecord(raw) || !isRecord(raw.answers)) throw new Error(`${name} returned a decision without answers.`);
      const result = raw as SystemOneResponse & { answers: Record<string, unknown> };

      const answers: Record<string, DecisionAnswer> = {};
      for (const [id, question] of Object.entries(questions)) {
        const answer = toAnswer(question, result.answers[id]);
        if (!answer) throw new Error(`${name} returned an invalid answer for question "${id}".`);
        answers[id] = answer;
      }

      const input = result.usage?.input_tokens ?? 0;
      const output = result.usage?.output_tokens ?? 0;
      return {
        model: typeof result.model === "string" ? result.model : endpoint.model,
        // Each answer was validated against its question above.
        answers: answers as DecisionAnswers<TQuestions>,
        usage: { input, output, total: input + output },
        raw: result,
      };
    } finally {
      try {
        onTransport?.(report);
      } catch {
        // Observation only: a failing hook must not change the request's outcome.
      }
    }
  },
});

/**
 * Typed decisions from any model behind the System One `/v1/systemone` API, such as
 * self-hosted Clef or Kev servers, or a hosted Jev endpoint.
 */
export const systemOneDecision = (model: string, options: SystemOneDecisionOptions): DecisionProvider => {
  const headers: Record<string, string> = { ...options.headers };
  if (options.apiKey) {
    for (const key of Object.keys(headers)) if (key.toLowerCase() === "authorization") delete headers[key];
    headers.Authorization = `Bearer ${options.apiKey}`;
  }
  return createDecisionProvider({
    name: options.name ?? "systemone",
    model,
    url: `${options.baseURL.replace(/\/+$/, "")}/systemone`,
    headers,
  });
};
