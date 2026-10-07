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
import { formatConnectionError } from "../shared/errors.js";

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

/** Shared System One client; presets differ only in URL, headers and response envelope. */
export const createDecisionProvider = (endpoint: DecisionEndpoint): DecisionProvider => ({
  name: endpoint.name,
  model: endpoint.model,
  async decide<const TQuestions extends DecisionQuestions>(
    { state, questions, images, signal }: DecisionRequest<TQuestions>,
  ): Promise<DecisionResult<TQuestions>> {
    signal?.throwIfAborted();
    const { name } = endpoint;
    const response = await fetch(endpoint.url, {
      method: "POST",
      headers: { ...endpoint.headers, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: endpoint.model,
        state,
        questions,
        ...(images?.length ? { images: images.map((image) => `data:${image.mediaType};base64,${image.data}`) } : {}),
      }),
      signal,
    }).catch((error: unknown) => {
      signal?.throwIfAborted();
      throw new Error(formatConnectionError(name, error));
    });

    const text = await response.text();
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      if (!response.ok) throw new Error(`${name} ${response.status}: ${text || response.statusText}`);
      throw new Error(`${name} returned invalid decision JSON.`);
    }
    if (!response.ok) {
      throw new Error(`${name} ${response.status}: ${decisionErrorMessage(body) ?? (text || response.statusText)}`);
    }

    const raw = endpoint.unwrap ? endpoint.unwrap(body) : body;
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
