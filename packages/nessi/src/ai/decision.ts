import type { Usage } from "./types.js";

// ----------------------------------------------------------------------------
// Questions
// ----------------------------------------------------------------------------

/** Question or option text. Decision models also accept structured JSON here. */
export type DecisionText = string | Record<string, unknown> | readonly unknown[];

/** A yes/no question. */
export type NoulQuestion = {
  type: "noul";
  instructions: DecisionText;
  /** Optional descriptions of what yes and no mean. */
  criteria?: { true?: DecisionText; false?: DecisionText };
};

/** Pick one of the named options. Keys are the option IDs returned as `choice`. */
export type ChoiceQuestion = {
  type: "choice";
  instructions: DecisionText;
  criteria: Record<string, DecisionText | null>;
};

/** Rate on an ordered scale, lowest level first. Levels are indexed from 0. */
export type ScoreQuestion = {
  type: "score";
  instructions: DecisionText;
  criteria: readonly DecisionText[];
};

export type DecisionQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion;

/** Questions keyed by ID. Answers are returned under the same IDs. */
export type DecisionQuestions = Record<string, DecisionQuestion>;

// ----------------------------------------------------------------------------
// Answers
// ----------------------------------------------------------------------------

export type NoulAnswer = {
  type: "noul";
  /** Probability that the answer is yes. */
  probability: number;
  /** `probability >= 0.5`. Compare `probability` with your own threshold when the cost of errors differs. */
  value: boolean;
};

export type ChoiceAnswer<TOption extends string = string> = {
  type: "choice";
  /** The most likely option. */
  choice: TOption;
  /** Probability of `choice`. */
  confidence: number;
  /** Probability per option; values sum to 1. */
  probabilities: Record<TOption, number>;
};

export type ScoreAnswer = {
  type: "score";
  /** Probability-weighted level; can land between levels. */
  score: number;
  /** Index of the most likely level. */
  level: number;
  /** Probability of `level`. */
  confidence: number;
  /** Probability per level, indexed like the question's `criteria`. */
  probabilities: number[];
};

export type DecisionAnswer<TQuestion extends DecisionQuestion = DecisionQuestion> =
  TQuestion extends { type: "noul" } ? NoulAnswer
    : TQuestion extends { type: "choice"; criteria: infer TCriteria } ? ChoiceAnswer<Extract<keyof TCriteria, string>>
      : TQuestion extends { type: "score" } ? ScoreAnswer
        : never;

export type DecisionAnswers<TQuestions extends DecisionQuestions> = {
  [TId in keyof TQuestions]: DecisionAnswer<TQuestions[TId]>;
};

// ----------------------------------------------------------------------------
// Requests and providers
// ----------------------------------------------------------------------------

/** Base64 image without a data URL prefix. Image support depends on the model. */
export type DecisionImage = { data: string; mediaType: string };

export type DecisionRequest<TQuestions extends DecisionQuestions> = {
  /** What to decide about: text or JSON-serializable data. */
  state: unknown;
  questions: TQuestions;
  images?: DecisionImage[];
  signal?: AbortSignal;
};

/** Response body of the System One `/v1/systemone` API, as sent by the provider. */
export type SystemOneResponse = {
  model?: string;
  answers?: Record<string, unknown>;
  usage?: { input_tokens?: number; output_tokens?: number };
  [key: string]: unknown;
};

export type DecisionResult<TQuestions extends DecisionQuestions> = {
  model: string;
  answers: DecisionAnswers<TQuestions>;
  usage: Usage;
  /** The unmodified provider response. */
  raw: SystemOneResponse;
};

export type DecisionProvider = {
  name: string;
  model: string;
  decide<const TQuestions extends DecisionQuestions>(
    request: DecisionRequest<TQuestions>,
  ): Promise<DecisionResult<TQuestions>>;
};
