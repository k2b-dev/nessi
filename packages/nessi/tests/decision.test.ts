import { afterEach, describe, expect, it } from "bun:test";
import {
  cloudflareDecision,
  systemOneDecision,
  type ChoiceAnswer,
  type DecisionProvider,
  type DecisionQuestions,
  type NoulAnswer,
  type ScoreAnswer,
} from "../src/ai/index.js";

const servers: ReturnType<typeof Bun.serve>[] = [];
afterEach(() => {
  for (const server of servers.splice(0)) server.stop(true);
});

const requests: Array<{ path: string; headers: Headers; body: Record<string, unknown> }> = [];
afterEach(() => {
  requests.length = 0;
});

/** HTTP server that records requests and replies with `reply(body)`. */
const endpoint = (reply: (body: Record<string, unknown>) => Response) => {
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    async fetch(request) {
      const body = (await request.json()) as Record<string, unknown>;
      requests.push({ path: new URL(request.url).pathname, headers: request.headers, body });
      return reply(body);
    },
  });
  servers.push(server);
  return `http://127.0.0.1:${server.port}`;
};

const ticketQuestions = {
  urgent: { type: "noul", instructions: "Is this support request urgent?" },
  team: {
    type: "choice",
    instructions: "Which team should handle this request?",
    criteria: { billing: "Payments and refunds", technical: "Outages and errors", sales: null },
  },
  severity: { type: "score", instructions: "How severe is the impact?", criteria: ["None", "Minor", "Major", "Critical"] },
} as const;

const systemOneAnswers = {
  urgent: { type: "noul", noul: 0.91 },
  team: { type: "choice", choice: "technical", confidence: 0.86, probabilities: { billing: 0.1, technical: 0.86, sales: 0.04 } },
  severity: {
    type: "score",
    score: 2.3,
    confidence: 0.6,
    legend: { 0: "None", 1: "Minor", 2: "Major", 3: "Critical" },
    probabilities: { 0: 0.02, 1: 0.08, 2: 0.48, 3: 0.42 },
  },
};

const decideTicket = (provider: DecisionProvider) =>
  provider.decide({ state: "Checkout fails for every customer.", questions: ticketQuestions });

describe("System One decisions", () => {
  it("sends the System One request and normalizes typed answers", async () => {
    const response = { model: "clef-flash", answers: systemOneAnswers, usage: { input_tokens: 42, output_tokens: 0 }, extra: true };
    const baseURL = endpoint(() => Response.json(response));
    const provider = systemOneDecision("clef-flash", { baseURL: `${baseURL}/v1/`, apiKey: "test-key" });

    const result = await provider.decide({
      state: { ticket: "Checkout fails" },
      questions: ticketQuestions,
      images: [{ data: "aGVsbG8=", mediaType: "image/png" }],
    });

    expect(result).toEqual({
      model: "clef-flash",
      answers: {
        urgent: { type: "noul", probability: 0.91, value: true },
        team: { type: "choice", choice: "technical", confidence: 0.86, probabilities: { billing: 0.1, technical: 0.86, sales: 0.04 } },
        severity: { type: "score", score: 2.3, level: 2, confidence: 0.6, probabilities: [0.02, 0.08, 0.48, 0.42] },
      },
      usage: { input: 42, output: 0, total: 42 },
      raw: response,
    });
    expect(requests[0]?.path).toBe("/v1/systemone");
    expect(requests[0]?.headers.get("authorization")).toBe("Bearer test-key");
    expect(requests[0]?.body).toEqual({
      model: "clef-flash",
      state: { ticket: "Checkout fails" },
      questions: ticketQuestions,
      images: ["data:image/png;base64,aGVsbG8="],
    });
  });

  it("derives missing confidence and treats missing option probabilities as zero", async () => {
    const baseURL = endpoint(() => Response.json({
      answers: {
        ...systemOneAnswers,
        team: { type: "choice", choice: "billing", probabilities: { billing: 0.7, technical: 0.3 } },
        severity: { type: "score", probabilities: { 0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4 } },
      },
    }));

    const { answers, model, usage } = await decideTicket(systemOneDecision("kev-4b", { baseURL }));

    expect(model).toBe("kev-4b");
    expect(usage).toEqual({ input: 0, output: 0, total: 0 });
    expect(answers.team).toEqual({ type: "choice", choice: "billing", confidence: 0.7, probabilities: { billing: 0.7, technical: 0.3, sales: 0 } });
    expect(answers.severity.level).toBe(3);
    expect(answers.severity.confidence).toBe(0.4);
    expect(answers.severity.score).toBeCloseTo(2);
  });

  it.each([
    ["a missing answer", { urgent: systemOneAnswers.urgent, team: systemOneAnswers.team }, "severity"],
    ["a choice outside the options", { ...systemOneAnswers, team: { type: "choice", choice: "legal" } }, "team"],
    ["a mismatched answer type", { ...systemOneAnswers, urgent: { type: "choice", choice: "yes" } }, "urgent"],
    ["a probability outside 0..1", { ...systemOneAnswers, urgent: { type: "noul", noul: 1.5 } }, "urgent"],
    ["a score without values", { ...systemOneAnswers, severity: { type: "score" } }, "severity"],
    ["a score beyond the last level", { ...systemOneAnswers, severity: { type: "score", score: 7 } }, "severity"],
  ])("rejects %s", async (_case, answers, question) => {
    const baseURL = endpoint(() => Response.json({ answers }));

    await expect(decideTicket(systemOneDecision("clef", { baseURL })))
      .rejects.toThrow(`systemone returned an invalid answer for question "${question}".`);
  });

  it("reports HTTP errors with the provider message", async () => {
    const baseURL = endpoint(() => Response.json({ detail: "criteria must not be empty" }, { status: 422 }));

    await expect(decideTicket(systemOneDecision("clef", { baseURL, name: "local-clef" })))
      .rejects.toThrow("local-clef 422: criteria must not be empty");
  });

  it("rejects malformed responses", async () => {
    const invalidJson = endpoint(() => new Response("not json", { status: 200 }));
    await expect(decideTicket(systemOneDecision("clef", { baseURL: invalidJson })))
      .rejects.toThrow("systemone returned invalid decision JSON.");

    const noAnswers = endpoint(() => Response.json({ model: "clef" }));
    await expect(decideTicket(systemOneDecision("clef", { baseURL: noAnswers })))
      .rejects.toThrow("systemone returned a decision without answers.");
  });

  it("cancels with the signal reason", async () => {
    const baseURL = endpoint(() => Response.json({ answers: systemOneAnswers }));
    const controller = new AbortController();
    controller.abort(new Error("user cancelled"));

    await expect(systemOneDecision("clef", { baseURL }).decide({
      state: "x",
      questions: ticketQuestions,
      signal: controller.signal,
    })).rejects.toThrow("user cancelled");
    expect(requests).toHaveLength(0);
  });

  it("reports connection failures", async () => {
    await expect(decideTicket(systemOneDecision("clef", { baseURL: "http://127.0.0.1:1/v1" })))
      .rejects.toThrow("systemone connection failed:");
  });
});

describe("Cloudflare decisions", () => {
  it("calls Workers AI and unwraps its response envelope", async () => {
    const baseURL = endpoint(() => Response.json({
      result: { model: "clef", answers: systemOneAnswers, usage: { input_tokens: 9, output_tokens: 0 } },
      success: true,
      errors: [],
      messages: [],
    }));
    const provider = cloudflareDecision("clef", { accountId: "acc 1", apiToken: "cf-token", baseURL });

    const { answers, usage } = await decideTicket(provider);

    expect(answers.team.choice).toBe("technical");
    expect(usage).toEqual({ input: 9, output: 0, total: 9 });
    expect(requests[0]?.path).toBe("/accounts/acc%201/ai/run/@cf/cloudflare/clef");
    expect(requests[0]?.headers.get("authorization")).toBe("Bearer cf-token");
    expect(requests[0]?.body.model).toBe("clef");
  });

  it("reports Cloudflare error envelopes", async () => {
    const failing = endpoint(() => Response.json({ success: false, errors: [{ code: 10000, message: "Authentication error" }] }, { status: 401 }));
    await expect(decideTicket(cloudflareDecision("clef", { accountId: "a", apiToken: "bad", baseURL: failing })))
      .rejects.toThrow("cloudflare 401: Authentication error");

    const unsuccessful = endpoint(() => Response.json({ result: null, success: false, errors: [{ message: "Model is busy" }] }));
    await expect(decideTicket(cloudflareDecision("clef", { accountId: "a", apiToken: "t", baseURL: unsuccessful })))
      .rejects.toThrow("cloudflare decision failed: Model is busy");
  });
});

// ----------------------------------------------------------------------------
// Compile-time checks (run by `bun run typecheck`)
// ----------------------------------------------------------------------------

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
const assertType = <T extends true>() => {};

export const typeChecks = async (provider: DecisionProvider) => {
  // Inline questions keep their literal option IDs without `as const`.
  const { answers } = await provider.decide({
    state: "Checkout fails.",
    questions: {
      urgent: { type: "noul", instructions: "Urgent?" },
      team: { type: "choice", instructions: "Which team?", criteria: { billing: "Payments", technical: "Outages" } },
      severity: { type: "score", instructions: "How severe?", criteria: ["None", "Major"] },
    },
  });

  assertType<Equal<typeof answers.urgent, NoulAnswer>>();
  assertType<Equal<typeof answers.team, ChoiceAnswer<"billing" | "technical">>>();
  assertType<Equal<typeof answers.team.choice, "billing" | "technical">>();
  assertType<Equal<keyof typeof answers.team.probabilities, "billing" | "technical">>();
  assertType<Equal<typeof answers.severity, ScoreAnswer>>();

  // Questions stored in a variable keep them with `satisfies` (or `as const`).
  const routing = {
    queue: { type: "choice", instructions: "Which queue?", criteria: { human: null, bot: null } },
  } satisfies DecisionQuestions;
  const routed = await provider.decide({ state: "Hi", questions: routing });
  assertType<Equal<typeof routed.answers.queue.choice, "human" | "bot">>();

  // @ts-expect-error unknown question IDs are not part of the answers
  void answers.missing;
  // @ts-expect-error a choice question needs criteria
  void provider.decide({ state: "", questions: { team: { type: "choice", instructions: "Which team?" } } });
};
