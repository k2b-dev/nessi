import { afterEach, expect, test } from "bun:test";
import { complete, openAICompatible, type AssistantContentBlock } from "../../../src/ai/index.js";
import { fixtureJson, fixtureText, stubFetch, textResponse } from "../helpers/fixtures.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });
const provider = (model = "kimi-k3", none = false) => openAICompatible({
  name: "recorded", model, baseURL: "https://example.test/v1", compat: { thinkingFormat: none ? "none" : "text" },
});
const sse = (deltas: Record<string, unknown>[], finish = true) => deltas.map((delta) =>
  `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: null }] })}\n\n`,
).join("") + (finish ? 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n' : "");
const blocks = async (parts: string[], model = "kimi-k3", none = false) => {
  stubFetch(() => textResponse(sse(parts.map((content) => ({ content }))), "text/event-stream"));
  return (await complete(provider(model, none), { messages: [] })).message.content;
};
const expected: AssistantContentBlock[] = [{ type: "thinking", thinking: "Plan." }, { type: "text", text: "Answer." }];

test("recorded Kimi K3 content becomes reasoning then text before the intact tool call", async () => {
  const recorded = await fixtureJson<{ reasoning: string; text: string }>("../fixtures/openai/kimi-k3-recorded.json");
  stubFetch(async () => textResponse(await fixtureText("../fixtures/openai/kimi-k3-recorded.sse"), "text/event-stream"));
  const events = [];
  for await (const event of provider().stream({ messages: [] })) events.push(event);
  expect(events.filter((event) => event.type === "issue")).toEqual([]);
  const result = events.flatMap((event) => event.type === "block_end" ? [event.block] : []);
  expect(result.slice(0, 2)).toEqual([{ type: "thinking", thinking: recorded.reasoning }, { type: "text", text: recorded.text }]);
  expect(result[2]).toMatchObject({ type: "tool_call", id: "code_write:20", name: "code_write", args: { path: "test/default.json" } });
  expect(events.filter((event) => event.type === "block_delta").map((event) => event.delta).join("")).not.toContain("<|");
});

for (const wire of ["Plan.<|close|>think<|sep|>Answer.", "<|open|>think<|sep|>Plan.<|close|>think<|sep|>Answer.", "<think>Plan.</think>Answer."]) {
  test(`all two-chunk splits and single-character deltas: ${wire}`, async () => {
    for (let split = 0; split <= wire.length; split++) expect(await blocks([wire.slice(0, split), wire.slice(split)])).toEqual(expected);
    expect(await blocks([...wire])).toEqual(expected);
  });
}

test("explicit markers work for other OpenAI-compatible models too", async () => {
  expect(await blocks(["Hello.<|open|>think<|sep|>Plan.<|close|>think<|sep|>Answer."], "other")).toEqual([
    { type: "text", text: "Hello." }, ...expected,
  ]);
});

test("complete and truncated closing markers at EOF never leak control syntax", async () => {
  expect(await blocks(["<think>Plan.</think>"])).toEqual([{ type: "thinking", thinking: "Plan." }]);
  expect(await blocks(["Plan.<|close|>think<|sep|>"])).toEqual([{ type: "thinking", thinking: "Plan." }]);
  const marker = "<|close|>think<|sep|>";
  for (let end = 2; end < marker.length; end++) {
    expect(await blocks(["Plan.", marker.slice(0, end)])).toEqual([{ type: "thinking", thinking: "Plan." }]);
    const result = await blocks(["<think>Plan.", marker.slice(0, end)]);
    expect(result).toEqual([{ type: "thinking", thinking: "Plan." }]);
  }
  expect(await blocks(["plain text <"])).toEqual([{ type: "text", text: "plain text <" }]);
  expect(await blocks(["x <thing> y"])).toEqual([{ type: "text", text: "x <thing> y" }]);
});

test("structured reasoning and inline reasoning coexist without leaking markers", async () => {
  stubFetch(() => textResponse(sse([{ reasoning_content: "First." }, { content: "<think>Plan.</think>Answer." }]), "text/event-stream"));
  expect((await complete(provider(), { messages: [] })).message.content).toEqual([
    { type: "thinking", thinking: "First.Plan." }, { type: "text", text: "Answer." },
  ]);
  expect(await blocks(["Plan.<|close|>think<|sep|>Answer."], "kimi-k3", true)).toEqual([{ type: "text", text: "Answer." }]);
});

test("marker-free Kimi text stays text on success and becomes thinking on truncated streams", async () => {
  expect(await blocks(["Answer."])).toEqual([{ type: "text", text: "Answer." }]);
  stubFetch(() => textResponse(sse([{ content: "Partial answer." }], false), "text/event-stream"));
  const events = [];
  for await (const event of provider().stream({ messages: [] })) events.push(event);
  expect(events.some((event) => event.type === "block_end" && event.block.type === "thinking" && event.block.thinking === "Partial answer.")).toBe(true);
  expect(events.some((event) => event.type === "issue")).toBe(true);
});

test("ordinary text and structured Kimi reasoning still stream before later frames arrive", async () => {
  for (const [model, delta, kind] of [["other", { content: "Answer." }, "text"], ["kimi-k3", { reasoning_content: "Plan." }, "thinking"]] as const) {
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    stubFetch(() => new Response(new ReadableStream({ start(value) { controller = value; } }), { headers: { "Content-Type": "text/event-stream" } }));
    const stream = provider(model).stream({ messages: [] })[Symbol.asyncIterator]();
    const pending = stream.next();
    await Promise.resolve();
    controller.enqueue(new TextEncoder().encode(sse([delta], false)));
    expect((await pending).value).toMatchObject({ type: "block_start", kind });
    controller.close();
    while (!(await stream.next()).done) { /* drain */ }
  }
});


test("literal HTML-like reasoning tags inside ordinary text and JSON remain unchanged", async () => {
  for (const text of ['Use </think> here and <think> in examples.', '```ts\nconst tags = "<think>plan</think>";\n```', '{"tags":"<think>x</think>"}']) {
    for (const model of ["other", "kimi-k3"]) expect(await blocks([...text], model)).toEqual([{ type: "text", text }]);
  }
  expect(await blocks([..."  <think>Plan.</think>Answer."], "other")).toEqual(expected);
});

test("length stops and provider errors keep the ambiguous prefix as reasoning", async () => {
  for (const end of ['data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n', 'data: {"error":{"message":"failed","type":"server_error"}}\n\n', 'data: invalid-json\n\n']) {
    stubFetch(() => textResponse(sse([{ content: "Plan." }], false) + end, "text/event-stream"));
    const events = [];
    for await (const event of provider().stream({ messages: [] })) events.push(event);
    expect(events.some((event) => event.type === "block_end" && event.block.type === "thinking" && event.block.thinking === "Plan.")).toBe(true);
    expect(events.some((event) => event.type === "block_start" && event.kind === "text")).toBe(false);
  }
});

test("explicitly disabled Kimi reasoning streams answer text immediately", async () => {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  stubFetch(() => new Response(new ReadableStream({ start(value) { controller = value; } }), { headers: { "Content-Type": "text/event-stream" } }));
  const stream = provider().stream({ messages: [], reasoningEffort: "none" })[Symbol.asyncIterator]();
  const pending = stream.next();
  await Promise.resolve();
  controller.enqueue(new TextEncoder().encode(sse([{ content: "Answer." }], false)));
  expect((await pending).value).toMatchObject({ type: "block_start", kind: "text" });
  controller.close();
  while (!(await stream.next()).done) { /* drain */ }
});

test("transport failures flush an ambiguous Kimi prefix into thinking before the issue", async () => {
  stubFetch(() => new Response(new ReadableStream<Uint8Array>({
    start(controller) { controller.enqueue(new TextEncoder().encode(sse([{ content: "Plan." }], false))); },
    pull(controller) { controller.error(new Error("connection lost")); },
  }), { headers: { "Content-Type": "text/event-stream" } }));
  const events = [];
  for await (const event of provider().stream({ messages: [] })) events.push(event);
  expect(events.findIndex((event) => event.type === "block_end" && event.block.type === "thinking" && event.block.thinking === "Plan.")).toBeLessThan(events.findIndex((event) => event.type === "issue"));
  expect(events.some((event) => event.type === "block_start" && event.kind === "text")).toBe(false);
});


test("an HTML closer inside Kimi reasoning remains literal until the Kimi closer", async () => {
  expect(await blocks([..."Use </think> as an example.<|close|>think<|sep|>Answer."])).toEqual([
    { type: "thinking", thinking: "Use </think> as an example." }, { type: "text", text: "Answer." },
  ]);
});

test("a refusal follows the buffered reasoning in its original order", async () => {
  stubFetch(() => textResponse(sse([{ content: "Plan." }, { refusal: "Cannot help." }]), "text/event-stream"));
  expect((await complete(provider(), { messages: [] })).message.content).toEqual([
    { type: "thinking", thinking: "Plan." }, { type: "text", text: "Cannot help." },
  ]);
});
