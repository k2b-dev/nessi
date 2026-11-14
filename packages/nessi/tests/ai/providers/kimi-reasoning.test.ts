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

test("marker-free Kimi text and truncated streams retain partial answers", async () => {
  expect(await blocks(["Answer."])).toEqual([{ type: "text", text: "Answer." }]);
  stubFetch(() => textResponse(sse([{ content: "Partial answer." }], false), "text/event-stream"));
  const events = [];
  for await (const event of provider().stream({ messages: [] })) events.push(event);
  expect(events.some((event) => event.type === "block_end" && event.block.type === "text" && event.block.text === "Partial answer.")).toBe(true);
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
