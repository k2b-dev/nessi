# Kimi K3 reasoning regression

Source: the assistant message at sequence 93 of fictional preview chat `hnzNvY`,
recorded on 2026-11-14 with model `kimi-k3` through Cortecs. It was read with
SELECT-only queries from `ai.messages`, joined to `ai.conversations` by ID.
Related `ai.turns` and `ai.inference_calls` rows were inspected; they do not
provide the original transport frames. No model request was made for this fixture.

The stored representation is already parsed: one text block contains an English
reasoning span, `<|close|>think<|sep|>`, and a German answer, followed by a tool call.
There is no opening marker or separate thinking block in this message.

`kimi-k3-recorded.sse` reconstructs Chat Completions deltas by splitting that
text into consecutive 37-character chunks, without changing any character.
The original tool ID, name and JSON arguments follow as one tool-call delta.
A synthetic `tool_calls` finish and `[DONE]` terminate the stream. Original
chunk boundaries, usage frames and transport timing were not retained and are
not claimed to be recorded. The data describes a fictional calculator and
notebook fixture; it contains no credentials.

`kimi-k3-recorded.json` contains the expected reasoning and answer, split at the
single recorded closing marker. Tests assert both spans and the intact tool call,
then exercise every two-chunk split and single-character deltas separately.
