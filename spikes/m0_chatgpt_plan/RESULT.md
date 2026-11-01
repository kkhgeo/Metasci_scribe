# M0 spike result: tool calling over a ChatGPT plan

T8 follows this file.

## Pass / fail

PASS, 2026-11-01. Commands: `spike_login.py`, `spike_login.py --status`, `spike_turn.py --tool-format namespace --model gpt-6.1-sol`, `spike_turn.py --tool-format additional --model gpt-6.1-sol`.

- `chatgpt.tokens.use.direct` granted, `GET /v1/models` returned 8 slugs.
- deepagents read a 39,531-character file through a tool, and the sentinel matched.
- Turn 2 completed with the replayed history. Each run took 18 to 21 seconds for 3 requests.

## Tool format

Both completed two turns. **T8 uses `additional`.** It needs no `namespace` on replayed `function_call` items. Send `additional_tools` as an `input` item on every request and send no top-level `tools`.

## Response items to replay

`message`, `function_call`, and `function_call_output` were enough. `--include-reasoning` and `--keep-item-ids` were not needed.

## Model slug

Tested `gpt-6.1-sol`. Also visible: `gpt-6-astra`, `gpt-6.1-terra`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`.

## Not covered

- `request_id` was not captured from response headers. T8 must find the right header.
- Usage-limit errors and token refresh were not exercised live. T7 and T8 cover them with unit tests.
