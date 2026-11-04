# Acceptance record

The initial version is complete when a researcher can do all seven steps below,
in a Windows terminal, from start to end. The steps are the completion criteria
of section 2 of `docs/superpowers/specs/2026-11-01-metasci-scribe-design.md`.

Fill this in by hand. The person who tests signs in to ChatGPT in the browser.

## Run

| Item | Value |
|---|---|
| Date | |
| Tester | |
| Windows version | |
| Windows user | clean user profile: yes / no |
| Python, uv, Git versions | |
| Install command used | |
| App version (`metasci-scribe --version`) | |
| Paper folder (local disk or drive letter) | |
| ChatGPT plan | |

## Completion criteria

Mark each line `[x]` when it passes and write what you saw under it. If a step
fails, write the message exactly and what you did next.

- [ ] **1. Install and start.** The app is installed and starts.

  Result:

- [ ] **2. `/login`.** `/login` connects to ChatGPT.

  Result:

- [ ] **3. `/init`.** `/init` creates the paper folder layout.

  Result:

- [ ] **4. Chat editing.** In chat, `outline.md` is edited and a paragraph is
  written. The agent reads and writes the files.

  Result:

- [ ] **5. `/run`.** `/run` writes autonomously into `Manuscript/` and `manuscript.md` is joined from it. `/run <section>` writes one section.

  Result:

- [ ] **6. Reopen and continue.** After closing and reopening the app, the same
  work continues.

  Result:

- [ ] **7. Usage limit.** At a usage limit the run stops without switching to
  another way of paying, and a later run continues where it stopped.

  Result:

## Terms re-check (before the repository is made public)

Read the current conditions of OpenAI's "Sign in with ChatGPT" documentation
again and record them here.

Checked after the repository had already been made public (2026-11-03), so
this record is late against the rule above.

| Item | Value |
|---|---|
| Date checked | 2026-11-03 |
| Page(s) read | developers.openai.com/siwc; /siwc/token-sharing-open-source (overview, sign-in, models-and-inference, errors-and-recovery, preview-limitations); /siwc/ui-ux-guidelines; /siwc/request-client-id; /siwc/website; openai.com/policies/usage-policies (version in force from 2026-11-20). The pages carry no date. openai.com/policies/row-terms-of-use (published 2026-01-01), read in a browser after the index refused automated reading (HTTP 403). |
| What the terms say now | The token-sharing path is for "open-source and locally hosted apps"; a paid or remotely hosted app must use the interest form instead. "Open source" is not defined. Still called a preview. No plan is named as supported (earlier: Plus and Pro). Rules: public Responses API only, never `chatgpt.com/backend-api`; `store=false`, `stream=true`; listed fields omitted; no silent switch to another billing path; on `subscription_sharing_usage_limit_exceeded` pause and point to chatgpt.com/settings/usage; on `user_not_eligible` do not retry or loop sign-in; token file owner-only and never logged. UI guidance: a "Using ChatGPT plan" indicator near the input. Usage policies forbid academic dishonesty. Terms of Use: account credentials are not to be shared or given to others, and data or output is not to be extracted automatically or programmatically; the Sign in with ChatGPT documentation is OpenAI's own sanctioned route for an app to call the API with the user's plan, and each user signs in with their own account. |
| Does it still allow this project's use? | Yes, as far as these pages say. The app is open source, runs locally, is not paid, and was built to these rules (`auth/chatgpt_plan.py`, `llm/chatgpt_plan.py`: field drop list, `store=False`, error-code mapping, owner-only DACL on Windows; status bar shows "Using ChatGPT plan"). `/siwc/request-client-id` says Sign in with ChatGPT is offered to "a select group of commercial partners"; the open-source pages need no secret or partner key, and the M0 spike (2026-11-01) registered dynamically and was granted `chatgpt.tokens.use.direct`, so the direct path was open to this project on that date. |
| Changes needed in the README or the app | README: "Plus and Pro are the plans documented so far" replaced (the docs name no plans); a note added that OpenAI's usage policies forbid academic dishonesty and that the user follows journal and institution rules on AI disclosure. App: none. Open: whether a text-only "Continue with ChatGPT" label in a terminal meets the branding guidance is not stated; the usage policies change on 2026-11-20 (the version read is that one). |

## Problems found

| Step | What happened | Fix or decision |
|---|---|---|
| | | |
