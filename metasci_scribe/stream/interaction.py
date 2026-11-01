"""Approval policy shared by the CLI, TUI and LangGraph gateway.

Config-rule HITL decisions (``shell_allow_list``, ``auto_approve``,
``dangerous_mode``) and the :class:`ApprovalOutcome` result type. Pure policy:
no agent, no graph, no transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def approve_decisions(action_requests: list) -> list[dict]:
    """Build the ``decisions`` payload that approves every action request.

    Length matches ``action_requests`` (with a floor of 1, so an empty
    request list still yields a single approve — the shape
    ``Command(resume=...)`` expects).
    """
    n = len(action_requests) or 1
    return [{"type": "approve"} for _ in range(n)]


def config_policy_snapshot(
    action_requests: list[dict],
) -> tuple[list[dict] | None, dict[int, dict]]:
    """One config load → ``(decisions, rejections)`` for an approval operation.

    The single evaluation point of the config policy: one ``_ensure_config()``
    produces both views an approval operation needs, so the policy verdicts
    the operation acts on are immutable for its whole duration — prompt,
    human wait, and reply. Re-loading config after the human replies would
    let a mid-wait ``auto_approve`` toggle flip a prompt-time REJECT into a
    PROMPT (no rejection), so a blanket approval would run the refused
    command (CWE-367 time-of-check/time-of-use).

    Returns ``(decisions, rejections)``:

    * ``decisions`` — the full per-request list when config clears every
      request, ``None`` when any request needs a human (the caller prompts).
      Same contract as :func:`resolve_config_decisions`, which is this
      snapshot's collapsed view. Empty requests yield ``([], {})``: a
      one-per-request decisions list would break ``HumanInTheLoopMiddleware``,
      which requires the counts to match.
    * ``rejections`` — per-index REJECT decisions (with their reasons),
      preserved even when the list collapses to ``None``: a batch that
      mixes a REJECT with a prompt-needed request still reports the REJECT
      here. The reply branches after a human approval consume these — see
      :func:`decisions_after_human_approval`.

    Config-load errors return ``(None, {})``: the auto path fails closed
    (the human is prompted), and the reply path fails open (when the policy
    cannot speak, the human's interactive approval stands).
    """
    try:
        from ..agent import _ensure_config

        # Session-effective config, not the file: --auto-approve / --dangerous
        # and METASCI_SCRIBE_* overrides live here (the same object
        # ``hitl_suppressed_for_run`` reads); the graph is armed on the runs that
        # reach this resolver, so this snapshot is the only gate for them.
        cfg = _ensure_config()
    except Exception:
        return None, {}

    shell_allow_list = (
        [s.strip() for s in cfg.shell_allow_list.split(",") if s.strip()]
        if cfg.shell_allow_list
        else []
    )

    decisions: list[dict] = []
    rejections: dict[int, dict] = {}
    needs_human = False
    for i, req in enumerate(action_requests):
        decision = _resolve_one_action_request(req, cfg, shell_allow_list)
        if decision is None:
            # Keep scanning: REJECTs later in a mixed batch must survive the
            # collapse to None (the reply branches consume them).
            needs_human = True
            continue
        decisions.append(decision)
        if decision.get("type") == "reject":
            rejections[i] = decision
    if needs_human:
        return None, rejections
    return decisions, rejections


def decisions_after_human_approval(
    action_requests: list[dict], policy_rejections: dict[int, dict]
) -> list[dict]:
    """Decisions to resume with after a human chose "approve"/"approve all".

    An interactive approval must not override the config policy's REJECTs:
    when a mixed batch reaches the prompt (a policy-rejected dangerous
    command plus a request that needed a human), approving the batch keeps
    the rejection — with its reason, so the model still gets the actionable
    refusal feedback — and approves only the rest. Without this, a single
    blanket "1" would run the dangerous command the policy just refused.

    *policy_rejections* are the per-index REJECT decisions from the
    prompt-time :func:`config_policy_snapshot` — the same single config
    load that decided to prompt. The function is pure (no config IO), so a
    config toggle while the human was deciding cannot change what the
    approval operation does with the reply.

    Requests the policy does not REJECT (approve, prompt — the human just
    decided those — or no verdict) are approved. A session "Approve all"
    grant never prompts, so it keeps its blanket semantics and does not come
    through here.
    """
    if not policy_rejections:
        return approve_decisions(action_requests)
    return [
        policy_rejections.get(i, {"type": "approve"})
        for i in range(len(action_requests))
    ]


def _resolve_one_action_request(
    req: dict, cfg: Any, shell_allow_list: list[str]
) -> dict | None:
    """One action request → its config decision, or ``None`` if a human must decide.

    Shared per-request core of the config policy. Malformed requests fail
    closed to ``None`` (prompt the human). Non-shell tools auto-approve.
    Shell tools go through
    :func:`~metasci_scribe.backends.resolve_action_decision`: allow-list
    token-boundary match, dangerous commands rejected with a reason under
    ``auto_approve``. ``None`` covers PROMPT verdicts, always-prompt tools,
    and malformed requests.
    """
    from ..config.settings import HITL_ALWAYS_PROMPT_TOOLS, HITL_SHELL_TOOLS

    if not isinstance(req, dict):
        return None
    name = req.get("name", "")
    if not isinstance(name, str) or not name:
        return None
    # dangerous_mode keeps its "trust everything" meaning: it must approve
    # ahead of the always-prompt set, or a --dangerous session is interrupted
    # on every delete / schedule_task while execute stays silent.
    if cfg.dangerous_mode:
        return {"type": "approve"}
    if name in HITL_ALWAYS_PROMPT_TOOLS:
        return None
    if name not in HITL_SHELL_TOOLS:
        return {"type": "approve"}
    args = req.get("args", {})
    command = args.get("command", "") if isinstance(args, dict) else ""
    if not isinstance(command, str) or not command:
        # Malformed shell request (missing/dict-typed/empty command):
        # an empty string would sail through auto-approve, so fail closed
        # to a human decision instead.
        return None
    from ..backends import ActionDecision, resolve_action_decision

    verdict = resolve_action_decision(
        command,
        auto_approve=cfg.auto_approve,
        dangerous_mode=cfg.dangerous_mode,
        allow_list=shell_allow_list,
    )
    if verdict.decision is ActionDecision.APPROVE:
        return {"type": "approve"}
    if verdict.decision is ActionDecision.REJECT:
        return {"type": "reject", "message": verdict.reason}
    return None


def resolve_config_decisions(action_requests: list[dict]) -> list[dict] | None:
    """Resolve HITL decisions from config rules alone, or ``None`` to prompt.

    Per-request policy chokepoint shared by every attended UI (CLI display, TUI):
    non-shell tools auto-approve; shell tools go through
    :func:`~metasci_scribe.backends.resolve_action_decision` (``shell_allow_list``
    token-boundary match, dangerous commands rejected with a reason under
    ``auto_approve``). Returns a full ``decisions`` list when config clears every
    request, or ``None`` when any request needs a human — the caller then prompts
    (mounts a widget, calls ``input()``). Fail-closed to
    ``None`` on config-load errors and malformed requests. The evaluation is
    :func:`config_policy_snapshot` (one config load); this collapsed view is
    what the attended UIs consume, so an ``auto_approve`` dangerous command
    is a REJECT with a reason everywhere, never a silent collapse to
    "needs a human".
    """
    return config_policy_snapshot(action_requests)[0]


@dataclass
class ApprovalOutcome:
    """Result of one human approval round, as reported by a ``hitl_outcome_fn``.

    ``decisions`` is the approve payload on approve / approve-all, or ``None``
    when the action was declined (reject / timeout / stop / unrecognized).

    ``unrecognized_reply`` carries the raw reply text when parsing failed.

    ``prompted`` is False only when the policy auto-resolved the interrupt
    without asking a human. The HITL loop counts only prompted rounds
    against its round budget (issue #469).

    ``session_granted`` is True only when ``decisions`` came from a session
    "approve all" grant that was already in force (all approve by
    construction). Config rules and human replies, the reply that grants
    included, leave it False: that reply can carry config rejections from a
    mixed batch, so a caller sends the grant only with an answer the grant
    produced. ``grant_in_force`` says whether the session holds a grant once
    this outcome is returned, which the reply that grants does too.
    ``rejected`` is True when a human replied and did not approve, a stop
    included; a decline without it (timeout, a prompt that could not be
    sent) was left unanswered.
    """

    decisions: list[dict] | None = None
    unrecognized_reply: str | None = None
    prompted: bool = False
    # True only when a human prompt was skipped because the caller's human
    # budget was already spent. Auto fast paths leave this False.
    budget_exhausted: bool = False
    session_granted: bool = False
    grant_in_force: bool = False
    rejected: bool = False
