"""The words around Sign in with ChatGPT, shared by ``/login`` and onboarding.

Terminal wording follows OpenAI's UI/UX guidelines
(https://developers.openai.com/siwc/ui-ux-guidelines): the sign-in is offered
as "Continue with ChatGPT", the first sign-in with plan usage on is welcomed
once, and a link to the usage page stays next to the plan. The decision for a
terminal, where there is no button or logo, is recorded in ``docs/baseline.md``.
"""

from __future__ import annotations

from .chatgpt_plan import (
    ChatGPTAuthError,
    CredentialStoreError,
    LoginCancelled,
    LoginFailed,
    LoginTimeout,
    NotSignedIn,
    PlanUsageDisabled,
    RefreshRejected,
    RefreshUnavailable,
)

CONTINUE_LABEL = "Continue with ChatGPT"
USAGE_URL = "https://chatgpt.com/settings/usage"

# Two lines, shown before the browser opens: what the agent can reach.
LOGIN_WARNING = (
    "The agent can read the files in this paper folder.",
    "Commands always need your approval in chat, and autonomous runs get no shell.",
)

WELCOME = (
    "You're using your ChatGPT plan.",
    "Eligible usage in this app uses your ChatGPT plan.",
)


def describe_auth_error(
    exc: ChatGPTAuthError, *, retry: str = "Run /login to try again."
) -> tuple[str, str]:
    """``(message, style)`` for a sign-in failure, in words a person can act on.

    ``retry`` says how to try again where a sign-in can be retried; onboarding
    has no ``/login``.
    """
    if isinstance(exc, LoginCancelled):
        return "Sign-in cancelled.", "yellow"
    if isinstance(exc, LoginTimeout):
        return (
            f"No answer from the browser in time. {retry}",
            "yellow",
        )
    if isinstance(exc, LoginFailed):
        if exc.code == "account_mismatch":
            return (
                "This is a different ChatGPT account from the one saved here. "
                "Run /logout first, then /login to switch accounts.",
                "red",
            )
        return f"{exc} {retry}", "red"
    if isinstance(exc, CredentialStoreError):
        return str(exc), "red"
    if isinstance(exc, RefreshRejected):
        return (
            "ChatGPT no longer accepts this sign-in. Run /login to sign in again.",
            "red",
        )
    if isinstance(exc, RefreshUnavailable):
        return (
            "Could not reach ChatGPT; your sign-in is kept. Try again shortly.",
            "yellow",
        )
    if isinstance(exc, PlanUsageDisabled):
        return (
            "ChatGPT plan usage is off for this sign-in. Run /login and approve "
            "plan usage, or use an OpenAI API key: metasci-scribe onboard.",
            "red",
        )
    if isinstance(exc, NotSignedIn):
        return "Not signed in to ChatGPT. Run /login.", "red"
    return str(exc), "red"


def plan_off_lines(who: str) -> tuple[str, str]:
    """What to say when the sign-in worked but the grant lacks plan usage."""
    return (
        f"{who}, but ChatGPT plan usage is off for this sign-in.",
        "Nothing was switched. Run /login again and approve plan usage, "
        "or use an OpenAI API key: metasci-scribe onboard.",
    )
