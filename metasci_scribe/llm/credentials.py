"""Whether a provider has a sign-in or key to build a chat model with.

Checked where the application builds its chat model, not inside
``get_chat_model``: that function builds a model for any caller, including the
ChatGPT plan route, which signs in lazily on the first request. A first run
that has neither a sign-in nor a key gets :class:`ModelNotConfigured` here, one
typed error every surface turns into the same message.
"""

from __future__ import annotations

import os

__all__ = ["ModelNotConfigured", "ensure_model_configured"]

_NOT_CONFIGURED = {
    "openai": (
        "No model is set up yet. Run /login to use your ChatGPT plan, or add an "
        "OpenAI API key: run `metasci-scribe onboard`, or set OPENAI_API_KEY."
    ),
    "anthropic": (
        "No Anthropic API key is set. Add one with `metasci-scribe onboard` or "
        "ANTHROPIC_API_KEY, or run /login to use your ChatGPT plan with an "
        "OpenAI model."
    ),
}


class ModelNotConfigured(RuntimeError):
    """No sign-in or API key exists for the chosen provider.

    Raised when the chat model is built, so every surface meets the same
    error and the same words, the single-shot run and the prompt alike. The
    message names ``/login`` and the API key alternative.
    """

    def __init__(self, provider: str = "openai") -> None:
        super().__init__(_NOT_CONFIGURED.get(provider, _NOT_CONFIGURED["openai"]))
        self.provider = provider


def _plan_sign_in_saved() -> bool:
    """Whether a ChatGPT sign-in is saved.

    A store that cannot be read counts as saved: the request then reports the
    real problem, which a "not signed in" message would hide.
    """
    from ..auth.chatgpt_plan import ChatGPTAuthError, load

    try:
        return load() is not None
    except ChatGPTAuthError:
        return True


def ensure_model_configured(provider: str | None) -> None:
    """Raise :class:`ModelNotConfigured` when ``provider`` cannot be used yet.

    OpenAI needs a saved ChatGPT sign-in in plan mode, otherwise an API key.
    Anthropic needs an API key, or a base URL for a gateway that holds it.
    """
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get(
            "ANTHROPIC_BASE_URL"
        ):
            raise ModelNotConfigured("anthropic")
        return
    if provider not in (None, "openai"):
        return
    if os.environ.get("METASCI_SCRIBE_OPENAI_AUTH_MODE") == "chatgpt_plan":
        if not _plan_sign_in_saved():
            raise ModelNotConfigured("openai")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise ModelNotConfigured("openai")
