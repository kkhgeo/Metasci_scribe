"""Input validators for the onboarding wizard.

- IntegerValidator / ChoiceValidator: prompt_toolkit Validators
- validate_*_key: per-provider API key validators (live HTTP probes) that
  return ``(is_valid, message)``. The checks live in
  :mod:`metasci_scribe.config.provider_keys`; these wrap them for the wizard.
"""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.validation import ValidationError, Validator

from .. import provider_keys
from ..provider_keys import KeyCheck


class IntegerValidator(Validator):
    """Validates that input is a positive integer."""

    def __init__(self, min_value: int = 1, max_value: int = 100):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, document) -> None:
        text = document.text.strip()
        if not text:
            return  # Allow empty for default
        try:
            value = int(text)
            if value < self.min_value or value > self.max_value:
                raise ValidationError(
                    message=f"Must be between {self.min_value} and {self.max_value}"
                )
        except ValueError as e:
            raise ValidationError(message="Must be a valid integer") from e


class ChoiceValidator(Validator):
    """Validates that input is one of the allowed choices."""

    def __init__(self, choices: list[str], allow_empty: bool = True):
        self.choices = choices
        self.allow_empty = allow_empty

    def validate(self, document) -> None:
        text = document.text.strip().lower()
        if not text and self.allow_empty:
            return
        if text not in [c.lower() for c in self.choices]:
            raise ValidationError(message=f"Must be one of: {', '.join(self.choices)}")


# =============================================================================
# API Key Validation
# =============================================================================


def as_pair(result: KeyCheck) -> tuple[bool, str]:
    """``(is_valid, message)`` for the wizard: a skipped (empty) key passes,
    anything not conclusively valid fails."""
    return result.status in ("valid", "empty"), result.message


def _pair(check_name: str) -> Callable[..., tuple[bool, str]]:
    # Looked up on every call so a patched check in provider_keys applies.
    def validate(*args, **kwargs) -> tuple[bool, str]:
        return as_pair(getattr(provider_keys, check_name)(*args, **kwargs))

    validate.__name__ = check_name.replace("check_", "validate_")
    validate.__doc__ = f"``{check_name}`` as ``(is_valid, message)``."
    return validate


validate_anthropic_key = _pair("check_anthropic_key")
validate_openai_key = _pair("check_openai_key")
