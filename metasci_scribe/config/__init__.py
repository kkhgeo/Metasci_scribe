"""Configuration package for Metasci Scribe.

Re-exports all public symbols from settings and onboard submodules
so that existing ``from metasci_scribe.config import X`` imports continue
to work without modification.

The onboard module is loaded lazily because it pulls in heavy dependencies
(langchain, llm) that are not needed for normal config operations.
"""

from .settings import (
    GatewaySurface,
    MemoryControls,
    MemoryObservationTarget,
    MemoryObservationWriter,
    MemorySkillSynthesisCadence,
    MemorySkillSynthesisMode,
    ScribeConfig,
    apply_config_to_env,
    get_config_dir,
    get_config_path,
    get_config_value,
    get_effective_config,
    list_config,
    load_config,
    reset_config,
    resolve_gateway_backend,
    save_config,
    set_config_value,
)

__all__ = [
    "GatewaySurface",
    "MemoryControls",
    "MemoryObservationTarget",
    "MemoryObservationWriter",
    "MemorySkillSynthesisCadence",
    "MemorySkillSynthesisMode",
    "ScribeConfig",
    "apply_config_to_env",
    # settings
    "get_config_dir",
    "get_config_path",
    "get_config_value",
    "get_effective_config",
    "list_config",
    "load_config",
    "reset_config",
    "resolve_gateway_backend",
    # onboard (lazy)
    "run_onboard",
    "save_config",
    "set_config_value",
]


def __getattr__(name: str):
    if name == "run_onboard":
        from .onboard import run_onboard

        return run_onboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
