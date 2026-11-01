"""HITL interrupt policy: which tools require approval on attended agents."""


def test_hitl_interrupt_on_arms_expected_tools():
    from metasci_scribe.agent import HITL_INTERRUPT_ON

    assert HITL_INTERRUPT_ON == {
        "execute": True,
        "run_in_background": True,
        "delete": True,
        "skill_manager": True,
    }


def test_skill_manager_pauses_for_installs_only():
    from metasci_scribe import policy

    assert policy.hitl_armed("skill_manager", {"action": "install", "source": "o/r"})
    assert not policy.hitl_armed("skill_manager", {"action": "list"})
    assert policy.hitl_armed("execute", {"command": "ls"})
    assert policy.hitl_armed("delete", {"path": "/x"})
