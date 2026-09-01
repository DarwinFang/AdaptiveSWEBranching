from __future__ import annotations

import inspect

from adaptive_swe_branching.agents.openhands import _completion_from_status
from adaptive_swe_branching.environments import verifier


def test_verifier_uses_active_python_interpreter() -> None:
    source = inspect.getsource(verifier.SWESmithVerifier.verify)
    assert "sys.executable" in source
    assert '["python"' not in source


def test_agent_stuck_is_a_valid_policy_termination_not_infrastructure_error() -> None:
    assert _completion_from_status(
        status_name="stuck",
        answer=None,
        previously_finished=False,
        previous_reason=None,
    ) == (True, "agent_stuck")
