from __future__ import annotations

import inspect
from types import SimpleNamespace

from adaptive_swe_branching.agents.openhands import _completion_from_status
from adaptive_swe_branching.data.records import Outcome
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


def test_empty_patch_is_unsolved_without_starting_the_harness(
    tmp_path, monkeypatch
) -> None:
    adapter = verifier.SWESmithVerifier(
        harness_path=tmp_path,
        timeout_seconds=1,
    )
    task = SimpleNamespace(record=SimpleNamespace(task_id="task"))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("the harness must not run for an empty patch")

    monkeypatch.setattr(verifier.subprocess, "run", fail_if_called)
    result = adapter.verify(task, "")

    assert result.record.outcome is Outcome.UNSOLVED
    assert result.record.cost.verifier_calls == 0
    assert result.report == {"reason": "empty_patch", "harness_invoked": False}
