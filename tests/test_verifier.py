from __future__ import annotations

import inspect

from adaptive_swe_branching.environments import verifier


def test_verifier_uses_active_python_interpreter() -> None:
    source = inspect.getsource(verifier.SWESmithVerifier.verify)
    assert "sys.executable" in source
    assert '["python"' not in source
