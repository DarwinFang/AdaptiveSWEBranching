from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_swe_branching.config import load_config
from adaptive_swe_branching.data.store import ExperimentManifest, RawStore
from adaptive_swe_branching.seeds import derive_seed


def test_config_includes_and_seed_are_stable(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("a:\n  x: 1\n  y: 2\n")
    (tmp_path / "child.yaml").write_text("include: base.yaml\na:\n  y: 3\n")
    assert load_config(tmp_path / "child.yaml") == {"a": {"x": 1, "y": 3}}
    assert derive_seed(12, "task", 3) == derive_seed(12, "task", 3)
    assert derive_seed(12, "task", 3) != derive_seed(12, "task", 4)


def test_raw_store_is_immutable(tmp_path: Path) -> None:
    store = RawStore(tmp_path / "run")
    manifest = ExperimentManifest.create(
        experiment_name="test",
        git_commit="abc",
        config={},
        root_seed=1,
        external_resources={},
    )
    store.initialise(manifest)
    store.put("task", "one", {"x": 1})
    store.put("task", "one", {"x": 1})
    with pytest.raises(FileExistsError):
        store.put("task", "one", {"x": 2})
