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


def test_phase4_manifest_freezes_single_shared_q_model() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs/experiments/phase4_oracle_pilot.yaml"
    )
    model = config["shared_success_probability_model"]
    assert model["separate_a_b_models"] is False
    assert model["pairwise_head"] is False
    assert config["label_rollout"]["nested_child_downstream_rollouts"] is False
    assert config["inference_roles"]["judger_b"]["selection"] == "argmax_q_child"
    assert config["label_rollout"]["target_valid_continuations_per_parent"] == 8
    assert (
        config["label_rollout"]["minimum_valid_continuations_for_formal_analysis"]
        == 6
    )
    assert config["label_rollout"]["cap_hit"]["main_analysis_outcome"] == "unsolved"
    retry = config["ranked_alternative_retry"]
    assert retry["children_n_sweep"] == [2, 4]
    assert retry["low_q_action"] == "cold_continue"
    assert retry["low_q_action_sweep"] == ["cold_continue", "ranked_rollback"]
    assert retry["max_candidate_attempts_p_sweep"] == [1, 2, 3, 4]
    assert retry["low_q_threshold_sweep"] == [0.1, 0.2, 0.3, 0.4]
    assert retry["q_reassessment_interval_steps_sweep"] == [1, 2, 4, 6]
    assert retry["ranked_rollback_after_p_attempts_exhausted"] == (
        "terminate_branch_search_as_unsolved"
    )


def test_child_q_audit_is_separate_and_bounded() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs/experiments/phase4_child_q_audit.yaml"
    )
    assert config["experiment"]["status"] == "planned_not_launched"
    assert config["experiment"]["separate_from_primary_phase4_dataset"] is True
    assert config["children"] == {
        "per_parent": 4,
        "branch_span_steps": 6,
        "source": "saved_prefix_checkpoints_from_distinct_primary_continuations",
        "sampling": "fixed_seed_without_replacement",
    }
    assert config["child_q_rollout"]["target_valid_continuations_per_child"] == 8


def test_screening_v6_freezes_work_conserving_four_slot_schedule() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs/experiments/difficulty_screen_v6_k5_20step.yaml"
    )
    screen = config["screening"]
    assert screen["scheduling"] == "rolling_work_conserving_task_pool"
    assert screen["parallel_tasks"] == 4
    assert screen["parallel_runs_per_task"] == 1
    assert screen["agent_base_urls"].count("http://127.0.0.1:11436") == 2
    assert screen["agent_base_urls"].count("http://127.0.0.1:11437") == 2
