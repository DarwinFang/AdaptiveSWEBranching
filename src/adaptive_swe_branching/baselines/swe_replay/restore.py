from __future__ import annotations

import re
from dataclasses import dataclass

from adaptive_swe_branching.baselines.swe_replay.types import ArchivedTrajectory


@dataclass(frozen=True)
class RestorePlan:
    mode: str
    checkpoint_id: str
    accumulated_patch: str
    prefix_actions: tuple[dict, ...]
    matched_non_repo_rules: tuple[str, ...]


_NON_REPO_RULES = {
    "package_manager": re.compile(
        r"\b(?:apt(?:-get)?|yum|dnf|apk|brew|pip(?:3)?|conda|npm|yarn|pnpm)\s+(?:install|remove|uninstall|update)\b",
        re.I,
    ),
    "absolute_write": re.compile(
        r"(?:>|>>|tee|rm\s+-|mv\s+|cp\s+)/(?!testbed(?:/|\b))", re.I
    ),
    "service_or_process": re.compile(
        r"\b(?:systemctl|service|kill|pkill|nohup)\b", re.I
    ),
    "environment": re.compile(r"\b(?:export|unset)\s+[A-Za-z_]", re.I),
    "permissions": re.compile(r"\b(?:chmod|chown)\b", re.I),
}


def restore_plan(archived: ArchivedTrajectory, step_index: int) -> RestorePlan:
    if not 0 <= step_index < len(archived.trajectory.steps):
        raise IndexError(step_index)
    prefix = archived.trajectory.steps[:step_index]
    action_text = "\n".join(step.action_text for step in prefix)
    matched = tuple(
        name for name, pattern in _NON_REPO_RULES.items() if pattern.search(action_text)
    )
    mode = "action_replay" if matched else "git_diff"
    return RestorePlan(
        mode=mode,
        checkpoint_id=archived.pre_step_checkpoint_ids[step_index],
        accumulated_patch=archived.accumulated_patches_before_step[step_index],
        prefix_actions=tuple(step.raw_action for step in prefix),
        matched_non_repo_rules=matched,
    )
