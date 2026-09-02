from __future__ import annotations

from adaptive_swe_branching.agents.openhands import (
    _assistant_text,
    _observation_text,
)


def test_assistant_text_accepts_openhands_content_blocks() -> None:
    events = [
        {
            "llm_message": {
                "role": "assistant",
                "tool_calls": [],
                "content": [{"type": "text", "text": "Finished cleanly."}],
            }
        }
    ]

    assert _assistant_text(events) == "Finished cleanly."


def test_observation_text_keeps_terminal_diagnostics() -> None:
    observation = {
        "observation": {
            "content": [{"type": "text", "text": ""}],
            "metadata": {
                "suffix": "The previous command is still running.",
            },
        }
    }

    assert "previous command is still running" in _observation_text(observation)
