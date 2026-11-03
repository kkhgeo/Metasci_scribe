"""Tests for shared persistent status-bar helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar

import pytest
from langchain_core.messages import HumanMessage

from metasci_scribe.cli.status_bar import (
    PLAN_LABEL,
    RunProgress,
    SessionStatusSnapshot,
    apply_assistant_text_to_snapshot,
    apply_user_text_to_snapshot,
    build_session_status_snapshot,
    build_status_fragments,
    build_status_text,
    make_usage_status_snapshot,
    read_run_progress,
    shorten_model_name,
    status_style_name,
    trim_status_text,
)
from metasci_scribe.memory import worker_activity
from metasci_scribe.run import RunState
from tests.fakes.common import FakeGraphGateway, FakeThreadStore, status_snapshot


def _render_fragments(fragments: list[tuple[str, str]]) -> str:
    return "".join(text for _, text in fragments)


def _display_len(text: str) -> int:
    from prompt_toolkit.utils import get_cwidth

    return get_cwidth(text)


@pytest.fixture
def reset_memory_activity():
    worker_activity.reset_memory_worker_status_for_tests()
    yield
    worker_activity.reset_memory_worker_status_for_tests()


def _default_snapshot() -> SessionStatusSnapshot:
    return SessionStatusSnapshot(
        model_full="openai/gpt-6",
        model_short="gpt-6",
        context_tokens=12_345,
        context_window=128_000,
        context_percent=10,
    )


def test_build_status_fragments_wide_layout():
    snapshot = SessionStatusSnapshot(
        model_full="openai/gpt-5.4",
        model_short="gpt-5.4",
        context_tokens=12_345,
        context_window=128_000,
        context_percent=10,
    )

    rendered = _render_fragments(
        build_status_fragments(
            snapshot,
            datetime.now() - timedelta(minutes=3),
            100,
        )
    )

    assert "gpt-5.4" in rendered
    assert "12.3K/128K" in rendered
    assert "[█████████░] 90% left" in rendered
    assert "3m" in rendered


def test_build_status_fragments_shows_saved_memory_counts(
    tmp_path,
    reset_memory_activity,
):
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    worker_activity.mark_memory_worker_started(
        thread_id="thread-1",
        run_id="run-1",
        memory_dir=memory_dir,
        before_outputs=before,
    )
    profile = memory_dir / "profile" / "USER_PROFILE.md"
    profile.parent.mkdir(parents=True)
    profile.write_text("# User profile\n\n- remembered\n", encoding="utf-8")
    observation = memory_dir / "observations" / "global" / "O-1.md"
    observation.parent.mkdir(parents=True)
    observation.write_text("# Observation\n", encoding="utf-8")
    worker_activity.mark_memory_worker_finished("thread-1", "run-1")

    rendered = _render_fragments(
        build_status_fragments(
            _default_snapshot(),
            datetime.now() - timedelta(minutes=3),
            100,
        )
    )

    assert "Saved 1 profile edit, 1 observation" in rendered


def test_build_status_fragments_shows_memory_worker_indicator_when_running(
    tmp_path,
    reset_memory_activity,
):
    worker_activity.mark_memory_worker_started(
        thread_id="thread-1",
        run_id="run-1",
        memory_dir=tmp_path / "memories",
    )

    rendered = _render_fragments(
        build_status_fragments(
            _default_snapshot(),
            datetime.now() - timedelta(minutes=3),
            100,
        )
    )

    assert "🧠" in rendered


def test_build_status_fragments_shows_observation_linker_indicator_when_running(
    reset_memory_activity,
):
    worker_activity.mark_observation_linker_started(
        thread_id="thread-1",
        run_id="run-1",
    )

    rendered = _render_fragments(
        build_status_fragments(
            _default_snapshot(),
            datetime.now() - timedelta(minutes=3),
            100,
        )
    )

    assert "🔗" in rendered
    assert "🧠" not in rendered


def test_build_status_fragments_shows_linked_relation_count(
    reset_memory_activity,
):
    worker_activity.mark_observation_relations_linked(1)

    rendered = _render_fragments(
        build_status_fragments(
            _default_snapshot(),
            datetime.now() - timedelta(minutes=3),
            100,
        )
    )

    assert "Created 1 memory link" in rendered
    assert "🔗" not in rendered


def test_build_status_fragments_hides_memory_indicator_when_idle(
    reset_memory_activity,
):
    fragments = build_status_fragments(
        _default_snapshot(),
        datetime.now() - timedelta(minutes=3),
        100,
    )

    assert not any(style == "class:status-bar-warn" for style, _text in fragments)


def test_build_status_fragments_medium_layout_drops_the_least_important_first():
    snapshot = SessionStatusSnapshot(
        model_full="openai/gpt-5.4",
        model_short="gpt-5.4",
        context_tokens=48_000,
        context_window=100_000,
        context_percent=48,
    )

    rendered = _render_fragments(
        build_status_fragments(
            snapshot,
            datetime.now() - timedelta(seconds=40),
            52,
            folder="my-paper",
            run_progress=RunProgress(done=3, total=12),
        )
    )

    assert _display_len(rendered) <= 52
    assert "my-paper" in rendered
    assert "gpt-5.4" in rendered
    assert "3/12 done" in rendered
    assert "52% left" in rendered
    # Tokens, then the session time, then the bar go first.
    assert "48K/100K" not in rendered
    assert "40s" not in rendered
    assert "[" not in rendered


def test_build_status_fragments_narrow_layout():
    snapshot = SessionStatusSnapshot(
        model_full="anthropic/claude-sonnet-4-20250514",
        model_short=shorten_model_name("anthropic/claude-sonnet-4-20250514"),
        context_tokens=95_000,
        context_window=100_000,
        context_percent=95,
    )

    rendered = _render_fragments(
        build_status_fragments(
            snapshot,
            datetime.now() - timedelta(hours=2),
            40,
            folder="my-paper",
        )
    )

    assert _display_len(rendered) <= 40
    assert snapshot.model_short in rendered
    assert "5% left" in rendered
    assert "my-paper" not in rendered
    assert "2h" not in rendered


def test_build_status_fragments_wide_layout_with_every_field():
    rendered = _render_fragments(
        build_status_fragments(
            SessionStatusSnapshot(
                model_full="openai/gpt-6.1-sol",
                model_short="gpt-6.1-sol",
                context_tokens=200_000,
                context_window=1_100_000,
                context_percent=18,
            ),
            datetime.now() - timedelta(minutes=3),
            120,
            folder="my-paper",
            on_plan=True,
            run_progress=RunProgress(done=3, total=12, blocked=1),
        )
    )

    assert rendered == (
        " my-paper │ gpt-6.1-sol │ Using ChatGPT plan │ 3/12 done, 1 blocked │ "
        "[████████░░] 82% left │ 200K/1.1M │ 3m "
    )


@pytest.mark.parametrize("width", [80, 52, 40, 30, 24, 19])
def test_the_plan_label_stays_at_every_width(width):
    rendered = _render_fragments(
        build_status_fragments(
            _default_snapshot(),
            datetime.now(),
            width,
            folder="a-long-paper-folder-name",
            on_plan=True,
            run_progress=RunProgress(done=3, total=12),
        )
    )

    assert PLAN_LABEL in rendered
    assert _display_len(rendered) <= width


def test_no_plan_label_without_the_plan():
    rendered = _render_fragments(
        build_status_fragments(_default_snapshot(), datetime.now(), 120)
    )

    assert PLAN_LABEL not in rendered


def test_trim_status_text_adds_ellipsis():
    assert trim_status_text("abcdefghijk", 8) == "abcde..."


def test_status_style_name_thresholds():
    assert status_style_name(49) == "good"
    assert status_style_name(50) == "warn"
    assert status_style_name(80) == "warn"
    assert status_style_name(81) == "bad"
    assert status_style_name(95) == "critical"


def test_build_status_text_uses_rich_styles():
    snapshot = SessionStatusSnapshot(
        model_full="openai/gpt-5.4",
        model_short="gpt-5.4",
        context_tokens=10_000,
        context_window=100_000,
        context_percent=10,
    )

    text = build_status_text(snapshot, datetime.now(), 100)

    assert str(text)
    assert text.spans


async def test_build_session_status_snapshot_uses_fallback_window(monkeypatch):
    class _FakeModel:
        model_name: ClassVar[str] = "provider/demo-model"
        profile: ClassVar[dict[str, object]] = {}

    def _fake_count(messages):
        assert len(messages) == 2
        assert messages[-1].content == "pending"
        return 42_000

    monkeypatch.setattr(
        "metasci_scribe.cli.status_bar._get_default_chat_model",
        lambda: _FakeModel(),
    )
    monkeypatch.setattr(
        "metasci_scribe.cli.status_bar.count_tokens_approximately",
        _fake_count,
    )

    snapshot = await build_session_status_snapshot(
        "thread-1",
        pending_user_text="pending",
        graph_gateway=FakeGraphGateway(
            thread_store=FakeThreadStore(messages=[HumanMessage(content="existing")])
        ),
    )

    assert snapshot.model_full == "provider/demo-model"
    assert snapshot.model_short == "demo-model"
    assert snapshot.context_tokens == 42_000
    assert snapshot.context_window == 200_000
    assert snapshot.context_percent == 21


def test_apply_assistant_text_to_snapshot_updates_context(monkeypatch):
    snapshot = SessionStatusSnapshot(
        model_full="provider/demo-model",
        model_short="demo-model",
        context_tokens=10_000,
        context_window=100_000,
        context_percent=10,
    )

    monkeypatch.setattr(
        "metasci_scribe.cli.status_bar.estimate_message_tokens",
        lambda text, message_type="ai": 550,
    )

    updated = apply_assistant_text_to_snapshot(snapshot, "streamed response")

    assert updated.context_tokens == 10_550
    assert updated.context_percent == 11


def test_apply_user_text_to_snapshot_updates_context(monkeypatch):
    snapshot = SessionStatusSnapshot(
        model_full="provider/demo-model",
        model_short="demo-model",
        context_tokens=46_751,
        context_window=163_840,
        context_percent=29,
        context_source="usage",
    )

    monkeypatch.setattr(
        "metasci_scribe.cli.status_bar.estimate_message_tokens",
        lambda text, message_type="human": 320,
    )

    updated = apply_user_text_to_snapshot(snapshot, "good")

    assert updated.context_tokens == 47_071
    assert updated.context_percent == 29
    assert updated.context_source == "usage"


def test_make_usage_status_snapshot_marks_usage_source(monkeypatch):
    class _FakeModel:
        model_name: ClassVar[str] = "provider/demo-model"
        profile: ClassVar[dict[str, object]] = {"max_input_tokens": 128_000}

    monkeypatch.setattr(
        "metasci_scribe.cli.status_bar._get_default_chat_model",
        lambda: _FakeModel(),
    )

    snapshot = make_usage_status_snapshot(42_000)

    assert snapshot.model_full == "provider/demo-model"
    assert snapshot.model_short == "demo-model"
    assert snapshot.context_tokens == 42_000
    assert snapshot.context_window == 128_000
    assert snapshot.context_percent == 33
    assert snapshot.context_source == "usage"


class TestRunProgress:
    def _save(self, root, paragraphs):
        RunState("thread-1", "digest", paragraphs).save(root)

    def test_counts_the_paragraphs(self, tmp_path):
        self._save(tmp_path, {"R1": "done", "R2": "blocked", "R3": "pending"})

        assert read_run_progress(tmp_path) == RunProgress(done=1, total=3, blocked=1)

    def test_no_file_means_no_field(self, tmp_path):
        assert read_run_progress(tmp_path) is None

    def test_a_damaged_file_hides_the_field(self, tmp_path):
        path = tmp_path / ".metasci-scribe" / "run.json"
        path.parent.mkdir()
        path.write_text("{not json", encoding="utf-8")

        assert read_run_progress(tmp_path) is None

    def test_reads_again_only_when_the_file_changes(self, tmp_path, monkeypatch):
        import os

        calls = []
        load = RunState.load.__func__

        def _counting_load(cls, root):
            calls.append(root)
            return load(cls, root)

        monkeypatch.setattr(RunState, "load", classmethod(_counting_load))
        self._save(tmp_path, {"R1": "pending", "R2": "pending"})

        assert read_run_progress(tmp_path) == RunProgress(done=0, total=2)
        assert read_run_progress(tmp_path) == RunProgress(done=0, total=2)
        assert len(calls) == 1

        self._save(tmp_path, {"R1": "done", "R2": "pending"})
        path = tmp_path / ".metasci-scribe" / "run.json"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        assert read_run_progress(tmp_path) == RunProgress(done=1, total=2)
        assert len(calls) == 2

    def test_a_file_that_goes_away_hides_the_field(self, tmp_path):
        self._save(tmp_path, {"R1": "done"})
        assert read_run_progress(tmp_path) == RunProgress(done=1, total=1)

        (tmp_path / ".metasci-scribe" / "run.json").unlink()

        assert read_run_progress(tmp_path) is None


class TestRubricBadge:
    def test_badge_is_a_strong_fragment_on_wide_bars(self):
        from datetime import datetime

        from metasci_scribe.cli.status_bar import build_status_fragments

        frags = build_status_fragments(
            status_snapshot(), datetime.now(), 120, rubric_badge="on+next"
        )
        assert ("class:status-bar-strong", "rubric: on+next") in frags
        plain = build_status_fragments(status_snapshot(), datetime.now(), 120)
        assert not any("rubric" in text for _style, text in plain)

    def test_badge_is_left_out_of_narrow_bars(self):
        from datetime import datetime

        from metasci_scribe.cli.status_bar import build_status_fragments

        narrow = build_status_fragments(
            status_snapshot(), datetime.now(), 60, rubric_badge="on"
        )
        assert not any("rubric" in text for _style, text in narrow)

    def test_rich_text_carries_the_badge(self):
        from datetime import datetime

        from metasci_scribe.cli.status_bar import build_status_text

        text = build_status_text(
            status_snapshot(), datetime.now(), 120, rubric_badge="next"
        )
        assert "rubric: next" in text.plain
