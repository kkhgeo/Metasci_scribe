"""PDF text extraction from ``Knowledge/``."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from filelock import FileLock

from metasci_scribe import knowledge
from metasci_scribe.knowledge import SyncReport, sync_knowledge
from metasci_scribe.project import extract_dir
from tests.fakes.pdf import (
    make_broken_pdf,
    make_encrypted_pdf,
    make_scan_pdf,
    make_text_pdf,
)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "paper"
    (root / "Knowledge").mkdir(parents=True)
    (root / ".metasci-scribe").mkdir()
    return root


def _outputs(root: Path) -> list[Path]:
    out = extract_dir(root)
    return sorted(p for p in out.glob("*.md")) if out.exists() else []


def test_extract_marks_pages(project):
    make_text_pdf(
        project / "Knowledge" / "smith.pdf", ["Alpha finding", "Beta finding"]
    )
    report = sync_knowledge(project)
    assert report.extracted == [project / "Knowledge" / "smith.pdf"]
    assert not report.failed
    assert not report.scanned
    (out,) = _outputs(project)
    text = out.read_text(encoding="utf-8")
    assert "[p. 1]" in text
    assert "[p. 2]" in text
    assert text.index("[p. 1]") < text.index("Alpha finding") < text.index("[p. 2]")
    assert text.index("[p. 2]") < text.index("Beta finding")
    assert "smith.pdf" in text.splitlines()[0]


def test_unchanged_file_is_skipped(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    first = sync_knowledge(project)
    assert len(first.extracted) == 1
    (out,) = _outputs(project)
    stamp = out.stat().st_mtime_ns
    second = sync_knowledge(project)
    assert second.unchanged == 1
    assert second.extracted == []
    assert out.stat().st_mtime_ns == stamp


def test_changed_file_is_reextracted(project):
    pdf = project / "Knowledge" / "a.pdf"
    make_text_pdf(pdf, "First version")
    sync_knowledge(project)
    make_text_pdf(pdf, ["Second version", "with a second page"])
    report = sync_knowledge(project)
    assert report.extracted == [pdf]
    assert report.unchanged == 0
    (out,) = _outputs(project)
    text = out.read_text(encoding="utf-8")
    assert "Second version" in text
    assert "First version" not in text


def test_same_size_edit_with_new_mtime_is_reextracted(project):
    pdf = project / "Knowledge" / "a.pdf"
    make_text_pdf(pdf, "Version AAAA")
    sync_knowledge(project)
    size = pdf.stat().st_size
    make_text_pdf(pdf, "Version BBBB")
    assert pdf.stat().st_size == size
    stat = pdf.stat()
    os.utime(pdf, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert sync_knowledge(project).extracted == [pdf]
    assert "BBBB" in _outputs(project)[0].read_text(encoding="utf-8")


def test_scan_pdf_reported_not_failed(project):
    scan = make_scan_pdf(project / "Knowledge" / "scan.pdf")
    report = sync_knowledge(project)
    assert report.scanned == [scan]
    assert report.failed == []
    assert report.extracted == []
    assert _outputs(project) == []
    # Known as a scan: it is reported again without reading it again.
    again = sync_knowledge(project)
    assert again.scanned == [scan]


def test_scan_then_text_version_is_extracted(project):
    pdf = project / "Knowledge" / "a.pdf"
    make_scan_pdf(pdf)
    sync_knowledge(project)
    make_text_pdf(pdf, "Now with text")
    report = sync_knowledge(project)
    assert report.extracted == [pdf]
    assert report.scanned == []


def test_broken_and_encrypted_do_not_stop_others(project):
    know = project / "Knowledge"
    broken = make_broken_pdf(know / "a-broken.pdf")
    locked = make_encrypted_pdf(know / "b-locked.pdf")
    good = make_text_pdf(know / "c-good.pdf", "Good text")
    empty = know / "d-empty.pdf"
    empty.write_bytes(b"")
    report = sync_knowledge(project)
    assert report.extracted == [good]
    failed = dict(report.failed)
    assert set(failed) == {broken, locked, empty}
    assert "password" in failed[locked].lower() or "encrypt" in failed[locked].lower()
    assert all(reason and "\n" not in reason for reason in failed.values())
    assert len(_outputs(project)) == 1
    # Unreadable files are reported again, not read again.
    assert {p for p, _ in sync_knowledge(project).failed} == {broken, locked, empty}


def test_korean_filename_with_spaces(project):
    pdf = make_text_pdf(project / "Knowledge" / "논문 한글 파일 (최종).pdf", "Body")
    report = sync_knowledge(project)
    assert report.extracted == [pdf]
    (out,) = _outputs(project)
    assert out.parent == extract_dir(project)
    assert "논문 한글 파일" in out.name
    assert " " in out.name
    assert "Body" in out.read_text(encoding="utf-8")


def test_original_pdf_unchanged(project):
    pdf = make_text_pdf(project / "Knowledge" / "a.pdf", ["x", "y"])
    scan = make_scan_pdf(project / "Knowledge" / "s.pdf")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (pdf, scan)}
    sync_knowledge(project)
    sync_knowledge(project)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (pdf, scan)}
    assert before == after


def test_searches_subfolders_and_ignores_other_files(project):
    know = project / "Knowledge"
    make_text_pdf(know / "deep" / "er" / "x.pdf", "Deep text")
    make_text_pdf(know / "UPPER.PDF", "Upper text")
    (know / "notes.txt").write_text("not a pdf", encoding="utf-8")
    report = sync_knowledge(project)
    assert {p.name for p in report.extracted} == {"x.pdf", "UPPER.PDF"}
    assert len(_outputs(project)) == 2


def test_same_names_in_subfolders_and_case_variants_do_not_collide(project):
    know = project / "Knowledge"
    make_text_pdf(know / "a" / "Paper.pdf", "from a upper")
    make_text_pdf(know / "b" / "paper.pdf", "from b lower")
    make_text_pdf(know / "c" / "Paper.pdf", "from c upper")
    make_text_pdf(know / "Paper.pdf", "from the top")
    report = sync_knowledge(project)
    assert len(report.extracted) == 4
    outs = _outputs(project)
    assert len(outs) == 4
    # No two names are equal on a case-insensitive disk.
    assert len({p.name.casefold() for p in outs}) == 4
    texts = " ".join(p.read_text(encoding="utf-8") for p in outs)
    for needle in ("from a upper", "from b lower", "from c upper", "from the top"):
        assert needle in texts


@pytest.mark.parametrize(
    "rel",
    [
        "plain.pdf",
        "CON.pdf",
        "nul",
        "a:b.pdf",
        "trailing dot..pdf",
        "end.",
        ".hidden.pdf",
        "..pdf",
        "...",
        "sub/..x/y.pdf",
        'q"u<o>t|e?s*.pdf',
        "tab\tname\x00nul.pdf",
        "긴" * 300 + ".pdf",
        "é.pdf",
    ],
)
def test_output_names_stay_inside_the_extract_dir(rel, tmp_path):
    out = tmp_path / "extracted"
    out.mkdir()
    name = knowledge._output_name(rel, set())
    assert name.endswith(".md")
    assert name == Path(name).name
    assert "/" not in name
    assert "\\" not in name
    assert "\x00" not in name
    assert not name.startswith(".")
    assert not name[:-3].endswith((" ", "."))
    assert len(name) < 120
    assert (out / name).parent == out
    assert (out / name).resolve().parent == out.resolve()
    (out / name).write_text("x", encoding="utf-8")  # a legal file name here


def test_output_names_are_stable_and_distinct():
    taken: set[str] = set()
    assert knowledge._output_name("a/x.pdf", taken) == knowledge._output_name(
        "a/x.pdf", taken
    )
    names = {
        knowledge._output_name(rel, set())
        for rel in ("a/x.pdf", "A/x.pdf", "a/X.pdf", "a__x.pdf", "a/x.PDF")
    }
    assert len({n.casefold() for n in names}) == 5


def test_name_clash_gets_another_name(monkeypatch):
    real = hashlib.sha256
    monkeypatch.setattr(knowledge.hashlib, "sha256", lambda data: real(b"same-for-all"))
    first = knowledge._output_name("a.pdf", set())
    second = knowledge._output_name("a.PDF", {first.casefold()})
    third = knowledge._output_name("a.Pdf", {first.casefold(), second.casefold()})
    assert len({first.casefold(), second.casefold(), third.casefold()}) == 3


def test_page_limit_is_a_failure_with_a_reason(project, monkeypatch):
    monkeypatch.setattr(knowledge, "MAX_PAGES", 2)
    big = make_text_pdf(project / "Knowledge" / "big.pdf", ["1", "2", "3"])
    ok = make_text_pdf(project / "Knowledge" / "ok.pdf", ["1", "2"])
    report = sync_knowledge(project)
    assert report.extracted == [ok]
    ((path, reason),) = report.failed
    assert path == big
    assert "3 pages" in reason
    assert "2" in reason
    assert len(_outputs(project)) == 1


def test_size_limit_is_a_failure_with_a_reason(project, monkeypatch):
    monkeypatch.setattr(knowledge, "MAX_BYTES", 100)
    big = make_text_pdf(project / "Knowledge" / "big.pdf", "x" * 500)
    report = sync_knowledge(project)
    ((path, reason),) = report.failed
    assert path == big
    assert "too large" in reason.lower()


def _sleeper(monkeypatch, seconds=60):
    monkeypatch.setattr(
        knowledge,
        "_worker_command",
        lambda *a: [sys.executable, "-c", f"import time; time.sleep({seconds})"],
    )


def test_time_limit_kills_a_hung_extraction(project, monkeypatch):
    stuck = make_text_pdf(project / "Knowledge" / "stuck.pdf")
    monkeypatch.setattr(knowledge, "MAX_SECONDS", 1)
    _sleeper(monkeypatch)
    start = time.monotonic()
    report = sync_knowledge(project)
    assert time.monotonic() - start < 20
    ((path, reason),) = report.failed
    assert path == stuck
    assert "1 s" in reason
    assert not [p for p in extract_dir(project).iterdir() if "tmp" in p.name]


def test_cancel_stops_between_files_and_kills_the_worker(project, monkeypatch):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    make_text_pdf(project / "Knowledge" / "b.pdf")
    _sleeper(monkeypatch)
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    start = time.monotonic()
    report = sync_knowledge(project, cancel=cancel)
    assert time.monotonic() - start < 20
    assert report.failed == []
    assert report.extracted == []


def test_file_that_vanishes_mid_run_is_skipped(project, monkeypatch):
    gone = make_text_pdf(project / "Knowledge" / "gone.pdf")
    keep = make_text_pdf(project / "Knowledge" / "keep.pdf")
    real = knowledge._run_worker

    def run(pdf, tmp, label, cancel):
        if pdf.name == "gone.pdf":
            pdf.unlink()
        return real(pdf, tmp, label, cancel)

    monkeypatch.setattr(knowledge, "_run_worker", run)
    report = sync_knowledge(project)
    assert report.extracted == [keep]
    assert report.failed == []
    assert report.scanned == []
    assert not gone.exists()
    assert len(_outputs(project)) == 1


def test_file_missing_when_listed_is_skipped(project, monkeypatch):
    make_text_pdf(project / "Knowledge" / "keep.pdf")
    make_text_pdf(project / "Knowledge" / "gone.pdf")
    real_stat = os.stat

    def stat(path, *a, **kw):
        if str(path).endswith("gone.pdf"):
            raise FileNotFoundError(path)
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(knowledge.os, "stat", stat)
    report = sync_knowledge(project)
    assert [p.name for p in report.extracted] == ["keep.pdf"]
    assert report.failed == []


def test_deleted_pdf_loses_its_extract(project):
    know = project / "Knowledge"
    a = make_text_pdf(know / "a.pdf", "Text of a")
    make_text_pdf(know / "b.pdf", "Text of b")
    sync_knowledge(project)
    assert len(_outputs(project)) == 2
    a.unlink()
    report = sync_knowledge(project)
    assert report.unchanged == 1
    (out,) = _outputs(project)
    assert "Text of b" in out.read_text(encoding="utf-8")


def test_missing_knowledge_folder_is_not_an_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert sync_knowledge(root) == SyncReport([], 0, [], [])
    assert not (root / ".metasci-scribe").exists()


def test_missing_knowledge_folder_keeps_old_extracts(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    sync_knowledge(project)
    (project / "Knowledge" / "a.pdf").unlink()
    (project / "Knowledge").rmdir()
    sync_knowledge(project)
    assert len(_outputs(project)) == 1


def test_damaged_index_is_rebuilt(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    sync_knowledge(project)
    (extract_dir(project) / knowledge.INDEX_NAME).write_text("{ not json", "utf-8")
    report = sync_knowledge(project)
    assert len(report.extracted) == 1
    assert sync_knowledge(project).unchanged == 1


def test_output_deleted_by_hand_is_extracted_again(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    sync_knowledge(project)
    _outputs(project)[0].unlink()
    assert len(sync_knowledge(project).extracted) == 1


def test_leftover_temp_files_are_removed(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    out = extract_dir(project)
    out.mkdir(parents=True)
    leftover = out / f"{knowledge.TMP_PREFIX}dead.md"
    leftover.write_text("half", encoding="utf-8")
    sync_knowledge(project)
    assert not leftover.exists()


def test_files_not_made_by_the_app_are_left_alone(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    out = extract_dir(project)
    out.mkdir(parents=True)
    mine = out / "my notes.md"
    mine.write_text("keep me", encoding="utf-8")
    sync_knowledge(project)
    assert mine.read_text(encoding="utf-8") == "keep me"


def test_progress_is_reported_before_each_extraction(project):
    know = project / "Knowledge"
    make_text_pdf(know / "a.pdf")
    make_text_pdf(know / "b.pdf")
    calls: list[tuple[int, int, str]] = []
    sync_knowledge(project, progress=lambda i, n, p: calls.append((i, n, p.name)))
    assert [(i, n) for i, n, _ in calls] == [(1, 2), (2, 2)]
    calls.clear()
    sync_knowledge(project, progress=lambda i, n, p: calls.append((i, n, p.name)))
    assert calls == []


def test_a_failing_progress_callback_does_not_stop_the_sync(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")

    def boom(i, n, p):
        raise RuntimeError("ui is gone")

    assert len(sync_knowledge(project, progress=boom).extracted) == 1


def test_link_out_of_knowledge_is_not_read(project, tmp_path):
    outside = make_text_pdf(tmp_path / "outside.pdf", "Outside secret")
    link = project / "Knowledge" / "link.pdf"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    report = sync_knowledge(project)
    assert report.extracted == []
    ((path, reason),) = report.failed
    assert path == link
    assert "outside" in reason
    assert _outputs(project) == []


def test_lone_surrogates_are_replaced_in_the_output(tmp_path, monkeypatch):
    from pypdf import PageObject

    from metasci_scribe import _pdf_worker

    pdf = make_text_pdf(tmp_path / "a.pdf", "x")
    monkeypatch.setattr(
        PageObject, "extract_text", lambda self, *a, **k: "bad \ud800 text 한글"
    )
    out = tmp_path / "out.md"
    result = _pdf_worker.extract(pdf, out, 10, "Knowledge/a.pdf")
    assert result["status"] == "ok"
    text = out.read_bytes().decode("utf-8")
    assert "bad" in text
    assert "text 한글" in text
    assert "\ud800" not in text


def test_a_page_that_fails_does_not_lose_the_others(tmp_path, monkeypatch):
    from pypdf import PageObject

    from metasci_scribe import _pdf_worker

    pdf = make_text_pdf(tmp_path / "a.pdf", ["one", "two", "three"])
    real = PageObject.extract_text
    calls = {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("bad page")
        return real(self, *a, **k)

    monkeypatch.setattr(PageObject, "extract_text", flaky)
    out = tmp_path / "out.md"
    result = _pdf_worker.extract(pdf, out, 10, "a.pdf")
    assert result["status"] == "ok"
    text = out.read_text(encoding="utf-8")
    assert "one" in text
    assert "three" in text
    assert "[p. 2]" in text


def test_worker_reports_a_missing_file(tmp_path):
    from metasci_scribe import _pdf_worker

    result = _pdf_worker.extract(tmp_path / "nope.pdf", tmp_path / "o.md", 10, "x")
    assert result["status"] == "missing"


def test_worker_prints_one_json_line_and_ignores_the_cwd(tmp_path):
    """The real command: isolated, so the folder it runs in cannot shadow modules."""
    import subprocess

    pdf = make_text_pdf(tmp_path / "a.pdf", "Hi")
    (tmp_path / "pypdf.py").write_text("raise SystemExit('shadowed')", "utf-8")
    out = tmp_path / "o.md"
    done = subprocess.run(
        knowledge._worker_command(pdf, out, "a.pdf"),
        capture_output=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.decode().strip().splitlines()[-1])["status"] == "ok"


def test_agent_can_read_but_not_write_the_extracts(project):
    from metasci_scribe.backends import CustomSandboxBackend

    make_text_pdf(project / "Knowledge" / "논문 A.pdf", ["Cited claim", "More"])
    sync_knowledge(project)
    (out,) = _outputs(project)
    backend = CustomSandboxBackend(root_dir=str(project), virtual_mode=True)
    key = f"/.metasci-scribe/extracted/{out.name}"
    read = backend.read(key)
    assert read.error is None
    assert "[p. 2]" in read.file_data["content"]
    assert "Cited claim" in read.file_data["content"]
    listing = backend.ls("/.metasci-scribe/extracted")
    assert out.name in [e["path"].rsplit("/", 1)[-1] for e in listing.entries]
    assert backend.write(key, "tampered").error
    assert backend.edit(key, "Cited claim", "Invented claim").error
    assert backend.delete(key).error
    assert backend.write("/.metasci-scribe/extracted/new.md", "x").error
    assert "Cited claim" in out.read_text(encoding="utf-8")
    assert not (extract_dir(project) / "new.md").exists()


# -- report text -------------------------------------------------------------


def test_report_messages_are_quiet_when_nothing_changed(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    sync_knowledge(project)
    assert knowledge.report_messages(sync_knowledge(project), project) == []


def test_report_messages_name_scans_and_failures(project):
    know = project / "Knowledge"
    make_text_pdf(know / "ok.pdf")
    make_scan_pdf(know / "scan one.pdf")
    make_broken_pdf(know / "bad.pdf")
    messages = knowledge.report_messages(sync_knowledge(project), project)
    joined = "\n".join(messages)
    assert "1 PDF" in messages[0]
    assert "text" in messages[0]
    assert "scan one.pdf" in joined
    assert "OCR" in joined
    assert "bad.pdf" in joined


def test_report_messages_cut_long_lists():
    root = Path("/p")
    report = SyncReport(
        [],
        0,
        [root / "Knowledge" / f"s{i}.pdf" for i in range(12)],
        [],
    )
    (line,) = knowledge.report_messages(report, root)
    assert "12 PDFs" in line
    assert "s0.pdf" in line
    assert "and 7 more" in line
    assert "s11.pdf" not in line


# -- start of the app --------------------------------------------------------


def test_start_sync_does_not_block_the_event_loop(project, monkeypatch):
    from metasci_scribe.cli import interactive

    release = threading.Event()
    started = threading.Event()

    def slow(root, *, progress=None, cancel=None):
        started.set()
        progress(1, 3, root / "Knowledge" / "a.pdf")
        release.wait(10)
        return SyncReport([root / "Knowledge" / "a.pdf"], 0, [], [])

    monkeypatch.setattr(knowledge, "sync_knowledge", slow)
    lines: list[str] = []

    async def scenario():
        handle = interactive.start_knowledge_sync(project, lines.append)
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        # The loop is free while the thread works.
        await asyncio.sleep(0.05)
        assert not handle.done()
        release.set()
        await asyncio.wait_for(handle.wait(), 10)

    asyncio.run(scenario())
    assert any("3 PDF" in line and "background" in line for line in lines)
    assert any("text" in line for line in lines[1:])


def test_start_sync_swallows_a_crash_with_a_warning(project, monkeypatch):
    from metasci_scribe.cli import interactive

    def boom(root, *, progress=None, cancel=None):
        raise OSError("disk went away")

    monkeypatch.setattr(knowledge, "sync_knowledge", boom)
    lines: list[str] = []

    async def scenario():
        handle = interactive.start_knowledge_sync(project, lines.append)
        await asyncio.wait_for(handle.wait(), 10)

    asyncio.run(scenario())
    assert len(lines) == 1
    assert "disk went away" in lines[0]


def test_start_sync_cancel_reaches_the_sync(project, monkeypatch):
    from metasci_scribe.cli import interactive

    seen: dict[str, threading.Event] = {}

    def waits(root, *, progress=None, cancel=None):
        seen["cancel"] = cancel
        cancel.wait(10)
        return SyncReport([], 0, [], [])

    monkeypatch.setattr(knowledge, "sync_knowledge", waits)

    async def scenario():
        handle = interactive.start_knowledge_sync(project, lambda line: None)
        await asyncio.sleep(0.1)
        handle.cancel()
        await asyncio.wait_for(handle.wait(), 10)

    asyncio.run(scenario())
    assert seen["cancel"].is_set()


# -- fix round 1 ---------------------------------------------------------------


def test_a_busy_lock_is_reported_as_busy_not_as_nothing_to_do(project, monkeypatch):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    out = extract_dir(project)
    out.mkdir(parents=True)
    monkeypatch.setattr(knowledge, "LOCK_WAIT_SECONDS", 0.2)
    with FileLock(out / knowledge.LOCK_NAME):
        report = sync_knowledge(project)
        assert report.busy
        assert report != SyncReport([], 0, [], [])
        start = time.monotonic()
        assert sync_knowledge(project, wait=0).busy
        assert time.monotonic() - start < 2
    assert _outputs(project) == []
    assert not sync_knowledge(project).busy


def test_a_waiting_sync_goes_on_when_the_lock_is_released(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    out = extract_dir(project)
    out.mkdir(parents=True)
    held = threading.Event()

    def hold():  # a lock is released by the thread that took it
        with FileLock(out / knowledge.LOCK_NAME):
            held.set()
            time.sleep(0.5)

    threading.Thread(target=hold).start()
    assert held.wait(10)
    report = sync_knowledge(project, wait=30)
    assert not report.busy
    assert len(report.extracted) == 1


def test_cancel_ends_the_wait_for_a_busy_lock(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    out = extract_dir(project)
    out.mkdir(parents=True)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    with FileLock(out / knowledge.LOCK_NAME):
        start = time.monotonic()
        report = sync_knowledge(project, cancel=cancel, wait=60)
    assert time.monotonic() - start < 10
    assert not report.busy


def test_report_messages_say_when_the_folder_is_busy(project):
    lines = knowledge.report_messages(SyncReport([], 0, [], [], busy=True), project)
    assert len(lines) == 1
    assert "another" in lines[0]


def _fake_reparse_lstat(monkeypatch, target_name):
    """lstat that gives *target_name* the reparse-point bit, as a junction
    looks on Windows (and on 3.11, which has no os.path.isjunction)."""
    real = os.lstat

    def lstat(path, *a, **kw):
        info = real(path, *a, **kw)
        if Path(path).name != target_name:
            return info

        class Fake:
            st_mode = info.st_mode
            st_file_attributes = 0x400

        return Fake()

    monkeypatch.setattr(knowledge.os, "lstat", lstat)


def test_a_reparse_point_is_a_link_without_os_path_isjunction(project, monkeypatch):
    monkeypatch.delattr(os.path, "isjunction", raising=False)
    make_text_pdf(project / "Knowledge" / "ok.pdf", "fine")
    make_text_pdf(project / "Knowledge" / "jump" / "inner.pdf", "behind the point")
    _fake_reparse_lstat(monkeypatch, "jump")
    found, complete = knowledge._discover(project / "Knowledge")
    assert [s.rel for s in found] == ["ok.pdf"]
    assert complete
    assert knowledge._is_link(str(project / "Knowledge" / "jump"))
    assert not knowledge._is_link(str(project / "Knowledge" / "ok.pdf"))


@pytest.mark.skipif(os.name != "nt", reason="junctions are a Windows thing")
def test_a_real_junction_is_not_followed(project, tmp_path):
    import subprocess

    outside = tmp_path / "elsewhere"
    make_text_pdf(outside / "secret.pdf", "Outside secret")
    make_text_pdf(project / "Knowledge" / "ok.pdf", "fine")
    junction = project / "Knowledge" / "jump"
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
    )
    if done.returncode != 0:
        pytest.skip("junctions could not be created here")
    report = sync_knowledge(project)
    assert [p.name for p in report.extracted] == ["ok.pdf"]
    assert report.failed == []
    texts = " ".join(p.read_text(encoding="utf-8") for p in _outputs(project))
    assert "Outside secret" not in texts


def test_a_file_whose_real_path_is_outside_is_never_read(project, monkeypatch):
    make_text_pdf(project / "Knowledge" / "evil.pdf", "Outside secret")
    make_text_pdf(project / "Knowledge" / "ok.pdf", "fine")
    real = os.path.realpath

    def realpath(path, *a, **kw):
        if str(path).endswith("evil.pdf"):
            return os.path.join(os.path.dirname(real(project)), "elsewhere", "x.pdf")
        return real(path, *a, **kw)

    monkeypatch.setattr(knowledge.os.path, "realpath", realpath)
    report = sync_knowledge(project)
    assert [p.name for p in report.extracted] == ["ok.pdf"]
    ((path, reason),) = report.failed
    assert path.name == "evil.pdf"
    assert "outside" in reason


def test_worker_stops_itself_after_its_own_time_limit(tmp_path):
    import subprocess

    code = (
        "import time, sys;"
        "import metasci_scribe._pdf_worker as w;"
        "w.extract = lambda *a: time.sleep(60);"
        "sys.exit(w.main(['w', 'a.pdf', 'o.md', '10', 'x', '1']))"
    )
    start = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, cwd=tmp_path, timeout=40
    )
    assert time.monotonic() - start < 30
    assert done.returncode == 0
    last = done.stdout.decode().strip().splitlines()[-1]
    assert json.loads(last) == {"status": "timeout"}


def test_worker_is_given_a_limit_longer_than_the_parents(tmp_path):
    command = knowledge._worker_command(tmp_path / "a.pdf", tmp_path / "o", "x")
    assert float(command[-1]) > knowledge.MAX_SECONDS


def test_deeply_nested_index_is_rebuilt(project):
    make_text_pdf(project / "Knowledge" / "a.pdf")
    sync_knowledge(project)
    depth = 200_000
    (extract_dir(project) / knowledge.INDEX_NAME).write_text(
        '{"version": 1, "files": ' + "[" * depth + "]" * depth + "}", "utf-8"
    )
    assert len(sync_knowledge(project).extracted) == 1
    assert sync_knowledge(project).unchanged == 1


@pytest.mark.parametrize(
    "name",
    [
        "../evil.md",
        "..\\evil.md",
        "evil.md",
        ".hidden.0123456789.md",
        "a.0123456789.txt",
        "a.xyz.md",
        "a/b.0123456789.md",
        "a\x00b.0123456789.md",
        "",
        None,
        3,
    ],
)
def test_index_output_names_must_look_like_ours(name):
    assert not knowledge._safe_out(name)


def test_index_output_names_we_make_are_accepted():
    for rel in ("a.pdf", "논문 A.pdf", "x/CON.pdf", "é.pdf"):
        assert knowledge._safe_out(knowledge._output_name(rel, set()))
    first = knowledge._output_name("a.pdf", set())
    assert knowledge._safe_out(knowledge._output_name("a.PDF", {first.casefold()}))


def test_a_tampered_index_cannot_make_us_delete_other_files(project):
    know = project / "Knowledge"
    pdf = make_text_pdf(know / "a.pdf", "Text")
    sync_knowledge(project)
    out = extract_dir(project)
    victim = out / "victim.md"
    victim.write_text("not ours", encoding="utf-8")
    outside = project / "outside.md"
    outside.write_text("not ours", encoding="utf-8")
    index_path = out / knowledge.INDEX_NAME
    data = json.loads(index_path.read_text(encoding="utf-8"))
    for rel, target in (("gone.pdf", "victim.md"), ("gone2.pdf", "../../outside.md")):
        data["files"][rel] = {"status": "ok", "size": 1, "mtime_ns": 1, "out": target}
    index_path.write_text(json.dumps(data), encoding="utf-8")
    pdf.unlink()
    sync_knowledge(project)
    assert victim.exists()
    assert outside.exists()


def test_two_index_entries_claiming_one_file_are_read_again(project):
    know = project / "Knowledge"
    make_text_pdf(know / "a.pdf", "Text of a")
    make_text_pdf(know / "b.pdf", "Text of b")
    sync_knowledge(project)
    index_path = extract_dir(project) / knowledge.INDEX_NAME
    data = json.loads(index_path.read_text(encoding="utf-8"))
    files = data["files"]
    files["b.pdf"]["out"] = files["a.pdf"]["out"]
    index_path.write_text(json.dumps(data), encoding="utf-8")
    report = sync_knowledge(project)
    assert [p.name for p in report.extracted] == ["b.pdf"]
    texts = " ".join(p.read_text(encoding="utf-8") for p in _outputs(project))
    assert len(_outputs(project)) == 2
    assert "Text of a" in texts
    assert "Text of b" in texts


def test_a_path_too_long_for_windows_is_a_failure_not_a_vanished_file(
    project, monkeypatch
):
    pdf = make_text_pdf(project / "Knowledge" / "longname.pdf")
    make_text_pdf(project / "Knowledge" / "ok.pdf")
    monkeypatch.setattr(knowledge, "_IS_WINDOWS", True)
    monkeypatch.setattr(knowledge, "_PATH_LIMIT", len(str(pdf)))
    real_stat = os.stat

    def stat(path, *a, **kw):
        if str(path).endswith("longname.pdf"):
            raise FileNotFoundError(path)
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(knowledge.os, "stat", stat)
    report = sync_knowledge(project)
    ((path, reason),) = report.failed
    assert path.name == "longname.pdf"
    assert "too long" in reason


def test_a_worker_that_cannot_open_a_long_path_is_a_failure(project, monkeypatch):
    pdf = make_text_pdf(project / "Knowledge" / "longname.pdf")
    monkeypatch.setattr(knowledge, "_IS_WINDOWS", True)
    monkeypatch.setattr(knowledge, "_PATH_LIMIT", len(str(pdf)))
    monkeypatch.setattr(knowledge, "_run_worker", lambda *a: {"status": "missing"})
    ((path, reason),) = sync_knowledge(project).failed
    assert path == pdf
    assert "too long" in reason


def test_a_name_that_cannot_be_passed_to_the_reader_is_reported(project, monkeypatch):
    make_text_pdf(project / "Knowledge" / "odd.pdf")

    def command(pdf, tmp, label):
        raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")

    monkeypatch.setattr(knowledge, "_worker_command", command)
    report = sync_knowledge(project)
    ((path, reason),) = report.failed
    assert path.name == "odd.pdf"
    assert "file name" in reason
    assert report.extracted == []


@pytest.mark.parametrize("status", ["timeout", "crashed"])
def test_a_timeout_or_crash_is_tried_once_more_later(project, monkeypatch, status):
    pdf = make_text_pdf(project / "Knowledge" / "a.pdf")
    calls: list[str] = []

    def run(pdf_path, tmp, label, cancel):
        calls.append(pdf_path.name)
        return {"status": status, "error": "the reader stopped unexpectedly"}

    monkeypatch.setattr(knowledge, "_run_worker", run)
    first = sync_knowledge(project)
    assert [p for p, _ in first.failed] == [pdf]
    assert len(calls) == 1
    second = sync_knowledge(project)
    assert [p for p, _ in second.failed] == [pdf]
    assert len(calls) == 2
    third = sync_knowledge(project)
    assert [p for p, _ in third.failed] == [pdf]
    assert len(calls) == 2  # now remembered
    # A changed file starts again.
    make_text_pdf(pdf, ["changed", "again"])
    sync_knowledge(project)
    assert len(calls) == 3


def test_a_retry_that_succeeds_extracts_the_file(project, monkeypatch):
    pdf = make_text_pdf(project / "Knowledge" / "a.pdf", "Fine text")
    real = knowledge._run_worker
    monkeypatch.setattr(knowledge, "_run_worker", lambda *a: {"status": "timeout"})
    assert len(sync_knowledge(project).failed) == 1
    monkeypatch.setattr(knowledge, "_run_worker", real)
    report = sync_knowledge(project)
    assert report.extracted == [pdf]
    assert report.failed == []


def test_start_sync_survives_a_thread_that_cannot_start(project, monkeypatch):
    from metasci_scribe.cli import interactive

    def refuse(self):
        raise RuntimeError("no more threads")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    lines: list[str] = []

    async def scenario():
        handle = interactive.start_knowledge_sync(project, lines.append)
        await asyncio.wait_for(handle.wait(), 10)
        assert handle.done()

    asyncio.run(scenario())
    assert len(lines) == 1
    assert "no more threads" in lines[0]


def test_start_sync_survives_a_failing_say_and_an_import_error(project, monkeypatch):
    from metasci_scribe.cli import interactive

    def boom(root, *, progress=None, cancel=None):
        raise ImportError("no filelock")

    monkeypatch.setattr(knowledge, "sync_knowledge", boom)

    def bad_say(line):
        raise RuntimeError("console is gone")

    async def scenario():
        handle = interactive.start_knowledge_sync(project, bad_say)
        await asyncio.wait_for(handle.wait(), 10)

    asyncio.run(scenario())


def test_start_sync_without_a_running_loop_does_not_raise(project, monkeypatch):
    from metasci_scribe.cli import interactive

    done = threading.Event()

    def quick(root, *, progress=None, cancel=None):
        done.set()
        return SyncReport([], 0, [], [])

    monkeypatch.setattr(knowledge, "sync_knowledge", quick)
    handle = interactive.start_knowledge_sync(project, lambda line: None)
    assert done.wait(10)
    assert handle.done()


def test_stop_sync_cancels_and_waits_for_the_thread(project, monkeypatch):
    from metasci_scribe.cli import interactive

    ended = threading.Event()

    def waits(root, *, progress=None, cancel=None):
        cancel.wait(10)
        time.sleep(0.2)
        ended.set()
        return SyncReport([], 0, [], [])

    monkeypatch.setattr(knowledge, "sync_knowledge", waits)

    async def scenario():
        handle = interactive.start_knowledge_sync(project, lambda line: None)
        await asyncio.sleep(0.1)
        await interactive.stop_knowledge_sync(handle, 5)
        assert ended.is_set()

    asyncio.run(scenario())


def test_stop_sync_gives_up_after_its_time(project, monkeypatch):
    from metasci_scribe.cli import interactive

    release = threading.Event()

    def stuck(root, *, progress=None, cancel=None):
        release.wait(20)
        return SyncReport([], 0, [], [])

    monkeypatch.setattr(knowledge, "sync_knowledge", stuck)

    async def scenario():
        handle = interactive.start_knowledge_sync(project, lambda line: None)
        await asyncio.sleep(0.1)
        start = time.monotonic()
        await interactive.stop_knowledge_sync(handle, 0.3)
        assert time.monotonic() - start < 5
        release.set()

    asyncio.run(scenario())
