"""An outline drawn as an HTML page the author opens in a browser.

``outline.md`` (or a proposal written in its format) is read into sections and
paragraphs with their fields, and drawn as one self-contained page: the key
message, and for each section a flow of paragraph cards, the short table
(번호 · 역할 · 핵심 내용) and the detail table (전달할 내용 · 근거 · 연결). Open
choices (◇), evidence still to check and missing evidence are marked.

The app writes the page itself, under ``.metasci-scribe/view/``, so nothing in
the paper folder changes. Nothing is fetched from the network: the page has
its own styles and no scripts.
"""

from __future__ import annotations

import contextlib
import html
import os
import re
import tempfile
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .paths import Workspace

__all__ = [
    "OutlineView",
    "parse_outline_view",
    "render_outline_html",
    "show_outline",
]

_MARKER = "◇"
_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
_SECTION = re.compile(r"^##[ \t]+(?P<name>\S.*?)[ \t]*#*[ \t]*$")
_PARAGRAPH = re.compile(
    r"^###[ \t]+(?P<id>[A-Z]+\d+)[ \t]*·[ \t]*"
    r"(?:(?P<num>[①-⑳㉑-㊿]|\d+)(?![^\s]))?[ \t]*(?P<title>.*)$"
)
_FIELD = re.compile(
    r"^[ \t]*[-*][ \t]+(?P<key>[^:：◇]{1,20}?)[ \t]*[:：][ \t]*(?P<value>.*)$"
)
_ITEM = re.compile(r"^[ \t]*[-*][ \t]+(?P<text>.*)$")
_KEY_MESSAGE = re.compile(r"^핵심[ \t]*메시지[ \t]*[:：][ \t]*(?P<value>.*)$")


@dataclass
class Paragraph:
    id: str
    number: str
    title: str
    fields: dict[str, str] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)

    def value(self, name: str) -> str:
        return self.fields.get(name, "")

    @property
    def open(self) -> bool:
        return bool(self.decisions) or any(
            _MARKER in v for v in (self.title, *self.fields.values())
        )


@dataclass
class Section:
    name: str
    notes: list[str] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


@dataclass
class OutlineView:
    key_message: str = ""
    sections: list[Section] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    @property
    def paragraph_count(self) -> int:
        return sum(len(s.paragraphs) for s in self.sections)

    @property
    def open_count(self) -> int:
        marks = len(self.decisions) + (_MARKER in self.key_message)
        for section in self.sections:
            marks += len(section.decisions)
            marks += sum(1 for p in section.paragraphs if p.open)
        return marks


def parse_outline_view(text: str) -> OutlineView:
    """Read *text* in the ``outline.md`` format. Lenient: lines it does not
    recognise are left out, and a comment before the first section (the
    template's guide) is not shown."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    view = OutlineView()
    section: Section | None = None
    paragraph: Paragraph | None = None
    last_key: str | None = None

    # Comments become notes of the section they sit in; their text is cut out
    # of the lines so a heading inside a comment is not read.
    pieces: list[tuple[str, str]] = []
    pos = 0
    for m in _COMMENT.finditer(text):
        pieces.append(("text", text[pos : m.start()]))
        pieces.append(("comment", m.group(1)))
        pos = m.end()
    pieces.append(("text", text[pos:]))

    for kind, chunk in pieces:
        if kind == "comment":
            note = "\n".join(line.strip() for line in chunk.strip().split("\n"))
            if section is not None and note:
                section.notes.append(note)
            continue
        for raw in chunk.split("\n"):
            line = raw.rstrip()
            if not line.strip():
                continue
            if m := _SECTION.match(line):
                section = Section(m.group("name"))
                view.sections.append(section)
                paragraph, last_key = None, None
                continue
            if m := _PARAGRAPH.match(line):
                if section is None:
                    section = Section("")
                    view.sections.append(section)
                paragraph = Paragraph(
                    m.group("id"), m.group("num") or "", m.group("title").strip()
                )
                section.paragraphs.append(paragraph)
                last_key = None
                continue
            if section is None:
                if m := _KEY_MESSAGE.match(line.strip()):
                    view.key_message = m.group("value").strip()
                elif _MARKER in line:
                    view.decisions.append(line.strip())
                continue
            target = paragraph
            if target is not None and (m := _FIELD.match(line)):
                last_key = m.group("key").strip()
                target.fields[last_key] = m.group("value").strip()
                continue
            item = _ITEM.match(line)
            stripped = item.group("text") if item else line.strip()
            if _MARKER in stripped:
                (target.decisions if target else section.decisions).append(stripped)
                continue
            if target is not None and last_key and not item:
                target.fields[last_key] += " " + stripped
    return view


# ------------------------------------------------------------------ rendering

_CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#5d6676;--line:#dde2ea;
--accent:#00897b;--accent-soft:#e0f2f1;--warn:#b26a00;--warn-soft:#fff4e0;
--bad:#c0392b;--bad-soft:#fdecea;--chip:#eef1f5;--shadow:0 1px 2px rgba(20,30,50,.06)}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--panel:#1a1f27;--ink:#e6e9ee;
--muted:#9aa3b2;--line:#2c333e;--accent:#4db6ac;--accent-soft:#16302d;--warn:#f0b357;
--warn-soft:#33270f;--bad:#f08070;--bad-soft:#3a1d1a;--chip:#232a34;--shadow:none}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 "Pretendard","Apple SD Gothic Neo","Malgun Gothic","Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}
header .kind{color:var(--accent);font-weight:700;letter-spacing:.04em;font-size:13px}
h1{font-size:26px;margin:4px 0 6px;line-height:1.3}
.meta{color:var(--muted);font-size:13px}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:700}
.badge.draft{background:var(--warn-soft);color:var(--warn)}
.badge.saved{background:var(--accent-soft);color:var(--accent)}
.key{margin:18px 0 8px;padding:14px 18px;background:var(--panel);border:1px solid var(--line);
border-left:4px solid var(--accent);border-radius:10px;box-shadow:var(--shadow)}
.key b{display:block;font-size:12px;color:var(--muted);font-weight:600;margin-bottom:2px}
.stats{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0}
.stats span{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:4px 10px;font-size:13px}
nav{display:flex;gap:6px;flex-wrap:wrap;margin:16px 0 4px}
nav a{color:var(--ink);text-decoration:none;background:var(--chip);border-radius:999px;padding:3px 12px;font-size:13px}
section.sec{margin-top:34px}
h2{font-size:20px;margin:0 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:14px;color:var(--muted);margin:22px 0 8px;font-weight:700}
.notes{background:var(--panel);border:1px dashed var(--line);border-radius:10px;padding:10px 14px;
color:var(--muted);font-size:13.5px;white-space:pre-wrap}
.flow{display:flex;flex-direction:column;align-items:stretch;max-width:720px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 16px;box-shadow:var(--shadow)}
.card.open{border-color:var(--warn);background:linear-gradient(0deg,var(--warn-soft),var(--warn-soft)),var(--panel)}
.card .top{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.pid{font:700 12px/1.4 "D2Coding",Consolas,monospace;background:var(--accent-soft);color:var(--accent);
border-radius:6px;padding:2px 7px;white-space:nowrap}
.card .title{font-weight:700}
.card .role{color:var(--muted);font-size:13px}
.card .claim{margin-top:4px}
.arrow{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px;padding:4px 0 4px 22px}
.arrow::before{content:"↓";font-size:18px;color:var(--accent);font-weight:700}
.tag{display:inline-block;font-size:11.5px;font-weight:700;border-radius:6px;padding:1px 7px}
.tag.open{background:var(--warn-soft);color:var(--warn)}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
border-radius:10px;overflow:hidden;box-shadow:var(--shadow)}
th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--line)}
th{font-size:12.5px;color:var(--muted);background:var(--chip);font-weight:700}
tr:last-child td{border-bottom:0}
td.no{white-space:nowrap;width:1%}
tr.open td{background:var(--warn-soft)}
.ev{display:flex;flex-direction:column;gap:4px}
.chip{display:inline-block;border-radius:6px;padding:2px 8px;font-size:12.5px;background:var(--chip)}
.chip code{font:12px "D2Coding",Consolas,monospace}
.chip.check{background:var(--warn-soft);color:var(--warn)}
.chip.miss{background:var(--bad-soft);color:var(--bad)}
.decide{margin:8px 0 0;padding:0;list-style:none}
.decide li{background:var(--warn-soft);color:var(--warn);border-radius:8px;padding:6px 10px;margin-top:6px;font-size:13.5px}
.sentences{margin-top:6px;font-size:13px;color:var(--muted)}
code{font-family:"D2Coding",Consolas,monospace;font-size:.92em}
footer{margin-top:40px;color:var(--muted);font-size:12px}
.chip{overflow-wrap:anywhere}
@media (max-width:720px){table,tbody,tr,td{display:block;width:100%}
tr.head{display:none}tr{border-bottom:1px solid var(--line)}td{border:0;padding:6px 12px}
td[data-label]::before{content:attr(data-label);display:block;font-size:11.5px;color:var(--muted);font-weight:700}
td.no{width:auto;padding-top:10px}}
"""


def _inline(text: str) -> str:
    """*text* escaped, with `code` and **bold** kept."""
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    return out.replace(_MARKER, f'<span class="tag open">{_MARKER}</span>')


def _evidence(value: str) -> str:
    if not value.strip():
        return ""
    chips = []
    for item in (part.strip() for part in value.split(";")):
        if not item:
            continue
        if "자료없음" in item or item.startswith("⚠"):
            chips.append(f'<span class="chip miss">{_inline(item)}</span>')
        elif item.startswith("확인 필요"):
            chips.append(f'<span class="chip check">{_inline(item)}</span>')
        else:
            path, sep, what = item.partition(" — ")
            body = f"<code>{html.escape(path)}</code>" + (
                f" — {_inline(what)}" if sep else ""
            )
            chips.append(f'<span class="chip">{body}</span>')
    return '<div class="ev">' + "".join(chips) + "</div>"


def _label(p: Paragraph) -> str:
    return f"{p.id} · {p.number}" if p.number else p.id


def _decisions(items: list[str]) -> str:
    if not items:
        return ""
    return (
        '<ul class="decide">'
        + "".join(f"<li>{_inline(item)}</li>" for item in items)
        + "</ul>"
    )


def _section_html(index: int, s: Section) -> str:
    parts = [
        f'<section class="sec" id="s{index}"><h2>{html.escape(s.name or "(섹션 없음)")}</h2>'
    ]
    if s.notes:
        parts.append(
            '<div class="notes">'
            + "\n\n".join(html.escape(n) for n in s.notes)
            + "</div>"
        )
    parts.append(_decisions(s.decisions))
    if not s.paragraphs:
        parts.append('<p class="meta">아직 단락이 없습니다.</p></section>')
        return "".join(parts)

    parts.append('<h3>흐름</h3><div class="flow">')
    for i, p in enumerate(s.paragraphs):
        role = p.value("역할")
        parts.append(
            f'<div class="card{" open" if p.open else ""}"><div class="top">'
            f'<span class="pid">{html.escape(_label(p))}</span>'
            f'<span class="title">{_inline(p.title)}</span>'
            + (f'<span class="role">{_inline(role)}</span>' if role else "")
            + (' <span class="tag open">결정 필요</span>' if p.open else "")
            + "</div>"
            + (
                f'<div class="claim">{_inline(p.value("핵심 내용"))}</div>'
                if p.value("핵심 내용")
                else ""
            )
            + (
                f'<div class="sentences">문장 구성: {_inline(p.value("문장 구성"))}</div>'
                if p.value("문장 구성")
                else ""
            )
            + _decisions(p.decisions)
            + "</div>"
        )
        if i < len(s.paragraphs) - 1:
            link = p.value("연결")
            parts.append(f'<div class="arrow">{_inline(link)}</div>')
    parts.append("</div>")

    parts.append(
        '<h3>간략 구성표</h3><table><tr class="head"><th>번호</th><th>역할</th>'
        "<th>핵심 내용</th></tr>"
    )
    for p in s.paragraphs:
        parts.append(
            f'<tr class="{"open" if p.open else ""}"><td class="no">'
            f'<span class="pid">{html.escape(_label(p))}</span></td>'
            f'<td data-label="역할">{_inline(p.value("역할"))}</td>'
            f'<td data-label="핵심 내용">{_inline(p.value("핵심 내용"))}</td></tr>'
        )
    parts.append("</table>")

    parts.append(
        '<h3>상세 설명표</h3><table><tr class="head"><th>번호</th><th>전달할 내용</th>'
        "<th>근거</th><th>연결</th></tr>"
    )
    for p in s.paragraphs:
        parts.append(
            f'<tr class="{"open" if p.open else ""}"><td class="no">'
            f'<span class="pid">{html.escape(_label(p))}</span></td>'
            f'<td data-label="전달할 내용">{_inline(p.value("전달할 내용"))}</td>'
            f'<td data-label="근거">{_evidence(p.value("근거"))}</td>'
            f'<td data-label="연결">{_inline(p.value("연결"))}</td></tr>'
        )
    parts.append("</table></section>")
    return "".join(parts)


def render_outline_html(
    view: OutlineView, *, title: str = "", saved: bool = True, source: str = ""
) -> str:
    """The page for *view*. *saved* tells a saved ``outline.md`` from a
    proposal not yet saved; *source* is shown in the header."""
    heading = title.strip() or "논문 구성 지도"
    status = (
        '<span class="badge saved">저장된 outline.md</span>'
        if saved
        else '<span class="badge draft">제안 — 아직 저장하지 않음</span>'
    )
    stats = (
        f"<span>섹션 {len(view.sections)}</span>"
        f"<span>단락 {view.paragraph_count}</span>"
        + (
            f'<span class="tag open">결정 필요 {view.open_count}</span>'
            if view.open_count
            else "<span>남은 결정 없음</span>"
        )
    )
    nav = "".join(
        f'<a href="#s{i}">{html.escape(s.name or "(섹션 없음)")}</a>'
        for i, s in enumerate(view.sections)
    )
    key = (
        f'<div class="key"><b>핵심 메시지</b>{_inline(view.key_message)}</div>'
        if view.key_message
        else ""
    )
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "".join(_section_html(i, s) for i, s in enumerate(view.sections))
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(heading)} — 구성 지도</title><style>{_CSS}</style></head>"
        f'<body><main><header><div class="kind">METASCI SCRIBE · 구성 지도</div>'
        f"<h1>{html.escape(heading)}</h1>"
        f'<div class="meta">{status} &nbsp;{html.escape(source)} · {when}</div>'
        f'<div class="stats">{stats}</div></header>{key}'
        + _decisions(view.decisions)
        + f"<nav>{nav}</nav>{body}"
        "<footer>이 페이지는 Metasci Scribe가 만든 미리보기입니다. "
        "고칠 때는 outline.md를 고치거나 대화로 요청하세요.</footer>"
        "</main></body></html>"
    )


def _paper_title(root: Path) -> str:
    try:
        text = (root / "program.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r'^title:\s*"?([^"\n#]*)"?', text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def show_outline(
    root: Path, outline: str | None = None, *, open_browser: bool = True
) -> tuple[Path, OutlineView]:
    """Draw *outline* (a proposal), or ``outline.md`` when it is None, as
    ``.metasci-scribe/view/outline.html`` and open it in the browser.

    Raises ``FileNotFoundError`` when there is no ``outline.md`` to draw.
    """
    root = Path(root)
    saved = outline is None
    if saved:
        outline = (root / "outline.md").read_text(encoding="utf-8", errors="replace")
    view = parse_outline_view(outline)
    page = render_outline_html(
        view,
        title=_paper_title(root),
        saved=saved,
        source="outline.md" if saved else "대화의 제안",
    )
    folder = Workspace(root).state_dir / "view"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "outline.html"
    fd, tmp = tempfile.mkstemp(prefix=".outline.", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(page)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(target.resolve().as_uri())
    return target, view
