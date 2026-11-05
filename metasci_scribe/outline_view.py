"""An outline drawn as an HTML page the author opens in a browser.

``outline.md`` (or a proposal written in its format) is read into sections and
paragraphs with their fields, and drawn as one self-contained page: the key
message, and for each section a flow of paragraph cards, the short table
(번호 · 역할 · 핵심 내용) and the detail table (전달할 내용 · 근거 · 연결). Open
choices (◇), evidence still to check and missing evidence are marked.

The app writes the page itself, under ``.metasci-scribe/view/``, so nothing in
the paper folder changes. The page has its own styles and no scripts. It is
set in Paperlogy: the installed font when there is one, otherwise the web font
from jsDelivr (the only thing the page fetches; no outline text leaves).
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

_CDN = "https://cdn.jsdelivr.net/gh/fonts-archive/Paperlogy"
_WEIGHTS = {
    300: "3Light",
    400: "4Regular",
    500: "5Medium",
    600: "6SemiBold",
    700: "7Bold",
}

# Paperlogy: the installed font first, the web font only where it is missing.
_FONTS = "".join(
    "@font-face{font-family:'Paperlogy';font-style:normal;font-display:swap;"
    f"font-weight:{weight};src:local('Paperlogy {name[0]} {name[1:]}'),"
    f"local('Paperlogy-{name}'),url('{_CDN}/Paperlogy-{name}.woff2') format('woff2')}}"
    for weight, name in _WEIGHTS.items()
)

_CSS = """
:root{--bg:#fff;--ink:#16181d;--muted:#6b7079;--faint:#a3a8b0;--rule:#e7e9ec;
--accent:#00897b;--open:#b26a00;--miss:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--ink:#e8eaed;--muted:#9aa0a8;
--faint:#5f656e;--rule:#2a2e34;--accent:#4db6ac;--open:#e0a54a;--miss:#ef8a7e}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:400 16px/1.75 'Paperlogy','Malgun Gothic','Apple SD Gothic Neo',sans-serif;
-webkit-font-smoothing:antialiased;word-break:keep-all;overflow-wrap:anywhere}
main{max-width:860px;margin:0 auto;padding:64px 20px 96px}
.meta{color:var(--muted);font-size:13px;letter-spacing:.01em}
h1{font-weight:700;font-size:34px;line-height:1.3;margin:10px 0 0;letter-spacing:-.01em}
.key{font-weight:400;font-size:20px;line-height:1.6;margin:22px 0 0;color:var(--ink)}
.key::before{content:"핵심 메시지";display:block;font-size:12px;color:var(--muted);margin-bottom:4px}
.counts{color:var(--muted);font-size:13px;margin-top:22px}
.open-text{color:var(--open)}
section{margin-top:72px}
h2{font-weight:600;font-size:24px;margin:0 0 6px;padding-top:20px;border-top:1px solid var(--ink)}
h3{font-weight:500;font-size:13px;color:var(--muted);margin:36px 0 12px;letter-spacing:.02em}
.notes{color:var(--muted);font-size:14px;white-space:pre-wrap;margin:10px 0 0}
.decide{list-style:none;margin:10px 0 0;padding:0}
.decide li{color:var(--open);font-size:14px}
ol.flow{list-style:none;margin:0;padding:0}
ol.flow>li{position:relative;padding:0 0 26px 26px;border-left:1px solid var(--rule);margin-left:5px}
ol.flow>li:last-child{border-left-color:transparent;padding-bottom:0}
ol.flow>li::before{content:"";position:absolute;left:-5px;top:9px;width:9px;height:9px;
border-radius:50%;background:var(--bg);border:1.5px solid var(--accent)}
ol.flow>li.open::before{border-color:var(--open)}
.pid{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums;margin-right:8px}
.ptitle{font-weight:600}
.role{color:var(--muted);font-size:13px;margin-left:8px}
.claim{margin-top:2px}
.link{color:var(--muted);font-size:13px;margin-top:6px}
.link::before{content:"→ ";color:var(--faint)}
.sentences{color:var(--muted);font-size:13px;margin-top:4px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;vertical-align:top;padding:12px 14px 12px 0;border-bottom:1px solid var(--rule)}
th{font-weight:500;font-size:12px;color:var(--muted);border-bottom-color:var(--ink);padding-top:0}
td.no{white-space:nowrap;width:1%;color:var(--muted);font-size:13px}
tr.open td.no{color:var(--open)}
.ev div{font-size:13.5px}
.path{color:var(--muted)}
.check{color:var(--open)}
.miss{color:var(--miss)}
.mark{color:var(--open)}
code{font-family:'D2Coding',Consolas,monospace;font-size:.9em}
footer{margin-top:80px;color:var(--faint);font-size:12px}
@media (max-width:720px){main{padding:40px 18px 72px}h1{font-size:28px}.key{font-size:18px}
table,tbody,tr,td{display:block;width:100%}tr.head{display:none}
tr{border-bottom:1px solid var(--rule);padding:10px 0}td{border:0;padding:3px 0}
td[data-label]::before{content:attr(data-label);display:block;font-size:12px;color:var(--muted)}}
"""


def _inline(text: str) -> str:
    """*text* escaped, with `code` and **bold** kept."""
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    return out.replace(_MARKER, f'<span class="mark">{_MARKER}</span>')


def _evidence(value: str) -> str:
    if not value.strip():
        return ""
    rows = []
    for item in (part.strip() for part in value.split(";")):
        if not item:
            continue
        if "자료없음" in item or item.startswith("⚠"):
            rows.append(f'<div class="miss">{_inline(item)}</div>')
        elif item.startswith("확인 필요"):
            rows.append(f'<div class="check">{_inline(item)}</div>')
        else:
            path, sep, what = item.partition(" — ")
            what_html = f" — {_inline(what)}" if sep else ""
            rows.append(
                f'<div><span class="path">{html.escape(path)}</span>{what_html}</div>'
            )
    return '<div class="ev">' + "".join(rows) + "</div>"


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


def _flow_item(p: Paragraph, last: bool) -> str:
    role, claim, link = p.value("역할"), p.value("핵심 내용"), p.value("연결")
    sentences = p.value("문장 구성")
    return (
        f'<li class="{"open" if p.open else ""}">'
        f'<span class="pid">{html.escape(_label(p))}</span>'
        f'<span class="ptitle">{_inline(p.title)}</span>'
        + (f'<span class="role">{_inline(role)}</span>' if role else "")
        + (f'<div class="claim">{_inline(claim)}</div>' if claim else "")
        + (
            f'<div class="sentences">문장 구성 {_inline(sentences)}</div>'
            if sentences
            else ""
        )
        + _decisions(p.decisions)
        + (f'<div class="link">{_inline(link)}</div>' if link and not last else "")
        + "</li>"
    )


def _section_html(s: Section) -> str:
    parts = [f"<section><h2>{html.escape(s.name or '(섹션 없음)')}</h2>"]
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

    count = len(s.paragraphs)
    parts.append('<h3>흐름</h3><ol class="flow">')
    parts.extend(_flow_item(p, i == count - 1) for i, p in enumerate(s.paragraphs))
    parts.append("</ol>")

    parts.append(
        '<h3>간략 구성표</h3><table><tr class="head"><th>번호</th><th>역할</th>'
        "<th>핵심 내용</th></tr>"
    )
    for p in s.paragraphs:
        parts.append(
            f'<tr class="{"open" if p.open else ""}">'
            f'<td class="no">{html.escape(_label(p))}</td>'
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
            f'<tr class="{"open" if p.open else ""}">'
            f'<td class="no">{html.escape(_label(p))}</td>'
            f'<td data-label="전달할 내용">{_inline(p.value("전달할 내용"))}</td>'
            f'<td data-label="근거">{_evidence(p.value("근거"))}</td>'
            f'<td data-label="연결">{_inline(p.value("연결"))}</td></tr>'
        )
    parts.append("</table></section>")
    return "".join(parts)


def render_outline_html(
    view: OutlineView, *, title: str = "", saved: bool = True
) -> str:
    """The page for *view*. *saved* tells a saved ``outline.md`` from a
    proposal not yet saved."""
    heading = title.strip() or "논문 구성 지도"
    status = "저장된 outline.md" if saved else "제안 · 아직 저장하지 않음"
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = " · ".join(("구성 지도", status, when))
    opens = (
        f' · <span class="open-text">◇ 결정 {view.open_count}</span>'
        if view.open_count
        else ""
    )
    counts = f"섹션 {len(view.sections)} · 단락 {view.paragraph_count}{opens}"
    key = f'<p class="key">{_inline(view.key_message)}</p>' if view.key_message else ""
    body = "".join(_section_html(s) for s in view.sections)
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(heading)} — 구성 지도</title>"
        f"<style>{_FONTS}{_CSS}</style></head>"
        f'<body><main><header><div class="meta">{html.escape(meta)}</div>'
        f"<h1>{html.escape(heading)}</h1>{key}"
        f'<div class="counts">{counts}</div>'
        + _decisions(view.decisions)
        + f"</header>{body}"
        "<footer>Metasci Scribe가 만든 미리보기입니다. "
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
