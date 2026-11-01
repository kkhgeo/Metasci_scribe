"""Reads one PDF and writes its text; run as a child process by ``knowledge``.

A PDF is untrusted input, and a damaged one can make the reader loop or use
all the memory. The parent starts this file with ``python -I`` (the folder it
runs in is not on the import path), gives it a time limit and kills it when
the limit passes. A timer here stops it too, a little later, for the case
where the parent has quit and cannot kill it.

Usage: ``_pdf_worker.py PDF OUT MAX_PAGES LABEL [SECONDS]``. After SECONDS it
reports ``timeout`` and exits. The last line printed is one JSON object:

* ``{"status": "ok", "pages": N}``: the text of the pages is in OUT;
* ``{"status": "scanned", "pages": N}``: the pages hold no text;
* ``{"status": "failed", "error": "..."}``: unreadable, encrypted or too long;
* ``{"status": "missing"}``: the file is gone;
* ``{"status": "unreadable", "error": "..."}``: the file could not be opened
  now (locked, no permission); the parent tries again next time.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

# Control characters but tab, newline and carriage return.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _short(exc: BaseException) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= 160 else text[:157] + "..."


def _failed(reason: str) -> dict:
    return {"status": "failed", "error": reason}


def extract(pdf: Path, out: Path, max_pages: int, label: str) -> dict:
    """Write the text of *pdf* to *out*, one ``[p. N]`` block per page."""
    try:
        from pypdf import PdfReader

        with open(pdf, "rb") as handle:
            reader = PdfReader(handle)
            if reader.is_encrypted:
                try:
                    opened = reader.decrypt("")
                except Exception as exc:
                    return _failed(f"encrypted and could not be opened ({_short(exc)})")
                if not opened:
                    return _failed("encrypted, needs a password")
            count = len(reader.pages)
            if count > max_pages:
                return _failed(f"{count} pages, the limit is {max_pages}")
            if count == 0:
                return _failed("no pages")
            blocks: list[str] = []
            has_text = False
            unreadable = 0
            first_error: BaseException | None = None
            for number in range(count):
                try:
                    text = reader.pages[number].extract_text() or ""
                except Exception as exc:
                    unreadable += 1
                    first_error = first_error or exc
                    text = ""
                text = _CONTROL.sub("", text).strip()
                has_text = has_text or bool(text)
                blocks.append(f"[p. {number + 1}]\n{text}\n")
    except FileNotFoundError:
        return {"status": "missing"}
    except OSError as exc:
        return {"status": "unreadable", "error": _short(exc)}
    except Exception as exc:
        return _failed(f"not readable as a PDF ({_short(exc)})")

    if not has_text:
        if unreadable and first_error is not None:
            return _failed(
                f"text unreadable on {unreadable} of {count} pages "
                f"({_short(first_error)})"
            )
        return {"status": "scanned", "pages": count}

    header = f"Source: {label} ({count} pages; [p. N] is the PDF page number)\n\n"
    # Lone surrogates in a name or a text cannot be encoded; they become "?".
    with open(out, "w", encoding="utf-8", errors="replace", newline="\n") as f:
        f.write(header)
        f.write("\n".join(blocks))
    return {"status": "ok", "pages": count}


def _give_up() -> None:
    """The parent kills a worker that overruns; this is for the one it cannot
    reach (the parent quit first)."""
    print(json.dumps({"status": "timeout"}), flush=True)
    os._exit(0)


def main(argv: list[str]) -> int:
    if len(argv) > 5:
        watchdog = threading.Timer(float(argv[5]), _give_up)
        watchdog.daemon = True
        watchdog.start()
    try:
        pdf, out, max_pages, label = argv[1:5]
        result = extract(Path(pdf), Path(out), int(max_pages), label)
    except Exception as exc:
        result = _failed(f"not readable ({_short(exc)})")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
