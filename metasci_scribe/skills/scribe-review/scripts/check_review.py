#!/usr/bin/env python3
"""Check a meta-review report against its draft, Knowledge files, and earlier round.

Run: python scripts/check_review.py draft.txt review.md [--knowledge DIR] [--previous REVIEW.md]

Exit 0 when every check passes, 1 when a check fails, 2 on unusable input.
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

DRAFT_QUOTE = re.compile(r"«([^»]+)»")
KNOWLEDGE_QUOTE = re.compile(r"〔([^:〕]+?):\s*([^〕]+)〕")
SECTION = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)
FINDING = re.compile(r"^### +(\d+)\. +\[(치명|주요|경미)\] *(.*)$", re.MULTILINE)
FIELD = re.compile(r"^\s*- ([^:\n]+?):\s*(.*)$")
CLAIM = re.compile(r"^- (C\d+):(.*)$")
CLAIM_FIELD = re.compile(r"^\s+- (근거|수준|어긋난 점):\s*(.*)$")
COVERAGE_LINE = re.compile(r"^- (기여|논리|데이터|문헌):\s*(검토함|해당 없음|검토 못 함)(.*)$")
PREVIOUS_LINE = re.compile(r"^- 이전 (\d+):\s*(해결됨|일부 해결|남음|판단 불가)(.*)$")
TRIM_LINE = re.compile(r"^- \[(빼기|옮기기|줄이기)\]")
QUOTE_MARKS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
LINE_HYPHEN = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")

TYPES = ("기여", "논리", "데이터", "문헌")
SEVERITY_RANK = {"치명": 0, "주요": 1, "경미": 2}
PATHS = ("서술 수정", "기존 결과 보고", "새 분석", "새 자료")
SUPPORT = ("충분", "부분", "부족", "판단 불가")
REQUIRED = ("연구 맥락 이해", "핵심 주장과 근거", "결론", "강점", "지적 사항",
            "확인 필요", "검토 범위")
# A problem worded as a contradiction or an impossibility must carry a 검증 line.
STRONG = re.compile(
    r"모순|불가능|성립하지 않|성립할 수 없|양립할 수 없|있을 수 없|뒷받침할 수 없"
    r"|주장할 수 없|결론지을 수 없|contradict|impossible|cannot|incompatible",
    re.IGNORECASE)


def normalize(text):
    """Ignore differences that line wrapping and typography introduce."""
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_MARKS)
    return re.sub(r"\s+", " ", text).strip()


def draft_forms(draft):
    """The draft as written, and with line-break hyphens joined or kept."""
    return {normalize(draft),
            normalize(LINE_HYPHEN.sub(r"\1\2", draft)),
            normalize(LINE_HYPHEN.sub(r"\1-\2", draft))}


def read(path):
    try:
        return Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError) as error:
        print(f"ERROR cannot read {path}: {error}")
        sys.exit(2)


def reason(rest):
    """Text after the dash that follows a status, or an empty string."""
    return rest.strip().lstrip("—–-:").strip()


def sections(report):
    """Map each ## heading to its body; the first heading wins on a repeat."""
    found, matches = {}, list(SECTION.finditer(report))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report)
        found.setdefault(match.group(1).strip(), report[match.end():end].strip())
    return found


def section(found, name):
    """Body of the section whose heading starts with name, or None."""
    return next((body for title, body in found.items() if title.startswith(name)), None)


def fields(block):
    """Read '- key: value' lines; other non-empty lines continue the last value."""
    values, key = {}, None
    for line in block.splitlines()[1:]:
        match = FIELD.match(line)
        if match:
            key = match.group(1).strip()
            values[key] = match.group(2).strip()
        elif key and line.strip():
            values[key] += " " + line.strip()
    return values


def starts_with_any(value, options):
    return next((option for option in options if value.startswith(option)), None)


def check_claims(body):
    claims, current, errors = {}, None, []
    for line in body.splitlines():
        match = CLAIM.match(line)
        if match:
            current = match.group(1)
            if current in claims:
                errors.append(f"주장 번호 {current}가 두 번 있습니다")
            claims[current] = {"head": match.group(2)}
            continue
        sub = CLAIM_FIELD.match(line)
        if sub and current:
            claims[current][sub.group(1)] = sub.group(2).strip()
    if not claims and not body.startswith("없음"):
        errors.append("핵심 주장과 근거에 C 번호 주장이 없습니다. 주장이 없는 글이면 없음으로 씁니다")
    for cid, data in claims.items():
        if not DRAFT_QUOTE.search(data["head"]):
            errors.append(f"{cid}: 주장을 «»로 인용하지 않았습니다")
        if not data.get("근거"):
            errors.append(f"{cid}: 근거 줄이 없습니다")
        if not starts_with_any(data.get("수준", ""), SUPPORT):
            errors.append(f"{cid}: 수준은 {', '.join(SUPPORT)} 중 하나여야 합니다")
        if "어긋난 점" not in data:
            errors.append(f"{cid}: 어긋난 점 줄이 없습니다. 없으면 없음으로 씁니다")
    return set(claims), errors


def check_finding(number, severity, block, claim_ids):
    label, values, errors = f"지적 {number}", fields(block), []
    kind = starts_with_any(values.get("유형", ""), TYPES)
    if not kind:
        errors.append(f"{label}: 유형은 {', '.join(TYPES)} 중 하나여야 합니다")
    related = values.get("관련 주장", "")
    named = re.findall(r"C\d+", related)
    if not related.startswith("없음") and not named:
        errors.append(f"{label}: 관련 주장에 C 번호나 없음을 씁니다")
    errors += [f"{label}: 관련 주장 {cid}가 핵심 주장과 근거에 없습니다"
               for cid in named if cid not in claim_ids]
    if not DRAFT_QUOTE.search(values.get("원문", "")):
        errors.append(f"{label}: 원문에 «초고 인용»이 없습니다")
    problem = values.get("문제", "")
    if not problem:
        errors.append(f"{label}: 문제 줄이 없습니다")
    elif STRONG.search(problem) and not values.get("검증"):
        errors.append(f"{label}: 문제가 모순이나 불가능을 단정하므로 검증 줄"
                      "(전제 → 결론, 찾아본 반례)이 필요합니다")
    if severity in ("치명", "주요") and not values.get("다른 곳 확인"):
        errors.append(f"{label}: {severity} 지적에는 다른 곳 확인 줄이 필요합니다")
    path = starts_with_any(values.get("해결 경로", ""), PATHS)
    if not path:
        errors.append(f"{label}: 해결 경로는 {', '.join(PATHS)} 중 하나여야 합니다")
    elif path in ("새 분석", "새 자료") and not values.get("필요 이유"):
        errors.append(f"{label}: {path}에는 기존 결과로 답할 수 없는 필요 이유가 필요합니다")
    if not values.get("해결 기준"):
        errors.append(f"{label}: 해결 기준 줄이 없습니다")
    if kind == "문헌" and not KNOWLEDGE_QUOTE.search(block):
        errors.append(f"{label}: 문헌 지적에 〔지식 파일 인용〕이 없습니다")
    return kind, errors


def check_findings(body, claim_ids):
    matches = list(FINDING.finditer(body))
    if not matches:
        ok = body.startswith("없음")
        return set(), 0, [] if ok else ["지적 사항에 지적이 없습니다. 없으면 없음으로 씁니다"]
    kinds, errors, worst = set(), [], 0
    for expected, match in enumerate(matches, 1):
        number, severity = int(match.group(1)), match.group(2)
        end = matches[expected].start() if expected < len(matches) else len(body)
        if number != expected:
            errors.append(f"지적 번호가 {expected}이어야 하는데 {number}입니다")
        if SEVERITY_RANK[severity] < worst:
            errors.append(f"지적 {number}: 심각도 순서(치명 → 주요 → 경미)를 어깁니다")
        worst = max(worst, SEVERITY_RANK[severity])
        kind, found = check_finding(number, severity, body[match.start():end], claim_ids)
        kinds.add(kind)
        errors += found
    return kinds - {None}, len(matches), errors


def check_coverage(body, kinds, has_knowledge):
    status, errors = {}, []
    for line in body.splitlines():
        match = COVERAGE_LINE.match(line.strip())
        if not match:
            continue
        kind, state, rest = match.groups()
        if kind in status:
            errors.append(f"검토 범위에 {kind}가 두 번 있습니다")
        status[kind] = state
        if state != "검토함" and not reason(rest):
            errors.append(f"검토 범위: {kind}의 {state} 뒤에 — 와 이유를 씁니다")
    errors += [f"검토 범위에 {kind} 줄이 없습니다" for kind in TYPES if kind not in status]
    errors += [f"검토 범위: {kind} 지적이 있는데 {status[kind]}로 표시했습니다"
               for kind in sorted(kinds) if status.get(kind, "검토함") != "검토함"]
    if status.get("문헌") == "검토함" and not has_knowledge:
        errors.append("검토 범위: --knowledge 없이 문헌을 검토함으로 표시했습니다")
    return errors


def check_trim(body):
    errors = []
    for line in body.splitlines():
        if not line.startswith("- "):
            continue
        if not TRIM_LINE.match(line):
            errors.append(f"덜어낼 것: 항목은 [빼기], [옮기기], [줄이기]로 시작합니다: {line[:40]}")
        elif not DRAFT_QUOTE.search(line):
            errors.append(f"덜어낼 것: 항목에 «초고 인용»이 없습니다: {line[:40]}")
    return errors


def check_previous(body, previous):
    earlier = [int(m.group(1)) for m in FINDING.finditer(previous)]
    if body is None:
        return ["--previous가 있는데 이전 지적 대조 섹션이 없습니다"] if earlier else []
    seen, errors = set(), []
    for line in body.splitlines():
        line = line.strip()
        match = PREVIOUS_LINE.match(line)
        if not match:
            if line.startswith("- 이전"):
                errors.append("이전 지적 대조: 상태는 해결됨, 일부 해결, 남음, 판단 불가 중 "
                              f"하나여야 합니다: {line[:40]}")
            continue
        number = int(match.group(1))
        if number in seen:
            errors.append(f"이전 지적 대조: 이전 {number}이 두 번 있습니다")
        seen.add(number)
        if number not in earlier:
            errors.append(f"이전 지적 대조: 이전 리뷰에 {number}번 지적이 없습니다")
        if not reason(match.group(3)):
            errors.append(f"이전 지적 대조: 이전 {number}의 상태 뒤에 — 와 근거를 씁니다")
    errors += [f"이전 지적 대조: 이전 {n}번 지적의 상태가 없습니다" for n in earlier if n not in seen]
    return errors


def find_knowledge_file(root, name):
    """Return the file called name under root, or None if absent."""
    return next((p for p in sorted(Path(root).rglob("*")) if p.name == name), None)


def check_quotes(report, draft, root):
    forms, errors = draft_forms(draft), []
    errors += [f"초고에 없음: «{quote}»" for quote in DRAFT_QUOTE.findall(report)
               if not any(normalize(quote) in form for form in forms)]
    for name, quote in KNOWLEDGE_QUOTE.findall(report):
        name = name.strip()
        if root is None:
            errors.append(f"--knowledge 없이 지식 파일을 인용했습니다: {name}")
            continue
        path = find_knowledge_file(root, name)
        if path is None:
            errors.append(f"지식 파일 없음: {name}")
        elif normalize(quote) not in normalize(read(path)):
            errors.append(f"{name}에 없음: 〔{quote}〕")
    return errors


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft")
    parser.add_argument("report")
    parser.add_argument("--knowledge")
    parser.add_argument("--previous")
    args = parser.parse_args()
    if args.knowledge and not Path(args.knowledge).is_dir():
        print(f"ERROR knowledge folder not found: {args.knowledge}")
        sys.exit(2)
    draft, report = read(args.draft), read(args.report)
    previous = read(args.previous) if args.previous else None

    found, errors = sections(report), []
    for name in REQUIRED:
        body = section(found, name)
        if body is None:
            errors.append(f"{name} 섹션이 없습니다")
        elif not body:
            errors.append(f"{name} 섹션이 비어 있습니다")

    claim_ids, claim_errors = check_claims(section(found, "핵심 주장과 근거") or "없음")
    kinds, count, finding_errors = check_findings(section(found, "지적 사항") or "없음", claim_ids)
    errors += claim_errors + finding_errors
    strengths = section(found, "강점")
    if strengths and not any(line.startswith("- ") for line in strengths.splitlines()):
        errors.append("강점에 - 로 시작하는 항목이 없습니다")
    if section(found, "검토 범위"):
        errors += check_coverage(section(found, "검토 범위"), kinds, bool(args.knowledge))
    trim = section(found, "덜어낼 것")
    if trim and not trim.startswith("없음"):
        errors += check_trim(trim)
    compared = section(found, "이전 지적 대조")
    if previous is not None:
        errors += check_previous(compared, previous)
    elif compared is not None:
        errors.append("--previous 없이 이전 지적 대조 섹션이 있습니다")
    errors += check_quotes(report, draft, args.knowledge)

    if errors:
        print(f"FAIL {len(errors)}건")
        print("\n".join(f"- {error}" for error in errors))
        sys.exit(1)
    print(f"PASS 지적 {count}건, 주장 {len(claim_ids)}개, "
          f"초고 인용 {len(DRAFT_QUOTE.findall(report))}건, "
          f"지식 파일 인용 {len(KNOWLEDGE_QUOTE.findall(report))}건"
          + (", 이전 지적 대조 완료" if previous is not None else ""))


if __name__ == "__main__":
    main()
