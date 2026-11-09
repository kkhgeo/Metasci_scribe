# Report format

`scripts/check_review.py` parses this format, so keep the headings, field names, and markers exactly.

## Markers

- `«...»`: a verbatim, contiguous quotation from the current `draft.txt`. No ellipsis, no paraphrase. Describe text from an earlier draft or a baseline in words; never put it in `«»`.
- `〔Chen2024.md: ...〕`: a verbatim quotation from that Knowledge file.
- `### 1. [치명] 제목`: a finding heading. Number from 1 and order by severity: 치명, then 주요, then 경미.

## Finding fields

| Field | Rule |
|---|---|
| 유형 | 기여, 논리, 데이터, or 문헌 |
| 관련 주장 | C numbers from 핵심 주장과 근거, or 없음 |
| 원문 | at least one `«»` |
| 문제 | why it is a problem and how a reader would take it |
| 검증 | required when 문제 asserts a contradiction or impossibility, or that a conclusion cannot follow (the script looks for words such as 모순, 불가능, 성립하지 않, 주장할 수 없): premise → conclusion, and the counterexample you looked for |
| 다른 곳 확인 | required for 치명 and 주요: where else you looked and what you found |
| 해결 경로 | 서술 수정, 기존 결과 보고, 새 분석, or 새 자료 |
| 필요 이유 | required for 새 분석 and 새 자료: why existing results cannot answer it |
| 해결 기준 | one sentence that any fix must satisfy |

A 문헌 finding also needs a `〔〕` line.

## Sections

- Required: 연구 맥락 이해, 핵심 주장과 근거, 결론, 강점, 지적 사항, 확인 필요, 검토 범위. Write 없음 under 지적 사항 or 확인 필요 when there is nothing. 핵심 주장과 근거 may be 없음 only for text that makes no claim, such as a Methods passage.
- 핵심 주장과 근거: 수준 is 충분, 부분, 부족, or 판단 불가. Write 어긋난 점: 없음 when claim and evidence agree.
- 검토 범위: list each of the four types once as 검토함, 해당 없음, or 검토 못 함. The last two need a reason after `—`. A type with a finding must be 검토함, and 문헌 can be 검토함 only with Knowledge files.
- Optional: 덜어낼 것 (each item starts with [빼기], [옮기기], or [줄이기] and quotes the draft), 이전 지적 대조 (re-review only), 섹션 확인 (drafts with several sections only).
- 이전 지적 대조: one line per earlier finding, with the status 해결됨, 일부 해결, 남음, or 판단 불가 and the evidence after `—`.

## Template

```markdown
# 메타리뷰: <원고 또는 섹션 이름>

## 연구 맥락 이해
(이 글이 무엇을, 누구에게, 어떤 근거로 보이려 하는지 2~4문장. 비판하지 않는다)

참고 자료: (읽은 자료. 지식 파일은 파일 이름까지)
(이 세션이 쓰거나 고친 글이면 그렇게 적고 새 세션에서 다시 리뷰할 것을 권한다)

## 핵심 주장과 근거
- C1: «주장 원문»
  - 근거: 표 2, 그림 3b
  - 수준: 부분
  - 어긋난 점: 범위 — 한 유역의 자료로 전국 결론을 낸다

## 결론
(가장 중요한 문제와 전체 판단 2~3문장. 문제가 없으면 없다고 쓴다)

## 이전 지적 대조
- 이전 1: 해결됨 — «새 원고에서 인용»
- 이전 2: 남음 — 해당 문장이 그대로 있다

## 강점
- (구체적인 강점과, 수정할 때 지켜야 할 부분)

## 지적 사항

### 1. [치명] 문제를 요약한 제목
- 유형: 기여
- 관련 주장: C1
- 원문: «초고에서 그대로 인용»
- 문제: 왜 문제인지, 독자가 어떻게 받아들일지
- 검증: 전제 → 결론, 찾아본 반례
- 다른 곳 확인: 살펴본 곳과 결과
- 해결 경로: 서술 수정
- 해결 기준: 어떤 수정이든 충족해야 할 조건 한 문장

### 2. [주요] 문제를 요약한 제목
- 유형: 문헌
- 관련 주장: 없음
- 원문: «초고에서 그대로 인용»
- 근거: 〔Chen2024.md: 지식 파일에서 그대로 인용〕
- 문제: 왜 문제인지
- 다른 곳 확인: 살펴본 곳과 결과
- 해결 경로: 새 분석
- 필요 이유: 기존 결과로 답할 수 없는 이유
- 해결 기준: 어떤 수정이든 충족해야 할 조건 한 문장

## 덜어낼 것
- [줄이기] «초고에서 그대로 인용» — 무엇을 왜 덜어낼지

## 확인 필요
- X였다면 문제가 없고, 아니라면 Z가 된다. 판단에 필요한 정보: ...

## 섹션 확인
- 서론: «그 섹션에서 그대로 인용»

## 검토 범위
- 기여: 검토함
- 논리: 검토함
- 데이터: 해당 없음 — 수치와 그림이 없는 단락
- 문헌: 검토 못 함 — 지식 파일이 없음
```
