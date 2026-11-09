# Examples

Synthetic examples of finished findings. Match their structure and precision, not their wording or subject. The study, numbers, and sentences are invented.

## 1. Aim, design, and conclusion do not match (치명)

The draft samples 23 wells once and concludes that nitrate is rising.

Weak, do not write this:

> 결론이 과장되었다. 추세를 주장할 근거가 없다.

It has no location, no reason a reader can check, and no criterion for a fix.

Strong:

```markdown
### 1. [치명] 한 차례 채수로 증가 추세를 결론 내림
- 유형: 기여
- 관련 주장: C1
- 원문: «2024년 8월 한 차례 23개 관정에서 시료를 채취하였다»
- 문제: 결론은 «질산성 질소 농도가 증가하는 추세를 보였다»라고 쓰지만, 시료는 한 시점에만 채취되어 시간에 따른 변화를 볼 수 없다. 이 설계가 보여 줄 수 있는 것은 관정 사이의 차이까지다. 독자는 이 문장을 시계열 결과로 읽는다.
- 검증: 전제(채수 1회, 과거 자료 없음) → 결론(시간 추세를 판단할 근거가 없음). 과거 측정값과 비교했다면 반례가 되므로 찾아보았으나 본문, 표, 부록에 없다.
- 다른 곳 확인: 방법 2.2, 표 1, 고찰 4.1을 확인했다. 반복 측정이나 과거 자료는 없다.
- 해결 경로: 서술 수정
- 해결 기준: 초록과 결론이 한 시점의 공간 분포만 말하고, 추세는 후속 연구 과제로 표시된다.
```

The finding comes first because the trend claim is the paper's headline. The weak sampling design is not filed as a separate finding; it is the cause of this one.

## 2. Abstract and table disagree (주요)

```markdown
### 2. [주요] 초록의 기준 초과 비율이 표 2와 다름
- 유형: 데이터
- 관련 주장: C2
- 원문: «조사 관정의 35%가 먹는물 수질 기준을 초과하였다»
- 문제: 표 2는 초과 관정을 «7 (30.4)»로 적어 23개 중 30.4%다. 두 수치 중 하나가 틀렸거나 분모가 다르다. 독자는 어느 값을 인용할지 알 수 없다.
- 다른 곳 확인: 결과 3.1은 표 2와 같은 30.4%를 쓰고, 분모를 바꾼다는 설명은 없다.
- 해결 경로: 서술 수정
- 해결 기준: 초록, 결과 3.1, 표 2의 초과 비율과 분모가 같다.
```

## 3. A strong candidate that did not survive its check

Candidate: the draft reports «중앙값 4.2 mg/L(사분위 범위 2.1~6.3 mg/L)» and «최댓값 48.0 mg/L», and the two look contradictory.

Check: premise (the interquartile range is 2.1 to 6.3) → conclusion (48.0 cannot occur). An interquartile range says nothing about the top quarter, so one well at 48.0 is a valid counterexample. The candidate is not a finding. Whether the value is real is still open, so it goes under 확인 필요 as a condition:

```markdown
## 확인 필요
- 최댓값 48.0 mg/L이 실제 측정값이라면 문제가 없고, 단위 환산이나 입력 오류라면 결과 3.2의 최댓값 서술과 그림 2의 축 범위가 바뀐다. 판단에 필요한 정보: 해당 관정의 원자료와 측정 단위.
```
