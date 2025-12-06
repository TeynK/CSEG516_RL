프로젝트 계획서: 스플랜더를 통한 On-Policy vs Off-Policy 리스크 성향 및 수렴성 분석
1. 연구 개요 (Overview)
본 연구는 강화학습의 두 가지 주요 접근 방식인 On-Policy(PPO)와 Off-Policy(DQN) 알고리즘이 불확실성이 존재하는 환경(스플랜더)에서 보이는 **리스크 감수 성향(Risk Appetite)**의 차이를 분석한다.
특히, Vanilla DQN이 가지는 구조적 결함인 **과대평가 편향(Overestimation Bias)**을 '리스크 추구 성향'으로 재해석하여, 이것이 실제 게임 전략에서 공격적인 행동으로 발현되는지를 통계적으로 검증한다. 또한, 자기 학습(Self-Play)이 이러한 알고리즘 고유의 성향(Bias)을 중화시키는지 확인한다.
2. 이론적 배경 (Theoretical Background)
2.1. Vanilla DQN: 낙관적 편향과 리스크 추구
핵심 기제: $Q$-Learning의 Target 계산식에 포함된 max 연산자는 노이즈(불확실성)의 양수 부분을 선택하는 경향이 있다.$$E[\max_a Q(s, a)] \ge \max_a E[Q(s, a)]$$
리스크와의 연결: 스플랜더에서 성공 확률은 낮지만 보상이 큰(High Variance) 전략(예: 3티어 킵)은 $Q$값의 분산($\sigma$)을 키운다. Thrun & Schwartz(1993)에 따르면 과대평가 편향은 분산에 비례하므로, DQN은 객관적 확률보다 해당 전략을 고평가(Overestimation)하여 선택하게 된다. 이를 **리스크 추구(Risk-Seeking)**로 정의한다.
2.2. PPO: 평균 회귀와 리스크 회피
핵심 기제: On-Policy 방식은 현재 정책 $\pi$ 하에서의 기대 보상(Mean Expectation)을 최적화한다.$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$
리스크와의 연결: 대박 가능성이 있어도 실패 확률이 높아 평균 기대값이 낮다면, PPO는 해당 행동을 선택하지 않는다. Trust Region(Clip)은 정책의 급격한 변화를 억제하므로, 안정적이고 분산이 낮은(Low Variance) 전략을 선호하게 된다. 이를 **리스크 회피/중립(Risk-Averse/Neutral)**으로 정의한다.
3. 실험 환경 (Environment Setup)
게임: Splendor (2인용)
State Space: 보드 상태, 플레이어 토큰/카드 보유 현황 (Masking 적용)
Action Space: 토큰 가져오기, 카드 구매, 카드 킵 (Masking 적용)
Reward Function:
승리: $+1$
패배: $-1$
설명: 보상 스케일링($\pm 100$)은 학습 불안정성만 야기할 뿐 성향 차이의 본질적 원인이 아니므로, 표준적인 $\pm 1$ 보상을 사용한다. 대신 **승률 예측의 편향(Bias)**을 측정하여 리스크 성향을 분석한다.
알고리즘 설정 (Critical):
DQN: Vanilla DQN (No Double DQN, No Dueling). Replay Buffer 사용. Target Update 주기를 길게 설정하여 Bias 유지.
PPO: Standard PPO (Clip ratio 0.2). GAE 사용.
Common: Action Masking 필수 적용. 유효하지 않은 행동 공간을 사전에 차단하여 학습 복잡도를 낮추고 수렴성을 확보함.
4. 가설 (Hypotheses)
가설 1 (Risk Propensity)
동일한 횟수만큼 학습된 모델을 비교했을 때, Vanilla DQN은 PPO보다 '불확실성이 높은 행동(High Variance Action)'을 유의미하게 많이 수행할 것이다.
지표: 3티어 카드 킵/구매 비율, 예측 승률과 실제 승률의 괴리(Optimism Bias).
가설 2 (Convergence via Self-Play)
충분한 시간차를 두고 자기 학습(Self-Play)을 수행하면, 두 알고리즘 모두 상대의 전략에 대응하며 **내쉬 균형(Nash Equilibrium)**에 도달하여, 초기의 성향 차이(공격 vs 수비)가 감소하고 전략적 유사도가 증가할 것이다.
5. 실험 절차 (Experiment Sequence)
Phase 1: 기준점 마련 (Pre-training)
목적: 무작위 행동(Random Walk)에 의한 노이즈 제거 및 기초 전략 수립.
방법: 룰 기반 휴리스틱 봇(Heuristic Bot)을 상대로 두 알고리즘을 승률 50% 수준까지 학습시킴.
Phase 2: 리스크 성향 분석 (Comparative Analysis)
방법: Phase 1에서 학습된 PPO와 DQN을 상호 대결(혹은 고정된 휴리스틱 봇과 대결)시키며 로그 수집. ($\epsilon=0$ 적용)
측정 지표 (Metrics):
행동 분포 (Behavioral Metric):
전체 구매 카드 중 3티어 카드의 비율.
카드 'Keep' 횟수 (미래를 위한 불확실한 투자).
턴 당 평균 칩 보유량 (유동성 리스크 감수 여부).
낙관성 편향 (Optimism Bias - 핵심 지표):
특정 상태 $S$에서 선택한 행동 $A$에 대해 알고리즘이 예측한 승률($Q_{pred}$ or $V_{pred}$)과, 해당 시점에서 몬테카를로 시뮬레이션(100회)으로 구한 실제 승률($P_{actual}$)의 차이.
$$Bias = \frac{1}{N} \sum (V_{predicted} - V_{actual})$$
DQN은 $Bias \gg 0$, PPO는 $Bias \approx 0$ 예상.
Phase 3: 자기 학습 및 수렴성 검증 (Self-Play)
방법: 각 알고리즘이 자신의 과거 버전(Frozen Copy)과 대결하며 추가 학습 진행.
분석:
Phase 2와 동일한 지표를 측정하여, 학습이 진행됨에 따라 DQN의 Optimism Bias가 감소하는지, PPO의 전략이 공격적으로 변화하는지(혹은 그 반대) 확인.
두 모델 간의 행동 분포 거리(KL-Divergence 등) 측정.
6. 예상 결과 및 해석 가이드
| 구분 | PPO (On-Policy) | Vanilla DQN (Off-Policy) | 해석 |
| 3티어 카드 선호도 | 낮음 | 높음 | DQN은 낮은 확률의 '대박'을 과대평가함 |
| 카드 킵(Keep) 빈도 | 낮음 (확실할 때만) | 높음 (견제/선점) | 불확실성을 기회로 인식하는 공격성 |
| 낙관성 지표(Bias) | 0에 근접 | 양수(+) | 실제 승률보다 자신을 과신(Overconfidence) |
| Self-Play 후 변화 | 전략 다변화 | Bias 감소 | 상호 적응을 통해 편향이 중화됨 |
7. 결론 도출 (Conclusion)
위 데이터를 통해 "DQN의 공격적인 플레이는 알고리즘의 수학적 특성(Overestimation)에서 기인한 리스크 추구 성향이며, 이는 PPO의 안정적 성향과 대비된다"는 결론을 도출한다.
