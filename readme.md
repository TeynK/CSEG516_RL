#  Splendor AI: RL Agent Strategy Analysis
> **CSEG516 Reinforcement Learning Project**  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Framework](https://img.shields.io/badge/Framework-PettingZoo%20%7C%20StableBaselines3-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 1. 프로젝트 개요 (Project Overview)

본 프로젝트는 **CSEG516 강화학습** 수업의 일환으로 진행된 연구입니다. 전략 보드게임 **Splendor**를 플레이하는 강화학습 에이전트(PPO, DQN)를 구현하고, 각 알고리즘이 가진 이론적 특성이 실제 게임 전략으로 어떻게 발현되는지를 분석했습니다.

###  연구 목표: 수학적 성향에서 전략적 성격으로
강화학습 알고리즘은 고유의 손실 함수(Loss Function)와 학습 메커니즘에 따라 서로 다른 **수학적 성향** 을 내포합니다. 본 연구는 이러한 이론적 특성이 자원 관리와 레이싱 요소가 결합된 복잡한 환경인 Splendor에서 인간이 해석 가능한 **구체적 전략** 으로 전이되는 과정을 규명합니다.

*   **On-Policy (PPO):** 학습 신호의 신뢰성을 중시하는 특성이 **저분산 안정적** 점수 분포를 가지게 하는가?
*   **Off-Policy (DQN):** 과거 경험을 재사용하며 낙관적 가치 추정을 하는 특성이 **'고위험-고수익(High-Risk)'** 점수 분포를 가지게 하는가?
*   **Profiling:** 게임 내 전략이 내포한 수학적 위험도(Variance)와 기회비용의 정량적 프로파일링.

###  주요 기능
*   **Custom Environment:** PettingZoo 기반의 Splendor AEC 환경 및 Gymnasium Wrapper 구현.
*   **Advanced Algorithms:** 유효하지 않은 행동을 원천 차단하는 **Action Masking**이 적용된 Maskable PPO 및 Maskable DQN 구현.
*   **Curriculum Learning:** Random Bot $\rightarrow$ Heuristic Bot (Aggressive/Defensive/Balanced)으로 이어지는 단계적 학습 파이프라인.
*   **Analysis Tools:** 에이전트 간 대전 시뮬레이션, 승률 분석, 전략적 성향(Tier 선호도 등) 시각화 도구.

---

## 2. 실험 결과 (Experimental Results)

실험 결과, 두 에이전트는 약 50:50으로 대등한 승률을 보였으나, **승리를 쟁취하는 과정(Quality of Victory)** 과 **전략의 결**은 극명하게 갈렸습니다.

###  전략적 성향 프로파일링 (Strategic Profiling)

| Feature | **PPO Agent** | **DQN Agent** |
| :--- | :--- | :--- |
| **Mathematical Basis** | **Reliability-Seeking** <br> ($L^{CLIP}$ 통한 급격한 변화 억제) | **Optimistic Bias** <br> (Max Operator에 의한 가치 과대평가) |
| **Risk Profile** | **Low Variance** (안정적 승리) | **High Variance** (모 아니면 도) |
| **Value Estimation** | **Realistic** <br> (전황에 따라 가치가 민감하게 반응) | **Overestimation** <br> (패배 직전에도 승리할 수 있다는 낙관적 사고) |
| **Converged Strategy** | **Tier 2 중심의 빌딩** <br> 비용 대비 효율을 중시하며, <br>안정적으로 자원을 확보하는 정석 플레이 | **Tier 1 + Tier 3/Noble** <br> 값싼 카드로 칩을 모아, <br>고득점 카드를 노리는 기회주의적 플레이 |

---

## 3. 설치 방법 (Installation)

본 프로젝트는 **Python 3.10+** 환경에서 동작을 보장합니다.

```bash
# 1. 저장소 클론
git clone https://github.com/TeynK/CSEG516_RL.git
cd CSEG516_RL

# 2. 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 패키지 설치
pip install -r requirements.txt

# 4. 프로젝트 패키지 설치 (Editable 모드)
pip install -e .
```

---

## 4. 사용 방법 (Usage)

### 4.1 에이전트 학습 (Training)
`scripts/train.py`를 사용하여 에이전트를 학습시킵니다. 하이퍼파라미터는 `configs/` 디렉토리 내의 YAML 파일에서 수정 가능합니다.

```bash
# DQN 에이전트 학습
python -m scripts.train --model DQN

# PPO 에이전트 학습
python -m scripts.train --model PPO
```

### 4.2 성능 평가 및 대전 (Estimation)
학습이 완료된 모델 간의 대전을 수행하고, 승률 및 로그 데이터를 추출하여 분석합니다.

```bash
# 100회 대전 수행 및 결과 출력
python -m scripts.estimate --games 100
```

### 4.3 모니터링 (Monitoring)
TensorBoard를 통해 학습 과정의 리워드 추이와 손실 그래프를 실시간으로 확인합니다.

```bash
tensorboard --logdir results/logs
```

---

## 5. 프로젝트 구조 (Project Structure)

```plaintext
CSEG516_RL/
├── agents/             # 에이전트 구현체 (DQN, PPO, Random, Heuristic)
├── configs/            # 학습 하이퍼파라미터 설정 (config.yaml)
├── envs/               # Splendor Gym 환경 및 Wrapper (PettingZoo 기반)
├── results/            # 실험 결과 저장소
│   ├── logs/           # TensorBoard 로그
│   ├── models/         # 학습된 모델 체크포인트 (.zip)
│   └── plots/          # 분석 결과 시각화 이미지 (Action Dist, Score Curve 등)
├── scripts/            # 실행 스크립트
│   ├── train.py        # 학습 스크립트
│   └── estimate.py     # 평가 및 대전 스크립트
├── splendor_game/      # Splendor 게임 코어 로직
├── setup.py            # 패키지 설정
└── requirements.txt    # 필요 라이브러리 목록
```
