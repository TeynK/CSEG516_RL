# CSEG516 Reinforcement Learning Project: Splendor AI
본 프로젝트는 CSEG516 강화학습 수업의 일환으로 진행된 프로젝트로, 보드 게임 Splendor를 플레이하는 강화학습 에이전트(DQN, PPO)를 구현하고 평가한 결과물입니다.
작성자: 20241584 김태윤
1. 프로젝트 개요
   이 리포지토리는 Splendor 게임 환경을 구축하고, 강화학습 알고리즘 간의 정책 차이를 통해 역으로 전략이 어떤 범주에 속하는지를 도출해내는 실험 프로젝트입니다.
   주요 구현 사항은 다음과 같습니다:
    Splendor 게임 로직: Python 기반의 자체 게임 엔진 구현 (splendor_game/)
    Gym/PettingZoo 환경: 강화학습을 위한 OpenAI Gym 호환 래퍼 (envs/)
    알고리즘:
        Maskable DQN (Deep Q-Network with Action Masking)
        PPO (Proximal Policy Optimization)
    학습용 봇: Random Bot, Weak Heuristic Bot, Heuristic Bot
2. 설치 방법 (Installation)본 프로젝트는 Python 3.10+ 환경에서 동작을 보장합니다. 필요한 라이브러리는 requirements.txt에 명시되어 있습니다.
```
# 의존성 패키지 설치
pip install -r requirements.txt

# 프로젝트 패키지 설치 (Editable 모드)
pip install -e .
```
3. 프로젝트 구조 (Project Structure)
```
├── agents/                 # 에이전트 구현체 (DQN, Random, Heuristic 등)
├── configs/                # 학습 하이퍼파라미터 설정 (YAML)
├── envs/                   # Splendor Gym 환경 및 Wrapper
├── results/                # 학습 결과 (Logs, Models, Plots)
│   ├── logs/               # TensorBoard 로그
│   ├── models/             # 학습된 모델 (.zip)
│   └── plots/              # 분석 그래프
├── scripts/                # 실행 스크립트 (학습 및 평가)
├── splendor_game/          # Splendor 게임 코어 로직
├── setup.py                # 패키지 설정 파일
└── requirements.txt        # 필요 라이브러리 목록
```
4. 사용 방법 (Usage)
4.1 에이전트 학습 (Training)scripts/train.py를 사용하여 에이전트를 학습시킬 수 있습니다. 설정 파일은 configs/ 디렉토리에 있습니다.
```
# DQN 에이전트 학습 예시
py -m scripts.train --model DQN

# PPO 에이전트 학습 예시
py -m scripts.train --model PPO
```
4.2 에이전트 경쟁 (Estimate)
학습된 모델끼리의 경쟁한 결과물을 출력하는 코드입니다.
```
py -m scripts.estimate --games 100
```
5. 실험 결과 (Results)
학습 결과 및 봇 대결 분석 차트는 results/plots/ 디렉토리에서 확인할 수 있습니다.
    PPO vs DQN: 두 강화학습 에이전트 간의 대결 지표 (Win Rate, Score Distribution 등)
주요 학습 로그는 results/logs/에 저장되어 있으며, TensorBoard를 통해 시각화할 수 있습니다.
```
tensorboard --logdir results/logs
```
6. 수학적 배경 (Mathematical Background)
Maskable DQN
    기존 DQN의 $Q(s, a)$ 함수 업데이트 시, 유효하지 않은 액션(invalid actions)을 마스킹하여 탐색 효율성을 높였습니다.
    $$Q_{target} = r + \gamma \max_{a' \in \mathcal{A}_{valid}} Q(s', a'; \theta^{-})$$
    여기서 $\mathcal{A}_{valid}$는 현재 상태 $s'$에서 가능한 유효 액션 집합을 의미합니다.
PPO (Proximal Policy Optimization)
    정책 업데이트 시 급격한 변화를 방지하기 위해 Clipped Objective를 사용합니다.
    $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$
    이때 $r_t(\theta)$는 확률 비율(probability ratio), $\hat{A}_t$는 어드밴티지 함수(advantage function) 추정값입니다.