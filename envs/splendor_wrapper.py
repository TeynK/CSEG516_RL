# envs/splendor_wrapper.py

"""
PettingZoo AECEnv Wrapper for the SplendorGame engine.

This file translates the object-oriented SplendorGame state into
fixed-size NumPy arrays (observation and action mask) suitable
for deep reinforcement learning agents.
"""

import copy
import itertools
import dataclasses
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict as GymDict
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.conversions import to_parallel

# --- Import from our custom game engine ---
# (이 import 경로는 splendor_rl_project/ 최상위에서 실행하는 것을 기준)
from splendor_game.game import SplendorGame
from splendor_game.constants import (
    GemColor,
    CARD_LEVELS,
    FACE_UP_CARDS_PER_LEVEL,
    MAX_RESERVED_CARDS,
    SETUP_CONFIG,
    MAX_PLAYERS,
    WINNING_SCORE
)
from splendor_game.actions import Action, ActionType
from splendor_game.card import DevelopmentCard, NobleTile

# --- Helper functions for PettingZoo ---

def env(**kwargs):
    """AECEnv 래퍼 헬퍼"""
    internal_env = SplendorEnv(**kwargs)
    # PPO/DQN은 병렬 API를 선호하는 경우가 많으므로 to_parallel을 적용할 수 있습니다.
    # (단, PPO/DQN 트레이너에서 직접 병렬 환경을 관리한다면 원본 AECEnv 사용)
    # internal_env = to_parallel(internal_env)
    return internal_env

def raw_env(**kwargs):
    """래퍼 없는 순수 AECEnv 반환"""
    return SplendorEnv(**kwargs)

class SplendorEnv(AECEnv):
    """
    PettingZoo AECEnv 인터페이스로 SplendorGame 엔진을 래핑합니다.

    관측(Observation):
    - 1D NumPy 벡터로 평탄화됩니다.
    - (현재 플레이어 상태, 1번 상대 상태, ..., N번 상대 상태, 보드 상태)
    
    행동(Action):
    - 하나의 정수(Integer)로 매핑됩니다.
    - 총 45개의 고정된 행동으로 구성됩니다.
    - 유효한 행동은 'action_mask'로 제공됩니다.
    """
    
    metadata = {
        "name": "splendor_v0",
        "render_modes": ["human"],
        "is_parallelizable": True,
    }

    # --- 1. 초기화 및 공간 정의 ---

    def __init__(self, num_players=2, render_mode=None):
        super().__init__()

        if not (2 <= num_players <= 4):
            raise ValueError(f"플레이어 수는 2에서 4 사이여야 합니다. (입력: {num_players})")
        
        self.num_players = num_players
        self.render_mode = render_mode

        # 1. 코어 게임 엔진 인스턴스 생성
        self.game = SplendorGame(num_players=self.num_players)

        # 2. 에이전트 설정
        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)

        # 3. 상태 저장 변수 (AECEnv 표준)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection: str = ""
        
        # --- 4. 행동 공간 정의 ---
        # (int -> Action 객체) / (Action 객체 식별자 -> int) 맵 생성
        (
            self.action_map_int_to_obj,
            self.action_map_obj_to_int
        ) = self._create_action_maps()
        
        self.TOTAL_ACTIONS = len(self.action_map_int_to_obj) # 총 45개

        self.action_spaces = {
            agent: Discrete(self.TOTAL_ACTIONS) for agent in self.agents
        }

        # --- 5. 관측 공간 정의 ---
        self._gem_colors_all = GemColor.get_all_gems() # 6 (standard + gold)
        self._gem_colors_standard = GemColor.get_standard_gems() # 5
        
        # Noble/Card 특징 벡터 크기 계산
        self._card_feature_size = 1 + 1 + len(self._gem_colors_standard) + len(self._gem_colors_all) # level, points, bonus(one-hot), cost(6) = 13
        self._noble_feature_size = 1 + len(self._gem_colors_standard) # points, cost(5) = 6
        
        # 최대 플레이어 수(4) 기준으로 공간을 고정 (padding)
        self._max_nobles = SETUP_CONFIG[MAX_PLAYERS]['nobles']

        self.OBS_VECTOR_SIZE = self._calculate_obs_size()

        self.observation_spaces = {
            agent: GymDict({
                "observation": Box(
                    low=0, high=1.0, shape=(self.OBS_VECTOR_SIZE,), dtype=np.float32
                ),
                "action_mask": Box(
                    low=0, high=1, shape=(self.TOTAL_ACTIONS,), dtype=np.int8
                )
            })
            for agent in self.agents
        }

    def _create_action_maps(self) -> Tuple[Dict[int, Action], Dict[Any, int]]:
        """
        모든 60개의 가능한 행동에 대해 (정수 <-> Action 객체) 매핑을 생성합니다.
        
        - 10 (Take 3 - 3 gems): 5C3
        - 10 (Take 3 - 2 gems): 5C2
        - 5  (Take 3 - 1 gem): 5C1
        - 5  (Take 2): 5C1
        - 12 (Buy Face-up): 3 levels * 4 slots
        - 3  (Buy Reserved): 3 slots
        - 12 (Reserve Face-up): 3 levels * 4 slots
        - 3  (Reserve Deck): 3 levels
        Total = 60
        """
        int_to_obj: Dict[int, Action] = {}
        obj_to_int: Dict[Any, int] = {}
        idx = 0

        standard_gems = GemColor.get_standard_gems()

        # --- 1. Take 3 (모든 조합: 10 + 10 + 5 = 25 actions) ---
        # (행동 타입, frozenset)을 키로 사용하여 TAKE_THREE의 하위 조합을 구분
        
        # 1-1. 3개 가져오기 (10 actions)
        for combo in itertools.combinations(standard_gems, 3):
            gems = {c: 1 for c in combo}
            action = Action(ActionType.TAKE_THREE_GEMS, gems=gems)
            int_to_obj[idx] = action
            key = (ActionType.TAKE_THREE_GEMS, frozenset(combo))
            obj_to_int[key] = idx 
            idx += 1
            
        # 1-2. 2개 가져오기 (10 actions)
        for combo in itertools.combinations(standard_gems, 2):
            gems = {c: 1 for c in combo}
            action = Action(ActionType.TAKE_THREE_GEMS, gems=gems)
            int_to_obj[idx] = action
            key = (ActionType.TAKE_THREE_GEMS, frozenset(combo))
            obj_to_int[key] = idx 
            idx += 1

        # 1-3. 1개 가져오기 (5 actions)
        for combo in itertools.combinations(standard_gems, 1):
            gems = {c: 1 for c in combo}
            action = Action(ActionType.TAKE_THREE_GEMS, gems=gems)
            int_to_obj[idx] = action
            key = (ActionType.TAKE_THREE_GEMS, frozenset(combo))
            obj_to_int[key] = idx 
            idx += 1

        # 2. Take 2 (5 actions)
        for color in standard_gems:
            gems = {color: 2}
            action = Action(ActionType.TAKE_TWO_GEMS, gems=gems)
            int_to_obj[idx] = action
            key = (ActionType.TAKE_TWO_GEMS, color) # (TAKE_TWO, GemColor.RED)
            obj_to_int[key] = idx 
            idx += 1

        # 3. Buy Face-up (12 actions)
        for level in CARD_LEVELS:
            for index in range(FACE_UP_CARDS_PER_LEVEL):
                action = Action(ActionType.BUY_CARD, level=level, index=index)
                int_to_obj[idx] = action
                key = (ActionType.BUY_CARD, level, index, False)
                obj_to_int[key] = idx 
                idx += 1
        
        # 4. Buy Reserved (3 actions)
        for index in range(MAX_RESERVED_CARDS):
            action = Action(ActionType.BUY_CARD, index=index, is_reserved_buy=True)
            int_to_obj[idx] = action
            key = (ActionType.BUY_CARD, None, index, True)
            obj_to_int[key] = idx
            idx += 1

        # 5. Reserve Face-up (12 actions)
        for level in CARD_LEVELS:
            for index in range(FACE_UP_CARDS_PER_LEVEL):
                action = Action(ActionType.RESERVE_CARD, level=level, index=index)
                int_to_obj[idx] = action
                key = (ActionType.RESERVE_CARD, level, index, False)
                obj_to_int[key] = idx
                idx += 1

        # 6. Reserve Deck (3 actions)
        for level in CARD_LEVELS:
            action = Action(ActionType.RESERVE_CARD, level=level, is_deck_reserve=True)
            int_to_obj[idx] = action
            key = (ActionType.RESERVE_CARD, level, None, True)
            obj_to_int[key] = idx
            idx += 1

        assert idx == 60, f"총 행동 개수가 60이 아닙니다: {idx}"
        return int_to_obj, obj_to_int

    def _calculate_obs_size(self) -> int:
        """관측 벡터(1D)의 총 크기를 계산합니다."""
        
        size = 0
        
        # 1. 플레이어 상태 (x 최대 4명, 패딩 포함)
        player_state_size = 0
        player_state_size += len(self._gem_colors_all) # 보유 보석 (6)
        player_state_size += len(self._gem_colors_standard) # 보너스 (5)
        player_state_size += 1 # 점수
        player_state_size += 1 # 예약 카드 수
        player_state_size += MAX_RESERVED_CARDS * self._card_feature_size # 예약 카드 특징 (3 * 13)
        
        size += MAX_PLAYERS * player_state_size

        # 2. 보드 상태
        size += len(self._gem_colors_all) # 보드 보석 스택 (6)
        size += len(CARD_LEVELS) # 덱 남은 카드 수 (3)
        size += self._max_nobles * self._noble_feature_size # 귀족 타일 특징 (5 * 6)
        size += (3 * 4) * self._card_feature_size # 공개 카드 특징 (12 * 13)

        # 2인플 기준 (예시):
        # player_state = 6 + 5 + 1 + 1 + (3*13) = 13 + 39 = 52
        # (2명 플레이어지만 4명분 공간 할당) 4 * 52 = 208
        # board_state = 6 (gems) + 3 (decks) + (5*6) (nobles) + (12*13) (cards)
        #             = 6 + 3 + 30 + 156 = 195
        # Total = 208 + 195 = 403
        
        return size

    # --- 2. 핵심 메서드: observe, step, reset ---

    def _get_obs_vector(self, player_id: int) -> np.ndarray:
        """
        현재 게임 상태를 player_id 에이전트의 관점에서 1D 벡터로 변환합니다.
        항상 (현재 플레이어, 다음 플레이어, ...) 순서로 정렬됩니다.
        """
        obs = np.zeros(self.OBS_VECTOR_SIZE, dtype=np.float32)
        idx = 0
        game = self.game

        # --- 1. 플레이어 상태 (항상 4명분 공간) ---
        player_order = [(player_id + i) % self.num_players for i in range(self.num_players)]
        
        player_state_size_per_player = 0 # 1인당 벡터 크기 (패딩 계산용)

        for i, pid in enumerate(player_order):
            player = game.players[pid]
            player_start_idx = idx
            
            # 1-1. 보유 보석 (6)
            for color in self._gem_colors_all:
                obs[idx] = player.gems[color] / 10.0 # 정규화 (최대 10개)
                idx += 1
            # 1-2. 보너스 (5)
            for color in self._gem_colors_standard:
                obs[idx] = player.bonuses[color] / 15.0 # 정규화 (15점=15개?)
                idx += 1
                
            # 1-3. 점수 (1)
            # *** 수정: min(1.0, ...) 클리핑 추가 ***
            obs[idx] = min(1.0, player.score / WINNING_SCORE) # 정규화 (1.0 초과 방지)
            idx += 1
            
            # 1-4. 예약 카드 수 (1)
            obs[idx] = len(player.reserved_cards) / MAX_RESERVED_CARDS
            idx += 1
            
            # 1-5. 예약 카드 특징 (3 * 13) - 패딩 포함
            for j in range(MAX_RESERVED_CARDS):
                if j < len(player.reserved_cards):
                    card = player.reserved_cards[j]
                    idx = self._encode_card(obs, idx, card)
                else:
                    idx += self._card_feature_size # 빈 슬롯 스킵
            
            if i == 0:
                player_state_size_per_player = idx - player_start_idx


        # 비어있는 플레이어 슬롯 패딩 (e.g., 2인플 시 2명분)
        unused_player_slots = MAX_PLAYERS - self.num_players
        idx += unused_player_slots * player_state_size_per_player

        # --- 2. 보드 상태 ---
        # 2-1. 보드 보석 스택 (6)
        for color in self._gem_colors_all:
            obs[idx] = game.board.gem_stacks[color] / 7.0 # 정규화 (최대 7개)
            idx += 1
        
        # 2-2. 덱 남은 카드 수 (3)
        max_deck_sizes = {1: 40, 2: 30, 3: 20}
        for level in CARD_LEVELS:
            obs[idx] = len(game.board.decks[level]) / max_deck_sizes[level]
            idx += 1
            
        # 2-3. 귀족 타일 (최대 5 * 6) - 패딩 포함
        for i in range(self._max_nobles):
            if i < len(game.board.nobles):
                noble = game.board.nobles[i]
                idx = self._encode_noble(obs, idx, noble)
            else:
                idx += self._noble_feature_size # 빈 슬롯 스킵
                
        # 2-4. 공개 카드 (12 * 13) - 패딩 포함
        for level in CARD_LEVELS:
            for i in range(FACE_UP_CARDS_PER_LEVEL):
                card = game.board.face_up_cards[level][i]
                if card:
                    idx = self._encode_card(obs, idx, card)
                else:
                    idx += self._card_feature_size # 빈 슬롯 스킵
        
        assert idx == self.OBS_VECTOR_SIZE, f"관측 벡터 크기 불일치: {idx} != {self.OBS_VECTOR_SIZE}"
        return obs

    def _encode_card(self, obs: np.ndarray, idx: int, card: DevelopmentCard) -> int:
        """obs 벡터에 카드 1장의 특징(13)을 인코딩하고 다음 인덱스를 반환"""
        # 1. Level (1)
        obs[idx] = card.level / 3.0
        idx += 1
        # 2. Points (1)
        obs[idx] = card.points / 5.0
        idx += 1
        # 3. Bonus (one-hot, 5)
        bonus_idx = self._gem_colors_standard.index(card.gem_type)
        obs[idx + bonus_idx] = 1.0
        idx += 5
        # 4. Cost (6)
        for color in self._gem_colors_all:
            obs[idx] = card.cost.get(color, 0) / 7.0 # (L3 최대 7)
            idx += 1
        return idx

    def _encode_noble(self, obs: np.ndarray, idx: int, noble: NobleTile) -> int:
        """obs 벡터에 귀족 1장의 특징(6)을 인코딩하고 다음 인덱스를 반환"""
        # 1. Points (1)
        obs[idx] = noble.points / 3.0
        idx += 1
        # 2. Cost (5)
        for color in self._gem_colors_standard:
            obs[idx] = noble.cost.get(color, 0) / 4.0 # (최대 4)
            idx += 1
        return idx
        
    def _get_action_mask(self) -> np.ndarray:
        """현재 플레이어의 유효한 행동 마스크(60)를 생성합니다."""
        # *** 수정: TOTAL_ACTIONS -> 60 ***
        mask = np.zeros(self.TOTAL_ACTIONS, dtype=np.int8)
        legal_actions = self.game.get_legal_actions()

        for action in legal_actions:
            key: Any = None
            if action.action_type == ActionType.TAKE_THREE_GEMS:
                # *** 수정: 1, 2, 3개 조합 모두 처리 ***
                key = (ActionType.TAKE_THREE_GEMS, frozenset(action.gems.keys()))
            elif action.action_type == ActionType.TAKE_TWO_GEMS:
                # *** 수정: (Type, Color) 튜플로 키 변경 ***
                key = (ActionType.TAKE_TWO_GEMS, list(action.gems.keys())[0])
            elif action.action_type == ActionType.BUY_CARD:
                key = (ActionType.BUY_CARD, action.level, action.index, action.is_reserved_buy)
            elif action.action_type == ActionType.RESERVE_CARD:
                key = (ActionType.RESERVE_CARD, action.level, action.index, action.is_deck_reserve)
            
            if key in self.action_map_obj_to_int:
                int_idx = self.action_map_obj_to_int[key]
                mask[int_idx] = 1
            else:
                # 이 경고가 뜨면 _create_action_maps의 키 정의 로직에 문제
                print(f"[경고] 유효한 행동 {action}을 정수 인덱스로 변환할 수 없습니다. (키: {key})")
                
        return mask

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """AECEnv: 현재 에이전트의 관측을 반환합니다."""
        player_id = self.agents.index(agent)
        observation = self._get_obs_vector(player_id)
        
        # 현재 턴인 에이전트에게만 유효한 액션 마스크 제공
        if agent == self.agent_selection:
            action_mask = self._get_action_mask()
        else:
            action_mask = np.zeros(self.TOTAL_ACTIONS, dtype=np.int8)
            
        return {"observation": observation, "action_mask": action_mask}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """AECEnv: 환경을 초기화합니다."""
        # (참고: PettingZoo는 seed() 메서드를 별도로 호출하는 것을 권장하지만,
        #  gymnasium 스타일로 reset에 seed를 포함하는 것도 일반적임)
        # if seed is not None:
        #     self.seed(seed)

        self.game.reset()

        # PettingZoo 상태 초기화
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action: int) -> None:
        """
        AECEnv: 에이전트가 선택한 정수(action)를 실행합니다.
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # 이미 종료된 에이전트의 행동은 스킵
            return self._was_dead_step(action)

        current_agent = self.agent_selection
        current_player_id = self.agents.index(current_agent)
        player = self.game.players[current_player_id]

        # --- 1. 정수 -> Action 객체 변환 ---
        if self._get_action_mask()[action] == 0:
            # (디버깅) 에이전트가 유효하지 않은 행동을 선택함
            print(f"[경고] 에이전트 {current_agent}가 유효하지 않은 행동({action})을 선택했습니다.")
            
            # *** 수정: 잘못된 행동 시, 게임을 비정상 종료(Truncation)시킴 ***
            self.truncations = {agent: True for agent in self.agents}
            self.infos[current_agent]["error"] = "Invalid action submitted."
            # 행동을 실행하지 않고 즉시 반환
            return

        template_action = self.action_map_int_to_obj[action]
        # 정수로부터 얻은 템플릿 Action에는 실제 카드 객체가 포함되어 있지 않으므로,
        # BUY/RESERVE 액션의 경우 카드 객체를 포함한 새로운 Action 인스턴스를 생성합니다.
        final_action = copy.deepcopy(template_action)

        try:
            if template_action.action_type == ActionType.BUY_CARD:
                card_to_buy = None
                if template_action.is_reserved_buy:
                    # 예약된 카드 구매
                    card_to_buy = player.reserved_cards[template_action.index]
                else:
                    # 공개된 카드 구매
                    card_to_buy = self.game.board.face_up_cards[template_action.level][template_action.index]
                
                final_action = Action(
                    action_type=template_action.action_type,
                    level=template_action.level,
                    index=template_action.index,
                    is_reserved_buy=template_action.is_reserved_buy,
                    card=card_to_buy
                )

            elif template_action.action_type == ActionType.RESERVE_CARD:
                # 덱에서 예약하는 경우는 card 객체가 필요 없음
                if not template_action.is_deck_reserve:
                    card_to_reserve = self.game.board.face_up_cards[template_action.level][template_action.index]
                    final_action = Action(
                        action_type=template_action.action_type,
                        level=template_action.level,
                        index=template_action.index,
                        is_deck_reserve=template_action.is_deck_reserve,
                        card=card_to_reserve
                    )
        except IndexError:
            # (디버깅) 예약된 카드가 없는데 예약 구매를 시도하는 등
            print(f"[오류] Action 객체 생성 중 오류: {template_action}")
            # 이 경우 게임을 비정상 종료시킴
            self.truncations = {agent: True for agent in self.agents}
            self.infos[current_agent]["error"] = "Invalid action object creation"
            return
        
        # --- 2. 게임 엔진 실행 ---
        is_game_over = self.game.step(final_action)

        # --- 3. 다음 에이전트 선택 ---
        self.agent_selection = self._agent_selector.next()

        # --- 4. 보상 및 종료 상태 업데이트 ---
        # 턴 기반 보상 (0)
        self.rewards = {agent: 0.0 for agent in self.agents}

        if is_game_over:
            # 게임 종료: 승/패에 따라 보상 분배
            self.terminations = {agent: True for agent in self.agents}
            winner_id = self.game.winner_id
            
            for i, agent in enumerate(self.agents):
                if i == winner_id:
                    self.rewards[agent] = 1.0
                else:
                    self.rewards[agent] = -1.0 # (또는 0.0)
            
            self.infos = {agent: {"game_winner": winner_id} for agent in self.agents}
        
        # 누적 보상 업데이트
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

        # 5. 렌더링
        if self.render_mode == "human":
            self.render()

    def render(self) -> None:
        """(선택 사항) 터미널에 현재 상태 출력"""
        if self.render_mode == "human":
            print("\n" + "="*60)
            print(f"--- 턴: {self.game.current_player_index}, 현재 에이전트: {self.agent_selection} ---")
            print(self.game.board)
            for i in range(self.num_players):
                player = self.game.players[i]
                print(player)
            print("="*60)

    # --- PettingZoo 필수 프로퍼티 ---
    
    def action_space(self, agent: str) -> Discrete:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> GymDict:
        return self.observation_spaces[agent]
    
    def close(self) -> None:
        pass