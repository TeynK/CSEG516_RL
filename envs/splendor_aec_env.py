import numpy as np
from typing import Tuple, Dict, Any, Optional
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from collections import defaultdict
import itertools
import dataclasses

from splendor_game.game import SplendorGame
from splendor_game.constants import GemColor, CARD_LEVELS, FACE_UP_CARDS_PER_LEVEL, MAX_RESERVED_CARDS, WINNING_SCORE, MAX_PLAYERS, SETUP_CONFIG
from splendor_game.actions import Action, ActionType
from splendor_game.card import DevelopmentCard, NobleTile

def env(**kwargs):
    internal_env = SplendorEnv(**kwargs)
    return internal_env

def raw_env(**kwargs):
    return SplendorEnv(**kwargs)

class SplendorEnv(AECEnv):
    metadata = {
        "name": "splendor_v0",
        "render_modes": ["human"],
        "is_parallelizable": True,
    }
    
    def __init__(self, num_players=2, render_mode=None):
        super().__init__()
        if not (2 <= num_players <= 4):
            raise ValueError(f"Invalid number of players: {num_players}")
        self.num_players = num_players
        self.render_mode = render_mode
        self.game = SplendorGame(num_players=self.num_players)
        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection: str = ""
        self.action_map_int_to_obj, self.action_map_obj_to_int = self.create_action_maps()
        self.TOTAL_ACTIONS = len(self.action_map_int_to_obj)
        self.action_spaces = {agent: Discrete(self.TOTAL_ACTIONS) for agent in self.agents}
        self._gem_colors_all = GemColor.get_all_gems()
        self._gem_colors_standard = GemColor.get_standard_gems()
        self._card_feature_size = 2 + len(self._gem_colors_all) + len(self._gem_colors_standard)
        self._noble_feature_size = 1 + len(self._gem_colors_standard)
        self._max_nobles =  SETUP_CONFIG[MAX_PLAYERS]['nobles']
        self.OBS_VECTOR_SIZE = self.calculate_obs_size()
        self.observation_spaces = {agent: Box(low=0, high=1.0, shape=(self.OBS_VECTOR_SIZE,), dtype=np.float32) for agent in self.agents}

    def create_action_maps(self) -> Tuple[Dict[int, Action], Dict[Any, int]]:
        int_to_obj: Dict[int, Action] = {}
        obj_to_int: Dict[Any, int] = {}
        idx = 0
        standard_gems = GemColor.get_standard_gems()
        for k in [3, 2, 1]:
            for combo in itertools.combinations(standard_gems, k):
                gems = {c: 1 for c in combo}
                action = Action(ActionType.TAKE_THREE_GEMS, gems=gems)
                int_to_obj[idx] = action
                key = (ActionType.TAKE_THREE_GEMS, frozenset(combo))
                obj_to_int[key] = idx
                idx += 1
        for color in standard_gems:
            gems = {color: 2}
            action = Action(ActionType.TAKE_TWO_GEMS, gems=gems)
            int_to_obj[idx] = action
            key = (ActionType.TAKE_TWO_GEMS, color)
            obj_to_int[key] = idx
            idx += 1
        for level in CARD_LEVELS:
            for index in range(FACE_UP_CARDS_PER_LEVEL):
                action = Action(ActionType.BUY_CARD, level=level, index=index, is_reserved_buy=False)
                int_to_obj[idx] = action
                key = (ActionType.BUY_CARD, level, index, False)
                obj_to_int[key] = idx
                idx += 1
        for index in range(MAX_RESERVED_CARDS):
            action = Action(ActionType.BUY_CARD, index=index, is_reserved_buy=True)
            int_to_obj[idx] = action
            key = (ActionType.BUY_CARD, None, index, True)
            obj_to_int[key] = idx
            idx += 1
        for level in CARD_LEVELS:
            for index in range(FACE_UP_CARDS_PER_LEVEL):
                action = Action(ActionType.RESERVE_CARD, level=level, index=index, is_deck_reserve=False)
                int_to_obj[idx] = action
                key = (ActionType.RESERVE_CARD, level, index, False)
                obj_to_int[key] = idx
                idx += 1
        for level in CARD_LEVELS:
            action = Action(ActionType.RESERVE_CARD, level=level, is_deck_reserve=True)
            int_to_obj[idx] = action
            key = (ActionType.RESERVE_CARD, level, None, True)
            obj_to_int[key] = idx
            idx += 1
        all_gem_colors = GemColor.get_all_gems()
        for combo in itertools.combinations(all_gem_colors, 1):
            gems = {combo[0]: 1}
            action = Action(ActionType.RETURN_GEMS, gems=gems)
            int_to_obj[idx] = action
            key = (ActionType.RETURN_GEMS, frozenset(gems.items()))
            obj_to_int[key] = idx
            idx += 1
        for combo in itertools.combinations_with_replacement(all_gem_colors, 2):
            gems = defaultdict(int)
            for color in combo:
                gems[color] += 1
            action = Action(ActionType.RETURN_GEMS, gems=gems)
            int_to_obj[idx] = action
            key = (ActionType.RETURN_GEMS, frozenset(gems.items()))
            obj_to_int[key] = idx
            idx += 1
        for combo in itertools.combinations_with_replacement(all_gem_colors, 3):
            gems = defaultdict(int)
            for color in combo:
                gems[color] += 1
            action = Action(ActionType.RETURN_GEMS, gems=gems)
            int_to_obj[idx] = action
            key = (ActionType.RETURN_GEMS, frozenset(gems.items()))
            obj_to_int[key] = idx
            idx += 1
        assert idx == 143, f"총 행동 개수가 143이 아닙니다: {idx}"
        return int_to_obj, obj_to_int

    def calculate_obs_size(self) -> int:
        size = 0
        player_state_size = 0
        player_state_size += len(self._gem_colors_all)
        player_state_size += len(self._gem_colors_standard)
        player_state_size += 1
        player_state_size += 1
        player_state_size += MAX_RESERVED_CARDS * self._card_feature_size
        size += MAX_PLAYERS * player_state_size
        size += len(self._gem_colors_all)
        size += len(CARD_LEVELS)
        size += self._max_nobles * self._noble_feature_size
        size += (3 * 4) * self._card_feature_size
        return size
    
    def encode_card(self, obs: np.ndarray, idx: int, card: DevelopmentCard) -> int:
        obs[idx] = card.level / 3.0
        idx += 1
        obs[idx] = card.points / 5.0
        idx += 1
        try:
            bonus_idx = self._gem_colors_standard.index(card.gem_type)
            obs[idx + bonus_idx] = 1.0
        except ValueError:
            pass 
        idx += len(self._gem_colors_standard)
        for color in self._gem_colors_all:
            obs[idx] = card.cost.get(color, 0) / 7.0
            idx += 1
        return idx
    
    def encode_noble(self, obs: np.ndarray, idx: int, noble: NobleTile) -> int:
        obs[idx] = noble.points / 3.0
        idx += 1
        for color in self._gem_colors_standard:
            obs[idx] = noble.cost.get(color, 0) / 4.0 
            idx += 1
        return idx
        
    def get_obs_vector(self, player_id: int) -> np.ndarray:
        obs = np.zeros(self.OBS_VECTOR_SIZE, dtype=np.float32)
        idx = 0
        game = self.game
        player_order = [(player_id + i) % self.num_players for i in range(self.num_players)]
        player_state_size_per_player = 0
        for i, pid in enumerate(player_order):
            player = game.players[pid]
            player_start_idx = idx
            for color in self._gem_colors_all:
                obs[idx] = player.gems[color] / 10.0
                idx += 1
            for color in self._gem_colors_standard:
                obs[idx] = player.bonuses[color] / 15.0
                idx += 1
            obs[idx] = min(1.0, player.score / WINNING_SCORE)
            idx += 1
            obs[idx] = len(player.reserved_cards) / MAX_RESERVED_CARDS
            idx += 1
            for j in range(MAX_RESERVED_CARDS):
                if j < len(player.reserved_cards):
                    card = player.reserved_cards[j]
                    idx = self.encode_card(obs, idx, card)
                else:
                    idx += self._card_feature_size
            if i == 0:
                player_state_size_per_player = idx - player_start_idx
        unused_player_slots = MAX_PLAYERS - self.num_players
        idx += unused_player_slots * player_state_size_per_player
        max_gems_on_board = SETUP_CONFIG[4]['gems'] 
        for color in self._gem_colors_standard:
            obs[idx] = game.board.gem_stacks[color] / max_gems_on_board
            idx += 1
        obs[idx] = game.board.gem_stacks[GemColor.GOLD] / SETUP_CONFIG[4]['gold']
        idx += 1
        max_deck_sizes = {1: 40, 2: 30, 3: 20}
        for level in CARD_LEVELS:
            obs[idx] = len(game.board.decks[level]) / max_deck_sizes[level]
            idx += 1
        for i in range(self._max_nobles):
            if i < len(game.board.nobles):
                noble = game.board.nobles[i]
                idx = self.encode_noble(obs, idx, noble)
            else:
                idx += self._noble_feature_size
        for level in CARD_LEVELS:
            for i in range(FACE_UP_CARDS_PER_LEVEL):
                card = game.board.face_up_cards[level][i]
                if card:
                    idx = self.encode_card(obs, idx, card)
                else:
                    idx += self._card_feature_size 
        
        assert idx == self.OBS_VECTOR_SIZE, f"관측 벡터 크기 불일치: {idx} != {self.OBS_VECTOR_SIZE}"
        return obs

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.TOTAL_ACTIONS, dtype=np.int8)
        legal_actions = self.game.get_legal_actions()
        for action in legal_actions:
            key: Any = None
            if action.action_type == ActionType.TAKE_THREE_GEMS:
                key = (ActionType.TAKE_THREE_GEMS, frozenset(action.gems.keys()))
            elif action.action_type == ActionType.TAKE_TWO_GEMS:
                key = (ActionType.TAKE_TWO_GEMS, list(action.gems.keys())[0])
            elif action.action_type == ActionType.BUY_CARD:
                key = (ActionType.BUY_CARD, action.level, action.index, action.is_reserved_buy)
            elif action.action_type == ActionType.RESERVE_CARD:
                key = (ActionType.RESERVE_CARD, action.level, action.index, action.is_deck_reserve)
            elif action.action_type == ActionType.RETURN_GEMS:
                key = (ActionType.RETURN_GEMS, frozenset(action.gems.items()))
            
            if key in self.action_map_obj_to_int:
                int_idx = self.action_map_obj_to_int[key]
                mask[int_idx] = 1
            else:
                print(f"[경고] 유효한 행동 {action}을 정수 인덱스로 변환할 수 없습니다. (키: {key})")
        return mask

    def observe(self, agent: str) -> np.ndarray:
        player_id = self.agents.index(agent)
        observation = self.get_obs_vector(player_id)
        self.game.current_player_index = player_id
        self.infos[agent]["action_mask"] = self.get_action_mask()
        return observation

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        self.game.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action: int) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        current_agent = self.agent_selection
        current_player_id = self.agents.index(current_agent)
        self.game.current_player_index = current_player_id
        player = self.game.players[current_player_id]
        current_mask = self.get_action_mask()
        current_mask = self.get_action_mask()
        has_no_legal_actions = np.sum(current_mask) == 0
        if action is None:
            if has_no_legal_actions:
                print(f"[경고] 에이전트 {current_agent}가 행동 불능(deadlock) 상태입니다. 게임을 기권패로 종료합니다.")
                self.terminations = {agent: True for agent in self.agents}
                self.truncations = {agent: True for agent in self.agents}
                self.rewards = {agent: 0.1 for agent in self.agents}
                self.rewards[current_agent] = -1.0
                self.infos = {agent: {"game_winner": -1, "deadlock": True} for agent in self.agents}
            else:
                print(f"[오류] 에이전트 {current_agent}가 None 행동을 전달했지만 유효한 행동이 있습니다. 턴을 넘깁니다.")
            self.agent_selection = self._agent_selector.next()
            return
        if current_mask[action] == 0:
            if has_no_legal_actions:
                print(f"[경고] 에이전트 {current_agent}가 행동 불능(deadlock) 상태입니다. (행동 {action} 선택됨) 게임을 기권패로 종료합니다.")
                self.terminations = {agent: True for agent in self.agents}
                self.truncations = {agent: True for agent in self.agents}
                self.rewards = {agent: 0.1 for agent in self.agents}
                self.rewards[current_agent] = -1.0
                self.infos = {agent: {"game_winner": -1, "deadlock": True} for agent in self.agents}
            else:
                print(f"[경고] 에이전트 {current_agent}가 유효하지 않은 행동({action})을 선택했습니다.")
                self.truncations = {agent: True for agent in self.agents}
                self.infos[current_agent]["error"] = "Invalid action submitted."

            self.agent_selection = self._agent_selector.next()
            return
        if current_mask[action] == 0:
            print(f"[경고] 에이전트 {current_agent}가 유효하지 않은 행동({action})을 선택했습니다.")
            self.truncations = {agent: True for agent in self.agents}
            self.infos[current_agent]["error"] = "Invalid action submitted."
            self.agent_selection = self._agent_selector.next()
            return
        template_action = self.action_map_int_to_obj[action]
        final_action = template_action
        is_game_over = False
        try:
            if template_action.action_type == ActionType.BUY_CARD:
                card_to_buy = None
                if template_action.is_reserved_buy:
                    if template_action.index >= len(player.reserved_cards):
                        raise IndexError(f"잘못된 예약 카드 인덱스: {template_action.index}")
                    card_to_buy = player.reserved_cards[template_action.index]
                else:
                    card_to_buy = self.game.board.face_up_cards[template_action.level][template_action.index]
                if card_to_buy is None:
                    raise ValueError("구매하려는 카드가 비어있습니다 (None).")
                final_action = dataclasses.replace(template_action, card=card_to_buy)
            elif template_action.action_type == ActionType.RESERVE_CARD:
                if not template_action.is_deck_reserve:
                    card_to_reserve = self.game.board.face_up_cards[template_action.level][template_action.index]
                    if card_to_reserve is None:
                        raise ValueError("예약하려는 카드가 비어있습니다 (None).")
                    final_action = dataclasses.replace(template_action, card=card_to_reserve)
            if final_action.action_type == ActionType.BUY_CARD:
                can_afford, _ = player.get_payment_details(final_action.card)
                if not can_afford:
                    print("\n--- DEBUG: POTENTIAL VALUE ERROR (SYNC FIX APPLIED) ---")
                    print(f"Agent: {current_agent}")
                    print(f"Action: {final_action}")
                    print(f"Card to buy: {final_action.card}")
                    print(f"Card cost: {final_action.card.cost}")
                    print(f"Player Gems: {player.gems}")
                    print(f"Player Bonuses: {player.bonuses}")
                    print(f"Calculated can_afford: {can_afford}")
                    print("--- END DEBUG ---\n")
                    raise ValueError("마스크와 실제 'can_afford' 상태가 일치하지 않습니다. (SYNC FIX 후에도 발생)")
            is_game_over = self.game.step(final_action)
        except (IndexError, ValueError, TypeError) as e:
            print(f"[오류] Action 객체 생성 또는 step 실행 중 오류: {template_action} (오류: {e})")
            self.truncations = {agent: True for agent in self.agents}
            self.infos[current_agent]["error"] = "Action object creation or step execution error"
            self.agent_selection = self._agent_selector.next()
            return
        if self.game.current_player_state != "RETURN_GEMS":
            self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0.0 for agent in self.agents}
        if is_game_over:
            self.terminations = {agent: True for agent in self.agents}
            winner_id = self.game.winner_id
            for i, agent in enumerate(self.agents):
                if i == winner_id:
                    self.rewards[agent] = 1.0 
                else:
                    self.rewards[agent] = -1.0
            for agent in self.agents:
                self.infos[agent] = {"game_winner": winner_id}
        
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
        
        if self.render_mode == "human":
            self.render()

    def render(self) -> None:
        if self.render_mode == "human":
            current_player_id = self.agents.index(self.agent_selection)
            self.game.current_player_index = current_player_id
            current_player = self.game.get_current_player()
            print("\n" + "="*60)
            print(f"--- 턴: {self.game.current_player_index}, "
                  f"현재 에이전트: player_{current_player.player_id} ({self.game.current_player_state}) ---")
            print(self.game.board)
            for i in range(self.num_players):
                player = self.game.players[i]
                print(player)
            print("="*60)

    def action_space(self, agent: str) -> Discrete:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> Box:
        return self.observation_spaces[agent]
    
    def close(self) -> None:
        pass

