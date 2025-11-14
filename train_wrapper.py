# train_wrapper.py
# [수정됨] MaskablePPO의 'ActionMasker' 래퍼를 지원하도록
# 관찰 공간을 'Box'로 되돌립니다.

import gymnasium as gym
import gymnasium.spaces
import numpy as np
from pettingzoo.utils.env import AECEnv
from typing import Any # [수정] Any 타입 임포트

from envs.splendor_wrapper import env as splendor_aec_env
from agents.heuristic_bot import HeuristicBot
from splendor_game.actions import get_legal_return_gems_actions
# [수정] Action, ActionType 임포트 추가
from splendor_game.actions import Action, ActionType 

class SplendorGymWrapper(gym.Env):
    """
    [수정됨] Petting Zoo AECEnv를 'ActionMasker' 래퍼와 호환되는
    'Box' 관찰 공간을 가진 gym.Env로 래핑합니다.
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=2):
        super().__init__()
        
        self.aec_env: AECEnv = splendor_aec_env(num_players=num_players)
        
        assert num_players == 2, "이 래퍼는 2인용(AI vs Bot) 전용입니다."
        
        # --- [수정된 관찰 공간] ---
        self.action_space = self.aec_env.action_space("player_0")
        self.observation_space = self.aec_env.observation_space("player_0")
        # --- [수정 끝] ---
        
        self.bot = HeuristicBot(player_id=1)
        self.agents = ["player_0", "player_1"]

        self.current_action_mast = None

    def action_mask(self) -> np.ndarray:
        """
        ActionMasker 래퍼가 호출할 메서드.
        현재 캐시된 액션 마스크를 반환합니다.
        """
        assert self.current_action_mask is not None, "Action mask is None. reset()이 먼저 호출되어야 합니다."
        return self.current_action_mask

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """
        [수정됨] AECEnv에서 관찰 값(np.ndarray)만 반환합니다.
        'action_mask'는 observe() 함수에 의해 info 딕셔너리에 자동으로 채워집니다.
        """
        obs_array = self.aec_env.observe(agent_id)
        return obs_array

    # --- [새로 추가된 헬퍼 함수] ---
    def _action_obj_to_key(self, action: Action) -> Any:
        """
        SplendorEnv에 action_to_key가 없으므로, SplendorEnv의
        _get_action_mask 로직을 기반으로 key를 수동 생성합니다.
        """
        key: Any = None
        if action.action_type == ActionType.TAKE_THREE_GEMS:
            key = (ActionType.TAKE_THREE_GEMS, frozenset(action.gems.keys()))
        elif action.action_type == ActionType.TAKE_TWO_GEMS:
            if not action.gems:
                return None # 잘못된 액션
            key = (ActionType.TAKE_TWO_GEMS, list(action.gems.keys())[0])
        elif action.action_type == ActionType.BUY_CARD:
            key = (ActionType.BUY_CARD, action.level, action.index, action.is_reserved_buy)
        elif action.action_type == ActionType.RESERVE_CARD:
            key = (ActionType.RESERVE_CARD, action.level, action.index, action.is_deck_reserve)
        elif action.action_type == ActionType.RETURN_GEMS:
            key = (ActionType.RETURN_GEMS, frozenset(action.gems.items()))
        
        return key
    # --- [헬퍼 함수 끝] ---

    def _run_bot_turn(self):
        """휴리스틱 봇(player_1)의 턴을 자동으로 실행합니다."""
        bot_agent = "player_1"
        
        while self.aec_env.agent_selection == bot_agent:
            termination = self.aec_env.terminations[bot_agent]
            truncation = self.aec_env.truncations[bot_agent]
            
            if termination or truncation:
                return True # 게임 종료

            current_player_obj = self.aec_env.game.players[1]
            if self.aec_env.game.current_player_state == "RETURN_GEMS":
                legal_actions = get_legal_return_gems_actions(current_player_obj)
            else:
                legal_actions = self.aec_env.game.get_legal_actions()

            if not legal_actions:
                self.aec_env.step(None) 
                continue
                
            bot_action_obj = self.bot.choose_action(self.aec_env.game.board, current_player_obj, legal_actions)

            if bot_action_obj is None:
                bot_action_obj = legal_actions[0]

            bot_action_int = -1
            try:
                # [수정] self.aec_env.action_to_key -> self._action_obj_to_key
                key = self._action_obj_to_key(bot_action_obj)
                
                if key in self.aec_env.action_map_obj_to_int:
                    bot_action_int = self.aec_env.action_map_obj_to_int[key]
                else:
                    found_match = False
                    for template_action in legal_actions:
                        # (템플릿 액션과 봇 액션의 속성 비교)
                        if (template_action.action_type == bot_action_obj.action_type and
                            template_action.level == bot_action_obj.level and
                            template_action.index == bot_action_obj.index and
                            template_action.is_reserved_buy == bot_action_obj.is_reserved_buy and
                            template_action.is_deck_reserve == bot_action_obj.is_deck_reserve):
                            
                            # [수정] self.aec_env.action_to_key -> self._action_obj_to_key
                            template_key = self._action_obj_to_key(template_action)
                            if template_key in self.aec_env.action_map_obj_to_int:
                                bot_action_int = self.aec_env.action_map_obj_to_int[template_key]
                                found_match = True
                                break
                    
                    if not found_match:
                        # [수정] self.aec_env.action_to_key -> self._action_obj_to_key
                        template_key = self._action_obj_to_key(legal_actions[0])
                        bot_action_int = self.aec_env.action_map_obj_to_int[template_key]

            except Exception as e:
                print(f"[Wrapper 오류] 봇 액션 변환 중 예외: {e}")
                # [수정] self.aec_env.action_to_key -> self._action_obj_to_key
                key = self._action_obj_to_key(legal_actions[0])
                bot_action_int = self.aec_env.action_map_obj_to_int[key]

            self.aec_env.step(bot_action_int)
            
        return False # 게임 계속 (AI 턴)

    def step(self, action: int):
        """AI(player_0)의 행동을 받고, 봇의 턴까지 실행한 뒤, AI의 다음 상태를 반환합니다."""
        
        ai_agent = "player_0"
        
        self.aec_env.step(action)
        game_over = self._run_bot_turn()
        
        if not game_over:
            observation = self._get_obs(ai_agent) 
            reward = self.aec_env.rewards[ai_agent]
            termination = self.aec_env.terminations[ai_agent]
            truncation = self.aec_env.truncations[ai_agent]
            info = self.aec_env.infos[ai_agent] 
        else:
            observation = self._get_obs(ai_agent) 
            reward = self.aec_env.rewards[ai_agent]
            termination = True 
            truncation = True
            info = self.aec_env.infos[ai_agent] 

        self.current_action_mask = info.get("action_mask")
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        """환경을 리셋하고 AI(player_0)의 첫 번째 관찰 값(np.ndarray)을 반환합니다."""
        self.aec_env.reset(seed=seed)
        
        if self.aec_env.agent_selection != "player_0":
            self._run_bot_turn()
            
        observation = self._get_obs("player_0") 
        info = self.aec_env.infos["player_0"]  
        self.current_action_mask = info.get("action_mask")
        
        return observation, info

    def render(self):
        self.aec_env.render()

    def close(self):
        self.aec_env.close()