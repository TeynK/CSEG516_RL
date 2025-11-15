import gymnasium as gym
# Dict와 Box를 gymnasium.spaces에서 직접 임포트합니다.
from gymnasium.spaces import Dict, Box 
import numpy as np
from pettingzoo.utils.env import AECEnv
from typing import Any

from envs.splendor_aec_env import env as splendor_aec_env
from agents.heuristic_bot import HeuristicBot
from agents.random_bot import RandomBot
from splendor_game.actions import get_legal_return_gems_actions
from splendor_game.actions import Action, ActionType 

class SplendorGymWrapper(gym.Env):
    """
    [수정됨] MaskableDQN 또는 MaskablePPO와 호환되도록
    gym.Env로 래핑합니다.
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=2):
        super().__init__()
        
        self.aec_env: AECEnv = splendor_aec_env(num_players=num_players)
        
        assert num_players == 2, "이 래퍼는 2인용(AI vs Bot) 전용입니다."
        
        # --- [MaskableDQN 호환을 위한 수정] ---
        
        # 1. 원본 Box 관찰 공간
        original_obs_space = self.aec_env.observation_space("player_0")
        
        # 2. 액션 마스크 공간 정의
        action_mask_space = Box(
            low=0, 
            high=1, 
            shape=(self.aec_env.TOTAL_ACTIONS,), 
            dtype=np.int8
        )
        
        self.action_space = self.aec_env.action_space("player_0")
        
        # 3. 관찰 공간을 Dict로 재정의합니다.
        self.observation_space = Dict({
            "observation": original_obs_space,
            "action_mask": action_mask_space
        })
        # --- [수정 끝] ---
        
        self.bot = HeuristicBot(player_id=1)
        # self.bot = RandomBot(player_id=1)
        self.agents = ["player_0", "player_1"]

        self.current_action_mask = None

    def action_mask(self) -> np.ndarray:
        """
        ActionMasker 래퍼가 호출할 메서드.
        (MaskablePPO에서만 사용됨)
        """
        assert self.current_action_mask is not None, "Action mask is None. reset()이 먼저 호출되어야 합니다."
        return self.current_action_mask

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """
        AECEnv에서 관찰 값(np.ndarray)만 반환합니다.
        """
        obs_array = self.aec_env.observe(agent_id)
        return obs_array

    # --- [헬퍼 함수 _action_obj_to_key 는 변경 없음] ---
    def _action_obj_to_key(self, action: Action) -> Any:
        key: Any = None
        if action.action_type == ActionType.TAKE_THREE_GEMS:
            key = (ActionType.TAKE_THREE_GEMS, frozenset(action.gems.keys()))
        elif action.action_type == ActionType.TAKE_TWO_GEMS:
            if not action.gems:
                return None 
            key = (ActionType.TAKE_TWO_GEMS, list(action.gems.keys())[0])
        elif action.action_type == ActionType.BUY_CARD:
            key = (ActionType.BUY_CARD, action.level, action.index, action.is_reserved_buy)
        elif action.action_type == ActionType.RESERVE_CARD:
            key = (ActionType.RESERVE_CARD, action.level, action.index, action.is_deck_reserve)
        elif action.action_type == ActionType.RETURN_GEMS:
            key = (ActionType.RETURN_GEMS, frozenset(action.gems.items()))
        
        return key
    # --- [헬퍼 함수 끝] ---

    # --- [_run_bot_turn 함수는 변경 없음] ---
    def _run_bot_turn(self):
        bot_agent = "player_1"
        
        while self.aec_env.agent_selection == bot_agent:
            termination = self.aec_env.terminations[bot_agent]
            truncation = self.aec_env.truncations[bot_agent]
            
            if termination or truncation:
                return True 

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
                key = self._action_obj_to_key(bot_action_obj)
                
                if key in self.aec_env.action_map_obj_to_int:
                    bot_action_int = self.aec_env.action_map_obj_to_int[key]
                else:
                    found_match = False
                    for template_action in legal_actions:
                        if (template_action.action_type == bot_action_obj.action_type and
                            template_action.level == bot_action_obj.level and
                            template_action.index == bot_action_obj.index and
                            template_action.is_reserved_buy == bot_action_obj.is_reserved_buy and
                            template_action.is_deck_reserve == bot_action_obj.is_deck_reserve):
                            
                            template_key = self._action_obj_to_key(template_action)
                            if template_key in self.aec_env.action_map_obj_to_int:
                                bot_action_int = self.aec_env.action_map_obj_to_int[template_key]
                                found_match = True
                                break
                    
                    if not found_match:
                        template_key = self._action_obj_to_key(legal_actions[0])
                        bot_action_int = self.aec_env.action_map_obj_to_int[template_key]

            except Exception as e:
                print(f"[Wrapper 오류] 봇 액션 변환 중 예외: {e}")
                key = self._action_obj_to_key(legal_actions[0])
                bot_action_int = self.aec_env.action_map_obj_to_int[key]

            self.aec_env.step(bot_action_int)
            
        return False 
    # --- [_run_bot_turn 끝] ---

    def step(self, action: int):
        ai_agent = "player_0"
        is_valid_action = self.current_action_mask[action] == 1
        
        final_action_to_env = action
        reward_override = None 

        if not is_valid_action:
            valid_actions_indices = np.where(self.current_action_mask == 1)[0]
            
            if len(valid_actions_indices) > 0:
                final_action_to_env = np.random.choice(valid_actions_indices)
            else:
                final_action_to_env = None 
            
            reward_override = -1.0
        
        self.aec_env.step(final_action_to_env)
        game_over = self._run_bot_turn()
        
        # --- [MaskableDQN 호환을 위한 수정된 반환] ---
        if not game_over:
            observation_array = self._get_obs(ai_agent) 
            reward = self.aec_env.rewards[ai_agent]
            termination = self.aec_env.terminations[ai_agent]
            truncation = self.aec_env.truncations[ai_agent]
            info = self.aec_env.infos[ai_agent] 
        else:
            observation_array = self._get_obs(ai_agent) 
            reward = self.aec_env.rewards[ai_agent]
            termination = True 
            truncation = True
            info = self.aec_env.infos[ai_agent] 

        if reward_override is not None:
            if termination or truncation:
                 reward = reward_override
            else:
                reward = reward_override + reward

        self.current_action_mask = info.get("action_mask")
        
        # 관찰 값을 딕셔너리 형태로 조합
        observation_dict = {
            "observation": observation_array,
            # 봇 턴 실행 후 업데이트된 AI의 마스크
            "action_mask": self.current_action_mask 
        }
        
        return observation_dict, reward, termination, truncation, info
        # --- [수정 끝] ---

    def reset(self, seed=None, options=None):
        """환경을 리셋하고 AI(player_0)의 첫 번째 관찰 값(Dict)을 반환합니다."""
        self.aec_env.reset(seed=seed)
        
        if self.aec_env.agent_selection != "player_0":
            self._run_bot_turn()
            
        # --- [수정된 부분] ---
        # 1. _get_obs 대신 observe()를 직접 호출하여 info 딕셔너리가
        #    "action_mask"로 채워지도록 보장합니다.
        observation_array = self.aec_env.observe("player_0") 
        
        # 2. observe()가 info를 채웠으므로, 이제 안전하게 가져올 수 있습니다.
        info = self.aec_env.infos["player_0"]  
        self.current_action_mask = info.get("action_mask")
        
        # 3. 마스크가 None이 아닌지 확인 (방어 코드)
        if self.current_action_mask is None:
            raise ValueError(
                "Reset failed: Action mask is None. "
                "Check splendor_aec_env.py observe() method."
            )
        # --- [수정 끝] ---

        observation_dict = {
            "observation": observation_array,
            "action_mask": self.current_action_mask
        }
        
        return observation_dict, info

    def render(self):
        self.aec_env.render()

    def close(self):
        self.aec_env.close()