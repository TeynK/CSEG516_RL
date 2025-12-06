import gymnasium as gym
from gymnasium.spaces import Dict, Box 
import numpy as np
from pettingzoo.utils.env import AECEnv
from typing import Any
import random

from envs.splendor_aec_env import env as splendor_aec_env
from agents.heuristic_bot import HeuristicBot
from agents.random_bot import RandomBot
from agents.weak_heuristic_bot import WeakHeuristicBot
from splendor_game.actions import get_legal_return_gems_actions
from splendor_game.actions import Action, ActionType 

class SplendorGymWrapper(gym.Env):    
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=2):
        super().__init__()
        self.aec_env: AECEnv = splendor_aec_env(num_players=num_players)
        assert num_players == 2, "이 래퍼는 2인용(AI vs Bot) 전용입니다."
        
        original_obs_space = self.aec_env.observation_space("player_0")
        action_mask_space = Box(
            low=0, 
            high=1, 
            shape=(self.aec_env.TOTAL_ACTIONS,), 
            dtype=np.int8
        )
        self.action_space = self.aec_env.action_space("player_0")
        self.observation_space = Dict({
            "observation": original_obs_space,
            "action_mask": action_mask_space
        })
        
        self.bot = HeuristicBot(player_id=1, style=random.choice(["aggressive", "balanced", "defensive"]))
        # self.bot = WeakHeuristicBot(player_id=1)
        # self.bot = RandomBot(player_id=1)
        self.agents = ["player_0", "player_1"]
        self.current_action_mask = None

        self.turns = 0

    def action_mask(self) -> np.ndarray:
        if self.current_action_mask is None:
            return np.ones(self.aec_env.TOTAL_ACTIONS, dtype=np.int8)
        return self.current_action_mask

    def _get_obs(self, agent_id: str) -> np.ndarray:
        return self.aec_env.observe(agent_id)
    
    def _action_obj_to_key(self, action: Action) -> Any:
        key: Any = None
        if action.action_type == ActionType.TAKE_THREE_GEMS:
            key = (ActionType.TAKE_THREE_GEMS, frozenset(action.gems.keys()))
        elif action.action_type == ActionType.TAKE_TWO_GEMS:
            if not action.gems: return None 
            key = (ActionType.TAKE_TWO_GEMS, list(action.gems.keys())[0])
        elif action.action_type == ActionType.BUY_CARD:
            key = (ActionType.BUY_CARD, action.level, action.index, action.is_reserved_buy)
        elif action.action_type == ActionType.RESERVE_CARD:
            key = (ActionType.RESERVE_CARD, action.level, action.index, action.is_deck_reserve)
        elif action.action_type == ActionType.RETURN_GEMS:
            key = (ActionType.RETURN_GEMS, frozenset(action.gems.items()))
        return key

    def _run_bot_turn(self):
        bot_agent = "player_1"
        while self.aec_env.agent_selection == bot_agent:
            termination = self.aec_env.terminations[bot_agent]
            truncation = self.aec_env.truncations[bot_agent]
            if termination or truncation: return True 
            
            current_player_obj = self.aec_env.game.players[1]
            if self.aec_env.game.current_player_state == "RETURN_GEMS":
                legal_actions = get_legal_return_gems_actions(current_player_obj)
            else:
                legal_actions = self.aec_env.game.get_legal_actions()
            
            if not legal_actions:
                self.aec_env.step(None) 
                continue

            bot_action_obj = self.bot.choose_action(self.aec_env.game.board, current_player_obj, legal_actions)
            if bot_action_obj is None: bot_action_obj = legal_actions[0]
            
            bot_action_int = -1
            try:
                key = self._action_obj_to_key(bot_action_obj)
                if key in self.aec_env.action_map_obj_to_int:
                    bot_action_int = self.aec_env.action_map_obj_to_int[key]
                else:
                    tk = self._action_obj_to_key(legal_actions[0])
                    bot_action_int = self.aec_env.action_map_obj_to_int[tk]
            except:
                tk = self._action_obj_to_key(legal_actions[0])
                bot_action_int = self.aec_env.action_map_obj_to_int[tk]

            self.aec_env.step(bot_action_int)  
        return False 

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
        self.turns += 1

        observation_array = self._get_obs(ai_agent) 
        reward = self.aec_env.rewards[ai_agent]
        termination = self.aec_env.terminations[ai_agent] or game_over
        truncation = self.aec_env.truncations[ai_agent]
        info = self.aec_env.infos[ai_agent] 

        if reward_override is not None:
            if termination or truncation: reward = reward_override
            else: reward = reward_override + reward

        self.current_action_mask = info.get("action_mask")
        
        if termination or truncation:
            game = self.aec_env.game
            agent_player = game.players[0]
            
            cards = agent_player.cards
            tier_1_cnt = sum(1 for c in cards if c.level == 1)
            tier_2_cnt = sum(1 for c in cards if c.level == 2)
            tier_3_cnt = sum(1 for c in cards if c.level == 3)
            
            info["agent_stats"] = {
                "score": agent_player.score,
                "total_cards": len(cards),
                "tier_1_count": tier_1_cnt,
                "tier_2_count": tier_2_cnt,
                "tier_3_count": tier_3_cnt,
                "reserved_count": len(agent_player.reserved_cards),
                "turn_count": self.turns,
                "noble_count": len(agent_player.nobles)
            }

        observation_dict = {
            "observation": observation_array,
            "action_mask": self.current_action_mask 
        }
        
        return observation_dict, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        self.turns = 0 # 턴 초기화
        new_style = random.choice(["aggressive", "balanced", "defensive"])
        self.bot = HeuristicBot(player_id=1, style=new_style)
        self.aec_env.reset(seed=seed)
        if self.aec_env.agent_selection != "player_0":
            self._run_bot_turn()
        
        observation_array = self.aec_env.observe("player_0") 
        info = self.aec_env.infos["player_0"]  
        self.current_action_mask = info.get("action_mask")
        
        observation_dict = {
            "observation": observation_array,
            "action_mask": self.current_action_mask
        }
        return observation_dict, info

    def render(self):
        self.aec_env.render()

    def close(self):
        self.aec_env.close()