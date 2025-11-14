import random
from typing import Optional, List

from splendor_game.board import Board
from splendor_game.player import Player
from splendor_game.actions import Action, ActionType

class RandomBot:
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.name = f"RandomBot (Player {player_id})"

    def choose_action(self, board: Board, player: Player, legal_actions: List[Action]) -> Optional[Action]:
        """
        주어진 합법적인 행동 리스트에서 무작위로 하나를 선택합니다.
        """
        # 1. 행동이 없으면 None 반환
        if not legal_actions:
            return None
            
        # 2. 보석 반납은 유일한 행동이므로 즉시 반환
        if legal_actions[0].action_type == ActionType.RETURN_GEMS:
            return legal_actions[0]

        # 3. 보석 반납이 아닌 경우, 합법적인 모든 행동 중 하나를 무작위로 선택
        return random.choice(legal_actions)