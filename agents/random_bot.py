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
        if not legal_actions:
            return None
        if legal_actions[0].action_type == ActionType.RETURN_GEMS:
            return legal_actions[0]
        return random.choice(legal_actions)