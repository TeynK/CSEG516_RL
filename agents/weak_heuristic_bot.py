import random
from typing import List, Optional
from splendor_game.board import Board
from splendor_game.player import Player
from splendor_game.card import DevelopmentCard
from splendor_game.actions import Action, ActionType
from splendor_game.constants import GemColor

class WeakHeuristicBot:
    def __init__(self, player_id: int, mistake_prob: float = 0.3):
        self.player_id = player_id
        self.mistake_prob = mistake_prob

    def choose_action(self, board: Board, player: Player, legal_actions: List[Action]) -> Optional[Action]:
        if not legal_actions:
            return None
        
        if random.random() < self.mistake_prob:
            return random.choice(legal_actions)

        best_score = -float('inf')
        best_action = None
        random.shuffle(legal_actions)

        for action in legal_actions:
            score = self.evaluate_action(action, board, player)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def evaluate_action(self, action: Action, board: Board, player: Player) -> float:
        score = 0.0

        if action.action_type == ActionType.BUY_CARD:
            score += 50.0
            card = self._get_card(action, board, player)
            if card:
                score += card.points * 10.0
        
        elif action.action_type == ActionType.RESERVE_CARD:
            score += 5.0
            if board.gem_stacks[GemColor.GOLD] > 0:
                score += 5.0
        
        elif action.action_type in [ActionType.TAKE_THREE_GEMS, ActionType.TAKE_TWO_GEMS]:
            score += sum(action.gems.values()) * 2.0
            total_gems = sum(player.gems.values()) + sum(action.gems.values())
            if total_gems > 10:
                score -= 10.0

        elif action.action_type == ActionType.RETURN_GEMS:
            score -= 20.0

        return score

    def _get_card(self, action: Action, board: Board, player: Player) -> Optional[DevelopmentCard]:
        if action.card:
            return action.card
        if action.action_type == ActionType.BUY_CARD and action.is_reserved_buy:
            if action.index < len(player.reserved_cards):
                return player.reserved_cards[action.index]
        if action.level is not None and action.index is not None:
            if action.action_type == ActionType.RESERVE_CARD and action.is_deck_reserve:
                return None
            if 0 <= action.index < len(board.face_up_cards[action.level]):
                return board.face_up_cards[action.level][action.index]
        return None