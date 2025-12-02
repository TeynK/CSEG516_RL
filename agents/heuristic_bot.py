import random
from typing import List, Optional, Dict
from splendor_game.board import Board
from splendor_game.player import Player
from splendor_game.card import DevelopmentCard
from splendor_game.actions import Action, ActionType
from splendor_game.constants import GemColor, CARD_LEVELS

class HeuristicBot:
    def __init__(self, player_id: int, style: str = "balanced"):
        self.player_id = player_id
        self.style = style
        self.set_weights(style)

    def set_weights(self, style: str):
        if style == "aggressive":
            self.w_points = 20.0
            self.w_development = 5.0
            self.w_defense = 0.5
            self.w_reserve = 1.0
        elif style == "defensive":
            self.w_points = 15.0
            self.w_development = 4.0
            self.w_defense = 10.0
            self.w_reserve = 5.0
        else:
            self.w_points = 18.0
            self.w_development = 4.5
            self.w_defense = 3.0
            self.w_reserve = 2.0

    def choose_action(self, board: Board, player: Player, legal_actions: List[Action]) -> Optional[Action]:
        if not legal_actions:
            return None
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
            card = self._get_card_from_action(action, board, player)
            if card:
                score += card.points * self.w_points
                score += self.w_development
                for noble in board.nobles:
                    if card.gem_type in noble.cost:
                        if player.bonuses[card.gem_type] < noble.cost[card.gem_type]:
                            score += 5.0
                score += 100.0 
        elif action.action_type == ActionType.RESERVE_CARD:
            card = self._get_card_from_action(action, board, player)
            if card:
                my_utility = (card.points * self.w_points) + self.w_development
                defense_score = 0.0
                if not action.is_deck_reserve and card.points >= 2:
                    defense_score = card.points * self.w_defense
                score += my_utility * 0.3
                score += defense_score
                score += self.w_reserve
                if board.gem_stacks[GemColor.GOLD] > 0:
                    score += 5.0
        elif action.action_type in [ActionType.TAKE_THREE_GEMS, ActionType.TAKE_TWO_GEMS]:
            gems_acquired = action.gems
            target_card = self._find_best_target_card(board, player)
            if target_card:
                needed_colors = []
                for color, cost in target_card.cost.items():
                    effective_cost = max(0, cost - player.bonuses[color])
                    if player.gems[color] < effective_cost:
                        needed_colors.append(color)
                match_count = sum(1 for color in gems_acquired if color in needed_colors)
                score += match_count * 10.0
                if action.action_type == ActionType.TAKE_TWO_GEMS and match_count > 0:
                    score += 5.0
            else:
                score += sum(gems_acquired.values()) * 2.0
            current_gems = sum(player.gems.values())
            taken_gems = sum(gems_acquired.values())
            if current_gems + taken_gems > 10:
                score -= 20.0 
        elif action.action_type == ActionType.RETURN_GEMS:
            score -= 100.0
            for color, count in action.gems.items():
                if player.gems[color] > 3:
                    score += 5.0
        return score

    def _get_card_from_action(self, action: Action, board: Board, player: Player) -> Optional[DevelopmentCard]:
        if action.card:
            return action.card
        if action.action_type == ActionType.BUY_CARD and action.is_reserved_buy:
            if action.index < len(player.reserved_cards):
                return player.reserved_cards[action.index]
            return None
        if action.level is not None and action.index is not None:
            if action.action_type == ActionType.RESERVE_CARD and action.is_deck_reserve:
                return None
            if 0 <= action.index < len(board.face_up_cards[action.level]):
                return board.face_up_cards[action.level][action.index]
        return None

    def _find_best_target_card(self, board: Board, player: Player) -> Optional[DevelopmentCard]:
        candidates = []
        for level in CARD_LEVELS:
            for card in board.face_up_cards[level]:
                if card: candidates.append(card)
        candidates.extend(player.reserved_cards)
        best_card = None
        max_roi = -float('inf')
        for card in candidates:
            effective_cost = player.calculate_effective_cost(card)
            shortfall = 0
            for color, cost in effective_cost.items():
                shortfall += max(0, cost - player.gems[color])
            if shortfall > 5:
                continue
            roi = (card.points * 10) + 3 - (shortfall * 2)
            if roi > max_roi:
                max_roi = roi
                best_card = card
        return best_card