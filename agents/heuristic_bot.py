from typing import Optional, List, Tuple
from collections import defaultdict

from splendor_game.board import Board
from splendor_game.player import Player
from splendor_game.actions import Action, ActionType
from splendor_game.card import DevelopmentCard, GemColor, CostDict
from splendor_game.constants import WINNING_SCORE

class HeuristicBot:
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.target_card: Optional[DevelopmentCard] = None
        self.name = f"HeuristicBot (Player {player_id})"

    def choose_action(self, board: Board, player: Player, legal_actions: List[Action]) -> Optional[Action]:
        if not legal_actions:
            return None
            
        if legal_actions[0].action_type == ActionType.RETURN_GEMS:
            return legal_actions[0]
        action = self._find_winning_move(board, player)
        if action:
            return action
        action = self._buy_reserved_target(player)
        if action:
            return action
        if self.target_card:
            action = self._gather_for_target(board, player, legal_actions)
            if action:
                return action
        action = self._set_and_reserve_target(board, player)
        if action:
            return action
        best_buy_action = None
        lowest_cost = 99
        
        all_buyable_cards_with_actions = self._get_all_buyable_cards(board, player)
        
        for card, action in all_buyable_cards_with_actions:
            if card != self.target_card:
                cost = sum(card.cost.values())
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_buy_action = action
                        
        if best_buy_action:
            return best_buy_action
            
        if player.can_reserve():
            best_reserve_action = None
            lowest_cost = 99
            
            for i, card in enumerate(board.face_up_cards[1]):
                if card:
                    cost = sum(card.cost.values())
                    if cost < lowest_cost:
                        lowest_cost = cost
                        best_reserve_action = Action(
                            action_type=ActionType.RESERVE_CARD,
                            card=card,
                            level=1,
                            index=i,
                            is_deck_reserve=False
                        )
            
            if best_reserve_action:
                return best_reserve_action

        for action in legal_actions:
            if action.action_type == ActionType.TAKE_THREE_GEMS:
                return action

        for action in legal_actions:
            if action.action_type == ActionType.TAKE_TWO_GEMS:
                return action
                
        for action in legal_actions:
            if action.action_type == ActionType.RESERVE_CARD and action.is_deck_reserve:
                 return action

        return legal_actions[0] if legal_actions else None

    def _get_all_buyable_cards(self, board: Board, player: Player) -> List[Tuple[DevelopmentCard, Action]]:
        buyable = []
        
        for i, card in enumerate(player.reserved_cards):
            if player.can_afford(card):
                action = Action(ActionType.BUY_CARD, card=card, index=i, is_reserved_buy=True)
                buyable.append((card, action))
                
        for level in board.face_up_cards:
            for i, card in enumerate(board.face_up_cards[level]):
                if card and player.can_afford(card):
                    action = Action(ActionType.BUY_CARD, card=card, level=level, index=i, is_reserved_buy=False)
                    buyable.append((card, action))
        return buyable

    def _find_winning_move(self, board: Board, player: Player) -> Optional[Action]:
        all_buyable = self._get_all_buyable_cards(board, player)
        for card, action in all_buyable:
            if (player.score + card.points) >= WINNING_SCORE:
                return action
        return None

    def _buy_reserved_target(self, player: Player) -> Optional[Action]:
        if not self.target_card:
            return None

        for i, card in enumerate(player.reserved_cards):
            if card == self.target_card:
                if player.can_afford(card):
                    self.target_card = None
                    return Action(ActionType.BUY_CARD, card=card, index=i, is_reserved_buy=True)
                else:
                    return None
        
        self.target_card = None 
        return None

    def _calculate_cost_score(self, card: DevelopmentCard) -> float:
        cost_sum = sum(card.cost.values())
        return card.points / (cost_sum + 0.1) 

    def _set_and_reserve_target(self, board: Board, player: Player) -> Optional[Action]:
        if self.target_card or not player.can_reserve():
            return None

        best_card: Optional[DevelopmentCard] = None
        best_index: int = -1
        best_level: int = -1
        best_score: float = -1.0

        for i, card in enumerate(board.face_up_cards[3]):
            if card:
                effective_cost_dict = player.calculate_effective_cost(card)
                total_effective_cost = sum(effective_cost_dict.values())
                
                if total_effective_cost < 10:
                    score = self._calculate_cost_score(card)
                    if score > best_score:
                        best_score = score
                        best_card = card
                        best_index = i
                        best_level = 3
        
        if best_card is None:
            best_score = -1.0
            for i, card in enumerate(board.face_up_cards[2]):
                if card:
                    effective_cost_dict = player.calculate_effective_cost(card)
                    total_effective_cost = sum(effective_cost_dict.values())
                    
                    if total_effective_cost < 8:
                        score = self._calculate_cost_score(card)
                        if score > best_score:
                            best_score = score
                            best_card = card
                            best_index = i
                            best_level = 2

        if best_card is None:
            best_score = -1.0
            lowest_cost = 99
            
            for i, card in enumerate(board.face_up_cards[1]):
                if card:
                    cost = sum(card.cost.values())
                    if cost < lowest_cost:
                        lowest_cost = cost
                        best_card = card
                        best_index = i
                        best_level = 1

        if best_card:
            self.target_card = best_card
            return Action(ActionType.RESERVE_CARD, card=best_card, level=best_level, index=best_index, is_deck_reserve=False)
        
        return None
    
    def _get_needed_gems(self, player: Player) -> CostDict:
        needed_gems = defaultdict(int)
        if not self.target_card:
            return needed_gems

        effective_cost = player.calculate_effective_cost(self.target_card)
        
        for color, cost in effective_cost.items():
            current_gems = player.gems.get(color, 0)
            if current_gems < cost:
                needed_gems[color] = cost - current_gems
                
        return needed_gems

    def _gather_for_target(self, board: Board, player: Player, legal_actions: List[Action]) -> Optional[Action]:
        if not self.target_card:
            return None

        needed_gems = self._get_needed_gems(player)
        if not needed_gems:
            return self._buy_reserved_target(player)

        sorted_needed = sorted(needed_gems.items(), key=lambda item: item[1], reverse=True)
        most_needed_color, _ = sorted_needed[0]

        if board.gem_stacks.get(most_needed_color, 0) >= 4:
            return Action(ActionType.TAKE_TWO_GEMS, gems={most_needed_color: 2})

        needed_bonus_color = most_needed_color
        for i, card in enumerate(board.face_up_cards[1]):
            if card and card.gem_type == needed_bonus_color and player.can_afford(card):
                return Action(ActionType.BUY_CARD, card=card, level=1, index=i, is_reserved_buy=False)

        colors_to_take = []
        for color, _ in sorted_needed:
            if board.gem_stacks.get(color, 0) > 0:
                colors_to_take.append(color)
                
        if len(colors_to_take) >= 3:
            gem_dict = {colors_to_take[0]: 1, colors_to_take[1]: 1, colors_to_take[2]: 1}
            return Action(ActionType.TAKE_THREE_GEMS, gems=gem_dict)
        elif 0 < len(colors_to_take) < 3:
            gem_dict = {c: 1 for c in colors_to_take}
            available_gems = GemColor.get_standard_gems()
            for color in available_gems:
                if color not in gem_dict and board.gem_stacks.get(color, 0) > 0:
                    gem_dict[color] = 1
                    if len(gem_dict) == 3:
                        break
            
            if len(gem_dict) >= 1: 
                 return Action(ActionType.TAKE_THREE_GEMS, gems=gem_dict)
        
        return None