# splendor_game/actions.py

import itertools
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
from collections import Counter

from .constants import GemColor, CARD_LEVELS, MAX_GEMS_PER_PLAYER
from .card import DevelopmentCard, CostDict
from .board import Board
from .player import Player

class ActionType(Enum):
    TAKE_THREE_GEMS = "take_three"
    TAKE_TWO_GEMS = "take_two"
    BUY_CARD = "buy_card"
    RESERVE_CARD = "reserve_card"
    RETURN_GEMS = "return_gems"

@dataclass(frozen=True)
class Action:
    action_type: ActionType
    gems: CostDict = field(default_factory=dict)
    card: Optional[DevelopmentCard] = None
    level: Optional[int] = None
    index: Optional[int] = None
    is_reserved_buy: bool = False
    is_deck_reserve: bool = False

    def __repr__(self) -> str:
        """Provides a human-readable representation for debugging."""
        if self.action_type == ActionType.TAKE_THREE_GEMS:
            colors = [g.value for g in self.gems.keys()]
            return f"Action(TAKE_THREE: {', '.join(colors)})"
        if self.action_type == ActionType.TAKE_TWO_GEMS:
            color = list(self.gems.keys())[0].value
            return f"Action(TAKE_TWO: {color})"
        if self.action_type == ActionType.BUY_CARD:
            if self.is_reserved_buy:
                return f"Action(BUY_RESERVED: Card at index {self.index})"
            return f"Action(BUY: L{self.level}, Idx{self.index})"
        if self.action_type == ActionType.RESERVE_CARD:
            if self.is_deck_reserve:
                return f"Action(RESERVE_DECK: L{self.level})"
            return f"Action(RESERVE: L{self.level}, Idx{self.index})"
        if self.action_type == ActionType.RETURN_GEMS:
            gems_str = ", ".join(f"{c.value}: {v}" for c, v in self.gems.items() if v > 0)
            return f"Action(RETURN_GEMS: {gems_str})"
        return "Action(Unknown)"

def get_legal_actions(board: Board, player: Player) -> List[Action]:
    legal_actions: List[Action] = []
    legal_actions.extend(get_legal_gem_actions(board))
    legal_actions.extend(get_legal_buy_actions(board, player))
    legal_actions.extend(get_legal_reserve_actions(board, player))
    return legal_actions

def get_legal_gem_actions(board: Board) -> List[Action]:
    actions: List[Action] = []

    available_colors = [color for color in GemColor.get_standard_gems() if board.gem_stacks.get(color, 0) > 0]
    n_available = len(available_colors)
    if n_available >= 3:
        for combo in itertools.combinations(available_colors, 3):
            gems_to_take = {c: 1 for c in combo}
            actions.append(Action(ActionType.TAKE_THREE_GEMS, gems=gems_to_take))
    elif n_available == 2:
        gems_to_take = {available_colors[0]: 1, available_colors[1]: 1}
        actions.append(Action(ActionType.TAKE_THREE_GEMS, gems=gems_to_take))
    elif n_available == 1:
        gems_to_take = {available_colors[0]: 1}
        actions.append(Action(ActionType.TAKE_THREE_GEMS, gems=gems_to_take))
    for color in GemColor.get_standard_gems():
        if board.gem_stacks.get(color, 0) >= 4:
            gems_to_take = {color: 2}
            actions.append(Action(ActionType.TAKE_TWO_GEMS, gems=gems_to_take))
    return actions

def get_legal_buy_actions(board: Board, player: Player) -> List[Action]:
    actions: List[Action] = []
    for level in CARD_LEVELS:
        for index, card in enumerate(board.face_up_cards[level]):
            if card is not None and player.can_afford(card):
                actions.append(Action(action_type=ActionType.BUY_CARD, level=level, index=index, is_reserved_buy=False))
    for index, card in enumerate(player.reserved_cards):
        if player.can_afford(card):
            actions.append(Action(action_type=ActionType.BUY_CARD, index=index, is_reserved_buy=True))
    return actions

def get_legal_reserve_actions(board: Board, player: Player) -> List[Action]:
    actions: List[Action] = []
    if not player.can_reserve():
        return actions
    for level in CARD_LEVELS:
        for index, card in enumerate(board.face_up_cards[level]):
            if card is not None:
                actions.append(Action(action_type=ActionType.RESERVE_CARD, level=level, index=index, is_deck_reserve=False))
    for level in CARD_LEVELS:
        if board.decks[level]:
            actions.append(Action(action_type=ActionType.RESERVE_CARD, level=level, is_deck_reserve=True))
    return actions

def get_legal_return_gems_actions(player: Player) -> List[Action]:
    total_gems = player.get_total_gems()
    gems_to_return_count = total_gems - MAX_GEMS_PER_PLAYER
    if gems_to_return_count <= 0:
        return []
    actions: List[Action] = []
    gem_pool = []
    for color, count in player.gems.items():
        if count > 0:
            gem_pool.extend([color] * count)
    unique_gem_combos = set(itertools.combinations(gem_pool, gems_to_return_count))
    for combo in unique_gem_combos:
        gems_to_return_dict = dict(Counter(combo))
        actions.append(Action(action_type=ActionType.RETURN_GEMS, gems=gems_to_return_dict))
    return actions