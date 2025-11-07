# splendor_game/actions.py

"""
Defines the Action data structure and the logic for generating
all legal actions for a given game state.

This is the core "rules engine" for determining valid moves.
"""

import itertools
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

from .constants import GemColor, CARD_LEVELS, FACE_UP_CARDS_PER_LEVEL
from .card import DevelopmentCard, CostDict
from .board import Board
from .player import Player


class ActionType(Enum):
    """Enumeration of the distinct action types a player can take."""
    TAKE_THREE_GEMS = "take_three"  # Take 3 different gems
    TAKE_TWO_GEMS = "take_two"      # Take 2 gems of the same color
    BUY_CARD = "buy_card"           # Buy a face-up or reserved card
    RESERVE_CARD = "reserve_card"   # Reserve a face-up card or from deck


@dataclass(frozen=True)
class Action:
    """
    Represents a single, unique, legal action a player can perform.
    'frozen=True' makes instances hashable, usable in sets.
    """
    action_type: ActionType

    # --- For TAKE actions ---
    # e.g., {GemColor.RED: 1, GemColor.BLUE: 1, GemColor.GREEN: 1}
    # e.g., {GemColor.WHITE: 2}
    gems: CostDict = field(default_factory=dict)

    # --- For BUY / RESERVE actions ---
    card: Optional[DevelopmentCard] = None  # The card object being bought/reserved
    level: Optional[int] = None # Level (1, 2, or 3)
    index: Optional[int] = None # Slot index (0-3 for face-up, or 0-2 for reserved)
    is_reserved_buy: bool = False # Distinguishes buying reserved vs. face-up
    is_deck_reserve: bool = False # Distinguishes reserving deck vs. face-up

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
        return "Action(Unknown)"


# --- Public Function ---

def get_legal_actions(board: Board, player: Player) -> List[Action]:
    """
    Generates a list of all possible legal actions for the current player.
    This is the main function to be called by the game engine.
    """
    # *** 수정: set() -> [] ***
    legal_actions: List[Action] = []

    # 1. Add "Take Gems" actions
    # *** 수정: .update() -> .extend() ***
    legal_actions.extend(_get_legal_gem_actions(board))

    # 2. Add "Buy Card" actions
    # *** 수정: .update() -> .extend() ***
    legal_actions.extend(_get_legal_buy_actions(board, player))

    # 3. Add "Reserve Card" actions
    # *** 수정: .update() -> .extend() ***
    legal_actions.extend(_get_legal_reserve_actions(board, player))

    return legal_actions


# --- Private Helper Functions ---

def _get_legal_gem_actions(board: Board) -> List[Action]: # *** 수정: Set -> List ***
    """Generates all valid gem-taking actions."""
    # *** 수정: set() -> [] ***
    actions: List[Action] = []
    
    # Get standard gems available in the stacks
    available_colors = [
        color for color in GemColor.get_standard_gems()
        if board.gem_stacks.get(color, 0) > 0
    ]

    # --- ActionType.TAKE_THREE_GEMS ---
    # Can take 3, 2, or 1 different gems, based on availability
    n_available = len(available_colors)
    
    if n_available >= 3:
        # Standard case: Take 3 different
        for combo in itertools.combinations(available_colors, 3):
            gems_to_take = {c: 1 for c in combo}
            # *** 수정: .add() -> .append() ***
            actions.append(Action(ActionType.TAKE_THREE_GEMS, gems=gems_to_take))
    
    elif n_available == 2:
        # Special case: Only 2 colors left, must take 2
        gems_to_take = {available_colors[0]: 1, available_colors[1]: 1}
        # *** 수정: .add() -> .append() ***
        actions.append(Action(ActionType.TAKE_THREE_GEMS, gems=gems_to_take))
        
    elif n_available == 1:
        # Special case: Only 1 color left, must take 1
        gems_to_take = {available_colors[0]: 1}
        # *** 수정: .add() -> .append() ***
        actions.append(Action(ActionType.TAKE_THREE_GEMS, gems=gems_to_take))

    # --- ActionType.TAKE_TWO_GEMS ---
    # Can take 2 of the same color if the stack has >= 4
    for color in GemColor.get_standard_gems():
        if board.gem_stacks.get(color, 0) >= 4:
            gems_to_take = {color: 2}
            # *** 수정: .add() -> .append() ***
            actions.append(Action(ActionType.TAKE_TWO_GEMS, gems=gems_to_take))

    return actions


def _get_legal_buy_actions(board: Board, player: Player) -> List[Action]: # *** 수정: Set -> List ***
    """Generates all valid card-buying actions."""
    # *** 수정: set() -> [] ***
    actions: List[Action] = []

    # 1. Check face-up cards on the board
    for level in CARD_LEVELS:
        for index, card in enumerate(board.face_up_cards[level]):
            if card is not None and player.can_afford(card):
                # *** 수정: .add() -> .append() ***
                actions.append(Action(
                    action_type=ActionType.BUY_CARD,
                    card=card,
                    level=level,
                    index=index,
                    is_reserved_buy=False
                ))

    # 2. Check player's reserved cards
    for index, card in enumerate(player.reserved_cards):
        if player.can_afford(card):
            # *** 수정: .add() -> .append() ***
            actions.append(Action(
                action_type=ActionType.BUY_CARD,
                card=card,
                index=index,
                is_reserved_buy=True
            ))
            
    return actions


def _get_legal_reserve_actions(board: Board, player: Player) -> List[Action]: # *** 수정: Set -> List ***
    """Generates all valid card-reserving actions."""
    # *** 수정: set() -> [] ***
    actions: List[Action] = []

    # Check if player has space to reserve
    if not player.can_reserve():
        return actions  # Return empty list

    # 1. Check face-up cards on the board
    for level in CARD_LEVELS:
        for index, card in enumerate(board.face_up_cards[level]):
            if card is not None:
                # *** 수정: .add() -> .append() ***
                actions.append(Action(
                    action_type=ActionType.RESERVE_CARD,
                    card=card,
                    level=level,
                    index=index,
                    is_deck_reserve=False
                ))

    # 2. Check decks (if not empty)
    for level in CARD_LEVELS:
        if board.decks[level]:  # Check if deck has cards
            # *** 수정: .add() -> .append() ***
            actions.append(Action(
                action_type=ActionType.RESERVE_CARD,
                level=level,
                is_deck_reserve=True
            ))

    return actions