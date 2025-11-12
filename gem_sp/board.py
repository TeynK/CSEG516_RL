# splendor_game/board.py

"""
Defines the Board class, which manages the shared state of the game.

This includes:
- The central gem stacks (tokens).
- The three decks of development cards.
- The face-up development cards.
- The available noble tiles.
"""

from typing import Dict, List, Optional

# Import constants and data loaders
from .constants import (
    SETUP_CONFIG,
    GemColor,
    CARD_LEVELS,
    FACE_UP_CARDS_PER_LEVEL
)
from .card import DevelopmentCard, NobleTile, load_development_cards, load_noble_tiles


class Board:
    """
    Manages all shared resources on the game board.
    """

    def __init__(self, num_players: int):
        """
        Initializes the game board based on the number of players.

        Args:
            num_players: The number of players (2, 3, or 4).
        """
        if num_players not in SETUP_CONFIG:
            raise ValueError(f"Invalid number of players: {num_players}. Must be in {list(SETUP_CONFIG.keys())}")
        
        self.num_players = num_players
        config = SETUP_CONFIG[self.num_players]

        # 1. Initialize Gem Stacks
        self.gem_stacks: Dict[GemColor, int] = {}
        standard_gem_count = config['gems']
        for color in GemColor.get_standard_gems():
            self.gem_stacks[color] = standard_gem_count
        self.gem_stacks[GemColor.GOLD] = config['gold']

        # 2. Load and set up Development Cards
        self.decks: Dict[int, List[DevelopmentCard]] = load_development_cards()
        self.face_up_cards: Dict[int, List[Optional[DevelopmentCard]]] = {}
        
        for level in CARD_LEVELS:
            self.face_up_cards[level] = []
            for _ in range(FACE_UP_CARDS_PER_LEVEL):
                # Draw a card from the deck, or None if the deck is empty
                card = self.draw_card_from_deck(level)
                self.face_up_cards[level].append(card)

        # 3. Load and set up Noble Tiles
        all_nobles = load_noble_tiles()
        num_nobles = config['nobles']
        self.nobles: List[NobleTile] = all_nobles[:num_nobles]

    def draw_card_from_deck(self, level: int) -> Optional[DevelopmentCard]:
        """
        Draws one card from the top of the specified level's deck.
        Returns None if the deck is empty.
        """
        if self.decks[level]:
            return self.decks[level].pop()
        return None

    def replace_face_up_card(self, level: int, index: int) -> None:
        """
        Replaces a taken face-up card with a new one from the deck.
        If the deck is empty, the slot becomes empty (None).
        
        Args:
            level: The card level (1, 2, or 3).
            index: The slot index (0-3) of the card to replace.
        """
        if not (0 <= index < FACE_UP_CARDS_PER_LEVEL):
            raise IndexError(f"Invalid card index: {index}. Must be 0-3.")
            
        new_card = self.draw_card_from_deck(level)
        self.face_up_cards[level][index] = new_card

    def take_gems(self, gems_to_take: Dict[GemColor, int]) -> None:
        """
        Removes the specified gems from the board's stacks.
        Assumes the action has already been validated.
        """
        for color, count in gems_to_take.items():
            if self.gem_stacks[color] < count:
                # This should ideally not be hit if actions are pre-validated
                raise ValueError(f"Not enough {color.value} gems in the stack.")
            self.gem_stacks[color] -= count

    def return_gems(self, gems_to_return: Dict[GemColor, int]) -> None:
        """
        Returns the specified gems to the board's stacks.
        (e.g., when a player has > 10 gems).
        """
        for color, count in gems_to_return.items():
            self.gem_stacks[color] += count

    def __repr__(self) -> str:
        """Provides a simple text representation of the board state."""
        rep_str = "--- Splendor Board State ---\n"
        rep_str += "Gems: " + ", ".join(f"{c.value}: {v}" for c, v in self.gem_stacks.items()) + "\n"
        rep_str += "Nobles: " + str([n.points for n in self.nobles]) + "\n"
        
        for level in sorted(self.face_up_cards.keys(), reverse=True):
            rep_str += f"Level {level} Deck: ({len(self.decks[level])} cards left)\n"
            for i, card in enumerate(self.face_up_cards[level]):
                if card:
                    rep_str += f"  [{i}]: {card}\n"
                else:
                    rep_str += f"  [{i}]: (Empty)\n"
        
        rep_str += "-----------------------------"
        return rep_str