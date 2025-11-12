# splendor_game/player.py

"""
Defines the Player class, which manages the state for a single player.

This includes:
- Gems (tokens) held by the player.
- Development cards purchased.
- Calculated bonuses from purchased cards.
- Reserved cards.
- Acquired noble tiles.
- Total score.
"""

from typing import Dict, List, Optional
from collections import defaultdict

# Import constants and data structures
from .constants import (
    GemColor,
    MAX_GEMS_PER_PLAYER,
    MAX_RESERVED_CARDS
)
from .card import DevelopmentCard, NobleTile, CostDict

class Player:
    """
    Manages the private state and resources of a single player.
    """

    def __init__(self, player_id: int):
        """
        Initializes a new player.

        Args:
            player_id: A unique identifier (e.g., 0, 1, 2, 3).
        """
        self.player_id: int = player_id

        # 1. Gem Tokens (movable)
        self.gems: CostDict = defaultdict(int)

        # 2. Cards (permanent bonuses)
        self.cards: List[DevelopmentCard] = []
        # Stored separately for fast lookups (O(1)) during cost calculation
        self.bonuses: CostDict = defaultdict(int)

        # 3. Reserved Cards
        self.reserved_cards: List[DevelopmentCard] = []

        # 4. Nobles
        self.nobles: List[NobleTile] = []

        # 5. Score
        self.score: int = 0

    def get_total_gems(self) -> int:
        """Returns the total number of gem tokens held by the player."""
        return sum(self.gems.values())
    
    def can_reserve(self) -> bool:
        """Checks if the player can reserve another card."""
        return len(self.reserved_cards) < MAX_RESERVED_CARDS
    
    def add_gems(self, gems_to_add: CostDict) -> None:
        """Adds gem tokens to the player's hand."""
        for color, count in gems_to_add.items():
            self.gems[color] += count
    
    def remove_gems(self, gems_to_remove: CostDict) -> None:
        """Removes gem tokens from the player's hand (e.g., when buying)."""
        for color, count in gems_to_remove.items():
            if self.gems[color] < count:
                raise ValueError(f"Player {self.player_id} does not have "
                                 f"enough {color.value} gems to spend.")
            self.gems[color] -= count
    
    def add_card(self, card: DevelopmentCard) -> None:
        """Adds a purchased card to the player's collection."""
        self.cards.append(card)
        # Update score
        self.score += card.points
        # Update permanent bonus
        self.bonuses[card.gem_type] += 1
    
    def add_reserved_card(self, card: DevelopmentCard) -> None:
        """Adds a card to the player's reserved list."""
        if not self.can_reserve():
            raise ValueError(f"Player {self.player_id} cannot reserve "
                             f"more than {MAX_RESERVED_CARDS} cards.")
        self.reserved_cards.append(card)

    def add_noble(self, noble: NobleTile) -> None:
        """Adds an acquired noble tile to the player's collection."""
        self.nobles.append(noble)
        self.score += noble.points

    def calculate_effective_cost(self, card: DevelopmentCard) -> CostDict:
        """
        Calculates the actual gem cost for a card, applying bonuses.
        Gold tokens are not considered here, only gem-for-gem reduction.
        """
        effective_cost = defaultdict(int)
        for color, cost in card.cost.items():
            # Reduce cost by player's bonus, but not below 0
            cost_after_bonus = max(0, cost - self.bonuses[color])
            effective_cost[color] = cost_after_bonus
        return effective_cost

    def can_afford(self, card: DevelopmentCard) -> bool:
        """
        Checks if the player can afford a card using their gems, bonuses,
        and gold tokens as wildcards.
        """
        effective_cost = self.calculate_effective_cost(card)
        
        # Calculate the remaining cost (shortfall) after applying gems
        shortfall = 0
        for color, cost in effective_cost.items():
            if self.gems[color] < cost:
                shortfall += (cost - self.gems[color])
        
        # Check if the player's gold gems can cover the shortfall
        return self.gems[GemColor.GOLD] >= shortfall
    
    def __repr__(self) -> str:
        """Provides a simple text representation of the player's state."""
        rep_str = f"--- Player {self.player_id} (Score: {self.score}) ---\n"
        rep_str += "Gems: " + ", ".join(f"{c.value}: {v}" for c, v in self.gems.items() if v > 0) + "\n"
        rep_str += "Bonuses: " + ", ".join(f"{c.value}: {v}" for c, v in self.bonuses.items() if v > 0) + "\n"
        rep_str += f"Reserved: {len(self.reserved_cards)} cards\n"
        rep_str += f"Nobles: {len(self.nobles)}\n"
        rep_str += "-----------------------------"
        return rep_str
    