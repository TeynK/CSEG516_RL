# splendor_game/card.py

"""
Defines the data structures for Development Cards and Noble Tiles.

This file uses @dataclass for clean object representation and
hard-codes all 90 development cards and 10 noble tiles,
as the game data is fixed.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Import the color definition from constants
from .constants import GemColor

# Type alias for clarity
CostDict = Dict[GemColor, int]

@dataclass(frozen=True)
class DevelopmentCard:
    """
    Represents a single Development Card.
    Instances are immutable ('frozen=True').
    """
    level: int
    points: int
    gem_type: GemColor
    cost: CostDict = field(default_factory=dict)

    def __repr__(self) -> str:
        cost_str = ", ".join(f"{c.value}: {v}" for c, v in self.cost.items())
        return f"Noble({self.points}pts, Requires:[{cost_str}])"
    
@dataclass(frozen=True)
class NobleTile:
    """
    Represents a single Noble Tile.
    Instances are immutable.
    """
    points: int
    cost: CostDict = field(default_factory=dict)  # Cost in *card bonuses*

    def __repr__(self) -> str:
        cost_str = ", ".join(f"{c.value}: {v}" for c, v in self.cost.items())
        return f"Noble({self.points}pts, Requires:[{cost_str}])"


# --- Private Hard-coded Game Data ---
# All 90 Development Cards and 10 Nobles

# Format: (points, gem_type, cost_dict)
_LEVEL_1_CARDS_DATA = [
    # Blue Bonus
    (0, GemColor.BLUE, {GemColor.WHITE: 1, GemColor.GREEN: 1, GemColor.RED: 1, GemColor.BLACK: 1}),
    (0, GemColor.BLUE, {GemColor.WHITE: 1, GemColor.GREEN: 1, GemColor.RED: 2, GemColor.BLACK: 1}),
    (0, GemColor.BLUE, {GemColor.WHITE: 2, GemColor.GREEN: 2, GemColor.BLACK: 1}),
    (0, GemColor.BLUE, {GemColor.GREEN: 1, GemColor.RED: 3, GemColor.BLACK: 1}),
    (0, GemColor.BLUE, {GemColor.WHITE: 4}),
    (1, GemColor.BLUE, {GemColor.BLUE: 2, GemColor.GREEN: 2, GemColor.RED: 3}),
    (0, GemColor.BLUE, {GemColor.WHITE: 2, GemColor.GREEN: 1}),
    (0, GemColor.BLUE, {GemColor.WHITE: 3}),
    # Green Bonus
    (0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.RED: 1, GemColor.BLACK: 1}),
    (0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.RED: 1, GemColor.BLACK: 2}),
    (0, GemColor.GREEN, {GemColor.BLUE: 2, GemColor.RED: 2, GemColor.BLACK: 1}),
    (0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 3, GemColor.BLACK: 1}),
    (0, GemColor.GREEN, {GemColor.BLUE: 4}),
    (1, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.GREEN: 2, GemColor.BLACK: 2}),
    (0, GemColor.GREEN, {GemColor.BLUE: 2, GemColor.BLACK: 1}),
    (0, GemColor.GREEN, {GemColor.BLUE: 3}),
    # Red Bonus
    (0, GemColor.RED, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.BLACK: 1}),
    (0, GemColor.RED, {GemColor.WHITE: 2, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.BLACK: 1}),
    (0, GemColor.RED, {GemColor.WHITE: 1, GemColor.GREEN: 2, GemColor.BLACK: 2}),
    (0, GemColor.RED, {GemColor.WHITE: 3, GemColor.GREEN: 1, GemColor.BLACK: 1}),
    (0, GemColor.RED, {GemColor.GREEN: 4}),
    (1, GemColor.RED, {GemColor.WHITE: 2, GemColor.RED: 3, GemColor.BLACK: 2}),
    (0, GemColor.RED, {GemColor.GREEN: 2, GemColor.BLACK: 1}),
    (0, GemColor.RED, {GemColor.GREEN: 3}),
    # Black Bonus
    (0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.RED: 1}),
    (0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.BLUE: 2, GemColor.GREEN: 1, GemColor.RED: 1}),
    (0, GemColor.BLACK, {GemColor.WHITE: 2, GemColor.BLUE: 1, GemColor.GREEN: 2}),
    (0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.RED: 3}),
    (0, GemColor.BLACK, {GemColor.RED: 4}),
    (1, GemColor.BLACK, {GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.RED: 2}),
    (0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.RED: 2}),
    (0, GemColor.BLACK, {GemColor.RED: 3}),
    # White Bonus
    (0, GemColor.WHITE, {GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.RED: 1, GemColor.BLACK: 1}),
    (0, GemColor.WHITE, {GemColor.BLUE: 1, GemColor.GREEN: 2, GemColor.RED: 1, GemColor.BLACK: 1}),
    (0, GemColor.WHITE, {GemColor.BLUE: 2, GemColor.GREEN: 1, GemColor.RED: 2}),
    (0, GemColor.WHITE, {GemColor.BLUE: 1, GemColor.GREEN: 3, GemColor.RED: 1}),
    (0, GemColor.WHITE, {GemColor.BLACK: 4}),
    (1, GemColor.WHITE, {GemColor.WHITE: 2, GemColor.BLUE: 2, GemColor.BLACK: 3}),
    (0, GemColor.WHITE, {GemColor.BLUE: 1, GemColor.GREEN: 2}),
    (0, GemColor.WHITE, {GemColor.BLACK: 3}),
]

_LEVEL_2_CARDS_DATA = [
    # Blue Bonus
    (1, GemColor.BLUE, {GemColor.BLUE: 2, GemColor.GREEN: 4, GemColor.RED: 1}),
    (2, GemColor.BLUE, {GemColor.BLUE: 5}),
    (2, GemColor.BLUE, {GemColor.WHITE: 5, GemColor.BLUE: 3}),
    (2, GemColor.BLUE, {GemColor.GREEN: 5, GemColor.RED: 3}),
    (3, GemColor.BLUE, {GemColor.BLUE: 6}),
    (1, GemColor.BLUE, {GemColor.WHITE: 3, GemColor.BLUE: 2, GemColor.BLACK: 3}),
    # Green Bonus
    (1, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 4, GemColor.GREEN: 2}),
    (2, GemColor.GREEN, {GemColor.GREEN: 5}),
    (2, GemColor.GREEN, {GemColor.GREEN: 5, GemColor.RED: 3}),
    (2, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.BLACK: 5}),
    (3, GemColor.GREEN, {GemColor.GREEN: 6}),
    (1, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.GREEN: 2, GemColor.RED: 3}),
    # Red Bonus
    (1, GemColor.RED, {GemColor.WHITE: 4, GemColor.RED: 2, GemColor.BLACK: 1}),
    (2, GemColor.RED, {GemColor.RED: 5}),
    (2, GemColor.RED, {GemColor.BLUE: 5, GemColor.RED: 3}),
    (2, GemColor.RED, {GemColor.GREEN: 3, GemColor.RED: 5}),
    (3, GemColor.RED, {GemColor.RED: 6}),
    (1, GemColor.RED, {GemColor.BLUE: 3, GemColor.RED: 2, GemColor.BLACK: 3}),
    # Black Bonus
    (1, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.RED: 4, GemColor.BLACK: 2}),
    (2, GemColor.BLACK, {GemColor.BLACK: 5}),
    (2, GemColor.BLACK, {GemColor.WHITE: 5, GemColor.BLACK: 3}),
    (2, GemColor.BLACK, {GemColor.BLUE: 3, GemColor.BLACK: 5}),
    (3, GemColor.BLACK, {GemColor.BLACK: 6}),
    (1, GemColor.BLACK, {GemColor.WHITE: 2, GemColor.GREEN: 3, GemColor.BLACK: 3}),
    # White Bonus
    (1, GemColor.WHITE, {GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.BLACK: 4}),
    (2, GemColor.WHITE, {GemColor.WHITE: 5}),
    (2, GemColor.WHITE, {GemColor.RED: 5, GemColor.BLACK: 3}),
    (2, GemColor.WHITE, {GemColor.WHITE: 5, GemColor.GREEN: 3}),
    (3, GemColor.WHITE, {GemColor.WHITE: 6}),
    (1, GemColor.WHITE, {GemColor.GREEN: 3, GemColor.RED: 2, GemColor.BLACK: 2}),
]

_LEVEL_3_CARDS_DATA = [
    # Blue Bonus
    (3, GemColor.BLUE, {GemColor.WHITE: 3, GemColor.GREEN: 3, GemColor.RED: 5, GemColor.BLACK: 3}),
    (4, GemColor.BLUE, {GemColor.WHITE: 7}),
    (4, GemColor.BLUE, {GemColor.WHITE: 6, GemColor.BLUE: 3, GemColor.BLACK: 3}),
    (5, GemColor.BLUE, {GemColor.WHITE: 7, GemColor.BLUE: 3}),
    # Green Bonus
    (3, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.RED: 3, GemColor.BLACK: 5}),
    (4, GemColor.GREEN, {GemColor.BLUE: 7}),
    (4, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.BLUE: 6, GemColor.GREEN: 3}),
    (5, GemColor.GREEN, {GemColor.BLUE: 7, GemColor.GREEN: 3}),
    # Red Bonus
    (3, GemColor.RED, {GemColor.WHITE: 5, GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.BLACK: 3}),
    (4, GemColor.RED, {GemColor.GREEN: 7}),
    (4, GemColor.RED, {GemColor.BLUE: 3, GemColor.GREEN: 6, GemColor.RED: 3}),
    (5, GemColor.RED, {GemColor.GREEN: 7, GemColor.RED: 3}),
    # Black Bonus
    (3, GemColor.BLACK, {GemColor.WHITE: 3, GemColor.BLUE: 5, GemColor.GREEN: 3, GemColor.RED: 3}),
    (4, GemColor.BLACK, {GemColor.RED: 7}),
    (4, GemColor.BLACK, {GemColor.GREEN: 3, GemColor.RED: 6, GemColor.BLACK: 3}),
    (5, GemColor.BLACK, {GemColor.RED: 7, GemColor.BLACK: 3}),
    # White Bonus
    (3, GemColor.WHITE, {GemColor.BLUE: 3, GemColor.GREEN: 5, GemColor.RED: 3, GemColor.BLACK: 3}),
    (4, GemColor.WHITE, {GemColor.BLACK: 7}),
    (4, GemColor.WHITE, {GemColor.WHITE: 3, GemColor.RED: 3, GemColor.BLACK: 6}),
    (5, GemColor.WHITE, {GemColor.WHITE: 3, GemColor.BLACK: 7}),
]

# Format: (points, cost_dict)
_NOBLES_DATA = [
    (3, {GemColor.BLUE: 4, GemColor.GREEN: 4}),
    (3, {GemColor.GREEN: 4, GemColor.RED: 4}),
    (3, {GemColor.RED: 4, GemColor.BLACK: 4}),
    (3, {GemColor.BLACK: 4, GemColor.WHITE: 4}),
    (3, {GemColor.WHITE: 4, GemColor.BLUE: 4}),
    (3, {GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.RED: 3}),
    (3, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.GREEN: 3}),
    (3, {GemColor.WHITE: 3, GemColor.RED: 3, GemColor.BLACK: 3}),
    (3, {GemColor.GREEN: 3, GemColor.RED: 3, GemColor.BLACK: 3}),
    (3, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.BLACK: 3}),
]

# Format: (points, cost_dict)
_NOBLES_DATA = [
    (3, {GemColor.BLUE: 4, GemColor.GREEN: 4}),
    (3, {GemColor.GREEN: 4, GemColor.RED: 4}),
    (3, {GemColor.RED: 4, GemColor.BLACK: 4}),
    (3, {GemColor.BLACK: 4, GemColor.WHITE: 4}),
    (3, {GemColor.WHITE: 4, GemColor.BLUE: 4}),
    (3, {GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.RED: 3}),
    (3, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.GREEN: 3}),
    (3, {GemColor.WHITE: 3, GemColor.RED: 3, GemColor.BLACK: 3}),
    (3, {GemColor.GREEN: 3, GemColor.RED: 3, GemColor.BLACK: 3}),
    (3, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.BLACK: 3}),
]

# --- Public Functions to Load Data ---

def load_development_cards() -> Dict[int, list[DevelopmentCard]]:
    """
    Parses the hard-coded data and returns a dictionary mapping
    card level to a shuffled list of DevelopmentCard objects.
    """
    decks = {}
    # Process Level 1
    deck1 = [DevelopmentCard(level=1, points=p, gem_type=g, cost=c)
             for p, g, c in _LEVEL_1_CARDS_DATA]
    random.shuffle(deck1)
    decks[1] = deck1

    # Process Level 2
    deck2 = [DevelopmentCard(level=2, points=p, gem_type=g, cost=c)
             for p, g, c in _LEVEL_2_CARDS_DATA]
    random.shuffle(deck2)
    decks[2] = deck2

    # Process Level 3
    deck3 = [DevelopmentCard(level=3, points=p, gem_type=g, cost=c)
             for p, g, c in _LEVEL_3_CARDS_DATA]
    random.shuffle(deck3)
    decks[3] = deck3
    
    return decks

def load_noble_tiles() -> List[NobleTile]:
    """
    Parses the hard-coded data and returns a shuffled list of NobleTile objects.
    """
    nobles = [NobleTile(points=p, cost=c) for p, c in _NOBLES_DATA]
    random.shuffle(nobles)
    return nobles