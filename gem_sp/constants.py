# splendor_game/constants.py

"""
Global Constants for the Splendor Game Logic

This file defines all the 'magic numbers' and configuration settings
for the game rules. All other modules should import these constants
instead of hard-coding values.
"""

from enum import Enum

# --- Game Setup ---

# Player configuration
MIN_PLAYERS = 2
MAX_PLAYERS = 4

# Board setup based on player count
# (Num Players) -> (Nobles, Gold Gems, Other Gems)
SETUP_CONFIG = {
    2: {'nobles': 3, 'gold': 5, 'gems': 4},
    3: {'nobles': 4, 'gold': 5, 'gems': 5},
    4: {'nobles': 5, 'gold': 5, 'gems': 7},
}

# --- Card and Noble Constants ---

# Development Card Levels
CARD_LEVELS = (1, 2, 3)
FACE_UP_CARDS_PER_LEVEL = 4

# --- Player State ---

# Maximum number of gems a player can hold at the end of their turn
MAX_GEMS_PER_PLAYER = 10

# Maximum number of cards a player can reserve
MAX_RESERVED_CARDS = 3

# --- Win Condition ---

# The score a player must reach to trigger the end game
WINNING_SCORE = 15

# --- Gem (Color) Definitions ---

class GemColor(Enum):
    """Enumeration of all gem colors, including Gold."""
    # Standard gems
    WHITE = "white"
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    BLACK = "black"
    # Wildcard gem
    GOLD = "gold"

    @classmethod
    def get_standard_gems(cls):
        """Returns a list of all standard (non-gold) gem colors."""
        return [cls.WHITE, cls.BLUE, cls.GREEN, cls.RED, cls.BLACK]
    
    @classmethod
    def get_all_gems(cls):
        """Returns a list of all gem colors, including gold."""
        return [cls.WHITE, cls.BLUE, cls.GREEN, cls.RED, cls.BLACK, cls.GOLD]