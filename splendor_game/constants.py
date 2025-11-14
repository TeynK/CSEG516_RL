from enum import Enum

MIN_PLAYERS = 2
MAX_PLAYERS = 4

SETUP_CONFIG = {
    2: {'nobles': 3, 'gold': 5, 'gems': 4},
    3: {'nobles': 4, 'gold': 5, 'gems': 5},
    4: {'nobles': 5, 'gold': 5, 'gems': 7},
}

CARD_LEVELS = (1, 2, 3)
FACE_UP_CARDS_PER_LEVEL = 4

MAX_GEMS_PER_PLAYER = 10
MAX_RESERVED_CARDS = 3

WINNING_SCORE = 15

class GemColor(Enum):
    WHITE = "white"
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    BLACK = "black"
    GOLD = "gold"

    @classmethod
    def get_standard_gems(cls):
        return [cls.WHITE, cls.BLUE, cls.GREEN, cls.RED, cls.BLACK]

    @classmethod
    def get_all_gems(cls):
        return [cls.WHITE, cls.BLUE, cls.GREEN, cls.RED, cls.BLACK, cls.GOLD]
    
_LEVEL_1_CARDS_DATA = [
    
    (0, GemColor.BLUE, {GemColor.WHITE: 1, GemColor.GREEN: 1, GemColor.RED: 1, GemColor.BLACK: 1}),
    (0, GemColor.BLUE, {GemColor.WHITE: 1, GemColor.GREEN: 1, GemColor.RED: 2, GemColor.BLACK: 1}),
    (0, GemColor.BLUE, {GemColor.WHITE: 2, GemColor.GREEN: 2, GemColor.BLACK: 1}),
    (0, GemColor.BLUE, {GemColor.GREEN: 1, GemColor.RED: 3, GemColor.BLACK: 1}),
    (0, GemColor.BLUE, {GemColor.WHITE: 4}),
    (1, GemColor.BLUE, {GemColor.BLUE: 2, GemColor.GREEN: 2, GemColor.RED: 3}),
    (0, GemColor.BLUE, {GemColor.WHITE: 2, GemColor.GREEN: 1}),
    (0, GemColor.BLUE, {GemColor.WHITE: 3}),
    
    (0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.RED: 1, GemColor.BLACK: 1}),
    (0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.RED: 1, GemColor.BLACK: 2}),
    (0, GemColor.GREEN, {GemColor.BLUE: 2, GemColor.RED: 2, GemColor.BLACK: 1}),
    (0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 3, GemColor.BLACK: 1}),
    (0, GemColor.GREEN, {GemColor.BLUE: 4}),
    (1, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.GREEN: 2, GemColor.BLACK: 2}),
    (0, GemColor.GREEN, {GemColor.BLUE: 2, GemColor.BLACK: 1}),
    (0, GemColor.GREEN, {GemColor.BLUE: 3}),

    (0, GemColor.RED, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.BLACK: 1}),
    (0, GemColor.RED, {GemColor.WHITE: 2, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.BLACK: 1}),
    (0, GemColor.RED, {GemColor.WHITE: 1, GemColor.GREEN: 2, GemColor.BLACK: 2}),
    (0, GemColor.RED, {GemColor.WHITE: 3, GemColor.GREEN: 1, GemColor.BLACK: 1}),
    (0, GemColor.RED, {GemColor.GREEN: 4}),
    (1, GemColor.RED, {GemColor.WHITE: 2, GemColor.RED: 3, GemColor.BLACK: 2}),
    (0, GemColor.RED, {GemColor.GREEN: 2, GemColor.BLACK: 1}),
    (0, GemColor.RED, {GemColor.GREEN: 3}),
    
    (0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.RED: 1}),
    (0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.BLUE: 2, GemColor.GREEN: 1, GemColor.RED: 1}),
    (0, GemColor.BLACK, {GemColor.WHITE: 2, GemColor.BLUE: 1, GemColor.GREEN: 2}),
    (0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.RED: 3}),
    (0, GemColor.BLACK, {GemColor.RED: 4}),
    (1, GemColor.BLACK, {GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.RED: 2}),
    (0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.RED: 2}),
    (0, GemColor.BLACK, {GemColor.RED: 3}),
    
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
    
    (1, GemColor.BLUE, {GemColor.BLUE: 2, GemColor.GREEN: 4, GemColor.RED: 1}),
    (2, GemColor.BLUE, {GemColor.BLUE: 5}),
    (2, GemColor.BLUE, {GemColor.WHITE: 5, GemColor.BLUE: 3}),
    (2, GemColor.BLUE, {GemColor.GREEN: 5, GemColor.RED: 3}),
    (3, GemColor.BLUE, {GemColor.BLUE: 6}),
    (1, GemColor.BLUE, {GemColor.WHITE: 3, GemColor.BLUE: 2, GemColor.BLACK: 3}),
    
    (1, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 4, GemColor.GREEN: 2}),
    (2, GemColor.GREEN, {GemColor.GREEN: 5}),
    (2, GemColor.GREEN, {GemColor.GREEN: 5, GemColor.RED: 3}),
    (2, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.BLACK: 5}),
    (3, GemColor.GREEN, {GemColor.GREEN: 6}),
    (1, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.GREEN: 2, GemColor.RED: 3}),

    (1, GemColor.RED, {GemColor.WHITE: 4, GemColor.RED: 2, GemColor.BLACK: 1}),
    (2, GemColor.RED, {GemColor.RED: 5}),
    (2, GemColor.RED, {GemColor.BLUE: 5, GemColor.RED: 3}),
    (2, GemColor.RED, {GemColor.GREEN: 3, GemColor.RED: 5}),
    (3, GemColor.RED, {GemColor.RED: 6}),
    (1, GemColor.RED, {GemColor.BLUE: 3, GemColor.RED: 2, GemColor.BLACK: 3}),
    
    (1, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.RED: 4, GemColor.BLACK: 2}),
    (2, GemColor.BLACK, {GemColor.BLACK: 5}),
    (2, GemColor.BLACK, {GemColor.WHITE: 5, GemColor.BLACK: 3}),
    (2, GemColor.BLACK, {GemColor.BLUE: 3, GemColor.BLACK: 5}),
    (3, GemColor.BLACK, {GemColor.BLACK: 6}),
    (1, GemColor.BLACK, {GemColor.WHITE: 2, GemColor.GREEN: 3, GemColor.BLACK: 3}),
    
    (1, GemColor.WHITE, {GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.BLACK: 4}),
    (2, GemColor.WHITE, {GemColor.WHITE: 5}),
    (2, GemColor.WHITE, {GemColor.RED: 5, GemColor.BLACK: 3}),
    (2, GemColor.WHITE, {GemColor.WHITE: 5, GemColor.GREEN: 3}),
    (3, GemColor.WHITE, {GemColor.WHITE: 6}),
    (1, GemColor.WHITE, {GemColor.GREEN: 3, GemColor.RED: 2, GemColor.BLACK: 2}),
]

_LEVEL_3_CARDS_DATA = [
    
    (3, GemColor.BLUE, {GemColor.WHITE: 3, GemColor.GREEN: 3, GemColor.RED: 5, GemColor.BLACK: 3}),
    (4, GemColor.BLUE, {GemColor.WHITE: 7}),
    (4, GemColor.BLUE, {GemColor.WHITE: 6, GemColor.BLUE: 3, GemColor.BLACK: 3}),
    (5, GemColor.BLUE, {GemColor.WHITE: 7, GemColor.BLUE: 3}),
    
    (3, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.RED: 3, GemColor.BLACK: 5}),
    (4, GemColor.GREEN, {GemColor.BLUE: 7}),
    (4, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.BLUE: 6, GemColor.GREEN: 3}),
    (5, GemColor.GREEN, {GemColor.BLUE: 7, GemColor.GREEN: 3}),

    (3, GemColor.RED, {GemColor.WHITE: 5, GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.BLACK: 3}),
    (4, GemColor.RED, {GemColor.GREEN: 7}),
    (4, GemColor.RED, {GemColor.BLUE: 3, GemColor.GREEN: 6, GemColor.RED: 3}),
    (5, GemColor.RED, {GemColor.GREEN: 7, GemColor.RED: 3}),
    
    (3, GemColor.BLACK, {GemColor.WHITE: 3, GemColor.BLUE: 5, GemColor.GREEN: 3, GemColor.RED: 3}),
    (4, GemColor.BLACK, {GemColor.RED: 7}),
    (4, GemColor.BLACK, {GemColor.GREEN: 3, GemColor.RED: 6, GemColor.BLACK: 3}),
    (5, GemColor.BLACK, {GemColor.RED: 7, GemColor.BLACK: 3}),
    
    (3, GemColor.WHITE, {GemColor.BLUE: 3, GemColor.GREEN: 5, GemColor.RED: 3, GemColor.BLACK: 3}),
    (4, GemColor.WHITE, {GemColor.BLACK: 7}),
    (4, GemColor.WHITE, {GemColor.WHITE: 3, GemColor.RED: 3, GemColor.BLACK: 6}),
    (5, GemColor.WHITE, {GemColor.WHITE: 3, GemColor.BLACK: 7}),
]

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