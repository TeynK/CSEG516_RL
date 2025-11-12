# splendor_game/board.py

from typing import Dict, List, Optional

from .constants import SETUP_CONFIG, GemColor, CARD_LEVELS, FACE_UP_CARDS_PER_LEVEL
from .card import DevelopmentCard, NobleTile, load_development_cards, load_noble_tiles

class Board:
    def __init__(self, num_players: int):
        if num_players not in SETUP_CONFIG:
            raise ValueError(f"Invalid number of players: {num_players}")
        self.num_players = num_players
        config = SETUP_CONFIG[self.num_players]

        self.gem_stacks: Dict[GemColor, int] = {}
        standard_gem_count = config['gems']
        for color in GemColor.get_standard_gems():
            self.gem_stacks[color] = standard_gem_count
        self.gem_stacks[GemColor.GOLD] = config['gold']

        self.decks: Dict[int, list[DevelopmentCard]] = load_development_cards()
        self.face_up_cards: Dict[int, List[Optional[DevelopmentCard]]] = {}

        for level in CARD_LEVELS:
            self.face_up_cards[level] = []
            for _ in range(FACE_UP_CARDS_PER_LEVEL):
                self.face_up_cards[level].append(self.draw_card_from_deck(level))

        self.nobles: List[NobleTile] = load_noble_tiles()[:config['nobles']]

    def draw_card_from_deck(self, level: int) -> Optional[DevelopmentCard]:
        return self.decks[level].pop() if self.decks[level] else None
    
    def replace_face_up_card(self, level: int, index: int) -> None:
        self.face_up_cards[level][index] = self.draw_card_from_deck(level)
    
    def take_gems(self, gems_to_take: Dict[GemColor, int]) -> None:
        for color, count in gems_to_take.items():
            if self.gem_stacks[color] < count:
                raise ValueError(f"Not enough {color.value} gems to take")
            self.gem_stacks[color] -= count

    def return_gems(self, gems_to_return: Dict[GemColor, int]) -> None:
        for color, count in gems_to_return.items():
            self.gem_stacks[color] += count
    
    def __repr__(self) -> str:
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