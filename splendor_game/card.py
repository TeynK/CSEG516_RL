# splendor_game/card.py

import random
from dataclasses import dataclass, field
from typing import Dict

from .constants import GemColor, _LEVEL_1_CARDS_DATA, _LEVEL_2_CARDS_DATA, _LEVEL_3_CARDS_DATA, _NOBLES_DATA

CostDict = Dict[GemColor, int]

@dataclass(frozen=True)
class DevelopmentCard:
    level: int
    points: int
    gem_type: GemColor
    cost: CostDict = field(default_factory=dict)

    def __repr__(self) -> str:
        cost_str = ", ".join(f"{c.value}: {v}" for c, v in self.cost.items())
        return f"DevCard(L{self.level}, {self.gem_type.value}, {self.points}pts, Cost:[{cost_str}])"

@dataclass(frozen=True)
class NobleTile:
    points: int
    cost: CostDict = field(default_factory=dict)

    def __repr__(self):
        cost_str = ", ".join(f"{c.value}: {v}" for c, v in self.cost.items())
        return f"Noble({self.points}pts, Requires:[{cost_str}])"
    
def load_development_cards() -> Dict[int, list[DevelopmentCard]]:
    decks = {}
    deck1 = [DevelopmentCard(level=1, points=p, gem_type=g, cost=c) for p, g, c in _LEVEL_1_CARDS_DATA]
    random.shuffle(deck1)
    decks[1] = deck1
    deck2 = [DevelopmentCard(level=2, points=p, gem_type=g, cost=c) for p, g, c in _LEVEL_2_CARDS_DATA]
    random.shuffle(deck2)
    decks[2] = deck2
    deck3 = [DevelopmentCard(level=3, points=p, gem_type=g, cost=c) for p, g, c in _LEVEL_3_CARDS_DATA]
    random.shuffle(deck3)
    decks[3] = deck3
    return decks

def load_noble_tiles() -> list[NobleTile]:
    nobles = [NobleTile(points=p, cost=c) for p, c in _NOBLES_DATA]
    random.shuffle(nobles)
    return nobles