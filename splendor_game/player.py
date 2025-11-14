from typing import List, Tuple
from collections import defaultdict

from .constants import GemColor, MAX_RESERVED_CARDS
from .card import DevelopmentCard, NobleTile, CostDict

class Player:
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.gems: CostDict = defaultdict(int)
        self.cards: List[DevelopmentCard] = []
        self.bonuses: CostDict = defaultdict(int)
        self.reserved_cards: List[DevelopmentCard] = []
        self.nobles: List[NobleTile] = []
        self.score: int = 0

    def get_total_gems(self) -> int:
        return sum(self.gems.values())
    
    def can_reserve(self) -> bool:
        return len(self.reserved_cards) < MAX_RESERVED_CARDS
    
    def add_gems(self, gems_to_add: CostDict) -> None:
        for color, count in gems_to_add.items():
            self.gems[color] += count
    
    def remove_gems(self, gems_to_remove: CostDict) -> None:
        for color, count in gems_to_remove.items():
            if self.gems[color] < count:
                raise ValueError(f"Not enough {color.value} gems to remove")
            self.gems[color] -= count
    
    def add_card(self, card: DevelopmentCard) -> None:
        self.cards.append(card)
        self.score += card.points
        self.bonuses[card.gem_type] += 1
    
    def add_reserved_card(self, card: DevelopmentCard) -> None:
        if not self.can_reserve():
            raise ValueError("Player has reached the maximum number of reserved cards")
        self.reserved_cards.append(card)

    def add_noble(self, noble: NobleTile) -> None:
        self.nobles.append(noble)
        self.score += noble.points
    
    def calculate_effective_cost(self, card: DevelopmentCard) -> CostDict:
        effective_cost = defaultdict(int)
        for color, cost in card.cost.items():
            cost_after_bonus = max(0, cost - self.bonuses.get(color, 0))
            effective_cost[color] = cost_after_bonus
        return effective_cost
    
    def get_payment_details(self, card: DevelopmentCard) -> Tuple[bool, CostDict]:
        effective_cost = self.calculate_effective_cost(card)
        gems_to_spend = defaultdict(int)
        shortfall = 0
        for color, cost in effective_cost.items():
            spend = min(self.gems.get(color, 0), cost)
            gems_to_spend[color] = spend
            shortfall += max(0, cost - spend)
        
        can_buy = self.gems.get(GemColor.GOLD, 0) >= shortfall
        if can_buy:
            gems_to_spend[GemColor.GOLD] += shortfall
            return True, gems_to_spend
        else:
            return False, defaultdict(int)
    
    def can_afford(self, card: DevelopmentCard) -> bool:
        can_buy, _ = self.get_payment_details(card)
        return can_buy
    
    def __repr__(self) -> str:
        rep_str = f"--- Player {self.player_id} (Score: {self.score}) ---\n"
        rep_str += "Gems: " + ", ".join(f"{c.value}: {v}" for c, v in self.gems.items() if v > 0) + "\n"
        rep_str += "Bonuses: " + ", ".join(f"{c.value}: {v}" for c, v in self.bonuses.items() if v > 0) + "\n"
        rep_str += f"Reserved: {len(self.reserved_cards)} cards\n"
        rep_str += f"Nobles: {len(self.nobles)}\n"
        rep_str += "-----------------------------"
        return rep_str