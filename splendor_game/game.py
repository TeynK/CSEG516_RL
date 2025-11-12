# splendor_game/game.py

import itertools
from typing import List, Optional
from collections import defaultdict

from .constants import GemColor, WINNING_SCORE, MAX_GEMS_PER_PLAYER
from .card import DevelopmentCard
from .board import Board
from .player import Player
from .actions import Action, ActionType, get_legal_actions, get_legal_return_gems_actions

class SplendorGame:
    def __init__(self, num_players: int):
        if not (2 <= num_players <= 4):
            raise ValueError(f"Invalid number of players: {num_players}")
        self.num_players = num_players
        self.board = Board(num_players)
        self.players: List[Player] = [Player(i) for i in range(num_players)]
        self.current_player_index = 0
        self.game_over: bool = False
        self.winner_id: Optional[int] = None
        self.current_player_state: str = "NORMAL"
        self.last_round_player_index: Optional[int] = None

    def reset(self) -> None:
        self.__init__(self.num_players)
    
    def get_current_player(self) -> Player:
        return self.players[self.current_player_index]
    
    def get_legal_actions(self) -> List[Action]:
        if self.game_over:
            return []
        player = self.get_current_player()
        if self.current_player_state == "RETURN_GEMS":
            return get_legal_return_gems_actions(player)
        if self.current_player_state == "NORMAL":
            return get_legal_actions(self.board, player)
        return []

    def step(self, action: Action) -> bool:
        if self.game_over:
            return True
        advance_turn = True
        player = self.get_current_player()
        if action.action_type == ActionType.TAKE_THREE_GEMS or action.action_type == ActionType.TAKE_TWO_GEMS:
            self.execute_take_gems(player, action)
            if player.get_total_gems() > MAX_GEMS_PER_PLAYER:
                self.current_player_state = "RETURN_GEMS"
                advance_turn = False
        elif action.action_type == ActionType.BUY_CARD:
            self.execute_buy_card(player, action)
            self.check_nobles(player)
        elif action.action_type == ActionType.RESERVE_CARD:
            self.execute_reserve_card(player, action)
            if player.get_total_gems() > MAX_GEMS_PER_PLAYER:
                self.current_player_state = "RETURN_GEMS"
                advance_turn = False
        elif action.action_type == ActionType.RETURN_GEMS:
            self.execute_return_gems(player, action)
            self.current_player_state = "NORMAL"
            advance_turn = True
        if self.last_round_player_index is None:
            if player.score >= WINNING_SCORE:
                self.last_round_player_index = self.current_player_index
        if advance_turn:
            self.next_turn()
        if self.last_round_player_index is not None and self.current_player_index == self.last_round_player_index:
            self.game_over = True
            self.determine_winner()
        return self.game_over
    

    def execute_take_gems(self, player: Player, action: Action) -> None:
        self.board.take_gems(action.gems)
        player.add_gems(action.gems)

    def execute_return_gems(self, player: Player, action: Action) -> None:
        player.remove_gems(action.gems)
        self.board.return_gems(action.gems)
    
    def execute_buy_card(self, player: Player, action: Action) -> None:
        card = action.card
        if card is None:
            raise ValueError("Action (BUY_CARD) is missing a 'card' object.")
        effective_cost = player.calculate_effective_cost(card)
        gems_to_spend = defaultdict(int)
        shortfall = 0
        for color, cost in effective_cost.items():
            spend = min(player.gems[color], cost)
            gems_to_spend[color] += spend
            shortfall += max(0, cost - spend)
        if shortfall > 0:
            if player.gems[GemColor.GOLD] < shortfall:
                raise ValueError("Player cannot afford card (not enough gold).")
            gems_to_spend[GemColor.GOLD] += shortfall
        player.remove_gems(gems_to_spend)
        self.board.return_gems(gems_to_spend)
        if action.is_reserved_buy:
            card_to_remove = player.reserved_cards[action.index]
            if card_to_remove != card:
                raise ValueError("Card mismatch in reserved buy.")
            player.reserved_cards.pop(action.index)
        else:
            self.board.replace_face_up_card(action.level, action.index)
        player.add_card(card)
    
    def execute_reserve_card(self, player: Player, action: Action) -> None:
        if self.board.gem_stacks[GemColor.GOLD] > 0:
            self.board.take_gems({GemColor.GOLD: 1})
            player.add_gems({GemColor.GOLD: 1})
        card_to_reserve: Optional[DevelopmentCard]
        if action.is_deck_reserve:
            card_to_reserve = self.board.draw_card_from_deck(action.level)
        else:
            card_to_reserve = action.card
            if card_to_reserve is None:
                raise ValueError("Action (RESERVE_CARD) is missing 'card' object.")
            self.board.replace_face_up_card(action.level, action.index)
        if card_to_reserve:
            player.add_reserved_card(card_to_reserve)
    
    def check_nobles(self, player: Player) -> None:
        for i in range(len(self.board.nobles) - 1, -1, -1):
            noble = self.board.nobles[i]
            can_visit = True
            for color, cost in noble.cost.items():
                if player.bonuses[color] < cost:
                    can_visit = False
                    break
            if can_visit:
                player.add_noble(noble)
                self.board.nobles.pop(i)
                break

    def next_turn(self) -> None:
        self.current_player_index = (self.current_player_index + 1) % self.num_players
    
    def determine_winner(self) -> None:
        max_score = -1
        for p in self.players:
            max_score = max(max_score, p.score)
        finalists = [p for p in self.players if p.score == max_score]
        if len(finalists) == 1:
            self.winner_id = finalists[0].player_id
            return
        min_cards = float('inf')
        for p in finalists:
            min_cards = min(min_cards, len(p.cards))
        dev_card_winners = [p for p in finalists if len(p.cards) == min_cards]
        self.winner_id = dev_card_winners[0].player_id
        return