# splendor_game/game.py

"""
Defines the main SplendorGame class, which orchestrates the entire game.

This class holds the board state, the list of players, and manages
the game loop (state machine).
"""

import itertools # <<< NEW
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

from .constants import GemColor, WINNING_SCORE, MAX_GEMS_PER_PLAYER
from .card import DevelopmentCard, NobleTile, CostDict
from .board import Board
from .player import Player
# <<< MODIFIED >>>
from .actions import Action, ActionType, get_legal_actions

class SplendorGame:
    """
    The main game engine. Manages the game state, player turns,
    action execution, and win conditions.
    """

    def __init__(self, num_players: int):
        """
        Initializes a new game of Splendor.
        
        Args:
            num_players: The number of players (2, 3, or 4).
        """
        if not (2 <= num_players <= 4):
            raise ValueError("Splendor must be played with 2, 3, or 4 players.")
            
        self.num_players = num_players
        self.board: Board = Board(num_players)
        self.players: List[Player] = [Player(i) for i in range(num_players)]
        
        self.current_player_index: int = 0
        
        # State tracking
        self.game_over: bool = False
        self.winner_id: Optional[int] = None
        
        # <<< NEW: Game state for gem returns >>>
        self.current_game_state: str = "NORMAL" # "NORMAL" or "RETURN_GEMS"
        
        # Index of the player who triggers the last round.
        # The game ends when the turn returns to this player.
        self.last_round_player_index: Optional[int] = None

    def reset(self) -> None:
        """Resets the game to its initial state."""
        # Re-initialize the game
        self.__init__(self.num_players)

    def get_current_player(self) -> Player:
        """Returns the Player object for the current turn."""
        return self.players[self.current_player_index]

    def get_legal_actions(self) -> List[Action]:
        """
        Gets all legal actions for the current player.
        
        Returns:
            A list of valid Action objects.
            Returns an empty list if the game is over.
        """
        if self.game_over:
            return []
            
        player = self.get_current_player()
        
        # ---
        # <<< MODIFIED: Handle 'Return Gems' state >>>
        # If the player is in the return state OR has too many gems
        # (the latter is a safeguard), force return actions.
        if self.current_game_state == "RETURN_GEMS" or \
           player.get_total_gems() > MAX_GEMS_PER_PLAYER:
            
            # Force state transition just in case
            self.current_game_state = "RETURN_GEMS"
            return self._get_gem_return_actions(player)
        # ---
        
        # If state is "NORMAL" and gems are fine, get standard actions
        # This calls splendor_game.actions.get_legal_actions
        return get_legal_actions(self.board, player)

    # <<< NEW: Helper for get_legal_actions >>>
    def _get_gem_return_actions(self, player: Player) -> List[Action]:
        """Generates all valid actions for returning gems."""
        gems_to_return_count = player.get_total_gems() - MAX_GEMS_PER_PLAYER
        
        if gems_to_return_count <= 0:
            # This shouldn't be called if count is <= 0, but as a safeguard
            self.current_game_state = "NORMAL"
            return []

        actions: List[Action] = []
        
        # Create a flat list of all gems the player has (including gold)
        player_gems_list: List[GemColor] = []
        for color, count in player.gems.items():
            if count > 0:
                player_gems_list.extend([color] * count)
        
        # Find all unique combinations of gems to return
        # Using combinations on the flat list handles counts correctly
        all_combos = set(itertools.combinations(player_gems_list, gems_to_return_count))
        
        # Use a set to store unique dictionaries (via frozenset)
        unique_gem_dicts = set()
        
        for combo_tuple in all_combos:
            gems_to_return_dict = defaultdict(int)
            for color in combo_tuple:
                gems_to_return_dict[color] += 1
            
            # Use frozenset to find unique *sets* of gems to return
            # (e.g., {White: 2} is one set, {White: 1, Blue: 1} is another)
            frozen_dict = frozenset(gems_to_return_dict.items())
            
            if frozen_dict not in unique_gem_dicts:
                unique_gem_dicts.add(frozen_dict)
                action = Action(ActionType.RETURN_GEMS, gems=gems_to_return_dict)
                actions.append(action)
                
        return actions

    def step(self, action: Action) -> bool:
        """
        Executes a single action and advances the game state.

        Args:
            action: The Action object to be performed.

        Returns:
            bool: True if the game is over after this step, False otherwise.
        """
        if self.game_over:
            # print("Cannot take step: Game is already over.")
            return True

        player = self.get_current_player()
        
        # --- Flag to track turn advancement ---
        advance_turn = True

        # --- 1. Execute the core action ---
        
        # <<< NEW: Handle RETURN_GEMS action >>>
        if action.action_type == ActionType.RETURN_GEMS:
            self._execute_return_gems(player, action)
            
            if player.get_total_gems() <= MAX_GEMS_PER_PLAYER:
                # Player is now compliant, change state and advance turn
                self.current_game_state = "NORMAL"
                advance_turn = True
            else:
                # Player still not compliant (e.g., if we allowed partial returns)
                # Keep state as "RETURN_GEMS"
                advance_turn = False # Do not advance turn

        elif action.action_type == ActionType.TAKE_THREE_GEMS or \
           action.action_type == ActionType.TAKE_TWO_GEMS:
            
            self._execute_take_gems(player, action)
            
            # <<< MODIFIED: Check for gem overflow >>>
            if player.get_total_gems() > MAX_GEMS_PER_PLAYER:
                # Trigger return gem state
                self.current_game_state = "RETURN_GEMS"
                advance_turn = False # Do not advance turn

        elif action.action_type == ActionType.BUY_CARD:
            self._execute_buy_card(player, action)
            self._check_nobles(player)
            # Note: Buying a card *reduces* gems, so no overflow check needed
            advance_turn = True

        elif action.action_type == ActionType.RESERVE_CARD:
            self._execute_reserve_card(player, action)
            
            # <<< MODIFIED: Check for gem overflow >>>
            if player.get_total_gems() > MAX_GEMS_PER_PLAYER:
                # Trigger return gem state (from taking gold)
                self.current_game_state = "RETURN_GEMS"
                advance_turn = False # Do not advance turn

        # --- 2. Check for game end trigger ---
        # <<< MODIFIED: Only check on turns that advance >>>
        if advance_turn and self.last_round_player_index is None:
            if player.score >= WINNING_SCORE:
                # This player triggered the end!
                self.last_round_player_index = (self.current_player_index - 1 + self.num_players) % self.num_players


        # --- 3. Advance the turn (if flagged) ---
        # <<< MODIFIED >>>
        if advance_turn:
            self._next_turn()

        # --- 4. Check if the game has officially ended ---
        # <<< MODIFIED: Only check on turns that advance >>>
        if advance_turn and self.last_round_player_index == self.current_player_index:
            self.game_over = True
            self._determine_winner()

        return self.game_over

    # --- Private Helper Methods ---

    # <<< NEW: Helper for step >>>
    def _execute_return_gems(self, player: Player, action: Action) -> None:
        """Executes gem returning logic."""
        player.remove_gems(action.gems)
        self.board.return_gems(action.gems)

    def _execute_take_gems(self, player: Player, action: Action) -> None:
        """Executes gem taking logic."""
        self.board.take_gems(action.gems)
        player.add_gems(action.gems)

    def _execute_buy_card(self, player: Player, action: Action) -> None:
        """Executes card buying logic, including payment."""
        card = action.card
        if card is None:
            raise ValueError("Action (BUY_CARD) is missing a 'card' object.")

        # 1. Calculate cost and determine gems to spend
        effective_cost = player.calculate_effective_cost(card)
        gems_to_spend = defaultdict(int)
        shortfall = 0

        for color, cost in effective_cost.items():
            spend = min(player.gems[color], cost)
            gems_to_spend[color] += spend
            if spend < cost:
                shortfall += (cost - spend)
        
        # 2. Use gold to cover any remaining shortfall
        if shortfall > 0:
            if player.gems[GemColor.GOLD] < shortfall:
                raise ValueError("Player cannot afford card (not enough gold).")
            gems_to_spend[GemColor.GOLD] += shortfall

        # 3. Perform transactions
        player.remove_gems(gems_to_spend)
        self.board.return_gems(gems_to_spend)
        
        # 4. Give card to player
        if action.is_reserved_buy:
            # Find and remove from reserved list
            # We use action.index to safely remove the correct card
            card_to_remove = player.reserved_cards[action.index]
            if card_to_remove != card:
                raise ValueError("Card mismatch in reserved buy.")
            player.reserved_cards.pop(action.index)
        else:
            # Replace card on the board
            self.board.replace_face_up_card(action.level, action.index)
        
        player.add_card(card) # This updates score and bonuses

    def _execute_reserve_card(self, player: Player, action: Action) -> None:
        """Executes card reserving logic."""
        
        # 1. Take a gold gem (if available)
        if self.board.gem_stacks[GemColor.GOLD] > 0:
            self.board.take_gems({GemColor.GOLD: 1})
            player.add_gems({GemColor.GOLD: 1})
        
        # 2. Get the card
        card_to_reserve: Optional[DevelopmentCard]
        if action.is_deck_reserve:
            # Reserve from deck
            card_to_reserve = self.board.draw_card_from_deck(action.level)
        else:
            # Reserve from face-up
            card_to_reserve = action.card
            if card_to_reserve is None:
                 raise ValueError("Action (RESERVE_CARD) is missing 'card' object.")
            self.board.replace_face_up_card(action.level, action.index)
        
        # 3. Add to player's hand (if a card was available)
        if card_to_reserve:
            player.add_reserved_card(card_to_reserve)

    def _check_nobles(self, player: Player) -> None:
        """
        Checks if the player's bonuses attract any available nobles.
        A player can only attract one noble per turn.
        """
        # Iterate in reverse to allow safe removal while iterating
        for i in range(len(self.board.nobles) - 1, -1, -1):
            noble = self.board.nobles[i]
            
            can_visit = True
            for color, cost in noble.cost.items():
                if player.bonuses[color] < cost:
                    can_visit = False
                    break
            
            if can_visit:
                # Player gets the noble
                player.add_noble(noble)
                # Remove noble from the board
                self.board.nobles.pop(i)
                # Rule: Only one noble per turn.
                break 

    def _next_turn(self) -> None:
        """Advances the turn to the next player."""
        self.current_player_index = (self.current_player_index + 1) % self.num_players

    def _determine_winner(self) -> None:
        """
        Calculates the winner based on score and tie-breaker rules
        (fewer development cards).
        """
        max_score = -1
        for p in self.players:
            max_score = max(max_score, p.score)
        
        # Get all players with the max score
        finalists = [p for p in self.players if p.score == max_score]
        
        if len(finalists) == 1:
            self.winner_id = finalists[0].player_id
            return

        # Tie-breaker: Fewer development cards
        min_cards = float('inf')
        for p in finalists:
            min_cards = min(min_cards, len(p.cards))
            
        dev_card_winners = [p for p in finalists if len(p.cards) == min_cards]
        
        # If still tied, the player who is *earliest* in the original
        # turn order among the tie-breakers wins.
        # (Note: Some house rules say it's a shared victory)
        self.winner_id = dev_card_winners[0].player_id