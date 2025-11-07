# splendor_game/main_text.py

"""
A text-based interface to run and test the SplendorGame engine.

This script allows human players to play the game in the terminal
to verify that all game logic, state transitions, and rules
are working correctly.

To run this file, execute it as a module from the PARENT directory:
$ python -m splendor_game.main_text
"""

# Import from the package using relative imports
from typing import Dict, List, Optional
from .game import SplendorGame
from .player import Player
from .board import Board
from .actions import Action


def print_game_state(game: SplendorGame):
    """Prints the current board and player state."""
    player = game.get_current_player()
    
    print("\n" + "="*50)
    print(f"---  TURN FOR PLAYER {player.player_id} (Score: {player.score}) ---")
    print("="*50)
    
    # Print Board State
    print(game.board)
    
    # Print Player State
    print(player)


def get_player_input(legal_actions: List[Action]) -> Action:
    """
    Lists all legal actions and prompts the user to select one.
    Returns the chosen Action object.
    """
    print("\n--- Legal Actions ---")
    if not legal_actions:
        print("No legal actions available. This should not happen.")
        return None # Should ideally not be reached

    for i, action in enumerate(legal_actions):
        print(f"  [{i+1}] {action}")
    
    while True:
        try:
            choice_str = input(f"\nChoose an action (1-{len(legal_actions)}): ")
            choice_int = int(choice_str)
            
            if 1 <= choice_int <= len(legal_actions):
                # Valid choice
                chosen_action = legal_actions[choice_int - 1]
                return chosen_action
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(legal_actions)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting game.")
            raise

def main():
    """Main game loop for the text-based version."""
    
    # --- 1. Game Setup ---
    while True:
        try:
            num_players_str = input("Enter number of players (2-4): ")
            num_players = int(num_players_str)
            if 2 <= num_players <= 4:
                break
            print("Invalid number. Must be between 2 and 4.")
        except ValueError:
            print("Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            return # Exit script
            
    game = SplendorGame(num_players)
    
    # --- 2. Main Game Loop ---
    try:
        while not game.game_over:
            # 1. Display current state
            print_game_state(game)
            
            # 2. Get legal actions
            legal_actions = game.get_legal_actions()
            
            # TODO: Handle the "Return Gems" state
            # As noted in game.py, if a player has > 10 gems,
            # get_legal_actions() should return *only* return-gem actions.
            # This test script assumes the simple action path for now.
            
            # 3. Get player choice
            chosen_action = get_player_input(legal_actions)
            if chosen_action is None: # Should not happen
                break
                
            print(f"\nPlayer {game.get_current_player().player_id} chose: {chosen_action}")
            
            # 4. Execute action
            game.step(chosen_action)
            
    except (KeyboardInterrupt, EOFError):
        print("\nGame interrupted. Exiting.")
        return

    # --- 3. Game End ---
    print("\n" + "="*50)
    print("--- GAME OVER ---")
    print(f"🏆 Winner is Player {game.winner_id}!")
    print("="*50)
    
    print("\nFinal Scores:")
    for player in game.players:
        print(f"  Player {player.player_id}: {player.score} points, {len(player.cards)} cards")

if __name__ == "__main__":
    main()