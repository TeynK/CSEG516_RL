from splendor_game.game import SplendorGame
from splendor_game.actions import Action

def print_game_state(game: SplendorGame):
    print("\n" + "="*50)
    print(game.board)
    print("\n" + "-"*50)
    for player in game.players:
        print(player)
    print("-" * 50)

def main():
    try:
        num_players = int(input("Enter number of players (2-4): "))
        if not (2 <= num_players <= 4):
            print("Invalid number of players. Exiting.")
            return
    except ValueError:
        print("Invalid input. Please enter a number. Exiting.")
        return
    game = SplendorGame(num_players)

    while not game.game_over:
        print_game_state(game)
        current_player = game.get_current_player()
        print(f"\n>>> Player {current_player.player_id}'s Turn <<<")
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            print("No legal actions available. Something might be wrong.")
            break
        print("Legal Actions:")
        for i, action in enumerate(legal_actions):
            print(f"  [{i}]: {action}")
        try:
            action_index = int(input(f"Choose an action (0-{len(legal_actions) - 1}): "))
            if not (0 <= action_index < len(legal_actions)):
                print("Invalid action index. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number. Please try again.")
            continue
        selected_action = legal_actions[action_index]
        print(f"Executing: {selected_action}")
        game.step(selected_action)
    print("\n" + "="*50)
    print("GAME OVER")
    print_game_state(game)
    if game.winner_id is not None:
        print(f"Player {game.winner_id} is the winner!")
    else:
        print("The game ended in a draw or with no winner.")
    print("="*50)

if __name__ == "__main__":
    main()