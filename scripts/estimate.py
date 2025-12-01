import numpy as np

from sb3_contrib import MaskablePPO
from agents.maskable_dqn import MaskableDQN
from envs.splendor_aec_env import env as splendor_env
from splendor_game.actions import ActionType

def get_action(model, obs, action_mask, deterministic=True):
    obs_dict = {"observation": obs, "action_mask": action_mask}
    action, _ = model.predict(obs_dict, deterministic=deterministic)
    return int(action)

def evaluate(PPO_path, DQN_path, num_games=100):
    PPO = MaskablePPO.load(PPO_path)
    DQN = MaskableDQN.load(DQN_path)
    env = splendor_env(num_players=2)
    stats = {
        "PPO": {"wins": 0, "turns": [], "tier_1_buys": 0, "tier_2_buys": 0, "tier_3_buys": 0, "reserves": 0},
        "DQN": {"wins": 0, "turns": [], "tier_1_buys": 0, "tier_2_buys": 0, "tier_3_buys": 0, "reserves": 0},
        "Draws": 0,
        "PPO_starts_wins": 0,
        "DQN_starts_wins": 0  
    }
    for i in range(num_games):
        env.reset()
        if np.random.rand() < 0.5:
            player_map = {0: ("PPO", PPO, True), 1: ("DQN", DQN, True)}
            starter = "PPO"
        else:
            player_map = {0: ("DQN", DQN, True), 1: ("PPO", DQN, True)}
            starter = "DQN"
        game_metrics = {
            "PPO": {"t1": 0, "t2": 0, "t3": 0, "res": 0},
            "DQN": {"t1": 0, "t2": 0, "t3": 0, "res": 0}
        }
        turns = 0
        done = False
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                player_idx = env.agents.index(agent)
                model_name, model, deterministic = player_map[player_idx]
                mask = info["action_mask"]
                action = get_action(model, observation, mask, deterministic)
                action_obj = env.action_map_int_to_obj[action]
                if action_obj.action_type == ActionType.BUY_CARD:
                    if hasattr(action_obj, 'level') and action_obj.level is not None:
                        if action_obj.level == 1: game_metrics[model_name]["t1"] += 1
                        elif action_obj.level == 2: game_metrics[model_name]["t2"] += 1
                        elif action_obj.level == 3: game_metrics[model_name]["t3"] += 1
                elif action_obj.action_type == ActionType.RESERVE_CARD:
                    game_metrics[model_name]["res"] += 1
            env.step(action)
            if agent == "player_0":
                turns += 1
        winner_id = env.game.winner_id
        if winner_id is not None:
            winner_model_name = player_map[winner_id][0]
            stats[winner_model_name]["wins"] += 1
            stats[winner_model_name]["turns"].append(turns)
            for m_name in ["PPO", "DQN"]:
                stats[m_name]["tier_1_buys"] += game_metrics[m_name]["t1"]
                stats[m_name]["tier_2_buys"] += game_metrics[m_name]["t2"]
                stats[m_name]["tier_3_buys"] += game_metrics[m_name]["t3"]
                stats[m_name]["reserves"] += game_metrics[m_name]["res"]
            if winner_model_name == "PPO":
                if starter == "PPO": stats["PPO_starts_wins"] += 1
            else:
                if starter == "DQN": stats["DQN_starts_wins"] += 1
        else:
            stats["Draws"] += 1
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{num_games} games done.")

    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    total_games = num_games - stats["Draws"]
    if total_games == 0: total_games = 1
    print(f"Total Games: {num_games} (Draws: {stats['Draws']})")
    for m_name in ["PPO", "DQN"]:
        wins = stats[m_name]["wins"]
        win_rate = (wins / num_games) * 100
        avg_turns = np.mean(stats[m_name]["turns"]) if stats[m_name]["turns"] else 0
        
        # 1게임당 평균 행동 횟수
        avg_t1 = stats[m_name]["tier_1_buys"] / num_games
        avg_t2 = stats[m_name]["tier_2_buys"] / num_games
        avg_t3 = stats[m_name]["tier_3_buys"] / num_games
        avg_res = stats[m_name]["reserves"] / num_games
        
        print(f"\n>>> {m_name} Stats <<<")
        print(f"  Win Rate: {win_rate:.2f}% ({wins} wins)")
        print(f"  Avg Turns to Win: {avg_turns:.2f}")
        print(f"  Strategy Profile (Avg Actions per Game):")
        print(f"    - Buy Tier 1: {avg_t1:.2f}")
        print(f"    - Buy Tier 2: {avg_t2:.2f}")
        print(f"    - Buy Tier 3: {avg_t3:.2f}")
        print(f"    - Reserves  : {avg_res:.2f}")
    
    print("-" * 50)
    print("First Move Advantage Analysis:")
    approx_starts = num_games / 2
    print(f" PPO Win Rate when Starting: {(stats['PPO_starts_wins'] / approx_starts * 100):.2f}% (Approx)")
    print(f" DQN Win Rate when Starting: {(stats['DQN_starts_wins'] / approx_starts * 100):.2f}% (Approx)")


if __name__ == "__main__":
    Maskable_PPO = "results/models/PPO/ppo_final.zip"
    Maskable_DQN = "results/models/DQN/dqn_final.zip"
    NUM_GAMES = 1000
    evaluate(Maskable_PPO, Maskable_DQN, NUM_GAMES)
