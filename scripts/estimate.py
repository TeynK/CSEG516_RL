import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from agents.maskable_dqn import MaskableDQN
from envs.splendor_aec_env import env as splendor_aec_env
from splendor_game.actions import ActionType

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

MAX_GAME_TURNS = 500

class BattleArena:
    def __init__(self, ppo_path, dqn_path, render_mode=None):
        self.env = splendor_aec_env(num_players=2, render_mode=render_mode)
        self.ppo_model = self._load_model(ppo_path, "PPO")
        self.dqn_model = self._load_model(dqn_path, "DQN")
        self.models = {"PPO": self.ppo_model, "DQN": self.dqn_model}
        
    def _load_model(self, path, model_type):
        if not os.path.exists(path):
            print(f"[경고] 모델 파일을 찾을 수 없습니다: {path}")
            return None
        print(f"[{model_type}] 모델 로딩 중: {path}")
        if model_type == "PPO":
            return MaskablePPO.load(path)
        elif model_type == "DQN":
            return MaskableDQN.load(path)
        else:
            raise ValueError(f"알 수 없는 모델 타입: {model_type}")

    def run_battle(self, num_games=100):
        results = []
        all_actions_history = []

        if self.ppo_model is None or self.dqn_model is None:
            print("모델 로드 실패로 대전을 진행할 수 없습니다.")
            return pd.DataFrame(), pd.DataFrame()

        print(f"\n--- {num_games} 게임 대전 시작 (PPO vs DQN) ---")
        
        for i in tqdm(range(num_games)):
            if random.random() < 0.5:
                agent_mapping = {0: "PPO", 1: "DQN"}
            else:
                agent_mapping = {0: "DQN", 1: "PPO"}
            
            verbose = (i == 0)
            game_stats, action_log = self._play_single_game(agent_mapping, game_id=i, verbose=verbose)
            
            results.append(game_stats)
            all_actions_history.extend(action_log)
            
        return pd.DataFrame(results), pd.DataFrame(all_actions_history)

    def _play_single_game(self, agent_mapping, game_id, verbose=False):
        self.env.reset()
        
        stats = {
            "game_id": game_id,
            "winner": None,
            "total_turns": 0,
            "PPO_score": 0, "DQN_score": 0,
            "PPO_role": "First" if agent_mapping[0] == "PPO" else "Second",
            "DQN_role": "First" if agent_mapping[0] == "DQN" else "Second",
            "PPO_actions": defaultdict(int), "DQN_actions": defaultdict(int),
            "PPO_cards": {1: 0, 2: 0, 3: 0}, "DQN_cards": {1: 0, 2: 0, 3: 0},
            "PPO_nobles": 0, "DQN_nobles": 0,
            "termination_reason": "Normal"
        }
        action_log = []
        
        turn_counter = 0
        if verbose: print(f"\n[Game {game_id}] Start: {agent_mapping[0]} vs {agent_mapping[1]}")

        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            
            if termination or truncation:
                if verbose and truncation: print(f"  -> Game Truncated!")
                self.env.step(None)
                continue
            
            if turn_counter >= MAX_GAME_TURNS:
                stats["termination_reason"] = "Max Turns Reached"
                self.env.step(None)
                break

            player_idx = self.env.agents.index(agent)
            model_name = agent_mapping[player_idx]
            model = self.models[model_name]
            mask = info["action_mask"]
            
            obs_dict = {"observation": observation, "action_mask": mask}
            
            try:
                if model_name == "PPO":
                    action_idx, _ = model.predict(obs_dict, action_masks=mask, deterministic=False)
                else:
                    action_idx, _ = model.predict(obs_dict, deterministic=True)
            except Exception as e:
                print(f"[Error] Predict failed for {model_name}: {e}")
                break
            
            if isinstance(action_idx, np.ndarray):
                action_int = int(action_idx.item())
            else:
                action_int = int(action_idx)
            if action_int in self.env.action_map_int_to_obj:
                action_obj = self.env.action_map_int_to_obj[action_int]
                act_type_name = action_obj.action_type.name
                
                stats[f"{model_name}_actions"][act_type_name] += 1
                action_log.append({
                    "game_id": game_id,
                    "turn": turn_counter,
                    "model": model_name,
                    "action_type": act_type_name
                })
                
                if verbose:
                    print(f"  Turn {turn_counter}: {model_name} -> {act_type_name}")
            
            self.env.step(action_int)
            
            if player_idx == 1:
                turn_counter += 1
        
        game_instance = self.env.game
        winner_id = game_instance.winner_id
        
        stats["total_turns"] = turn_counter
        if winner_id is not None:
            stats["winner"] = agent_mapping[winner_id]
        else:
            stats["winner"] = "Draw"
            if stats["termination_reason"] == "Normal":
                 stats["termination_reason"] = "Stalemate/Error"

        for pid in [0, 1]:
            p_obj = game_instance.players[pid]
            m_name = agent_mapping[pid]
            stats[f"{m_name}_score"] = p_obj.score
            stats[f"{m_name}_nobles"] = len(p_obj.nobles)
            for card in p_obj.cards:
                stats[f"{m_name}_cards"][card.level] += 1
                
        return stats, action_log

class Visualizer:
    def __init__(self, df, action_df, save_dir="results/plots/ppo_vs_dqn"):
        self.df = df
        self.action_df = action_df
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
    def plot_all(self):
        if self.df.empty:
            print("데이터가 없어 시각화를 건너뜁니다.")
            return

        print("\n--- 결과 시각화 생성 중 ---")
        try:
            self.plot_win_rate()
            self.plot_score_distribution()
            self.plot_turn_distribution()
            self.plot_card_tier_analysis()
            self.plot_action_distribution()
            self.plot_nobles_analysis()
            if not self.action_df.empty:
                self.plot_turn_action_distribution(model_name="PPO")
                self.plot_turn_action_distribution(model_name="DQN")
                
            print(f"모든 그래프가 '{self.save_dir}'에 저장되었습니다.")
        except Exception as e:
            print(f"시각화 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    def plot_win_rate(self):
        plt.figure(figsize=(8, 6))
        win_counts = self.df['winner'].value_counts()
        colors = {'PPO': '#3498db', 'DQN': '#e74c3c', 'Draw': '#95a5a6'}
        plt.pie(win_counts, labels=win_counts.index, autopct='%1.1f%%', 
                colors=[colors.get(x, '#333333') for x in win_counts.index],
                startangle=90, counterclock=False, shadow=True)
        plt.title('Win Rate: PPO vs DQN')
        plt.savefig(os.path.join(self.save_dir, 'win_rate.png'))
        plt.close()

    def plot_score_distribution(self):
        plt.figure(figsize=(10, 6))
        max_score = max(self.df['PPO_score'].max(), self.df['DQN_score'].max())
        bin_w = 1 if max_score > 0 else 0.5
        sns.histplot(self.df['PPO_score'], color='blue', label='PPO', kde=False, alpha=0.5, binwidth=bin_w)
        sns.histplot(self.df['DQN_score'], color='red', label='DQN', kde=False, alpha=0.5, binwidth=bin_w)
        plt.title('Score Distribution')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'score_dist.png'))
        plt.close()
        
    def plot_turn_distribution(self):
        plt.figure(figsize=(10, 6))
        if len(self.df['winner'].unique()) > 0:
            sns.boxplot(x='winner', y='total_turns', data=self.df, palette={'PPO': 'blue', 'DQN': 'red', 'Draw': 'gray'})
            plt.title('Game Duration (Turns) by Winner')
            plt.savefig(os.path.join(self.save_dir, 'turn_dist.png'))
        plt.close()

    def plot_card_tier_analysis(self):
        ppo_cards = pd.DataFrame(self.df['PPO_cards'].tolist()).mean()
        dqn_cards = pd.DataFrame(self.df['DQN_cards'].tolist()).mean()
        labels = ['Tier 1', 'Tier 2', 'Tier 3']
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(10, 6))
        p_vals = [ppo_cards.get(i, 0) for i in [1, 2, 3]]
        d_vals = [dqn_cards.get(i, 0) for i in [1, 2, 3]]
        plt.bar(x - width/2, p_vals, width, label='PPO', color='blue', alpha=0.7)
        plt.bar(x + width/2, d_vals, width, label='DQN', color='red', alpha=0.7)
        plt.ylabel('Avg Cards Purchased')
        plt.title('Average Cards Purchased by Tier')
        plt.xticks(x, labels)
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'card_tiers.png'))
        plt.close()

    def plot_action_distribution(self):
        if 'PPO_actions' not in self.df.columns: return
        ppo_acts = pd.DataFrame(self.df['PPO_actions'].tolist()).fillna(0).mean()
        dqn_acts = pd.DataFrame(self.df['DQN_actions'].tolist()).fillna(0).mean()
        all_actions = set(ppo_acts.index) | set(dqn_acts.index)
        actions_list = sorted(list(all_actions))
        if not actions_list: return
        ppo_vals = [ppo_acts.get(a, 0) for a in actions_list]
        dqn_vals = [dqn_acts.get(a, 0) for a in actions_list]
        x = np.arange(len(actions_list))
        width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, ppo_vals, width, label='PPO', color='blue', alpha=0.7)
        plt.bar(x + width/2, dqn_vals, width, label='DQN', color='red', alpha=0.7)
        plt.ylabel('Avg Count per Game')
        plt.title('Action Type Distribution')
        plt.xticks(x, actions_list, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'action_dist.png'))
        plt.close()

    # [신규 기능] 턴별 행동 분포 시각화
    def plot_turn_action_distribution(self, model_name):
        subset = self.action_df[self.action_df['model'] == model_name]
        if subset.empty: return

        # 데이터 변환: (turn, action_type) -> count
        # 각 턴마다 전체 게임 수로 나누어 비율(Probability) 계산
        # 1. 턴별, 행동별 카운트 집계
        counts = subset.groupby(['turn', 'action_type']).size().unstack(fill_value=0)
        
        # 2. 턴별 총합으로 나누어 비율(%)로 변환
        props = counts.div(counts.sum(axis=1), axis=0)
        
        # 너무 긴 턴은 잘라내기 (데이터가 희소해져서 노이즈가 됨, 예: 60턴까지만)
        limit_turn = min(60, props.index.max())
        props = props.loc[:limit_turn]

        # 색상 매핑 (직관적으로 설정)
        color_map = {
            'TAKE_THREE_GEMS': '#3498db', # 파랑
            'TAKE_TWO_GEMS': '#2980b9',   # 진한 파랑
            'BUY_CARD': '#2ecc71',        # 초록 (핵심)
            'RESERVE_CARD': '#f1c40f',    # 노랑
            'RETURN_GEMS': '#e74c3c'      # 빨강 (부정적/반납)
        }
        # 데이터에 있는 컬럼만 색상 지정
        colors = [color_map.get(col, '#95a5a6') for col in props.columns]

        # 그래프 그리기 (Stacked Area Chart가 흐름 보기에 좋음, 혹은 Bar)
        plt.figure(figsize=(14, 7))
        props.plot(kind='bar', stacked=True, width=1.0, color=colors, figsize=(14, 7), ax=plt.gca())
        
        plt.title(f'Action Distribution over Turns ({model_name})')
        plt.xlabel('Turn Number')
        plt.ylabel('Proportion of Actions')
        plt.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'turn_action_dist_{model_name}.png')
        plt.savefig(save_path)
        plt.close()
        print(f" - {model_name} 턴별 행동 그래프 저장됨: {save_path}")

    def plot_nobles_analysis(self):
        if 'PPO_nobles' not in self.df.columns: return
        
        # 데이터 변환 (Wide -> Long format)
        # 예: game_id, Model(PPO/DQN), Nobles(개수) 형태로 변환해야 seaborn으로 그리기 편함
        noble_df = self.df[['PPO_nobles', 'DQN_nobles']].melt(var_name='Model', value_name='Nobles')
        
        # 이름 정리 ('PPO_nobles' -> 'PPO')
        noble_df['Model'] = noble_df['Model'].str.replace('_nobles', '')
        
        plt.figure(figsize=(10, 6))
        
        # Countplot: 각 에이전트가 귀족을 0개, 1개, 2개... 획득한 게임이 몇 판인지 보여줌
        ax = sns.countplot(data=noble_df, x='Nobles', hue='Model', palette={'PPO': 'blue', 'DQN': 'red'})
        
        plt.title('Distribution of Nobles Acquired per Game')
        plt.xlabel('Number of Nobles Acquired')
        plt.ylabel('Game Count')
        plt.legend(title='Agent')
        
        # 막대 위에 숫자 표시 (선택 사항)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width()/2., height + 0.1, int(height), ha="center")

        save_path = os.path.join(self.save_dir, 'nobles_dist.png')
        plt.savefig(save_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evalute PPO vs DQN models")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate")
    parser.add_argument("--ppo_path", type=str, 
                        default="results/models/PPO/PPO_vs_bot_final.zip",
                        help="Path to PPO model")
    parser.add_argument("--dqn_path", type=str, 
                        default="results/models/DQN/DQN_vs_bot_final.zip",
                        help="Path to DQN model")
    args = parser.parse_args()
    
    if not os.path.exists(args.ppo_path) or not os.path.exists(args.dqn_path):
        print("\n[오류] 모델 파일을 찾을 수 없습니다.")
        if not os.path.exists(args.ppo_path): print(f" - Missing PPO: {args.ppo_path}")
        if not os.path.exists(args.dqn_path): print(f" - Missing DQN: {args.dqn_path}")
        return

    try:
        arena = BattleArena(args.ppo_path, args.dqn_path)
        df, action_df = arena.run_battle(num_games=args.games)
        
        if df.empty: return

        print("\n" + "="*40)
        print("       FINAL RESULTS SUMMARY       ")
        print("="*40)
        print(f"Total Games: {args.games}")
        print("-" * 20)
        print(df['winner'].value_counts())
        
        if 'Draw' in df['winner'].values:
            draws = df[df['winner'] == 'Draw']
            if 'termination_reason' in draws.columns:
                print("\n[Draw Reasons]")
                print(draws['termination_reason'].value_counts())

        print("-" * 20)
        print(f"PPO Avg Score: {df['PPO_score'].mean():.2f}")
        print(f"DQN Avg Score: {df['DQN_score'].mean():.2f}")
        print(f"Avg Turns: {df['total_turns'].mean():.1f}")
        print("="*40)
        viz = Visualizer(df, action_df)
        viz.plot_all()
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()