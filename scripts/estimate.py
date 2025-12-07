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

# 프로젝트 루트 경로 설정 (상위 폴더 인식용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from agents.maskable_dqn import MaskableDQN
from envs.splendor_aec_env import env as splendor_aec_env
from splendor_game.actions import ActionType

# 그래프 폰트 및 스타일 설정
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
        all_score_history = [] 
        all_value_history = []

        if self.ppo_model is None or self.dqn_model is None:
            print("모델 로드 실패로 대전을 진행할 수 없습니다.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        print(f"\n--- {num_games} 게임 대전 시작 (PPO vs DQN) ---")
        
        for i in tqdm(range(num_games)):
            # 선공/후공 랜덤 결정
            if random.random() < 0.5:
                agent_mapping = {0: "PPO", 1: "DQN"}
            else:
                agent_mapping = {0: "DQN", 1: "PPO"}
            
            verbose = (i == 0)
            game_stats, action_log, score_log, value_log = self._play_single_game(agent_mapping, game_id=i, verbose=verbose)
            
            # 승패 정보 주입 (행동 로그용)
            winner_model = game_stats["winner"]
            for entry in action_log:
                if winner_model == "Draw":
                    entry["result"] = "Draw"
                elif entry["model"] == winner_model:
                    entry["result"] = "Win"
                else:
                    entry["result"] = "Loss"

            results.append(game_stats)
            all_actions_history.extend(action_log)
            all_score_history.extend(score_log)
            all_value_history.extend(value_log)
            
        return pd.DataFrame(results), pd.DataFrame(all_actions_history), pd.DataFrame(all_score_history), pd.DataFrame(all_value_history)

    def _estimate_value(self, model, obs, mask):
        """현재 상태에 대한 모델의 가치 추정값(Predicted Value)을 반환"""
        import torch as th
        with th.no_grad():
            if isinstance(model, MaskablePPO):
                # PPO: Value Function 예측값
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                value = model.policy.predict_values(obs_tensor)
                return value.item()
            elif isinstance(model, MaskableDQN):
                # DQN: Q-Network가 예측한 Q값 중 최댓값 (Max Q)
                obs_tensor = model.q_net.obs_to_tensor(obs)[0]
                q_values = model.q_net(obs_tensor)
                max_q = q_values.max(dim=1).values
                return max_q.item()
        return 0.0

    def _play_single_game(self, agent_mapping, game_id, verbose=False):
        self.env.reset()
        
        # [중요] 통계 딕셔너리 초기화 (KeyError 방지를 위해 모든 키 미리 선언)
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
        score_log = []
        value_log = [] 
        
        turn_counter = 0
        if verbose: print(f"\n[Game {game_id}] Start: {agent_mapping[0]} vs {agent_mapping[1]}")

        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            
            if termination or truncation:
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
            
            # 1. 가치 추정 (Estimation)
            pred_val = self._estimate_value(model, obs_dict, mask)
            
            # 2. 행동 선택
            try:
                if model_name == "PPO":
                    action_idx, _ = model.predict(obs_dict, action_masks=mask, deterministic=False)
                else:
                    action_idx, _ = model.predict(obs_dict, deterministic=True)
            except Exception as e:
                print(f"[Error] Predict failed for {model_name}: {e}")
                break
            
            action_int = int(action_idx.item()) if isinstance(action_idx, np.ndarray) else int(action_idx)
            
            if action_int in self.env.action_map_int_to_obj:
                action_obj = self.env.action_map_int_to_obj[action_int]
                act_type_name = action_obj.action_type.name
                
                # 3. 행동 상세 이름 (카드 티어 포함)
                detailed_action_name = act_type_name
                if action_obj.action_type == ActionType.BUY_CARD:
                    card_level = None
                    if action_obj.is_reserved_buy:
                        current_player = self.env.game.players[player_idx]
                        if action_obj.index < len(current_player.reserved_cards):
                            card = current_player.reserved_cards[action_obj.index]
                            card_level = card.level
                    else:
                        card_level = action_obj.level
                    
                    if card_level is not None:
                        detailed_action_name = f"BUY_CARD_T{card_level}"

                # 4. 통계 기록
                stats[f"{model_name}_actions"][act_type_name] += 1
                
                # 가치 로그 (Predicted Value)
                value_log.append({
                    "game_id": game_id,
                    "turn": turn_counter,
                    "model": model_name,
                    "predicted_value": pred_val
                    # "final_score"는 게임 종료 후 채움
                })

                # 행동 로그
                action_log.append({
                    "game_id": game_id,
                    "turn": turn_counter,
                    "model": model_name,
                    "action_type": detailed_action_name
                })
            
            self.env.step(action_int)
            
            # 라운드 종료 시 점수 기록 (후공 플레이어 턴 끝)
            if player_idx == 1:
                scores = {m: self.env.game.players[pid].score for pid, m in agent_mapping.items()}
                score_log.append({
                    "game_id": game_id, 
                    "turn": turn_counter, 
                    "PPO_score": scores["PPO"], 
                    "DQN_score": scores["DQN"]
                })
                turn_counter += 1
        
        # 게임 종료 후처리
        game_instance = self.env.game
        winner_id = game_instance.winner_id
        
        stats["total_turns"] = turn_counter
        stats["winner"] = agent_mapping[winner_id] if winner_id is not None else "Draw"
        if stats["termination_reason"] == "Normal" and winner_id is None:
             stats["termination_reason"] = "Stalemate"

        # 최종 점수 기록
        final_scores = {
            agent_mapping[0]: game_instance.players[0].score,
            agent_mapping[1]: game_instance.players[1].score
        }
        stats["PPO_score"] = final_scores["PPO"]
        stats["DQN_score"] = final_scores["DQN"]
        
        # value_log에 실제 최종 점수(Actual Return) 주입
        for entry in value_log:
            entry["final_score"] = final_scores[entry["model"]]

        # 카드 및 귀족 통계 집계 (KeyError 해결의 핵심)
        for pid in [0, 1]:
            p_obj = game_instance.players[pid]
            m_name = agent_mapping[pid]
            stats[f"{m_name}_nobles"] = len(p_obj.nobles)
            for card in p_obj.cards:
                stats[f"{m_name}_cards"][card.level] += 1
                
        return stats, action_log, score_log, value_log

class Visualizer:
    def __init__(self, df, action_df, score_df, value_df, save_dir="results/plots/ppo_vs_dqn"):
        self.df = df
        self.action_df = action_df
        self.score_df = score_df
        self.value_df = value_df
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
            self.plot_average_score_per_turn()
            self.plot_card_tier_analysis()
            self.plot_action_distribution()
            self.plot_nobles_analysis()
            self.plot_value_estimation()
            
            if not self.action_df.empty:
                for model in ["PPO", "DQN"]:
                    for result in ["Win", "Loss"]:
                        self.plot_turn_action_distribution(model_name=model, result_filter=result)
                
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

    def plot_average_score_per_turn(self):
        if self.score_df.empty: return

        # 1. score_df에 승패 정보 병합
        merged_score = self.score_df.merge(self.df[['game_id', 'winner']], on='game_id', how='left')

        # 2. 케이스별로 데이터 분리
        ppo_win_games = merged_score[merged_score['winner'] == 'PPO']
        ppo_loss_games = merged_score[merged_score['winner'] == 'DQN']
        dqn_win_games = merged_score[merged_score['winner'] == 'DQN']
        dqn_loss_games = merged_score[merged_score['winner'] == 'PPO']

        # 3. 각 케이스별 평균 점수 계산 (Forward Fill 적용)
        def get_avg_trajectory(df_subset, score_col):
            if df_subset.empty: return None
            pivot = df_subset.pivot(index='turn', columns='game_id', values=score_col)
            pivot = pivot.ffill() # 종료된 게임 점수 유지
            return pivot.mean(axis=1)

        ppo_win_curve = get_avg_trajectory(ppo_win_games, 'PPO_score')
        ppo_loss_curve = get_avg_trajectory(ppo_loss_games, 'PPO_score')
        dqn_win_curve = get_avg_trajectory(dqn_win_games, 'DQN_score')
        dqn_loss_curve = get_avg_trajectory(dqn_loss_games, 'DQN_score')

        # 4. 시각화
        plt.figure(figsize=(12, 6))
        
        if ppo_win_curve is not None:
            plt.plot(ppo_win_curve.index, ppo_win_curve.values, label='PPO (Win)', color='blue', linestyle='-', linewidth=2)
        if ppo_loss_curve is not None:
            plt.plot(ppo_loss_curve.index, ppo_loss_curve.values, label='PPO (Loss)', color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
            
        if dqn_win_curve is not None:
            plt.plot(dqn_win_curve.index, dqn_win_curve.values, label='DQN (Win)', color='red', linestyle='-', linewidth=2)
        if dqn_loss_curve is not None:
            plt.plot(dqn_loss_curve.index, dqn_loss_curve.values, label='DQN (Loss)', color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        plt.title('Average Score per Turn (Win vs Loss)')
        plt.xlabel('Turn')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, 'avg_score_per_turn.png')
        plt.savefig(save_path)
        plt.close()

    def plot_card_tier_analysis(self):
        # [KeyError 발생지점] 데이터프레임 생성 시 'PPO_cards' 키가 없으면 에러 발생.
        # 위에서 stats 초기화를 올바르게 했다면 에러가 나지 않음.
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

    def plot_nobles_analysis(self):
        if 'PPO_nobles' not in self.df.columns: return
        noble_df = self.df[['PPO_nobles', 'DQN_nobles']].melt(var_name='Model', value_name='Nobles')
        noble_df['Model'] = noble_df['Model'].str.replace('_nobles', '')
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=noble_df, x='Nobles', hue='Model', palette={'PPO': 'blue', 'DQN': 'red'})
        plt.title('Distribution of Nobles Acquired per Game')
        plt.xlabel('Number of Nobles Acquired')
        plt.ylabel('Game Count')
        plt.legend(title='Agent')
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width()/2., height + 0.1, int(height), ha="center")
        plt.savefig(os.path.join(self.save_dir, 'nobles_dist.png'))
        plt.close()

    def plot_turn_action_distribution(self, model_name, result_filter="Win"):
        subset = self.action_df[
            (self.action_df['model'] == model_name) & 
            (self.action_df['result'] == result_filter)
        ]
        
        if subset.empty:
            print(f" - {model_name} ({result_filter}) 데이터가 없어 그래프를 건너뜁니다.")
            return

        # 3% 미만 표본 절삭 로직
        total_games_in_subset = subset['game_id'].nunique()
        min_sample_threshold = total_games_in_subset * 0.03 

        active_games_per_turn = subset.groupby('turn')['game_id'].nunique()
        valid_turns = active_games_per_turn[active_games_per_turn >= min_sample_threshold].index
        
        if len(valid_turns) == 0:
            print(f" - {model_name} ({result_filter}) 표본 수가 너무 적어 그래프를 그리지 않습니다.")
            return
            
        max_valid_turn = valid_turns.max()
        subset_filtered = subset[subset['turn'] <= max_valid_turn]

        counts = subset_filtered.groupby(['turn', 'action_type']).size().unstack(fill_value=0)
        props = counts.div(counts.sum(axis=1), axis=0)
        
        color_map = {
            'TAKE_THREE_GEMS': '#3498db',
            'TAKE_TWO_GEMS': '#2980b9',
            'BUY_CARD_T1': '#a8e6cf',
            'BUY_CARD_T2': '#2ecc71',
            'BUY_CARD_T3': '#1e8449',
            'BUY_CARD': '#2ecc71',
            'RESERVE_CARD': '#f1c40f',
            'RETURN_GEMS': '#e74c3c'
        }
        colors = [color_map.get(col, '#95a5a6') for col in props.columns]

        plt.figure(figsize=(14, 7))
        props.plot(kind='bar', stacked=True, width=1.0, color=colors, figsize=(14, 7), ax=plt.gca())
        
        plt.title(f'Action Distribution: {model_name} ({result_filter}) (Cutoff at Turn {max_valid_turn})')
        plt.xlabel('Turn Number')
        plt.ylabel('Proportion of Actions')
        plt.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'turn_action_dist_{model_name}_{result_filter}.png')
        plt.savefig(save_path)
        plt.close()
        print(f" - {model_name} ({result_filter}) 행동 그래프 저장됨: {save_path} (Max Turn: {max_valid_turn})")

    def plot_value_estimation(self):
        """
        [Final Clean Version]
        기능:
        1. PPO 승리 / DQN 승리 시나리오를 각각 분리하여 1행 2열로 그립니다.
        2. 전체 평균이 아닌, '플레이 타임 중간값(Median)'을 가진 대표 게임 1개를 선정합니다.
        3. Dual Axis를 사용합니다.
           - Left Axis (점선): 실제 점수 (Actual Score)
           - Right Axis (실선): 예측 Q값 (Predicted Q-Value)
        """
        if self.value_df.empty or self.score_df.empty:
            print("[Skip] 가치 추정 그래프를 그리기 위한 데이터가 부족합니다.")
            return

        # 시나리오 정의: (그래프 제목, 승자 모델명)
        scenarios = [("Scenario A: PPO Wins", "PPO"), ("Scenario B: DQN Wins", "DQN")]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # 데이터 안전하게 그리는 헬퍼 함수 (X, Y 길이 불일치 방지)
        def safe_plot(ax, x, y, **kwargs):
            min_len = min(len(x), len(y))
            return ax.plot(x[:min_len], y[:min_len], **kwargs)

        for idx, (title, winner_name) in enumerate(scenarios):
            ax_score = axes[idx] # 왼쪽 축 (점수용)
            
            # 1. 승자가 'winner_name'인 게임 ID 리스트 찾기
            winning_games = self.df[self.df['winner'] == winner_name]
            
            if winning_games.empty:
                ax_score.text(0.5, 0.5, f"No Data for {winner_name} Wins", ha='center', fontsize=12)
                ax_score.set_title(title)
                continue

            # 2. 대표 게임 선정 (플레이 턴 수가 중간값인 게임)
            # 너무 짧은 게임(광속패)이나 너무 긴 게임(지루한 공방) 제외
            sorted_games = winning_games.sort_values('total_turns')
            median_row = sorted_games.iloc[len(sorted_games) // 2]
            target_game_id = median_row['game_id']
            total_turns = median_row['total_turns']
            
            loser_name = "DQN" if winner_name == "PPO" else "PPO"
            print(f"[{title}] Plotting Game ID: {target_game_id} (Total Turns: {total_turns})")

            # -------------------------------------------------------
            # [Data Fetching] 해당 게임의 데이터만 추출
            # -------------------------------------------------------
            
            # (1) 점수 데이터: 턴별로 유일해야 함 (중복 제거)
            # score_df는 보통 턴당 1줄씩 기록되므로 game_id로 필터링만 하면 됨
            game_score_df = self.score_df[self.score_df['game_id'] == target_game_id].sort_values('turn')
            
            # 턴 인덱스 (X축)
            turns_x = game_score_df['turn'].values
            
            # (2) 예측(Q) 데이터: 모델별로 따로 추출
            game_value_df = self.value_df[self.value_df['game_id'] == target_game_id]
            
            win_q_df = game_value_df[game_value_df['model'] == winner_name].sort_values('turn')
            loss_q_df = game_value_df[game_value_df['model'] == loser_name].sort_values('turn')

            # -------------------------------------------------------
            # [Axis 1: Actual Score] 실제 점수 (점선, 배경)
            # -------------------------------------------------------
            # 승자 점수 (Winner Color)
            l1, = safe_plot(ax_score, turns_x, game_score_df[f'{winner_name}_score'].values,
                            color='blue' if winner_name == 'PPO' else 'red',
                            linestyle=':', linewidth=2, alpha=0.6, label=f'{winner_name} Score')
            
            # 패자 점수 (Gray)
            l2, = safe_plot(ax_score, turns_x, game_score_df[f'{loser_name}_score'].values,
                            color='gray',
                            linestyle=':', linewidth=2, alpha=0.6, label=f'{loser_name} Score')

            ax_score.set_xlabel("Turn")
            ax_score.set_ylabel("Actual Score (0 ~ 15)", fontweight='bold')
            ax_score.set_ylim(-1, 17) # 점수 범위 고정
            ax_score.grid(True, alpha=0.3)

            # -------------------------------------------------------
            # [Axis 2: Predicted Q] 예측 가치 (실선, 메인)
            # -------------------------------------------------------
            ax_q = ax_score.twinx()
            
            # Q값 스케일 자동 감지 (점수 예측인지, 승률 예측인지)
            # 데이터가 비어있을 수 있으므로 예외처리
            max_q_val = 0
            if not win_q_df.empty: max_q_val = max(max_q_val, win_q_df['predicted_value'].max())
            if not loss_q_df.empty: max_q_val = max(max_q_val, loss_q_df['predicted_value'].max())

            if max_q_val > 2.0:
                q_ylim = (-5, 24)
                y_label = "Predicted Score (Q)"
            else:
                q_ylim = (-1.5, 1.5)
                y_label = "Win Probability (Q)"

            # 승자 Q값 (Winner Color, Thick Line)
            l3, = safe_plot(ax_q, win_q_df['turn'].values, win_q_df['predicted_value'].values,
                            color='blue' if winner_name == 'PPO' else 'red',
                            linestyle='-', linewidth=3, label=f'{winner_name} Predicted Q')
            
            # 패자 Q값 (Gray, Thin Line)
            l4, = safe_plot(ax_q, loss_q_df['turn'].values, loss_q_df['predicted_value'].values,
                            color='gray',
                            linestyle='-', linewidth=2, alpha=0.8, label=f'{loser_name} Predicted Q')

            ax_q.set_ylabel(y_label, fontweight='bold', color='purple')
            ax_q.set_ylim(q_ylim)
            ax_q.tick_params(axis='y', labelcolor='purple')

            # -------------------------------------------------------
            # [Legend & Title]
            # -------------------------------------------------------
            # 범례 통합 (Score축과 Q축의 라벨을 한곳에 모음)
            lines = [l1, l2, l3, l4]
            labels = [l.get_label() for l in lines]
            ax_score.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)
            
            ax_score.set_title(f"{title}\n(Game ID: {target_game_id})", fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'value_estimation_single_game.png')
        plt.savefig(save_path)
        plt.close()
        print(f"가치 추정 그래프 저장 완료: {save_path}")

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
        # 4개의 데이터프레임 반환
        df, action_df, score_df, value_df = arena.run_battle(num_games=args.games)
        
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
        
        # 시각화 실행
        viz = Visualizer(df, action_df, score_df, value_df)
        viz.plot_all()
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()