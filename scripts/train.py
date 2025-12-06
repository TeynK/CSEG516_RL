import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
import time

import gymnasium as gym
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from envs.splendor_gym_wrapper import SplendorGymWrapper
from agents.maskable_dqn import MaskableDQN

class RichStatsCallback(BaseCallback):
    def __init__(self, total_timesteps: int, check_freq: int, verbose=1):
        super(RichStatsCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.best_win_rate = -float('inf')
        self.save_path = os.path.join("results", "models", "best_model")
        os.makedirs(self.save_path, exist_ok=True)
        
        self.recent_results = deque(maxlen=100)
        
        self.stats_history = {
            "episode": [], "win_rate": [], 
            "agent_score": [], "agent_turns": [], "agent_nobles": [],
            "agent_total_cards": [], "agent_t1": [], "agent_t2": [], "agent_t3": [],
            "agent_reserve_count": []
        }
        self.episode_count = 0
        self.start_time = None
        self.current_win_rate = 0.0
        self.last_game_result = "N/A"
        
        self.live = None
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total} Steps"),
        )
        self.task_id = self.progress.add_task("[cyan]Training...", total=total_timesteps)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        layout = self.make_layout()
        self.live = Live(layout, refresh_per_second=4, console=self.console)
        self.live.start()

    def _on_training_end(self) -> None:
        if self.live:
            self.live.stop()

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        return layout

    def generate_table(self) -> Table:
        table = Table(title="Training Status", expand=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0

        table.add_row("Total Episodes", str(self.episode_count))
        table.add_row("Current Timesteps", f"{self.num_timesteps:,}")
        table.add_row("FPS", str(fps))
        
        win_rate_str = f"{self.current_win_rate * 100:.1f}%"
        table.add_row("Win Rate (Last 100)", win_rate_str)
        
        best_str = f"{self.best_win_rate * 100:.1f}%"
        table.add_row("Best Win Rate", f"[bold green]{best_str}[/]")
        
        result_color = "green" if self.last_game_result == "WIN" else "red" if self.last_game_result == "LOSE" else "white"
        table.add_row("Last Game Result", f"[{result_color}]{self.last_game_result}[/]")
        
        return Panel(table, title="General Metrics", border_style="blue")

    def generate_agent_stats(self) -> Table:
        table = Table(title="In-Game Analysis (RL Agent)", expand=True)
        table.add_column("Stat", style="yellow")
        table.add_column("Value", style="white")

        if self.stats_history["agent_score"]:
            avg_score = np.mean(self.stats_history["agent_score"][-50:])
            avg_turns = np.mean(self.stats_history["agent_turns"][-50:])
            avg_nobles = np.mean(self.stats_history["agent_nobles"][-50:])
            
            avg_total = np.mean(self.stats_history["agent_total_cards"][-50:])
            avg_t1 = np.mean(self.stats_history["agent_t1"][-50:])
            avg_t2 = np.mean(self.stats_history["agent_t2"][-50:])
            avg_t3 = np.mean(self.stats_history["agent_t3"][-50:])
            
            avg_reserve = np.mean(self.stats_history["agent_reserve_count"][-50:])
        else:
            avg_score, avg_turns, avg_nobles, avg_total, avg_t1, avg_t2, avg_t3, avg_reserve = 0,0,0,0,0,0,0,0

        table.add_row("Avg Score", f"{avg_score:.2f}")
        table.add_row("Avg Turns", f"{avg_turns:.1f}")
        table.add_row("Avg Nobles", f"{avg_nobles:.2f}")
        table.add_row("Avg Cards Total", f"{avg_total:.1f}")
        table.add_row(" - Tier 1", f"{avg_t1:.1f}")
        table.add_row(" - Tier 2", f"{avg_t2:.1f}")
        table.add_row(" - Tier 3", f"{avg_t3:.1f}")
        table.add_row("Avg Reserves", f"{avg_reserve:.2f}")
        
        return Panel(table, title="Agent Behavior (Last 50 Games)", border_style="green")

    def _on_step(self) -> bool:
        self.progress.update(self.task_id, completed=self.num_timesteps)

        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                info = infos[i]
                episode_reward = info.get('episode', {}).get('r', 0)
                
                is_win = 1 if episode_reward > 0 else 0
                self.recent_results.append(is_win)
                self.last_game_result = "WIN" if is_win else "LOSE"
                
                agent_stats = info.get("agent_stats")
                
                if agent_stats:
                    self.current_win_rate = np.mean(self.recent_results) if self.recent_results else 0.0
                    
                    self.stats_history["episode"].append(self.episode_count)
                    self.stats_history["win_rate"].append(self.current_win_rate)
                    
                    self.stats_history["agent_score"].append(agent_stats['score'])
                    self.stats_history["agent_turns"].append(agent_stats['turn_count'])
                    self.stats_history["agent_nobles"].append(agent_stats['noble_count'])
                    
                    self.stats_history["agent_total_cards"].append(agent_stats['total_cards'])
                    self.stats_history["agent_t1"].append(agent_stats['tier_1_count'])
                    self.stats_history["agent_t2"].append(agent_stats['tier_2_count'])
                    self.stats_history["agent_t3"].append(agent_stats['tier_3_count'])
                    self.stats_history["agent_reserve_count"].append(agent_stats['reserved_count'])

                    if len(self.recent_results) >= 100:
                        if self.current_win_rate > self.best_win_rate:
                            self.best_win_rate = self.current_win_rate

        if self.live:
            layout = self.live.get_renderable()
            layout["header"].update(Panel(f"[bold white]Splendor RL Training[/] [dim]({datetime.now().strftime('%H:%M:%S')})[/]", style="bold blue"))
            layout["left"].update(self.generate_table())
            layout["right"].update(self.generate_agent_stats())
            layout["footer"].update(self.progress)
            
        return True

    def save_plots(self, save_path, model_name):
        if not self.stats_history["episode"]: return
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Analysis: {model_name} (RL Agent)', fontsize=16)

        axs[0, 0].plot(self.stats_history["episode"], self.stats_history["win_rate"], color='blue')
        axs[0, 0].set_title('Win Rate Trend')

        axs[0, 1].hist(self.stats_history["agent_score"], bins=range(0, 20), color='green', alpha=0.7)
        axs[0, 1].set_title('Final Score Distribution')

        axs[0, 2].plot(self.stats_history["episode"], self.stats_history["agent_turns"], color='brown', alpha=0.5)
        axs[0, 2].set_title('Game Duration (Turns)')
        axs[0, 2].invert_yaxis()

        def moving_average(a, n=50):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        episodes = self.stats_history["episode"]
        if len(episodes) > 50:
            ma_episodes = episodes[49:]
            ma_t1 = moving_average(self.stats_history["agent_t1"])
            ma_t2 = moving_average(self.stats_history["agent_t2"])
            ma_t3 = moving_average(self.stats_history["agent_t3"])
            
            axs[1, 0].plot(ma_episodes, ma_t1, label='Tier 1', color='skyblue')
            axs[1, 0].plot(ma_episodes, ma_t2, label='Tier 2', color='orange')
            axs[1, 0].plot(ma_episodes, ma_t3, label='Tier 3', color='red')
            axs[1, 0].set_title('Card Tier Composition (MA)')
            axs[1, 0].legend()
        else:
            axs[1, 0].text(0.5, 0.5, 'Need > 50 episodes', ha='center')

        axs[1, 1].plot(self.stats_history["episode"], self.stats_history["agent_reserve_count"], color='purple', alpha=0.5)
        axs[1, 1].set_title('Reserved Cards Count')

        axs[1, 2].plot(self.stats_history["episode"], self.stats_history["agent_nobles"], color='gold', alpha=0.8)
        axs[1, 2].set_title('Nobles Acquired')

        plt.tight_layout()
        plot_file = os.path.join(save_path, f"{model_name}_analysis.png")
        plt.savefig(plot_file)
        self.console.print(f"[bold green]Analysis plot saved to {plot_file}[/bold green]")
        plt.close()

def make_env(rank: int, seed: int = 0):
    def _init():
        env = SplendorGymWrapper()
        env = Monitor(env)
        env = ActionMasker(env, lambda env: env.unwrapped.action_mask())
        env.reset(seed=seed + rank) 
        return env
    set_random_seed(seed)
    return _init

def load_config(model_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, "configs", f"{model_type.lower()}_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Train Splendor RL Agent")
    parser.add_argument("--model", type=str, required=True, choices=["PPO", "DQN"], help="Model type to train")
    parser.add_argument("--n_envs", type=int, default=None, help="Number of parallel environments (Overrides config)")
    args = parser.parse_args()
    model_type = args.model.upper()
    console = Console()
    try:
        config = load_config(model_type)
        console.print(f"[green]Loaded config for {model_type}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        return

    if args.n_envs is not None:
        n_envs = args.n_envs
    else:
        n_envs = config.get("num_cpu", 4)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    yaml_model_dir = config.get("model_dir", "results/models")
    if not os.path.isabs(yaml_model_dir):
        model_save_dir = os.path.join(project_root, yaml_model_dir)
    else:
        model_save_dir = yaml_model_dir
        
    yaml_log_dir = config.get("log_dir", "results/logs")
    if not os.path.isabs(yaml_log_dir):
        tensorboard_log_dir = os.path.join(project_root, yaml_log_dir)
    else:
        tensorboard_log_dir = yaml_log_dir

    model_name_base = config.get("model_name", f"{model_type}_vs_bot")
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    plot_save_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(plot_save_dir, exist_ok=True)

    model_path = os.path.join(model_save_dir, f"{model_name_base}_final.zip")

    console.print(f"\n[bold blue][{model_type}] Training Start...[/bold blue]")
    console.print(f"Parallel Environments: {n_envs}")
    console.print(f"TensorBoard Logs: {tensorboard_log_dir}")
    console.print(f"Model Save Path: {model_path}")
    
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    model_hyperparams = config.get("model_hyperparameters", {})
    total_timesteps = config.get("total_timesteps", 1000000)

    if os.path.exists(model_path):
        console.print(f"[yellow]Loading existing model from {model_path}[/yellow]")
        if model_type == "PPO":
            model = MaskablePPO.load(model_path, env=env, tensorboard_log=tensorboard_log_dir, **model_hyperparams)
        elif model_type == "DQN":
            model = MaskableDQN.load(model_path, env=env, tensorboard_log=tensorboard_log_dir, **model_hyperparams)
    else:
        console.print("[green]Creating new model...[/green]")
        if model_type == "PPO":
            model = MaskablePPO(
                "MultiInputPolicy", 
                env, 
                verbose=0, 
                tensorboard_log=tensorboard_log_dir, 
                **model_hyperparams
            )
        elif model_type == "DQN":
            model = MaskableDQN(
                "MultiInputPolicy", 
                env, 
                verbose=0, 
                tensorboard_log=tensorboard_log_dir, 
                **model_hyperparams
            )

    stats_callback = RichStatsCallback(
        total_timesteps=total_timesteps,
        check_freq=1000
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=stats_callback,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        console.print("\n[bold red]Training interrupted by user.[/bold red]")
    finally:
        if stats_callback.live:
            stats_callback.live.stop()
        env.close()

    console.print(f"Saving model to {model_path}...", style="bold cyan")
    model.save(model_path)
    
    stats_callback.save_plots(plot_save_dir, model_name_base)
    console.print("[bold green]Training Finished.[/bold green]")

if __name__ == "__main__":
    main()