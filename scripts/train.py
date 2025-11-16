import os
import yaml
import argparse
import sys
from typing import Dict, Any, Type, Callable

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from envs.splendor_gym_wrapper import SplendorGymWrapper
from agents.maskable_dqn import MaskableDQN

def load_config(config_path: str) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, '..', config_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"--- Config loaded from: {full_path} ---")
    return config

def get_model_class(class_name: str) -> Type:
    if class_name == "MaskablePPO":
        return MaskablePPO
    elif class_name == "DQN":
        return DQN
    elif class_name == "MaskableDQN":
        return MaskableDQN
    else:
        raise ValueError(f"Unknown model class: {class_name}. (Supported: MaskablePPO, DQN, MaskableDQN)")

def make_env_thunk(model_class_name: str) -> Callable:
    """
    환경을 생성하는 Thunk 함수.
    MaskablePPO일 때만 ActionMasker 래퍼를 적용합니다.
    """
    def _thunk():
        env = SplendorGymWrapper(num_players=2)
        if model_class_name == "MaskablePPO":
            env = ActionMasker(env, lambda env: env.action_mask())
        return env
    return _thunk
# --- [수정 끝] ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_config.yaml",
                        help="Path to the config file (e.g., configs/ppo_config.yaml or configs/dqn_config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)

    log_dir = config.get("log_dir", "PPO")
    model_dir = config.get("model_dir", "PPO")
    
    full_log_dir = os.path.join("results", "logs", log_dir)
    full_model_dir = os.path.join("results", "models", model_dir)
    os.makedirs(full_log_dir, exist_ok=True)
    os.makedirs(full_model_dir, exist_ok=True)

    num_cpu = config.get("num_cpu", 8)
    
    # --- [수정된 부분 2] ---
    # model_class_name을 env 생성 전에 미리 가져옴
    model_class_name = config.get("model_class")
    ModelClass = get_model_class(model_class_name)
    # --- [수정 끝] ---

    model_params = config.get("model_hyperparameters", {})
    load_path = config.get("load_model_path", "")

    # --- [수정된 부분 3] ---
    # env_thunk에 model_class_name 전달
    env_thunk = make_env_thunk(model_class_name)
    vec_env = make_vec_env(env_thunk, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    # --- [수정 끝] ---

    print(f"--- Using {num_cpu} CPUs for '{model_class_name}' parallel training ---")

    model = None
    is_valid_load_path = load_path and os.path.exists(load_path)

    if is_valid_load_path:
        print(f"--- Loading existing model from: {load_path} ---")
        model = ModelClass.load(load_path, env=vec_env, tensorboard_log=full_log_dir)
        print("--- Model loaded. Resuming training ---")
    else:
        if load_path:
            print(f"--- WARNING: load_model_path not found, creating new model: {load_path} ---")
        print(f"--- Creating new '{model_class_name}' model ---")
        
        policy_name = config.get("policy")
        policy_to_use = None

        if model_class_name == "MaskablePPO":
            policy_to_use = "MultiInputPolicy"
            print(f"Using policy: MaskableActorCriticPolicy (required for MaskablePPO)")
        elif policy_name:
            policy_to_use = policy_name
            print(f"Using policy: {policy_name} (from config)")
        else:
            # MaskableDQN은 내부적으로 DQNPolicy를 사용하므로,
            # policy_name이 None이어도 괜찮지만, 명시성을 위해 MultiInputPolicy를 권장합니다.
            raise ValueError("Policy must be specified in config if not using MaskablePPO")
            
        model = ModelClass(
            policy=policy_to_use,
            env=vec_env,
            verbose=1,
            tensorboard_log=full_log_dir,
            **model_params
        )

    total_timesteps = config.get("total_timesteps", 1000000)
    save_interval = config.get("save_interval", 100000)
    model_name = config.get("model_name", "model")

    iterations = max(1, total_timesteps // save_interval)
    timesteps_per_iteration = save_interval

    print(f"--- Starting training for {total_timesteps} timesteps ({iterations} iterations of {timesteps_per_iteration} steps) ---")

    try:
        for i in range(iterations):
            model.learn(
                total_timesteps=timesteps_per_iteration,
                reset_num_timesteps=(i == 0 and not is_valid_load_path),
                tb_log_name=model_name
            )
            
            steps = (i + 1) * timesteps_per_iteration
            save_path = os.path.join(full_model_dir, f"{model_name}_{steps}_steps.zip")
            model.save(save_path)
            print(f"--- Checkpoint saved to: {save_path} ---")
    
    except KeyboardInterrupt:
        print("\n--- Training interrupted by user. Saving model... ---")
        interrupted_path = os.path.join(full_model_dir, f"{model_name}_interrupted.zip")
        model.save(interrupted_path)
        print(f"Model saved to: {interrupted_path}")

    print("--- Training complete ---")
    final_model_path = os.path.join(full_model_dir, f"{model_name}_final.zip")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    vec_env.close()