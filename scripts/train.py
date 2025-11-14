# train_ppo.py
# [수정됨] 'ActionMasker'에 'action_mask_fn'을 수동으로 전달

import os
import yaml
import argparse
import gymnasium

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from envs.splendor_gym_wrapper import SplendorGymWrapper
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN

def get_action_mask_from_env(env):
    return env.action_mask()

def make_env_sb3(use_action_masker=True):
    env = SplendorGymWrapper(num_players=2)
    if use_action_masker:
        env = ActionMasker(env, get_action_mask_from_env)
    return env

def load_config(config_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, config_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {full_path}")
        
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_model_class(class_name):
    if class_name == "MaskablePPO" and MaskablePPO:
        return MaskablePPO
    if class_name == "DQN" and DQN:
        return DQN
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/ppo_config.yaml", 
        help="사용할 설정 파일 경로"
    )
    args = parser.parse_args()
    print(f"--- 설정 파일 로드: {args.config} ---")
    config = load_config(args.config)
    agent_type = config.get("agent_type", "PPO")
    model_class_name = config['model_class']
    ModelClass = get_model_class(model_class_name)
    log_dir = config['log_dir']
    model_dir = config['model_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    num_cpu = config['num_cpu']
    use_masker = (agent_type == "PPO") 
    
    vec_env = make_vec_env(
        lambda: make_env_sb3(use_action_masker=use_masker),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv
    )
    print(f"--- {num_cpu}개 CPU 코어로 '{model_class_name}' 병렬 학습 시작 ---")
    model_params = config.get("model_hyperparameters", {})
    model_params["policy"] = config.get("policy", "MlpPolicy") 

    model = ModelClass(
        env=vec_env, 
        verbose=1, 
        tensorboard_log=log_dir,
        **model_params
    )
    total_timesteps = config['total_timesteps'] 
    save_interval = config['save_interval'] 
    model_name = config['model_name'] 

    print(f"--- {total_timesteps} 타임스텝 학습 시작 ---")
    for i in range(1, (total_timesteps // save_interval) + 1):
        model.learn(
            total_timesteps=save_interval, 
            reset_num_timesteps=False, 
            tb_log_name=model_name
        )
        model.save(f"{model_dir}{model_name}_{i * save_interval}_steps")
        print(f"--- 모델 저장: {model_dir}{model_name}_{i * save_interval}_steps ---")

    print(f"--- 학습 완료 ---")
    
    final_model_path = f"{model_dir}{model_name}_final"
    model.save(final_model_path)
    print(f"최종 모델이 {final_model_path}.zip 에 저장되었습니다.")
    
    vec_env.close()
    
