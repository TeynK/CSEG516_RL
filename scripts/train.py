# train.py (DQN/PPO 호환 및 학습 재개 기능 추가)

import os
import yaml
import argparse
import gymnasium
from gymnasium.wrappers import FlattenObservation # [추가] DQN을 위한 래퍼

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from envs.splendor_gym_wrapper import SplendorGymWrapper
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

def get_action_mask_from_env(env: gymnasium.Env):
    """
    환경의 action_mask() 메소드를 호출하는 헬퍼 함수
    """
    # SplendorGymWrapper에 정의된 메소드 호출
    return env.action_mask()

def make_env_sb3(agent_type="PPO"):
    """
    에이전트 타입(PPO/DQN)에 따라 적절한 래퍼를 적용한 환경을 생성합니다.
    """
    env = SplendorGymWrapper(num_players=2) #
    
    if agent_type == "PPO":
        # PPO는 ActionMasker 래퍼를 사용
        env = ActionMasker(env, get_action_mask_from_env)
    elif agent_type == "DQN":
        env = FlattenObservation(env)
        
    return env

def load_config(config_path):
    """
    프로젝트 루트 기준의 설정 파일을 불러옵니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, '..', config_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {full_path}")
        
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_model_class(class_name):
    """
    설정 파일의 문자열(class_name)을 실제 모델 클래스로 변환합니다.
    """
    if class_name == "MaskablePPO" and MaskablePPO:
        return MaskablePPO
    if class_name == "DQN" and DQN:
        return DQN
    # [오류 처리] 지원하지 않는 모델 클래스
    raise ValueError(f"'{class_name}'은(는) 지원하지 않는 모델 클래스입니다. (MaskablePPO 또는 DQN)")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/ppo_config.yaml", #
        help="사용할 설정 파일 경로 (예: configs/ppo_config.yaml 또는 configs/dqn_config.yaml)"
    )
    args = parser.parse_args()
    
    print(f"--- 설정 파일 로드: {args.config} ---")
    config = load_config(args.config)
    
    # --- 설정 값 불러오기 ---
    agent_type = config.get("agent_type", "PPO") # PPO 또는 DQN
    model_class_name = config['model_class']
    ModelClass = get_model_class(model_class_name)
    
    log_dir = os.path.join("results/logs", config['log_dir']) # [수정] logs/ 하위로
    model_dir = os.path.join("results/models", config['model_dir']) # [수정] models/ 하위로
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    num_cpu = config['num_cpu']
    
    # --- 병렬 환경 생성 ---
    # [수정] make_env_sb3에 agent_type 전달
    vec_env = make_vec_env(
        lambda: make_env_sb3(agent_type=agent_type),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv
    )
    
    print(f"--- {num_cpu}개 CPU 코어로 '{model_class_name}' 병렬 학습 시작 ---")
    
    model_params = config.get("model_hyperparameters", {})
    
    # --- [수정] 학습 재개 로직 ---
    load_path = config.get("load_model_path", None)
    
    # load_path가 유효한 파일 경로인지 확인
    is_valid_load_path = load_path and load_path.strip() and os.path.exists(load_path)

    if is_valid_load_path:
        print(f"\n--- 이전 모델을 불러옵니다 ---")
        print(f"경로: {load_path}")
        model = ModelClass.load(
            load_path,
            env=vec_env,
            tensorboard_log=log_dir 
        )
        print("--- 모델 로드 완료 ---")
    
    else:
        if load_path: # 경로가 지정되었으나 파일이 없는 경우
             print(f"경고: load_model_path가 {load_path}로 지정되었으나, 파일을 찾을 수 없습니다. 새 모델을 생성합니다.")
        
        print(f"\n--- 새 '{model_class_name}' 모델을 생성합니다 ---")
        
        # [수정] 에이전트 타입에 따라 올바른 정책(Policy) 설정
        if agent_type == "PPO":
            policy = MaskableActorCriticPolicy
            if "policy" in model_params: # config에 MlpPolicy 등이 있어도 무시
                del model_params["policy"]
        else: # DQN 등
            policy = config.get("policy", "MlpPolicy") 
            if "policy" in model_params:
                del model_params["policy"] # model_params 대신 인자로 전달
        
        model = ModelClass(
            policy=policy,
            env=vec_env, 
            verbose=1, 
            tensorboard_log=log_dir,
            **model_params
        )
        print("--- 모델 생성 완료 ---")

    # --- 학습 파라미터 ---
    total_timesteps = config['total_timesteps'] 
    save_interval = config['save_interval'] 
    model_name = config['model_name'] 
    
    model_save_path = os.path.join(model_dir, model_name)

    print(f"--- {total_timesteps} 타임스텝 학습 시작 ---")
    
    # (total_timesteps // save_interval) 만큼 반복
    iterations = max(1, total_timesteps // save_interval) 
    
    for i in range(1, iterations + 1):
        model.learn(
            total_timesteps=save_interval, 
            reset_num_timesteps=False, # [중요] True로 바꾸면 이어서 학습이 안됨
            tb_log_name=model_name
        )
        save_path = f"{model_save_path}_{i * save_interval}_steps"
        model.save(save_path)
        print(f"--- 모델 저장: {save_path}.zip ---")

    print(f"--- 학습 완료 ---")
    
    final_model_path = f"{model_save_path}_final"
    model.save(final_model_path)
    print(f"최종 모델이 {final_model_path}.zip 에 저장되었습니다.")
    
    vec_env.close()