# train_ppo.py
# [수정됨] 'ActionMasker'에 'action_mask_fn'을 수동으로 전달

import os
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker

from train_wrapper import SplendorGymWrapper

# --- [새로 추가된 헬퍼 함수] ---
def get_action_mask_from_env(env):
    """
    ActionMasker에 전달할 콜백 함수입니다.
    이제 SplendorGymWrapper에 .action_mask() 메서드가 있으므로
    이 호출은 성공합니다.
    """
    return env.action_mask()

def make_env():
    """
    환경을 생성하고 ActionMasker 래퍼로 감싸는 함수입니다.
    """
    env = SplendorGymWrapper(num_players=2)
    
    # [수정] ActionMasker 래퍼를 다시 사용합니다.
    env = ActionMasker(env, get_action_mask_from_env) 
    return env


if __name__ == "__main__":
    
    # --- 1. 학습 환경 설정 ---
    log_dir = "./logs/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    num_cpu = 8 
    
    print(f"--- {num_cpu}개의 CPU 코어를 사용하여 'MaskablePPO' + 'ActionMasker' 병렬 학습 시작 ---")

    # 'make_env' 헬퍼 함수를 전달 (이전과 동일)
    vec_env = make_vec_env(
        make_env,
        n_envs=num_cpu, 
        vec_env_cls=SubprocVecEnv
    )

    # --- 2. 모델(MaskablePPO) 정의 ---
    
    # "MlpPolicy" 사용 (이전과 동일)
    model = MaskablePPO(
        "MlpPolicy",
        vec_env, 
        verbose=1, 
        tensorboard_log=log_dir,
    )

    # --- 3. 학습 시작 ---
    # (이하 코드는 수정할 필요 없음)
    total_timesteps = 1_000_000 
    save_interval = 100_000 
    model_name = "maskable_ppo_splendor_vs_bot" 

    print(f"--- {total_timesteps} 타임스텝 학습 시작 ---")
    
    for i in range(1, (total_timesteps // save_interval) + 1):
        model.learn(
            total_timesteps=save_interval, 
            reset_num_timesteps=False, 
            tb_log_name=model_name
        )
        model.save(f"{model_dir}/{model_name}_{i * save_interval}_steps")
        print(f"--- 모델 저장: {model_name}_{i * save_interval}_steps ---")

    print(f"--- 학습 완료 ---")
    
    final_model_path = f"{model_dir}/{model_name}_final"
    model.save(final_model_path)
    print(f"최종 모델이 {final_model_path}.zip 에 저장되었습니다.")
    
    vec_env.close()