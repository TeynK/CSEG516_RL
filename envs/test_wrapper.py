# test_wrapper.py
# (splendor_rl_project/envs/ 폴더에 생성)

import numpy as np

from pettingzoo.test import api_test
from .splendor_wrapper import raw_env # raw_env를 임포트

# 2인플 환경으로 테스트
env = raw_env(num_players=2)

print("PettingZoo API 테스트 시작...")
try:
    # api_test는 reset, step, observe 등이
    # PettingZoo 규격에 맞게 작동하는지 자동으로 테스트합니다.
    api_test(env, num_cycles=1000, verbose_progress=True)
    
    print("\nAPI 테스트 통과!")
    print("수동 reset/step 테스트 시작...")
    
    # 수동 테스트 (몇 스텝 진행)
    env.reset()
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        
        if terminated or truncated:
            action = None
        else:
            # action_mask에서 유효한 행동 중 하나를 무작위로 선택
            action_mask = obs["action_mask"]
            legal_actions = np.where(action_mask == 1)[0]
            if len(legal_actions) == 0:
                print(f"[경고] {agent}가 할 수 있는 행동이 없습니다.")
                action = env.action_space(agent).sample() # 강제 샘플링
            else:
                action = np.random.choice(legal_actions)
        
        env.step(action)
        
        if env.terminations[agent] or env.truncations[agent]:
            print(f"에이전트 {agent} 종료됨.")
            break
            
    print("수동 테스트 완료.")

except AssertionError as e:
    print("\n[!!!] PettingZoo API 테스트 실패:")
    print(e)