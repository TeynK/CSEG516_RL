from envs.splendor_wrapper import env as splendor_env
from pettingzoo.test import api_test

if __name__ == "__main__":
    env = splendor_env(num_players=2)
    api_test(env, num_cycles=1000000, verbose_progress=False)
    print("PettingZoo API 테스트 통과!")