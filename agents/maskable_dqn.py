import torch as th
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Dict, Optional, Tuple

class MaskableDQN(DQN):
    def _setup_model(self):
        super()._setup_model()
        self.replay_buffer = DictReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        for _ in range(gradient_steps):
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            real_obs = replay_data.observations
            real_next_obs = replay_data.next_observations
            next_action_mask = replay_data.next_observations["action_mask"]
            with th.no_grad():
                next_q_values_main = self.q_net(real_next_obs)
                masked_next_q_main = th.where(
                    next_action_mask.bool(),
                    next_q_values_main,
                    th.tensor(-float("inf")).to(self.device)
                )
                next_actions = masked_next_q_main.argmax(dim=1).unsqueeze(-1)
                next_q_values_target = self.q_net_target(real_next_obs)
                next_q_values = th.gather(next_q_values_target, dim=1, index=next_actions).squeeze(-1)
                all_masked = (next_action_mask.float().sum(dim=1) == 0)
                next_q_values[all_masked] = 0.0
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            current_q_values = self.q_net(real_obs)
            action_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions)
            loss = th.nn.functional.smooth_l1_loss(action_q_values, target_q_values)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

    def predict(self, observation: Dict[str, np.ndarray], state: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        action_mask = observation["action_mask"]
        obs_tensor, _ = self.policy.obs_to_tensor(observation)
        mask_tensor = th.as_tensor(action_mask).to(self.device)
        with th.no_grad():
            q_values = self.policy.q_net(obs_tensor)
            masked_q_values = th.where(
                mask_tensor.bool(),
                q_values,
                th.tensor(-float("inf")).to(self.device)
            )
            exploitation_actions = masked_q_values.argmax(dim=1)
        if not deterministic and np.random.rand() < self.exploration_rate:
            float_mask = mask_tensor.float()
            all_invalid_mask = (float_mask.sum(dim=1) == 0)
            safe_mask = float_mask.clone()
            safe_mask[all_invalid_mask, 0] = 1.0
            exploration_actions = th.multinomial(safe_mask, num_samples=1).squeeze(1)
            action = exploration_actions
        else:
            action = exploitation_actions
        action = action.reshape(-1)
        return action.cpu().numpy(), state
            