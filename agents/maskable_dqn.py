import torch as th
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Type, Any, Dict, Optional, Tuple

class MaskableDQN(DQN):
    def _setup_model(self):
        original_obs_space = self.observation_space
        self.observation_space = self.observation_space.spaces["observation"]
        super()._setup_model()
        self.observation_space = original_obs_space
        self.replay_buffer = DictReplayBuffer(self.buffer_size, self.observation_space, self.action_space, self.device, n_envs=self.n_envs, optimize_memory_usage=self.optimize_memory_usage)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        for _ in range(gradient_steps):
            replay_data: ReplayBufferSamples = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            real_obs = replay_data.observations["observation"]
            real_next_obs = replay_data.next_observations["observation"]
            next_action_mask = replay_data.next_observations["action_mask"]
            with th.no_grad():
                next_q_values = self.q_net_target(real_next_obs)
                masked_next_q = th.where(
                    next_action_mask.bool(),
                    next_q_values,
                    th.tensor(-float("inf")).to(self.device)
                )
                next_q_values, _ = masked_next_q.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            current_q_values = self.q_net(real_obs)
            action_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions)
            loss = th.nn.functional.smooth_l1_loss(action_q_values, target_q_values)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
        self._on_train_step_end()

    def predict(self, observation: Dict[str, np.ndarray], state: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        real_obs = observation["observation"]
        action_mask = observation["action_mask"]
        obs_tensor, _ = self.obs_to_tensor(real_obs)
        mask_tensor = th.as_tensor(action_mask).to(self.device)
        with th.no_grad():
            q_values = self.q_net(obs_tensor)
            masked_q_values = th.where(
                mask_tensor.bool(),
                q_values,
                th.tensor(-float("inf")).to(self.device)
            )
        if not deterministic and np.random.rand() < self.exploration_rate:
            batch_size = mask_tensor.shape[0]
            actions = []
            for i in range(batch_size):
                valid_indices = th.where(mask_tensor[i])[0]
                if len(valid_indices) > 0:
                    actions.append(np.random.choice(valid_indices.cpu().numpy()))
                else:
                    actions.append(masked_q_values[i].argmax().item())
            action = th.as_tensor(actions, device=self.device)
        else:
            action = masked_q_values.argmax(dim=1)
        action = action.reshape(-1)
        return action.cpu().numpy(), state
            