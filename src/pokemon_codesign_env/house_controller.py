"""Morphology-conditioned recurrent PPO controller for House of Dextra."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class MorphologyConditionedGRUActorCritic(nn.Module):
    """Shared GRU actor-critic matching the House controller architecture."""

    def __init__(
        self,
        observation_space,
        action_count: int,
        morphology_dimension: int,
        hidden_size: int = 256,
        mlp_sizes: tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()
        self.feature_keys = tuple(key for key in observation_space.spaces if key != "action_mask")
        observation_dimension = sum(
            int(np.prod(observation_space[key].shape)) for key in self.feature_keys
        )
        self.input_dimension = observation_dimension + morphology_dimension
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(self.input_dimension, hidden_size)
        self.gru_norm = nn.LayerNorm(hidden_size)
        layers: list[nn.Module] = []
        width = self.input_dimension + hidden_size
        for output_width in mlp_sizes:
            layers.extend((nn.Linear(width, output_width), nn.ELU()))
            width = output_width
        self.mlp = nn.Sequential(*layers)
        self.action_head = nn.Linear(width, action_count)
        self.value_head = nn.Linear(width, 1)

    def initial_state(self, environments: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(environments, self.hidden_size, device=device)

    def _features(self, observations: dict[str, torch.Tensor], morphology: torch.Tensor):
        state = torch.cat(
            [
                observations[key].float().reshape(observations[key].shape[0], -1)
                for key in self.feature_keys
            ],
            dim=-1,
        )
        return torch.cat((state, morphology.float()), dim=-1)

    def step(
        self,
        observations: dict[str, torch.Tensor],
        morphology: torch.Tensor,
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        hidden = hidden * (~episode_starts.bool()).float().unsqueeze(-1)
        inputs = self._features(observations, morphology)
        next_hidden = self.gru(inputs, hidden)
        latent = self.mlp(torch.cat((inputs, self.gru_norm(next_hidden)), dim=-1))
        logits = self.action_head(latent)
        action_mask = observations["action_mask"].bool()
        no_legal_action = ~action_mask.any(dim=-1)
        if no_legal_action.any():
            action_mask = action_mask.clone()
            action_mask[no_legal_action, 0] = True
        distribution = Categorical(logits=logits.masked_fill(~action_mask, float("-inf")))
        return distribution, self.value_head(latent).squeeze(-1), next_hidden

    def sequence(
        self,
        observations: dict[str, torch.Tensor],
        morphology: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[list[Categorical], torch.Tensor]:
        hidden = initial_hidden
        distributions = []
        values = []
        for time_index in range(morphology.shape[0]):
            step_observations = {key: value[time_index] for key, value in observations.items()}
            distribution, value, hidden = self.step(
                step_observations,
                morphology[time_index],
                hidden,
                episode_starts[time_index],
            )
            distributions.append(distribution)
            values.append(value)
        return distributions, torch.stack(values)


@dataclass(frozen=True, slots=True)
class RecurrentPPOConfig:
    rollout_steps: int = 32
    epochs: int = 5
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coefficient: float = 4.0
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 1.0


def _tensor_observation(observation, device):
    return {key: torch.as_tensor(value, device=device) for key, value in observation.items()}


def train_recurrent_ppo(
    policy: MorphologyConditionedGRUActorCritic,
    env,
    morphology: np.ndarray,
    total_timesteps: int,
    config: RecurrentPPOConfig,
    *,
    device: torch.device,
    on_rollout_end: Callable[[int], np.ndarray] | None = None,
) -> list[dict[str, float]]:
    """Train masked recurrent PPO, preserving sequences across each rollout."""

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    observation = env.reset()
    environments = env.num_envs
    hidden = policy.initial_state(environments, device)
    episode_starts = np.ones(environments, dtype=bool)
    history = []
    completed_timesteps = 0

    while completed_timesteps < total_timesteps:
        steps = min(
            config.rollout_steps,
            (total_timesteps - completed_timesteps + environments - 1) // environments,
        )
        observations = {key: [] for key in observation}
        morphologies = []
        starts = []
        actions = []
        old_log_probabilities = []
        rewards = []
        dones = []
        values = []
        initial_hidden = hidden.detach()

        for _ in range(steps):
            tensor_observation = _tensor_observation(observation, device)
            tensor_morphology = torch.as_tensor(morphology, dtype=torch.float32, device=device)
            tensor_starts = torch.as_tensor(episode_starts, device=device)
            with torch.no_grad():
                distribution, value, hidden = policy.step(
                    tensor_observation, tensor_morphology, hidden, tensor_starts
                )
                action = distribution.sample()
                log_probability = distribution.log_prob(action)
            for key, value_array in observation.items():
                observations[key].append(np.asarray(value_array).copy())
            morphologies.append(morphology.copy())
            starts.append(episode_starts.copy())
            actions.append(action.cpu().numpy())
            old_log_probabilities.append(log_probability.cpu().numpy())
            values.append(value.cpu().numpy())
            observation, reward, done, _ = env.step(action.cpu().numpy())
            rewards.append(np.asarray(reward).copy())
            dones.append(np.asarray(done).copy())
            episode_starts = np.asarray(done, dtype=bool)

        with torch.no_grad():
            _, bootstrap_values, _ = policy.step(
                _tensor_observation(observation, device),
                torch.as_tensor(morphology, dtype=torch.float32, device=device),
                hidden,
                torch.as_tensor(episode_starts, device=device),
            )
        reward_array = np.asarray(rewards, dtype=np.float32)
        done_array = np.asarray(dones, dtype=bool)
        value_array = np.asarray(values, dtype=np.float32)
        advantages = np.zeros_like(reward_array)
        last_advantage = np.zeros(environments, dtype=np.float32)
        next_values = bootstrap_values.cpu().numpy()
        for time_index in reversed(range(steps)):
            not_done = 1.0 - done_array[time_index].astype(np.float32)
            delta = (
                reward_array[time_index]
                + config.gamma * next_values * not_done
                - value_array[time_index]
            )
            last_advantage = delta + config.gamma * config.gae_lambda * not_done * last_advantage
            advantages[time_index] = last_advantage
            next_values = value_array[time_index]
        returns = advantages + value_array

        tensor_observations = {
            key: torch.as_tensor(np.asarray(value), device=device)
            for key, value in observations.items()
        }
        tensor_morphologies = torch.as_tensor(np.asarray(morphologies), device=device)
        tensor_starts = torch.as_tensor(np.asarray(starts), device=device)
        tensor_actions = torch.as_tensor(np.asarray(actions), device=device)
        old_logs = torch.as_tensor(np.asarray(old_log_probabilities), device=device)
        tensor_advantages = torch.as_tensor(advantages, device=device)
        tensor_advantages = (tensor_advantages - tensor_advantages.mean()) / (
            tensor_advantages.std() + 1e-8
        )
        tensor_returns = torch.as_tensor(returns, device=device)

        for _ in range(config.epochs):
            distributions, predicted_values = policy.sequence(
                tensor_observations,
                tensor_morphologies,
                initial_hidden,
                tensor_starts,
            )
            new_logs = torch.stack(
                [
                    distribution.log_prob(tensor_actions[index])
                    for index, distribution in enumerate(distributions)
                ]
            )
            entropy = torch.stack([distribution.entropy() for distribution in distributions])
            ratio = (new_logs - old_logs).exp()
            unclipped = ratio * tensor_advantages
            clipped = ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * tensor_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (predicted_values - tensor_returns).square().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

        completed_timesteps += steps * environments
        history.append(
            {
                "timesteps": float(completed_timesteps),
                "mean_reward": float(reward_array.mean()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
            }
        )
        if on_rollout_end is not None and completed_timesteps < total_timesteps:
            morphology = on_rollout_end(completed_timesteps)
            observation = env.reset()
            hidden = policy.initial_state(environments, device)
            episode_starts = np.ones(environments, dtype=bool)

    return history
