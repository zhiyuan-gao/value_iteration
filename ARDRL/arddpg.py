import numpy as np
import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
import torch as th

class ActionRobustDDPG(DDPG):
    def __init__(self, *args, alpha=0.1, **kwargs):
        super(ActionRobustDDPG, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.adversary = self.actor
        self.target_adversary = self.actor
        self.adversary.optimizer = self.actor.optimizer
    
    def train(self, gradient_steps: int, batch_size: int) -> None:
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Adversary update
            actions_adversary = self.actor.forward(replay_data.observations)
            q_values_adversary = self.critic.forward(replay_data.observations, actions_adversary)
            adversary_loss = -q_values_adversary.mean()
            
            self.actor.optimizer.zero_grad()
            adversary_loss.backward()
            self.actor.optimizer.step()
            
            # Critic update
            actions_pi = self.actor.forward(replay_data.observations)
            q_values_pi = self.critic.forward(replay_data.observations, actions_pi)
            
            with th.no_grad():
                noise = NormalActionNoise(mean=0., sigma=0.1)
                actions_target = self.actor_target(replay_data.next_observations) + noise()
                q_values_target = self.critic_target(replay_data.next_observations, actions_target)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * q_values_target
            
            current_q = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = th.nn.functional.mse_loss(current_q, target_q)
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # Actor update
            actions_pi = self.actor(replay_data.observations)
            q_values_pi = self.critic(replay_data.observations, actions_pi)
            actor_loss = -q_values_pi.mean()
            
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            # Update target networks
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.soft_update(self.actor_target, self.actor, self.tau)

if __name__ == "__main__":
    import random
    import numpy as np
    import torch

    env = gym.make('Pendulum-v1')

    seed = 42  # Set your desired seed value here
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    action_min = env.action_space.low
    action_max = env.action_space.high

    action_range = (action_min, action_max)


    agent = ActionRobustDDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_range=action_range,
        max_steps_per_episode=200,
        batch_size=64,
        num_episodes=1000,
        hidden_dim1=400,
        hidden_dim2=300
    )
    
    agent.train(env, num_episodes=agent.num_episodes, max_steps_per_episode=agent.max_steps_per_episode, batch_size=agent.batch_size, seed=seed)
    
    # Save the model
    agent.save_checkpoint('ddpg_checkpoint_final.pth')
    
    # Load the model
    agent.load_checkpoint('ddpg_checkpoint_final.pth')
    
    # Continue training or use the agent for inference
    agent.train(env, num_episodes=agent.num_episodes, max_steps_per_episode=agent.max_steps_per_episode, batch_size=agent.batch_size, seed=seed)
