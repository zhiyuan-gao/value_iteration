import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=400, hidden_dim2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Using tanh to constrain output to [-1, 1]
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=400, hidden_dim2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)  # No activation function on the output layer
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Linear output
        return x

class Adversary(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=400, hidden_dim2=300):
        super(Adversary, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Using tanh to constrain output to [-1, 1]
        return x

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

class ActionRobustDDPG:
    def __init__(self, state_dim, action_dim, action_range, log_dir='logs', max_buffer_size=100000, actor_lr=1e-4, critic_lr=1e-3, adversary_lr=1e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, actor_update_steps=10, hidden_dim1=400, hidden_dim2=300,
                 max_steps_per_episode=200, batch_size=64, num_episodes=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.actor = Actor(state_dim, action_dim, hidden_dim1, hidden_dim2)
        self.critic = Critic(state_dim, action_dim, hidden_dim1, hidden_dim2)
        self.adversary = Adversary(state_dim, action_dim, hidden_dim1, hidden_dim2)
        
        self.actor_target = Actor(state_dim, action_dim, hidden_dim1, hidden_dim2)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim1, hidden_dim2)
        self.adversary_target = Adversary(state_dim, action_dim, hidden_dim1, hidden_dim2)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.adversary_target.load_state_dict(self.adversary.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.adversary_optimizer = optim.Adam(self.adversary.parameters(), lr=adversary_lr)
        
        self.replay_buffer = ReplayBuffer(max_size=max_buffer_size)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.actor_update_steps = actor_update_steps
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.max_steps_per_episode = max_steps_per_episode
        self.batch_size = batch_size
        self.num_episodes = num_episodes

        self.writer = SummaryWriter(log_dir)

    def update_networks(self, batch_size, step):
        for _ in range(self.actor_update_steps):
            states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)
            
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)

            # Update actor
            actor_loss = self.compute_actor_loss(states)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update critic
            critic_loss = self.compute_critic_loss(states, actions, rewards, next_states)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Log losses to TensorBoard
            self.writer.add_scalar('Loss/Actor', actor_loss.item(), step)
            self.writer.add_scalar('Loss/Critic', critic_loss.item(), step)
        
        states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)
        
        states = torch.tensor(states, dtype=torch.float32)
        
        # Update adversary
        adversary_loss = self.compute_adversary_loss(states)
        self.adversary_optimizer.zero_grad()
        adversary_loss.backward()
        self.adversary_optimizer.step()

        # Log adversary loss to TensorBoard
        self.writer.add_scalar('Loss/Adversary', adversary_loss.item(), step)
        
        # Update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.adversary_target, self.adversary)
    
    def compute_actor_loss(self, states):
        actions = (1 - self.alpha) * self.actor(states) + self.alpha * self.adversary(states)
        q_values = self.critic(states, actions)
        return -q_values.mean()
    
    def compute_critic_loss(self, states, actions, rewards, next_states):
        with torch.no_grad():
            next_actions = (1 - self.alpha) * self.actor_target(next_states) + self.alpha * self.adversary_target(next_states)
            target_q_values = rewards + self.gamma * self.critic_target(next_states, next_actions)
        q_values = self.critic(states, actions)
        return nn.MSELoss()(q_values, target_q_values)
    
    def compute_adversary_loss(self, states):
        actions = (1 - self.alpha) * self.actor(states) + self.alpha * self.adversary(states)
        q_values = self.critic(states, actions)
        return -q_values.mean()
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def store_transition(self, transition):
        self.replay_buffer.add(transition)
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = (1 - self.alpha) * self.actor(state) + self.alpha * self.adversary(state)
        action = action.detach().numpy()[0]
        return np.clip(action, self.action_range[0], self.action_range[1])
    
    def train(self, env, num_episodes, max_steps_per_episode, batch_size):
        step = 0
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            for t in range(max_steps_per_episode):
                action = self.select_action(state)
                action = action + np.random.randn(self.action_dim)  # Add exploration noise
                
                next_state, reward, done, _ = env.step(action)
                
                self.store_transition((state, action, reward, next_state))
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if len(self.replay_buffer.buffer) > batch_size:
                    self.update_networks(batch_size, step)
                    step += 1
                
                if done:
                    break

            # Log episode reward and length to TensorBoard
            self.writer.add_scalar('Reward/Episode', episode_reward, episode)
            self.writer.add_scalar('Episode Length', episode_length, episode)

            # Save the model every 100 episodes
            if episode % 100 == 0:
                self.save_checkpoint(f'ddpg_checkpoint_episode_{episode}.pth')
                print(f'Episode {episode} - Reward: {episode_reward} - Episode Length: {episode_length}')

    def save_checkpoint(self, path):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'adversary_state_dict': self.adversary.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'adversary_target_state_dict': self.adversary_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'adversary_optimizer_state_dict': self.adversary_optimizer.state_dict(),
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha': self.alpha,
            'actor_update_steps': self.actor_update_steps,
            'hidden_dim1': self.hidden_dim1,
            'hidden_dim2': self.hidden_dim2,
            'max_steps_per_episode': self.max_steps_per_episode,
            'batch_size': self.batch_size,
            'num_episodes': self.num_episodes,
            'action_range': self.action_range
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.adversary.load_state_dict(checkpoint['adversary_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.adversary_target.load_state_dict(checkpoint['adversary_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.adversary_optimizer.load_state_dict(checkpoint['adversary_optimizer_state_dict'])
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.alpha = checkpoint['alpha']
        self.actor_update_steps = checkpoint['actor_update_steps']
        self.hidden_dim1 = checkpoint['hidden_dim1']
        self.hidden_dim2 = checkpoint['hidden_dim2']
        self.max_steps_per_episode = checkpoint['max_steps_per_episode']
        self.batch_size = checkpoint['batch_size']
        self.num_episodes = checkpoint['num_episodes']
        self.action_range = checkpoint['action_range']

# Example of how to use the class
if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
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
        alpha=0.01,
        hidden_dim1=64,
        hidden_dim2=64
    )
    
    agent.train(env, num_episodes=agent.num_episodes, max_steps_per_episode=agent.max_steps_per_episode, batch_size=agent.batch_size)
    
    # Save the model
    agent.save_checkpoint('ddpg_checkpoint_final.pth')
    
    # Load the model
    agent.load_checkpoint('ddpg_checkpoint_final.pth')
    
    # Continue training or use the agent for inference
    agent.train(env, num_episodes=agent.num_episodes, max_steps_per_episode=agent.max_steps_per_episode, batch_size=agent.batch_size)
