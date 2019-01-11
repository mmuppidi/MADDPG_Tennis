
import torch
from ddpg import Agent
from buffer import ReplayBuffer


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor


class MADDPG():
    def __init__(self, num_agents, state_size, action_size, random_seed):
        """ Initialize multiple Agents each with a Actor-Critic network
            but they share the replay buffer to learn from experience
        """
        self.num_agents = num_agents
        self.agents = []
        for _ in range(num_agents):
            agent = Agent(state_size, action_size, random_seed)
            self.agents.append(agent)
            
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def act(self, states, add_noise=True):
        clipped_actions = []
        for state, agent in zip(states, self.agents):
            clipped_actions.append(agent.act(state, add_noise))
        return clipped_actions
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    def learn(self, experiences, gamma):
        for agent in self.agents:
            agent.learn(experiences, gamma)
            
    def saveCheckPoints(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  f"checkpoints/actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoints/critic_agent_{i}.pth")
            
    def loadCheckPoints(self):
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load(f"checkpoints/actor_agent_{i}.pth"))
            agent.critic_local.load_state_dict(torch.load(f"checkpoints/critic_agent_{i}.pth"))
            
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience / reward
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for agent in self.agents:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)