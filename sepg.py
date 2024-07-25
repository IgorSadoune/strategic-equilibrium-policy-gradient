import torch
import numpy as np
from utils import get_joint_action_code, get_joint_action_from_code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CriticNetwork(torch.nn.Module):
    def __init__(self, state_dim, num_agent):
        super(CriticNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_agent)
        )

    def _set_seeds(self):
        if self.seed is not None:
            # Set the seed for PyTorch
            torch.manual_seed(self.seed)
            # CUDA (GPU)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups
            # Additional settings for ensuring reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def forward(self, x):
        return self.layers(x)
    
class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim):
        super(ActorNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def _set_seeds(self):
        if self.seed is not None:
            # Set the seed for PyTorch
            torch.manual_seed(self.seed)
            # CUDA (GPU)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups
            # Additional settings for ensuring reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def forward(self, x):
        return self.layers(x)

class SEPGCritic:
    def __init__(self, num_agents, state_dim, action_dim, batch_size, lr=1e-3):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.joint_action_dim = action_dim ** num_agents
        self.batch_size = batch_size
        self.critic = CriticNetwork(state_dim, self.num_agents).to(device)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_loss = 0.0
        self.joint_action_counts = np.zeros(self.joint_action_dim)
        self.avg_collective_returns = np.zeros(self.joint_action_dim)

    def _update_running_averages(self, actions, rewards): 
        # Computes averages in batch
        for joint_action_, joint_reward in zip(actions, rewards):
            joint_action = get_joint_action_code(joint_action_)
            self.joint_action_counts[joint_action] += 1
            self.avg_collective_returns[joint_action] += (joint_reward.sum() - self.avg_collective_returns[joint_action]) / self.joint_action_counts[joint_action]

    def _compute_critic_joint_prob(self, critic_output):
        critic_ind_probs = torch.stack((1 - critic_output, critic_output), dim=1)
        self.critic_joint_prob = critic_ind_probs[0]
        for probs in critic_ind_probs[1:]:
            print(self.critic_joint_prob.rehape(-1))
            self.critic_joint_prob = torch.outer(self.critic_joint_prob, probs).reshape(-1)
        self.critic_joint_prob /= self.critic_joint_prob.sum()
        self.critic_joint_prob = self.critic_joint_prob.flatten()

    def _compute_critic_loss(self, critic_output):
        self._compute_critic_joint_prob(critic_output)
        self.critic_loss = torch.tensor([0.0]).to(device)
        for joint_action in range(self.joint_action_dim):
            joint_prob = torch.FloatTensor(self.critic_joint_prob[joint_action]).to(device)
            avg_collective_return = torch.FloatTensor(self.avg_collective_returns[joint_action]).to(device)
            self.critic_loss -= joint_prob * avg_collective_return

    def learn(self, actions, rewards):

        # Latent space
        latent_input = torch.randn(self.batch_size, self.state_dim, device=device) # stateless critic

        # Update internal variables
        self._update_running_averages(actions, rewards)

        # Update
        critic_output = torch.sigmoid(self.critic(latent_input)) # p(a_i = 1) for all i
        self._compute_critic_loss(critic_output)
        self.optimizer_critic.zero_grad()
        self.critic_loss.backward()
        self.optimizer_critic.step()

        return critic_output.cpu().detach().numpy()
    
    def get_loss(self):
        return self.critic_loss.cpu().detach().numpy().item()

    
class SEPGActor:

    def __init__(self, num_agents, state_dim, action_dim, batch_size, lr, rho):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.joint_action_dim = action_dim ** num_agents
        self.lr = lr
        self.actor = ActorNetwork(state_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_loss = 0.0
        self.batch_size = batch_size
        self.action_counts = np.zeros((self.num_agents, self.action_dim))
        self.memory = []
        self.rho = rho
    
    def _compute_actor_loss(self, actions, rewards, agent_freqs_1, opponents_freqs_1, critic_output, policy_probs_1):
        
        # Create frequency tensors of size btach_size
        critic_probs = torch.where(actions==1, critic_output, 1-critic_output)
        opponents_freqs = torch.where(actions==1, opponents_freqs_1, 1-opponents_freqs_1)
        agent_freqs = torch.where(actions==1, agent_freqs_1, 1-agent_freqs_1)
        policy_probs = torch.where(actions==1, policy_probs_1, 1-policy_probs_1)

        # Target
        target = rewards * critic_probs * opponents_freqs # r_i * p^C(a_i) * p(a_-i)

        # Prediction
        prediction = rewards * agent_freqs * opponents_freqs # r_i * p(a_i) * p(a_-i)

        # MSE 
        mse = torch.nn.functional.mse_loss(target, prediction) 

        # Entropy
        entropy = -(policy_probs * torch.log(policy_probs + 1e-10)).sum(dim=1).mean()

        # Actor loss
        self.actor_loss = mse + self.rho * entropy

    def _reset_memory(self):
        self.memory = []

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(self.actor(state_tensor)) 
        action = torch.bernoulli(prob).item()
        return int(action)
    
    def remember(self, state, actions, rewards):

        # Compute and store action frequencies for each agent
        for agent_id, action in zip(range(self.num_agents), actions):
            self.action_counts[agent_id, action] += 1 # counts form the start until the current iteration
        agent_freq_1 = self.action_counts[0, 1]/(self.action_counts[0, :].sum() + 1e-5)
        opponents_freq_1 = self.action_counts[1:, 1].sum()/(self.action_counts[1:, :].sum() + 1e-5) # p(a_-i = 0) is the prob that at least one opponent plays 0

        # Increment memory
        action = actions[0]
        reward = rewards[0] 
        self.memory.append((state, action, reward, agent_freq_1, opponents_freq_1))

    def learn(self, critic_output):
        if len(self.memory) == 0:
            return

        # Collect
        states, actions, rewards, agent_freqs_1, opponents_freqs_1  = zip(*self.memory)

        # To tensor
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        agent_freqs_1 = torch.FloatTensor(np.array(agent_freqs_1)).to(device)
        opponents_freqs_1 = torch.FloatTensor(np.array(opponents_freqs_1)).to(device)
        critic_output = torch.FloatTensor(np.array(critic_output)).to(device)
        
        # Update actor
        policy_probs_1 = torch.sigmoid(self.actor(states)) # p(a_i=1)
        self._compute_actor_loss(actions, rewards, agent_freqs_1, opponents_freqs_1, critic_output, policy_probs_1)
        self.optimizer_actor.zero_grad()
        self.actor_loss.backward()
        self.optimizer_actor.step()

        # New batch
        self._reset_memory()

        return policy_probs_1.mean(axis=0).cpu().detach().numpy()

    def get_loss(self):
        return self.actor_loss.cpu().detach().numpy()