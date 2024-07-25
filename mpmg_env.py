import gym
import numpy as np
import random
import itertools 
from utils import get_joint_action_code

class MPMGEnv(gym.Env):
    def __init__(self, n_agents=2, action_size=2 ,sigma_beta=0, alpha=1.3):
        super(MPMGEnv, self).__init__()

        # Collusive potential parameters
        self.n_agents = n_agents
        self.sigma_beta = sigma_beta
        self.alpha = alpha

        # Internal state action variables
        self.action_size = action_size
        self.joint_action_size = action_size ** self.n_agents
        self.beta_size = self.n_agents
        self.state_size = self.n_agents + self.joint_action_size + self.beta_size
        self.state_space = {'action_frequencies': None,
                            'joint_action_frequencies': None,
                            'beta_parameters': None} 

    def _set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_power_parameters(self):
        if self.sigma_beta == 0:
            return np.ones(self.n_agents) / self.n_agents
        beta = np.abs(np.random.normal(1/self.n_agents, self.sigma_beta, self.n_agents))
        return beta / np.sum(beta)

    def _update_action_frequencies(self, actions):
        for agent_id, action in enumerate(actions): 
            if action == 1:
                self.action_counts[agent_id] += 1
        self.action_frequencies = self.action_counts/self.iteration

    def _update_joint_action_frequencies(self, actions):
        index = get_joint_action_code(actions)
        mask = [False] * self.joint_action_size
        mask[index] = True
        self.joint_action_counts[mask] += 1
        self.joint_action_frequencies = self.joint_action_counts/self.iteration

    def _update_beta(self):
        return None

    def _get_immediate_rewards(self, actions):
        mask_1 = np.array([action == 1 for action in actions])
        rewards = np.zeros(self.n_agents)
        if not all(mask_1):  # At least someone played 0
            # Mask for agents who played 0
            mask_0 = ~mask_1
            # Calculate total beta for agents who played 0
            beta_sum_0 = np.sum(self.beta_parameters[mask_0])
            # Normalized betas for agents who played 0
            normalized_betas_0 = self.beta_parameters[mask_0] / beta_sum_0
            # Distribute reward based on normalized betas
            rewards[mask_0] = normalized_betas_0
        else:  # Everybody played 1
            rewards = self.beta_parameters * self.alpha
        return rewards

    def _get_state(self):
        self.state_space['joint_action_frequencies'] = self.joint_action_frequencies
        self.state_space['action_frequencies'] = self.action_frequencies
        self.state_space['beta_parameters'] = self.beta_parameters
        return self.state_space
    
    def reset(self, seed=None):
        self._set_seed(seed)
        self.iteration = 1 # not 0 because it is a counter
        self.action_counts = np.zeros(self.n_agents)
        self.joint_action_counts = np.zeros(self.joint_action_size)
        self.beta_parameters = self._get_power_parameters()
        joint_actions = list(itertools.product(range(self.action_size), repeat=self.n_agents))
        for actions in joint_actions:
            _, __, ___ = self.step(actions)
        return self._get_state()

    def step(self, actions):
        
        # Update internal state variables
        self._update_action_frequencies(actions)
        self._update_joint_action_frequencies(actions)
        #self.beta_parameters = self._update_beta()

        # Get immediate rewards
        immediate_rewards = self._get_immediate_rewards(actions)

        # Nest state
        next_state = self._get_state()

        # Update counters
        self.iteration += 1

        return immediate_rewards, next_state, True

