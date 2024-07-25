#utils.py

'''
This file contains utilitary functions non-specific to scripts or modules. 
'''

import itertools
import numpy as np
import torch

def get_joint_action_from_code(joint_action, n_agents, action_dim):
    actions = []
    for _ in range(n_agents):
        action = joint_action % action_dim
        actions.append(action)
        joint_action = joint_action // action_dim
        actions = actions[::-1]
    return actions  # Reverse to get actions in the correct order

def get_joint_action_code(actions):
    action_code = 0
    for action in actions:
        action_code = (action_code << 1) | action
    return action_code

def get_joint_action_code_with_permutation(actions, agent_id):
    '''
    The action of the reference agent (the agent for whom the code is being calculated) 
    is always considered first.
    '''
    # Rotate the actions list so that the action of the agent_id is first
    rotated_actions = actions[agent_id:] + actions[:agent_id]
    # Compute the code
    action_code = 0
    for action in rotated_actions:
        action_code = (action_code << 1) | action
    return action_code

def permute_array(array, agent_index):
    n = len(array)
    if agent_index < 1 or agent_index > n:
        raise ValueError(f"agent_index should be in the range 1 to {n}")
    # Roll the array by agent_index - 1 positions
    return np.roll(array, -(agent_index - 1))

def get_joint_action_space(num_agents, num_actions):
    """
    Generate a list of tuples representing the joint action space.

    Parameters:
    num_agents (int): The number of agents.
    num_actions (int): The number of actions each agent can take.

    Returns:
    list of tuples: The joint action space.
    """
    actions = range(num_actions)
    joint_action_space = list(itertools.product(actions, repeat=num_agents))
    return joint_action_space

def process_state_info(state, n_agents, individual_state_size):
    # Initialize arrays for a single instance
    individual_states = np.zeros((n_agents, individual_state_size))
    joint_state = []
    # Process each agent's state
    for agent_id in range(n_agents):
        individual_key = f'individual_state_{agent_id}'
        individual_state = state[individual_key]
        # Flatten and store individual state
        individual_states[agent_id] = np.hstack(list(individual_state.values()))
        joint_state.append(individual_states[agent_id])
    # Flatten and store joint state
    joint_state = [item for sublist in joint_state for item in sublist]
    collective_features = [state['expected_collective_return'], state['immediate_collective_reward'], state['cumulative_collective_reward']]
    joint_state = np.hstack([joint_state, collective_features])
    return individual_states, joint_state

def reference_agent_permutation(agent_array, agent_id, n_agents):
    # Assuming binary actions
    num_actions = 2
    # Total number of joint actions
    num_joint_actions = num_actions ** n_agents
    # Create a new array to hold the permuted frequencies
    # Assuming agent_array shape is [batch_size, num_joint_actions]
    batch_size = agent_array.shape[0]
    permuted_freq = np.zeros_like(agent_array)
    # Iterate through each joint action
    for joint_action in range(num_joint_actions):
        # Convert joint action to binary representation
        binary_repr = format(joint_action, f'0{n_agents}b')
        # Rotate the binary representation
        rotated_repr = binary_repr[-agent_id:] + binary_repr[:-agent_id]
        # Convert back to an integer
        rotated_action = int(rotated_repr, 2)
        # Assign the frequency to the permuted array for each batch
        for batch_idx in range(batch_size):
            permuted_freq[batch_idx, rotated_action] = agent_array[batch_idx, joint_action]
    return permuted_freq

def map_to_joint_probability(probabilities):
    """
    Maps individual action probabilities to joint action probabilities.
    probabilities: Tensor of shape [batch_size, n_agents]
    """
    batch_size, n_agents = probabilities.size()
    # Initialize joint probabilities tensor
    joint_probs = torch.ones(batch_size, 2**n_agents)
    # Iterate through each agent and calculate joint probabilities
    for i in range(2**n_agents):
        prob = torch.ones(batch_size).to(probabilities.device)
        for agent_id in range(n_agents):
            action = (i >> agent_id) & 1
            prob *= probabilities[:, agent_id] if action == 1 else (1 - probabilities[:, agent_id])
        joint_probs[:, i] = prob
    return joint_probs.to(probabilities.device)

def standardize(array):
    if array.ndim == 1:
        mean = np.mean(array)
        std_dev = np.std(array)
        return (array - mean) / std_dev
    else:
        standardized_array = np.zeros_like(array)
        for i in range(array.shape[0]):
            mean = np.mean(array[i])
            std_dev = np.std(array[i])
            standardized_array[i] = (array[i] - mean) / std_dev
        return standardized_array
    
def has_converged(losses, threshold, window_size):
    return all(abs(loss - threshold) < 0.1 for loss in losses[-window_size:])

import numpy as np

def get_joint_action_code_with_permutation_batch(batch_actions, agent_id):
    '''
    This function handles a batch of action sequences as NumPy arrays. 
    The action of the reference agent (the agent for whom the code is being calculated)
    is always considered first in each sequence.

    Parameters:
    batch_actions (numpy array): A batch of action sequences.
    agent_id (int): The agent ID for which the action code is being calculated.

    Returns:
    numpy array: An array of action codes, one for each action sequence in the batch.
    '''
    action_codes = []
    for actions in batch_actions:
        # Rotate the actions array so that the action of the agent_id is first
        rotated_actions = np.concatenate((actions[agent_id:], actions[:agent_id]))
        
        # Compute the code for this sequence of actions
        action_code = 0
        for action in rotated_actions:
            action_code = (action_code << 1) | action
        action_codes.append(action_code)
    
    return np.array(action_codes)

def flatten_state(state_dict):
    return np.concatenate([v.flatten() for v in state_dict.values()])

def set_nested_value(d, keys, value):
    """
    Set a value in a nested dictionary.
    
    :param d: The dictionary to modify.
    :param keys: A list of keys specifying the path in the dictionary.
    :param value: The value to set at the specified path.
    """
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value

def permute_state(state, num_agent, num_action):
    joint_action_size = num_action**num_agent
    state_0 = permute_array(state[:num_agent], 2)
    # permute_array(state[num_agent+2:num_agent+(joint_action_size-2)], 2)
    state_1 = [state[2]]
    state_2 = permute_array(state[3:5], 2)
    state_3 = [state[5]]
    state_4 = permute_array(state[-2:], 2)
    nested_state = [state_0, state_1, state_2, state_3, state_4]
    state = [item for sublist in nested_state for item in sublist]
    return np.array([float(num) for num in state])