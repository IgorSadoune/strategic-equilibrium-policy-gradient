import torch
import numpy as np
import random
import os
import shutil
import pickle
from tqdm import tqdm
from time import sleep
from mpmg_env import MPMGEnv
import matplotlib.pyplot as plt
import argparse 
from utils import flatten_state, get_joint_action_from_code, permute_array, permute_state
from sepg import SEPGCritic, SEPGActor

parser = argparse.ArgumentParser(description='SEPG')
parser.add_argument('--num_agents', type=int, default=2, help='Number of agents')
parser.add_argument('--sigma_beta', type=float, default=0, help='Sigma beta value')
parser.add_argument('--num_repeat', type=int, default=100, help='Number of repeats')
parser.add_argument('--num_episode', type=int, default=100, help='Number of episodes')
parser.add_argument('--num_offline_iteration', type=int, default=100, help='Number of offline iterations for critic optimization')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--rho', type=float, default=0.99, help='Penalty sclaing factor')

args = parser.parse_args()

current_directory = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_AGENT = args.num_agents
SIGMA_BETA = args.sigma_beta

env = MPMGEnv(n_agents=NUM_AGENT, sigma_beta=SIGMA_BETA) 
STATE_DIM = env.state_size
ACTION_DIM = env.action_size
JOINT_ACTION_DIM = env.joint_action_size
NUM_OFFLINE_ITERATION = args.num_offline_iteration
SEED = 42

metrics = {'critic_losses': np.zeros(NUM_OFFLINE_ITERATION)}

metrics_dir = os.path.join(current_directory, "sepg", "metrics_critic")
if os.path.exists(metrics_dir):
    shutil.rmtree(metrics_dir)

#################### Offline critic training
SEPGAgent_critic = SEPGCritic(NUM_AGENT, STATE_DIM, ACTION_DIM, CRITIC_BATCH_SIZE)

# Seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
sleep(0.01)

# Initial state
_ = env.reset(seed=SEED)

# Training loop
for offline_iter in range(NUM_OFFLINE_ITERATION):

    # Create batch
    actions_, rewards_ = [], []
    for step in range(CRITIC_BATCH_SIZE):
        actions = [random.randint(0, 1) for _ in range(NUM_AGENT)]
        rewards, _, _ = env.step(actions)
        actions_.append(actions)
        rewards_.append(rewards)

# Critic agent update
critic_output = SEPGAgent_critic.learn(actions_, rewards_)

# Collect metrics
critic_loss = SEPGAgent_critic.get_loss()
metrics['critic_losses'][offline_iter] = critic_loss

# Save metrics
os.makedirs(metrics_dir, exist_ok=True)
metric_path = os.path.join(metrics_dir, 'critic_losses.pkl')
with open(metric_path, 'wb') as f:
    pickle.dump(metrics, f)

# Plot critic losses
plt.style.use('ggplot')

plot_dir = os.path.join(current_directory, "sepg", "plots", "critic")
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir, exist_ok=True)

critic_losses = metrics['critic_losses']
plot_path = os.path.join(plot_dir, "critic_losses")
plt.figure()
plt.plot(np.arange(NUM_OFFLINE_ITERATION), critic_losses)
plt.xlabel('Training Episode')
plt.ylabel('Critic Loss')
plt.savefig(plot_path)

print("CRITIC OUTPUT", critic_output)

############################### Online actor training
LR = args.learning_rate
NUM_REPEAT = args.num_repeat
NUM_EPISODE = args.num_episode
NUM_EVAL_EPISODE = 100
BATCH_SIZE = args.batch_size
RHO = args.rho
critic_output = [1.0]*NUM_AGENT

metrics = { 'train': 
                {
                    'action_frequencies': np.zeros((NUM_AGENT, NUM_REPEAT, NUM_EPISODE)),
                    'joint_action_frequencies': np.zeros((env.joint_action_size, NUM_REPEAT, NUM_EPISODE)),
                    'actor_losses': np.zeros((NUM_AGENT, NUM_REPEAT, NUM_EPISODE)),
                    'policy_1': np.zeros((NUM_AGENT, NUM_REPEAT, NUM_EPISODE))
                },
            'eval': {
                    'action_frequencies': np.zeros((NUM_AGENT, NUM_REPEAT, NUM_EVAL_EPISODE)),
                    'joint_action_frequencies': np.zeros((env.joint_action_size, NUM_REPEAT, NUM_EVAL_EPISODE)),
                    'total_return': np.zeros(NUM_REPEAT)
            }
                }

metrics_dir = os.path.join(current_directory, "sepg", f"metrics_actor_{NUM_AGENT}_{SIGMA_BETA}")
if os.path.exists(metrics_dir):
    shutil.rmtree(metrics_dir)

collusive_repeats, non_collusive_repeats = [], []
for repeat in tqdm(range(NUM_REPEAT), desc="Repeat"):
    repeat_seed = SEED + repeat
    random.seed(repeat_seed)
    np.random.seed(repeat_seed)
    torch.manual_seed(repeat_seed)
    sleep(0.01)

    agents = [SEPGActor(NUM_AGENT, STATE_DIM, ACTION_DIM, BATCH_SIZE, LR, RHO) for _ in range(NUM_AGENT)]

    state_dict = env.reset(seed=repeat_seed)
    state = flatten_state(state_dict)

    episode = -1
    for iteration in range(NUM_EPISODE*BATCH_SIZE):
        
        # Act
        actions = []
        for agent, agent_id in zip(agents, range(NUM_AGENT)):
            action = agent.act(state)

            # if agent_id ==0:
            #     action = agent.act(state)
            # else: 
            #     state_ = permute_state(state, NUM_AGENT, num_action=2)
            #     action = agent.act(state_) 
            actions.append(action)
        rewards, next_state_dict, _ = env.step(actions)
        next_state = flatten_state(next_state_dict)

        # Increment memory
        for agent, agent_id in zip(agents, range(NUM_AGENT)):
            agent.remember(state, actions, rewards)

            # if agent_id == 0:
            #     agent.remember(state, actions, rewards)
            # else:
            #     actions = permute_array(actions, agent_id+1) # every agents think they are agent 0
            #     rewards = permute_array(rewards, agent_id+1) # every agents think they are agent 0
            #     agent.remember(state_, actions, rewards)

        # Agent update once every batch_size
        if (iteration+1) % BATCH_SIZE == 0:
            episode += 1

            # Collect metrics and update agent
            for joint_action_frequency in range(env.joint_action_size):
                metrics['train']['joint_action_frequencies'][joint_action_frequency, repeat, episode] = state_dict['joint_action_frequencies'][joint_action_frequency]
            for agent_id in range(NUM_AGENT):
                prob_1 = agents[agent_id].learn(critic_output[agent_id])
                metrics['train']['action_frequencies'][agent_id, repeat, episode] = state_dict['action_frequencies'][agent_id]
                metrics['train']['actor_losses'][agent_id][repeat][episode] = agents[agent_id].get_loss()
                metrics['train']['policy_1'][agent_id][repeat][episode] = prob_1

        # Next state
        state = next_state

    # Evaluation
    total_return = 0
    state_dict = env.reset(seed=repeat_seed)
    state = flatten_state(state_dict)
    for eval_episode in range(NUM_EVAL_EPISODE):
        for joint_action_frequency in range(env.joint_action_size):
            metrics['eval']['joint_action_frequencies'][joint_action_frequency, repeat, eval_episode] = state_dict['joint_action_frequencies'][joint_action_frequency]
        for agent_id in range(NUM_AGENT):
            metrics['eval']['action_frequencies'][agent_id, repeat, eval_episode] = state_dict['action_frequencies'][agent_id]

        actions = [agent.act(state) for agent in agents]
        rewards, next_state_dict, _ = env.step(actions)
        
        total_return += sum(rewards)

        # Next state
        state_dict = next_state_dict
        state = flatten_state(state_dict)

    # Collect total returns
    metrics['eval']['total_return'][repeat] = total_return
    if total_return > 1.2*NUM_EVAL_EPISODE:
        collusive_repeats.append(repeat)
    if total_return <= 1.1*NUM_EVAL_EPISODE:
        non_collusive_repeats.append(repeat)

# Save metrics
os.makedirs(metrics_dir, exist_ok=True)
metric_path = os.path.join(metrics_dir, "metrics.pkl")
with open(metric_path, 'wb') as f:
    pickle.dump(metrics, f)

# Collusion check
count_collusion = np.sum(metrics['eval']['total_return'] > 1.1*NUM_EVAL_EPISODE)
average_total_return = np.mean(metrics['eval']['total_return'])
std_total_return = np.std(metrics['eval']['total_return'])
max_total_return = np.max(metrics['eval']['total_return'])
min_total_return = np.min(metrics['eval']['total_return'])

total_return = {
    'count': count_collusion,
    'mean': average_total_return,
    'std': std_total_return,
    'max': max_total_return,
    'min': min_total_return}

total_return_path = os.path.join(metrics_dir, "total_return.pkl")
with open(total_return_path, 'wb') as f:
    pickle.dump(total_return, f)

print(total_return)

print("Training and evaluation completed.")

# Plot training metrics
average_action_frequencies = np.mean(metrics['train']['action_frequencies'], axis=1)
std_action_frequencies = np.std(metrics['train']['action_frequencies'], axis=1)

average_joint_action_frequencies = np.mean(metrics['train']['joint_action_frequencies'], axis=1)
std_joint_action_frequencies = np.std(metrics['train']['joint_action_frequencies'], axis=1)

average_actor_losses = np.mean(metrics['train']['actor_losses'], axis=1)
std_actor_losses = np.std(metrics['train']['actor_losses'], axis=1)

average_policy_1 = np.mean(metrics['train']['policy_1'], axis=1)
std_policy_1 = np.std(metrics['train']['policy_1'], axis=1)

plt.style.use('ggplot')

plot_dir = os.path.join(current_directory, "sepg", "plots", f"actor_{NUM_AGENT}_{SIGMA_BETA}")
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir, exist_ok=True)

# Plot training average action frequencies with confidence intervals
plot_path = os.path.join(plot_dir, "train_action_frequencies")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), average_action_frequencies[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), average_action_frequencies[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    average_action_frequencies[agent_id] - std_action_frequencies[agent_id],
                    average_action_frequencies[agent_id] + std_action_frequencies[agent_id],
                    alpha=0.3)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Action Frequency', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot training average joint action frequencies with confidence intervals
plot_path = os.path.join(plot_dir, "train_joint_action_frequencies")
plt.figure(figsize=(10, 6))
for joint_action_code in range(JOINT_ACTION_DIM):
    joint_action = get_joint_action_from_code(joint_action_code, NUM_AGENT, ACTION_DIM)
    if NUM_AGENT > 2:
        plt.plot(np.arange(NUM_EPISODE), average_joint_action_frequencies[joint_action_code])
    else:
        plt.plot(np.arange(NUM_EPISODE), average_joint_action_frequencies[joint_action_code], label=f'Joint Action {joint_action}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    average_joint_action_frequencies[joint_action_code] - std_joint_action_frequencies[joint_action_code],
                    average_joint_action_frequencies[joint_action_code] + std_joint_action_frequencies[joint_action_code],
                    alpha=0.3)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Joint Action Frequency', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 2:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot average actor losses with confidence intervals
plot_path = os.path.join(plot_dir, "actor_losses")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), average_actor_losses[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), average_actor_losses[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    average_actor_losses[agent_id] - std_actor_losses[agent_id],
                    average_actor_losses[agent_id] + std_actor_losses[agent_id],
                    alpha=0.3)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Actor Loss', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot average policy_1 with confidence intervals
plot_path = os.path.join(plot_dir, "policy_1")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), average_policy_1[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), average_policy_1[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    average_policy_1[agent_id] - std_policy_1[agent_id],
                    average_policy_1[agent_id] + std_policy_1[agent_id],
                    alpha=0.3)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Policy for action 1: $P(a_i = 1)$', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot evaluation metrics
average_action_frequencies = np.mean(metrics['eval']['action_frequencies'], axis=1)
std_action_frequencies = np.std(metrics['eval']['action_frequencies'], axis=1)

average_joint_action_frequencies = np.mean(metrics['eval']['joint_action_frequencies'], axis=1)

# Plot eval average action frequencies with confidence intervals
plot_path = os.path.join(plot_dir, "eval_action_frequencies")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    plt.plot(np.arange(NUM_EVAL_EPISODE), average_action_frequencies[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EVAL_EPISODE),
                    average_action_frequencies[agent_id] - std_action_frequencies[agent_id],
                    average_action_frequencies[agent_id] + std_action_frequencies[agent_id],
                    alpha=0.3)
plt.xlabel('Evaluation Episode', fontsize=12, fontweight='bold')
plt.ylabel('Action Frequency', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot eval average joint action frequencies with confidence intervals
plot_path = os.path.join(plot_dir, "eval_joint_action_frequencies")
plt.figure(figsize=(10, 6))
for joint_action_code in range(JOINT_ACTION_DIM):
    joint_action = get_joint_action_from_code(joint_action_code, NUM_AGENT, ACTION_DIM)
    plt.plot(np.arange(NUM_EVAL_EPISODE), average_joint_action_frequencies[joint_action_code], label=f'Joint Action {joint_action}')
plt.xlabel('Evaluation Episode', fontsize=12, fontweight='bold')
plt.ylabel('Joint Action Frequency', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot collusive repeats
collusive_action_frequencies = metrics['train']['action_frequencies'][:, collusive_repeats, :]
collusive_joint_action_frequencies = metrics['train']['joint_action_frequencies'][:, collusive_repeats, :]
collusive_actor_losses = metrics['train']['actor_losses'][:, collusive_repeats, :]
collusive_policy_1 = metrics['train']['policy_1'][:, collusive_repeats, :]

collusive_average_action_frequencies = np.mean(collusive_action_frequencies, axis=1)
collusive_std_action_frequencies = np.std(collusive_action_frequencies, axis=1)

collusive_average_joint_action_frequencies = np.mean(collusive_joint_action_frequencies, axis=1)
collusive_std_joint_action_frequencies = np.std(collusive_joint_action_frequencies, axis=1)

collusive_average_actor_losses = np.mean(collusive_actor_losses, axis=1)
collusive_std_actor_losses = np.std(collusive_actor_losses, axis=1)

collusive_average_policy_1 = np.mean(collusive_policy_1, axis=1)
collusive_std_policy_1 = np.std(collusive_policy_1, axis=1)

plt.style.use('ggplot')

plot_dir = os.path.join(current_directory, "sepg", "plots", f"collusive_metrics_actor_{NUM_AGENT}_{SIGMA_BETA}")
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir, exist_ok=True)

# Plot training average action frequencies with confidence intervals
plot_path = os.path.join(plot_dir, "train_action_frequencies")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), collusive_average_action_frequencies[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), collusive_average_action_frequencies[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    collusive_average_action_frequencies[agent_id] - collusive_std_action_frequencies[agent_id],
                    collusive_average_action_frequencies[agent_id] + collusive_std_action_frequencies[agent_id],
                    alpha=0.2)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Action Frequency', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot training average joint action frequencies with confidence intervals
plot_path = os.path.join(plot_dir, "train_joint_action_frequencies")
plt.figure(figsize=(10, 6))
for joint_action_code in range(JOINT_ACTION_DIM):
    joint_action = get_joint_action_from_code(joint_action_code, NUM_AGENT, ACTION_DIM)
    if NUM_AGENT > 2:
        plt.plot(np.arange(NUM_EPISODE), collusive_average_joint_action_frequencies[joint_action_code])
    else:
        plt.plot(np.arange(NUM_EPISODE), collusive_average_joint_action_frequencies[joint_action_code], label=f'Joint Action {joint_action}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    collusive_average_joint_action_frequencies[joint_action_code] - collusive_std_joint_action_frequencies[joint_action_code],
                    collusive_average_joint_action_frequencies[joint_action_code] + collusive_std_joint_action_frequencies[joint_action_code],
                    alpha=0.2)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Joint Action Frequency', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 2:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot average actor losses with confidence intervals
plot_path = os.path.join(plot_dir, "actor_losses")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), collusive_average_actor_losses[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), collusive_average_actor_losses[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    collusive_average_actor_losses[agent_id] - collusive_std_actor_losses[agent_id],
                    collusive_average_actor_losses[agent_id] + collusive_std_actor_losses[agent_id],
                    alpha=0.2)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Actor Loss', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot average policy_1 with confidence intervals
plot_path = os.path.join(plot_dir, "policy_1")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), collusive_average_policy_1[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), collusive_average_policy_1[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    collusive_average_policy_1[agent_id] - collusive_std_policy_1[agent_id],
                    collusive_average_policy_1[agent_id] + collusive_std_policy_1[agent_id],
                    alpha=0.2)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Policy for action 1: $P(a_i = 1)$', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot non-collusive repeats
non_collusive_action_frequencies = metrics['train']['action_frequencies'][:, non_collusive_repeats, :]
non_collusive_joint_action_frequencies = metrics['train']['joint_action_frequencies'][:, non_collusive_repeats, :]
non_collusive_actor_losses = metrics['train']['actor_losses'][:, non_collusive_repeats, :]
non_collusive_policy_1 = metrics['train']['policy_1'][:, non_collusive_repeats, :]

non_collusive_average_action_frequencies = np.mean(non_collusive_action_frequencies, axis=1)
non_collusive_std_action_frequencies = np.std(non_collusive_action_frequencies, axis=1)

non_collusive_average_joint_action_frequencies = np.mean(non_collusive_joint_action_frequencies, axis=1)
non_collusive_std_joint_action_frequencies = np.std(non_collusive_joint_action_frequencies, axis=1)

non_collusive_average_actor_losses = np.mean(non_collusive_actor_losses, axis=1)
non_collusive_std_actor_losses = np.std(non_collusive_actor_losses, axis=1)

non_collusive_average_policy_1 = np.mean(non_collusive_policy_1, axis=1)
non_collusive_std_policy_1 = np.std(non_collusive_policy_1, axis=1)

plt.style.use('ggplot')

plot_dir = os.path.join(current_directory, "sepg", "plots", f"non_collusive_metrics_actor_{NUM_AGENT}_{SIGMA_BETA}")
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir, exist_ok=True)

# Plot training average action frequencies with confidence intervals
plot_path = os.path.join(plot_dir, "train_action_frequencies")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), non_collusive_average_action_frequencies[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), non_collusive_average_action_frequencies[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    non_collusive_average_action_frequencies[agent_id] - non_collusive_std_action_frequencies[agent_id],
                    non_collusive_average_action_frequencies[agent_id] + non_collusive_std_action_frequencies[agent_id],
                    alpha=0.2)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Action Frequency', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot training average joint action frequencies with confidence intervals
plot_path = os.path.join(plot_dir, "train_joint_action_frequencies")
plt.figure(figsize=(10, 6))
for joint_action_code in range(JOINT_ACTION_DIM):
    joint_action = get_joint_action_from_code(joint_action_code, NUM_AGENT, ACTION_DIM)
    if NUM_AGENT > 2:
        plt.plot(np.arange(NUM_EPISODE), non_collusive_average_joint_action_frequencies[joint_action_code])
    else:
        plt.plot(np.arange(NUM_EPISODE), non_collusive_average_joint_action_frequencies[joint_action_code], label=f'Joint Action {joint_action}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    non_collusive_average_joint_action_frequencies[joint_action_code] - non_collusive_std_joint_action_frequencies[joint_action_code],
                    non_collusive_average_joint_action_frequencies[joint_action_code] + non_collusive_std_joint_action_frequencies[joint_action_code],
                    alpha=0.2)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Joint Action Frequency', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 2:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot average actor losses with confidence intervals
plot_path = os.path.join(plot_dir, "actor_losses")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), non_collusive_average_actor_losses[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), non_collusive_average_actor_losses[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    non_collusive_average_actor_losses[agent_id] - non_collusive_std_actor_losses[agent_id],
                    non_collusive_average_actor_losses[agent_id] + non_collusive_std_actor_losses[agent_id],
                    alpha=0.2)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Actor Loss', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Plot average policy_1 with confidence intervals
plot_path = os.path.join(plot_dir, "policy_1")
plt.figure(figsize=(10, 6))
for agent_id in range(NUM_AGENT):
    if NUM_AGENT > 5:
        plt.plot(np.arange(NUM_EPISODE), non_collusive_average_policy_1[agent_id])
    else:
        plt.plot(np.arange(NUM_EPISODE), non_collusive_average_policy_1[agent_id], label=f'Agent {agent_id}')
    plt.fill_between(np.arange(NUM_EPISODE),
                    non_collusive_average_policy_1[agent_id] - non_collusive_std_policy_1[agent_id],
                    non_collusive_average_policy_1[agent_id] + non_collusive_std_policy_1[agent_id],
                    alpha=0.2)
plt.xlabel('Training Episode', fontsize=12, fontweight='bold')
plt.ylabel('Policy for action 1: $P(a_i = 1)$', fontsize=12, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
if NUM_AGENT <= 5:
    plt.legend(loc='best')
plt.gca().set_facecolor('#f0f0f0')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')