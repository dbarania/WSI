import numpy as np
import random
import gym
import os
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

def run(episodes:int, t_max:int, gamma:float, beta:float,map_size:int = 4,epsilon:float = 0.1,random_seed:int = 42, is_slippery = False, render:bool = False):
    random.seed(random_seed)
    mode = 'human' if render else None
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=map_size),render_mode=mode, is_slippery=is_slippery)
    Q_matrix = np.zeros((env.observation_space.n,env.action_space.n))

    result_array = np.zeros((episodes))
    treasures = 0
    for e in range(episodes):
        print(f"Started episode {e}")
        state = env.reset()[0]
        for t in range(t_max):
            # Which action to use, greedy-epsilon
            strategy = random.random()
            action = None
            if strategy>=epsilon:
                # Pick the best action from Q
                a = Q_matrix[state]
                max_value = np.max(a)
                max_indices = np.where(a == max_value)[0]
                action = np.random.choice(max_indices)
            else:
                # Pick random avliable action
                action = env.action_space.sample()

            new_state, reward, terminated, _, _ = env.step(action)
            # print(f"I am at {state}, I decided to {action}, the reward I found {reward}, sum {np.sum(Q_matrix)}")
            
            delta = reward + gamma*np.max(Q_matrix[new_state])-Q_matrix[state,action]
            Q_matrix[state,action] += beta*delta
            state = new_state
            
            if terminated:
                # msg = "Found treasue" if reward else "Ended up in the lake"
                # print(f"Episode {e} finished \t {msg} \t {np.sum(Q_matrix)}")
                result_array[e] = np.sum(Q_matrix)
                treasures+= reward if reward==1 else 0
                break
    print(treasures)    
    return result_array

if __name__=="__main__":
    
    episodes = 1000
    t_max = 200
    gamma_default = 0.3
    beta_default = 0.5
    map_size = 6
    epsilon_default = 0.15
    random_seeds = [1,5,55437]
    is_slippery = False
    render = False

    betas = range(1,9,1)
    gammas = range(1,9,1)
    epsilons = range(5,20,1)


output_dir = 'imgs'
os.makedirs(output_dir, exist_ok=True)

# Plot and save figures for different betas
for i, beta in enumerate(betas):
    plt.figure()
    for seed in random_seeds:
        result = run(episodes, t_max, gamma_default, beta / 10, map_size, epsilon_default, seed, is_slippery, render)
        plt.plot(range(episodes), result, label=f'beta={beta/10}, seed={seed}')
    plt.legend()
    plt.title(f'Beta={beta/10}')
    plt.xlabel('Episode')
    plt.ylabel('Result')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'beta{i+1}.png'))
    plt.close()

# Plot and save figures for different gammas
for i, gamma in enumerate(gammas):
    plt.figure()
    for seed in random_seeds:
        result = run(episodes, t_max, gamma / 10, beta_default, map_size, epsilon_default, seed, is_slippery, render)
        plt.plot(range(episodes), result, label=f'gamma={gamma/10}, seed={seed}')
    plt.legend()
    plt.title(f'Gamma={gamma/10}')
    plt.xlabel('Episode')
    plt.ylabel('Result')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'gamma{i+1}.png'))
    plt.close()

# Plot and save figures for different epsilons
for i, epsilon in enumerate(epsilons):
    plt.figure()
    for seed in random_seeds:
        result = run(episodes, t_max, gamma_default, beta_default, map_size, epsilon / 100, seed, is_slippery, render)
        plt.plot(range(episodes), result, label=f'epsilon={epsilon/100}, seed={seed}')
    plt.legend()
    plt.title(f'Epsilon={epsilon/100}')
    plt.xlabel('Episode')
    plt.ylabel('Result')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'epsilon{i+1}.png'))
    plt.close()
