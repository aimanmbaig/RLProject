import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)
        steps_taken = 0
        while not terminated:
            # TODO: Get action from DQN.
            old_obs = obs
            action_tensor = dqn.act(obs, exploit=False)
            action = action_tensor.item()

            # Act in the true environment.
            obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess incoming observation.
            if not terminated:
                obs = preprocess(obs, env=args.env).unsqueeze(0)
            else:
                obs = None
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            reward_tensor = torch.tensor(reward, device=device)
            # obs = torch.tensor(obs, device=device) if obs is not None else None
            # print("NEXT_OBS: ", obs)
            memory.push(old_obs, action_tensor, obs, reward_tensor)

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps_taken % env_config['train_frequency'] == 0:
                optimize(dqn, target_dqn, memory, optimizer)
            
            

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if steps_taken % env_config['target_update_frequency'] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            steps_taken += 1

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')
            output_file = open(f'output/output.txt', 'a')
            output_file.write(f'{episode+1}, {mean_return}\n')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
