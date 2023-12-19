import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
import argparse
from PPO import PolicyNetwork, PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", "-f", type=str, default="results/ppo_agent.pth", help="path of the file to load the model from.")
    args = parser.parse_args()
    src_file = args.src_file

    env = gym.make('CartPole-v0', render_mode="human")
    observation_dimension = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPO(observation_dimension, action_dim)
    ppo_agent.policy_network.load_state_dict(torch.load(src_file))

    state, _ = env.reset()
    done = False
    total_reward = 0
    want_to_play = True
    while want_to_play:
        while not done:
            action, _ = ppo_agent.choose_action(state)
            state, reward, done, _, __ = env.step(action)
            total_reward += reward
            # env.render()
        print("Total reward of the game: {}".format(total_reward))
        print("Do you want to play again ? (y/n)")
        answer = input()
        if answer == "y":
            state, _ = env.reset()
            done = False
            total_reward = 0
        else:
            want_to_play = False

    env.close()