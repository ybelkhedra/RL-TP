import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
from gym.spaces import Discrete, Box
import random
import gym

##L'une des idees pour les methodes modernes dont l PPO st d'utilisr un loss surrogate. C'st à dir controllr la taill du stp siz pour vitr de fair ds pas trop gros.
#stp too far = mauvais policy car ça fauss ls informations qu'on aura plus tard car lls sront rcolt sur l nouveau modèle.  
#La TRPO, on chrch à optimisr la loss surrogat, tout n consrvant un faible distance (distance KL) ntr l'ancienn et la nouvelle policy.
#A la différence de la TRPO, la PPO permets d'éviter d'utiliser une erreur de second ordre, pour mieux s'adapter aux frameworks de deep learning

#Avec la TRPO, on cherche à optimisr la surrogate loss tout en ayant une contrainte KL entre l'ancienne et la nouvelle policy. Avec la PPO, on fusionne la surrogate loss et la contrainte KL en un seul problème d'optimisation.

# Environnement CartPole
env = gym.make('CartPole-v0')
observation_dimension = env.observation_space.shape[0]
action_dim = env.action_space.n

# Architecture du réseau neuronal
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_dimension, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        state_values = self.fc3(x)
        return action_probs, state_values

# Algorithme PPO
class PPO:
    def __init__(self):
        self.policy_network = PolicyNetwork()
        self.optimizer = Adam(self.policy_network.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 0.2  #valeur pour borner le clipping

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs, _ = self.policy_network(state) 
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample() #l'action est determiné à l'aide de la distribution de probabilité obtenue
        return action.item(), action_probs[0, action]

    def update_policy(self, states, actions, old_action_probs, advantages, returns):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_action_probs = torch.FloatTensor(old_action_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        for _ in range(5):  # Nombre d'itérations sur les échantillons
            action_probs, state_values = self.policy_network(states)
            action_distribution = torch.distributions.Categorical(action_probs)
            entropy = action_distribution.entropy()

            new_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = new_action_probs / old_action_probs
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean() #on minimise la policy_loss pour éviter qu'on sorte de notre zone de confiance

            value_loss = nn.MSELoss()(state_values.squeeze(), returns) #calcul de la loss des valeurs

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Initialisation de l'agent PPO
ppo_agent = PPO()

# Entraînement sur plusieurs épisodes
num_episodes = 1000
max_steps = 500
for episode in range(num_episodes):
    state, _ = env.reset()
    states, actions, rewards, old_action_probs = [], [], [], []
    total_reward = 0

    #on sample les actions, rewards et les probas d'activer l'action sur notre agent initial
    for step in range(max_steps):
        action, old_action_prob = ppo_agent.get_action(state) #on récupère la prochaine action et la proba de l'action actuelle
        next_state, reward, done, _, __ = env.step(action) # on récupère l'état suivant et la recompense en fonction de la prochaine action

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        old_action_probs.append(old_action_prob.item())

        total_reward += reward
        state = next_state

        if done:
            break

    # Calcul des avantages et des retours
    returns = []
    advantage = 0
    for r in rewards[::-1]:
        advantage = r + ppo_agent.gamma * advantage
        returns.insert(0, advantage)
    advantages = torch.tensor(returns) - torch.tensor(old_action_probs)

    # Normalisation des avantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Mettre à jour la politique
    ppo_agent.update_policy(states, actions, old_action_probs, advantages, returns)

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
        print("Average score of the policy: {}".format(total_reward / (episode + 1)))

env.close()
