# TP RL Yamine BELKHEDRA

## SUJET : Proximal Policy Optimization PPO

### Contextes

Dans ce rendu de TP, on cherche à résoudre le problème suivant pour un cas simple : nous avons un agent qui peut agir dans un environnement et à partir des observations qu'il peut faire, il va chercher à prendre les décisions qui vont maximiser ses gains dans cet environnement. Ici, nous choisissons d'entrainer un agent sur l'environnement CartPole V0 de Gym.

#### Environnemment : CartPole-v0

Ici, nous choisissons d'entrainer un agent sur l'environnement CartPole-v0 de Gym.

### Explications et intuitions de la PPO

L'idée principale de la méthode PPO est de contrôler nos pas d'apprentissage pour éviter de changer grandement notre politique entre chaque itération. En effet, si un mauvaise pas est pris, les nouvelles données d'entrainement seront complètement fausses. La régulation du pas d'apprentissga est permis grâce à une contrainte KL (Kullback-Leibler) entre l'ancienne et la nouvelle politique, mais aussi grâce à une nouvelle expression de la fonction objectif de la politique appelé surrogate loss.

Avec la TRPO, on cherche à optimisr la surrogate loss tout en ayant une contrainte KL entre l'ancienne et la nouvelle policy. Avec la PPO, on fusionne la surrogate loss et la contrainte KL en un seul problème d'optimisation.
#On cherche aussi à éviter que le ratio entre la nouvelle policy et l'ancienne n'est pas trop grande, on utilise donc un clipping pour borner le ratio entre 1-epsilon et 1+epsilon.


### Sources

"L4 TRPO and PPO (Foundations of Deep RL Series)" YouTube, uploaded by Pieter Abbeel, published August 25, 2021, https://www.youtube.com/watch?v=KjWF8VIMGiY

"Proximal Policy Optimization (PPO)", blog,  uploaded by Thomas Simonini, published August 5, 2022, https://huggingface.co/blog/deep-rl-ppo