# TP RL : Yamine BELKHEDRA

## SUJET : Proximal Policy Optimization PPO

### Contextes

Dans ce rendu de TP, on cherche à résoudre le problème suivant pour un cas simple : nous avons un agent qui peut agir dans un environnement et à partir des observations qu'il peut faire, il va chercher à prendre les décisions qui vont maximiser ses gains dans cet environnement. Ici, nous choisissons d'entraîner un agent sur l'environnement CartPole V0 de Gym.

#### Environnemment : CartPole-v0

Ici, nous choisissons d'entraîner un agent sur l'environnement CartPole-v0 de Gym, qui est discret.
Cet environnement simule un pendule inversé fixé sur un chariot en mouvement. L'objectif est de maintenir le pendule en équilibre vertical en évitant qu'il ne tombe complètement vers le haut (90 degrés) ou vers le bas (-90 degrés) en ayant comme actions possibles d'appliquer une force vers la gauche ou la droite sur le chariot. Les récompenses sont +1 à chaque pas de temps tant que le pendule reste dans des limites d'inclinaison acceptables. L'épisode se termine lorsque le pendule s'incline au-delà d'un certain angle ou lorsque le chariot sort de limites prédéfinies. Pour cet environnement le score maximal est 500, score qui représente le maintien parfait de l'équilibre pendant 500 pas de temps consécutifs.

#### Agent

Pour représenter notre agent, nous utiliserons un réseau de neurones qui prédira l'action à faire selon un état donné, et qu'on entraînera à faire les prédictions qui maximiseront le reward.

### Explications et intuitions de la PPO

L'idée principale de la méthode PPO est de contrôler notre pas d'apprentissage pour éviter de changer grandement notre politique entre chaque itération. En effet, si un mauvais pas est pris, les nouvelles données d'entrainement samplées seront loin d'une solution envisageable. De plus en faisant cela on obtient une meilleure stabilité du modèle en évitant les cas où on pourrait s'éloigner de l'optimum à cause d'un pas trop grand. La régulation du pas d'apprentissage est permise grâce à une contrainte KL (Kullback-Leibler) entre l'ancienne et la nouvelle politique, mais aussi grâce à une nouvelle expression de la fonction objective de la politique appelée surrogate loss.

Avec la TRPO, on cherche à optimiser l'erreur surrogate tout en ayant une contrainte KL entre l'ancienne et la nouvelle policy. Avec la PPO, on fusionne l'erreur surrogate et la contrainte KL en un seul problème d'optimisation. Ainsi, le problème devient plus simple et réalisable avec les bibliothèques de machine learning comme Pytorch. Le problème d'optimisation initiale peut s'écrire par la formule ci-dessous. La variable $\hat{A}_t$ correspond à l'avantage d'une action, calculée à partir de ce que nous font gagner les autres actions. Ainsi, s'il est positif, on souhaite augmenter la probabilité de choisir cette action à cet état car elle est meilleure que la moyenne des autres actions sur cet état. Si l'avantage est négatif, on souhaite alors diminuer la probabilité de prendre cette action.


$$ L^{CPI}(\theta) = \hat{\mathbb{E}}_{t} \left[ \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t [r_t(\theta) \hat{A}_t] $$

Avec PPO, on souhaite éviter que le ratio entre la nouvelle policy et l'ancienne ait des valeurs trop extrêmes (trop proches de 0 ou trop au-dessus de 1), on utilise alors un clipping pour borner le ratio entre $1-\epsilon$ et $1+\epsilon$. Les différents articles de recherche conseillent $\epsilon = 0.2$. La formule devient alors la suivante :

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right)\right]
 $$ 

Finalement, nous utiliserons la formule suivante :

$$ L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb{E}}_t \left[ L^{CLIP}_{t} (\theta) - c_{1} L^{VF}(\theta) + c_{2} S[\pi_{\theta}] \right] $$


où $c_1$ et $c_2$ représentent des coefficients, $L^{VF}(\theta)$ l'erreur moyenne au carré des valeurs de l'état et $S[\pi_\theta]$ une valeur d'entropie bonus qui garantit assez d'exploitation.

### Utilisation du projet

L'entraînement d'un agent avec la méthode PPO est possible avec l'utilisation de la commande :
```bash
python3 PPO.py [--num_episodes, -n] 1500 [--dest_file, -f] ppo_agent.pth
```

ou pour plus d'aides :

```bash
python3 PPO.py -h
```

Les poids du réseau de neurones seront sauvegardés dans un dossier results. Celui-ci sera créé s'il n'est pas déjà existant. Vous pouvez tester le réseau de neurones entraîné avec la commande suivante :

```bash
python3 test.py [--src_file, -f] results/ppo_agent.pth
```

### Résultats et observations

En lançant plusieurs expériences, on observe que 1500 épisodes suffisent pour que le modèle donne des résultats satisfaisants. Le modèle a l'air plutôt stable car les entraînements sont souvent bons. On observe en effet qu'au fil des épisodes, les scores totaux sont plus souvent près du score maximal de 500.

### Sources

"L4 TRPO and PPO (Foundations of Deep RL Series)" YouTube, uploaded by Pieter Abbeel, published August 25, 2021, https://www.youtube.com/watch?v=KjWF8VIMGiY

"Proximal Policy Optimization (PPO)", blog,  uploaded by Thomas Simonini, published August 5, 2022, https://huggingface.co/blog/deep-rl-ppo

"Cart Pole", Gym Documentation, https://www.gymlibrary.dev/environments/classic_control/cart_pole/