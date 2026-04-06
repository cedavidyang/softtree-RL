# Interpretable RL based on Soft Tree Actors

This repository is associated with the following paper:

> **Citation**
>
> Moayyedi, S.A., Yang, D.Y., 2026. Interpretable Deep Reinforcement Learning for Element-level Bridge Life-cycle Optimization. [arXiv:2604.02528](https://arxiv.org/abs/2604.02528).

## Overview

The `softtree_ppo` package includes Proximal Policy Optimization (PPO) trainers compatible with both neural network (`PPOTrainer`) and soft tree (`SofttreePPOTrainer`) actors.

* **Working Examples:** The scripts `nbe107_training_nn.py` and `nbe107_training_softtree.py` provide working examples for neural network and soft tree actors, respectively. Both utilize the `example_nbe107` training environment from the `bridge-gym` package available [here](https://github.com/InfraRiskGroup/bridge-gym.git).

The `SofttreePPOTrainer` class features a dedicated method (`convert_to_obtree_actor`) designed to freeze and prune a soft tree, converting it into an interpretable oblique decision tree.
* **Working Example:** A working example of this conversion process is available in `nbe_validation_obtree.py`.

## Replicating Study Results

You can replicate the cited study's findings using the following provided scripts:

* **Training & Learning Curves:** Run `nbe107_training_nn.py` and `nbe107_training_softtree.py`.
* **Validation Results:** Run `nbe107_validation_nn.py` and `nbe107_validation_softtree.py`.
* **Oblique Decision Tree Validation:** To validate the oblique decision tree converted from a soft tree baseline, run `nbe_validation_obtree.py`.

> **Note:** The decision rules implemented in this codebase deviate slightly from those described in the cited paper. However, there is a simple mathematical conversion between the two. See the [softtree repository](https://github.com/InfraRiskGroup/softtree.git) for more details.
