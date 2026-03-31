#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from torchrl.envs import GymWrapper

from bridge_gym.example_nbe107.rl_env import SingleElement
from bridge_gym.example_nbe107.settings import CS_PFS
from softtree_ppo.training import SofttreePPOTrainer

from nbe107_training_nn import max_steps, gamma
from nbe107_training_nn import include_step_count
from nbe107_training_nn import alpha_vector

# %%

if __name__ == '__main__':
    env_seed = 1034
    pruning_threshold = 1e-3
    lp_threshold = 1e-6
    num_episodes = 10_000
    cost_kwargs = {"normalizer": 1}

    actor_path = "./actors/softtree_d10b1lm1e-02_200yr_run11.pt"
    save_path = f"./results/val_obt_d10b1lm1e-02_200yr_{pruning_threshold:.0e}prune_run11.csv"

    gym_env = SingleElement(
        max_steps=max_steps, discount=gamma,
        include_step_count=include_step_count,
        reset_prob=None,
        dirichlet_alpha=alpha_vector,
        render_mode="ansi",
        seed=env_seed,
        cost_kwargs=cost_kwargs,
    )
    env = GymWrapper(gym_env, categorical_action_encoding=True)
    
    # get oblique tree actor
    STC_actor = SofttreePPOTrainer.load_actor(
        actor_path,
        env.action_spec,
    )
    OBT_actor, prune_mask = SofttreePPOTrainer.convert_to_obtree_actor(
        STC_actor, pruning_threshold=pruning_threshold,
        lp_threshold=lp_threshold,
        A_ub=[],
        b_ub=[],
        bounds=(0, 1),
    )

    # evaluate oblique tree actor
    eval_log = SofttreePPOTrainer.evaluate(
        OBT_actor,
        env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=True,
    )

    # plot testing results
    if include_step_count:
        init_states = np.array(eval_log["init_state"])[:, :-1]
    else:
        init_states = np.array(eval_log["init_state"])
    init_pf = init_states @ CS_PFS
    init_beta = -stats.norm.ppf(init_pf)
    eval_costs = -np.array(eval_log["eval_reward"])
    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        sns.scatterplot(x=init_beta, y=eval_costs, ax=ax)
        # ax.set_ylim(0, 1e6)

    # print results
    print(f"mean: {np.mean(eval_costs)} \t std: {np.std(eval_costs)}")

    # save results
    candidate_nodes = np.sum(prune_mask).item()
    internal_nodes = OBT_actor.module.tree.internal_num
    leaf_nodes = OBT_actor.module.tree.leaf_num
    pruned_internal = 2**OBT_actor.module.tree.max_depth - 1 - internal_nodes
    pruned_leaf = 2**OBT_actor.module.tree.max_depth - leaf_nodes
    val_res = {
        'init_beta': init_beta,
        'eval_costs': eval_costs,
        'internal_nodes': internal_nodes,
        'leaf_nodes': leaf_nodes,
        'candidate_nodes': candidate_nodes,
        'pruned_internal': pruned_internal,
        'pruned_leaf': pruned_leaf
    }
    pd.DataFrame(val_res).to_csv(
        save_path,
        index=False
    )
# %%
