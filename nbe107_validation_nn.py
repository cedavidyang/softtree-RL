#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from torchrl.envs import GymWrapper

from bridge_gym.example_nbe107.rl_env import SingleElement
from bridge_gym.example_nbe107.settings import CS_PFS
from softtree_ppo.training import PPOTrainer

from nbe107_training_nn import max_steps, gamma
from nbe107_training_nn import include_step_count
from nbe107_training_nn import alpha_vector

# %%

if __name__ == '__main__':
    actor_path = "./actors/nn_64x1_200yr_run1.pt"
    save_path = "./results/val_nn_64x1_200yr_run1.csv"
    
    env_seed = 508
    num_episodes = 1000
    cost_kwargs = {"normalizer": 1}

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
    
    actor = PPOTrainer.load_actor(
        actor_path,
        env.action_spec,
    )

    eval_log = PPOTrainer.evaluate(
        actor,
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

    # save results
    val_res = {
        'init_beta': init_beta,
        'eval_costs': eval_costs
    }
    pd.DataFrame(val_res).to_csv(
        save_path,
        index=False
    )
# %%
