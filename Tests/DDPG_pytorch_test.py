"""Legacy demo script for PyTorch policy evaluation."""

from __future__ import annotations

# ruff: noqa: E402

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

if __package__ in (None, ""):
    from _bootstrap import ensure_repo_root_on_path
else:
    from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from continuum_rl.env import ContinuumEnv
from continuum_rl.gym_compat import unpack_step_output
from continuum_robot.utils import (
    plot_average_error,
    plot_various_results,
    sub_plot_various_results,
)
from Pytorch.ddpg import config, validate_checkpoint_compatibility
from Pytorch.ddpg_agent import Agent


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = {store_name: {} for store_name in ["error", "pos", "kappa", "reward"]}
    storage["error"]["error_store"] = []
    storage["error"]["x"] = []
    storage["error"]["y"] = []
    storage["pos"]["x"] = []
    storage["pos"]["y"] = []
    storage["kappa"]["kappa1"] = []
    storage["kappa"]["kappa2"] = []
    storage["kappa"]["kappa3"] = []
    storage["reward"]["value"] = []
    storage["reward"]["effectiveness"] = []

    env = ContinuumEnv(observation_mode="canonical", goal_type=config["goal_type"])
    agent = Agent(state_size=env.obs_size, action_size=3, random_seed=10)

    repo_root = Path(__file__).resolve().parents[1]
    seed_dir = "seed_0"
    checkpoint_actor = (
        repo_root
        / "runs"
        / "pytorch"
        / config["goal_type"]
        / config["reward"]["file"]
        / seed_dir
        / "model"
        / "checkpoint_actor.pth"
    )
    checkpoint_critic = (
        repo_root
        / "runs"
        / "pytorch"
        / config["goal_type"]
        / config["reward"]["file"]
        / seed_dir
        / "model"
        / "checkpoint_critic.pth"
    )
    expected = {
        "state_dim": env.obs_size,
        "obstacle_count": env.num_obstacles,
        "goal_type": config["goal_type"],
        "reward_function": config["reward"]["function"],
    }
    validate_checkpoint_compatibility(checkpoint_actor, expected)
    validate_checkpoint_compatibility(checkpoint_critic, expected)

    agent.actor_local.load_state_dict(torch.load(checkpoint_actor, map_location=torch.device("cpu")))
    agent.critic_local.load_state_dict(torch.load(checkpoint_critic, map_location=torch.device("cpu")))

    state, _ = env.reset()
    env.time = 0.0
    env.start_kappa = [env.kappa1, env.kappa2, env.kappa3]
    env.render_init()
    n_steps = 750

    for t in range(n_steps):
        start = time.time()
        action = agent.act(state, add_noise=False)
        step_out = unpack_step_output(env.step(action, reward_function=config["reward"]["function"]))
        state, reward = step_out.obs, step_out.reward
        done = step_out.terminated or step_out.truncated

        storage["pos"]["x"].append(state[0])
        storage["pos"]["y"].append(state[1])
        env.render_calculate()

        print(f"{t}th action")
        print("Goal Position", state[2:4])
        if config["reward"]["function"] == "step_minus_euclidean_square":
            print("Error: {0}, Current State: {1}".format(math.sqrt(max(-1 * reward, 0.0)), state))
        else:
            print("Error: {0}, Current State: {1}".format(env.error, state))
        print("Action: {0},  Kappas {1}".format(action, [env.kappa1, env.kappa2, env.kappa3]))
        print("Reward is ", reward)
        print("--------------------------------------------------------------------------------")

        stop = time.time()
        env.time += stop - start

        if config["reward"]["function"] == "step_minus_euclidean_square":
            storage["error"]["error_store"].append(math.sqrt(max(-1 * reward, 0.0)))
        else:
            storage["error"]["error_store"].append(env.error)
        storage["kappa"]["kappa1"].append(env.kappa1)
        storage["kappa"]["kappa2"].append(env.kappa2)
        storage["kappa"]["kappa3"].append(env.kappa3)
        storage["error"]["x"].append(abs(state[0] - state[2]))
        storage["error"]["y"].append(abs(state[1] - state[3]))
        storage["reward"]["value"].append(reward)

        if done:
            break

    storage["reward"]["effectiveness"].append(t)

    print(f"{env.overshoot0} times robot tried to cross the task space")
    print(f"{env.overshoot1} times random goal was generated outside of the task space")
    print(f"Simulation took {(env.time)} seconds")
    effectiveness_score = float(sum(storage["reward"]["effectiveness"]) / len(storage["reward"]["effectiveness"]))
    print(f"Average Effectiveness Score is {effectiveness_score}")

    env.visualization(storage["pos"]["x"], storage["pos"]["y"])
    plt.xlabel("Position x [m]", fontsize=15)
    plt.ylabel("Position y [m]", fontsize=15)
    plt.savefig(output_dir / "pytorch_robot_trajectory.png", dpi=300, bbox_inches="tight")
    plt.show()

    sub_plot_various_results(
        error_store=storage["error"]["error_store"],
        error_x=storage["error"]["x"],
        error_y=storage["error"]["y"],
        pos_x=storage["pos"]["x"],
        pos_y=storage["pos"]["y"],
        kappa_1=storage["kappa"]["kappa1"],
        kappa_2=storage["kappa"]["kappa2"],
        kappa_3=storage["kappa"]["kappa3"],
        goal_x=state[2],
        goal_y=state[3],
        output_dir=output_dir,
        filename="pytorch_sub_plot_various_results.png",
    )
    for plot_choice in (1, 2, 3):
        plot_various_results(
            plot_choice=plot_choice,
            error_store=storage["error"]["error_store"],
            error_x=storage["error"]["x"],
            error_y=storage["error"]["y"],
            pos_x=storage["pos"]["x"],
            pos_y=storage["pos"]["y"],
            kappa_1=storage["kappa"]["kappa1"],
            kappa_2=storage["kappa"]["kappa2"],
            kappa_3=storage["kappa"]["kappa3"],
            goal_x=state[2],
            goal_y=state[3],
            output_dir=output_dir,
            file_prefix="pytorch_plot_various_results",
        )
    # Keep plot_average_error robust when rollout terminates before full horizon.
    error_window = max(len(storage["error"]["error_store"]), 1)
    plot_average_error(
        error_x=storage["error"]["x"],
        error_y=storage["error"]["y"],
        error_store=storage["error"]["error_store"],
        N=error_window,
        episode_number=1,
        output_dir=output_dir,
        file_prefix="pytorch_plot_average_error",
    )

    plt.figure()
    plt.plot(storage["reward"]["value"], linewidth=4)
    plt.xlabel("Step")
    plt.ylabel(f"Reward {config['reward']['function']}")
    plt.savefig(output_dir / "pytorch_reward_per_step.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
