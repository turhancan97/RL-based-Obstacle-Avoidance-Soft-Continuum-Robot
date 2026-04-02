"""Legacy demo script for PyTorch policy evaluation."""

from __future__ import annotations

# ruff: noqa: E402

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
from Pytorch.ddpg import config, validate_checkpoint_compatibility
from Pytorch.ddpg_agent import Agent


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ContinuumEnv(observation_mode="canonical", goal_type=config["goal_type"])
    agent = Agent(state_size=env.obs_size, action_size=3, random_seed=10)

    checkpoint_actor = Path(
        f"Pytorch/{config['goal_type']}/{config['reward']['file']}/model/checkpoint_actor.pth"
    )
    checkpoint_critic = Path(
        f"Pytorch/{config['goal_type']}/{config['reward']['file']}/model/checkpoint_critic.pth"
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
    env.start_kappa = [env.kappa1, env.kappa2, env.kappa3]
    initial_state = state[0:2]
    x_pos = []
    y_pos = []

    for t in range(750):
        start = time.time()
        action = agent.act(state, add_noise=False)
        step_out = unpack_step_output(env.step(action, reward_function=config["reward"]["function"]))
        state = step_out.obs
        x_pos.append(state[0])
        y_pos.append(state[1])
        print(f"{t}th action")
        print("Goal Position", state[2:4])
        print("Action: {0},  Kappas {1}".format(action, [env.kappa1, env.kappa2, env.kappa3]))
        print("Episodic Reward is {}".format(step_out.reward))
        print("--------------------------------------------------------------------------------")
        stop = time.time()
        env.time += stop - start
        if step_out.terminated or step_out.truncated:
            break

    env.visualization(x_pos, y_pos)
    plt.title(
        f"Initial Position is x: {initial_state[0]} y: {initial_state[1]} & "
        f"Target Position is x: {state[0]} y: {state[1]}"
    )
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.savefig(output_dir / "pytorch_robot_trajectory.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
