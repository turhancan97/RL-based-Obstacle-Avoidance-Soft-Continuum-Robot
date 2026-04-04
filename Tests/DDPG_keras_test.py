"""Legacy demo script for Keras policy evaluation."""

from __future__ import annotations

# ruff: noqa: E402

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

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
from Keras.DDPG import config, get_actor, validate_checkpoint_compatibility


def main() -> None:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
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
    state, _ = env.reset()
    env.time = 0
    env.start_kappa = [env.kappa1, env.kappa2, env.kappa3]
    env.render_init()

    repo_root = Path(__file__).resolve().parents[1]
    seed_dir = "seed_0"
    checkpoint_actor = (
        repo_root
        / "runs"
        / "keras"
        / config["goal_type"]
        / config["reward"]["file"]
        / seed_dir
        / "model"
        / "continuum_actor.weights.h5"
    )
    expected = {
        "state_dim": env.obs_size,
        "obstacle_count": env.num_obstacles,
        "goal_type": config["goal_type"],
        "reward_function": config["reward"]["function"],
    }
    validate_checkpoint_compatibility(checkpoint_actor, expected)

    resolved_actor = checkpoint_actor
    if not resolved_actor.exists():
        alt = checkpoint_actor.with_name("continuum_actor.h5")
        if alt.exists():
            resolved_actor = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_actor} (or {alt})")

    num_states = env.obs_size
    num_actions = env.action_space.shape[0]
    upper_bound = float(env.action_space.high[0])
    lower_bound = float(env.action_space.low[0])
    actor_model = get_actor(num_states, num_actions, upper_bound)
    actor_model.load_weights(resolved_actor)

    N = 500
    for step in range(N):
        start = time.time()
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = tf.squeeze(actor_model(tf_prev_state)).numpy()
        action = tf.clip_by_value(action, lower_bound, upper_bound).numpy()
        step_out = unpack_step_output(env.step(action, reward_function=config["reward"]["function"]))
        state, reward = step_out.obs, step_out.reward
        done = step_out.terminated or step_out.truncated

        storage["pos"]["x"].append(state[0])
        storage["pos"]["y"].append(state[1])
        env.render_calculate()

        print(f"{step}th action")
        print("Goal Position", state[2:4])
        if config["reward"]["function"] == "step_minus_euclidean_square":
            print("Error: {0}, Current State: {1}".format(math.sqrt(-1 * reward), state))
        else:
            print("Error: {0}, Current State: {1}".format(env.error, state))
        print("Action: {0},  Kappas {1}".format(action, [env.kappa1, env.kappa2, env.kappa3]))
        print("Reward is ", reward)
        print("--------------------------------------------------------------------------------")

        stop = time.time()
        env.time += stop - start
        if config["reward"]["function"] == "step_minus_euclidean_square":
            storage["error"]["error_store"].append(math.sqrt(-1 * reward))
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
    storage["reward"]["effectiveness"].append(step)

    print(f"{env.overshoot0} times robot tried to cross the task space")
    print(f"{env.overshoot1} times random goal was generated outside of the task space")
    print(f"Simulation took {(env.time)} seconds")
    effectiveness_score = float(sum(storage["reward"]["effectiveness"]) / len(storage["reward"]["effectiveness"]))
    print(f"Average Effectiveness Score is {effectiveness_score}")

    env.visualization(storage["pos"]["x"], storage["pos"]["y"])
    plt.xlabel("Position x [m]", fontsize=15)
    plt.ylabel("Position y [m]", fontsize=15)
    plt.savefig(output_dir / "keras_robot_trajectory.png", dpi=300, bbox_inches="tight")
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
        filename="keras_sub_plot_various_results.png",
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
            file_prefix="keras_plot_various_results",
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
        file_prefix="keras_plot_average_error",
    )
    plt.figure()
    plt.plot(storage["reward"]["value"], linewidth=4)
    plt.xlabel("Step")
    plt.ylabel(f"Reward {config['reward']['function']}")
    plt.savefig(output_dir / "keras_reward_per_step.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
