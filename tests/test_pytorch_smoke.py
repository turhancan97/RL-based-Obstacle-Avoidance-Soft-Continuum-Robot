from __future__ import annotations

from continuum_rl.env import ContinuumEnv
from continuum_rl.gym_compat import unpack_step_output
from Pytorch.ddpg_agent import Agent


def _run_agent_step(observation_mode: str):
    env = ContinuumEnv(observation_mode=observation_mode, goal_type="fixed_goal")
    agent = Agent(state_size=env.obs_size, action_size=3, random_seed=10)
    state, _ = env.reset(seed=0)
    action = agent.act(state, add_noise=False)
    step_out = unpack_step_output(env.step(action, reward_function="step_minus_weighted_euclidean"))
    assert step_out.obs.shape == (env.obs_size,)


def test_pytorch_smoke_canonical():
    _run_agent_step("canonical")


def test_pytorch_smoke_legacy4d():
    _run_agent_step("legacy4d")
