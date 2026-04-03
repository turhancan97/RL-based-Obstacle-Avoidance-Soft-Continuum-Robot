# Obstacle Avoidance with Reinforcement Learning for a Soft Continuum Robot

This repository trains and evaluates DDPG controllers (PyTorch and Keras) for a soft continuum robot with static obstacle avoidance.

For the base (non-obstacle-avoidance) project, see:
[RL-based-Control-of-a-Soft-Continuum-Robot](https://github.com/turhancan97/RL-based-Control-of-a-Soft-Continuum-Robot)

## Runtime Contract

- Observation mode is canonical-only.
- Canonical observation schema is `canonical_v3`.
- Canonical observation shape is `7 + 2 * obstacle_count`:
  `[x, y, goal_x, goal_y, kappa1, kappa2, kappa3, obstacle_x..., obstacle_y...]`.
- DDPG architecture is standardized across frameworks:
  - Actor: MLP `128 -> 128 -> action(tanh)`
  - Critic: concat(state, action) -> MLP `128 -> 128 -> Q`
- Default obstacle count is 3 static obstacles.
- Supported goal modes: `fixed_goal`, `random_goal`.
- Checkpoints must include matching metadata (`state_dim`, `obs_schema`, etc.); older checkpoints are intentionally incompatible.

## Setup

### Option A: conda

```bash
git clone https://github.com/turhancan97/RL-based-Obstacle-Avoidance-Soft-Continuum-Robot.git
cd RL-based-Obstacle-Avoidance-Soft-Continuum-Robot
conda env create -f environment.yml
conda activate continuum-rl
pip install -e .
```

### Option B: pip

```bash
git clone https://github.com/turhancan97/RL-based-Obstacle-Avoidance-Soft-Continuum-Robot.git
cd RL-based-Obstacle-Avoidance-Soft-Continuum-Robot
pip install -e .
```

Python target is `>=3.9,<3.10`.

## Primary CLI (Hydra)

Hydra is the primary interface. Use the console script:

```bash
continuum-rl --help
```

Single app task selector:

- `task=pytorch_train`
- `task=pytorch_eval_smoke`
- `task=keras_train`
- `task=keras_eval_smoke`
- `task=pytorch_reward_vis`
- `task=keras_reward_vis`
- `task=paper_figures`
- `task=gradio_demo`

### Training

PyTorch:

```bash
continuum-rl task=pytorch_train
```

Keras:

```bash
continuum-rl task=keras_train
```

### Inference / Smoke Eval

PyTorch:

```bash
continuum-rl task=pytorch_eval_smoke \
  task.checkpoint_actor=runs/pytorch/fixed_goal/reward_step_minus_weighted_euclidean/seed_0/model/checkpoint_actor.pth \
  task.checkpoint_critic=runs/pytorch/fixed_goal/reward_step_minus_weighted_euclidean/seed_0/model/checkpoint_critic.pth
```

Keras:

```bash
continuum-rl task=keras_eval_smoke \
  task.checkpoint_actor=runs/keras/fixed_goal/reward_step_minus_weighted_euclidean/seed_0/model/continuum_actor.h5
```

Smoke eval expects checkpoints to exist already. Run training first if checkpoint files are missing.
After the `canonical_v3` + architecture-standardization update, retrain models before running eval.

### Reward Visualization

PyTorch:

```bash
continuum-rl task=pytorch_reward_vis
```

Keras:

```bash
continuum-rl task=keras_reward_vis
```

Plots are saved under `<framework>/<goal_type>/<reward_type>/seed_<id>/rewards/plots/`.

### Conference-Grade Paper Figures

Generate the full paper figure suite (headless, JPEG-only, deterministic output path):

```bash
continuum-rl task=paper_figures
```

Input contract is strict local layout (no W&B ingestion):

```text
runs/<framework>/<goal_type>/<reward_id>/seed_<seed_id>/
  rewards/
    avg_reward_list.pickle
    scores.pickle               # PyTorch
    ep_reward_list.pickle       # Keras
  model/
    checkpoint_actor.pth        # PyTorch
    checkpoint_critic.pth       # PyTorch
    continuum_actor.weights.h5  # Keras
```

Default output:

```text
figures/paper/latest/
  *.jpeg
  manifest.json
```

Key overrides:

```bash
continuum-rl task=paper_figures \
  task.runs_root=runs \
  task.output_dir=figures/paper/latest \
  task.rollouts_per_seed=100 \
  task.include_goal_types=[fixed_goal,random_goal] \
  task.max_steps=750 \
  task.show=false
```

### Interactive Gradio Demo

Install UI extras once:

```bash
pip install -e .[ui]
```

Launch:

```bash
continuum-rl task=gradio_demo
```

Example overrides:

```bash
continuum-rl task=gradio_demo \
  task.framework=keras \
  task.control_mode=manual \
  task.max_steps=400 \
  task.seed=7 \
  task.device=cpu
```

Compatibility wrapper (deprecated):

```bash
python run.py gradio-demo --framework pytorch --control-mode policy
```

## Hydra Override Examples

Goal mode:

```bash
continuum-rl task=pytorch_train task.goal_type=random_goal
```

Reward function + folder:

```bash
continuum-rl task=keras_train \
  task.reward_function=step_distance_based

`task.reward_file` is optional and auto-derived as `reward_<reward_function>`.  
Set it explicitly only if you want a custom artifact folder name.
```

Episode / step counts:

```bash
continuum-rl task=pytorch_train task.episodes=10 task.max_t=100
continuum-rl task=keras_train task.episodes=10 task.max_steps=100
```

Reproducible and deterministic execution:

```bash
continuum-rl task=pytorch_train task.seed=123 task.deterministic=true
continuum-rl task=keras_train task.seed=123 task.deterministic=true
```

Output directories:

```bash
continuum-rl task=pytorch_train task.output_base_dir=runs/pytorch
continuum-rl task=keras_train task.output_base_dir=runs/keras
```

## Artifacts

### PyTorch

- model: `runs/pytorch/<goal_type>/<reward_file>/seed_<seed_id>/model/`
- rewards: `runs/pytorch/<goal_type>/<reward_file>/seed_<seed_id>/rewards/`
- files:
  - `checkpoint_actor.pth`
  - `checkpoint_critic.pth`
  - metadata sidecars: `*.metadata.json`
  - metadata includes `obs_schema: canonical_v3`
  - metadata includes `model_arch: ddpg_mlp_actor_128x128_critic_128x128_concat`

### Keras

- model: `runs/keras/<goal_type>/<reward_file>/seed_<seed_id>/model/`
- rewards: `runs/keras/<goal_type>/<reward_file>/seed_<seed_id>/rewards/`
- files:
  - `continuum_actor.weights.h5`
  - `continuum_critic.weights.h5`
  - `continuum_target_actor.weights.h5`
  - `continuum_target_critic.weights.h5`
  - metadata sidecars: `*.metadata.json`
  - metadata includes `obs_schema: canonical_v3`
  - metadata includes `model_arch: ddpg_mlp_actor_128x128_critic_128x128_concat`

## Training/Eval Semantics

- TD terminal masking uses only true termination (`terminated=True`), not time-limit truncation.
- Success counters track true goal-reaching terminations only.
- Truncation counts are reported separately.

## Deprecated Compatibility Commands

These still work during the deprecation window but emit warnings and will be removed in the next release milestone.

Legacy wrapper entrypoint:

```bash
python run.py pytorch-train --episodes 300 --max-t 750
python run.py pytorch-eval-smoke --checkpoint-actor <path> --checkpoint-critic <path>
python run.py keras-train --episodes 500 --max-steps 500
python run.py keras-eval-smoke --checkpoint-actor <path>
python run.py pytorch-reward-vis
python run.py keras-reward-vis
python run.py gradio-demo --framework pytorch --control-mode policy
```

Legacy module entrypoints:

```bash
python -m Pytorch.ddpg --mode train
python -m Pytorch.ddpg --mode eval-smoke --checkpoint-actor <actor.pth> --checkpoint-critic <critic.pth>

python -m Keras.DDPG --mode train
python -m Keras.DDPG --mode eval-smoke --checkpoint-actor <actor.h5>
```

## Dependency Source of Truth

Dependencies are pinned in `pyproject.toml`.

Compatibility exports are generated one-way:

```bash
python scripts/export_dependency_files.py
python scripts/export_dependency_files.py --check
```

This keeps `requirements.txt` and `environment.yml` aligned with `pyproject.toml`.

## Repository Cleanup

Use the deterministic cleanup workflow to remove generated files and redundant local artifacts:

```bash
python scripts/cleanup_repo.py
python scripts/cleanup_repo.py --apply
```

## Tests and Quality Checks

```bash
pytest
ruff check continuum_rl Pytorch Keras tests run.py scripts
python -m py_compile continuum_rl/*.py Pytorch/*.py Keras/*.py scripts/*.py run.py
```

## Demo Scripts

```bash
python Tests/DDPG_pytorch_test.py
python Tests/DDPG_keras_test.py
python Tests/amorphous_space_test.py
python Tests/polygon_space_test.py
```

Demo plots are saved under `Tests/visualizations/`.
