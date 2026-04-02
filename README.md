# Obstacle Avoidance with Reinforcement Learning for a Soft Continuum Robot

This repository trains and evaluates DDPG controllers for a soft continuum robot with static obstacle avoidance.

For the base (non-obstacle-avoidance) project, see:
[RL-based-Control-of-a-Soft-Continuum-Robot](https://github.com/turhancan97/RL-based-Control-of-a-Soft-Continuum-Robot)

## Setup

### Option A: conda (recommended)

```bash
git clone https://github.com/turhancan97/RL-based-Obstacle-Avoidance-Soft-Continuum-Robot.git
cd RL-based-Obstacle-Avoidance-Soft-Continuum-Robot
conda env create -f environment.yml
conda activate continuum-rl
```

### Option B: pip

```bash
git clone https://github.com/turhancan97/RL-based-Obstacle-Avoidance-Soft-Continuum-Robot.git
cd RL-based-Obstacle-Avoidance-Soft-Continuum-Robot
pip install -r requirements.txt
```

## Runtime Model (v2)

The repository now uses a compatibility-first runtime:

- Canonical env class: `continuum_rl.env.ContinuumEnv`
- Backward alias: `continuum_rl.env.continuumEnv`
- Observation modes:
  - `canonical` (default): `4 + 2 * obstacle_count` features
  - `legacy4d`: 4 features (`x, y, goal_x, goal_y`) for old checkpoints
- Goal modes:
  - `fixed_goal`
  - `random_goal`
- Default obstacles: 3 static obstacles (configurable via Python API)

## Recommended CLI (Run From Repo Root)

Use root entrypoint:

```bash
python run.py -h
python run.py pytorch-train -h
python run.py pytorch-eval-smoke -h
python run.py keras-train -h
python run.py keras-eval-smoke -h
```

### 1) Training

#### PyTorch training

```bash
python run.py pytorch-train \
  --episodes 300 \
  --max-t 750 \
  --print-every 25 \
  --observation-mode canonical \
  --goal-type fixed_goal \
  --reward-function step_minus_weighted_euclidean \
  --reward-file reward_step_minus_weighted_euclidean
```

#### Keras training

```bash
python run.py keras-train \
  --episodes 500 \
  --max-steps 500 \
  --observation-mode canonical \
  --goal-type fixed_goal \
  --reward-function step_minus_weighted_euclidean \
  --reward-file reward_step_minus_weighted_euclidean
```

### 2) Inference / Evaluation (Smoke)

Smoke eval runs a short rollout and checks checkpoint/runtime compatibility.

#### PyTorch smoke eval

```bash
python run.py pytorch-eval-smoke \
  --observation-mode legacy4d \
  --goal-type fixed_goal \
  --reward-function step_minus_weighted_euclidean \
  --max-t 20 \
  --checkpoint-actor Pytorch/fixed_goal/reward_step_minus_weighted_euclidean/model/checkpoint_actor.pth \
  --checkpoint-critic Pytorch/fixed_goal/reward_step_minus_weighted_euclidean/model/checkpoint_critic.pth
```

#### Keras smoke eval

```bash
python run.py keras-eval-smoke \
  --observation-mode legacy4d \
  --goal-type fixed_goal \
  --reward-function step_minus_weighted_euclidean \
  --max-steps 20 \
  --checkpoint-actor Keras/fixed_goal/reward_step_minus_weighted_euclidean/model/continuum_actor.h5
```

`Keras` smoke eval accepts both `.h5` and `.weights.h5` paths.

### 3) Quick sanity runs

```bash
python run.py pytorch-train --episodes 1 --max-t 1 --observation-mode legacy4d
python run.py keras-train --episodes 1 --max-steps 1 --observation-mode legacy4d
```

## Reward Functions

Available reward functions:

- `step_error_comparison`
- `step_minus_euclidean_square`
- `step_minus_weighted_euclidean`
- `step_distance_based`

Set with `--reward-function ...` and choose a matching `--reward-file ...`.

## Output Artifacts

Training writes outputs in both new and legacy-compatible locations.

### PyTorch

- v2 model: `Pytorch/v2/<goal_type>/<reward_file>/model/`
- v2 rewards: `Pytorch/v2/<goal_type>/<reward_file>/rewards/`
- legacy-compatible model/rewards: `Pytorch/experiment/`
- files:
  - `checkpoint_actor.pth`
  - `checkpoint_critic.pth`
  - metadata sidecars: `*.metadata.json`

### Keras

- v2 model: `Keras/v2/<goal_type>/<reward_file>/model/`
- v2 rewards: `Keras/v2/<goal_type>/<reward_file>/rewards/`
- legacy-compatible model/rewards: `Keras/experiment/`
- files:
  - `continuum_actor.weights.h5`
  - `continuum_critic.weights.h5`
  - `continuum_target_actor.weights.h5`
  - `continuum_target_critic.weights.h5`
  - metadata sidecars: `*.metadata.json`

## Legacy / Direct Module Entry Points

You can still run framework modules directly:

```bash
python -m Pytorch.ddpg --mode train
python -m Pytorch.ddpg --mode eval-smoke --checkpoint-actor <actor.pth> --checkpoint-critic <critic.pth>

python -m Keras.DDPG --mode train
python -m Keras.DDPG --mode eval-smoke --checkpoint-actor <actor.h5>
```

Cluster scripts are also available:

- `Pytorch/train.sh`
- `Keras/train.sh`

## Visualization

### Reward curve visualization

```bash
python -m Pytorch.reward_visualization.reward_vis
python -m Keras.reward_visualization.reward_vis
```

These scripts auto-detect available reward folders when the configured folder is missing.
Saved plots are written to `<reward_dir>/plots/`.

### Legacy interactive demos

These are explicit demo scripts (not collected by pytest):

```bash
python Tests/DDPG_pytorch_test.py
python Tests/DDPG_keras_test.py
python Tests/amorphous_space_test.py
python Tests/polygon_space_test.py
```

Demo plots are saved under `Tests/visualizations/`.

## Testing and Quality Checks

```bash
pytest
ruff check continuum_rl Pytorch Keras tests run.py
python -m py_compile continuum_rl/*.py Pytorch/*.py Keras/*.py run.py
```

## Python API Example

```python
from continuum_rl.env import ContinuumEnv

env = ContinuumEnv(observation_mode="canonical", goal_type="random_goal")
obs, info = env.reset(seed=0)
```

## Troubleshooting

- Checkpoint mismatch error (`state_dim` mismatch):
  - Use `--observation-mode legacy4d` for old 4D checkpoints.
  - Or use checkpoints trained with matching mode.
- Missing reward folder for visualization:
  - The reward-vis scripts print available folder candidates.
- Gym/Gymnasium differences:
  - Compatibility is handled internally by `continuum_rl.gym_compat`.
