# VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands

<div align="center">
<a href="https://vaic-humanoid.github.io/">
	<img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://arxiv.org/abs/2606.09286">
	<img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/ldt29/VAIC/stargazers">
	<img alt="GitHub stars" src="https://img.shields.io/github/stars/ldt29/VAIC?style=social"/>
</a>

</div>

This repository hosts the open-source release for the paper VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands.


## 🚀 Quick Start

```bash
# setup conda environment
conda create -n vaic python=3.11 -y
conda activate vaic

# install isaacsim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
isaacsim # test isaacsim

# install isaaclab
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh -i none

# install vaic
cd ..
git clone https://github.com/ldt29/VAIC
cd VAIC
pip install -e .
```

## Verify Your Data
Visualize motions in Isaac Sim with `task.command.replay_motion=true`:

```bash
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea task.command.replay_motion=true
```


## Train and Evaluate

Teacher policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_train task=G1/vaic/skateboard_tea
# evaluate policy
python scripts/play.py algo=ppo_vel_train task=G1/vaic/skateboard_tea checkpoint_path=run:<wandb-run-path>
```

Student policy

```bash
# train policy
python scripts/train.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
# evaluate policy
python scripts/play.py algo=ppo_vel_finetune task=G1/vaic/skateboard_stu checkpoint_path=run:<student_wandb-run-path>
```
To export trained policies, add `export_policy=true` to the play script.

Teacher policy with FlashSAC on the flat observation

`flashsac_vel_flat_train` trains the teacher with [FlashSAC](https://github.com/Holiday-Robot/FlashSAC) on the same task, rewards, observations and simulation as `ppo_vel_train`. The `flat` suffix indicates that its actor and critic share one flat observation vector.

```bash
# train policy (uses 1024 envs)
python scripts/train.py algo=flashsac_vel_flat_train task=G1/vaic/skateboard_general_tracking_tea
# evaluate policy (FlashSAC reads raw observations, so VecNorm must be off)
python scripts/play.py algo=flashsac_vel_flat_train task=G1/vaic/skateboard_general_tracking_tea vecnorm=null checkpoint_path=run:<wandb-run-path>
```

- The FlashSAC networks and update rule are vendored unmodified in `active_adaptation/learning/ppo/flashsac_upstream/` (commit and per-file sha256 in `SOURCE.json`). Hyperparameters follow upstream `scripts/run_isaaclab.sh` (1024 envs, 2 updates per env step, batch 2048, 3-step returns) except where noted: here the actor is 256 wide (`algo.actor_hidden_dim`, upstream 128). Updates start after 100K transitions (`algo.buffer_min_length`).
- The actor and the critic read the same observation, 1187 dims for the skateboard tasks: `command`, `policy`, `priv`, `object_`, `action_dr` and `ref_joint_pos_`, reduced as follows (the env still computes everything, so rewards and terminations are unchanged):
  - `algo.obs_keys` picks the groups. The object point cloud (`object_trans`, 384 dims) is left out; add `object_trans` to use it. `action_dr` is privileged: this episode's action delay (physics substeps) and low-pass filter alpha from `JointPosition`.
  - `algo.obs_drop_terms` removes single terms as `group/term`. By default it drops the noisy copies in `policy` of values `priv` holds without noise (`root_ang_vel_history`, `projected_gravity_history`, `joint_pos_history`, 180 dims), keeping `prev_actions`.
  - `algo.obs_future_steps` sets which of the task's future steps `[1, 2, 8, 16, 32]` every future-reference term keeps (every term whose name contains `future`, 11 in these tasks); default `[8, 32]`, `null` keeps all.
  - Checkpoints store this layout and refuse to load into a different one. Earlier layouts load with these overrides (both were trained with the 128-wide actor):
    - 2748 dims (runs before the observation options): `algo.actor_hidden_dim=128 algo.obs_keys=[command,policy,priv,object_,object_trans] algo.obs_drop_terms=[] algo.obs_future_steps=null`
    - 1320 dims (future steps `[1, 32]` on the per-body terms only): `algo.actor_hidden_dim=128 algo.obs_keys=[command,policy,priv,object_] algo.obs_future_steps=[1,32] algo.obs_future_terms=[command/ref_body_pos_future_local,priv/diff_body_pos_future_local,priv/diff_body_ori_future_local,priv/diff_body_lin_vel_future_local,priv/diff_body_ang_vel_future_local]`
- `algo.action_mode` sets how the actor output `u = tanh(z)` becomes the joint command. `ref_joint_pos_` stays in the observation either way, and checkpoints record the mode and scale and refuse to load into a different one.
  - `residual` (default): `action = ref_joint_pos_ + residual_scale * tanh(z)`. The policy starts on the reference motion, which is why it learns roughly twice as fast per frame early on, but it can never leave a band of `+-residual_scale` raw action units around the reference. `algo.residual_scale` defaults to 3.0; it was 1.0 until measurements on a trained `ppo_vel_flat` teacher showed that band binding — that policy commands `|action - ref_joint_pos_|` with p99 2.47 and a mean of 1.21 on both ankle pitch joints, and 94% of its control states need at least one joint beyond 1.0.
  - `absolute`: `action = action_scale * tanh(z)`, as upstream FlashSAC, which has no reference to center on. No band, but no reference prior either, so expect a much slower start. `algo.action_scale` defaults to 4.0, covering the p99 of 2.46 for `|action|` and 1.66 for `|ref_joint_pos_|` measured on that same teacher.
  - Note that until `algo.buffer_min_length` transitions are collected the actor is bypassed and `u` is drawn uniformly from `[-1, 1]`, so both scales also set how violent the warmup is.
- `algo.critic_state_encoder_dim` and `algo.critic_action_encoder_dim` (both default `256`) balance the inputs to the FlashSAC critic. Each Q head maps state with `BatchNorm -> unit linear -> ReLU` and action with `unit linear -> ReLU`, then concatenates the branch features. There is no branch-output BatchNorm; the original FlashSAC embedder applies `BatchNorm -> unit linear` immediately after concatenation and the rest of its critic is unchanged. The defaults change the 1187-state/23-action fusion into 256 state features plus 256 action features; set both options to `null` to restore the upstream critic. Checkpoints record both dimensions and the encoder architecture; checkpoints from the earlier branch encoders are not compatible.
- Command completion bootstraps like a time limit by default (`algo.bootstrap_on_command_finished=true`), matching this PPO implementation. Set it to `false` to treat command completion as terminal.
- Replay size is the one setting limited by hardware. Observations are stored as float16 (`algo.buffer_obs_dtype`), and the default `algo.buffer_max_length=6000000` takes 13.9 GiB of GPU memory: 12% of a 50M-frame run and 0.375% of the default 1.6B. Upstream IsaacLab keeps 10M transitions, 20% of its 50M-step runs. Memory scales with rows x observation dims, so revisit the buffer length when changing the observation options.
- `algo.buffer_device_type=cpu` keeps the replay in RAM instead (10M rows take 23 GiB at 1187 dims); batches are staged through pinned memory to the GPU, which made training about 5% slower in a 1024-env run. The run checks the available RAM before allocating.
- Besides the usual episode statistics, the off-policy loop logs how non-terminated episodes ended: `train/stats/episode_time_limit` and `train/stats/command_finished` (fractions of finished episodes).
- The learning-rate schedule spans `total_frames`; upstream IsaacLab runs use `total_frames=50_000_896`.

Actuator settings are part of the training distribution. A policy trained with
`task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0` has constant
`action_dr` inputs. Their BatchNorm variance can approach zero, making evaluation
with the task's default delay (2–6 physics substeps) and alpha (0.8–1.0) fail even
when tracking with the training actuator works well. Checkpoint loading now logs
these mismatches. To transfer an existing policy, see the measured
[default-delay investigation](diagnostics/flashsac_improve_20260918/REPORT.md) and
[fine-tuning/evaluation commands](diagnostics/flashsac_improve_20260918/REPRODUCE.md).
The experiment keeps the task's default physics, rewards, and termination rules.

Teacher policy with PPO on the FlashSAC observation

`ppo_vel_flat_train` isolates the effect of the input: PPO with `ppo_vel`'s networks, sizes and hyperparameters, reading the single flat vector `flashsac_vel_flat_train` builds instead of `ppo_vel_train`'s grouped tensors and adaptation modules.

```bash
python scripts/train.py algo=ppo_vel_flat_train task=G1/vaic/skateboard_general_tracking_tea
python scripts/play.py algo=ppo_vel_flat_train task=G1/vaic/skateboard_general_tracking_tea checkpoint_path=run:<wandb-run-path>
```

- `active_adaptation/learning/ppo/ppo_vel_flat.py` is self-contained: it imports nothing from `ppo_vel.py` or `flashsac_vel_flat.py`, so either can change without affecting it (and the observation options are duplicated there, not shared).
- The observation is the one described above and uses the same option names (`algo.obs_keys`, `algo.obs_drop_terms`, `algo.obs_future_steps`, `algo.obs_future_terms`), 1187 dims by default. Checkpoints store the selected indices and refuse to load into a different selection.
- Actor `[512, 256, 256]`, critic `[512, 256, 128]`, one value head per reward group, `lr=3e-4` with the `desired_kl=0.01` adaptive schedule, 3 epochs x 8 minibatches, `clip_param=0.2`, `gamma=0.99`, `lmbda=0.95` — all as in `ppo_vel`.
- The actor emits the joint command directly (`action = loc`), like `ppo_vel_finetune`, not as a residual on `ref_joint_pos_` like `ppo_vel_train` and `flashsac_vel_flat_train`. `ref_joint_pos_` is still the last block of the observation.
- `vecnorm` defaults to `train`: PPO reads raw-scale observations and needs the running normalizer, where FlashSAC's BatchNorm embedder does that job itself.
- The number of envs comes from the task (unlike `flashsac_vel_flat_train`, which fixes 1024); pass `task.num_envs=1024` to match FlashSAC's rollout width.


## Acknowledgments

This repository is built on top of [HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos](https://github.com/LeCAR-Lab/HDMI). We thank the authors for open-sourcing their work.

## Citation

If you find our work useful for your research, please consider citing us:

```bibtex
@article{li2026vaic,
  title = {VAIC: Vision-Guided Humanoid Agile Object Interaction Control via Decoupled Commands},
  author = {Li, Dongting and Wu, Qianyang and Chen, Xingyu and Li, Liang and Lin, Yuhang and Wu, Sikai and Zhang, Guoyao and Zhou, Mingliang and Xiang, Diyun and Zhang, Qiang and Xu, Renjing and Ma, Jianzhu},
  journal = {arXiv preprint arXiv:2606.09286},
  year = {2026}
}
```
