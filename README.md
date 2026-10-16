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

FlashSAC implementations, finetune variants, the adaptive schedule helper, and vendored
upstream code live in `active_adaptation/learning/flashsac/`. PPO implementations remain
in `active_adaptation/learning/ppo/`; FlashSAC imports shared observation/perception
components from there. Algorithm names such as `algo=flashsac_vel_finetune` are unchanged.
New configs use `active_adaptation.learning.flashsac.*` targets; old `learning.ppo.*`
FlashSAC imports remain aliases for loading existing configs and checkpoints.

`flashsac_vel_flat_train` trains the teacher with [FlashSAC](https://github.com/Holiday-Robot/FlashSAC) on the same task, rewards, observations and simulation as `ppo_vel_train`. The `flat` suffix indicates that its actor and critic share one flat observation vector.

```bash
# train policy (uses 1024 envs)
python scripts/train.py algo=flashsac_vel_flat_train task=G1/vaic/skateboard_general_tracking_tea
# evaluate policy (FlashSAC reads raw observations, so VecNorm must be off)
python scripts/play.py algo=flashsac_vel_flat_train task=G1/vaic/skateboard_general_tracking_tea vecnorm=null checkpoint_path=run:<wandb-run-path>
```

- The FlashSAC networks and update rule are vendored unmodified in `active_adaptation/learning/flashsac/flashsac_upstream/` (commit and per-file sha256 in `SOURCE.json`). Hyperparameters follow upstream `scripts/run_isaaclab.sh` (1024 envs, 2 updates per env step, batch 2048, 3-step returns) except where noted: here the actor is 256 wide (`algo.actor_hidden_dim`, upstream 128). Updates start after 100K transitions (`algo.buffer_min_length`).
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

Teacher policy with VAIC adaptation and FlashSAC

`flashsac_vel_train` keeps the privileged encoder, GRU adaptation module, object estimator,
and `actor_adapt` interfaces from `ppo_vel_train`. Its teacher actor and critic use the
FlashSAC architecture and update rule. The actor reads `[command, policy, priv_feature]`;
the critic keeps the flat observation described above.

```bash
python scripts/train.py algo=flashsac_vel_train task=G1/vaic/skateboard_general_tracking_tea
```

The replay buffer is stored in CPU RAM by default. The equivalent explicit command is:

```bash
python scripts/train.py \
  algo=flashsac_vel_train \
  task=G1/vaic/skateboard_general_tracking_tea \
  algo.buffer_device_type=cpu
```

Use `algo.buffer_device_type=cuda` to keep it on the GPU instead.

- Long-term replay stores full `command` and `policy`, the critic-selected subset of `priv`,
  and the remaining critic fields. Every sampled
  current and bootstrap observation is passed through the latest encoder, so replay never
  contains stale `priv_feature` values. The task's fixed object point template is stored once
  in the actor; `object_trans` is recomputed from each replayed `object_` pose and is not stored
  in every replay row.
- On the skateboard task this gives a 1,580-dimensional replay row: the actor keeps PPO's full
  command and policy inputs, while both its privileged encoder and the critic use the critic's
  928-dimensional privileged selection. At the default length this is about 30.42 GiB in host RAM.
  The algorithm defaults to `algo.buffer_max_length=10000000`, checks available RAM before
  allocation, and asynchronously stages sampled minibatches through pinned memory to the GPU.
- The recurrent adaptation module is not trained from random long-term replay entries.
  It and `actor_adapt` train from each latest contiguous `algo.train_every` rollout, matching
  the sequence treatment in `ppo_vel_train`.
- `actor_adapt` uses the same `FlashSACActor` backbone, UnitLinear/UnitNorm parameter
  constraints, and tanh-Gaussian action parameterization as the teacher. Its semantic input
  remains the PPOVEL student input `[vel_command, policy, priv_pred]`. On the skateboard task
  this is 525 dimensions: 20 velocity-command values, 249 policy values, and a 256-dimensional
  latent. Both PPOVEL and FlashSACVEL use this same student input.
- `actor_adapt` distills the teacher's final joint command with MSE. With residual actions both
  teacher and student commands are `ref_joint_pos_ + residual_scale * tanh(mean)`.
- W&B uses the same adaptation metric names as `ppo_vel_train`: `adapt/priv_loss` is the
  privileged-feature prediction MSE and `adapt/adapt_loss` is the actor distillation MSE.
- Continue with `flashsac_vel_finetune` below to train depth perception and then freeze it
  for student SAC training.

Adaptive FlashSAC student finetuning

```bash
python scripts/train.py \
  algo=flashsac_vel_finetune \
  task=G1/vaic/skateboard_general_tracking_stu \
  checkpoint_path=/home/hcc/research/VAIC/outputs/2026-09-23/03-53-05-G1SkateboardGeneralTracking-flashsac_vel/wandb/latest-run/files/checkpoint_3000.pt
```

Defaults: 4096 camera environments, `vecnorm=null`, 32 steps per iteration, CPU replay.
The student task enables cameras and retains its configured depth noise/delay. Teacher
checkpoints must contain the FlashSAC `actor_adapt` (`actor_adapt_arch=flashsac_v1`).

Headless finetuning defaults to `algo.enable_rtx=false`: depth images are computed
by the CUDA ray-cast sensor, so the RTX renderer is unnecessary. Keep
`task.enable_cameras=true` to retain depth observations. On the 512-env skateboard
task, disabling RTX reduced process RAM after environment/model initialization
from about 26.8 GiB to 8.0 GiB. Models and perception still run on the GPU. RGB
evaluation (`eval_render=true`) automatically enables RTX; it can also be enabled
explicitly with `app.enable_cameras=true` or `algo.enable_rtx=true`.

Stage 1 action noise is controlled by `algo.teacher_perception_rollout_noise_scale=1.6`:
`u = tanh(teacher_mean + teacher_std * noise * scale)`. Set it to `0.5` for half
the pre-tanh noise amplitude or `0.0` for deterministic teacher actions.
Student perception refreshes roll out the frozen student with
`algo.student_perception_rollout_noise_scale=1.0` in the same formula. Stage 2 keeps
its normal FlashSAC sampling (scale 1.0). The teacher's
predicted std comes from the checkpoint, not directly from `temp_target_sigma`.

The default `algo.cycle_mode=true` uses measured EMA latent error to alternate perception
and SAC, including the first SAC phase:

1. For `algo.perception_iters` **new** iterations (default 500), the frozen teacher rolls out
   with the configured exploration noise, including before any replay exists. Depth
   CNN/GRU, object estimator and adaptation GRU train from short contiguous rollouts using
   object MSE and privileged-latent MSE against the frozen teacher encoder. `actor_adapt`
   trains with action MSE using detached **predicted** latents from EMA perception and the
   teacher's deterministic action target. Teacher actor/encoder, critic/target critic and
   temperature stay fixed. No long-term replay is allocated in this stage.
2. Immediately after teacher perception, the depth/object/adaptation models and their EMA
   copies freeze and the first SAC phase begins. Environments and hidden
   states reset at the boundary. The distilled `actor_adapt` becomes the SAC actor; teacher
   critic/target critic, temperature and reward-normalization statistics are retained.
   Fresh RL optimizers/schedules start; the warm-start actor fills an empty replay until
   `buffer_min_length=2600000`, then SAC updates begin. No uniform random replay collection is used.
   Just before SAC first becomes eligible to update, recent student replay-filling rollouts
   establish the initial error reference. The current partial iteration is included if replay
   becomes ready mid-iteration. This measurement adds no perception training or rollout delay.
3. SAC and student perception refreshes alternate according to the measured error.
   After a minimum SAC duration, sustained error growth triggers another student refresh.
   The maximum SAC duration forces a refresh even without error growth. Each refresh releases
   replay and resets environments; subsequent SAC phases retain their optimizers. Adaptive
   cycles continue until the nominal training budget is exhausted, followed by
   the optional final SAC completion rule below.
   During refresh, the student actor is frozen and only perception trains. A refresh ends on
   a plateau or its iteration cap, then replaces the reference with its recent observed EMA error.
   Both the initial reference and refresh references are empirical errors, not minimum training loss.

Adaptive settings (iteration counts include replay filling):

| Setting | Default | Meaning |
| --- | ---: | --- |
| `perception_trigger` | `0.03` | Refresh when smoothed RMSE reaches `reference * 1.03` |
| `perception_error_window` | `20` | Recent rollout iterations used for RMSE smoothing and the reference |
| `perception_trigger_patience` | `3` | Consecutive eligible SAC iterations above the threshold |
| `cycle_sac_iters_min` / `cycle_sac_iters_max` | `100` / `1000` | SAC cooldown and forced-refresh limit |
| `plateau_window` | `60` | Compare the previous 60 and latest 60 refresh iterations |
| `plateau_tol` | `0.0025` | Relative RMSE improvement below 0.25% counts as no improvement |
| `plateau_check_interval` / `plateau_patience` | `10` / `3` | Check every 10 iterations; stop after 3 consecutive plateau checks |
| `cycle_perception_iters_min` / `cycle_perception_iters_max` | `100` / `300` | Minimum refresh duration and safety cap |
| `final_sac_iters` | `200` | Minimum uninterrupted SAC phase length at the end; `0` disables it; ignored when `cycle_mode=false` |

`final_sac_iters` takes effect only when the nominal `total_frames` budget is reached,
not that many iterations before the end. If already in SAC, the current phase counts
toward the minimum: 80 completed iterations with `final_sac_iters=200` need another
120, while 203 need none. If perception is still running, it finishes using the existing
plateau/patience rules or its existing iteration cap, then SAC runs for the requested
minimum. Initial teacher perception also finishes normally if the budget is unusually
short. During this finishing period no new refresh starts, even if the error trigger or
SAC cap fires on the exact budget boundary. Perception monitoring continues, and a SAC
phase already in progress retains its replay and optimizers.

This option can exceed `total_frames` by the remaining perception phase plus the missing
SAC iterations. Counts include replay filling (about 20 iterations with the default
buffer/environment settings), not only gradient updates. W&B `env_frames` records actual
frames, including this extension. Learning rates retain their original clock and clamp
to `learning_rate_end` once their schedule ends. Set `algo.final_sac_iters=0` to retain
strict nominal-budget termination in adaptive mode; fixed mode always retains it.

The error is `sqrt(sum of per-state latent MSE / valid state count)`, comparing rollout
EMA predictions against the frozen teacher encoder, excluding `step_count <= 1`. Windows
weight iterations by their valid state counts. This is a pre-update rollout measurement,
separate from online `adapt/priv_loss`. Invalid measurements reset contiguous windows and
patience. The initial reference may use fewer iterations when replay fills quickly; invalid
measurements keep it pending and are retried without delaying SAC. Error triggers require a
reference; the SAC duration cap still applies while it is pending. A cap can use a shorter
valid refresh reference window; a refresh with no valid reference
raises an error. Worsening RMSE also counts as no improvement for plateau detection.
The 3% trigger and plateau defaults are pilot candidates from checkpoint probes and offline
traces, not a guarantee of reward/success retention or a completed adaptive-training comparison.
Smoothing, patience and duration limits still require validation during full training.

`algo.sac_iters`, `algo.num_cycles`, `algo.cycle_perception_iters` and
`algo.cycle_sac_iters` are ignored in adaptive mode. To retain the fixed schedule, set
`algo.cycle_mode=false`: teacher perception is followed by `sac_iters=1000`, then
`num_cycles=10` repeats of `cycle_perception_iters=300` and `cycle_sac_iters=500`, followed
by SAC to the end. In fixed mode, `num_cycles=0` retains one-shot perception training.
Adaptive learning rates decay with global environment-step progress after teacher perception,
including refresh time; fixed mode retains its estimated SAC-update schedule.

On this skateboard task the actor input is 525 dimensions (`vel_command` 20 + full `policy`
249 + frozen latent 256). The replay observation is **1856** dimensions: existing teacher
fields (full command 356, full policy 249, selected priv 928, object 22, action DR 2, reference
joints 23) plus velocity command 20 and frozen latent 256. The critic still selects its
original **1187** dimensions. Replay stores neither depth nor recurrent hidden states;
next/terminal latents are computed with frozen perception before episode reset. Ten million
float16 observation rows plus transition metadata take about **35.6 GiB** of host RAM.
The default `algo.buffer_max_length=10000000` retains all 1856 observation dimensions;
only sampled batches are copied to the GPU for SAC updates. The RAM availability
check remains enabled. Terminate failed or suspended simulator runs before starting
another run, because they retain their scene memory even without a replay buffer.

Actions preserve the checkpoint's convention: by default
`ref_joint_pos_ + residual_scale * tanh(z)`. Thus this student still requires reference joint
positions for the final action even though they are outside the 525-dimensional actor input.
Mean-action MSE does not supervise the stochastic std head; SAC trains it in Stage 2.

W&B: `finetune/stage`, `finetune/perception_iters_completed`, `finetune/iters_completed`,
`finetune/cycles_completed`, `finetune/rl_phase_iters_completed`,
`adapt/priv_loss`, `adapt/object_loss`, `adapt/adapt_loss`, `adapt/teacher_student_action_rmse`,
latent/depth feature norms, and the existing FlashSAC actor/critic/temperature metrics.
Adaptive mode also logs `finetune/perception_ema_rmse`, `finetune/perception_ema_rmse_smoothed`,
`finetune/reference_error`, `finetune/relative_error`,
`finetune/plateau_improvement`, trigger/plateau streaks and `finetune/transition_reason`.
`finetune/initial_reference_pending` indicates whether the first student reference is still being collected.
`finetune/final_sac_active` indicates that the nominal budget has been reached and the
finishing rule is active; `finetune/final_sac_iters_remaining` reports the SAC iterations
still required (the full requirement while perception is running), and
`finetune/extra_iters` counts iterations beyond the nominal budget.
`finetune/student_perception_phase` distinguishes teacher perception from student refreshes;
`finetune/measurement_stage` and `finetune/measurement_student_rollout` identify the rollout
that produced the error, including at phase boundaries.
Evaluation uses the student path (EMA perception + actor_adapt), including Stage 1 checkpoints.

Teacher perception labels and actions share one privileged-encoder forward. During
multi-epoch perception updates, EMA weights stay fixed until the end, so recurrent EMA
latents are computed once in minibatch-sized chunks and reused across epochs. The
initial hidden states and reset masks remain the same as the original sequence forward;
warmup BC still updates its actor on every minibatch. Refresh actor diagnostics can also
be cached because that actor is frozen. Caches are rebuilt every iteration and are not
checkpointed. At the default two epochs/eight minibatches, EMA sequence forwards fall
from 16 to 8; refresh monitoring actor forwards also fall from 16 to 8. Adaptive error
measurements, phase decisions, SAC update counts and terminal bootstrapping are unchanged.

Version 9 finetune checkpoints preserve the phase, counters, depth models, EMA, optimizer
states, adaptive reference, initial-reference pending status, measurement windows and patience counters. The original adaptive
learning-rate clock and horizon are preserved on resume, even when the invocation frame budget
changes. Schedule parameters come from the current config; changes are reported. Changing
adaptive parameters clears the windows/patience but retains the reference. Versions 6, 7
and 8 checkpoints remain supported. A fixed SAC checkpoint switching to adaptive mode
collects its first reference while refilling replay; existing perception phases finish normally.
An older version 7 checkpoint saved during its initial student refresh retains that phase on resume.
Stage 2 resumes with frozen perception and
refills replay (replay and simulator hidden states are not checkpointed).
`total_frames` is normally the nominal frame budget for the current invocation, including
any remaining perception training. A checkpoint already in the final completion period
retains that original budget boundary and resumes its unfinished tail, rather than
starting a new full budget. The current config's `final_sac_iters` applies on resume;
a completed tail needs no more iterations unless that requirement is increased.
Disabling the option or switching to fixed mode restores ordinary invocation-budget behavior.
For a short adaptive smoke test set `algo.perception_iters=2
algo.cycle_perception_iters_min=2 algo.cycle_perception_iters_max=4
algo.cycle_sac_iters_min=2 algo.cycle_sac_iters_max=4 algo.perception_error_window=2
algo.plateau_window=2 algo.plateau_check_interval=1 algo.final_sac_iters=0` and smaller replay/batch/env counts.
For the equivalent fixed-schedule test, set `algo.cycle_mode=false algo.perception_iters=2
algo.sac_iters=2 algo.num_cycles=1 algo.cycle_perception_iters=2 algo.cycle_sac_iters=2`.
The self-contained CoSIL, TGRL and GADDPG variants below retain their fixed schedules;
these new adaptive settings apply to `flashsac_vel_finetune`.

Joint perception and SAC (`algo=joint_flashsac_vel_finetune`)

```bash
python scripts/train.py \
  algo=joint_flashsac_vel_finetune \
  task=G1/vaic/skateboard_general_tracking_stu \
  checkpoint_path=/path/to/flashsac_vel_teacher.pt \
  algo.buffer_max_length=8000000
```

This variant starts student rollout and perception training immediately
(`perception_iters=0`). SAC updates begin once `buffer_min_length` transitions are
ready. Every `train_every` steps, perception trains on the latest depth/GRU rollout
with the existing privileged-latent and object supervision losses, then updates
its EMA. Actor and critic updates use the existing FlashSAC losses. The teacher
remains fixed; the joint actor receives no additional behavior-cloning update.
SAC gradients do not flow through stored latents into perception.

Teacher latent and action labels share one privileged-encoder forward. During joint
training, action-error monitoring reuses the actual rollout EMA latent instead of
running EMA depth/GRUs again for each perception minibatch. Thus
`adapt/teacher_student_action_rmse` measures the current actor on collected perception
inputs; the perception/SAC training losses are unchanged. Only that latent is added
to the short rollout buffer, without duplicating its command/policy tensors.
Current replay observations are packed only for the first insertion; subsequent
rows already have the preceding next observation. Next-state EMA inference remains
necessary for terminal/truncation bootstrapping and changing perception weights.

There are no cycle, trigger, plateau or final-SAC-tail options. Perception stays
trainable throughout SAC, with no phase resets of replay or recurrent states.
Collected latents and hidden states are used as-is, without recomputing them after
perception updates. Depth sequences live only in the short perception buffer;
the long SAC replay keeps the same flat observations and collected latents as
`flashsac_vel_finetune`.

The other defaults match `flashsac_vel_finetune`, including the 10-million-row CPU
replay. The example uses 8 million rows (about 28.45 GiB for this task, versus
35.56 GiB at 10 million) to leave more host RAM available. An optional initial
teacher-only warmup can be enabled with `algo.perception_iters=500`.

W&B logs `finetune/joint_active`, `finetune/joint_iters_completed`,
`finetune/joint_perception_iters_completed`, the `adapt/*` losses, and existing
SAC metrics. `finetune/stage=2` means joint training here. Joint checkpoints save
both perception and SAC optimizer states and resume with empty replay; their
method is distinct from alternating finetune checkpoints. Start from a
`flashsac_vel` teacher checkpoint or resume a joint checkpoint.

CoSIL variant (`algo=cosil_flashsac_vel_finetune`) shares the configuration and phase scheduler
of `flashsac_vel_finetune`. With `algo.cycle_mode=true` (default), initial teacher perception
training goes directly into student SAC; later refreshes use the same RMSE trigger,
SAC min/max lengths, plateau checks and perception min/max lengths. `final_sac_iters`
works only in adaptive mode, including finishing a current refresh before the final SAC
tail. `algo.cycle_mode=false` uses `perception_iters`, `sac_iters`, `num_cycles`,
`cycle_perception_iters` and `cycle_sac_iters`. The `finetune/*` schedule logs are shared too.

It follows
CoSIL (Nguyen et al., CoRL 2022) and regularizes the student SAC actor toward the frozen teacher
with `D(s,h) = sum_j (tanh m_teacher,j(s) - tanh m_student,j(h))^2` and an adaptive `beta`:
the critic target subtracts `beta*D(s',h')`, the actor loss adds `beta*D(s,h)`, and
`beta` minimizes `beta*(target_divergence - E[D])`. `algo.cosil_mode=cosil` replaces FlashSAC's
entropy term with this penalty (paper); `cosil_entropy` (default) keeps the entropy term and adds it.
Defaults: `algo.target_divergence=0.8`, `algo.beta_init=0.01`. These losses run during
SAC phases; perception refreshes retain beta and its optimizer/scheduler state. Checkpoints
save both the shared schedule and CoSIL state.

`algo.cosil_divergence_input=student` uses the replayed perception latent for
the student side of the divergence. `algo.cosil_divergence_input=teacher` (default) uses the
frozen teacher encoder's latent instead, keeping the student's velocity command and policy
observations. This applies to the actor penalty, beta update and next-state critic penalty.
Rollouts and SAC Q/entropy calculations continue to use student perception. The extra
teacher-latent actor branch keeps BatchNorm statistics fixed while allowing actor gradients;
the teacher encoder remains frozen and is evaluated once for the current/next minibatch.
Teacher mode therefore measures actor disagreement given an ideal latent, rather than the
disagreement caused by the deployed perception model. `cosil/divergence_uses_teacher_latent`
logs the selection (1 for teacher, 0 for student). Checkpoints record it; an explicit change
on resume uses the current config and prints a warning.

Old CoSIL CLI names are now `perception_iters` (formerly `perception_warmup_iters`),
`sac_iters` (`rl_phase_iters`), `cycle_perception_iters` (`cycle_warmup_iters`), and
`cycle_sac_iters` (`cycle_rl_phase_iters`). Existing fixed-cycle CoSIL checkpoints are
translated on load; use `algo.cycle_mode=false` to retain their fixed schedule. Repeated
refreshes now always use the frozen student. For an old `cycle_warmup_mode=original` run,
resume a SAC-phase checkpoint rather than a checkpoint inside a repeated teacher phase.

TGRL variant (`algo=tgrl_flashsac_vel_finetune`, self-contained copy of this flow). It follows
TGRL (Shenfeld et al., ICML 2023). SAC phases train two students on one replay: the exported
`pi` maximizes `Q_R + c*(Q_I - cost(s,h))`, and the auxiliary `pi_R` maximizes `Q_R` only. Half the
envs act with each. `pi` and `pi_R` use the FlashSAC student actor architecture. `Q_I` is a second FlashSAC
categorical critic, with the same class, support and EMA target as `Q_R`. It learns the negative
future imitation return using a one-step target `gamma*(1-done_1)*(Q_I(s[t+1],a')-cost(s[t+1],h[t+1]))`.
This includes every future imitation cost while `Q_R` retains FlashSAC's n-step reward target.
The default cost is `algo.tgrl_cost=kl`, the pre-tanh Gaussian `KL(student || teacher)` used by the
linked TGRL implementation; `mse` remains an optional mean-action distance. The cost is clipped at
`algo.tgrl_cost_clip` and scaled by `normalized_G_max*(1-gamma)/tgrl_cost_clip` so it fits that support. Every
`algo.teacher_coef_update_iters` iterations, `c` moves by `+-algo.teacher_coef_update`, depending on
whether `pi`'s mean finished-episode return beats `pi_R`'s. It is clipped to `[0, teacher_coef_max]`.
Both actors train BatchNorm on concatenated current and n-step next observations, using only the
current half for the actor loss, as in `flashsac_vel_finetune`.

CoSIL + teacher intervention (`algo=gaddpg_cosil_flashsac_vel_finetune`, self-contained copy of the
CoSIL variant). The CoSIL losses are unchanged, including the default `algo.cosil_mode=cosil_entropy`.
`algo.cosil_mode=entropy` drops the CoSIL term and beta. Its SAC update is FlashSAC's own
`_update_networks`, bit-identical to `flashsac_vel_finetune`, so teacher intervention is the only
difference from that baseline. What changes is the SAC rollout:
- A fraction `algo.intervention_env_fraction` (default 0.5) of the envs is mixed, spread evenly over
  the env index. In a mixed env, the frozen teacher takes control when the student's mean action
  diverges, `D(s,h) > tau`. It then keeps control for `algo.intervention_hold_steps` (10) steps.
  Replay stores the executed action, which is the teacher's `tanh` mean during a takeover.
- `tau` follows a target takeover rate over the mixed envs. Once per SAC iteration it updates
  `log tau += algo.intervention_tau_lr*(rate - target)`. The target decays linearly from
  `algo.intervention_rate_init` (0.6) to `algo.intervention_rate_end` (0) over
  `algo.intervention_decay_fraction` (0.5) of all SAC iterations.
- The n-step window of a replay row stops before the first later step whose action was the
  teacher's. It bootstraps there with the student and a `gamma^k` discount, so the critic never
  credits student actions with the teacher's corrections.
- During SAC, `train/stats/*` and the env reward EMAs are computed from the pure-student envs
  only. The intervention is logged under `gaddpg/*`. GA-DDPG (Wang et al., CoRL 2021) mixes expert
  data into replay. This variant instead uses the teacher online, as an intervening controller.

Joint perception + CoSIL + teacher intervention
(`algo=dagger_cosil_joint_flashsac_vel_finetune`):

```bash
python scripts/train.py \
  algo=dagger_cosil_joint_flashsac_vel_finetune \
  task=G1/vaic/skateboard_general_tracking_stu \
  checkpoint_path=/path/to/teacher/checkpoint_3000.pt
```

- Starts joint training immediately (`perception_iters=0`). There are no public cycle,
  `sac_iters`, plateau, or final-SAC controls. An optional `perception_iters>0` gives a single
  teacher warmup before joint training. Perception trains while replay fills; SAC starts when
  the replay is ready. Thereafter both continue, without phase resets or replay clearing.
- Owns its config, rollout, teacher intervention, perception updates, CoSIL losses,
  intervention-aware replay and checkpoint handling in `dagger_cosil_joint_flashsac_vel_finetune.py`.
  It inherits only the generic `FlashSACVel` implementation; it does not import or inherit
  the GADDPG or plain-joint finetune implementations. Losses and defaults remain unchanged:
  `cosil_mode=cosil_entropy`,
  `target_divergence=0.8`, `beta_init=0.01`, and the intervention settings listed above.
  The actor receives CoSIL regularization, with no extra behavior-cloning optimizer steps
  during joint training. This is a DAgger-style variant, not vanilla DAgger.
  [DAgger](https://proceedings.mlr.press/v15/ross11a.html) aggregates expert labels on
  learner-visited states for supervised policy learning; here online teacher labels and
  aggregated replay are combined with CoSIL/SAC and divergence-triggered interventions.
  This is not a reproduction of the goal-auxiliary DDPG method in
  [GA-DDPG](https://sites.google.com/view/gaddpg/home).
- Online perception learns privileged-feature and object labels on the latest short depth/GRU
  rollout, including both mixed and pure-student envs. EMA perception keeps updating for
  rollout actions. Long SAC replay stores historical latents, **not depth images**; those
  latents and carried hidden states are not recomputed when perception changes, as in
  `joint_flashsac_vel_finetune`. SAC gradients do not update perception through replay.
- `train/*` episode metrics come from pure-student envs during joint training; evaluation
  never intervenes. Monitor `adapt/*`, `cosil/*`, `dagger/takeover_rate`, `dagger/target_rate`,
  `dagger/tau`, and `finetune/joint_perception_iters_completed`. Intervention counters include
  the initial replay-fill iterations, matching GADDPG's convention.
- Reuses sampled student means and teacher outputs within each rollout step. Teacher latent
  labels and takeover actions share a single encoder/policy forward. Action-error monitoring
  uses the stored rollout EMA latent rather than rerunning EMA depth/GRUs on each minibatch;
  `adapt/teacher_student_action_rmse` therefore measures the current actor on actual rollout
  perception inputs. Only the latent is cached in the short rollout; command/policy tensors
  are not duplicated. Critic-only CoSIL updates evaluate the teacher on next states only.
  Perception/SAC losses, takeover rules and hyperparameters are unchanged.
- When perception has multiple epochs, cache the actor's elementwise action errors once
  per rollout, in minibatch-sized chunks. The actor and collected inputs stay fixed during
  those epochs; the same errors can be reduced with each original shuffled minibatch/mask.
  At the default two epochs/eight minibatches, monitoring actor forwards fall from 16 to 8.
  This temporary cache is rebuilt after every rollout/SAC update and is not used for BC.
  With replay ready, the defaults still perform 256 critic, 128 actor, 128 beta and 128
  entropy-temperature updates per training iteration (`32 * 8` SAC updates, actor period 2),
  plus 16 perception optimizer steps (`2 * 8`). Teacher and EMA receive no gradient steps,
  and the separate warmup BC optimizer is inactive during joint training.
- Joint logs, intervention flags and checkpoint keys use `dagger` names. Older joint
  checkpoints with `gaddpg_*` intervention keys still load. The separate `gaddpg` module and
  its original algorithm registration remain available for existing cycle-based runs.
- Accepts a FlashSAC teacher checkpoint or its own joint checkpoint. Resume restores both
  optimizers, perception/EMA, CoSIL beta, takeover threshold and decay progress; replay and
  recurrent rollout state start fresh. Other finetune variants' checkpoints are rejected.
  Buffer size is unchanged from the existing variants; use `algo.buffer_max_length=8000000`
  if the default 10-million-row CPU replay exceeds available RAM.

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
