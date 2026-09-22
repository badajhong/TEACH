"""PPO teacher on the flat FlashSAC observation.

The networks, their sizes and the PPO hyperparameters are ppo_vel's; the input is the one
`flashsac_vel_flat_train` builds -- a single flat vector, shared by the actor and the critic,
assembled from the same observation groups with the same `obs_keys` / `obs_drop_terms` /
`obs_future_steps` options. The adaptation modules of ppo_vel (student actor, GRU adapt
module, object and depth encoders) have no place here and are left out: one actor, one
critic, one input.

Unlike ppo_vel_train and flashsac_vel_flat_train the actor emits the joint command directly
(`action = loc`), as ppo_vel_finetune does, instead of a residual on `ref_joint_pos_`;
`ref_joint_pos_` is still the last block of the observation.

This module is deliberately self-contained: nothing is imported from ppo_vel.py or
flashsac_vel_flat.py, so it can be changed without touching either of them.
"""
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.utils._pytree as pytree
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictModuleBase
from tensordict.nn import TensorDictSequential as Seq
from termcolor import colored
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor

from ..modules.distributions import IndependentNormal
from ..utils.valuenorm import ValueNorm1, ValueNormFake
from .common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    TERM_KEY,
    Actor,
    GAE,
    make_batch,
    make_mlp,
)

OBJECT_KEY = "object_"
OBJECT_GEO_KEY = "object_geo_"
OBJECT_TRANS_KEY = "object_trans"
HEIGHT_KEY = "height"
VEL_CMD_KEY = "vel_command"
REF_JPOS_KEY = "ref_joint_pos_"
# privileged: this episode's JointPosition delay (physics substeps) and low-pass filter alpha
ACTION_DR_KEY = "action_dr"
FLAT_OBS_KEY = "_flat_obs"


@dataclass
class PPOVelFlatConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_vel_flat.PPOVelFlat"
    name: str = "ppo_vel_flat"

    # --- observation, as in flashsac_vel_flat_train --------------------------------------
    # obs groups the env builds
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBJECT_KEY, OBS_PRIV_KEY, OBJECT_GEO_KEY, HEIGHT_KEY, VEL_CMD_KEY)
    # ... and the ones concatenated into the shared actor/critic input. Groups the task does not
    # define are skipped; ref_joint_pos_ is always appended. Add OBJECT_TRANS_KEY for the object
    # point cloud in the robot frame; ACTION_DR_KEY is the per-episode action delay and filter alpha.
    obs_keys: List[str] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY, OBJECT_KEY, HEIGHT_KEY, ACTION_DR_KEY)
    # "group/term" entries (term names as in the task's observation config) left out of the input;
    # by default the noisy copies in `policy` of values that `priv` holds without noise
    obs_drop_terms: List[str] = (
        "policy/root_ang_vel_history",
        "policy/projected_gravity_history",
        "policy/joint_pos_history",
    )
    # every future-reference term (term name contains "future") keeps only these of the task's
    # command.future_steps [1, 2, 8, 16, 32]; null keeps every step
    obs_future_steps: Optional[List[int]] = (8, 32)
    # null applies obs_future_steps to all future terms; a list of "group/term" limits it to those
    obs_future_terms: Optional[List[str]] = None

    # --- networks, sizes and PPO hyperparameters, as in ppo_vel ---------------------------
    actor_hidden_dims: List[int] = (512, 256, 256)
    critic_hidden_dims: List[int] = (512, 256, 128)
    layer_norm: Union[str, None] = "before"

    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8
    clip_param: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.95

    lr: float = 3e-4
    desired_kl: Optional[float] = 0.01  # adaptive learning rate; null keeps lr fixed
    max_grad_norm: float = 1.0

    entropy_coef_start: float = 0.001
    entropy_coef_end: float = 0.001
    entropy_decay_iters: int = 1000

    init_noise_scale: float = 1.0
    load_noise_scale: Optional[float] = 0.5

    normalize_ratio: bool = False
    normalize_before_sum: bool = False
    clip_neg_reward: bool = False
    value_norm: bool = False

    # PPO reads raw-scale observations, so it needs the running normalizer ppo_vel_train uses
    # (FlashSAC does without it because its embedder starts with a BatchNorm)
    vecnorm: Union[str, None] = "train"
    checkpoint_path: Union[str, None] = None


cs = ConfigStore.instance()
cs.store("ppo_vel_flat_train", node=PPOVelFlatConfig, group="algo")


class TransformObject(TensorDictModuleBase):
    """Object point cloud rotated and translated into the robot frame."""

    def __init__(self, in_keys, out_keys):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.points_dim = 128
        self.transform_dim = 12

    def forward(self, tensordict: TensorDictBase):
        geo_key = OBJECT_GEO_KEY
        vec_key = [k for k in self.in_keys if k != geo_key][0]

        object_geo_ = tensordict[geo_key].view(*tensordict[geo_key].shape[:-1], -1, 3)
        objects_num = object_geo_.shape[-2] // self.points_dim
        object_vec = tensordict[vec_key][..., -objects_num * self.transform_dim:]
        points_trans = []
        for i in range(objects_num):
            pos = object_vec[..., i * self.transform_dim:i * self.transform_dim + 3]
            ori = object_vec[..., i * self.transform_dim + 3:i * self.transform_dim + 12]
            ori = ori.view(*object_vec.shape[:-1], 3, 3)
            object_points = object_geo_[..., i * self.points_dim:(i + 1) * self.points_dim, :]
            points_rot = torch.matmul(object_points, ori.transpose(-1, -2))
            points_trans.append(points_rot + pos.unsqueeze(-2))
        points_trans = torch.cat(points_trans, dim=-2)

        tensordict[self.out_keys[0]] = points_trans.flatten(-2, -1)
        return tensordict


class FlatObs(TensorDictModuleBase):
    """Concatenates the selected observation groups into the shared actor/critic vector.

    `action_dr` is not an env observation: it is read from the action manager when the
    tensordict does not already carry it (during a rollout it is stored, so a replayed
    minibatch uses the delay and alpha of the episode it was collected in).
    """

    def __init__(self, env, obs_keys: List[str], env_obs_keys: List[str], index: torch.Tensor):
        super().__init__()
        self.in_keys = list(env_obs_keys)
        self.out_keys = [FLAT_OBS_KEY]
        self.obs_keys = list(obs_keys)
        if ACTION_DR_KEY in self.obs_keys:
            self.out_keys.append(ACTION_DR_KEY)
        object.__setattr__(self, "_env", env)
        self.object_transform = TransformObject([OBJECT_KEY, OBJECT_GEO_KEY], [OBJECT_TRANS_KEY])
        self.register_buffer("index", index, persistent=False)

    def forward(self, tensordict: TensorDictBase):
        if ACTION_DR_KEY in self.obs_keys and tensordict.get(ACTION_DR_KEY, None) is None:
            action_manager = self._env.action_manager
            tensordict.set(
                ACTION_DR_KEY,
                torch.cat([action_manager.delay.float(), action_manager.alpha.float()], dim=-1),
            )
        if OBJECT_TRANS_KEY in self.obs_keys:
            self.object_transform(tensordict)
        batch_size = tensordict.batch_size
        obs = torch.cat([tensordict[k].reshape(*batch_size, -1).float() for k in self.obs_keys], dim=-1)
        tensordict.set(FLAT_OBS_KEY, obs.index_select(-1, self.index))
        return tensordict


class PPOVelFlatRollout(TensorDictModuleBase):
    """Runs the actor and keeps only what the training loop stores."""

    def __init__(self, policy: "PPOVelFlat"):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.in_keys = list(policy.flat_obs.in_keys)
        self.out_keys = [ACTION_KEY, "sample_log_prob"] + list(policy.dist_keys)
        if ACTION_DR_KEY in policy.flat_obs.obs_keys:
            self.out_keys.append(ACTION_DR_KEY)

    def forward(self, tensordict: TensorDictBase):
        tensordict = self.policy.actor(tensordict)
        # the flat vector is rebuilt from the stored groups during the update, so it is not kept
        return tensordict.exclude(FLAT_OBS_KEY, "_actor_feature", inplace=True)


class PPOVelFlat(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOVelFlatConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
        env,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.observation_spec = observation_spec
        object.__setattr__(self, "env", env)

        self.entropy_coef = cfg.entropy_coef_start
        self.desired_kl = cfg.desired_kl
        self.clip_param = cfg.clip_param
        self.lr_policy = cfg.lr
        self.num_updates = 0

        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(gamma=cfg.gamma, lmbda=cfg.lmbda)
        self.reward_groups = list(env.cfg.reward.keys())
        num_reward_groups = len(self.reward_groups)
        self.reward_scales = torch.ones(num_reward_groups, device=self.device)
        self.reward_scales /= self.reward_scales.sum()
        value_norm_cls = ValueNorm1 if cfg.value_norm else ValueNormFake
        self.value_norm = value_norm_cls(input_shape=num_reward_groups).to(self.device)

        self.action_dim = action_spec.shape[-1]
        self.num_envs = observation_spec.shape[0]
        self.joint_names = env.action_manager.joint_names

        # --- shared actor/critic input -----------------------------------------------------
        spec_keys = set(observation_spec.keys())
        cmd_key = "command_" if "command_" in spec_keys else CMD_KEY
        self.obs_keys = []
        for key in cfg.obs_keys:
            key = cmd_key if key == CMD_KEY else key
            if key == OBJECT_TRANS_KEY:
                present = {OBJECT_KEY, OBJECT_GEO_KEY} <= spec_keys
            elif key == ACTION_DR_KEY:
                present = all(hasattr(env.action_manager, a) for a in ("delay", "alpha"))
            else:
                present = key in spec_keys
            if present:
                self.obs_keys.append(key)
            else:
                print(colored(f"[PPOFlat] obs group '{key}' is not defined by this task, skipped.", "yellow"))
        assert REF_JPOS_KEY in spec_keys, f"{REF_JPOS_KEY} is required"
        self.obs_keys.append(REF_JPOS_KEY)
        env_obs_keys = []
        for key in self.obs_keys:
            if key == OBJECT_TRANS_KEY:
                env_obs_keys += [OBJECT_KEY, OBJECT_GEO_KEY]
            elif key != ACTION_DR_KEY:
                env_obs_keys.append(key)

        rename = lambda name: cmd_key + name[len(CMD_KEY):] if name.startswith(f"{CMD_KEY}/") else name
        future_terms = None if cfg.obs_future_terms is None else {rename(n) for n in cfg.obs_future_terms}
        index, self.obs_layout, num_future = self._select_terms(
            env, observation_spec, {rename(n) for n in cfg.obs_drop_terms}, future_terms
        )
        self.obs_dim = len(index)
        print(colored(f"[PPOFlat] actor/critic observation dim {self.obs_dim}:", "green"))
        for line in self.obs_layout:
            print(colored(f"    {line}", "green"))
        if num_future:
            print(colored(f"    {num_future} future terms keep future steps {list(cfg.obs_future_steps)}", "green"))

        # `flat_obs` and `_actor_head` live inside `actor` and `critic`; holding them here as
        # plain attributes keeps them out of `named_children()`, so the checkpoint has one copy
        flat_obs = FlatObs(env, self.obs_keys, env_obs_keys, torch.tensor(index, device=self.device))
        object.__setattr__(self, "flat_obs", flat_obs)

        # --- actor and critic --------------------------------------------------------------
        self.dist_cls = IndependentNormal
        self.dist_keys = IndependentNormal.dist_keys

        actor_head = Actor(
            self.action_dim,
            init_noise_scale=cfg.init_noise_scale,
            load_noise_scale=cfg.load_noise_scale,
        )
        object.__setattr__(self, "_actor_head", actor_head)
        actor_module = Seq(
            flat_obs,
            Mod(make_mlp(list(cfg.actor_hidden_dims), norm=cfg.layer_norm), [FLAT_OBS_KEY], ["_actor_feature"]),
            Mod(actor_head, ["_actor_feature"], self.dist_keys),
        )
        self.actor = ProbabilisticActor(
            module=actor_module,
            in_keys=self.dist_keys,
            out_keys=[ACTION_KEY],
            distribution_class=self.dist_cls,
            return_log_prob=True,
        ).to(self.device)

        critic_mlp = nn.Sequential(
            make_mlp(list(cfg.critic_hidden_dims), norm=cfg.layer_norm),
            nn.LazyLinear(num_reward_groups),
        )
        self.critic = Seq(
            flat_obs,
            Mod(critic_mlp, [FLAT_OBS_KEY], ["state_value"]),
        ).to(self.device)

        # materialize the lazy layers, then initialize them as ppo_vel does
        fake_input = observation_spec.zero()
        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            if ACTION_DR_KEY in self.obs_keys:
                fake_input[ACTION_DR_KEY] = torch.zeros(fake_input.shape[0], 2)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.apply(init_)

        self.opt_policy = torch.optim.Adam(self.actor.parameters(), lr=self.lr_policy)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def _select_terms(self, env, observation_spec, drop_terms, future_terms):
        """Indices of the kept entries in the concatenation of `self.obs_keys`, a readable layout,
        and the number of future terms reduced to `obs_future_steps`."""
        future_steps = self.cfg.obs_future_steps
        if future_terms is None:
            is_future = lambda name: "future" in name.split("/", 1)[1]
        else:
            is_future = lambda name: name in future_terms
        index, layout, offset, matched, num_future = [], [], 0, set(), 0
        for group in self.obs_keys:
            if group == OBJECT_TRANS_KEY:
                terms = [(OBJECT_TRANS_KEY, math.prod(observation_spec[OBJECT_GEO_KEY].shape[1:]))]
            elif group == ACTION_DR_KEY:
                terms = [("delay", 1), ("alpha", 1)]
            else:
                terms = [
                    (t, f().reshape(self.num_envs, -1).shape[-1])
                    for t, f in env.observation_funcs[group].funcs.items()
                ]
                assert sum(d for _, d in terms) == math.prod(observation_spec[group].shape[1:]), group
            group_dim, notes = sum(d for _, d in terms), []
            for term, dim in terms:
                name = f"{group}/{term}"
                keep = range(dim)
                if name in drop_terms:
                    keep = range(0)
                    notes.append(f"-{term} ({dim})")
                elif future_steps and is_future(name):
                    all_steps = env.command_manager.future_steps.tolist()
                    missing = set(future_steps) - set(all_steps)
                    if missing or dim % len(all_steps):
                        raise ValueError(
                            f"cannot keep future steps {list(future_steps)} of {name} (task has {all_steps})"
                        )
                    per_step = dim // len(all_steps)  # future terms are laid out step-major
                    keep = [all_steps.index(s) * per_step + i for s in future_steps for i in range(per_step)]
                    notes.append(f"{term} {len(keep)}/{dim}")
                    num_future += 1
                matched.add(name)
                index.extend(offset + i for i in keep)
                offset += dim
            kept = sum(1 for i in index if i >= offset - group_dim)
            layout.append(f"{group} {kept}/{group_dim}" + (f": {', '.join(notes)}" if notes else ""))
        named = drop_terms | (future_terms or set())
        unknown = sorted(n for n in named if n.split("/")[0] in self.obs_keys and n not in matched)
        if unknown:
            raise ValueError(f"unknown observation terms in obs_drop_terms/obs_future_terms: {unknown}")
        return index, layout, num_future

    def get_rollout_policy(self, mode: str = "train"):
        return PPOVelFlatRollout(self)

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")
        info = self.train_policy(tensordict)
        self.num_updates += 1

        action_std = self._actor_head.actor_std.detach()
        for joint_name, std in zip(self.joint_names, action_std):
            info[f"actor_std/{joint_name}"] = std
        info["actor_std/mean"] = action_std.mean()
        return info

    def train_policy(self, tensordict: TensorDict):
        infos = []
        self._compute_advantage(tensordict)

        # entropy coef schedule
        entropy_progress = float(np.clip(self.env.current_iter / self.cfg.entropy_decay_iters, 0.0, 1.0))
        self.entropy_coef = self.cfg.entropy_coef_start + (
            self.cfg.entropy_coef_end - self.cfg.entropy_coef_start
        ) * entropy_progress

        for _ in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self._update_ppo(minibatch))

                if self.desired_kl is not None:  # adaptive learning rate
                    kl = infos[-1]["actor/kl"]
                    if kl > self.desired_kl * 2.0:
                        self.lr_policy = max(1e-5, self.lr_policy / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        self.lr_policy = min(1e-2, self.lr_policy * 1.5)
                for param_group in self.opt_policy.param_groups:
                    param_group["lr"] = self.lr_policy

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.lr_policy
        infos["actor/entropy_coef"] = self.entropy_coef

        ret = tensordict["ret"]
        ret_mean, ret_std = ret.mean(dim=(0, 1)), ret.std(dim=(0, 1))
        for i, group_name in enumerate(self.reward_groups):
            infos[f"critic/{group_name}.ret_mean"] = ret_mean[i].item()
            infos[f"critic/{group_name}.ret_std"] = ret_std[i].item()
            infos[f"critic/{group_name}.neg_rew_ratio"] = (tensordict[REWARD_KEY][:, :, i] <= 0.0).float().mean().item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(self, tensordict: TensorDict, adv_key: str = "adv", ret_key: str = "ret"):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                self.critic(tensordict_flat)
                self.critic(tensordict_flat["next"])

        values = self.value_norm.denormalize(tensordict["state_value"])
        next_values = self.value_norm.denormalize(tensordict["next", "state_value"])

        rewards = tensordict[REWARD_KEY]
        if self.cfg.clip_neg_reward:
            rewards = rewards.clamp_min(0.0)

        adv, ret = self.gae(
            rewards, tensordict[TERM_KEY], tensordict[DONE_KEY], values, next_values,
            tensordict["next", "discount"],
        )

        # [num_envs, num_steps, num_reward_groups]
        if self.cfg.normalize_before_sum:  # normalize, scale, sum
            adv_norm = (adv - adv.mean(dim=(0, 1))) / (adv.std(dim=(0, 1)) + 0.01)
            adv_final = (adv_norm * self.reward_scales).sum(dim=2, keepdim=True)
        else:  # scale, sum, normalize
            adv_sum = (adv * self.reward_scales).sum(dim=2, keepdim=True)
            adv_final = (adv_sum - adv_sum.mean(dim=(0, 1))) / (adv_sum.std(dim=(0, 1)) + 1e-8)

        self.value_norm.update(ret)
        tensordict.set(adv_key, adv_final)
        tensordict.set(ret_key, self.value_norm.normalize(ret))
        return tensordict

    def _update_ppo(self, tensordict: TensorDict):
        dist_kwargs_old = tensordict.select(*self.dist_keys)

        dist: D.Independent = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[ACTION_KEY])
        entropy = dist.entropy().mean()

        # the first steps of an episode carry Isaac's placeholder observations
        valid = (tensordict["step_count"] > 1).squeeze(-1)

        adv = tensordict["adv"]
        log_ratio = (log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        if self.cfg.normalize_ratio:
            clamped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param).detach()
            surr1, surr2 = surr1 / clamped_ratio, surr2 / clamped_ratio
        policy_loss = -(torch.min(surr1, surr2)[valid]).mean()
        entropy_loss = -self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)[valid].mean(dim=0)

        loss = policy_loss + entropy_loss + value_loss.mean()

        self.opt_policy.zero_grad()
        self.opt_critic.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt_policy.step()
        self.opt_critic.step()

        with torch.no_grad():
            explained_var = 1 - value_loss / b_returns[valid].var(dim=0)
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            kl = D.kl_divergence(self.dist_cls(**dist_kwargs_old), dist).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_std": tensordict["scale"].detach().mean(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
            "actor/approx_kl": ((ratio - 1) - log_ratio).mean(),
            "critic/grad_norm": critic_grad_norm,
        }
        for i, group_name in enumerate(self.reward_groups):
            info[f"critic/{group_name}.explained_var"] = explained_var[i]
            info[f"critic/{group_name}.value_loss"] = value_loss[i].detach()
        return info

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        state_dict["last_iter"] = self.env.current_iter
        state_dict["lr_policy"] = self.lr_policy
        state_dict["obs_layout"] = self.obs_layout
        # the layout counts entries, so it cannot tell [1, 32] from [8, 32]; the index can
        state_dict["obs_index"] = self.flat_obs.index.cpu()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        saved_index = state_dict.get("obs_index")
        if saved_index is not None and not torch.equal(saved_index, self.flat_obs.index.cpu()):
            raise ValueError(
                f"checkpoint input is {state_dict.get('obs_layout')} but the current obs options give "
                f"{self.obs_layout}. Set algo.obs_keys, obs_drop_terms, obs_future_steps and "
                "obs_future_terms as in the run's cfg.yaml."
            )
        succeed_keys, failed_keys = [], []
        for name, module in self.named_children():
            try:
                module.load_state_dict(state_dict.get(name, {}), strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")

        self.env.set_progress(state_dict.get("last_iter", 0))
        lr_policy = state_dict.get("lr_policy", None)
        if lr_policy is not None:
            self.lr_policy = lr_policy
            for param_group in self.opt_policy.param_groups:
                param_group["lr"] = self.lr_policy
        return failed_keys
