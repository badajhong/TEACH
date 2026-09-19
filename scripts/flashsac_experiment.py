"""Reproducible FlashSAC transfer experiments and first-episode evaluations.

Uses the unchanged task rewards, terminations, and default actuator settings.
All optional interventions are explicit under +experiment.* and saved in cfg.yaml.
"""
import json
import os
from pathlib import Path
import random
import shutil
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from isaaclab.app import AppLauncher
from torchrl.envs.utils import ExplorationType, set_exploration_type

from helpers import make_env_policy


@torch.no_grad()
def first_episodes(env, policy, seed):
    """Same per-env first completed episode metric as helpers.evaluate, streaming."""
    env.base_env.eval()
    env.eval()
    env.set_seed(seed)
    carry = env.reset()
    initial_delay = env.action_manager.delay.squeeze(-1).clone()
    initial_alpha = env.action_manager.alpha.squeeze(-1).clone()
    seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    samples = {}
    rollout = policy.get_rollout_policy("eval")
    with set_exploration_type(ExplorationType.MODE):
        for step in range(env.max_episode_length + 1):
            torch.compiler.cudagraph_mark_step_begin()
            td, carry = env.step_and_maybe_reset(rollout(carry))
            new = td["next", "done"].squeeze(-1) & ~seen
            if new.any():
                for key, value in td["next", "stats"].items(True, True):
                    name = "/".join(key) if isinstance(key, tuple) else key
                    if name not in samples:
                        samples[name] = torch.full((env.num_envs,), float("nan"), device=env.device)
                    samples[name][new] = value.reshape(env.num_envs)[new].float()
                seen |= new
            if seen.all():
                break
    if not seen.all():
        raise RuntimeError(f"{int((~seen).sum())} environments have no completed first episode")
    result = {"seed": seed, "episodes": env.num_envs, "steps": step + 1}
    for name, value in samples.items():
        if name not in ("success", "episode_len"):
            value = value / samples["episode_len"]
        result["eval/" + name] = value.mean().item()
        result["eval/" + name + "_std"] = value.std().item()
    p = result["eval/success"]
    n = env.num_envs
    center = (p + 1.96**2 / (2*n)) / (1 + 1.96**2/n)
    radius = 1.96 * ((p*(1-p)/n + 1.96**2/(4*n*n))**0.5) / (1 + 1.96**2/n)
    result["success_wilson95"] = [center-radius, center+radius]
    result["success_by_delay"] = {
        str(int(delay)): {"episodes": int((initial_delay == delay).sum()),
                          "success": samples["success"][initial_delay == delay].mean().item()}
        for delay in initial_delay.unique()
    }
    samples["initial_delay"] = initial_delay
    samples["initial_alpha"] = initial_alpha
    return result, {k: v.cpu() for k, v in samples.items()}


@torch.no_grad()
def neutralize_constant_dr(policy, method="scale"):
    """Initialize previously constant actuator features for transfer to randomization.

    Either zero BN scales or zero input-projection columns. The latter preserves
    the BN normalization budget. Subsequent SAC updates can learn the new inputs.
    """
    offset = 0
    indices = []
    for line in policy.obs_layout:
        group, dims = line.split()[:2]
        width = int(dims.split("/")[0].rstrip(":"))
        if group == "action_dr":
            indices = list(range(offset, offset + width))
        offset += width
    if len(indices) != 2:
        raise ValueError("Expected two action_dr features")
    actuator = policy.env.action_manager
    delay_count = actuator.max_delay - actuator.min_delay + 1
    alpha_low, alpha_high = actuator.alpha_range
    means = torch.tensor([(actuator.min_delay + actuator.max_delay) / 2,
                          (alpha_low + alpha_high) / 2], device=policy.device)
    variances = torch.tensor([(delay_count**2 - 1) / 12,
                              (alpha_high - alpha_low)**2 / 12], device=policy.device)
    audit = {}
    for name, norm in [("actor", policy.actor.embedder.norm),
                       ("critic", policy.critic.state_norm),
                       ("target_critic", policy.target_critic.state_norm)]:
        if norm is None:
            raise ValueError("This experiment requires critic state encoders")
        if not (norm.running_var[..., indices] < 1e-8).all():
            raise ValueError(f"{name}: action_dr was not constant during source training")
        audit[name] = {k: getattr(norm, k)[..., indices].tolist()
                       for k in ("running_mean", "running_var", "weight", "bias")}
        if method == "scale":
            norm.weight[..., indices] = 0
        elif method == "projection":
            # Preserve the input BN scale budget during subsequent unit normalization.
            # The untrained columns, rather than the BN scales, start at zero.
            projection = (policy.actor.embedder.w.w.weight if name == "actor" else
                          getattr(policy, name).state_projection.weight)
            projection[..., indices] = 0
        else:
            raise ValueError(f"Unknown transfer method: {method}")
        norm.running_mean[..., indices] = means
        norm.running_var[..., indices] = variances
    return audit


@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    ex = cfg.get("experiment", {})
    output = Path(ex.get("output", "diagnostics/flashsac_improve_20260918/experiment")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output / "experiment_source.py")
    # Seed before construction as well as reset: assets can randomize at creation.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    app = AppLauncher(OmegaConf.to_container(cfg.app)).app
    env, policy, vecnorm = make_env_policy(cfg)
    OmegaConf.save(cfg, output / "cfg.yaml")
    results = []

    def evaluate(label):
        for seed in ex.get("eval_seeds", [cfg.seed]):
            metrics, samples = first_episodes(env, policy, seed)
            metrics.update(label=label, checkpoint=str(cfg.checkpoint_path), scene_seed=cfg.seed)
            results.append(metrics)
            (output / "results.json").write_text(json.dumps(results, indent=2))
            torch.save(samples, output / f"{label}_seed{seed}_episodes.pt")
            print("RESULT " + json.dumps(metrics), flush=True)

    def save_checkpoint(path):
        torch.save({"policy": policy.state_dict(), "env": env.state_dict(),
                    "cfg": cfg, "experiment": OmegaConf.to_container(cfg.experiment)}, path)
        print("CHECKPOINT " + str(path), flush=True)

    if ex.get("baseline", True):
        evaluate("baseline")
    if ex.get("neutralize_dr", False):
        audit = neutralize_constant_dr(policy, ex.get("transfer_method", "scale"))
        (output / "dr_audit.json").write_text(json.dumps(audit, indent=2))
        save_checkpoint(output / "checkpoint_neutralized.pt")
        if ex.get("evaluate_transfer", True):
            evaluate("neutralized_dr")

    steps = ex.get("train_steps", 0)
    if steps:
        # New replay and optimizer states: explicitly a fine-tune, not exact resume.
        for name, network in policy._networks().items():
            if network.optimizer is not None:
                rate = ex.get(name + "_lr", ex.get("lr", 3e-5))
                network.optimizer.state.clear()
                for group in network.optimizer.param_groups:
                    group["lr"] = rate
                    group["initial_lr"] = rate
                network.scheduler = torch.optim.lr_scheduler.LambdaLR(network.optimizer, lambda _: 1.)
        env.train()
        env.base_env.train()
        env.set_seed(cfg.seed)
        carry = env.reset()
        rollout = policy.get_rollout_policy("train")
        deterministic = policy.get_rollout_policy("eval")
        start_iter = env.current_iter
        start = time.perf_counter()
        finished = successes = length_sum = 0.
        # Filling a new replay from a pretrained policy must not use uniform actions.
        for step in range(steps):
            env.set_progress(start_iter + step // cfg.algo.train_every)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                acting = rollout if policy.can_start_training() else deterministic
                td, carry = env.step_and_maybe_reset(acting(carry))
                policy.add_transition(td)
                done = td["next", "done"].squeeze(-1)
                finished += done.sum().item()
                successes += td["next", "stats", "success"][done].sum().item()
                length_sum += td["next", "stats", "episode_len"][done].sum().item()
            policy.update()
            if (step + 1) % 32 == 0:
                info = policy.pop_info()
                info.update(step=step+1, seconds=time.perf_counter()-start)
                if finished:
                    info.update(train_success=successes/finished, train_episode_len=length_sum/finished)
                finished = successes = length_sum = 0.
                print("TRAIN " + json.dumps(info), flush=True)
            if (step + 1) % ex.get("save_steps", steps) == 0 or step + 1 == steps:
                checkpoint = output / f"checkpoint_{step+1}.pt"
                save_checkpoint(checkpoint)
        cfg.checkpoint_path = str(checkpoint)
        evaluate("finetuned")
    for checkpoint_path in ex.get("eval_checkpoints", []):
        state = torch.load(checkpoint_path, weights_only=False, map_location=env.device)
        policy.load_state_dict(state["policy"])
        cfg.checkpoint_path = str(checkpoint_path)
        evaluate(Path(checkpoint_path).stem)
    print("COMPLETE", flush=True)
    # Isaac's shutdown can hang after successful evaluation; files are closed above.
    os._exit(0)


if __name__ == "__main__":
    main()
