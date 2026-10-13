"""Exercise the real training loop without importing/starting Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import torch
from tensordict import TensorDict


class EmptyEpisodeStats:
    def __init__(self, in_keys, device):
        self.in_keys = in_keys

    def add(self, td):
        pass

    def __len__(self):
        return 0


class FakeEnv:
    num_envs = 2
    max_episode_length = 100
    current_iter = 7
    device = 'cpu'
    extra = {}
    stats_ema = {}

    def __init__(self):
        self.progress = []
        self.steps = 0

    def reset(self):
        return TensorDict({}, batch_size=[self.num_envs])

    def set_progress(self, i):
        self.progress.append(i)

    def step_and_maybe_reset(self, carry):
        self.steps += 1
        next_td = TensorDict({
            'done': torch.zeros(self.num_envs, 1, dtype=torch.bool),
            'truncated': torch.zeros(self.num_envs, 1, dtype=torch.bool),
            'step_count': torch.ones(self.num_envs, 1, dtype=torch.long),
            'stats': TensorDict({}, batch_size=[self.num_envs]),
        }, batch_size=[self.num_envs])
        return TensorDict({'next': next_td}, batch_size=[self.num_envs]), carry


class PlainPolicy:
    def __init__(self):
        self.updates = 0
        self.transitions = 0

    def get_rollout_policy(self, mode):
        return lambda carry: carry

    def add_transition(self, td):
        self.transitions += 1

    def update(self):
        self.updates += 1

    def pop_info(self):
        return {'policy_updates': self.updates}


class BudgetPolicy(PlainPolicy):
    def __init__(self, required_iterations, train_every):
        super().__init__()
        self.required_updates = required_iterations * train_every
        self.configured = []

    def configure_training_budget(self, total_iters):
        self.configured.append(total_iters)

    def training_complete(self):
        if not self.configured:
            raise AssertionError('budget must be configured before completion checks')
        return self.updates >= self.required_updates


class OffPolicyBudgetTest(unittest.TestCase):
    def run_loop(self, policy, nominal_iters=2, main=True):
        # Compile only the production function, bypassing train.py's simulator
        # startup imports. Its control flow, updates, logging and saves are real.
        path = Path(__file__).parents[1] / 'scripts/train.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'train_off_policy')
        progress_totals = []

        def progress(iterable, total):
            progress_totals.append(total)
            return iterable

        namespace = dict(torch=torch, time=time, EpisodeStats=EmptyEpisodeStats,
                         tqdm=progress, aa=SimpleNamespace(get_world_size=lambda: 1,
                                                          is_main_process=lambda: main))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        env = FakeEnv()
        train_every = 3
        cfg = SimpleNamespace(algo=SimpleNamespace(train_every=train_every),
                              get=lambda key, default: nominal_iters * env.num_envs * train_every)
        logs, saves = [], []
        namespace['train_off_policy'](
            cfg, env, policy, SimpleNamespace(log=logs.append),
            EmptyEpisodeStats([], 'cpu'), lambda p, name: saves.append((name, p.updates)),
            lambda i: i > 0 and i % 2 == 0,
        )
        return env, logs, saves, progress_totals

    def test_other_agents_keep_exact_nominal_budget(self):
        policy = PlainPolicy()
        env, logs, saves, totals = self.run_loop(policy)
        self.assertEqual(env.progress, [7, 8])
        self.assertEqual(policy.updates, 6)
        self.assertEqual(policy.transitions, 6)
        self.assertEqual([r['env_frames'] for r in logs], [6, 12])
        self.assertEqual(saves, [])
        self.assertEqual(totals, [2])

    def test_budget_hook_can_extend_and_records_actual_frames_and_saves(self):
        policy = BudgetPolicy(required_iterations=4, train_every=3)
        env, logs, saves, totals = self.run_loop(policy)
        self.assertEqual(policy.configured, [2])
        self.assertEqual(env.progress, [7, 8, 9, 10])
        self.assertEqual(policy.updates, 12)
        self.assertEqual([r['env_frames'] for r in logs], [6, 12, 18, 24])
        self.assertEqual(saves, [('checkpoint_2', 9)])
        self.assertEqual(totals, [2])

    def test_restored_finishing_tail_may_complete_before_new_nominal_budget(self):
        policy = BudgetPolicy(required_iterations=1, train_every=3)
        env, logs, _, _ = self.run_loop(policy, nominal_iters=20)
        self.assertEqual(policy.configured, [20])
        self.assertEqual(env.progress, [7])
        self.assertEqual(logs[-1]['env_frames'], 6)

    def test_already_complete_tail_does_not_collect_extra_rollout(self):
        policy = BudgetPolicy(required_iterations=0, train_every=3)
        env, logs, saves, _ = self.run_loop(policy)
        self.assertEqual(env.steps, 0)
        self.assertEqual((logs, saves), ([], []))

    def test_non_main_process_uses_same_extension_without_progress_or_logging(self):
        policy = BudgetPolicy(required_iterations=4, train_every=3)
        env, logs, _, totals = self.run_loop(policy, main=False)
        self.assertEqual(env.steps, 12)
        self.assertEqual((logs, totals), ([], []))


if __name__ == '__main__':
    unittest.main()
