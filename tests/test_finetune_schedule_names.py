"""Schedule config overrides and checkpoint schedule warnings."""
import contextlib
import io
import unittest
from types import SimpleNamespace

from omegaconf import OmegaConf

from active_adaptation.learning.flashsac.flashsac_vel_finetune import (
    FlashSACVelFinetune, FlashSACVelFinetuneConfig,
)
from active_adaptation.learning.flashsac.gaddpg_cosil_flashsac_vel_finetune import (
    GADDPGCoSILFlashSACVelFinetune, GADDPGCoSILFlashSACVelFinetuneConfig,
)
from active_adaptation.learning.flashsac.tgrl_flashsac_vel_finetune import (
    TGRLFlashSACVelFinetune, TGRLFlashSACVelFinetuneConfig,
)


class ScheduleNamesTest(unittest.TestCase):
    def test_config_overrides_cycles_and_checkpoint_warnings(self):
        for policy, config in (
            (FlashSACVelFinetune, FlashSACVelFinetuneConfig),
            (GADDPGCoSILFlashSACVelFinetune, GADDPGCoSILFlashSACVelFinetuneConfig),
            (TGRLFlashSACVelFinetune, TGRLFlashSACVelFinetuneConfig),
        ):
            with self.subTest(policy=policy.__name__):
                values = dict(perception_iters=2, sac_iters=3,
                              cycle_perception_iters=4, cycle_sac_iters=5)
                cfg = OmegaConf.merge(OmegaConf.structured(config), values)
                if policy is FlashSACVelFinetune:
                    cfg.cycle_mode = False
                p = SimpleNamespace(cfg=cfg, cycles_completed=0)
                self.assertEqual(policy._perception_length(p), 2)
                self.assertEqual(policy._rl_phase_length(p), 3)
                p.cycles_completed = 1
                self.assertEqual(policy._perception_length(p), 4)
                self.assertEqual(policy._rl_phase_length(p), 5)
                for key, value in values.items():
                    state = {key: value + 1}
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        policy._warn_schedule_change(p, state)
                    self.assertIn(f'{key} {value + 1} -> {value}', output.getvalue())
                    self.assertEqual(state, {key: value + 1})
                    state[key] = value
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        policy._warn_schedule_change(p, state)
                    self.assertEqual(output.getvalue(), '')



if __name__ == '__main__':
    unittest.main()
