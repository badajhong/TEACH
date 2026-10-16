"""Run the return/discount contract against DAgger's independent replay."""
import unittest

import test_gaddpg_replay as replay_tests
from active_adaptation.learning.flashsac.dagger_cosil_joint_flashsac_vel_finetune import (
    InterventionNStepReplay,
)


class DAggerReplayTest(replay_tests.GADDPGReplayTest):
    replay_cls = InterventionNStepReplay


if __name__ == "__main__":
    unittest.main()
