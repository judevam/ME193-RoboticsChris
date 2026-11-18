"""Train the Walk-Straight task. Same as mjlab's own `train`, but it registers our task first.

    python train_walk.py Walk-Straight --env.scene.num-envs 4096
"""
import sys

import walk_task  # noqa: F401  (registers "Walk-Straight")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
