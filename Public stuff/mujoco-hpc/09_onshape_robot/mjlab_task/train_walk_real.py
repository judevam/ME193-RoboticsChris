"""Train the Walk-Straight-Real task (the version that can run on the real robot). Like train_walk.py.

    python train_walk_real.py Walk-Straight-Real --env.scene.num-envs 4096
"""
import sys

import walk_real_task  # noqa: F401  (registers "Walk-Straight-Real"; walk_task.py must be next to it)
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
