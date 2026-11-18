"""Train the Paddle-Pass task. Same as mjlab's own `train`, but it registers our task first.

    python train_paddle.py Paddle-Pass --env.scene.num-envs 4096
"""
import sys

import paddle_task  # noqa: F401  (registers "Paddle-Pass")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
