"""Record a video of a trained Paddle-Pass policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G \
      --with mjlab_paddle/paddle_task.py \
      --with runs/<your run>/logs/rsl_rl/paddle_pass/<date folder>/model_999.pt \
      mjlab_paddle/make_video_paddle.py model_999.pt

runIt.py brings the .mp4 home to ./runs/ and opens it. The video has a counter of passes and dropped balls, and
the job output prints the peak of every bounce and the totals.
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import paddle_task as pt  # registers "Paddle-Pass"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

TASK = "Paddle-Pass"
STEPS = 500          # 10 seconds

if len(sys.argv) != 2:
  sys.exit("usage: make_video_paddle.py model_999.pt")
checkpoint = Path(sys.argv[1])

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env = VideoRecorder(env, video_folder=".", step_trigger=lambda step: step == 0, video_length=STEPS, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

base = env.unwrapped
passes = drops = 0
apexes = []
trace = []                       # (passes, drops) after every step, for the text on the video
prev_vz = prev_x = None
prev_len = 0
obs = env.get_observations()
with torch.no_grad():
  for step in range(STEPS):
    obs = env.step(policy(obs))[0]
    q, v = pt._state(base)
    bx, bz, bvx, bvz = q[0, 2].item(), q[0, 3].item(), v[0, 2].item(), v[0, 3].item()
    length = int(base.episode_length_buf[0])
    if length < prev_len and prev_len < 499:      # the episode restarted early: the ball was lost
      drops += 1
      prev_vz = prev_x = None
    prev_len = length
    if prev_vz is not None and prev_vz > 0 and bvz <= 0:
      apexes.append(bz)
    if prev_x is not None and (prev_x > 0) != (bx > 0) and abs(bvx) > 0.3 and bz > 0.25:
      passes += 1
    prev_vz, prev_x = bvz, bx
    trace.append((passes, drops))
env.close()
print(f"in 10 s: {passes} passes across the middle, ball dropped {drops} times")
if apexes:
  print("bounce peaks (m):", ", ".join(f"{a:.2f}" for a in apexes))
  mean_error = sum(abs(a - pt.TARGET) for a in apexes) / len(apexes) * 100
  print(f"mean peak {sum(apexes) / len(apexes):.3f} m, mean error {mean_error:.1f} cm (target {pt.TARGET} m)")

# write the counters onto the video (if anything goes wrong the plain video is kept)
try:
  import imageio.v3 as iio
  import numpy as np
  from PIL import Image, ImageDraw, ImageFont

  video = max(Path(".").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
  frames = iio.imread(video, index=None)
  try:
    font = ImageFont.load_default(size=18)
  except TypeError:
    font = ImageFont.load_default()
  out = []
  for i, frame in enumerate(frames):
    p, d = trace[min(i * len(trace) // len(frames), len(trace) - 1)]
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text((10, 8), f"passes: {p}   dropped: {d}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    out.append(np.asarray(img))
  fps = 50 if len(frames) >= STEPS else max(1, round(len(frames) / (STEPS * 0.02)))
  iio.imwrite(video, np.stack(out), fps=fps)
  print(f"wrote {video} with the counters on it")
except Exception as e:
  print(f"wrote the video without the counters ({type(e).__name__}: {e})")
