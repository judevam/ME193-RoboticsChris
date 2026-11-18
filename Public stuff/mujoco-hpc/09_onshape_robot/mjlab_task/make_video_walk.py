"""Record a video of a trained Walk-Straight policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G \
      --with 09_onshape_robot/mjlab_task/walk_task.py --with 09_onshape_robot/onshape \
      --with runs/<your run>/logs/rsl_rl/walk_straight/<date folder>/model_999.pt \
      09_onshape_robot/mjlab_task/make_video_walk.py model_999.pt

runIt.py brings the .mp4 home to ./runs/ and opens it. The video shows the green ideal line, black and white squares
beside it, and live numbers (distance forward, distance off the line). It also prints where the robot is each second.
"""
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("WALK_TRACK", "1")   # draw the green line and the squares (must be set before walk_task is imported)

import torch

import walk_task  # noqa: F401  (registers "Walk-Straight")
try:
  import walk_real_task  # noqa: F401  (registers "Walk-Straight-Real" if the file was sent with --with)
except ImportError:
  pass
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

TASK = sys.argv[2] if len(sys.argv) == 3 else "Walk-Straight"   # or Walk-Straight-Real
STEPS = 500          # about 10 seconds of walking

if len(sys.argv) not in (2, 3):
  sys.exit("usage: make_video_walk.py model_999.pt [Walk-Straight-Real]")
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

robot = env.unwrapped.scene["robot"]
trace = []   # (distance forward, distance off the line) after every step, for the text on the video
print("  second   forward (cm)   off the line (cm)   heading (deg)   speed (m/s)")
obs = env.get_observations()
with torch.no_grad():
  for step in range(STEPS):
    obs = env.step(policy(obs))[0]
    x, y = robot.data.root_link_pos_w[0, :2].tolist()
    if step == 0:
      x0, y0 = x, y
    trace.append(((y - y0) * 100, (x - x0) * 100))
    if step % 50 == 49:   # once a second
      w, qx, qy, qz = robot.data.root_link_quat_w[0].tolist()
      yaw = torch.atan2(torch.tensor(2 * (w * qz + qx * qy)), torch.tensor(1 - 2 * (qy * qy + qz * qz))).item()
      vy = robot.data.root_link_lin_vel_w[0, 1].item()
      print(f"  {(step + 1) / 50:5.0f}  {trace[-1][0]:12.1f}  {trace[-1][1]:17.1f}  {yaw * 57.296:13.1f}  {vy:12.2f}")
env.close()

# write the live numbers onto the video (if anything goes wrong the plain video is kept)
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
    forward, off = trace[min(i * len(trace) // len(frames), len(trace) - 1)]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for k, text in enumerate((f"distance forward: {forward:+5.0f} cm", f"off the line:      {off:+5.0f} cm")):
      draw.text((10, 8 + 22 * k), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    out.append(np.asarray(img))
  fps = 50 if len(frames) >= STEPS else max(1, round(len(frames) / (STEPS * 0.02)))
  iio.imwrite(video, np.stack(out), fps=fps)
  print(f"wrote {video} with the distance shown on it")
except Exception as e:
  print(f"wrote the video without the distance text ({type(e).__name__}: {e})")
