"""Test the Onshape two-motor robot: drop it on the floor, spin its legs, film it. Run it with runIt.py.

    python3 runIt.py --gpu --with 09_onshape_robot/onshape --with 09_onshape_robot/walker.py 09_onshape_robot/walker_test.py

On your own computer (the model is on the laptop, and so is a screen):
    python3 walker_test.py             # film walker.mp4 (needs MUJOCO_GL=egl or glfw, see below)
    python3 walker_test.py --view      # open the interactive viewer: drag the legs, push the robot
    python3 walker_test.py --no-video  # just print the numbers

What the plan does: the robot drops and settles, then the legs turn the same way (it should move), then in opposite
ways (it should spin on the spot), then backwards. The printout says where the robot ended up after each part,
and whether it stayed upright. Change PLAN to try other things; the motor limits are at the top of walker.py.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco
import numpy as np

import walker

# What the legs do, in order: (until this second, forward speed, turn speed), in rad/s. forward > 0 should push the
# robot toward +y; turn > 0 should turn it counter-clockwise seen from above.
PLAN = [
    (1.0, 0, 0),       # settle on the floor
    (4.0, 10, 0),      # both legs turn the same way: move
    (7.0, 0, 8),       # legs turn opposite ways: spin
    (10.0, -10, 0),    # backwards
]
OUT = "walker.mp4"
FPS = 30

model = walker.load()
data = mujoco.MjData(model)
walker.reset(model, data)
chassis = model.body("chassis").id
print("legs:", [model.actuator(i).name for i in range(model.nu)],
      f"| mass {model.body_subtreemass[chassis] * 1000:.0f} g | torque limit {walker.MAX_TORQUE} N*m, speed limit {walker.MAX_SPEED} rad/s")

if "--view" in sys.argv:
    import mujoco.viewer
    mujoco.viewer.launch(model, data)
    sys.exit()


def status():
    up = data.xmat[chassis].reshape(3, 3)[2, 2]          # 1 = upright, 0 = on its side, -1 = upside down
    yaw = np.degrees(np.arctan2(data.xmat[chassis][3], data.xmat[chassis][0]))
    return up, yaw


video = "--no-video" not in sys.argv
if video:
    renderer = mujoco.Renderer(model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING     # follow the robot
    camera.trackbodyid = chassis
    camera.distance, camera.azimuth, camera.elevation = 0.6, 90, -35   # from behind and above, so drift shows as left or right of the green line
frames = []
start_x, start_y = data.qpos[:2]    # for the distance shown on the video
for end, forward, turn in PLAN:
    start = data.qpos[:2].copy()
    walker.command(model, data, forward, turn)
    while data.time < end:
        mujoco.mj_step(model, data)
        if video and data.time >= len(frames) / FPS:
            renderer.update_scene(data, camera=camera)
            frames.append(walker.annotate(renderer.render(),
                                          f"distance forward: {(data.qpos[1] - start_y) * 100:+5.0f} cm",
                                          f"off the line:      {(data.qpos[0] - start_x) * 100:+5.0f} cm",
                                          f"time: {data.time:4.1f} s"))
    up, yaw = status()
    dx, dy = data.qpos[:2] - start
    print(f"t={data.time:4.1f}s  forward={forward:+3g} turn={turn:+3g}:  moved x={dx * 100:+6.1f} cm, y={dy * 100:+6.1f} cm,"
          f"  heading {yaw:+6.0f} deg,  upright {up:+.2f}")

if video:
    renderer.close()
    imageio.mimwrite(OUT, frames, fps=FPS)
    print(f"wrote {OUT}: {len(frames)} frames")
