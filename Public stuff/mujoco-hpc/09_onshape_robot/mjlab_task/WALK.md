# Walk-Straight: teach the two-motor robot to walk straight with mjlab

`walk_task.py` turns the Onshape two-motor robot (`../onshape/robot.xml`) into an mjlab task called **Walk-Straight**.
4096 copies of the robot start with their legs at random angles, pointing a little off-axis, and learn to move along +y
while staying on the line x = 0. The policy outputs one speed per leg. There is no hand-written steering: it is all
learned with reinforcement learning (PPO, via mjlab).

| File | What it is |
|---|---|
| `walk_task.py` | The task: robot, motors, rewards, resets, domain randomization. **Tune this one.** |
| `train_walk.py` | mjlab's own train command, with our task registered first |
| `make_video_walk.py` | Records a video of a trained policy on the cluster: green ideal line, black and white squares, distance shown on screen |

Run everything from the `mujoco-hpc` folder in WSL (venv active, VPN on if off campus).

## 1. Check it loads (short run, about 10 minutes of waiting at most)

```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 --with 09_onshape_robot/mjlab_task/walk_task.py --with 09_onshape_robot/onshape 09_onshape_robot/mjlab_task/train_walk.py Walk-Straight --env.scene.num-envs 64 --agent.max-iterations 20
```

## 2. Train

```
python3 runIt.py --gpu --time 00:30:00 --mem 16G --cpus 4 --with 09_onshape_robot/mjlab_task/walk_task.py --with 09_onshape_robot/onshape 09_onshape_robot/mjlab_task/train_walk.py Walk-Straight --env.scene.num-envs 4096
```
The default is 1000 iterations. Checkpoints (`model_<n>.pt`, every 50 iterations) come back to
`runs/train_walk-<time>/logs/rsl_rl/walk_straight/<date>/`. Add `--agent.max-iterations 2000` (and `--time 01:00:00`) to train longer.

## 3. Watch the result

Use your own folder names (type `runs/train_walk-` and press Tab) and the last checkpoint (`model_999.pt` for 1000 iterations):
```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 --with 09_onshape_robot/mjlab_task/walk_task.py --with 09_onshape_robot/onshape --with runs/train_walk-XXXXXXXX-XXXXXX/logs/rsl_rl/walk_straight/XXXX-XX-XX_XX-XX-XX/model_999.pt 09_onshape_robot/mjlab_task/make_video_walk.py model_999.pt
```
The job output prints forward distance, sideways drift and heading every second. The video opens when it is done.

## Tuning (all in `walk_task.py`)

- **Drifts sideways:** raise the `stay_on_line` and `face_forward` weights, one at a time. Too high and the robot may
  prefer to stand still (the forward reward can no longer pay for it).
- **Barely moves or is too weak:** raise `LEG_TORQUE_LIMIT` or `LEG_KD`.
- **Jerky legs:** make the `action_rate` weight more negative.
- **Faster:** `TARGET_SPEED` (0.1 m/s) is just above what the robot can do (about 0.08), so the reward simply keeps paying
  for more speed.
- **Domain randomization** (friction, mass, motor strength, shoves) is in the `events` block. Remove an entry to turn it off.

Forward is +y, and the robot's own front is its +y axis. The left motor is wired the other way round, so `walk_task.py`
gives it a negative action scale: a policy output of +1 on both legs means "forward".

## What was tested

On a laptop CPU only (not yet on the cluster's GPU): the task builds and steps, the reward and observation
shapes are right (20 inputs, 2 outputs), domain randomization runs, the track compiles, and 3 iterations of
mjlab's PPO training loop run. No policy has been trained with this task yet.

## Walk-Straight-Real: train on what the LEGO Double Motor can actually sense

`walk_real_task.py` registers **Walk-Straight-Real**. Its policy sees only the packet the robot sends every 20 ms (18 numbers),
with noise and delay: IMU yaw (sin, cos), pitch and roll, gyroscope x y z, accelerometer x y z (in g, gravity included),
both leg angles, both leg speeds, and its own last command. Body velocity, sideways offset and "which way is up" are only
given to the critic during training. The accelerometer is computed from the change in body velocity over 20 ms, plus gravity.

```
python3 runIt.py --gpu --time 01:00:00 --mem 16G --cpus 4 --with 09_onshape_robot/mjlab_task/walk_task.py --with 09_onshape_robot/mjlab_task/walk_real_task.py --with 09_onshape_robot/onshape 09_onshape_robot/mjlab_task/train_walk_real.py Walk-Straight-Real --env.scene.num-envs 4096
```
Then `python export_policy.py model_1499.pt` and run it with `run_on_robot.py` (it detects the 18-input policy).

**Before trusting it on the robot**, run `python run_on_robot.py --imu-check`. The library does not document the IMU units or
axes, so `GYRO_UNIT_DPS`, `ACCEL_UNIT_G` and the axis maps at the top of `run_on_robot.py` are assumptions. Flat and still, one
accel axis should read about 1 g; the gyro z should match the yaw direction when you turn the robot by hand.
The old 11-input `walk_policy_real.npz` still runs; it just needs no retraining. This task has not been run yet.
