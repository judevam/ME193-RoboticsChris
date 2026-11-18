# Paddle-Pass: two tilted paddles pass a ball back and forth, trained with mjlab

`paddle_task.py` is an mjlab task called **Paddle-Pass**. Two paddles, each tilted 20 degrees toward the middle, slide up
and down. A ball moves in the plane between them and starts above the left paddle. The policy picks a height for each paddle
every 20 ms. 4096 copies of the court train at once on the GPU.

| File | What it is |
|---|---|
| `paddle_task.py` | The task: the court, the rewards, the resets. **Tune this one.** |
| `train_paddle.py` | mjlab's own train command, with our task registered first |
| `make_video_paddle.py` | Records a video of a trained policy on the cluster, with a pass counter |

Rewards (all in `paddle_task.py`): `apex_height` (+1, the top of each bounce at 1.0 m), `action_rate` (-0.01, smooth
paddles), `fell` (-10, ball hit the floor or left the court; the episode ends) and `pass_bonus` (+1, the ball crossed the
middle sideways in the air; my addition so the paddles are rewarded for passing, not only for bouncing).

Angles: MuJoCo reads the angles in the XML in degrees, so `TILT_DEG = 20` goes straight into the paddles' `euler` setting.
Inside the simulation, joint angles are radians; this task has only sliding joints, so it works in metres.

Run from the `mujoco-hpc` folder in WSL (venv active, VPN on if off campus).

## 1. Check it loads (short run)

```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 --with mjlab_paddle/paddle_task.py mjlab_paddle/train_paddle.py Paddle-Pass --env.scene.num-envs 64 --agent.max-iterations 20
```

## 2. Train

```
python3 runIt.py --gpu --time 00:30:00 --mem 16G --cpus 4 --with mjlab_paddle/paddle_task.py mjlab_paddle/train_paddle.py Paddle-Pass --env.scene.num-envs 4096
```
1000 iterations. Checkpoints (every 50) come back to `runs/train_paddle-<time>/logs/rsl_rl/paddle_pass/<date>/`.

## 3. Watch the result

Use your own folder names (type `runs/train_paddle-` and press Tab) and the last checkpoint (`model_999.pt`):
```
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 --with mjlab_paddle/paddle_task.py --with runs/train_paddle-XXXXXXXX-XXXXXX/logs/rsl_rl/paddle_pass/XXXX-XX-XX_XX-XX-XX/model_999.pt mjlab_paddle/make_video_paddle.py model_999.pt
```
The job output prints the peak of every bounce, the number of passes and the number of drops. The video shows the pass counter.

## Tuning (all in `paddle_task.py`)

- **Drops the ball:** raise the `fell` weight, or move the paddles closer (`SPACING`), or tilt them less or more (`TILT_DEG`).
- **Bounces too low or too high:** change `TARGET`.
- **Jerky paddles:** make the `action_rate` weight more negative.
- **Does not pass back and forth:** raise the `pass_bonus` weight.

## What was tested

On a laptop CPU only: the task builds, the paddles start at 0.5 m and the ball bounces between them (with the paddles
held still it makes about two passes and then escapes), and 3 iterations of mjlab's PPO training run. No policy has been
trained with this task yet, and `make_video_paddle.py` has not been run (it needs the GPU renderer).
