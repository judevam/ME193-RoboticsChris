"""Train two tilted paddles to pass a ball back and forth, over and over. Reinforcement learning (PPO), then a video.

    python 09_paddleDemo.py                  # train (about 5-10 min on a laptop CPU), then render paddle.mp4
    python 09_paddleDemo.py --iters 50       # a quick, rough try
    python 09_paddleDemo.py --eval           # skip training, replay the saved policy (paddle_policy.pt)

On the cluster:  python3 runIt.py --gpu scripts/09_paddleDemo.py     (the GPU node is for the video, EGL needs it)

The world: two paddles, one on the left (x = -0.4 m) and one on the right (x = +0.4 m). Each is a flat plate tilted by
TILT_DEG degrees so that it leans toward the middle: a ball that lands on the left paddle is bounced up and to the right,
and one that lands on the right paddle is bounced up and to the left. Each paddle can slide up and down (starts at
0.5 m). The ball moves in the x-z plane (side to side and up and down) and starts above the left paddle at 1.0 m.
The policy decides where each paddle should be, 50 times a second.
Goal: keep the ball going back and forth between the paddles, peaking at TARGET = 1.0 m, with little paddle motion.

A note on angles: MuJoCo reads the angles you write in the XML in DEGREES (the <compiler angle="degree"/> line below
says so, and degrees is also the default), so TILT_DEG is simply put in the XML as "euler". Inside the simulation,
joint angles and velocities (qpos, qvel) are in RADIANS. This demo has only sliding joints, so it uses metres.

Files written next to the script: paddle_policy.pt (the trained network), train_log.csv (reward terms per
iteration, for plotting), paddle.mp4 (the video, with a counter of passes).
"""
import argparse
import csv
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- the task
TILT_DEG = 20          # degrees: how far each paddle leans toward the middle (0 would be flat)
SPACING = 0.4          # m: each paddle is this far from the middle, left at -SPACING and right at +SPACING
TARGET = 1.0           # m: the height every bounce should peak at
PADDLE_START = 0.5     # m
BALL_START = 1.0       # m: the ball starts this high, above the left paddle
PADDLE_LO, PADDLE_HI = 0.2, 0.8   # m: how far each paddle can slide
OUT_OF_BOUNDS = 0.9    # m: the ball has been lost if it goes further than this to either side
G = 9.81
SUBSTEPS = 10          # physics steps (2 ms each) per decision, so the policy acts at 50 Hz
EPISODE_STEPS = 500    # 10 s per episode

# ---------------------------------------------------------------- the rewards (edit these and retrain!)
# Each is multiplied by its weight and added up every step. Positive = reward, negative = penalty.
W = dict(
    cross=3.0,          # + each time the ball crosses the middle (x = 0) moving sideways, in the air: a pass
    apex=3.0,           # + at the top of each bounce, exp(-(peak - 1.0)^2 / 0.07^2)  (peak at the target height)
    energy=0.5,         # - |ball's up-and-down energy - what a 1.0 m peak needs| / that energy
    alive=0.2,          # + for every step the ball is still in play
    off_court=2.0,      # - how far the ball is beyond 0.55 m from the middle
    paddle_speed=0.005,  # - paddle speed^2, both paddles          (smooth, gentle paddles)
    action_change=0.05,  # - (change in the paddle commands)^2     (minimal paddle movement)
    fall=5.0,           # - once, if the ball hits the floor or leaves the court (the episode ends)
)

# The world, in MuJoCo's XML. The paddles slide up and down; the ball slides in x and z. Everything stays in one plane.
XML = f"""
<mujoco>
  <compiler angle="degree"/>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 -1 4"/>
    <camera name="front" pos="0 -3 0.8" xyaxes="1 0 0 0 0 1"/>
    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>
    <site name="target" type="box" size="{OUT_OF_BOUNDS} .004 .004" pos="0 0 {TARGET}" rgba="1 0 0 1"/>

    <!-- left paddle: leans toward +x, so a falling ball is bounced to the right -->
    <body name="paddle_left" pos="{-SPACING} 0 0" gravcomp="1">
      <joint name="left_z" type="slide" axis="0 0 1" range="{PADDLE_LO} {PADDLE_HI}"/>
      <geom type="box" size=".2 .25 .02" euler="0 {TILT_DEG} 0" mass="1" rgba=".9 .6 .1 1"
            solref="-30000 -5" friction="0.05 0.005 0.0001"/>
    </body>
    <!-- right paddle: leans toward -x -->
    <body name="paddle_right" pos="{SPACING} 0 0" gravcomp="1">
      <joint name="right_z" type="slide" axis="0 0 1" range="{PADDLE_LO} {PADDLE_HI}"/>
      <geom type="box" size=".2 .25 .02" euler="0 {-TILT_DEG} 0" mass="1" rgba=".2 .7 .9 1"
            solref="-30000 -5" friction="0.05 0.005 0.0001"/>
    </body>

    <body name="ball" pos="0 0 0">
      <joint name="ball_x" type="slide" axis="1 0 0"/>
      <joint name="ball_z" type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".05" mass=".1" rgba="1 .3 .2 1" solref="-30000 -5" friction="0.05 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <!-- a spring-damper that pulls each paddle to the commanded height (critically damped for a 1 kg paddle).
         (the joints sit at the origin, so their position is the real height) -->
    <position name="left" joint="left_z" kp="1000" kv="63" ctrlrange="{PADDLE_LO} {PADDLE_HI}" forcerange="-300 300"/>
    <position name="right" joint="right_z" kp="1000" kv="63" ctrlrange="{PADDLE_LO} {PADDLE_HI}" forcerange="-300 300"/>
  </actuator>
</mujoco>
"""

OBS_DIM, ACT_DIM = 10, 2


class PaddleEnv:
    """N copies of the world. step() takes two actions per copy (left, right paddle) and returns (obs, reward, info)."""

    def __init__(self, n, seed=0):
        self.n = n
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = [mujoco.MjData(self.model) for _ in range(n)]
        self.rng = np.random.default_rng(seed)
        self.t = np.zeros(n, int)
        self.last_a = np.zeros((n, ACT_DIM))
        self.prev_bvz = np.zeros(n)
        self.prev_bx = np.zeros(n)
        for i in range(n):
            self.reset_one(i)

    def reset_one(self, i):
        d = self.data[i]
        mujoco.mj_resetData(self.model, d)
        # qpos is [left paddle height, right paddle height, ball x, ball z]
        d.qpos[:] = [PADDLE_START, PADDLE_START, -SPACING + self.rng.uniform(-0.03, 0.03), BALL_START + self.rng.uniform(-0.02, 0.02)]
        d.ctrl[:] = PADDLE_START
        mujoco.mj_forward(self.model, d)
        self.t[i] = 0
        self.last_a[i] = 0.0
        self.prev_bvz[i] = 0.0
        self.prev_bx[i] = d.qpos[2]

    def state(self):
        s = np.array([[d.qpos[0], d.qvel[0], d.qpos[1], d.qvel[1], d.qpos[2], d.qpos[3], d.qvel[2], d.qvel[3]] for d in self.data])
        return s.T   # lz, lv, rz, rv, bx, bz, bvx, bvz

    def obs(self):
        lz, lv, rz, rv, bx, bz, bvx, bvz = self.state()
        return np.stack([
            bx / 0.5, (bz - 0.8) / 0.5, bvx / 2, bvz / 3,
            (lz - 0.5) / 0.3, lv / 2, (rz - 0.5) / 0.3, rv / 2,
            self.last_a[:, 0], self.last_a[:, 1],
        ], 1).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1, 1)
        target = PADDLE_LO + (a + 1) / 2 * (PADDLE_HI - PADDLE_LO)   # action -1..1 -> paddle height 0.2..0.8 m
        for i, d in enumerate(self.data):
            d.ctrl[:] = target[i]
            mujoco.mj_step(self.model, d, nstep=SUBSTEPS)
        lz, lv, rz, rv, bx, bz, bvx, bvz = self.state()

        # --- rewards
        energy = G * bz + 0.5 * bvz ** 2           # the ball's up-and-down energy per kg
        energy_goal = G * TARGET
        topped = (self.prev_bvz > 0) & (bvz <= 0)  # the ball just reached the top of a bounce
        crossed = (np.sign(self.prev_bx) != np.sign(bx)) & (np.abs(bvx) > 0.3) & (bz > 0.25)   # a real pass, in the air
        fell = (bz < 0.07) | (np.abs(bx) > OUT_OF_BOUNDS)
        terms = dict(
            cross=crossed.astype(float),
            apex=np.where(topped, np.exp(-((bz - TARGET) / 0.07) ** 2), 0.0),
            energy=-np.abs(energy - energy_goal) / energy_goal,
            alive=np.ones(self.n),
            off_court=-np.maximum(np.abs(bx) - 0.55, 0.0),
            paddle_speed=-(lv ** 2 + rv ** 2),
            action_change=-((a - self.last_a) ** 2).sum(1),
            fall=-fell.astype(float),
        )
        reward = sum(W[k] * v for k, v in terms.items())

        self.last_a = a
        self.prev_bvz = bvz
        self.prev_bx = bx
        self.t += 1
        return self.obs(), reward, dict(terms=terms, apex=np.where(topped, bz, np.nan), fell=fell, crossed=crossed)

    def time_up(self):
        return self.t >= EPISODE_STEPS


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        mlp = lambda out: nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))
        self.actor, self.critic = mlp(ACT_DIM), mlp(1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x):
        return torch.distributions.Normal(self.actor(x), self.log_std.exp())


def train(args):
    torch.manual_seed(args.seed)
    env = PaddleEnv(args.envs, args.seed)
    env.t = env.rng.integers(0, EPISODE_STEPS, args.envs)   # stagger the episodes so each batch mixes all stages
    agent = Agent()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
    T, gamma, lam = 64, 0.98, 0.95
    obs = torch.tensor(env.obs())
    log = open("train_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["iter", "mean_step_reward", "apex_error_m", "passes_per_10_s", "falls_per_10_s"] + list(W))
    print(f"{'iter':>5} {'reward/step':>12} {'apex err (m)':>13} {'passes/10 s':>12} {'falls/10 s':>11}   (peak of each bounce vs {TARGET} m)")

    for it in range(1, args.iters + 1):
        O, A, LP, R, V, D = [], [], [], [], [], []
        sums = {k: 0.0 for k in W}
        apexes, crosses, falls = [], 0, 0
        for _ in range(T):
            with torch.no_grad():
                dist = agent.dist(obs)
                act = dist.sample()
                lp, val = dist.log_prob(act).sum(-1), agent.value(obs)
            nobs, rew, info = env.step(act.numpy())
            for k in W:
                sums[k] += float(np.mean(info["terms"][k]))
            apexes += list(info["apex"][~np.isnan(info["apex"])])
            crosses += int(info["crossed"].sum())
            falls += int(info["fell"].sum())
            timeup, done = env.time_up() & ~info["fell"], env.time_up() | info["fell"]
            nobs_t = torch.tensor(nobs)
            if done.any():   # the time limit is not a failure: add the value of where we stopped
                with torch.no_grad():
                    rew = rew + gamma * agent.value(nobs_t).numpy() * timeup
                for i in np.flatnonzero(done):
                    env.reset_one(i)
                nobs_t = torch.tensor(env.obs())
            O.append(obs); A.append(act); LP.append(lp); V.append(val)
            R.append(torch.tensor(rew, dtype=torch.float32)); D.append(torch.tensor(done, dtype=torch.float32))
            obs = nobs_t

        # advantages (GAE): how much better each action was than the critic expected
        with torch.no_grad():
            last = agent.value(obs)
        adv = torch.zeros(T, args.envs)
        g = torch.zeros(args.envs)
        for t in reversed(range(T)):
            nxt = last if t == T - 1 else V[t + 1]
            delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]
            g = delta + gamma * lam * (1 - D[t]) * g
            adv[t] = g
        ret = adv + torch.stack(V)
        O, A, LP = torch.cat(O), torch.cat(A), torch.cat(LP)
        adv, ret = adv.flatten(), ret.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(4):   # PPO: a few passes over the batch, with the update clipped
            for idx in torch.randperm(len(O)).chunk(4):
                dist = agent.dist(O[idx])
                ratio = (dist.log_prob(A[idx]).sum(-1) - LP[idx]).exp()
                pg = -torch.min(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx]).mean()
                vf = ((agent.value(O[idx]) - ret[idx]) ** 2).mean()
                loss = pg + 0.5 * vf - 0.003 * dist.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()

        mean_r = sum(W[k] * sums[k] for k in W) / T
        err = float(np.mean(np.abs(np.array(apexes) - TARGET))) if apexes else float("nan")
        per_10s = 10.0 / (T * args.envs * 0.02)       # turns a count over this batch into a count per 10 simulated seconds
        writer.writerow([it, f"{mean_r:.4f}", f"{err:.4f}", f"{crosses * per_10s:.2f}", f"{falls * per_10s:.2f}"] + [f"{W[k] * sums[k] / T:.4f}" for k in W])
        log.flush()
        if it % 10 == 0 or it == 1:
            print(f"{it:5d} {mean_r:12.3f} {err:13.3f} {crosses * per_10s:12.2f} {falls * per_10s:11.2f}", flush=True)
    log.close()
    torch.save(agent.state_dict(), "paddle_policy.pt")
    print("saved paddle_policy.pt and train_log.csv")
    return agent


def evaluate(agent, seconds=10.0, video="paddle.mp4"):
    """Run the trained policy (no exploration noise), report how it did, write a video."""
    env = PaddleEnv(1, seed=123)
    try:
        renderer = mujoco.Renderer(env.model, 480, 640)
    except Exception as e:   # no EGL (for example a CPU node): still report numbers
        print(f"no video ({type(e).__name__}: {e}); try a GPU node")
        renderer = None
    frames, apexes, passes, falls, travel = [], [], 0, 0, 0.0
    prev_lz, prev_rz = PADDLE_START, PADDLE_START
    for _ in range(int(seconds / (0.002 * SUBSTEPS))):
        with torch.no_grad():
            act = agent.actor(torch.tensor(env.obs())).numpy()   # the mean action
        _, _, info = env.step(act)
        d = env.data[0]
        travel += abs(d.qpos[0] - prev_lz) + abs(d.qpos[1] - prev_rz)
        prev_lz, prev_rz = d.qpos[0], d.qpos[1]
        passes += int(info["crossed"][0])
        if not np.isnan(info["apex"][0]):
            apexes.append(float(info["apex"][0]))
        if renderer:
            renderer.update_scene(d, camera="front")
            frame = renderer.render()
            try:   # write the pass counter on the video
                from PIL import Image, ImageDraw
                img = Image.fromarray(frame)
                ImageDraw.Draw(img).text((10, 10), f"passes: {passes}   dropped: {falls}", fill=(255, 255, 255))
                frame = np.asarray(img)
            except Exception:
                pass
            frames.append(frame)
        if info["fell"][0]:
            falls += 1
            env.reset_one(0)
            prev_lz, prev_rz = PADDLE_START, PADDLE_START
    if renderer:
        renderer.close()
        imageio.mimwrite(video, frames, fps=int(1 / (0.002 * SUBSTEPS)))
        print(f"wrote {video}: {len(frames)} frames")
    print(f"in {seconds:g} s: {passes} passes across the middle, ball dropped {falls} times")
    if apexes:
        print(f"bounce peaks (m): {', '.join(f'{a:.2f}' for a in apexes)}")
        print(f"mean peak {np.mean(apexes):.3f} m, mean error {np.mean(np.abs(np.array(apexes) - TARGET)) * 100:.1f} cm (target {TARGET} m)")
    print(f"the two paddles travelled {travel:.1f} m in total")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=600, help="training iterations (each is 4096 simulation steps)")
    ap.add_argument("--envs", type=int, default=64, help="how many worlds to simulate at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true", help="skip training; replay paddle_policy.pt")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.eval:
        agent = Agent()
        agent.load_state_dict(torch.load("paddle_policy.pt"))
    else:
        agent = train(args)
    evaluate(agent)
