"""Train a paddle to juggle a ball so that the ball peaks at 1.2 m. Reinforcement learning (PPO), then a video.

    python 09_paddleDemo.py                  # train (about 5 min on a laptop CPU), then render paddle.mp4
    python 09_paddleDemo.py --iters 50       # a quick, rough try
    python 09_paddleDemo.py --eval           # skip training, replay the saved policy (paddle_policy.pt)

On the cluster:  python3 runIt.py --gpu scripts/09_paddleDemo.py     (the GPU node is for the video, EGL needs it)

The world: a paddle that slides up and down (starts at 0.5 m) and a ball that only moves up and down (starts at 1.0 m).
The ball falls on the paddle; the policy decides where the paddle should be, 50 times a second.
Goal: every bounce should peak at TARGET = 1.2 m, with as little paddle motion as possible.

Files written next to the script: paddle_policy.pt (the trained network), train_log.csv (reward terms per
iteration, for plotting), paddle.mp4 (the video).
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
TARGET = 1.2           # m: the height every bounce should reach
PADDLE_START = 0.5     # m
BALL_START = 1.0       # m
PADDLE_LO, PADDLE_HI = 0.2, 0.8   # m: how far the paddle can travel
G = 9.81
SUBSTEPS = 10          # physics steps (2 ms each) per decision, so the policy acts at 50 Hz
EPISODE_STEPS = 300    # 6 s per episode

# ---------------------------------------------------------------- the rewards (edit these and retrain!)
# Each is multiplied by its weight and added up every step. Positive = reward, negative = penalty.
W = dict(
    energy=1.0,         # - |ball energy - energy needed for 1.2 m| / that energy   (the ball's speed and height together)
    height=1.0,         # + exp(-(ball height - 1.2)^2 / 0.15^2)                    (be near 1.2 m)
    apex=5.0,           # + at the top of each bounce, exp(-(peak - 1.2)^2 / 0.05^2) (the peak is what we care about)
    above_paddle=0.2,   # + how far the ball is above the paddle (0 to 0.7 m, scaled to 0..1)
    ball_speed=0.005,   # - ball speed^2                                            (do not throw it wildly)
    paddle_speed=0.01,  # - paddle speed^2                                          (smooth, gentle paddle)
    action_change=0.1,  # - (change in the paddle command)^2                        (minimal paddle movement)
    fall=5.0,           # - once, if the ball ends up below the paddle (the episode ends)
)

# The world, in MuJoCo's XML. Both joints are vertical sliders, so everything stays on one line.
XML = f"""
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 -1 4"/>
    <camera name="side" pos="0 -3 0.8" xyaxes="1 0 0 0 0 1"/>
    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>
    <site name="target" type="box" size=".45 .004 .004" pos="0 0 {TARGET}" rgba="1 0 0 1"/>

    <body name="paddle" pos="0 0 0" gravcomp="1">
      <joint name="paddle_z" type="slide" axis="0 0 1" range="{PADDLE_LO} {PADDLE_HI}"/>
      <geom type="box" size=".25 .25 .02" mass="1" rgba=".9 .6 .1 1" solref="-30000 -5"/>
    </body>

    <body name="ball" pos="0 0 0">
      <joint name="ball_z" type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".05" mass=".1" rgba="1 .3 .2 1" solref="-30000 -5"/>
    </body>
  </worldbody>
  <actuator>
    <!-- (the slide joints sit at the origin, so their position is the real height) a spring-damper that pulls the paddle to the commanded height (critically damped for a 1 kg paddle) -->
    <position joint="paddle_z" kp="1000" kv="63" ctrlrange="{PADDLE_LO} {PADDLE_HI}" forcerange="-300 300"/>
  </actuator>
</mujoco>
"""

OBS_DIM = 6


class PaddleEnv:
    """N copies of the world. step() takes one action per copy and returns (obs, reward, info)."""

    def __init__(self, n, seed=0):
        self.n = n
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = [mujoco.MjData(self.model) for _ in range(n)]
        self.rng = np.random.default_rng(seed)
        self.t = np.zeros(n, int)
        self.last_a = np.zeros(n)
        self.prev_bv = np.zeros(n)
        for i in range(n):
            self.reset_one(i)

    def reset_one(self, i):
        d = self.data[i]
        mujoco.mj_resetData(self.model, d)
        d.qpos[0] = PADDLE_START
        d.qpos[1] = BALL_START + self.rng.uniform(-0.01, 0.01)   # a hair of noise so it is not one fixed trajectory
        d.ctrl[0] = PADDLE_START
        mujoco.mj_forward(self.model, d)
        self.t[i] = 0
        self.last_a[i] = 0.0
        self.prev_bv[i] = 0.0

    def state(self):
        return np.array([[d.qpos[0], d.qvel[0], d.qpos[1], d.qvel[1]] for d in self.data])

    def obs(self):
        pz, pv, bz, bv = self.state().T
        return np.stack([(pz - 0.5) / 0.3, pv / 2, (bz - 0.8) / 0.5, bv / 3, (bz - pz - 0.5) / 0.5, self.last_a], 1).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1, 1)
        target = PADDLE_LO + (a + 1) / 2 * (PADDLE_HI - PADDLE_LO)   # action -1..1 -> paddle height 0.2..0.8 m
        for i, d in enumerate(self.data):
            d.ctrl[0] = target[i]
            mujoco.mj_step(self.model, d, nstep=SUBSTEPS)
        pz, pv, bz, bv = self.state().T

        # --- rewards
        energy = G * bz + 0.5 * bv ** 2            # ball energy per kg
        energy_goal = G * TARGET
        topped = (self.prev_bv > 0) & (bv <= 0)    # the ball just reached the top of a bounce
        apex = np.where(topped, bz, np.nan)
        fell = bz < pz                             # the ball got past the paddle: a failure
        terms = dict(
            energy=-np.abs(energy - energy_goal) / energy_goal,
            height=np.exp(-((bz - TARGET) / 0.15) ** 2),
            apex=np.where(topped, np.exp(-((bz - TARGET) / 0.05) ** 2), 0.0),
            above_paddle=np.clip((bz - pz) / 0.7, 0, 1),
            ball_speed=-bv ** 2,
            paddle_speed=-pv ** 2,
            action_change=-(a - self.last_a) ** 2,
            fall=-fell.astype(float),
        )
        reward = sum(W[k] * v for k, v in terms.items())

        self.last_a = a
        self.prev_bv = bv
        self.t += 1
        return self.obs(), reward, dict(terms=terms, apex=apex, fell=fell)

    def time_up(self):
        return self.t >= EPISODE_STEPS


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        mlp = lambda out: nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out))
        self.actor, self.critic = mlp(1), mlp(1)
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x):
        return torch.distributions.Normal(self.actor(x).squeeze(-1), self.log_std.exp())


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
    writer.writerow(["iter", "mean_step_reward", "apex_error_m", "n_apex"] + list(W))
    print(f"{'iter':>5} {'reward/step':>12} {'apex err (m)':>13}   (the peak of each bounce, vs {TARGET} m)")

    for it in range(1, args.iters + 1):
        O, A, LP, R, V, D = [], [], [], [], [], []
        sums = {k: 0.0 for k in W}
        apexes = []
        for _ in range(T):
            with torch.no_grad():
                dist = agent.dist(obs)
                act = dist.sample()
                lp, val = dist.log_prob(act), agent.value(obs)
            nobs, rew, info = env.step(act.numpy())
            for k in W:
                sums[k] += float(np.mean(info["terms"][k]))
            apexes += list(info["apex"][~np.isnan(info["apex"])])
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
                ratio = (dist.log_prob(A[idx]) - LP[idx]).exp()
                pg = -torch.min(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx]).mean()
                vf = ((agent.value(O[idx]) - ret[idx]) ** 2).mean()
                loss = pg + 0.5 * vf - 0.003 * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()

        mean_r = sum(W[k] * sums[k] for k in W) / T
        err = float(np.mean(np.abs(np.array(apexes) - TARGET))) if apexes else float("nan")
        writer.writerow([it, f"{mean_r:.4f}", f"{err:.4f}", len(apexes)] + [f"{W[k] * sums[k] / T:.4f}" for k in W])
        if it % 10 == 0 or it == 1:
            print(f"{it:5d} {mean_r:12.3f} {err:13.3f}", flush=True)
    log.close()
    torch.save(agent.state_dict(), "paddle_policy.pt")
    print("saved paddle_policy.pt and train_log.csv")
    return agent


def evaluate(agent, seconds=8.0, video="paddle.mp4"):
    """Run the trained policy (no exploration noise), report the bounce peaks, write a video."""
    env = PaddleEnv(1)
    obs = torch.tensor(env.obs())
    try:
        renderer = mujoco.Renderer(env.model, 480, 640)
    except Exception as e:   # no EGL (for example a CPU node): still report numbers
        print(f"no video ({type(e).__name__}: {e}); try a GPU node")
        renderer = None
    frames, apexes, travel, prev_pz, falls = [], [], 0.0, PADDLE_START, 0
    for _ in range(int(seconds / (0.002 * SUBSTEPS))):
        with torch.no_grad():
            act = agent.actor(obs).squeeze(-1).numpy()   # the mean action
        nobs, _, info = env.step(act)
        d = env.data[0]
        travel += abs(d.qpos[0] - prev_pz)
        prev_pz = d.qpos[0]
        if not np.isnan(info["apex"][0]):
            apexes.append(float(info["apex"][0]))
        if renderer:
            renderer.update_scene(d, camera="side")
            frames.append(renderer.render())
        obs = torch.tensor(nobs)
        if info["fell"][0]:
            falls += 1
            env.reset_one(0)
            obs = torch.tensor(env.obs())
    if renderer:
        renderer.close()
        imageio.mimwrite(video, frames, fps=int(1 / (0.002 * SUBSTEPS)))
        print(f"wrote {video}: {len(frames)} frames")
    print(f"bounce peaks (m): {', '.join(f'{a:.2f}' for a in apexes)}")
    if apexes:
        tail = apexes[2:] or apexes   # skip the first bounces while it gets going
        print(f"after the first two bounces: mean peak {np.mean(tail):.3f} m, mean error {np.mean(np.abs(np.array(tail) - TARGET)) * 100:.1f} cm (target {TARGET} m)")
    print(f"paddle travelled {travel:.1f} m in {seconds:g} s; ball got past the paddle {falls} times")


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
