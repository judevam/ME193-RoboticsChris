"""Teach the Onshape two-motor robot to walk (move forward, stay upright) with reinforcement learning (PPO).

    python3 runIt.py --gpu --time 01:00:00 --cpus 4 --with 09_onshape_robot/onshape --with 09_onshape_robot/walker.py 09_onshape_robot/walker_train.py
    python3 walker_train.py --iters 20      # a quick try on your own computer
    python3 walker_train.py --eval          # skip training, replay the saved policy (walker_policy.pt)

Files written next to the script: walker_policy.pt (the trained network), walker_log.csv (reward terms per iteration,
for plotting), walker_trained.mp4 (the policy running, with the numbers printed at the end).

The task: the robot starts on the floor with its legs at random angles. Every 20 ms the policy chooses how fast each
leg turns (-1 = full speed one way, +1 = full speed the other, in the direction that pushes the robot toward +y).
Goal: move toward +y as fast as possible, without tipping over, flipping, or drifting sideways, using little energy.
The motor limits are in walker.py: a policy can only be as good as the motors allow.
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

import walker

# ---------------------------------------------------------------- the task
SUBSTEPS = 10          # physics steps (2 ms each) per decision: the policy acts at 50 Hz
EPISODE_STEPS = 500    # 10 s per episode
FALL_UP = 0.3          # the robot has "fallen" when its up-axis points less than this much up (1 = upright, 0 = on its side)
SPEED_CAP = 0.3        # m/s: speed beyond this earns no extra reward

# ---------------------------------------------------------------- the rewards (edit these and retrain!)
# Each is multiplied by its weight and added up every step. Positive = reward, negative = penalty.
W = dict(
    forward=10.0,      # + speed toward +y (m/s, capped at SPEED_CAP)
    sideways=2.0,      # - |speed along x|                  (do not drift sideways)
    spin=0.2,          # - |turning rate about the vertical| (rad/s)
    heading=0.1,       # - (heading angle off +y)^2          (keep pointing the right way)
    upright=0.3,       # + how upright the body is (0 to 1)
    alive=0.0,         # + for every step it stays up (0 = off; the fall penalty already says 'stay up')
    energy=0.05,       # - motor power (watts)                (little effort)
    action_change=0.01,  # - (change in the leg commands)^2    (smooth legs)
    fall=5.0,          # - once, when it falls over (the episode ends)
)

OBS_DIM, ACT_DIM = 17, 2
FREEZE_NORM = 40       # training iterations during which the observation scaling is still adapting


class WalkerEnv:
    """N copies of the robot. step() takes the leg commands (N x 2, each -1..1) and returns (obs, reward, info)."""

    def __init__(self, n, seed=0):
        self.n = n
        self.model = walker.load()
        self.data = [mujoco.MjData(self.model) for _ in range(n)]
        self.rng = np.random.default_rng(seed)
        m = self.model
        self.chassis = m.body("chassis").id
        self.act = [m.actuator(name).id for name in (walker.LEFT, walker.RIGHT)]
        self.qadr = [m.joint(name).qposadr[0] for name in (walker.LEFT, walker.RIGHT)]
        self.vadr = [m.joint(name).dofadr[0] for name in (walker.LEFT, walker.RIGHT)]
        self.gyro = m.sensor_adr[m.sensor("gyro").id]
        self.sign = np.array([walker.SIGN[walker.LEFT], walker.SIGN[walker.RIGHT]])
        self.t = np.zeros(n, int)
        self.last_a = np.zeros((n, ACT_DIM))
        for i in range(n):
            self.reset_one(i)

    def reset_one(self, i):
        d = self.data[i]
        walker.reset(self.model, d, leg_angles=self.rng.uniform(-np.pi, np.pi, 2))
        self.t[i] = 0
        self.last_a[i] = 0.0

    def _obs_one(self, i):
        d = self.data[i]
        R = d.xmat[self.chassis].reshape(3, 3)
        angle = d.qpos[self.qadr]
        return np.concatenate([
            R[2],                                    # which way is up, seen from the robot
            d.sensordata[self.gyro:self.gyro + 3],   # turning rates
            np.sin(angle), np.cos(angle),            # the two leg angles
            d.qvel[self.vadr] / walker.MAX_SPEED,    # leg speeds
            R.T @ d.qvel[:3],                        # the robot's own velocity, in its own frame
            self.last_a[i],
        ])

    def obs(self):
        return np.array([self._obs_one(i) for i in range(self.n)], dtype=np.float32)

    def step(self, action):
        a = np.clip(action, -1, 1)
        reward, fell, terms = np.zeros(self.n), np.zeros(self.n, bool), {k: np.zeros(self.n) for k in W}
        for i, d in enumerate(self.data):
            d.ctrl[self.act] = self.sign * a[i] * walker.MAX_SPEED
            mujoco.mj_step(self.model, d, nstep=SUBSTEPS)
            R = d.xmat[self.chassis].reshape(3, 3)
            gyro = d.sensordata[self.gyro:self.gyro + 3]
            vx, vy = d.qvel[0], d.qvel[1]
            yaw = np.arctan2(R[1, 0], R[0, 0])
            up = R[2, 2]
            fell[i] = up < FALL_UP
            terms["forward"][i] = np.clip(vy, -SPEED_CAP, SPEED_CAP)
            terms["sideways"][i] = -abs(vx)
            terms["spin"][i] = -abs(R[2] @ gyro)
            terms["heading"][i] = -yaw ** 2
            terms["upright"][i] = max(up, 0.0)
            terms["alive"][i] = 1.0
            terms["energy"][i] = -np.abs(d.actuator_force[self.act] * d.qvel[self.vadr]).sum()
            terms["action_change"][i] = -((a[i] - self.last_a[i]) ** 2).sum()
            terms["fall"][i] = -float(fell[i])
        reward = sum(W[k] * v for k, v in terms.items())
        self.last_a = a
        self.t += 1
        return self.obs(), reward, dict(terms=terms, fell=fell, vy=np.array([d.qvel[1] for d in self.data]))

    def time_up(self):
        return self.t >= EPISODE_STEPS


class Normalizer:
    """Running mean and spread of the observations, so every input has a similar size."""

    def __init__(self):
        self.mean, self.var, self.count = np.zeros(OBS_DIM), np.ones(OBS_DIM), 1e-4

    def update(self, x):
        n = len(x)
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + x.var(0) * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    def __call__(self, x):
        return ((x - torch.tensor(self.mean, dtype=torch.float32)) / torch.tensor(np.sqrt(self.var) + 1e-6, dtype=torch.float32)).clamp(-10, 10)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        mlp = lambda out: nn.Sequential(nn.Linear(OBS_DIM, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, out))
        self.actor, self.critic = mlp(ACT_DIM), mlp(1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.0))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x):
        return torch.distributions.Normal(self.actor(x), self.log_std.exp())


def train(args):
    torch.manual_seed(args.seed)
    env = WalkerEnv(args.envs, args.seed)
    env.t = env.rng.integers(0, EPISODE_STEPS, args.envs)   # stagger the episodes so each batch mixes all stages
    agent, norm = Agent(), Normalizer()
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
    T, gamma, lam = 128, 0.99, 0.95
    raw = env.obs()
    norm.update(raw)
    obs = norm(torch.tensor(raw))
    log = open("walker_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["iter", "mean_step_reward", "forward_speed", "falls_per_100_s"] + list(W))
    print(f"{'iter':>5} {'reward/step':>12} {'speed (m/s)':>12} {'falls/100 s':>12}", flush=True)

    for it in range(1, args.iters + 1):
        for group in opt.param_groups:   # the learning rate shrinks as training goes on, which keeps it from unlearning
            group["lr"] = 3e-4 * max(0.1, 1 - it / args.iters)
        O, A, LP, R, V, D = [], [], [], [], [], []
        sums = {k: 0.0 for k in W}
        speed, falls = 0.0, 0
        for _ in range(T):
            with torch.no_grad():
                dist = agent.dist(obs)
                act = dist.sample()
                lp, val = dist.log_prob(act).sum(-1), agent.value(obs)
            nraw, rew, info = env.step(act.numpy())
            for k in W:
                sums[k] += float(np.mean(info["terms"][k]))
            speed += float(np.mean(info["vy"]))
            falls += int(info["fell"].sum())
            timeup, done = env.time_up() & ~info["fell"], env.time_up() | info["fell"]
            if it <= FREEZE_NORM:   # after this the observation scaling stays fixed, so the network sees stable inputs
                norm.update(nraw)
            nobs_t = norm(torch.tensor(nraw))
            if done.any():   # the time limit is not a failure: add the value of where we stopped
                with torch.no_grad():
                    rew = rew + gamma * agent.value(nobs_t).numpy() * timeup
                for i in np.flatnonzero(done):
                    env.reset_one(i)
                nraw = env.obs()
                nobs_t = norm(torch.tensor(nraw))
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
        fall_rate = falls / (T * args.envs * 0.02) * 100       # falls per 100 simulated seconds
        writer.writerow([it, f"{mean_r:.4f}", f"{speed / T:.4f}", f"{fall_rate:.2f}"] + [f"{W[k] * sums[k] / T:.4f}" for k in W])
        log.flush()
        if it % 10 == 0 or it == 1:
            print(f"{it:5d} {mean_r:12.3f} {speed / T:12.3f} {fall_rate:12.1f}", flush=True)
        if it % 50 == 0:
            torch.save(dict(agent=agent.state_dict(), mean=norm.mean, var=norm.var, count=norm.count), "walker_policy.pt")
    log.close()
    torch.save(dict(agent=agent.state_dict(), mean=norm.mean, var=norm.var, count=norm.count), "walker_policy.pt")
    print("saved walker_policy.pt and walker_log.csv")
    return agent, norm


def evaluate(agent, norm, seconds=10.0, video="walker_trained.mp4"):
    """Run the trained policy (no exploration noise), report how it did, write a video."""
    env = WalkerEnv(1, seed=123)
    try:
        renderer = mujoco.Renderer(env.model, 480, 640)
    except Exception as e:   # no EGL (for example a CPU node): still report numbers
        print(f"no video ({type(e).__name__}: {e}); try a GPU node")
        renderer = None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.chassis
    camera.distance, camera.azimuth, camera.elevation = 0.45, 0, -20
    d = env.data[0]
    start_y, frames, falls, energy = d.qpos[1], [], 0, 0.0
    for _ in range(int(seconds / 0.02)):
        with torch.no_grad():
            act = agent.actor(norm(torch.tensor(env.obs()))).numpy()   # the mean action
        _, _, info = env.step(act)
        energy += float(-info["terms"]["energy"][0]) * 0.02
        if renderer:
            renderer.update_scene(d, camera=camera)
            frames.append(renderer.render())
        if info["fell"][0]:
            falls += 1
            env.reset_one(0)
            start_y = d.qpos[1]
    if renderer:
        renderer.close()
        imageio.mimwrite(video, frames, fps=50)
        print(f"wrote {video}: {len(frames)} frames")
    print(f"in {seconds:g} s: moved {d.qpos[1] - start_y:+.2f} m toward +y since the last start, sideways {d.qpos[0]:+.2f} m, "
          f"fell {falls} times, used {energy:.2f} J")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=200, help="training iterations (each is 128 steps in each world)")
    ap.add_argument("--envs", type=int, default=64, help="how many robots to simulate at once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true", help="skip training; replay walker_policy.pt")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.eval:
        ckpt = torch.load("walker_policy.pt", weights_only=False)
        agent, norm = Agent(), Normalizer()
        agent.load_state_dict(ckpt["agent"])
        norm.mean, norm.var, norm.count = ckpt["mean"], ckpt["var"], ckpt["count"]
    else:
        agent, norm = train(args)
    evaluate(agent, norm)
