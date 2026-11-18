"""mjlab task: Walk-Straight-Real. The same walking task as walk_task.py, but made so the policy can run on the REAL robot.

The first policy (Walk-Straight) used inputs the LEGO Double Motor cannot measure (body velocity, sideways offset, which way
is up), so on the real robot it did worse than just spinning both legs at a constant speed. Here the policy (the "actor")
sees ONLY what the real robot has, with realistic noise and delay:

    heading      (2)  sin and cos of the IMU yaw
    tilt         (2)  IMU pitch and roll (rad)
    gyroscope    (3)  IMU angular velocity about the robot's x, y, z (rad/s); z is the yaw rate
    accelerometer(3)  IMU acceleration along the robot's x, y, z, in g (includes gravity, like a real accelerometer)
    leg angles   (4)  sin and cos of each leg's angle, from the motor position counters
    leg speeds   (2)  from the motor positions changing
    last command (2)
  = 18 inputs, the same packet the Double Motor sends every 20 ms (ImuDeviceNotification + MotorNotification).

The value network (the "critic", used only during training) still sees everything, which helps it learn.
To make the policy ready for real hardware, training also adds: noise on every sensor, a delay between sensor and policy,
and a delay between the policy's command and the motor (Bluetooth).

Importing this file registers "Walk-Straight-Real" (see train_walk.py: add `import walk_real_task` there, or use
train_walk_real.py). Numbers measured on the real robot: 100% motor command is about 15 rad/s (40% gave about 6 rad/s).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import walk_task as wt
try:
  from mjlab.utils.lab_api.math import quat_apply_inverse    # newer mjlab
except ImportError:
  from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse
from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_vel_rel, last_action
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.noise import GaussianNoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- realism settings you may want to change ---------------------------------------------------------------------
COMMAND_DELAY_STEPS = (10, 40)   # physics steps (2 ms each) between the policy's command and the motor: 20 to 80 ms of Bluetooth lag
SENSOR_DELAY_STEPS = (0, 2)      # control steps (20 ms each) of lag on the sensors
GYRO_NOISE = 0.15                # rad/s per axis (z was the old "yaw rate" noise)
ACCEL_NOISE = 0.08               # g per axis: a small MEMS accelerometer on a vibrating, gear-driven robot is noisy
TILT_NOISE = 0.02                # rad: pitch and roll
G = 9.81
YAW_RATE_NOISE = GYRO_NOISE            # rad/s: the real yaw rate is a difference of two readings in tenths of a degree
HEADING_NOISE = 0.03             # rad
LEG_ANGLE_NOISE = 0.03           # rad: the motor position counters have 1 degree steps
LEG_SPEED_NOISE = 0.6            # rad/s: a speed from 1-degree steps over 20 ms is very rough
# -------------------------------------------------------------------------------------------------------------------


def yaw_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Turning rate about the vertical (counter-clockwise positive), rad/s. Shape [num_envs, 1]."""
  return env.scene["robot"].data.root_link_ang_vel_b[:, 2:3]


def gyro(env: ManagerBasedRlEnv) -> torch.Tensor:
  """IMU gyroscope: angular velocity in the robot's own axes, rad/s. Shape [num_envs, 3]."""
  return env.scene["robot"].data.root_link_ang_vel_b


def tilt(env: ManagerBasedRlEnv) -> torch.Tensor:
  """IMU pitch and roll (rad), from the direction of gravity in the robot's frame. Shape [num_envs, 2]."""
  g = env.scene["robot"].data.projected_gravity_b     # unit vector, (0, 0, -1) when level
  pitch = torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))
  roll = torch.atan2(-g[:, 1], -g[:, 2])
  return torch.stack([pitch, roll], dim=-1)


def accelerometer(env: ManagerBasedRlEnv) -> torch.Tensor:
  """IMU accelerometer: specific force in the robot's own axes, in g. Shape [num_envs, 3].

  A real accelerometer reads (acceleration - gravity): +1 g up when the robot sits still. MuJoCo gives velocities, so the
  acceleration is the change in body velocity over one control step (20 ms, the same rate as the real notifications).
  """
  data = env.scene["robot"].data
  v = data.root_link_lin_vel_w
  prev = getattr(env, "_walk_prev_v", None)
  if prev is None or prev.shape != v.shape:
    prev = v.clone()
  first = (env.episode_length_buf == 0).unsqueeze(-1)     # a fresh episode: no velocity history yet
  a_w = torch.where(first, torch.zeros_like(v), (v - prev) / env.step_dt)
  env._walk_prev_v = v.clone()
  a_b = quat_apply_inverse(data.root_link_quat_w, a_w)
  # specific force = a - g_vec, and projected_gravity_b is g_vec / |g| in the body frame
  return (a_b - G * data.projected_gravity_b) / G


def heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw)]: which way the robot points compared with the start. Shape [num_envs, 2]."""
  yaw = wt._yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def _robot_cfg_real() -> EntityCfg:
  return EntityCfg(
    spec_fn=wt._get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.08), joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPdActuatorCfg(
          target_names_expr=(wt.LEFT, wt.RIGHT), stiffness=0.0, damping=wt.LEG_KD, effort_limit=wt.LEG_TORQUE_LIMIT,
          delay_min_lag=COMMAND_DELAY_STEPS[0], delay_max_lag=COMMAND_DELAY_STEPS[1],   # Bluetooth lag
        ),
      ),
    ),
  )


def walk_real_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = wt.walk_straight_env_cfg(play)
  cfg.scene.entities = {"robot": _robot_cfg_real()}

  def term(func, noise_std=None):
    return ObservationTermCfg(
      func=func,
      noise=GaussianNoiseCfg(std=noise_std) if noise_std else None,
      delay_min_lag=SENSOR_DELAY_STEPS[0], delay_max_lag=SENSOR_DELAY_STEPS[1],
    )

  actor_terms = {
    "heading": term(heading, HEADING_NOISE),
    "tilt": term(tilt, TILT_NOISE),
    "gyro": term(gyro, GYRO_NOISE),
    "accel": term(accelerometer, ACCEL_NOISE),
    "leg_phase": term(wt.leg_phase, LEG_ANGLE_NOISE),
    "leg_vel": term(joint_vel_rel, LEG_SPEED_NOISE),
    "last_action": ObservationTermCfg(func=last_action),
  }
  # the critic is used only in training, so it can see the true full state
  critic_terms = {
    "lin_vel": ObservationTermCfg(func=wt.base_lin_vel),
    "ang_vel": ObservationTermCfg(func=wt.base_ang_vel),
    "gravity": ObservationTermCfg(func=wt.projected_gravity),
    "heading_offset": ObservationTermCfg(func=wt.heading_and_offset),
    "leg_phase": ObservationTermCfg(func=wt.leg_phase),
    "leg_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg(critic_terms),
  }
  return cfg


def walk_real_ppo_runner_cfg():
  cfg = wt.walk_straight_ppo_runner_cfg()
  cfg.experiment_name = "walk_straight_real"
  cfg.max_iterations = 1500     # a harder job (noise, delay, fewer inputs) than the first task
  return cfg


register_mjlab_task(
  task_id="Walk-Straight-Real",
  env_cfg=walk_real_env_cfg(),
  play_env_cfg=walk_real_env_cfg(play=True),
  rl_cfg=walk_real_ppo_runner_cfg(),
)
