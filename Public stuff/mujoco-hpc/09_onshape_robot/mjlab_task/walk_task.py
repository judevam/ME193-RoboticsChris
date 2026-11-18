"""mjlab task: teach the Onshape two-motor robot to walk in a straight line along +y.

The robot starts at the origin with its legs at random angles, pointing a little off to one side (random yaw). It
must move forward (+y) while staying on the line x = 0 and keeping its heading. The policy sees its own velocities,
which way is up, its heading error, its sideways offset and its leg angles and speeds, and outputs one speed for each
leg. Nothing is hard-coded: all of the steering is learned.

Importing this file registers the task "Walk-Straight" (see train_walk.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import (
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  bad_orientation,
  joint_vel_rel,
  last_action,
  projected_gravity,
  push_by_setting_velocity,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# the Onshape export: next to this file when run by runIt.py, one folder up in the repo
_HERE = Path(__file__).resolve().parent
ROBOT_XML = next(p for p in (_HERE / "onshape" / "robot.xml", _HERE.parent / "onshape" / "robot.xml") if p.exists())
LEFT, RIGHT = "left_motor", "right_motor"       # joint names in robot.xml
TARGET_SPEED = 0.15         # m/s rewarded; above the ~0.07 the last policy reached, so the reward keeps paying for more speed
MAX_LEG_SPEED = 15.0        # rad/s; a policy output of 1.0 means this leg speed
LEG_KD = 0.01               # how hard the motor pushes toward the leg speed it was told (N*m per rad/s)
LEG_TORQUE_LIMIT = 0.10     # N*m: the most each motor can push (a small geared hobby motor)
JOINT_FRICTION = 0.0005     # the exporter's 0.1 N*m would stall the motors
START_YAW = 0.2             # rad; the robot starts pointing up to this far off straight
MAX_OFF_LINE = 0.5          # m; the episode ends if the robot wanders this far from the line x = 0
EPISODE_S = 10.0
# Visual only (for make_video_walk.py): the green ideal line and black and white squares beside it.
SHOW_TRACK = os.environ.get("WALK_TRACK", "0") == "1"
CHECK = 0.05                # m: size of one square
# -------------------------------------------------------------------------------------------------------

# The two shafts point in opposite directions (the right one along -x, the left one along +x), so for both legs to
# push the robot toward +y the left leg must turn the other way. A policy output of +1 on both legs means "forward".
_LEG_SCALE = {LEFT: -MAX_LEG_SPEED, RIGHT: +MAX_LEG_SPEED}


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  for actuator in list(spec.actuators):     # mjlab adds its own leg motors (below)
    spec.delete(actuator)
  chassis = spec.body("chassis")
  if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
    chassis.add_freejoint()                 # let the robot move around (newer exports already have one)
  for j in spec.joints:
    if j.type == mujoco.mjtJoint.mjJNT_HINGE:
      j.frictionloss = JOINT_FRICTION
      try:
        j.damping = JOINT_FRICTION          # MuJoCo 3.x
      except TypeError:
        j.damping = [JOINT_FRICTION] * 3    # newer MuJoCo takes one value per axis
  if SHOW_TRACK:                            # purely visual: no collisions
    centre = float(chassis.ipos[0])         # the robot's own centre line (its body origin sits to one side of its middle)

    def tile(name, x, y, half_x, half_y, z, rgba):
      spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_x, half_y, 0.0002], pos=[x, y, z], rgba=rgba,
        contype=0, conaffinity=0,
      )

    tile("ideal_path", centre, 1.0, 0.003, 1.5, 0.0003, [0.1, 0.9, 0.2, 1])
    for row in range(int(3.0 / CHECK)):
      for col in range(2):
        color = [0.05, 0.05, 0.05, 1] if (row + col) % 2 == 0 else [0.95, 0.95, 0.95, 1]
        for side in (-1, 1):
          x = centre + side * (0.06 + CHECK * (col + 0.5))
          tile(f"check_{row}_{col}_{'l' if side < 0 else 'r'}", x, -0.5 + CHECK * (row + 0.5), CHECK / 2, CHECK / 2, 0.0001, color)
  return spec


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    # the legs reach about 5 cm below the body at the lowest, so start a little above the floor and let it settle
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.08), joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the speed we ask for
        BuiltinPdActuatorCfg(target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=LEG_KD, effort_limit=LEG_TORQUE_LIMIT),
      ),
    ),
  )


# ---- observations and rewards: how the robot is doing relative to "straight along +y" ----------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How far the robot has turned about the vertical since it started (its own +y axis is its front)."""
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this env's start line (its x axis), in metres."""
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw), off-line distance]. Shape [num_envs, 3]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw), _off_line(env)], dim=-1)


def leg_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin, cos] of each leg's angle: where each leg is in its turn. Shape [num_envs, 4]."""
  angle = env.scene["robot"].data.joint_pos
  return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for going along +y at TARGET_SPEED: 0 when standing still, 1 at the target, less beyond it, negative backwards.

  It slopes everywhere (a narrow bump around the target would give a robot that starts out still nothing to learn from).
  """
  vy = env.scene["robot"].data.root_link_lin_vel_w[:, 1]
  return torch.clamp(1.0 - torch.square((vy - TARGET_SPEED) / TARGET_SPEED), min=-1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_off_line(env))


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_yaw(env))


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


def walk_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "gravity": ObservationTermCfg(func=projected_gravity),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "leg_phase": ObservationTermCfg(func=leg_phase),
    "leg_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "legs": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=_LEG_SCALE, use_default_offset=True
    ),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
    # start with the legs at random angles, so the policy works from any phase of the leg turn
    "reset_legs": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-math.pi, math.pi), "velocity_range": (0.0, 0.0)},
    ),
  }
  if not play:
    # domain randomization: each robot gets a slightly different body and world, so the policy is not tuned to one
    # exact simulation. (Note: a contact uses the larger of the two surfaces' friction and the floor's is 1.0, so
    # this range matters most above 1.)
    events["friction"] = EventTermCfg(
      func=envs_mdp.dr.geom_friction,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "operation": "abs", "distribution": "uniform", "ranges": (0.5, 1.5)},
    )
    events["mass"] = EventTermCfg(    # mass and inertia scale together
      func=envs_mdp.dr.pseudo_inertia,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "alpha_range": (-0.2, 0.2)},
    )
    events["motor_strength"] = EventTermCfg(
      func=envs_mdp.dr.effort_limits,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "effort_limit_range": (0.7, 1.3), "operation": "scale"},
    )
    events["push"] = EventTermCfg(    # a small random shove every few seconds
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 6.0),
      params={"velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
    )

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=20.0),    # 0.1 m off the line costs 0.2
    "face_forward": RewardTermCfg(func=face_forward, weight=2.0),     # 0.3 rad off straight costs 0.18
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(70.0)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=0.6,
      elevation=-35.0,
      azimuth=90.0,     # from behind and above, so sideways drift shows as left or right of the green line
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    decimation=10,      # the policy acts every 20 ms, like the other trainer
    episode_length_s=1e10 if play else EPISODE_S,
  )


def walk_straight_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="walk_straight",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=2000,
  )


register_mjlab_task(
  task_id="Walk-Straight",
  env_cfg=walk_straight_env_cfg(),
  play_env_cfg=walk_straight_env_cfg(play=True),
  rl_cfg=walk_straight_ppo_runner_cfg(),
)
