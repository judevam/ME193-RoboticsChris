"""mjlab task: two tilted paddles pass a ball back and forth. Importing this file registers the task "Paddle-Pass".

The world: a left paddle at x = -0.4 m and a right paddle at x = +0.4 m. Each is a flat plate tilted TILT_DEG degrees so it
leans toward the middle: a ball that lands on the left paddle is bounced up and to the right, and one that lands on the
right paddle is bounced up and to the left. Each paddle slides up and down (start 0.5 m). The ball moves in the x-z plane
(side to side and up and down) and starts 1.0 m above the left paddle. The policy picks a height for each paddle every 20 ms.

Rewards (the three you asked for, plus one of mine):
    apex_height   +  at the top of each bounce, how close the peak is to TARGET (1.0 m)
    action_rate   -  how much the paddle commands change from step to step
    fell_penalty  -  once, when the ball hits the floor or leaves the court (the episode ends)
    pass_bonus    +  each time the ball crosses the middle sideways in the air (my addition: without it nothing
                     tells the paddles that the ball should travel back and forth)

A note on angles: MuJoCo reads angles written in the XML in DEGREES (see the <compiler angle="degree"/> line below), so
TILT_DEG goes straight into the paddle's euler setting. Inside the simulation, joint angles (qpos/qvel) are radians; this
task only has sliding joints, so it works in metres.

Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import action_rate_l2, last_action, reset_joints_by_offset, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
TILT_DEG = 20          # degrees: how far each paddle leans toward the middle (0 would be flat)
SPACING = 0.4          # m: each paddle is this far from the middle
TARGET = 1.0           # m: the height every bounce should peak at
PADDLE_START = 0.5     # m
BALL_START = 1.0       # m
PADDLE_RANGE = 0.3     # m: a policy output of +-1 moves a paddle this far from its start (so 0.2 to 0.8 m)
OUT_OF_BOUNDS = 0.9    # m: the ball is lost if it goes further than this to either side
FLOOR = 0.07           # m: the ball is lost if its centre gets this low (it is touching the floor)
EPISODE_S = 10.0
# -------------------------------------------------------------------------------------------------------

JOINTS = ("left_z", "right_z", "ball_x", "ball_z")

# The world, in MuJoCo's XML. The paddles slide up and down; the ball slides in x and z. Everything stays in one plane.
# The "court" body is just a fixed frame that holds everything (an mjlab entity needs one root body).
XML = f"""
<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <body name="court" pos="0 0 0">
      <!-- a marker for the camera to look at, and a red line at the target height -->
      <body name="view" pos="0 0 0.7">
        <geom type="sphere" size=".001" contype="0" conaffinity="0" rgba="0 0 0 0"/>
      </body>
      <geom name="target_line" type="box" size="{OUT_OF_BOUNDS} .004 .004" pos="0 0 {TARGET}" rgba="1 0 0 1"
            contype="0" conaffinity="0"/>

      <!-- left paddle: leans toward +x, so a falling ball is bounced to the right -->
      <body name="paddle_left" pos="{-SPACING} 0 0" gravcomp="1">
        <joint name="left_z" type="slide" axis="0 0 1" range="0.2 0.8"/>
        <geom type="box" size=".2 .25 .02" euler="0 {TILT_DEG} 0" mass="1" rgba=".9 .6 .1 1"
              solref="-30000 -5" friction="0.05 0.005 0.0001"/>
      </body>
      <!-- right paddle: leans toward -x -->
      <body name="paddle_right" pos="{SPACING} 0 0" gravcomp="1">
        <joint name="right_z" type="slide" axis="0 0 1" range="0.2 0.8"/>
        <geom type="box" size=".2 .25 .02" euler="0 {-TILT_DEG} 0" mass="1" rgba=".2 .7 .9 1"
              solref="-30000 -5" friction="0.05 0.005 0.0001"/>
      </body>

      <body name="ball" pos="0 0 0">
        <joint name="ball_x" type="slide" axis="1 0 0"/>
        <joint name="ball_z" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass=".1" rgba="1 .3 .2 1" solref="-30000 -5" friction="0.05 0.005 0.0001"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(XML)


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      # the joints sit at the origin, so a joint's position is the real height (or side position, for the ball)
      joint_pos={"left_z": PADDLE_START, "right_z": PADDLE_START, "ball_x": -SPACING, "ball_z": BALL_START},
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # a spring-damper that pulls each paddle to the commanded height (critically damped for a 1 kg paddle)
        BuiltinPositionActuatorCfg(target_names_expr=("left_z", "right_z"), stiffness=1000.0, damping=63.0, effort_limit=300.0),
      ),
    ),
  )


# ---- the state the rewards and observations read ---------------------------------------------------------------


def _state(env: ManagerBasedRlEnv):
  """(positions, velocities), each [num_envs, 4] in the order left paddle, right paddle, ball x, ball z."""
  robot = env.scene["robot"]
  ids = robot.find_joints(JOINTS, preserve_order=True)[0]
  return robot.data.joint_pos[:, ids], robot.data.joint_vel[:, ids]


def court_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """What the policy sees: the ball and both paddles, scaled to about -1..1. Shape [num_envs, 8]."""
  q, v = _state(env)
  lz, rz, bx, bz = q.unbind(-1)
  lv, rv, bvx, bvz = v.unbind(-1)
  return torch.stack([bx / 0.5, (bz - 0.8) / 0.5, bvx / 2, bvz / 3, (lz - 0.5) / 0.3, lv / 2, (rz - 0.5) / 0.3, rv / 2], dim=-1)


def _previous(env: ManagerBasedRlEnv, name: str, current: torch.Tensor) -> torch.Tensor:
  """The value this quantity had on the previous step (for detecting a top of a bounce or a crossing), then remember the new one."""
  prev = getattr(env, name, None)
  if prev is None:
    prev = current.clone()
  prev = torch.where((env.episode_length_buf <= 1).view(-1), current, prev)   # just reset: no history
  setattr(env, name, current.clone())
  return prev


# ---- rewards -------------------------------------------------------------------------------------------------


def apex_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """At the top of each bounce: 1 if the peak is exactly TARGET, falling off smoothly (0.07 m scale). 0 on every other step."""
  q, v = _state(env)
  bz, bvz = q[:, 3], v[:, 3]
  prev_bvz = _previous(env, "_prev_ball_vz", bvz)
  topped = (prev_bvz > 0) & (bvz <= 0)
  return torch.where(topped, torch.exp(-torch.square((bz - TARGET) / 0.07)), torch.zeros_like(bz))


def ball_lost(env: ManagerBasedRlEnv) -> torch.Tensor:
  q, _ = _state(env)
  return (q[:, 3] < FLOOR) | (q[:, 2].abs() > OUT_OF_BOUNDS)


def fell_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 on the step the ball is lost (the weight makes it a penalty)."""
  return ball_lost(env).float()


def pass_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 on a step where the ball crosses the middle moving sideways (faster than 0.3 m/s), in the air (above 0.25 m)."""
  q, v = _state(env)
  bx, bz, bvx = q[:, 2], q[:, 3], v[:, 2]
  prev_bx = _previous(env, "_prev_ball_x", bx)
  crossed = (torch.sign(prev_bx) != torch.sign(bx)) & (bvx.abs() > 0.3) & (bz > 0.25)
  return crossed.float()


# ---- environment ----------------------------------------------------------------------------------------


def paddle_pass_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "court": ObservationTermCfg(func=court_state),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "paddles": JointPositionActionCfg(
      entity_name="robot", actuator_names=("left_z", "right_z"), scale=PADDLE_RANGE, use_default_offset=True
    ),
  }

  events = {
    # put the paddles back at their start height (a reset only writes the joints some event names)
    "reset_paddles": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("left_z", "right_z")),
      },
    ),
    # the ball starts a little off the exact spot each time, so it is not one fixed trajectory
    "reset_ball": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.03, 0.03),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("ball_x", "ball_z")),
      },
    ),
  }

  rewards = {
    "apex_height": RewardTermCfg(func=apex_height, weight=1.0),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.01),
    "fell": RewardTermCfg(func=fell_penalty, weight=-10.0),
    "pass_bonus": RewardTermCfg(func=pass_bonus, weight=1.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "ball_lost": TerminationTermCfg(func=ball_lost),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=3.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="view",       # a fixed marker in the middle of the court, so the camera does not move
      distance=3.0,
      elevation=0.0,
      azimuth=90.0,           # from the front, looking along +y: the left paddle is on the left
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    decimation=10,            # the policy acts every 20 ms
    episode_length_s=1e10 if play else EPISODE_S,
  )


def paddle_pass_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.98,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="paddle_pass",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )


register_mjlab_task(
  task_id="Paddle-Pass",
  env_cfg=paddle_pass_env_cfg(),
  play_env_cfg=paddle_pass_env_cfg(play=True),
  rl_cfg=paddle_pass_ppo_runner_cfg(),
)
