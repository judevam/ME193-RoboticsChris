"""Run the trained Walk-Straight policy on a real LEGO Education Double Motor. A first, rough try.

    pip install legoeducation numpy
    python run_on_robot.py --dry-run         # connect, read the sensors, print what the policy WOULD do (motors off)
    python run_on_robot.py --direction-test  # both legs "forward" at 30% for 3 s: which way does the robot go?
    python run_on_robot.py                   # run the policy for 10 s at half power
    python run_on_robot.py --open-loop 40    # NO policy: both legs "forward" at 40% for --seconds; prints yaw and legs
    python run_on_robot.py --no-yaw          # the policy, but its heading input is held at 0 (rules out a yaw sign problem)

Run it on your Windows laptop (not WSL: WSL cannot see Bluetooth), with the walk_policy.npz made by export_policy.py next
to this file. Same setup as Controls/pd_tracker.py. Ctrl-C stops the motors.

What it does, 50 times a second: read both motors' positions and the IMU yaw, build the 20 numbers the policy was
trained on, run the network, and send each motor a speed (in percent).

WHAT IS FAKED, because the real robot cannot measure it (so expect it to be rough):
  * body velocity (3 inputs) and sideways offset from the line: set to 0
  * "which way is up": assumed upright
  * turning rate: computed from how fast the IMU yaw changes
The policy was trained with all of these, so it is getting worse information than it expects.

BEFORE THE FIRST RUN
  * Lift the robot so the legs spin freely, run --direction-test, and check that BOTH legs turn so the robot would be
    pushed the same way. If one turns backwards, flip its sign in DIR below.
  * Set the IMU so yaw is turning about the vertical (put the robot flat and turn it: yaw should change). Run --dry-run,
    turn the robot counter-clockwise by hand: the printed yaw should go UP. If it goes down, set YAW_SIGN = -1.
  * Start slow (POWER = 0.3), and keep a hand ready to pick the robot up.
"""
import argparse
import math
import sys
import time

import numpy as np

# ---- settings you may want to change -------------------------------------------------------------------------------
POWER = 0.5            # fraction of the full motor speed a policy output of 1.0 turns into (the sim's legs ran at 15 rad/s)
DIR = {"left": -1, "right": +1}   # flip a sign if that leg turns the wrong way (see --direction-test)
# The two shafts face opposite ways, so with both signs +1 one leg usually pushes forward and the other backward and the
# robot spins on the spot. If so, set one of them to -1 (try {"left": -1, "right": +1} first).
YAW_SIGN = +1          # +1 if the printed yaw INCREASES when you turn the robot counter-clockwise (seen from above), else -1
YAW_UNIT_DEG = 0.1     # the IMU reports yaw in tenths of a degree (QLearn/main.py divides by 10 too)
LOOP_HZ = 50           # the policy was trained to act 50 times a second
RUN_SECONDS = 10.0
POLICY_FILE = "walk_policy_imu.npz"   # the 18-input Walk-Straight-Real policy, looked for next to this file
# Raw IMU units and axes. The library does not document them: CHECK with  python run_on_robot.py --imu-check
# (robot flat and still: accel should read about 1 g on one axis; rotate it by hand and watch the gyro).
GYRO_UNIT_DPS = 0.1    # degrees per second per raw gyro count (ASSUMED, decidegrees/s like the angles)
ACCEL_UNIT_G = 0.001   # g per raw accelerometer count (ASSUMED, milli-g)
# Map the IMU axes to the sim's robot axes (x = right, y = forward, z = up): entries are (imu axis index 0..2, sign).
GYRO_MAP = [(0, 1), (1, 1), (2, 1)]    # sim gyro x, y, z from the raw gyroscope X, Y, Z
ACCEL_MAP = [(0, 1), (1, 1), (2, 1)]   # sim accel x, y, z from the raw accelerometer X, Y, Z
PITCH_SIGN, ROLL_SIGN = +1, +1
# ---------------------------------------------------------------------------------------------------------------------


class Policy:
  """The trained network, in numpy."""

  def __init__(self, path):
    import os
    if not os.path.isabs(path) and not os.path.exists(path):
      path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)   # also works when started from another folder
    p = np.load(path)
    self.mean, self.std, self.eps = p["mean"], p["std"], float(p["eps"])
    self.layers = [(p["w0"], p["b0"]), (p["w1"], p["b1"]), (p["w2"], p["b2"])]
    # Walk-Straight uses 20 inputs. Walk-Straight-Real (old) used 11: turning rate, heading (2), legs (4+2), command (2).
    # Walk-Straight-Real (current) uses 18, all from the IMU and motor packets: see Observer.build_imu.
    self.pick = [5, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19] if len(self.mean) == 11 else None
    self.imu18 = len(self.mean) == 18
    print(f"policy with {len(self.mean)} inputs" + (" (the real-robot version)" if self.pick or self.imu18 else ""))

  def __call__(self, obs):
    obs = np.asarray(obs, dtype=np.float32)
    if self.pick is not None:
      obs = obs[self.pick]
    x = (obs - self.mean) / (self.std + self.eps)
    for i, (w, b) in enumerate(self.layers):
      x = w @ x + b
      if i < len(self.layers) - 1:
        x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU
    return x   # [left leg, right leg], about -1..1


class Observer:
  """Builds the policy's 20 inputs from the robot's sensors, in the order the task defined them."""

  def __init__(self):
    self.prev = None
    self.last_action = np.zeros(2)         # [left, right]
    self.zero_pos = None
    self.wz = 0.0

  def build(self, left_deg, right_deg, yaw_deg, now):
    if self.zero_pos is None:
      self.zero_pos = (left_deg, right_deg)
    # leg angles in radians, in the direction the leg turns for a positive command, then in the sim's convention:
    # in the sim a positive command turns the right joint forward (+) and the left joint backward (-)
    fwd_left = DIR["left"] * math.radians(left_deg - self.zero_pos[0])
    fwd_right = DIR["right"] * math.radians(right_deg - self.zero_pos[1])
    q_right, q_left = +fwd_right, -fwd_left
    yaw = YAW_SIGN * math.radians(yaw_deg)   # the sim counts counter-clockwise as positive
    if self.prev is None:
      qd_right = qd_left = 0.0
    else:
      dt = max(now - self.prev[0], 1e-3)
      qd_right = +DIR["right"] * math.radians(right_deg - self.prev[2]) / dt
      qd_left = -DIR["left"] * math.radians(left_deg - self.prev[1]) / dt
      wz = YAW_SIGN * math.radians(yaw_deg - self.prev[3]) / dt
      self.wz = 0.7 * self.wz + 0.3 * wz   # a little smoothing: this is a difference of two noisy readings
    self.prev = (now, left_deg, right_deg, yaw_deg)
    return np.array([
      0.0, 0.0, 0.0,                         # body velocity: not measurable, set to 0
      0.0, 0.0, self.wz,                     # turning rate, from the yaw
      0.0, 0.0, -1.0,                        # gravity: assumed upright
      math.sin(yaw), math.cos(yaw), 0.0,     # heading, and sideways offset (not measurable, set to 0)
      math.sin(q_right), math.sin(q_left), math.cos(q_right), math.cos(q_left),
      qd_right, qd_left,                     # leg speeds, rad/s
      self.last_action[0], self.last_action[1],
    ], dtype=np.float32)


def build_imu18(obs20, tilt, gyro, accel):
  """The 18 inputs of the current Walk-Straight-Real, in the task's order:
  heading sin,cos (2), pitch,roll (2), gyro xyz (3, rad/s), accel xyz (3, g), leg sin,cos (4), leg speeds (2), last command (2)."""
  return np.concatenate([obs20[9:11], tilt, gyro, accel, obs20[12:20]]).astype(np.float32)


class Robot:
  """The LEGO Education Double Motor, over Bluetooth."""

  def __init__(self):
    import legoeducation as le
    self.le = le
    self.dm = le.DoubleMotor()
    print("Connecting to the Double Motor (switch it on and keep it close)...")
    for attempt in range(5):
      try:
        self.dm.connect(device_notification_delay=20)   # sensor updates every 20 ms
        break
      except Exception as e:
        if "not ready" in str(e).lower() and attempt < 4:
          time.sleep(1)
        else:
          raise
    if not self.dm.connected:
      sys.exit("Could not connect to the Double Motor.")
    self.dm.imu_reset_yaw_axis(0)   # call with the robot pointing along the line you want it to walk
    time.sleep(0.5)

  def read(self):
    """(left angle in degrees, right angle in degrees, yaw in degrees), cumulative."""
    le = self.le
    return (float(self.dm.motor[le.MOTOR_LEFT].position), float(self.dm.motor[le.MOTOR_RIGHT].position),
            float(self.dm.imu_device.yaw) * YAW_UNIT_DEG)

  def imu_extras(self):
    """(tilt [pitch, roll] rad, gyro xyz rad/s, accel xyz g), converted to the sim's axes and units."""
    d = self.dm.imu_device
    raw_g = [float(d.gyroscopeX), float(d.gyroscopeY), float(d.gyroscopeZ)]
    raw_a = [float(d.accelerometerX), float(d.accelerometerY), float(d.accelerometerZ)]
    gyro = [sg * math.radians(raw_g[i] * GYRO_UNIT_DPS) for i, sg in GYRO_MAP]
    accel = [sg * raw_a[i] * ACCEL_UNIT_G for i, sg in ACCEL_MAP]
    tilt = [PITCH_SIGN * math.radians(float(d.pitch) * YAW_UNIT_DEG), ROLL_SIGN * math.radians(float(d.roll) * YAW_UNIT_DEG)]
    return np.array(tilt), np.array(gyro), np.array(accel)

  def send(self, left_pct, right_pct):
    le = self.le
    self.dm.motor_run(motor=le.MOTOR_LEFT, speed=int(left_pct), blocking=False)
    self.dm.motor_run(motor=le.MOTOR_RIGHT, speed=int(right_pct), blocking=False)

  def stop(self):
    try:
      self.dm.motor_stop()
    except Exception as e:
      print("stop failed:", e)


class FakeRobot:
  """For --selftest: no hardware, made-up sensors, to check the program runs."""

  def __init__(self):
    self.t = 0.0

  def imu_extras(self):
    return np.zeros(2), np.zeros(3), np.array([0.0, 0.0, 1.0])

  def read(self):
    self.t += 1.0 / LOOP_HZ
    return 200 * self.t, -150 * self.t, 5 * math.sin(self.t)   # (yaw already in degrees here)

  def send(self, left_pct, right_pct):
    pass

  def stop(self):
    pass


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dry-run", action="store_true", help="read sensors and print the policy's commands, but keep the motors off")
  ap.add_argument("--direction-test", action="store_true", help="both legs 'forward' at 30%% for 3 s")
  ap.add_argument("--selftest", action="store_true", help="no hardware: run the loop on made-up sensors")
  ap.add_argument("--open-loop", type=float, default=None, metavar="PCT", help="no policy: both legs 'forward' at PCT percent")
  ap.add_argument("--policy-file", default=POLICY_FILE, help="the exported policy (default walk_policy_imu.npz)")
  ap.add_argument("--imu-check", action="store_true", help="print the raw IMU and motor numbers for 15 s (motors off) to check units and axes")
  ap.add_argument("--no-yaw", action="store_true", help="hold the heading input at 0 (and the turning rate)")
  ap.add_argument("--seconds", type=float, default=RUN_SECONDS)
  ap.add_argument("--power", type=float, default=POWER, help="0 to 1: how much of the full speed a policy output of 1.0 gives")
  args = ap.parse_args()

  robot = FakeRobot() if args.selftest else Robot()
  try:
    if args.direction_test:
      print("Both legs 'forward' at 30% for 3 s. Watch which way each leg pushes; flip its sign in DIR if it is wrong.")
      robot.send(DIR["left"] * 30, DIR["right"] * 30)
      time.sleep(3)
      return
    if args.imu_check:
      print("Raw IMU (gyro, accel) and converted values. Flat and still: accel should be ~1 g on ONE axis, gyro ~0. Rotate by hand.")
      d = robot.dm.imu_device
      for _ in range(75):
        tilt, gyro, accel = robot.imu_extras()
        print(f"raw gyro {d.gyroscopeX:6.0f} {d.gyroscopeY:6.0f} {d.gyroscopeZ:6.0f}  raw accel {d.accelerometerX:6.0f} {d.accelerometerY:6.0f} "
              f"{d.accelerometerZ:6.0f}  yaw/pitch/roll {d.yaw:6.0f} {d.pitch:6.0f} {d.roll:6.0f} | gyro rad/s {np.round(gyro, 2)} accel g {np.round(accel, 2)}")
        time.sleep(0.2)
      return
    if args.open_loop is not None:
      print(f"No policy: both legs 'forward' at {args.open_loop:g}% for {args.seconds:g} s. The robot should walk roughly straight.")
      start = time.monotonic()
      robot.send(DIR["left"] * args.open_loop, DIR["right"] * args.open_loop)
      while time.monotonic() - start < args.seconds:
        left_deg, right_deg, yaw_deg = robot.read()
        print(f"t={time.monotonic() - start:4.1f}s  yaw {yaw_deg:+6.1f} deg   legs {left_deg:8.0f} {right_deg:8.0f} deg")
        time.sleep(0.2)
      return
    policy, obs_builder = Policy(args.policy_file), Observer()
    print(f"Running {args.seconds:g} s at power {args.power:g}{' (DRY RUN: motors off)' if args.dry_run else ''}. Ctrl-C to stop.")
    start = last = time.monotonic()
    tick = 0
    while time.monotonic() - start < args.seconds:
      now = time.monotonic()
      left_deg, right_deg, yaw_deg = robot.read()
      obs = obs_builder.build(left_deg, right_deg, 0.0 if args.no_yaw else yaw_deg, now)
      if policy.imu18:
        tilt, gyro, accel = robot.imu_extras()
        obs = build_imu18(obs, tilt, np.zeros(3) if args.no_yaw else gyro, accel)
      action = policy(obs)                              # [left, right]
      obs_builder.last_action = action.copy()
      command = np.clip(action, -1.0, 1.0) * args.power * 100
      left_pct, right_pct = DIR["left"] * command[0], DIR["right"] * command[1]
      if not args.dry_run:
        robot.send(left_pct, right_pct)
      if tick % (LOOP_HZ // 5) == 0:
        print(f"t={now - start:4.1f}s  yaw {yaw_deg:+6.1f} deg   legs {left_deg:8.0f} {right_deg:8.0f} deg   "
              f"policy {action[0]:+5.2f} {action[1]:+5.2f}   motors {left_pct:+4.0f}% {right_pct:+4.0f}%")
      tick += 1
      time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - now)))
  except KeyboardInterrupt:
    print("\nstopped")
  finally:
    robot.stop()
    print("motors stopped")


if __name__ == "__main__":
  main()
