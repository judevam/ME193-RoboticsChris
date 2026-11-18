"""Load the Onshape two-motor robot and make it ready to simulate. Used by walker_test.py (and later by training).

The Onshape export (onshape/robot.xml) is never edited, so you can re-export whenever the design changes. This file
fixes up what the exporter does not get right, in memory, each time the model is loaded:

  * motors:   velocity motors (command = leg speed in rad/s) with a torque limit that suits a 0.1 kg robot
              (the exporter's are strong position springs with no limit, far too strong for 6 g legs)
  * joints:   almost no friction (the exporter's 0.1 N*m would stall the motors)
  * sensors:  a gyro, the body orientation, and each leg's angle and speed
  * start:    the robot is lifted so it just clears the floor

Everything you may want to change is in the first block of constants.
"""
import mujoco
import numpy as np

MODEL = "onshape/scene.xml"     # the Onshape export (robot.xml plus a floor and a light)
MAX_TORQUE = 0.10    # N*m, the most each motor can push (a small geared hobby motor); raise it if the robot is too weak
MAX_SPEED = 15.0     # rad/s, the fastest each leg can turn (about 2.4 turns per second)
KV = 0.01            # N*m per rad/s: how hard the motor pushes to reach the commanded speed
JOINT_FRICTION = 0.0005
LEFT, RIGHT = "left_motor", "right_motor"
SHOW_IDEAL_PATH = True   # draw the straight line the robot should follow (runs toward +y) with black and white squares beside it; purely visual
CHECK = 0.05             # m: size of one black or white square

# The two shafts point in opposite directions (the right one along -x, the left one along +x), so to turn both legs the
# same physical way the commands need opposite signs. Use leg_commands() instead of thinking about it.
SIGN = {LEFT: -1.0, RIGHT: +1.0}


def load(path=MODEL):
    """Return the compiled MjModel with the fixes above applied."""
    spec = mujoco.MjSpec.from_file(path)
    chassis = spec.worldbody.first_body()
    if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
        chassis.add_freejoint()                      # let the robot move around

    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.frictionloss = JOINT_FRICTION
            try:
                j.damping = JOINT_FRICTION               # MuJoCo 3.x
            except TypeError:
                j.damping = [JOINT_FRICTION] * 3         # newer MuJoCo takes one value per axis

    for a in list(spec.actuators):                   # replace the exported position springs by velocity motors
        spec.delete(a)
    for name in (LEFT, RIGHT):
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_velocity(kv=KV)
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [-MAX_SPEED, MAX_SPEED]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-MAX_TORQUE, MAX_TORQUE]

    if SHOW_IDEAL_PATH:   # all of this is visual only: no collisions, nothing the policy can sense
        centre = chassis.ipos[0]    # the robot's own centre line (its body origin sits to one side of its middle)
        tile = lambda name, x, y, size, z, rgba: spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[size[0], size[1], 0.0002], pos=[x, y, z], rgba=rgba,
            contype=0, conaffinity=0)
        # the ideal path: a green strip from 0.5 m behind the start to 2.5 m ahead
        tile("ideal_path", centre, 1.0, (0.003, 1.5), 0.0003, [0.1, 0.9, 0.2, 1])
        # black and white squares on both sides of it: they slide past as the robot makes progress
        for row in range(int(3.0 / CHECK)):
            for col in range(2):
                color = [0.05, 0.05, 0.05, 1] if (row + col) % 2 == 0 else [0.95, 0.95, 0.95, 1]
                for side in (-1, 1):
                    x = centre + side * (0.06 + CHECK * (col + 0.5))
                    tile(f"check_{row}_{col}_{'l' if side < 0 else 'r'}", x, -0.5 + CHECK * (row + 0.5),
                         (CHECK / 2, CHECK / 2), 0.0001, color)

    chassis.add_site(name="imu", pos=chassis.ipos)   # at the centre of mass
    sensor = lambda name, kind, objtype, obj: spec.add_sensor(name=name, type=kind, objtype=objtype, objname=obj)
    sensor("gyro", mujoco.mjtSensor.mjSENS_GYRO, mujoco.mjtObj.mjOBJ_SITE, "imu")            # rad/s, in the body frame
    sensor("orientation", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_SITE, "imu")
    for name in (LEFT, RIGHT):
        sensor(f"{name}_angle", mujoco.mjtSensor.mjSENS_JOINTPOS, mujoco.mjtObj.mjOBJ_JOINT, name)
        sensor(f"{name}_speed", mujoco.mjtSensor.mjSENS_JOINTVEL, mujoco.mjtObj.mjOBJ_JOINT, name)
    return spec.compile()


def annotate(frame, *lines):
    """Write lines of text in the top-left corner of a video frame (a numpy image) and return the new frame."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:    # an older Pillow only has a small fixed font
        font = ImageFont.load_default()
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for k, text in enumerate(lines):
        draw.text((10, 8 + 22 * k), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def lowest_point(model, data):
    """The height of the lowest corner of any mesh in the robot."""
    low = np.inf
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_contype[g]:
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            low = min(low, (data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
    return low


def reset(model, data, clearance=0.005, leg_angles=None, yaw=0.0):
    """Put the robot back at the start, just above the floor.
    leg_angles: optional (left, right) start angles in rad. yaw: start heading in rad (0 = the body axes line up with the world)."""
    mujoco.mj_resetData(model, data)
    if yaw:
        data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]    # the free joint's orientation: a turn about vertical
    if leg_angles is not None:
        for name, angle in zip((LEFT, RIGHT), leg_angles):
            data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    data.qpos[2] += clearance - lowest_point(model, data)    # the free joint's first 3 numbers are x, y, z
    mujoco.mj_forward(model, data)


def leg_commands(forward, turn=0.0):
    """Leg speeds to command, in rad/s. forward > 0 turns both legs the same way, turn > 0 turns it counter-clockwise seen from above (checked in simulation)."""
    return {LEFT: SIGN[LEFT] * (forward + turn), RIGHT: SIGN[RIGHT] * (forward - turn)}


def command(model, data, forward, turn=0.0):
    for name, speed in leg_commands(forward, turn).items():
        data.ctrl[model.actuator(name).id] = speed
