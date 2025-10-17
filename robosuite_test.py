import numpy as np
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)
from robosuite.environments import MujocoEnv
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

options = {}

options["robots"] = "Panda"
arm_controller_config = suite.load_part_controller_config(
    default_controller="JOINT_POSITION"
)
robot = (
    options["robots"][0] if isinstance(options["robots"], list) else options["robots"]
)
options["controller_configs"] = refactor_composite_controller_config(
    arm_controller_config, robot, ["right", "left"]
)

env: ManipulationEnv = suite.make(
    "Lift",
    use_camera_obs=False,
    has_renderer=True,
    has_offscreen_renderer=False,
    reward_shaping=True,
    control_freq=20,
    **options,
)

env.reset()

# gripper_joint_indices = env.robots[0].gripper["right"]
# print(gripper_joint_indices)
# gripper_joint_limits = env.sim.model.jnt_range  # Joint limits
# print(gripper_joint_limits)

for _ in range(1000):
    # print(*env.action_spec[0].shape)
    # action = np.random.randn(*env.action_spec[0].shape) * 0.1
    action = np.zeros(*env.action_spec[0].shape)
    action[-1] = 1.0
    obs, reward, done, info = env.step(action)
    print(obs["robot0_gripper_qpos"])
    env.render()

env.close()
