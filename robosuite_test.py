import numpy as np
import robosuite as suite
from robosuite.environments import MujocoEnv
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

env: ManipulationEnv = suite.make(
    "Lift",
    robots="Panda",
    use_camera_obs=False,
    has_renderer=True,
    has_offscreen_renderer=False,
    reward_shaping=True,
    control_freq=20,
)

env.reset()


for _ in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
