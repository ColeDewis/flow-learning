# ref: https://github.com/Lifelong-Robot-Learning/LIBERO/issues/101

import os

import numpy as np
import robosuite
from robosuite.utils import camera_utils, transform_utils


class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


## Generate Cloud Point from
class PointCloud:
    def __init__(self):
        pass

    def set_cam(self, env, width, height, camera_name):
        self.camera_name = camera_name
        intrinsic_matrix = camera_utils.get_camera_intrinsic_matrix(
            env.sim, camera_name, height, width
        )
        self.cam = CameraInfo(
            width,
            height,
            fx=intrinsic_matrix[0, 0],
            fy=intrinsic_matrix[1, 1],
            cx=intrinsic_matrix[0, 2],
            cy=intrinsic_matrix[1, 2],
        )

    def gen_points(self, obs, env, cam):
        H, W = obs[cam + "_image"].shape[:2]  # modified
        self.set_cam(env, W, H, cam)
        colors = obs[cam + "_image"]
        depths = obs[cam + "_depth"]
        colors = colors.astype(np.float32) / 255
        depths = camera_utils.get_real_depth_map(env.sim, depths)[:, :, 0]

        # IMPORTANT!!!!!! flip color and depth !!!
        colors = colors[::-1]
        depths = depths[::-1]

        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap, indexing="xy")  # modified
        points_z = depths
        points_x = (xmap - self.cam.cx) / self.cam.fx * points_z
        points_y = (ymap - self.cam.cy) / self.cam.fy * points_z

        mask = (points_z > 0) & (points_z < 5)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)
        return points, colors  # modified, also returns colors

    def pc_cam_to_pc_world(self, pc, extrinsic):
        # extrinsic is ^{world} _{camera} \mathbf{T}
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3]
        pc = (R @ pc.T).T + T
        return pc

    def pc_world_to_pc_cam(self, pc, extrinsic):
        # extrinsic is ^{world} _{camera} \mathbf{T}
        extrinsic_inv = np.linalg.inv(extrinsic)
        R = extrinsic_inv[:3, :3]
        T = extrinsic_inv[:3, 3]
        pc = (R @ pc.T).T + T
        return pc


pointcloud = PointCloud()

env = robosuite.make(
    "NutAssemblySquare",
    renderer="mujoco",
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=240,
    camera_widths=320,
    has_renderer=True,
    has_offscreen_renderer=True,
    render_camera="agentview",
)

for _ in range(10):
    obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])

# `get_camera_extrinsic_matrix` returns cam pose under world, i.e. ^{world} _{camera} \mathbf{T}
# no need to inverse extrinsic in `pc_cam_to_pc_world`
agent_ex = camera_utils.get_camera_extrinsic_matrix(env.sim, "agentview")
hand_ex = camera_utils.get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")

agent_points, agent_colors = pointcloud.gen_points(obs, env, "agentview")
hand_points, hand_colors = pointcloud.gen_points(obs, env, "robot0_eye_in_hand")
# transfer to world coordinate system
agent_points = pointcloud.pc_cam_to_pc_world(agent_points, agent_ex)
hand_points = pointcloud.pc_cam_to_pc_world(hand_points, hand_ex)

# Can combine with open3d it seems
# points = np.concatenate([agent_points, hand_points], axis=0)
# colors = np.concatenate([agent_colors, hand_colors], axis=0)
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)
# point_cloud.colors = o3d.utility.Vector3dVector(colors)
# o3d.io.write_point_cloud("debug.ply", point_cloud)
