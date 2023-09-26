# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import math
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from env.tasks.humanoid_amp import HumanoidAMP

from utils import torch_utils


class HumanoidSit(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._tar_dist_min = 0.5
        self._tar_dist_max = 2.0
        self._near_dist = 1
        self._near_prob = 0.6

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        sit_body_names = cfg["env"]["sitBodyNames"]
        self._sit_body_ids = self._build_sit_body_ids_tensor(self.envs[0], self.humanoid_handles[0],
                                                                   sit_body_names)
        self._build_target_tensors()
        self.bbox_pos = torch.tensor(((-0.25, -0.241, 0), (0.222, -0.241, 0), (0.222, -0.241, 0.782),(-0.25, -0.241, 0.782),
                             (-0.25, 0.242, 0), (0.222, 0.242, 0), (0.222, 0.242, 0.782), (-0.25, 0.242, 0.782)), device=self.device, dtype=torch.float)

        self.sit_pos = torch.tensor((0.00005, 0.000434, 0.429), device=self.device, dtype=torch.float)
        self.chair_orientation = torch.tensor((1.0, 0.0, 0.0), device=self.device, dtype=torch.float)
        self.unmove_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 29
        return obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return

    def _load_target_asset(self):
        asset_root = "data/assets/mjcf/"
        asset_file = "Chair_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 30.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()

        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group,
                                              col_filter, segmentation_id)
        self._target_handles.append(target_handle)

        return

    def _build_sit_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]

        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidAMP.StateInit.Fixed:
            self._reset_fixed(env_ids)
        elif self._state_init == HumanoidAMP.StateInit.Default:
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start
              or self._state_init == HumanoidAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidAMP.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_fixed(self, env_ids):
        n = len(env_ids)

        fixed_distance = 0.4

        self._humanoid_root_states[env_ids, 0] = fixed_distance + self._target_states[env_ids, 0]
        self._humanoid_root_states[env_ids, 1] = self._target_states[env_ids, 1]
        self._humanoid_root_states[env_ids, 2] = self._initial_humanoid_root_states[env_ids, 2]

        self._humanoid_root_states[env_ids, 3:7] = self._initial_humanoid_root_states[env_ids, 3:7]
        self._humanoid_root_states[env_ids, 7:10] = self._initial_humanoid_root_states[env_ids, 7:10]
        self._humanoid_root_states[env_ids, 10:13] = self._initial_humanoid_root_states[env_ids, 10:13]

        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return



    def _reset_default(self, env_ids):
        n = len(env_ids)

        init_near = torch.rand([n], dtype=self._target_states.dtype,
                               device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype,
                                                   device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                 device=self._target_states.device) + self._tar_dist_min

        rand_theta = np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) \
                     + -0.5 * np.pi * torch.ones([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._humanoid_root_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + self._target_states[env_ids, 0]
        self._humanoid_root_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + self._target_states[env_ids, 1]
        self._humanoid_root_states[env_ids, 2] = self._initial_humanoid_root_states[env_ids, 2]

        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        self._humanoid_root_states[env_ids, 3:7] = rand_rot
        self._humanoid_root_states[env_ids, 7:10] = self._initial_humanoid_root_states[env_ids, 7:10]
        self._humanoid_root_states[env_ids, 10:13] = self._initial_humanoid_root_states[env_ids, 10:13]

        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self._state_init == HumanoidAMP.StateInit.Random
                or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == HumanoidAMP.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)

        init_near = torch.rand([num_envs], dtype=self._target_states.dtype,
                               device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([num_envs], dtype=self._target_states.dtype,
                                                   device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([num_envs], dtype=self._target_states.dtype,
                                                                 device=self._target_states.device) + self._tar_dist_min

        rand_theta = np.pi * torch.rand([num_envs], dtype=self._target_states.dtype, device=self._target_states.device) \
                     + -0.5 * np.pi * torch.ones([num_envs], dtype=self._target_states.dtype,
                                                 device=self._target_states.device)
        root_pos[:, 0] = rand_dist * torch.cos(rand_theta) + self._target_states[env_ids, 0]
        root_pos[:, 1] = rand_dist * torch.sin(rand_theta) + self._target_states[env_ids, 1]

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
        else:
            root_states = self._humanoid_root_states[env_ids]

        obs = compute_sit_observations(root_states, self.bbox_pos, self.sit_pos, self.chair_orientation)
        return obs

    def _compute_reward(self, actions):
        tar_pos = self._target_states[..., 0:3]
        char_root_state = self._humanoid_root_states

        self.rew_buf[:] = compute_sit_reward(tar_pos, self.sit_pos, char_root_state,
                                                self._prev_root_pos,
                                                self.chair_orientation,
                                                self.dt)
        return

    def _compute_reset(self):
        root_pos_err_threshold = 0.02
        root_pos_diff = self._humanoid_root_states[..., 0:3] - self._prev_root_pos
        root_pos_err = torch.sqrt(torch.sum(root_pos_diff*root_pos_diff, dim=-1))
        self.unmove_counter = torch.where(root_pos_err < root_pos_err_threshold, self.unmove_counter+1, self.unmove_counter)

        self.reset_buf[:], self._terminate_buf[:], self.unmove_counter[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                               self._contact_forces, self._contact_body_ids,
                                                                               self._rigid_body_pos,
                                                                               self._humanoid_root_states,
                                                                               self.chair_orientation,
                                                                               self._tar_contact_forces,
                                                                               self._sit_body_ids,
                                                                               self.max_episode_length,
                                                                               self._enable_early_termination,
                                                                               self._termination_heights,
                                                                               self.unmove_counter)
        return

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._target_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_sit_observations(root_states, bbox_pos, sit_pos, chair_orientation):
    # chair_orientation; bbox; target position;All these
    # goal features are recorded in the character’s local frame.
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    chair_orientation = chair_orientation.repeat(root_rot.shape[0], 1)
    local_chair_orientation = quat_rotate(heading_rot, chair_orientation)

    bbox_pos = bbox_pos.unsqueeze(0).repeat(root_rot.shape[0], 1, 1)
    local_bbox_pos = bbox_pos - root_pos.unsqueeze(1)
    local_bbox_pos[..., -1] = bbox_pos[..., -1]
    flat_local_bbox_pos = local_bbox_pos.view(local_bbox_pos.shape[0]*local_bbox_pos.shape[1], local_bbox_pos.shape[2])
    tmp_heading_rot = heading_rot.unsqueeze(1).repeat(1, bbox_pos.shape[1], 1)
    tmp_heading_rot = tmp_heading_rot.view(tmp_heading_rot.shape[0]*tmp_heading_rot.shape[1], tmp_heading_rot.shape[2])
    flat_local_bbox_pos = quat_rotate(tmp_heading_rot, flat_local_bbox_pos)
    local_bbox_pos = flat_local_bbox_pos.view(local_bbox_pos.shape[0], local_bbox_pos.shape[1] * local_bbox_pos.shape[2])

    sit_pos = sit_pos.repeat(root_rot.shape[0], 1)
    local_sit_pos = sit_pos - root_pos
    local_sit_pos[..., -1] = sit_pos[..., -1]
    local_sit_pos = quat_rotate(heading_rot, local_sit_pos)

    obs = torch.cat([local_chair_orientation[:, 0:2], local_bbox_pos, local_sit_pos], dim=-1)
    return obs


@torch.jit.script
def compute_sit_reward(tar_pos, tar_sit_pos, root_state, prev_root_pos, chair_orientation, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    tar_speed = 1.5
    vel_err_scale = 2.0
    tar_pos_err_scale = 0.5
    sit_pos_err_scale = 10

    near_reward_w = 0.7
    far_reward_w = 0.3

    tar_pos_reward_w = 0.5
    vel_reward_w = 0.4
    facing_reward_w = 0.1

    root_pos = root_state[..., 0:3]
    tar_pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_pos_err = torch.sum(tar_pos_diff * tar_pos_diff, dim=-1)
    tar_pos_reward = torch.exp(-tar_pos_err_scale * tar_pos_err)

    far_case = torch.sqrt(tar_pos_err) > 0.5

    tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    root_rot = root_state[..., 3:7]
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    tar_sit_pos_diff = tar_sit_pos - root_pos
    tar_sit_pos_err = torch.sum(tar_sit_pos_diff * tar_sit_pos_diff, dim=-1)
    tar_sit_pos_reward = torch.exp(-sit_pos_err_scale * tar_sit_pos_err)

    chair_orientation = chair_orientation.repeat(root_rot.shape[0], 1)
    near_facing_err = torch.sum(chair_orientation[..., 0:2] * facing_dir[..., 0:2], dim=-1)
    near_facing_reward = torch.clamp_min(near_facing_err, 0.0)

    far_reward = tar_pos_reward_w * tar_pos_reward + vel_reward_w * vel_reward \
                 + facing_reward_w * facing_reward
    # near_reward = torch.where(far_case, tar_sit_pos_reward, 0.5 * tar_sit_pos_reward + 0.5 * near_facing_reward)
    near_reward = tar_sit_pos_reward

    reward = torch.where(far_case, near_reward_w * near_reward + far_reward_w * far_reward,
                         near_reward_w * near_reward + far_reward_w)

    return reward


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, root_state,
                           chair_orientation, tar_contact_forces, sit_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, unmove_counter):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    contact_force_threshold = 1.0
    unmove_counter_threshold = 30

    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_failed = torch.logical_and(fall_contact, fall_height)

        # tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)
        #
        # root_rot = root_state[..., 3:7]
        # root_pos = root_state[..., 0:3]
        # heading_rot = torch_utils.calc_heading_quat(root_rot)
        # facing_dir = torch.zeros_like(root_pos)
        # facing_dir[..., 0] = 1.0
        # facing_dir = quat_rotate(heading_rot, facing_dir)
        # chair_orientation = chair_orientation.repeat(root_rot.shape[0], 1)
        # dir_diff = torch.sum(chair_orientation[..., 0:2] * facing_dir[..., 0:2], dim=-1)
        # is_reverse = dir_diff < 0
        # contact_and_is_reverse = torch.logical_and(tar_has_contact, is_reverse)
        #
        # has_failed = torch.logical_or(has_failed, contact_and_is_reverse)

        is_unmove = unmove_counter > unmove_counter_threshold
        has_failed = torch.logical_or(has_failed, is_unmove)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    unmove_counter = torch.where(reset, torch.zeros_like(unmove_counter), unmove_counter)

    return reset, terminated, unmove_counter
