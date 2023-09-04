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
import os

from collections import OrderedDict
from enum import Enum
import pandas as pd
import torch
from gym import spaces
from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch
from NCP_envs.tasks.humanoid import Humanoid, dof_to_obs
from utils import torch_utils
from utils.motion_lib import MotionLib
import uuid


class HumanoidNCP(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidNCP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._random_init_prob = cfg["env"]["randomInitProp"]
        self._changing_motion_steps = cfg["env"]["changingMotionSteps"]
        self._reset_default_env_ids = []
        self.enable_kin = not headless and cfg["env"].get('enable_render', True)
        if self.enable_kin:
            cfg["env"]["numHumanoid"] = 2
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if self.enable_kin:
            self._build_kin_tensors()
        self._time = torch.zeros(self.num_envs, device=self.device)
        self._motion_ids = torch.ones(self.num_envs, device=self.device, dtype=torch.long) * -1
        self._changing_motion_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_start_env = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        motion_file = cfg['env']['motion_file']
        motion_index = cfg["env"]['motion_index']
        self._load_motion(motion_file, motion_index)
        self._flat_local_key_pos_kin = torch.zeros((self.num_envs, self._key_body_ids.shape[0] * 3), device=self.device,
                                                   dtype=torch.float)
        self._flat_local_key_pos_dyn = torch.zeros((self.num_envs, self._key_body_ids.shape[0] * 3), device=self.device,
                                                   dtype=torch.float)

        self._allow_early_termination = cfg["env"].get('allow_early_termination', True)

        self._rewards_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._max_steps = self._motion_lib._motion_lengths // self.dt
        self._update_motion_weight_num = 0
        self._average_reward = torch.zeros(len(self._motion_lib._motion_lengths), device=self.device, dtype=torch.float)
        self._motion_sampled_times = torch.zeros(len(self._motion_lib._motion_lengths), device=self.device,
                                                 dtype=torch.float)
        self._dynamic_sample_strategy = cfg['env']['dynamic_sample_strategy']
        self._actor_id = str(uuid.uuid1())[:8]

        self._dynamic_sample_save_path = cfg['env']['dynamic_sample_save_path']
        if self._dynamic_sample_save_path != 'None':
            self._dynamic_sample_save_path = self._dynamic_sample_save_path + \
                                             self._actor_id + '_dynamic_sample_result.csv'
            if os.path.exists(self._dynamic_sample_save_path):
                dynamic_sample_result = np.array(pd.read_csv(self._dynamic_sample_save_path))
                if dynamic_sample_result.shape[0] > 0:
                    self._average_reward[:] = torch.from_numpy(
                        dynamic_sample_result[-1][0:len(self._motion_lib._motion_lengths)])
                self._motion_lib._motion_weights[:] = (1 - self._average_reward) ** self._dynamic_sample_strategy
                self._motion_lib._motion_weights /= self._motion_lib._motion_weights.sum()
        # allocate buffers
        if self.asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            self._num_future = 284
        elif self.asset_file == "mjcf/amp_humanoid.xml" or self.asset_file == "mjcf/amp_humanoid_boxing.xml":
            self._num_future = 254
        else:
            raise ValueError

        self.future_buf = torch.zeros(
            (self.num_envs, self._num_future), device=self.device, dtype=torch.float)
        dict_obs_space = OrderedDict({
            'prop': spaces.Box(0, 0, shape=(self._num_obs,)),
            'future': spaces.Box(0, 0, shape=(self._num_future,)),
        })

        self.observation_space = spaces.Dict(dict_obs_space)
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1., np.ones(self._num_actions) * 1.)

        return

    def pre_physics_step(self, actions):
        self._update_changing_motion_count()

        actions = actions + self._dof_pos
        if self.enable_kin:
            actions = torch.hstack((actions, torch.zeros_like(actions))).reshape(2 * actions.shape[0], actions.shape[1])
            forces = torch.zeros_like(actions)
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                            force_tensor,
                                                            gymtorch.unwrap_tensor(self._humanoid_actor_ids + 1),
                                                            len(self._humanoid_actor_ids))
        # self.actions = torch.clamp(actions, -1.0, 1.0).to(self.device).clone()
        # pd_tar = self._action_to_pd_targets(self.actions)
        pd_tar = actions
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        pd_tar_tensor,
                                                        gymtorch.unwrap_tensor(self._humanoid_actor_ids),
                                                        len(self._humanoid_actor_ids))

        return

    def _update_changing_motion_count(self):
        self._changing_motion_counter -= 1
        self._recovery_counter = torch.clamp_min(self._changing_motion_counter, 0)

    def post_physics_step(self):
        super().post_physics_step()
        if self.enable_kin:
            self._motion_sync()
        self._time += self.dt
        return

    @staticmethod
    def motion_post_process(motion, motion_type):
        def mapping(data):
            body_num = 17
            mapping_from = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], device=data.device)
            mapping_to = torch.tensor([0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16], device=data.device)
            res = torch.ones((data.shape[0], body_num, data.shape[2]), device=data.device) * float('nan')
            res[:, mapping_to] = data[:, mapping_from]
            return res

        if motion_type == 'humanoid':
            motion.global_translation = mapping(motion.global_translation)
            motion.global_rotation = mapping(motion.global_rotation)
            motion.local_rotation = mapping(motion.local_rotation)

    def _load_motion(self, motion_file, motion_index=None):
        if self.asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            motion_post_process = self.motion_post_process
        elif self.asset_file == "mjcf/amp_humanoid.xml" or self.asset_file == "mjcf/amp_humanoid_boxing.xml":
            motion_post_process = None
        else:
            raise ValueError

        assert (self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(),
                                     equal_motion_weights=self._equal_motion_weights,
                                     device=self.device)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if self.enable_kin:
            self._opponent_handle = []
            self._load_kin_humanoid()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_kin_humanoid(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.fix_base_link = True
        self._opponent_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(self._opponent_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(self._opponent_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(self._opponent_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(self._opponent_asset, left_foot_idx, sensor_pose)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self.enable_kin:
            self._build_kin_humanoid(env_id, env_ptr)
        return

    def _build_kin_humanoid(self, env_id, env_ptr):
        col_group = env_id + self.num_envs
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.p.x = 1.0

        humanoid_handle = self.gym.create_actor(env_ptr, self._opponent_asset, start_pose,
                                                "kin", col_group, col_filter + 1, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(0.78, 0.19, 0.19))
        self._opponent_handle.append(humanoid_handle)

        return

    def _build_kin_tensors(self):
        # get gym GPU state tensors
        num_actors = self.get_num_actors_per_env()

        self._kin_root_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[...,
                                1, :]
        self._all_actor_ids = torch.arange(self.num_envs * num_actors, device=self.device, dtype=torch.int32).view(
            self.num_envs, num_actors)
        num_humanoid = 2
        dofs_per_env = self._dof_state.shape[0] // self.num_envs // num_humanoid
        self._kin_dof_pos = self._dof_state.view(self.num_envs, num_actors, dofs_per_env, 2)[:, 1, :self.num_dof, 0]
        self._kin_dof_vel = self._dof_state.view(self.num_envs, num_actors, dofs_per_env, 2)[:, 1, :self.num_dof, 1]
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        super()._reset_envs(env_ids)
        return

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidNCP.StateInit.Default:
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidNCP.StateInit.Start
              or self._state_init == HumanoidNCP.StateInit.Random
              or self._state_init == HumanoidNCP.StateInit.Hybrid):
            self._reset_ref_state_init(env_ids)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_default(self, env_ids):
        num_envs = env_ids.shape[0]
        # self._teacher_sample_prob = self._teacher_init_sample_prob * torch.exp(
        #     -self._teacher_decay * self._total_step_num).repeat(num_envs)
        self._time[env_ids] = torch.zeros(num_envs, device=self.device)
        self._motion_ids[env_ids] = self._motion_lib.sample_motions(num_envs)

        dyn_root_states = self._initial_humanoid_root_states[env_ids].clone()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(self._motion_ids[env_ids], self._time[env_ids])

        default_heading_rot = torch_utils.calc_heading_quat(dyn_root_states[:, 3:7])
        kin_heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

        dyn_root_states[:, 3:7] = quat_mul(default_heading_rot,
                                           quat_mul(kin_heading_rot_inv, root_rot))
        dyn_root_states[:, :2] = root_pos[:, :2]

        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        reset_start_env_ids = env_ids
        if (self._dynamic_sample_strategy is not None and len(reset_start_env_ids) > 0):
            motion_id = self._motion_ids[reset_start_env_ids]
            # motion_id may be same, in this case only the first number will be set
            # e.g. self._average_reward[[0, 0, 0]] = torch.tensor([1, 2, 3.0], device='cuda:0') occurs no error
            self._average_reward[motion_id] = self._rewards_sum[reset_start_env_ids] / self._max_steps[motion_id]
            assert isinstance(self._dynamic_sample_strategy, int)
            assert (self._average_reward <= 1.0).all()
            self._motion_lib._motion_weights[:] = (1 - self._average_reward) ** self._dynamic_sample_strategy
            self._motion_lib._motion_weights /= self._motion_lib._motion_weights.sum()
            if self._update_motion_weight_num % 500 == 0:
                if self._dynamic_sample_save_path is not None:
                    import time
                    save_result = torch.vstack((torch.ones_like(self._average_reward) * time.time(),
                                                self._average_reward, self._motion_sampled_times))
                    pd.DataFrame(save_result.cpu().numpy().reshape(1, -1)).to_csv(self._dynamic_sample_save_path,
                                                                                  mode='a', header=False, index=False)
                self._update_motion_weight_num = 0
                self._motion_sampled_times = torch.zeros(len(self._motion_lib._motion_lengths),
                                                         device=self.device, dtype=torch.float)
            self._update_motion_weight_num += 1

        self._rewards_sum[env_ids] = 0.0
        self._motion_ids[env_ids] += 1
        self._motion_ids[env_ids] = self._motion_ids[env_ids] % self._motion_lib.num_motions()
        self._motion_sampled_times[self._motion_ids[env_ids]] += 1

        if self._state_init == HumanoidNCP.StateInit.Random or self._state_init == HumanoidNCP.StateInit.Start:
            if self._state_init == HumanoidNCP.StateInit.Random:
                self._time[env_ids] = self._motion_lib.sample_time(self._motion_ids[env_ids])
            else:
                self._time[env_ids] = torch.zeros(num_envs, device=self.device)
                # self._reset_start_env[env_ids] = 1.0

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(self._motion_ids[env_ids], self._time[env_ids])
            self._set_env_state(env_ids=env_ids,
                                root_pos=root_pos,
                                root_rot=root_rot,
                                dof_pos=dof_pos,
                                root_vel=root_vel,
                                root_ang_vel=root_ang_vel,
                                dof_vel=dof_vel)
            return

        if self._state_init == HumanoidNCP.StateInit.Hybrid:
            self._changing_motion_counter[env_ids] = 0
            from_start_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
            from_start_mask = torch.bernoulli(from_start_probs) == 1.0

            from_start_ids = env_ids[from_start_mask]
            if len(from_start_ids) > 0:
                self._time[from_start_ids] = torch.zeros(len(from_start_ids), device=self.device)
                self._reset_start_env[from_start_ids] = 1.0

            from_random_mask = torch.logical_not(from_start_mask)
            from_random_ids = env_ids[from_random_mask]

            motion_id = self._motion_ids.clone()
            time = self._time.clone()

            root_pos_original, root_rot_original = None, None
            if len(from_random_ids) > 0:
                time[from_random_ids] = self._time[from_random_ids] = self._motion_lib.sample_time(
                    self._motion_ids[from_random_ids])
                self._reset_start_env[from_random_ids] = 0.0

                random_init_probs = to_torch(np.array([self._random_init_prob] * len(from_random_ids)),
                                             device=self.device)
                random_init_mask = torch.bernoulli(random_init_probs) == 1.0
                random_init_ids = from_random_ids[random_init_mask]

                if len(random_init_ids) > 0:
                    motion_id[random_init_ids] = self._motion_lib.sample_motions(len(random_init_ids))
                    time[random_init_ids] = self._motion_lib.sample_time(motion_id[random_init_ids])
                    self._changing_motion_counter[random_init_ids] = self._changing_motion_steps
                    root_pos_original, root_rot_original, _, _, _, _, _ \
                        = self._motion_lib.get_motion_state(self._motion_ids[random_init_ids],
                                                            self._time[random_init_ids])
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(motion_id[env_ids], time[env_ids])
            if root_pos_original is not None:
                refine_state_mask = from_random_mask.clone()
                refine_state_mask[from_random_mask] = random_init_mask
                root_pos[refine_state_mask, :2] = root_pos_original[:, :2]
                original_heading_rot = torch_utils.calc_heading_quat(root_rot_original)
                kin_heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot[refine_state_mask])

                root_rot[refine_state_mask] = quat_mul(original_heading_rot,
                                                       quat_mul(kin_heading_rot_inv, root_rot[refine_state_mask]))

            self._set_env_state(env_ids=env_ids,
                                root_pos=root_pos,
                                root_rot=root_rot,
                                dof_pos=dof_pos,
                                root_vel=root_vel,
                                root_ang_vel=root_ang_vel,
                                dof_vel=dof_vel)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        return

    def _update_tracking_future(self):
        num_future = 2
        future_time = self._time.unsqueeze(-1) + (torch.arange(num_future, device=self.device) + 1) * self.dt
        motion_id = self._motion_ids.unsqueeze(-1).repeat((1, num_future))
        flat_future_time = future_time.view(future_time.shape[0] * num_future)
        flat_motion_id = motion_id.view(motion_id.shape[0] * num_future)

        root_pos_kin, root_rot_kin, dof_pos_kin, root_vel_kin, root_ang_vel_kin, dof_vel_kin, key_pos_kin \
            = self._motion_lib.get_motion_state(flat_motion_id, flat_future_time)

        future = compute_tracking_future(self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
                                         root_pos_kin, root_rot_kin, root_vel_kin, root_ang_vel_kin,
                                         dof_pos_kin, dof_vel_kin, key_pos_kin,
                                         self._local_root_obs, self._dof_obs_size, self._dof_offsets,
                                         num_future)
        self.future_buf[:] = future.view(future.shape[0] // num_future, num_future * future.shape[1])

        return

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._update_tracking_future()
        obs = OrderedDict({
            'prop': self.obs_buf.to(self.device),
            'future': self.future_buf.to(self.device),
        })
        return obs

    def step(self, actions):
        # self._total_step_num += 1
        super().step(actions)
        self._update_tracking_future()
        obs = OrderedDict({
            'prop': self.obs_buf.to(self.device),
            'future': self.future_buf.to(self.device),
        })
        # self.extras['rigid_body_pos'] = self._rigid_body_pos
        # self.extras['rigid_body_rot'] = self._rigid_body_rot
        return obs, self.rew_buf.to(self.device), self.reset_buf.to(self.device), self.extras

    def _compute_reward(self):
        self.rew_buf[:] = self._compute_humanoid_tracking_reward()
        self._rewards_sum[:] += self.rew_buf
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        # todo not working, because of refreshing after
        rigid_body_state_reshaped[env_ids, 0, :] = self._humanoid_root_states[env_ids]
        rigid_body_state_reshaped[env_ids, 1:, :] = 0
        rigid_body_state_reshaped[env_ids, 1:, 6] = 1
        return

    def _motion_sync(self):
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(self._motion_ids, self._time)

        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_kin_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel)

        env_ids_int32 = self._humanoid_actor_ids[env_ids] + 1
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _set_kin_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        root_pos[:, 0] += 1.0
        self._kin_root_states[env_ids, 0:3] = root_pos
        self._kin_root_states[env_ids, 3:7] = root_rot
        self._kin_root_states[env_ids, 7:10] = root_vel
        self._kin_root_states[env_ids, 10:13] = root_ang_vel
        self._kin_dof_pos[env_ids] = dof_pos
        self._kin_dof_vel[env_ids] = dof_vel
        return

    def _compute_humanoid_tracking_reward(self):
        root_pos_kin, root_rot_kin, dof_pos_kin, root_vel_kin, root_ang_vel_kin, dof_vel_kin, key_pos_kin \
            = self._motion_lib.get_motion_state(self._motion_ids, self._time)

        _, local_root_vel_kin, local_root_ang_vel_kin, dof_obs_kin, self._flat_local_key_pos_kin[:] = \
            extract_tracking_feature(root_pos_kin, root_rot_kin, root_vel_kin, root_ang_vel_kin,
                                     dof_pos_kin, key_pos_kin,
                                     self._local_root_obs, self._dof_obs_size, self._dof_offsets)

        root_rot_dyn, local_root_vel_dyn, local_root_ang_vel_dyn, dof_obs_dyn, self._flat_local_key_pos_dyn[:] = \
            extract_tracking_feature(self._rigid_body_pos[:, 0, :],
                                     self._rigid_body_rot[:, 0, :],
                                     self._rigid_body_vel[:, 0, :],
                                     self._rigid_body_ang_vel[:, 0, :],
                                     self._dof_pos, self._rigid_body_pos[:, self._key_body_ids, :],
                                     self._local_root_obs, self._dof_obs_size, self._dof_offsets)
        reward = _compute_humanoid_tracking_reward(self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
                                                   local_root_vel_dyn,
                                                   local_root_ang_vel_dyn, dof_obs_dyn, self._dof_vel,
                                                   self._flat_local_key_pos_dyn[:],

                                                   root_pos_kin, root_rot_kin, local_root_vel_kin,
                                                   local_root_ang_vel_kin, dof_obs_kin, dof_vel_kin,
                                                   self._flat_local_key_pos_kin[:])
        # frame_idx = self._motion_lib.get_motion_idx(self._motion_ids, self._time)
        # env_idx = reward > self.max_reward[frame_idx]
        # better_frame_idx = frame_idx[env_idx]
        # self.save_joint_pos[better_frame_idx] = dof_obs_dyn[env_idx]
        # self.save_vel[better_frame_idx] = local_root_vel_dyn[env_idx]
        # self.save_ang_vel[better_frame_idx] = local_root_ang_vel_dyn[env_idx]
        # self.max_reward[frame_idx] = torch.where(reward > self.max_reward[frame_idx], reward, self.max_reward[frame_idx])
        # print("steps:", self._total_step_num)
        # print(sum(torch.all(self.save_joint_pos == 0, dim=1)))
        # # print("counts:", sum(self.max_reward == 0.0))
        # print("min_reward {}, avg_reward {}".format(min(self.max_reward), torch.mean(self.max_reward)))
        # print("num of reward < 0.6 {}".format(torch.sum(self.max_reward < 0.6)))
        # print("num of reward < 0.7 {}".format(torch.sum(self.max_reward < 0.7)))
        # if self._total_step_num % 100 == 0:
        #     data = {
        #         'reward': self.max_reward.cpu().numpy(),
        #     }
        #     np.save('tracking_reward_0519.npy', data)
        return reward

    def _compute_reset(self):
        super()._compute_reset()
        if self._allow_early_termination:
            self._compute_dyn_kin_difference_reset()
        time_end = self._time >= self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf = torch.where(time_end, torch.ones_like(self.reset_buf), self.reset_buf)
        self._terminate_buf[:] = self.reset_buf

    def _compute_dyn_kin_difference_reset(self):
        root_pos_kin, root_rot_kin, dof_pos_kin, root_vel_kin, root_ang_vel_kin, dof_vel_kin, key_pos_kin \
            = self._motion_lib.get_motion_state(self._motion_ids, self._time)
        root_rot_dyn = self._rigid_body_rot[:, 0, :]
        root_rot_err = quat_mul(root_rot_kin, quat_conjugate(root_rot_dyn))
        root_rot_err_angle, _ = torch_utils.quat_to_angle_axis(root_rot_err)

        key_body_err = self._flat_local_key_pos_kin - self._flat_local_key_pos_dyn
        key_body_err = torch.where(key_body_err.isnan(), torch.zeros_like(key_body_err), key_body_err)
        key_body_err = torch.linalg.norm(key_body_err.view(key_body_err.shape[0], self._key_body_ids.shape[0], 3),
                                         dim=-1)

        kin_global_rotation = self._motion_lib.get_motion_global_rotation(self._motion_ids, self._time)
        sword_global_rotation_kin = kin_global_rotation[:, 6]
        sword_global_rotation_dyn = self._rigid_body_rot[:, 6]

        sword_rot_err = quat_mul(sword_global_rotation_kin, quat_conjugate(sword_global_rotation_dyn))
        sword_rot_err_angle, _ = torch_utils.quat_to_angle_axis(sword_rot_err)
        big_root_rot_err_angle = torch.abs(root_rot_err_angle) > 0.5

        big_sword_rot_err_angle = torch.abs(sword_rot_err_angle) > 1.57

        if self.asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            big_key_body_pos_threshold = torch.tensor([1.0, 0.5, 0.5, 0.5, 1.5, 0.5], device=self._key_body_ids.device)
        elif self.asset_file == "mjcf/amp_humanoid.xml" or self.asset_file == "mjcf/amp_humanoid_boxing.xml":
            big_key_body_pos_threshold = torch.tensor([0.5, 0.5, 0.5, 0.5], device=self._key_body_ids.device)
            big_sword_rot_err_angle[:] = 0.0
        else:
            raise ValueError
        big_key_body_err = torch.any(torch.abs(key_body_err) > big_key_body_pos_threshold, dim=1)

        has_big_diff = torch.logical_or(torch.logical_or(big_root_rot_err_angle,
                                                         big_key_body_err),
                                        big_sword_rot_err_angle)
        has_big_diff *= (self.progress_buf > 1)
        changing_motion = self._changing_motion_counter > 0
        has_big_diff[changing_motion] = 0
        # if torch.abs(root_rot_err_angle[0]) > 0.5:
        #     print("root_rot_err", torch.abs(root_rot_err_angle[0]))
        # if torch.abs(sword_rot_err_angle[0]) > 1.57:
        #     print("sword_rot_err", sword_rot_err_angle[0])
        # if big_key_body_err[0]:
        #     print("big_key_body_err", torch.abs(key_body_err[0]))

        self.reset_buf = torch.where(has_big_diff, torch.ones_like(self.reset_buf), self.reset_buf)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def _compute_humanoid_tracking_reward(root_pos_dyn, root_rot_dyn, local_root_vel_dyn, local_root_ang_vel_dyn,
                                      dof_obs_dyn, dof_vel_dyn, flat_local_key_pos_dyn,
                                      root_pos_kin, root_rot_kin, local_root_vel_kin, local_root_ang_vel_kin,
                                      dof_obs_kin, dof_vel_kin, flat_local_key_pos_kin):
    # type:  (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    joint_pos_w = 0.30
    joint_vel_w = 0.1
    key_body_w = 0.30
    root_w = 0.2
    root_vel_w = 0.1

    total_w = joint_pos_w + joint_vel_w + key_body_w + root_w + root_vel_w

    joint_pos_w /= total_w
    joint_vel_w /= total_w
    key_body_w /= total_w
    root_w /= total_w
    root_vel_w /= total_w

    joint_pos_scale = 2.0
    joint_vel_scale = 0.1
    key_body_scale = 10
    root_scale = 20.0
    root_vel_scale = 2.0

    joint_pos_err = torch.linalg.norm((dof_obs_kin - dof_obs_dyn), dim=1)
    joint_vel_err = torch.linalg.norm((dof_vel_kin - dof_vel_dyn), dim=1)

    root_pos_err = torch.sum(torch.square(root_pos_kin - root_pos_dyn), dim=1)
    root_rot_err = quat_mul(root_rot_kin, quat_conjugate(root_rot_dyn))
    root_rot_err_angle, _ = torch_utils.quat_to_angle_axis(root_rot_err)
    root_err = root_pos_err + 0.5 * torch.square(root_rot_err_angle)

    key_body_err = flat_local_key_pos_kin - flat_local_key_pos_dyn
    key_body_err = torch.where(key_body_err.isnan(), torch.zeros_like(key_body_err), key_body_err)
    key_body_err = torch.linalg.norm(key_body_err, dim=1)

    root_vel_err = torch.sum(torch.square(local_root_vel_kin - local_root_vel_dyn), dim=1)
    root_ang_vel_err = torch.sum(torch.square(local_root_ang_vel_kin - local_root_ang_vel_dyn), dim=1)
    root_vel_err = root_vel_err + 0.1 * root_ang_vel_err

    joint_pos_reward = torch.exp(-joint_pos_scale * joint_pos_err)
    joint_vel_reward = torch.exp(-joint_vel_scale * joint_vel_err)
    key_body_reward = torch.exp(-key_body_scale * key_body_err)
    root_reward = torch.exp(-root_scale * root_err)
    root_vel_reward = torch.exp(-root_vel_scale * root_vel_err)
    reward = joint_pos_w * joint_pos_reward + joint_vel_w * joint_vel_reward + key_body_w * key_body_reward + root_w * root_reward + root_vel_w * root_vel_reward
    return reward


@torch.jit.script
def extract_tracking_feature(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, key_body_pos,
                             local_root_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, int, List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
                                           local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0],
                                            local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    return root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, flat_local_key_pos


# 140 dim height(1) + rotation(6) + vel(3) + ang_vel(3) + dof_obs(78) + dof_vel(31) + key_body_pos(18=6*3)
@torch.jit.script
def compute_tracking_future(root_pos_dyn, root_rot_dyn, root_pos_kin, root_rot_kin, root_vel_kin, root_ang_vel_kin,
                            dof_pos_kin, dof_vel_kin, key_pos_kin,
                            local_root_obs, dof_obs_size, dof_offsets, num_future):
    # type:  (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, int, List[int], int) -> Tensor

    _, local_root_vel_kin, local_root_ang_vel_kin, dof_obs_kin, flat_local_key_pos_kin = \
        extract_tracking_feature(root_pos_kin, root_rot_kin, root_vel_kin, root_ang_vel_kin,
                                 dof_pos_kin, key_pos_kin,
                                 local_root_obs, dof_obs_size, dof_offsets)
    pos_dyn_expand = root_pos_dyn.unsqueeze(-2).repeat((1, num_future, 1))
    flat_pos_dyn = pos_dyn_expand.view(pos_dyn_expand.shape[0] * pos_dyn_expand.shape[1], pos_dyn_expand.shape[2])
    pos_diff = root_pos_kin - flat_pos_dyn

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot_dyn)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, num_future, 1))
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_pos_diff = quat_rotate(flat_heading_rot, pos_diff)

    rot_dyn_expand = root_rot_dyn.unsqueeze(-2).repeat((1, num_future, 1))
    flat_rot_dyn_expand = rot_dyn_expand.view(rot_dyn_expand.shape[0] * rot_dyn_expand.shape[1],
                                              rot_dyn_expand.shape[2])
    root_rot_err = quat_mul(quat_conjugate(flat_rot_dyn_expand), root_rot_kin)
    root_rot_err_obs = torch_utils.quat_to_tan_norm(root_rot_err)
    flat_local_key_pos_kin = torch.where(flat_local_key_pos_kin.isnan(), torch.zeros_like(flat_local_key_pos_kin),
                                         flat_local_key_pos_kin)
    future = torch.cat(
        (local_pos_diff, root_rot_err_obs, local_root_vel_kin, local_root_ang_vel_kin, dof_obs_kin, dof_vel_kin,
         flat_local_key_pos_kin), dim=1)
    return future
