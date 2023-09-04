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
import numpy as np
from gym import spaces
from isaacgym import gymtorch
from utils.motion_lib import MotionLib
from NCP_envs.tasks.humanoid_ncp import HumanoidNCP
from collections import OrderedDict

class HumanoidViewMotion(HumanoidNCP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt

        cfg["env"]["controlFrequencyInv"] = 1
        cfg["env"]["pdControl"] = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.asset_file = self.cfg["env"]["asset"]["assetFileName"]

        motion_file = cfg['env']['motion_file']
        motion_index = cfg["env"]['motion_index']
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)

        self.humanoid_sword_shield_body_names = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm',
                                                 'right_hand', 'sword', 'left_upper_arm', 'left_lower_arm', 'shield',
                                                 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh',
                                                 'left_shin', 'left_foot']
        self.humanoid_body_names = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand',
                                    'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin',
                                    'right_foot', 'left_thigh', 'left_shin', 'left_foot']

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

    def _load_motion(self, motion_file, motion_index):
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

    # def pre_physics_step(self, actions):
    #     self.actions = actions.to(self.device).clone()
    #     forces = torch.zeros_like(self.actions)
    #     force_tensor = gymtorch.unwrap_tensor(forces)
    #     self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
    #     return

    def post_physics_step(self):
        super().post_physics_step()
        self._motion_sync()
        return

    def _compute_observations(self, env_ids=None):
        return

    def _get_humanoid_collision_filter(self):
        return 1 # disable self collisions

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        motion_times = self.progress_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, motion_times)
        
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt)
        return

    def _reset_actors(self, env_ids):
        return

    def _reset_env_tensors(self, env_ids):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return


@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset, terminated
