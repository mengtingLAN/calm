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

from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn

from learning import amp_network_builder

ENC_LOGIT_INIT_SCALE = 0.1


class NCPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = NCPBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            actions_dim = kwargs.get('actions_num')
            ob_space = kwargs.get('input_shape')
            self.value_size = kwargs.get('value_size', 1)
            self.num_seqs = kwargs.get('num_seqs', 1)
            actions_log_std_init = torch.tensor(-2.0)
            self.actions_log_std = nn.Parameter(torch.ones(actions_dim) * actions_log_std_init, requires_grad=True)

            prop_dim = ob_space.spaces['prop'].shape[0]
            future_dim = ob_space.spaces['future'].shape[0]
            self.rms_prop = RunningMeanStd(prop_dim, params['rms_momentum'])
            self.rms_future = RunningMeanStd(future_dim, params['rms_momentum'])

            params['prop_dim'] = prop_dim
            params['future_dim'] = future_dim
            params['actions_dim'] = actions_dim
            nc = TrackingZConfig(**params)
            self.value_net = ValueNet(nc)
            self.pi_net = PiNet(nc)

            return

        def forward(self, obs_dict):
            is_train = obs_dict.get('is_train', True)
            prop = obs_dict['obs']['prop']
            future = obs_dict['obs']['future']

            prop_rms, prop_rms_loss = self.rms_prop(prop)
            future_rms, future_rms_loss = self.rms_future(future)

            prop_rms = prop_rms.clamp(-5.0, 5.0)
            future_rms = future_rms.clamp(-5.0, 5.0)

            action_mean, quantized, encoder_z = self.pi_net(prop_rms, future_rms)
            actor_outputs = (action_mean, self.actions_log_std)
            value = self.value_net(prop_rms, future_rms)
            states = obs_dict.get('rnn_states', None)

            if is_train:
                rms_loss = torch.mean(prop_rms_loss, dim=(0, 1)) + torch.mean(future_rms_loss, dim=(0, 1))
                e_latent_loss = torch.mean((quantized.detach() - encoder_z) ** 2, dim=(0, 1))
                q_latent_loss = torch.mean((quantized - encoder_z.detach()) ** 2, dim=(0, 1))
            else:
                rms_loss = None
                e_latent_loss = None
                q_latent_loss = None

            output = actor_outputs + (value, states) + (rms_loss, e_latent_loss, q_latent_loss)

            return output

        def load(self, params):
            self.separate = params.get('separate', False)
            self.has_rnn = 'rnn' in params

        def eval_critic(self, obs):
            prop = obs['prop']
            future = obs['future']

            prop_rms, _ = self.rms_prop(prop)
            future_rms, _ = self.rms_future(future)

            prop_rms = prop_rms.clamp(-5.0, 5.0)
            future_rms = future_rms.clamp(-5.0, 5.0)
            value = self.value_net(prop_rms, future_rms)
            return value




class ValueNet(nn.Module):
    def __init__(self, nc):
        super(ValueNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nc.prop_dim + nc.future_dim, nc.embed_dim),
            nn.ReLU(),
            nn.Linear(nc.embed_dim, nc.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(nc.embed_dim // 2, nc.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(nc.embed_dim // 4, 1)
        )

    def forward(self, prop, future):
        embed = torch.hstack((prop, future))
        return self.mlp(embed)

    def reset(self):
        pass

class Encoder(nn.Module):
    def __init__(self, nc):
        super(Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nc.prop_dim + nc.future_dim, nc.embed_dim),
            nn.ReLU(),
            nn.Linear(nc.embed_dim, nc.embed_dim),
            nn.ReLU(),
            nn.Linear(nc.embed_dim, nc.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(nc.embed_dim // 2, nc.z_len)
        )

    def forward(self, prop, future):
        embed = torch.hstack((prop, future))
        z = self.mlp(embed)
        return z

class LLC(nn.Module):
    def __init__(self, nc):
        super(LLC, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(nc.prop_dim + nc.z_len, nc.embed_dim),
            nn.ReLU(),
            nn.Linear(nc.embed_dim, nc.embed_dim),
            nn.ReLU(),
            nn.Linear(nc.embed_dim, nc.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(nc.embed_dim // 2, nc.actions_dim)
            # nn.Linear(nc.embed_dim // 2, nc.ac_space['A_LLC'].shape[0])
        )

    def forward(self, prop, z):
        z = z.squeeze()
        embed = torch.hstack((prop, z))
        out = self.mlp(embed)
        return out


class PiNet(nn.Module):
    def __init__(self, nc):
        super(PiNet, self).__init__()
        self.encoder = Encoder(nc)
        self.nc = nc
        self.embeddings = nn.Embedding(nc.num_embeddings, nc.z_len // nc.code_num)
        self.embeddings.weight.data.uniform_(-3.0 / nc.num_embeddings, 3.0 / nc.num_embeddings)
        self.num_embeddings = nc.num_embeddings

        self.llc = LLC(nc)

    def forward(self, prop, future):
        encoder_z = self.encoder(prop, future)
        flat_z = encoder_z.view(encoder_z.shape[0] * self.nc.code_num, self.nc.z_len // self.nc.code_num)
        encoding_indices = self.get_code_indices(flat_z)
        # print(encoding_indices)
        # encoding_indices = torch.randint(self.num_embeddings, size=encoding_indices.shape)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(encoder_z)  # [B, H, W, C]
        z_curr = encoder_z + (quantized - encoder_z).detach()
        out = self.llc(prop, z_curr)
        return out, quantized, encoder_z


    def reset(self):
        self.hidden_encoder = None
        self.hidden_llc = None

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings.weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

class RunningMeanStd(nn.Module):
    def __init__(self,
                 input_size,
                 momentum=0.99,
                 begin_norm_axis=0,
                 moving_mean_initializer=None,
                 moving_std_initializer=None,
                 trainable=True):
        super(RunningMeanStd, self).__init__()
        self.begin_norm_axis = begin_norm_axis
        self.momentum = momentum
        self.moving_mean = nn.Parameter(torch.zeros((1, input_size))) if moving_mean_initializer is None else moving_mean_initializer
        self.moving_std = nn.Parameter(torch.ones((1, input_size))) if moving_std_initializer is None else moving_std_initializer

    def forward(self, x):
        mean = torch.mean(x, dim=self.begin_norm_axis, keepdim=True)
        variance = torch.var(x, dim=self.begin_norm_axis, keepdim=True)
        rms_loss = 0.5 * self.momentum * (torch.square(self.moving_mean - mean.detach()) +
                                     torch.square(self.moving_std - torch.sqrt(variance.detach())))
        output = (x - self.moving_mean) / (self.moving_std + 1e-8)
        return output, rms_loss

class TrackingZConfig(object):
    def __init__(self, **kwargs):
        # allow partially overwriting
        for k, v in kwargs.items():
            self.__dict__[k] = v