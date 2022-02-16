import torch
import torch.nn as nn

from .network_builder import NetworkBuilder, A2CBuilder
from ..modules.pointnet_modules import pointnet


class A2CPNBuilder(A2CBuilder):
    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            print(f"Input shape of PNBuilder: {input_shape}")

            assert "pointcloud" in input_shape
            assert "pointnet" in params

            input_shape_pc = input_shape["pointcloud"]
            # input_shape_pc = (input_shape_pc[0], input_shape_pc[2], input_shape_pc[1])  # (B, N, 3) -> (B, 3, N)
            input_shape_state = input_shape.get("obs", 0)

            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            self.value_size = kwargs.pop('value_size', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            pn_local_shape = params["pointnet"]["local_units"]
            pn_global_shape = params["pointnet"]["global_units"]
            pn_output_shape = pn_global_shape[-1]
            in_mlp_shape = pn_output_shape + input_shape_state[0]

            if len(self.units) == 0:
                out_size = in_mlp_shape
            else:
                out_size = self.units[-1]

            # Build PointNet
            self.pointnet = pointnet.PointNet(in_channels=3, local_channels=pn_local_shape,
                                              global_channels=pn_global_shape)

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                else:
                    rnn_in_size = in_mlp_shape
                    in_mlp_shape = self.rnn_units
                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                # self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size': in_mlp_shape,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear
            }

            self.mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            self.flatten_act = self.activations_factory.create(self.activation)
            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                                              requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            # Initialization
            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)

            if self.is_discrete:
                mlp_init(self.logits.weight)
            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            mlp_init(self.value.weight)

        def forward(self, obs_dict):
            obs_dict = obs_dict['obs']
            obs = obs_dict['pointcloud']
            obs = obs.permute((0, 2, 1))
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            out = obs
            out = self.pointnet(out)["feature"]
            out = out.flatten(1)

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    out = self.mlp(out)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)
                if len(states) == 1:
                    states = states[0]
                out, states = self.rnn(out, states)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)
                if type(states) is not tuple:
                    states = (states,)

                if self.is_rnn_before_mlp:
                    for l in self.mlp:
                        out = l(out)
            else:
                # Modification.
                if "obs" in obs_dict:
                    out = torch.cat([out, obs_dict["obs"]], dim=-1)

                for l in self.mlp:
                    out = l(out)

            value = self.value_act(self.value(out))

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.space_config['fixed_sigma']:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu * 0 + sigma, value, states

        def load(self, params):
            self.separate = params['separate']
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous' in params['space']
            self.is_multi_discrete = 'multi_discrete' in params['space']
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            if self.is_continuous:
                self.space_config = params['space']['continuous']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
            elif self.is_multi_discrete:
                self.space_config = params['space']['multi_discrete']
            self.has_rnn = 'rnn' in params
            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            num_layers = self.rnn_layers
            if self.rnn_name == 'lstm':
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)),
                        torch.zeros((num_layers, self.num_seqs, self.rnn_units)))
            else:
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)))

    def build(self, name, **kwargs):
        net = A2CPNBuilder.Network(self.params, **kwargs)
        return net
