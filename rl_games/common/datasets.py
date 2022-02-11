import torch
import copy
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class PPODataset(Dataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        self.is_rnn = is_rnn
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.is_discrete = is_discrete
        self.is_continuous = not is_discrete
        total_games = self.batch_size // self.seq_len
        self.num_games_batch = self.minibatch_size // self.seq_len
        self.game_indexes = torch.arange(total_games, dtype=torch.long, device=self.device)
        self.flat_indexes = torch.arange(total_games * self.seq_len, dtype=torch.long, device=self.device).reshape(
            total_games, self.seq_len)

        self.special_names = ['rnn_states']

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.values_dict['mu'][start:end] = mu
        self.values_dict['sigma'][start:end] = sigma

    def __len__(self):
        return self.length

    def _get_item_rnn(self, idx):
        gstart = idx * self.num_games_batch
        gend = (idx + 1) * self.num_games_batch
        start = gstart * self.seq_len
        end = gend * self.seq_len
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names:
                if v is dict:
                    v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                    input_dict[k] = v_dict
                else:
                    if v is not None:
                        input_dict[k] = v[start:end]
                    else:
                        input_dict[k] = None

        rnn_states = self.values_dict['rnn_states']
        input_dict['rnn_states'] = [s[:, gstart:gend, :].contiguous() for s in rnn_states]

        return input_dict

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                if type(v) is dict:
                    v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[start:end]

        return input_dict

    def __getitem__(self, idx):
        if self.is_rnn:
            sample = self._get_item_rnn(idx)
        else:
            sample = self._get_item(idx)
        return sample


class StateBasedDataset:
    def __init__(self, dataset_path: str):
        if not Path(dataset_path).exists() or dataset_path == "":
            print(f"Dataset file {dataset_path} not exist")
            self.length = 0
            return
        self.is_hdf5 = False
        if dataset_path.endswith(("pkl", "pickle")):
            data = np.load(dataset_path, allow_pickle=True)
            data_obs = []
            data_action = []
            for k, v in data.items():
                data_obs.append(v['observations'])
                data_action.append(v['actions'])
            self.data_obs = np.concatenate(data_obs, axis=0)
            self.data_action = np.concatenate(data_action, axis=0)
            assert len(self.data_obs) == len(self.data_action), "Demo Dataset Error: Obs num does not match Action num."
            self.length = len(self.data_obs)
        elif dataset_path.endswith("hdf5"):
            import h5py
            self.is_hdf5 = True
            self.file = h5py.File(dataset_path)
            self.data = self.file["data"]
            self.data_obs = self.data["observations"]
            self.data_action = self.data["actions"]
            self.length = np.prod(self.data_obs.shape[:-1])
            self.data_ind = 0

        print("Demo dataset loaded, length = {}".format(self.length))

    def __len__(self):
        return self.length

    def get_random_batch(self, batchsize):
        if self.length == 0:
            raise RuntimeError
        idxes = np.random.randint(0, self.length, batchsize)
        if not self.is_hdf5:
            batch_obs = self.data_obs[idxes]
            batch_actions = self.data_action[idxes]
        else:
            ind_after = self.data_ind + batchsize
            if ind_after <= self.length:
                batch_obs = self.data_obs[self.data_ind: ind_after, ...]
                batch_actions = self.data_action[self.data_ind: ind_after, ...]
                self.data_ind = ind_after
            else:
                addition_ind = batchsize - (self.length - self.data_ind)
                batch_obs_before = self.data_obs[self.data_ind:, ...]
                batch_obs_after = self.data_obs[:addition_ind, ...]
                batch_obs = np.concatenate([batch_obs_before, batch_obs_after])
                batch_actions_before = self.data_action[self.data_ind:, ...]
                batch_actions_after = self.data_action[:addition_ind, ...]
                batch_actions = np.concatenate([batch_actions_before, batch_actions_after])
                self.data_ind = addition_ind

        return batch_obs, batch_actions


class DemoAugmentedPPODataset(Dataset):
    def __init__(self, ppo_dataset, dataset_path, demo_batch_size):
        self.demo_dataset = StateBasedDataset(dataset_path)
        self.ppo_dataset = ppo_dataset
        self.dapg_batch_size = demo_batch_size
        self.device = self.ppo_dataset.device
        self.is_rnn = self.ppo_dataset.is_rnn

    def update_values_dict(self, values_dict):
        self.ppo_dataset.values_dict = values_dict

    def update_mu_sigma(self, mu, sigma):
        self.ppo_dataset.update_mu_sigma(mu, sigma)

    def __len__(self):
        return self.ppo_dataset.length

    @property
    def demo_len(self):
        return self.demo_dataset.length

    def get_discrim_batch(self):
        input_dict = self.get_bc_batch(self.dapg_batch_size)

        obs = self.values_dict['obs']
        actions = self.values_dict['actions']
        idxes = np.random.randint(0, len(obs), self.dapg_batchsize)
        input_dict['obs'] = obs[idxes]
        input_dict['actions'] = actions[idxes]
        return input_dict

    def get_bc_batch(self, batch_size):
        input_dict = {}
        demo_obs, demo_actions = self.demo_dataset.get_random_batch(batch_size)
        input_dict['demo_obs'] = torch.from_numpy(demo_obs).float().to(self.device)
        input_dict['demo_actions'] = torch.from_numpy(demo_actions).float().to(self.device)
        return input_dict

    def _get_item(self, idx):
        input_dict = self.ppo_dataset._get_item(idx)
        demo_dict = self.get_bc_batch(self.dapg_batch_size)
        input_dict.update(demo_dict)

        return input_dict

    def __getitem__(self, idx):
        if self.is_rnn:
            raise NotImplementedError
        else:
            sample = self._get_item(idx)
        return sample


class DatasetList(Dataset):
    def __init__(self):
        self.dataset_list = []

    def __len__(self):
        return self.dataset_list[0].length * len(self.dataset_list)

    def add_dataset(self, dataset):
        self.dataset_list.append(copy.deepcopy(dataset))

    def clear(self):
        self.dataset_list = []

    def __getitem__(self, idx):
        ds_len = len(self.dataset_list)
        ds_idx = idx % ds_len
        in_idx = idx // ds_len
        return self.dataset_list[ds_idx].__getitem__(in_idx)
