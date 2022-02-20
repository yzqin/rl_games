# The code is written by Zhaoheng Yin and modified by Yuzhe

import os

import torch
from torch import nn
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from rl_games.algos_torch import torch_ext, a2c_continuous
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.common import common_losses, datasets
from rl_games.common import env_configurations
from rl_games.algos_torch import players
import time


def omegaconf_to_dict(d):
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


class OnlineBCAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        # Demo setting
        bc_config = self.config.get("bc", {})
        if len(bc_config) > 0:
            demo_path = bc_config["demo_path"]
            print(f"BC demo path {demo_path}")
            self.dataset = datasets.DemoAugmentedPPODataset(self.dataset, demo_path, demo_batch_size=-1)
        else:
            pass  # keep original PPO dataset

        # Expert policy setting
        expert_checkpoint = self.config["expert"]["checkpoint"]
        config_path = Path(expert_checkpoint).parent.parent / "config.yaml"
        cfg = OmegaConf.load(config_path)
        params = omegaconf_to_dict(cfg["train"]["params"])
        self.expert_model_builder = ModelBuilder()
        self.expert_model = self.expert_model_builder.load(params)
        expert_config = params["config"]
        expert_config["network"] = self.expert_model
        self.expert_params = params
        self.expert_checkpoint = expert_checkpoint
        print(f"Loading expert checkpoint: {expert_checkpoint}")
        expert_modality = params["network"].get("modality", ["obs"])
        self.expert_modality = expert_modality

    def train(self):
        bc_config = self.config.get("bc", {})
        self.set_train()
        use_bc_init = len(bc_config)

        # Build expert
        env_info = env_configurations.get_env_info(self.vec_env.env)
        self.expert_params["config"]["env_info"] = env_info
        self.expert_player = players.PpoPlayerContinuous(self.expert_params)
        self.expert_player.restore(self.expert_checkpoint)
        self.expert_player.has_batch_dimension = True

        # Skip BC pretrain if bc config is not found
        if use_bc_init > 0:
            bc_epoch = bc_config["epoch"]
            bc_batch_size = bc_config["batch_size"]
            print(f"Start to do behavioral cloning with {bc_epoch}!")
            bc_steps = 0

            # Behavior cloning
            for bc_epoch in range(bc_epoch):
                for i in range(self.dataset.demo_len // bc_batch_size):
                    bc_training_dict = self.dataset.get_bc_batch(bc_batch_size)
                    obs, action = bc_training_dict['demo_obs'], bc_training_dict['demo_actions']

                    batch_dict_demo = {
                        'is_train': True,
                        'prev_actions': action,
                        'obs': obs,
                    }

                    res_dict_demo = self.model(batch_dict_demo)
                    action_log_probs_demo = res_dict_demo['prev_neglogp']
                    loss = common_losses.behavioral_cloning_actor_loss(action_log_probs_demo).mean()

                    if self.multi_gpu:
                        self.optimizer.zero_grad()
                    else:
                        for param in self.model.parameters():
                            param.grad = None

                    bc_steps += 1

                    self.scaler.scale(loss).backward()
                    # TODO: Refactor this ugliest code of they year
                    if self.truncate_grads:
                        if self.multi_gpu:
                            self.optimizer.synchronize()
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                            with self.optimizer.skip_synchronize():
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                        else:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                # Save BC checkpoint for each epoch
                print(f"BC Step = {bc_steps}, Epoch = {bc_epoch}, Loss = {loss.item()}")
                bc_checkpoint_name = self.config['name'] + 'BC_Init_' + str(bc_steps) + '_Loss_' + str(loss.item())
                self.save(os.path.join(self.nn_dir, bc_checkpoint_name))

            print("Behavioral Cloning Finished.")

        # Online distillation
        super().train()

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)
        if self.ewma_ppo:
            self.ewma_model.reset()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')

        loss_type = self.config["expert"]["loss"]
        expert_batch_size = self.config["expert"]["batch_size"]
        obs = batch_dict["obses"]
        is_obs_dict = isinstance(obs, dict)
        if is_obs_dict:
            obs_expert = {k: v for k, v in obs.items() if k in self.expert_modality}
        else:
            obs_expert = obs

        expert_action = self.expert_player.get_action(obs_expert, is_determenistic=True)
        batch_size = min(expert_batch_size, expert_action.shape[0])

        play_time_end = time.time()
        update_time_start = time.time()

        assert loss_type in ["mse", "logp"]
        export_bc_losses = []
        for mini_ep in range(0, self.mini_epochs_num):
            for batch_num in range(int(expert_action.shape[0] / batch_size)):
                start_num = batch_num * batch_size
                end_num = min(start_num + batch_size, expert_action.shape[0])
                if is_obs_dict:
                    obs_slice = {k: v[start_num: end_num] for k,v in obs.items()}
                else:
                    obs_slice = obs[start_num: end_num, :]
                batch_dict_demo = {
                    'is_train': True,
                    'prev_actions': expert_action[start_num: end_num, :],
                    'obs': obs_slice,
                }

                res_dict_demo = self.model(batch_dict_demo)
                if loss_type == "logp":
                    action_log_probs_demo = res_dict_demo['prev_neglogp']
                    export_bc_loss = common_losses.behavioral_cloning_actor_loss(action_log_probs_demo).mean()
                elif loss_type == "mse":
                    student_action = res_dict_demo["mus"]
                    export_bc_loss = common_losses.l2_loss(batch_dict_demo["prev_actions"], student_action).mean()
                else:
                    raise NotImplementedError

                loss = export_bc_loss
                export_bc_losses.append(export_bc_loss.detach().clone())

                if self.multi_gpu:
                    self.optimizer.zero_grad()
                else:
                    for param in self.model.parameters():
                        param.grad = None

                self.scaler.scale(loss).backward()
                # TODO: Refactor this ugliest code of they year
                if self.truncate_grads:
                    if self.multi_gpu:
                        self.optimizer.synchronize()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                        with self.optimizer.skip_synchronize():
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                    else:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

        # self.prepare_dataset(batch_dict)
        # self.algo_observer.after_steps()
        #
        # if self.has_central_value:
        #     self.train_central_value()
        #
        # a_losses = []
        # c_losses = []
        # b_losses = []
        # entropies = []
        # kls = []
        #
        # for mini_ep in range(0, self.mini_epochs_num):
        #     ep_kls = []
        #     for i in range(len(self.dataset)):
        #         a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(
        #             self.dataset[i])
        #         a_losses.append(a_loss)
        #         c_losses.append(c_loss)
        #         ep_kls.append(kl)
        #         entropies.append(entropy)
        #         if self.bounds_loss_coef is not None:
        #             b_losses.append(b_loss)
        #
        #         self.dataset.update_mu_sigma(cmu, csigma)
        #
        #         if self.schedule_type == 'legacy':
        #             if self.multi_gpu:
        #                 kl = self.hvd.average_value(kl, 'ep_kls')
        #             self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef,
        #                                                                     self.epoch_num, 0, kl.item())
        #             self.update_lr(self.last_lr)
        #
        #     av_kls = torch_ext.mean_list(ep_kls)
        #
        #     if self.schedule_type == 'standard':
        #         if self.multi_gpu:
        #             av_kls = self.hvd.average_value(av_kls, 'ep_kls')
        #         self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num,
        #                                                                 0, av_kls.item())
        #         self.update_lr(self.last_lr)
        #     kls.append(av_kls)
        #     self.diagnostics.mini_epoch(self, mini_ep)
        #     if self.normalize_input:
        #         self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch
        # if self.schedule_type == 'standard_epoch':
        #     if self.multi_gpu:
        #         av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
        #     self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,
        #                                                             av_kls.item())
        #     self.update_lr(self.last_lr)
        #
        # if self.has_phasic_policy_gradients:
        #     self.ppg_aux_loss.train_net(self)
        #
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        loss_length = len(export_bc_losses)
        a_losses = export_bc_losses
        zero_tensor = torch.zeros(1, device=self.device)
        c_losses, b_losses, entropies, kls, last_lr, lr_mul = [zero_tensor] * loss_length, [
            zero_tensor] * loss_length, [zero_tensor] * loss_length, [zero_tensor] * (loss_length // 100), 0, 1

        return batch_dict[
                   'step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul
