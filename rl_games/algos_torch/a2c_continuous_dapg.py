# The code is written by Zhaoheng Yin and modified by Yuzhe

import os

import torch
from torch import nn

from rl_games.algos_torch import torch_ext, a2c_continuous
from rl_games.common import common_losses, datasets


class DAPGA2CAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        # Demo setting
        dapg_config = self.config['dapg']
        demo_path = dapg_config["demo_path"]
        batch_size = dapg_config["batch_size"]
        print(f"DAPG demo path {demo_path}")
        self.dataset = datasets.DemoAugmentedPPODataset(self.dataset, demo_path, batch_size)
        self.lambda_0 = dapg_config['lambda_0']
        self.lambda_1 = dapg_config['lambda_1']

        self.saved_data = 0
        print("DAPG Lambda: ", self.lambda_0, self.lambda_1)

    def calc_gradients(self, input_dict):
        # Original PPO data
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']

        # if self.modality_dict is not None:
        #     obs_batch = {}
        #     for modality_name in self.modality_dict:
        #         obs_batch[modality_name] = input_dict[modality_name]
        #     obs_batch = self._preproc_obs(obs_batch)
        #
        # else:
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        # Demo data
        # TODO: modality
        demo_obs_batch = input_dict['demo_obs']
        demo_obs_batch = self._preproc_obs(demo_obs_batch)
        demo_actions_batch = input_dict['demo_actions']

        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        batch_dict_demo = {
            'is_train': True,
            'prev_actions': demo_actions_batch,
            'obs': demo_obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # Demo augmentation
            res_dict_demo = self.model(batch_dict_demo)
            action_log_probs_demo = res_dict_demo['prev_neglogp']

            if self.ewma_ppo:
                ewma_dict = self.ewma_model(batch_dict)
                proxy_neglogp = ewma_dict['prev_neglogp']
                a_loss = common_losses.decoupled_actor_loss(old_action_log_probs_batch, action_log_probs, proxy_neglogp,
                                                            advantage, curr_e_clip)
                old_action_log_probs_batch = proxy_neglogp  # to get right statistic later
                old_mu_batch = ewma_dict['mus']
                old_sigma_batch = ewma_dict['sigmas']

                # DAPG Demo loss
                demo_loss = common_losses.behavioral_cloning_actor_loss(action_log_probs_demo)
            else:
                a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo,
                                                  curr_e_clip)

                # DAPG Demo loss
                demo_loss = common_losses.behavioral_cloning_actor_loss(action_log_probs_demo)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch,
                                                   self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # Loss coefficient.
            demo_loss_discounted = demo_loss.mean() * self.lambda_0 \
                                   * (self.lambda_1 ** self.epoch_num)

            loss = a_loss + demo_loss_discounted + 0.5 * c_loss * self.critic_coef \
                   - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

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

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        if self.ewma_ppo:
            self.ewma_model.update()

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)

        # Writer logs.
        self.demo_loss = demo_loss.mean()
        self.demo_loss_discounted = demo_loss_discounted

        self.train_result = (a_loss, c_loss, entropy, \
                             kl_dist, self.last_lr, lr_mul, \
                             mu.detach(), sigma.detach(), b_loss)

    def train(self):
        bc_config = self.config.get('bc')
        bc_epoch = bc_config["epoch"]
        bc_batch_size = bc_config["batch_size"]
        print(f"Start to do behavioral cloning with {bc_epoch}!")
        bc_steps = 0

        self.set_train()
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

        # Dapg.
        super().train()

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls,
                    last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        super().write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies,
                            kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
        self.writer.add_scalar('losses/demo_loss', self.demo_loss.item(), frame)
        self.writer.add_scalar('losses/discounted_demo_loss', self.demo_loss_discounted.item(), frame)
