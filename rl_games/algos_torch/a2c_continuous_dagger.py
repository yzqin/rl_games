from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.algos_torch import ppg_aux
from rl_games.common.ewma_model import EwmaModel
from rl_games.algos_torch import players
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.common import env_configurations

import time
from torch import optim
import torch
from torch import nn
import numpy as np
import gym
import os


def omegaconf_to_dict(d):
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


class A2CAgent(a2c_common.ContinuousA2CBase):
    def __init__(self, base_name, config):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, config)
        obs_shape = self.obs_shape
        self.config = config

        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1)
        }

        # Instantiate the model from the network builder.
        self.model = self.network.build(config)
        self.model.to(self.ppo_device)

        # Target model, used to provide kl loss for dapg update.
        # self.target_model = self.network.build(config)
        # self.target_model.to(self.ppo_device)
        # self.target_model.eval()

        self.states = None
        if self.ewma_ppo:
            self.ewma_model = EwmaModel(self.model, ewma_decay=0.889)
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08,
                                    weight_decay=self.weight_decay)

        if self.normalize_input:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.running_mean_std = RunningMeanStdObs(obs_shape).to(self.ppo_device)
            else:
                self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)

        if self.has_central_value:
            cv_config = {
                'state_shape': self.state_shape,
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'num_steps': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'model': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'max_epochs': self.max_epochs,
                'multi_gpu': self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)

        demo_path = self.config['demo']
        print("Demo path = ", demo_path)
        self.dataset = datasets.DemoAugmentedPPODataset(demo_path,
                                                        self.batch_size, self.minibatch_size,
                                                        self.is_discrete, self.is_rnn,
                                                        self.ppo_device, self.seq_len, self.config['dapg_bs'])
        self.lambda_0 = self.config['lambda_0']
        self.lambda_1 = self.config['lambda_1']
        self.last_max_advantage = 1.0

        print("LAMBDA0_COEFF", self.lambda_0, self.lambda_1)

        if 'phasic_policy_gradients' in self.config:
            self.has_phasic_policy_gradients = True
            self.ppg_aux_loss = ppg_aux.PPGAux(self, self.config['phasic_policy_gradients'])
        self.has_value_loss = (self.has_central_value and self.use_experimental_cv) \
                              or (not self.has_phasic_policy_gradients and not self.has_central_value)
        self.algo_observer.after_init(self)

        # Expert policy
        expert_checkpoint = self.config["expert"]["checkpoint"]
        config_path = Path(expert_checkpoint).parent.parent / "config.yaml"
        cfg = OmegaConf.load(config_path)
        params = omegaconf_to_dict(cfg["train"]["params"])
        self.expert_model_builder = ModelBuilder()
        self.expert_model = self.expert_model_builder.load(params)
        expert_config = params["config"]
        expert_config["network"] = self.expert_model
        self.expert_config = expert_config
        self.expert_checkpoint = expert_checkpoint

    def update_epoch(self):
        self.epoch_num += 1

        return self.epoch_num

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

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

                # demo_loss = common_losses.l2_loss(action_mu_demo, demo_actions_batch)
            else:
                a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo,
                                                  curr_e_clip)

                # DAPG Demo loss
                demo_loss = common_losses.behavioral_cloning_actor_loss(action_log_probs_demo)
                # demo_loss = common_losses.l2_loss(action_mu_demo, demo_actions_batch)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch,
                                                   self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            b_loss = self.bound_loss(mu)
            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # Loss coefficient.
            demo_loss_discounted = demo_loss.mean() * self.lambda_0 \
                                   * (self.lambda_1 ** self.epoch_num)  # * self.dataset.last_max_advantage

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
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
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
        self.last_max_advantage = self.dataset.last_max_advantage.item()

        self.train_result = (a_loss, c_loss, entropy, \
                             kl_dist, self.last_lr, lr_mul, \
                             mu.detach(), sigma.detach(), b_loss)

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame)
        if self.ewma_ppo:
            self.ewma_model.reset()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')

        obs = batch_dict["obses"]
        expert_action = self.expert_player.get_action(obs, is_determenistic=True)

        batch_size = 512
        bc_losses = []
        loss_type = self.config["expert"]["loss"]
        assert loss_type in ["mse", "logp"]
        for mini_ep in range(0, self.mini_epochs_num):
            for batch_num in range(int(expert_action.shape[0] / batch_size)):
                start_num = batch_num * batch_size
                end_num = min(start_num + batch_size, expert_action.shape[0])
                batch_dict_demo = {
                    'is_train': True,
                    'prev_actions': expert_action[start_num: end_num, :],
                    'obs': obs[start_num: end_num, :],
                }

                res_dict_demo = self.model(batch_dict_demo)
                if loss_type == "logp":
                    action_log_probs_demo = res_dict_demo['prev_neglogp']
                    loss = common_losses.behavioral_cloning_actor_loss(action_log_probs_demo).mean()
                elif loss_type == "mse":
                    student_action = res_dict_demo["mus"]
                    loss = common_losses.l2_loss(batch_dict_demo["prev_actions"], student_action).mean()
                else:
                    raise NotImplementedError

                bc_losses.append(loss.detach().clone())

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

        # if self.has_central_value:
        #     self.train_central_value()

        # a_losses = []
        # c_losses = []
        # b_losses = []
        # entropies = []
        # kls = []
        #
        # self.target_model.eval()

        # if self.is_rnn:
        #     frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
        #     print(frames_mask_ratio)
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
        #
        # if self.schedule_type == 'standard_epoch':
        #     if self.multi_gpu:
        #         av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
        #     self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,
        #                                                             av_kls.item())
        #     self.update_lr(self.last_lr)
        #
        # if self.has_phasic_policy_gradients:
        #     self.ppg_aux_loss.train_net(self)

        # self.target_model.load_state_dict(self.model.state_dict())

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        loss_length = len(bc_losses)
        zero_tensor = torch.zeros(1, device=self.device)
        c_losses, b_losses, entropies, kls, last_lr, lr_mul = [zero_tensor] * loss_length, [
            zero_tensor] * loss_length, [zero_tensor] * loss_length, [zero_tensor] * (loss_length // 100), 0, 1

        # return batch_dict[
        #            'step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul
        return batch_dict[
                   'step_time'], play_time, update_time, total_time, bc_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)
        self.dataset.last_max_advantage = advantages.max()

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        env_info = env_configurations.get_env_info(self.vec_env.env)
        self.expert_config["env_info"] = env_info
        self.expert_player = players.PpoPlayerContinuous(self.expert_config)
        self.expert_player.restore(self.expert_checkpoint)
        self.expert_player.has_batch_dimension = True
        # We do behavior cloning to initialize the model.
        print("Start to do behavioral cloning!")
        bc_steps = 0

        self.set_train()
        for bc_epoch in range(self.config.get('bc_epoch', 100)):
            for i in range(10000 // 128):
                bc_training_dict = self.dataset.get_bc_batch(128)
                obs, action = bc_training_dict['demo_obs'], bc_training_dict['demo_actions']

                batch_dict_demo = {
                    'is_train': True,
                    'prev_actions': action,
                    'obs': obs,
                }

                res_dict_demo = self.model(batch_dict_demo)
                action_log_probs_demo = res_dict_demo['prev_neglogp']
                loss = common_losses.behavioral_cloning_actor_loss(action_log_probs_demo).mean()
                # loss.backward()

                if self.multi_gpu:
                    self.optimizer.zero_grad()
                else:
                    for param in self.model.parameters():
                        param.grad = None

                if bc_steps % 1000 == 0:
                    print("BC Step = {}, Loss = {}".format(bc_steps, loss.item()))
                    bc_checkpoint_name = self.config['name'] + 'BC_Init_' + str(bc_steps) + '_Loss_' + str(loss.item())
                    self.save(os.path.join(self.nn_dir, bc_checkpoint_name))

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

        print("Behavioral Cloning Finished.")

        # Dapg.
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0

        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            if self.multi_gpu:
                self.hvd.sync_stats(self)
            should_exit = False
            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = sum_time  # self.num_agents * sum_time
                scaled_play_time = play_time  # self.num_agents * play_time
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f'fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f}  fps total: {fps_total:.1f}')
                    print(f"Mean reward: {self.mean_rewards}")

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses,
                                 entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                self.post_write_stats(frame)
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                if epoch_num > self.max_epochs:
                    self.save(os.path.join(self.nn_dir,
                                           'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(
                                               mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit).float()
                self.hvd.broadcast_value(should_exit_t, 'should_exit')
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    def post_write_stats(self, frame):
        # self.writer.add_scalar('losses/demo_loss', self.demo_loss.item(), frame)
        # self.writer.add_scalar('losses/discounted_demo_loss', self.demo_loss_discounted.item(), frame)
        # self.writer.add_scalar('losses/last_max_advantage', self.last_max_advantage, frame)
        return

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
