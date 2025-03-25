import sys
import gymnasium as gym
import safety_gymnasium
import numpy as np
import copy
import os
import pickle
import shutil
import torch
import numpy as np
import tqdm

# import gym
# import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym
import pickle
from matplotlib import pyplot as plt
from Lyapunov_related import lyapunov_neural_network, lyapunov_train, lyapunov_train_pre
from utilities import Dynamics_NN
import safe_learning
FLAGS = flags.FLAGS


# flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_integer('start_training',  int(1e3),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', True, 'Log wandb.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 20, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)
FLAGS(sys.argv)
def main(_):
    seed_list = [0, 1, 3, 5, 7, 42, 1234, 2560, 5678, 100]
    for seed_new in seed_list:
        FLAGS.seed = seed_new
        np.random.seed(FLAGS.seed)
        observation_ly = []
        observation_gym = []
        action_gym = []
        observation_train = []
        observation_next = []
        cost = []
        index_all = []
        reward_pre = []
        observation_terminated = []
        reward_collection_50noise_eval = []
        q = []
        a_loss = []
        c_loss = []
        x_velocities = []
        velocity_test = []
        cost_test = []
        term_test = []
        length_eval_pure = []
        reward_done_50noise_eval_pure = []
        hopperVelocityThreshold = 0.7402  # Hopper
        walker2dVelocityThreshold = 2.3415
        antVelocityThreshold = 	2.6222
        env_name = "SafetyAntVelocity-v1"
        env = safety_gymnasium.make(env_name)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        # env = wrap_gym(env, rescale_actions=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        # env.set_seed(FLAGS.seed)
        test_env = safety_gymnasium.make(env_name)
        eval_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(test_env)
        # eval_env = wrap_gym(eval_env, rescale_actions=False)
        # eval_env.set_seed(FLAGS.seed + 42)
        # print("Max episode steps:", env._max_episode_steps)
        kwargs = dict(FLAGS.config)
        agent = SACLearner.create(FLAGS.seed, env.observation_space, env.action_space, **kwargs)

        chkpt_dir = 'saved_lynn_'+env_name+'_'+str(FLAGS.seed)+'/checkpoints'
        os.makedirs(chkpt_dir, exist_ok=True)
        buffer_dir = 'saved_lynn_'+env_name+'_'+str(FLAGS.seed)+'/buffers'

        last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

        if last_checkpoint is None:
            start_i = 0
            replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                         FLAGS.max_steps)
            replay_buffer.seed(FLAGS.seed)
        else:
            start_i = int(last_checkpoint.split('_')[-1])

            agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

            with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
                replay_buffer = pickle.load(f)

        observation, info = env.reset(seed=FLAGS.seed)
        observation_train.append(observation)
        observation_ly.append(observation)
        cost.append(0.)
        observation_terminated.append(False)
        x_velocities.append(-0.0)
        index_all.append(0)
        # observation_done.append(done)
        for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                           smoothing=0.1,
                           disable=not FLAGS.tqdm):
        # for i in range(1000000):
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action, agent = agent.sample_actions(observation)
            # step_result = env.step(action)
            # print(step_result)
            next_observation, reward, terminated, truncated, info = env.step(action)
            reward_pre.append(reward)
            observation_next.append(next_observation)
            observation_gym.append(observation)
            cost.append(info['cost'])
            x_velocities.append(info['x_velocity'])
            observation_terminated.append(terminated)

            if i >= FLAGS.start_training:
                reward_torch = torch.tensor(reward, dtype=torch.float64).detach()
                next_observation_torch = torch.tensor(next_observation, dtype=torch.float64).detach()
                next_observation_torch = next_observation_torch.reshape(1, -1)
                with torch.no_grad():
                    value = lyapunov_nn.lyapunov_function(next_observation_torch)
                # value = lyapunov_nn.lyapunov_function(m_ly)
                # safe_error = (value > lyapunov_nn.c_max).numpy().astype(int)[0][0]
                if (value > lyapunov_nn.c_max)[0][0].numpy():
                    safe_error = ((value - lyapunov_nn.c_max) / (lyapunov_nn.max - lyapunov_nn.c_max))[0][0].numpy()
                else:
                    safe_error = 0
                # if i<= 20000:
                #     a = 0.5
                # else:
                #     a = 1
                a = 0.1
                reward = reward - a * safe_error

            if not terminated or truncated:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(
                dict(observations=observation,
                     actions=action,
                     rewards=reward,
                     masks=mask,
                     dones=terminated,
                     next_observations=next_observation))
            observation = next_observation
            observation_train.append(observation)
            observation_ly.append(observation)
            index_all.append(i+1)
            # observation_train = np.array(observation_train).reshape(2, 72)
            # if done:
            #     observation, done = env.reset(), False
            #     for k, v in info['episode'].items():
            #         decode = {'r': 'return', 'l': 'length', 't': 'time'}
            #         wandb.log({f'training/{decode[k]}': v}, step=i)
            #         if k == 'r':
            #             reward_collection_seed1234_50noise_ep.append(v)
            #             reward_done_seed1234_50noise_pure.append(i)
            if terminated or truncated:
               observation, info = env.reset()

            if i >= FLAGS.start_training:
                batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                agent, update_info = agent.update(batch, FLAGS.utd_ratio)
                #
                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        if k == 'q':
                            q.append(v)
                        if k == 'actor_loss':
                            a_loss.append(v)
                        if k == 'critic_loss':
                            c_loss.append(v)
                #         wandb.log({f'training/{k}': v}, step=i)

            if i % FLAGS.eval_interval == 0:
                if not FLAGS.real_robot:
                    eval_info = evaluate(agent,
                                         eval_env,
                                         num_episodes=FLAGS.eval_episodes, tseed=FLAGS.seed)
                    # for k, v in eval_info.items():
                    #     wandb.log({f'evaluation/{k}': v}, step=i)
                    # length_eval_pure.append(eval_info['length'])
                    reward_collection_50noise_eval.append(eval_info['return'])
                    length_eval_pure.append(eval_info['length'])
                    cost_test.append(eval_info['cost'])
                    velocity_test.append(eval_info['velocity'])
                    term_test.append(eval_info['term'])

                checkpoints.save_checkpoint(chkpt_dir,
                                            agent,
                                            step=i + 1,
                                            keep=20,
                                            overwrite=True)

                try:
                    shutil.rmtree(buffer_dir)
                except:
                    pass

                os.makedirs(buffer_dir, exist_ok=True)
                with open(os.path.join(buffer_dir, f'buffer_{i + 1}'), 'wb') as f:
                    pickle.dump(replay_buffer, f)

            if i == 999:
                torch.set_num_threads(16)
                states_all = np.array(observation_ly).astype(np.float64).reshape(-1, env.observation_space.shape[0])
                index_ly = np.array(index_all).astype(np.float64).reshape(-1, 1)
                # cost_ly = np.array(cost)
                cost_ly = np.array(cost).reshape(1, -1)
                term = np.array(observation_terminated).reshape(1, -1)
                x_ly = states_all[:-1, :]
                x_ly = np.hstack((x_ly, index_ly[:-1]))
                y_ly = states_all[1:, :]
                y_ly = np.hstack((y_ly, index_ly[1:]))
                safety_standars = np.zeros(x_ly.shape[0])
                x_velocities_ly = np.array(x_velocities).reshape(-1, 1)
                for i in range(x_ly.shape[0]):
                    # roll, pitch, _ = np.rad2deg(quat_to_euler(states_all[i, 36:40]))
                    # rollandpitch[i] = np.logical_and(np.abs(roll) < 30, np.abs(pitch) < 30, not observation_done_early[i])
                    safety_standars[i] = (cost_ly[0][i] == 0.) and (term[0][i] == np.bool_(False)) and (x_velocities_ly[i][0] < antVelocityThreshold)
                initial_safe_set = safety_standars

                def policy():
                    return True

                states_dim = x_ly.shape[1]
                lyapunov_function, L_v = lyapunov_neural_network(env.observation_space.shape[0])
                tau = 0
                L_dyn = 1.3
                dynamics = Dynamics_NN(x_ly, y_ly)
                horizon = 5

                states = x_ly

                lyapunov_nn = safe_learning.Lyapunov(states, states_all, lyapunov_function, dynamics, L_dyn, L_v, tau, policy,
                                                     initial_safe_set)
                lyapunov_states = torch.tensor(lyapunov_nn.states, dtype=torch.float64).detach()
                lyapunov_nn.values = lyapunov_nn.lyapunov_function(lyapunov_states[:, :-1]).detach()
                print("Max value:", torch.max(lyapunov_nn.values))
                print("Min value:", torch.min(lyapunov_nn.values))
                lyapunov_nn.update_c()
                lyapunov_train(lyapunov_nn, FLAGS.seed, cost_ly, term)

            if i % 2000 ==0 and i > 1000 and i<= 50000:
                # torch.set_num_threads(16)
                states_all = np.array(observation_ly).astype(np.float64).reshape(-1, env.observation_space.shape[0])
                index_ly = np.array(index_all).astype(np.float64).reshape(-1, 1)
                # cost_ly = np.array(cost)
                cost_ly = np.array(cost).reshape(1, -1)
                term = np.array(observation_terminated).reshape(1, -1)
                x_ly = states_all[:-1, :]
                x_ly = np.hstack((x_ly, index_ly[:-1]))
                y_ly = states_all[1:, :]
                y_ly = np.hstack((y_ly, index_ly[1:]))
                safety_standars = np.zeros(x_ly.shape[0])
                x_velocities_ly = np.array(x_velocities).reshape(-1, 1)
                for i in range(x_ly.shape[0]):
                    # roll, pitch, _ = np.rad2deg(quat_to_euler(states_all[i, 36:40]))
                    # rollandpitch[i] = np.logical_and(np.abs(roll) < 30, np.abs(pitch) < 30, not observation_done_early[i])
                    safety_standars[i] = (cost_ly[0][i] == 0.) and (term[0][i] == np.bool_(False)) and (x_velocities_ly[i][0] < antVelocityThreshold)
                initial_safe_set = safety_standars

                def policy():
                    return True

                states_dim = x_ly.shape[1]
                lyapunov_function, L_v = lyapunov_neural_network(env.observation_space.shape[0])
                tau = 0
                L_dyn = 1.3
                dynamics = Dynamics_NN(x_ly, y_ly)
                horizon = 5
                # for i in range(safety_standars.shape[0]):
                #     for k in range(horizon):
                #         roll, pitch, _ = np.rad2deg(tr.quat_to_euler(states_all[i+k, 36:40]))
                #         if (np.abs(roll) > 30) | (np.abs(pitch) > 30) | (observation_done_all[i]):
                #             safety_standars[i] = False
                #             break

                states = x_ly

                lyapunov_nn = safe_learning.Lyapunov(states, states_all, lyapunov_function, dynamics, L_dyn, L_v, tau,
                                                     policy,
                                                     initial_safe_set)
                lyapunov_states = torch.tensor(lyapunov_nn.states, dtype=torch.float64).detach()
                lyapunov_nn.values = lyapunov_nn.lyapunov_function(lyapunov_states[:, :-1]).detach()
                print("Max value:", torch.max(lyapunov_nn.values))
                print("Min value:", torch.min(lyapunov_nn.values))
                lyapunov_nn.update_c()
                lyapunov_train(lyapunov_nn, FLAGS.seed, cost_ly, term)
                for index in range(1000, len(replay_buffer)):
                    # 修改 reward，例如将 reward 翻倍
                    reward_pre_temp = np.array(reward_pre)
                    next_observation_torch_temp = torch.tensor(replay_buffer.dataset_dict["next_observations"][i], dtype=torch.float64).detach()
                    next_observation_torch_temp = next_observation_torch_temp.reshape(1, -1)
                    with torch.no_grad():
                        value_temp = lyapunov_nn.lyapunov_function(next_observation_torch_temp)
                    # value = lyapunov_nn.lyapunov_function(m_ly)
                    # safe_error = (value > lyapunov_nn.c_max).numpy().astype(int)[0][0]
                    if (value_temp > lyapunov_nn.c_max)[0][0].numpy():
                        safe_error_temp = ((value_temp - lyapunov_nn.c_max) / (lyapunov_nn.max - lyapunov_nn.c_max))[0][0].numpy()
                    else:
                        safe_error_temp = 0
                    # if i<= 20000:
                    #     a = 0.01
                    # else:
                    #     a = 0.1
                    a = 0.1
                    reward_temp = reward_pre_temp[i] - a * safe_error_temp
                    replay_buffer.dataset_dict["rewards"][i] = reward_temp

        savedir = 'lynn_'+env_name+'_'+str(FLAGS.seed)
        torch.save(lyapunov_nn.lyapunov_function.state_dict(), savedir)
        with open('./ant_cost/reward_collection_ly_seed'+str(FLAGS.seed)+'_eval.pkl', 'wb') as f:
            pickle.dump(reward_collection_50noise_eval, f)
        # with open('length_eval_pure_seed'+str(FLAGS.seed)+'.pkl', 'wb') as f:
        #     pickle.dump(length_eval_pure, f)
        with open('./ant_cost/cost_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(cost, f)
        with open('./ant_cost/terminated_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(observation_terminated, f)
        with open('./ant_cost/actor_loss_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(a_loss, f)
        with open('./ant_cost/critic_loss_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(c_loss, f)
        with open('./ant_cost/q_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(q, f)
        with open('./ant_cost/velocity_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(x_velocities, f)
        with open('./ant_cost/observation_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(observation_gym, f)
        with open('./ant_cost/cost_test_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(cost_test, f)
        with open('./ant_cost/velocity_test_ly_train_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(velocity_test, f)
        with open('./ant_cost/terminated_test_ly_train_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(term_test, f)
        with open('./ant_cost/length_ly_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(length_eval_pure, f)
        # with open('./reward_velocity_walker/cost_1M'+str(FLAGS.seed)+'.pkl', 'wb') as f:
        #     pickle.dump(cost, f)
        env.close()

if __name__ == '__main__':
    app.run(main)
