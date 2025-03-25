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
import time
from matplotlib import pyplot as plt
FLAGS = flags.FLAGS


# flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(3e5), 'Number of training steps.')
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
    seed_list = [0, 1, 3, 5, 7, 1234, 2560, 5678, 42, 100]
    # seed_list = [5]
    for seed_new in seed_list:
        FLAGS.seed = seed_new
        np.random.seed(FLAGS.seed)
        observation_ly = []
        observation_gym = []
        action_gym = []
        observation_train = []
        observation_next = []
        cost = []
        cost_test = []
        index_all = []
        action_train = []
        observation_terminated = []
        observation_with_done = []
        x_velocities = []
        velocity_test = []
        terminated_train = []
        terminated_test = []
        length_test = []
        reward_collection_50noise_eval = []
        a_loss = []
        c_loss = []
        q = []
        action_train = []
        length_eval_pure = []
        reward_done_50noise_eval_pure = []

        env = safety_gymnasium.make("SafetyWalker2dVelocity-v1")
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        # env = gym.wrappers.Monitor(env, './videos/'+str(time()), video_callable=lambda x: True, force=True)
        # env = wrap_gym(env, rescale_actions=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        # env.set_seed(FLAGS.seed)
        test_env = safety_gymnasium.make("SafetyWalker2dVelocity-v1")
        eval_env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(test_env)
        kwargs = dict(FLAGS.config)
        agent = SACLearner.create(FLAGS.seed, env.observation_space, env.action_space, **kwargs)
        saved_path = 'saved_walker_cost'+str(FLAGS.seed)
        chkpt_dir = saved_path+'/checkpoints'
        os.makedirs(chkpt_dir, exist_ok=True)
        buffer_dir = saved_path+'/buffers'

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
        for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                           smoothing=0.1,
                           disable=not FLAGS.tqdm):
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action, agent = agent.sample_actions(observation)
            # step_result = env.step(action)
            # print(step_result)
            next_observation, reward, terminated, truncated, info = env.step(action)
            observation_next.append(next_observation)
            observation_gym.append(observation)
            cost.append(info['cost'])
            x_velocities.append(info['x_velocity'])
            observation_terminated.append(terminated)
            action_train.append(action)


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
            if terminated or truncated:
               observation, info = env.reset(seed=FLAGS.seed)

            if i >= FLAGS.start_training:
                batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                agent, update_info = agent.update(batch, FLAGS.utd_ratio)
                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        if k == 'actor_loss':
                            a_loss.append(v)
                        if k == 'critic_loss':
                            c_loss.append(v)
                        if k == 'q':
                            q.append(v)
                #
                # if i % FLAGS.log_interval == 0:
                #     for k, v in update_info.items():
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
                    velocity_test.append(eval_info['velocity'])
                    terminated_test.append(eval_info['term'])
                    cost_test.append(eval_info['cost'])
                    length_test.append(eval_info['length'])


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



        with open('./walker_cost/reward_collection_1M_seed'+str(FLAGS.seed)+'_eval.pkl', 'wb') as f:
            pickle.dump(reward_collection_50noise_eval, f)
        # with open('length_eval_pure_seed'+str(FLAGS.seed)+'.pkl', 'wb') as f:
        #     pickle.dump(length_eval_pure, f)
        with open('./walker_cost/terminated_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(observation_terminated, f)
        with open('./walker_cost/cost_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(cost, f)
        with open('./walker_cost/cost_test_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(cost_test, f)
        with open('./walker_cost/actor_loss_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(a_loss, f)
        with open('./walker_cost/critic_loss_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(c_loss, f)
        with open('./walker_cost/q_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(q, f)
        with open('./walker_cost/action_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(action_train, f)
        with open('./walker_cost/observation_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(observation_train, f)
        with open('./walker_cost/length_test_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(length_test, f)
        with open('./walker_cost/velocity_test_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(velocity_test, f)
        with open('./walker_cost/velocity_1M_'+str(FLAGS.seed)+'.pkl', 'wb') as f:
            pickle.dump(x_velocities, f)

        # dir_path = 'saved'

        # 检查目录是否存在
        # if os.path.exists(dir_path):
        #     # 删除目录及其内容
        #     shutil.rmtree(dir_path)
        #     print(f"目录 {dir_path} 及其内容已删除")
        # else:
        #     print(f"目录 {dir_path} 不存在")

        env.close()

if __name__ == '__main__':
    app.run(main)