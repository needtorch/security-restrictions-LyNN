from __future__ import division, print_function
import imp

import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

# import tensorflow as tf
import os
import safe_learning
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
# from sklearn import preprocessing
import math
from scipy.interpolate import interp2d
from utilities import (LyapunovNetwork, balanced_class_weights,Loss)
# writer = SummaryWriter('C:/Users/admin/Desktop/runs')
torch.autograd.set_detect_anomaly(True)
import pickle
import transformations as tr

def min_max_normalization(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def scatter_heatmap(states, values, c_max, safe_set , index):
    x = states[:, 0] * 100
    y = states[:, 1] * 100
    c = values
    plt.figure(figsize=(8, 5), dpi=200)
    col = []
    for i in range(1000):
        if c[i] < c_max:
            col.append('r')
        elif safe_set[i]==True:
            col.append('g')
        else:
            col.append('b')
    # plt.scatter(x, y, c=c, s=0.4, marker='o', cmap='rainbow', vmin=np.min(c), vmax=np.max(c))
    plt.scatter(x, y, c=col, s=0.4, marker='o')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.colorbar()
    plt.savefig('C:/Users/lenovo-pc/Desktop/4_结果图/{}.png'.format(index))
    plt.show()


def plot_safe_set(function, num_states, index, seed, iteration, show=False):
    limits = np.array([[-10, 10], ] * 2)
    num_points = np.broadcast_to(num_states, len(limits))
    discrete_points = [np.linspace(low, up, n)
                       for (low, up), n in zip(limits, num_points)]
    mesh = np.meshgrid(*discrete_points, indexing='ij')  # mesh[0].shape: (num_states, num_states)
    points = np.column_stack(col.ravel() for col in mesh)
    ss = np.ones(points.shape[0]) * 0.44
    sd = np.ones(points.shape[0]) * -0.0604
    w = np.ones(points.shape[0]) * 4.08216509
    wx = np.ones(points.shape[0]) * 1.325
    wy = np.ones(points.shape[0]) * 1.5253
    en = np.ones(points.shape[0]) * 0.039
    dj = np.ones(points.shape[0]) * 0.0266
    points = np.concatenate((points[:, 0].reshape(-1, 1), points[:, 1].reshape(-1, 1), ss.reshape(-1, 1), sd.reshape(-1, 1), wx.reshape(-1, 1), wy.reshape(-1, 1), en.reshape(-1, 1), dj.reshape(-1, 1)), axis=1)
    data = function(points).numpy()
    data = data.reshape([num_states, num_states]).T
    data = np.flipud(data)
    column_labels = np.arange(-int(10), int(10), 0.1)
    row_labels = np.arange(-int(10), int(10), 0.1)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap="rainbow",
    vmin=np.nanmin(data), vmax=np.nanmax(data))
    # heatmap.cmap.set_under('black')
    bar = fig.colorbar(heatmap, extend='both')
    # put the major ticks at the middle of each cell

    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

    # want a more natural, table-like display

    # ax.invert_yaxis()
    # ax.xaxis.tick_top()
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig('F:/实验数据/heatmap_test/seed{}_iteration{}_{}.png'.format(seed, iteration, index))
    # if show:
    #     plt.show()


def polar_coordinates(fuction, index, seed, iteration, show=False):
    ss = np.linspace(0, 4, 10)
    sd = np.linspace(-np.pi, np.pi, 50)
    # sd = np.linspace(0, 2 * np.pi, 50)
    ws = 8
    # wd = np.linspace(-np.pi, np.pi, 20)
    wd = np.linspace(0, 2 * np.pi, 20)
    eng = 0.039
    rra = 0.027

    # mean = np.mean(x, axis=0)
    # std = np.std(x, axis=0)
    values = np.zeros((ss.shape[0], sd.shape[0]))
    wwww = []
    test = np.zeros(wd.shape[0])
    # for index_wd in range(wd.shape[0]):
    #     wwww.append([ws*np.sin(wd[index_wd]), ws*np.cos(wd[index_wd])])
    #     input_array = [ss[-1]*np.sin(sd[0]), ss[-1]*np.cos(sd[0]), ws*math.sin(wd[index_wd]), ws*math.cos(wd[index_wd])]
    #     input_ = np.array(input_array)
    #     input_ = np.append(input_, eng)
    #     input_ = np.append(input_, rra)
    #     test[index_wd] = function(input_)

    # wwww = np.array(wwww)
    for index_wd in range(wd.shape[0]):
        for index_ss in range(ss.shape[0]):
            for index_sd in range(sd.shape[0]):
                input_array = [ss[index_ss]*np.sin(sd[index_sd]), ss[index_ss]*np.cos(sd[index_sd]), ws*math.sin(wd[index_wd]), ws*math.cos(wd[index_wd])]
                input_ = np.array(input_array)
                # input_ = np.append(input_, eng)
                # input_ = np.append(input_, rra)
                # input_ = (input_ - mean)/std
                values[index_ss, index_sd] = fuction(input_)
        r, theta = np.meshgrid(ss, sd)
        # func = interp2d(sd, ss, values, kind='cubic')
        # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        # r_new = np.linspace(0, 2, 10)  # y

        # v_plot = func(t_new, r_new)
        # t_plot, r_plot = np.meshgrid(t_new, r_new)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(projection='polar')
        # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet')
        ax.set_rgrids([0.3, 0.6, 0.9])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        plt.grid(c='black')
        plt.colorbar(pc, shrink=.8)
        # plt.show()
        # print('over')
        plt.savefig('F:/实验数据/polar_coordinates/seed{}_iteration{}_epoch{}_angle_{}.png'.format(seed, iteration, index, (wd[index_wd]/np.pi*180)))
    print('plot over')

class Options(object):
    def __init__(self, **kwargs) -> object:
        super(Options, self).__init__()  # 子类调用父类的构造函数
        self.__dict__.update(kwargs)  # args表示任意多个无名参数，返回一个tuple；kwargs表示关键字参数，返回一个dict。


OPTIONS = Options(np_dtype              = safe_learning.config.np_dtype,  # return numpy dtype
                  py_dtype              = safe_learning.config.dtype,  # tf.float64
                  eps                   = 1e-8,                            # numerical tolerance 决定接近答案的结果是否被当作正确
                  saturate              = True,                            # apply saturation constraints to the control input 用饱和度限制控制输入
                  use_zero_threshold    = True,                            # assume the discretization is infinitely fine (i.e., tau = 0) limit分割区间大小趋近于0
                  pre_train             = True,                            # pre-train the neural network to match a given candidate in a supervised approach
                  dpi                   = 150,
                  num_cores             = 4,
                  num_sockets           = 1,
                  tf_checkpoint_path    = "./tmp/lyapunov_function_learning.ckpt")

# mkl设置，intel优化
os.environ["KMP_BLOCKTIME"]    = str(0)  # 设置线程在睡眠之前完成并行区域执行后应该等待的时间（以毫秒为单位）
os.environ["KMP_SETTINGS"]     = str(1)  # 在程序执行期间启用（true）或禁用（false）打印OpenMP *运行时库环境变量。
os.environ["KMP_AFFINITY"]     = 'granularity=fine,noverbose,compact,1,0'  # 启用运行时库将线程绑定到物理处理单元。
os.environ["OMP_NUM_THREADS"]  = str(OPTIONS.num_cores)  # 指定要使用的线程数。

# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads  = OPTIONS.num_cores,  # 可以并行运算的线程数
#                                 inter_op_parallelism_threads  = OPTIONS.num_sockets,  # 建议将其设置为等于套接字数量。
#                                 allow_soft_placement          = False,  # True：不满足要求自动分配
#                                 device_count                  = {'CPU': OPTIONS.num_cores})


# Set random seed to reproduce results
seed = 1
torch.random.manual_seed(seed)
np.random.seed(seed)


def lyapunov_neural_network(state_dims):
    """
    定义神经网络，李普希兹常数
    states_dim:传入多少个维度
    """
    layer_dims = [256, 256, 256]
    # layer_dims = [64, 64, 64, 64, 64]
    activations = [torch.tanh, torch.tanh, torch.tanh]
    lyapunov_function = LyapunovNetwork(state_dims, layer_dims, activations, OPTIONS.eps)

    # Approximate local Lipschitz constants with gradients
    # grad_lyapunov_function = lambda x: tf.gradients(lyapunov_function(x), x)[0]  # 对第一个参数的梯度

    grad_lyapunov_function = lambda x: torch.autograd.grad(outputs=lyapunov_function(x), inputs=x, grad_outputs=torch.ones_like(lyapunov_function(x)), create_graph=True)[0]

    # L_v = lambda x: tf.norm(grad_lyapunov_function(x), ord=1, axis=1, keepdims=True)  # 变成列向量，保持维度

    L_v = lambda x: torch.norm(grad_lyapunov_function(x), p=1, dim=1, keepdims=True)

    return lyapunov_function, L_v


def lyapunov_train(lyapunov_nn: safe_learning.Lyapunov, seed, cost, term):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    states = lyapunov_nn.states
    dynamics = lyapunov_nn.dynamics
    policy = lyapunov_nn.policy
    print('Lyapunov Training ...')
    value_collection = []
    roa_estimate = np.copy(lyapunov_nn.safe_set)
    c_max = [lyapunov_nn.c_max.detach().numpy(), ]
    # Training hyperparameters
    train_batch_size = 512
    outer_iters = 20
    inner_iters = 10
    horizon = 5
    test_size = int(train_batch_size)
    level_multiplier = 1.2
    batch_size = train_batch_size
    learning_rate_init = 3e-4
    learning_rate_20 = 3e-4
    learning_rate_30 = 3e-4
    learning_rate_40 = 3e-4
    # states_stand = torch.tensor(lyapunov_nn.states, dtype=torch.float64).detach()
    # x_ly_mean = torch.mean(states_stand, dim=0)
    # x_ly_mean = torch.tensor(x_ly_mean, dtype=torch.float64).detach()
    # x_ly_std = torch.std(states_stand, dim=0)
    # x_ly_std = torch.tensor(x_ly_std, dtype=torch.float64).detach()
    # data_min = states_stand.min(axis=0)[0]
    # data_min = torch.tensor(data_min, dtype=torch.float64).detach()
    # data_max = states_stand.max(axis=0)[0]
    # data_max = torch.tensor(data_max, dtype=torch.float64).detach()
    # epsilon = torch.tensor(1e-8, dtype=torch.float64)
    # ranges = data_max - data_min + epsilon
    # ranges = torch.tensor(ranges, dtype=torch.float64).detach()
    # learning_rate = 7e-2
    lagrange_multiplier = 100
    loss = Loss(lagrange_multiplier, OPTIONS.eps)
    # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
    # Train
    print('Current metrics ...')
    # c = lyapunov_nn.c_max.numpy()
    c = lyapunov_nn.c_max.detach().numpy()
    # num_safe = lyapunov_nn.safe_set.sum()
    print('Safe level (c_k): {}'.format(c))
    print('')
    optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate_init)
    time.sleep(0.5)
    safe_fraction = []
    losses = []
    c_plot = []
    for i in range(outer_iters):
        if i == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_20
        if i == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_30

        if i == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_40
        index_outer_iters = i
        print('Iteration (k): {}'.format(len(c_max)))
        print('ROAESTIMATE: {}'.format(roa_estimate.sum()))
        time.sleep(0.5)
        # Identify the "gap" states, i.e., those between V(c_k) and V(a * c_k) for a > 1
        c = lyapunov_nn.c_max

        # idx_decrease = ((lyapunov_nn.lyapunov_function(torch.tensor(dynamics.build_evaluation(lyapunov_nn.states))) -
        #                  lyapunov_nn.lyapunov_function(torch.tensor(lyapunov_nn.states))) < 0).numpy().ravel()
        #
        # # idx_decrease_all = np.zeros(lyapunov_nn.values.shape[0],dtype=bool)
        # # for i in range(lyapunov_nn.values.shape[0]):
        # #     if idx_decrease[i]:
        # #         idx_decrease_all[i] = True
        # decrease_region = lyapunov_nn.values.clone()
        # for i in range(lyapunov_nn.values.shape[0]):
        #     if not idx_decrease[i]:
        #         decrease_region[i] = 1000
        #
        # idx_small = (decrease_region <= c).numpy().ravel()
        # idx_big = (decrease_region <= level_multiplier * c).numpy().ravel()
        # idx_gap = np.logical_and(idx_big, ~idx_small)

        idx_small = (lyapunov_nn.values <= c).numpy().ravel()
        idx_big = (lyapunov_nn.values <= level_multiplier * c).numpy().ravel()
        idx_gap = np.logical_and(idx_big, ~idx_small)
        # # 判断gap区域的点
        gap_states = states[idx_gap]
        if gap_states.any():
            for _ in range(horizon):
                # action_gap = policy(gap_states)
                # gap_states = dynamics(gap_states, action_gap)
                gap_states = dynamics.build_evaluation(gap_states)
            # dist = []
            # for i in range(gap_states.shape[0]):
            #     #TODO 考虑位置？需更改
            #     dist.append(math.sqrt(gap_states[i][0] ** 2 + gap_states[i][1] ** 2))
            with torch.no_grad():
                gap_future_values = lyapunov_nn.lyapunov_function((torch.tensor(gap_states[:, :-1], dtype=torch.float64))).detach()
            # roa_estimate[idx_gap] |= np.array(dist) <= 7
            # roa_estimate[idx_small] |= idx_small  # 或运算
            roa_estimate = np.logical_or(roa_estimate, idx_small)
            roa_estimate[idx_gap] |= (gap_future_values <= c).numpy().ravel()  # 或运算
        # for index in range(roa_estimate.shape[0]-25):
        #     for k in range(25):
        #         roll, pitch, _ = np.rad2deg(tr.quat_to_euler(lyapunov_nn.states_all[index+k, 36:40]))
        #         if (np.abs(roll) > 30) | (np.abs(pitch) > 30):
        #             roa_estimate[index] = False
        #             break

        target_idx = np.logical_or(idx_big, roa_estimate)
        target_set = states[target_idx]
        print('target_set_size:{}'.format(target_set.shape[0]))
        # # 整个ROA进行分类为+1，0，后续梯度下降会换成+1，-1
        target_labels = roa_estimate[target_idx].astype(OPTIONS.np_dtype).reshape([-1, 1])
        idx_range = target_set.shape[0]
        # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
        # SGD for classification
        for _ in tqdm(range(inner_iters)):
            # Training step
            # idx_batch_eval = tf.compat.v1.random_uniform([batch_size, ], 0, idx_range, dtype=tf.int32,
            #                                              name='batch_sample')
            torch.set_printoptions(precision=8)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                print(name, param.data)
            optimizer.zero_grad()  # 重置梯度
            # for name, param in lyapunov_nn.lyapunov_function.named_parameters():
            #     print(name, param.data)
            idx_batch_eval = torch.randint(low=0, high=idx_range, size=(batch_size,),
                                           dtype=torch.int32,requires_grad=False)
            train_states = target_set[idx_batch_eval]
            train_state_torch = torch.tensor(train_states[:, :-1],dtype=torch.float64)

            # writer.add_graph(lyapunov_nn.lyapunov_function,torch.tensor(train_state_torch, dtype=torch.float64))
            # writer.close()
            # train_states = states
            train_level = c
            train_roa_labels = target_labels[idx_batch_eval]  # 确定他们的标签
            # train_roa_labels = lyapunov_nn.safe_set.astype(OPTIONS.np_dtype).reshape([-1, 1])
            class_weights, class_counts = balanced_class_weights(train_roa_labels.astype(bool), scale_by_total=True)



            # 计算loss，min
            # def loss(value,nn_train_value):
            #     class_labels = 2 * train_roa_labels - 1
            #     decision_distance = train_level - value
            #     class_labels_torch = torch.tensor(class_labels, dtype=torch.float64)
            #     class_weights_torch = torch.tensor(class_weights, dtype=torch.float64)
            #     # 创建一个300行1列的全0矩阵，数据类型为float64
            #     zero_matrix = torch.zeros((300, 1), dtype=torch.float64,requires_grad=False)
            #
            #     classifier_loss = class_weights_torch * torch.maximum(- class_labels_torch * decision_distance,  zero_matrix)
            #     classifier_loss_ = torch.tensor(classifier_loss, dtype=torch.float64, requires_grad=True)
            #     tf_dv_nn_train = nn_train_value - value
            #     stop_gra=value.detach()
            #     train_roa_labels_torch = torch.tensor(train_roa_labels, dtype=torch.float64)
            #     decrease_loss = train_roa_labels_torch * torch.maximum(tf_dv_nn_train, torch.zeros([300,1])) / (stop_gra + OPTIONS.eps)
            #     decrease_loss_ = torch.tensor(decrease_loss, dtype=torch.float64, requires_grad=True)
            #     res = (classifier_loss_ + lagrange_multiplier * decrease_loss_).mean()
            #     return res

            # print(lyapunov_nn.lyapunov_function)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            # batch_states = train_states.clone().detach()
            # train_state_torch = (train_state_torch - x_ly_mean) / x_ly_std  #标准化
            # train_state_torch = (train_state_torch - data_min) / ranges  # 归一化
            values = lyapunov_nn.lyapunov_function(train_state_torch)
            value_collection.append(torch.max(values).detach().numpy())
            print('values:{}'.format(values))
            # print(id(values))
            next_states = dynamics.build_evaluation(train_states)
            next_states_torch = torch.tensor(next_states[:, :-1],dtype=torch.float64)
            # next_states_torch = (next_states_torch - x_ly_mean) / x_ly_std
            # next_states_torch = (next_states_torch - data_min) / ranges    # 归一化
            nn_train_value = lyapunov_nn.lyapunov_function(next_states_torch)
            # print('nn_train_value:{}'.format(nn_train_value))
            # print(id(nn_train_value))
            # values = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64, requires_grad=False)
            # loss = Loss(lagrange_multiplier,OPTIONS.eps)
            objective = loss(values, nn_train_value, train_level, train_roa_labels, class_weights)
            # criterion = nn.MSELoss()
            # # values = torch.tensor(values, dtype=torch.float64,requires_grad=True)
            # # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64,requires_grad=True)
            # objective = criterion(values, train_roa_labels)
            print(' loss: {}'.format(objective))
            # 在训练循环中
            loss_value = objective.item()
            losses.append(loss_value)

             # 绘制图形
            # plt.plot(losses)
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.draw()
            # plt.pause(0.1)  # 暂停一会，以便图形更新


            objective.backward()  # 自动计算梯度
            # 检查是否可以进行第二次backward
            # try:
            #     outputs = lyapunov_nn.lyapunov_function.forward(train_states)
            #     lyapunov_nn.lyapunov_function.forward(train_states)
            #     loss = loss(outputs, nn_train_value)
            #     loss.backward()  # 第二次backward
            # except RuntimeError as e:
            #     print("捕获到错误：", e)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            optimizer.step()  # 应用梯度更新



        # Update Lyapunov values and ROA estimate, based on new parameter values
        with torch.no_grad():
            lyapunov_nn.values = lyapunov_nn.lyapunov_function(torch.tensor(states[:, :-1], dtype=torch.float64)).detach()
        lyapunov_nn.update_c()

        # ss = np.linspace(0, 7.5, 10)
        # sd = np.linspace(-np.pi, np.pi, 50)
        # ws = 8
        # wd = np.linspace(0, 2 * np.pi, 20)
        # values = np.zeros((ss.shape[0], sd.shape[0]))
        # for index_ss in range(ss.shape[0]):
        #     for index_sd in range(sd.shape[0]):
        #         input_array = [ss[index_ss] * np.sin(sd[index_sd]), ss[index_ss] * np.cos(sd[index_sd]),
        #                        ws * math.sin(wd[0]), ws * math.cos(wd[0])]
        #         input_ = np.array(input_array).reshape(1, 4)
        #         input_ = torch.tensor(input_, dtype=torch.float64)
        #         values[index_ss, index_sd] = lyapunov_nn.lyapunov_function(input_).detach()
        # r, theta = np.meshgrid(ss, sd)
        # func = interp2d(sd, ss, values, kind='cubic')
        # # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        # # r_new = np.linspace(0, 2, 10)  # y
        #
        # # v_plot = func(t_new, r_new)
        # # t_plot, r_plot = np.meshgrid(t_new, r_new)
        #
        # plt.figure(figsize=(10, 10))
        # ax = plt.subplot(projection='polar')
        # # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        # # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet', vmax= max_ly, vmin=min_ly)
        # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet')
        # ax.set_rgrids([1, 3, 5, 7])
        # ax.set_theta_zero_location('N')
        # ax.set_theta_direction(-1)
        # plt.grid(c='black')
        # plt.colorbar(pc, shrink=.8)
        # # plt.savefig('heatmap2/heatmap_epoch_{}_w8_{}.png'.format(i, index_wd))
        # plt.savefig('heatmap2/heatmap_epoch_{}.png'.format(i))

        # [-0.04213739  1.3599194 - 1.9086523   0.1175271   1.1737214 - 1.6618658
        #  - 0.01790294  1.6895126 - 1.4089415   0.02065039  1.524482 - 1.0981531
        #
        #  - 0.40626395 - 1.1319396 - 0.558339    0.53220224  2.689868    2.5254738
        #  - 0.76223963 - 0.02025877  2.2770653 - 0.5148225   0.60094297  1.2247174
        #
        #  - 0.11555663  1.0826958 - 1.97015     0.17678288  1.5650104 - 0.916298
        #  - 0.11042087  1.6241096 - 1.0401481 - 0.04433194  1.5419348 - 0.9460856
        #
        #  0.90024114  0.06285354 - 0.01576584  0.43054238 - 0.14796436  1.1092571
        #  - 0.6419087   0.7881173 - 0.11366197 - 0.65463597]

        # FR_hip = np.array(-0.04213739).astype(np.float32)
        # FR_thigh = np.linspace(0.15, 1.9, 20)
        # FR_calf = np.array(-1.9086523).astype(np.float32)
        # FL_hip = np.array(0.1175271).astype(np.float32)
        # FL_thigh = np.array(1.1737214).astype(np.float32)
        # FL_calf = np.array(-1.6618658).astype(np.float32)
        # RR_hip = np.array( -0.01790294).astype(np.float32)
        # RR_thigh = np.array(1.6895126).astype(np.float32)
        # RR_calf = np.array(-1.4089415).astype(np.float32)
        # RL_hip = np.array(0.02065039).astype(np.float32)
        # RL_thigh = np.array(1.524482).astype(np.float32)
        # RL_calf = np.array(-1.0981531).astype(np.float32)
        #
        # FR_hip_vel = np.array(-0.40626395).astype(np.float32)
        # FR_thigh_vel = np.linspace(-6.5, 5.6, 20)
        # FR_calf_vel = np.array(-0.558339).astype(np.float32)
        # FL_hip_vel = np.array(0.53220224).astype(np.float32)
        # FL_thigh_vel = np.array(2.689868).astype(np.float32)
        # FL_calf_vel = np.array(2.5254738).astype(np.float32)
        # RR_hip_vel = np.array( -0.76223963).astype(np.float32)
        # RR_thigh_vel = np.array(-0.02025877).astype(np.float32)
        # RR_calf_vel = np.array(2.2770653).astype(np.float32)
        # RL_hip_vel = np.array(-0.5148225).astype(np.float32)
        # RL_thigh_vel = np.array(0.60094297).astype(np.float32)
        # RL_calf_vel = np.array(1.2247174).astype(np.float32)
        # prev_action = np.array([-0.11555663,  1.0826958, -1.97015, 0.17678288,  1.5650104, -0.916298,
        #  -0.11042087,  1.6241096, -1.0401481, -0.04433194,  1.5419348, -0.9460856]).astype(np.float32)
        # values = np.zeros((FR_thigh.shape[0],  FR_thigh_vel.shape[0]))
        # for index_fr in range(FR_thigh.shape[0]):
        #     for index_frv in range(FR_thigh_vel.shape[0]):
        #         input_array = [FR_hip, FR_thigh[index_fr], FR_calf, FL_hip, FL_thigh,  FL_calf,
        #                        RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf,
        #                        FR_hip_vel, FR_thigh_vel[index_frv], FR_calf_vel,
        #                        FL_hip_vel, FL_thigh_vel, FL_calf_vel,
        #                        RR_hip_vel, RR_thigh_vel, RR_calf_vel,
        #                        RL_hip_vel, RL_thigh_vel, RL_calf_vel,
        #                        -0.11555663,  1.0826958, -1.97015, 0.17678288,  1.5650104, -0.916298,
        #                        -0.11042087,  1.6241096, -1.0401481, -0.04433194,  1.5419348, -0.9460856]
        #         input_ = np.array(input_array).reshape(1, 36)
        #         input_ = torch.tensor(input_, dtype=torch.float64).detach()
        #         input_ = (input_ - x_ly_mean) / x_ly_std
        #         # input_ = (input_ - data_min) / ranges
        #         values[index_fr, index_frv] = lyapunov_nn.lyapunov_function(input_).detach()
        # r, theta = np.meshgrid(abs(FR_thigh_vel), FR_thigh)
        # # func = interp2d(FR_thigh, FR_thigh_vel, values, kind='cubic')
        # # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        # # r_new = np.linspace(0, 2, 10)  # y
        #
        # # v_plot = func(t_new, r_new)
        # # t_plot, r_plot = np.meshgrid(t_new, r_new)
        #
        # plt.figure(figsize=(10, 10))
        # ax = plt.subplot(projection='polar')
        # # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        # # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet', vmax= max_ly, vmin=min_ly)
        # pc = ax.pcolor(theta, r, values, shading='auto', cmap='jet')
        # ax.set_rgrids([1.3, 2.6, 3.9, 5.2])
        # ax.set_theta_zero_location('N')
        # ax.set_theta_direction(-1)
        # plt.grid(c='black')
        # plt.colorbar(pc, shrink=.8)
        # # plt.savefig('heatmap2/heatmap_epoch_{}_w8_{}.png'.format(i, index_wd))
        # plt.savefig('heatmap_3w_step/heatmap_epoch_{}.png'.format(i))





        # ss = np.linspace(0, 7.5, 10)
        # sd = np.linspace(-np.pi, np.pi, 50)
        # ws = 8
        # wd = np.linspace(0, 2 * np.pi, 20)
        # values = np.zeros((ss.shape[0], sd.shape[0]))
        # for index_ss in range(ss.shape[0]):
        #     for index_sd in range(sd.shape[0]):
        #         input_array = [ss[index_ss] * np.sin(sd[index_sd]), ss[index_ss] * np.cos(sd[index_sd]),
        #                        ws * math.sin(wd[0]), ws * math.cos(wd[0])]
        #         input_ = np.array(input_array).reshape(1, 4)
        #         input_ = torch.tensor(input_, dtype=torch.float64)
        #         values[index_ss, index_sd] = lyapunov_nn.lyapunov_function(input_).detach()
        # r, theta = np.meshgrid(ss, sd)
        # func = interp2d(sd, ss, values, kind='cubic')
        # # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        # # r_new = np.linspace(0, 2, 10)  # y
        #
        # # v_plot = func(t_new, r_new)
        # # t_plot, r_plot = np.meshgrid(t_new, r_new)
        #
        # plt.figure(figsize=(10, 10))
        # ax = plt.subplot(projection='polar')
        # # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        # # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet', vmax= max_ly, vmin=min_ly)
        # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet')
        # ax.set_rgrids([1, 3, 5, 7])
        # ax.set_theta_zero_location('N')
        # ax.set_theta_direction(-1)
        # plt.grid(c='black')
        # plt.colorbar(pc, shrink=.8)
        # # plt.savefig('heatmap2/heatmap_epoch_{}_w8_{}.png'.format(i, index_wd))
        # plt.savefig('heatmap_2w_step/heatmap_epoch_{}.png'.format(i))

        # ss = np.linspace(0, 7.5, 10)
        # sd = np.linspace(-np.pi, np.pi, 50)
        # ws = 8
        # wd = np.linspace(0, 2 * np.pi, 20)
        # values = np.zeros((ss.shape[0], sd.shape[0]))

        # for index_wd in range(wd.shape[0]):
        #     for index_ss in range(ss.shape[0]):
        #         for index_sd in range(sd.shape[0]):
        #             input_array = [ss[index_ss]*np.sin(sd[index_sd]), ss[index_ss]*np.cos(sd[index_sd]), ws*math.sin(wd[index_wd]), ws*math.cos(wd[index_wd])]
        #             input_ = np.array(input_array)
        #             values[index_ss, index_sd] = lyapunov_nn.lyapunov_function(input_)
        #     r, theta = np.meshgrid(ss, sd)
        #     func = interp2d(sd, ss, values, kind='cubic')
        #     # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        #     # r_new = np.linspace(0, 2, 10)  # y

        #     # v_plot = func(t_new, r_new)
        #     # t_plot, r_plot = np.meshgrid(t_new, r_new)

        #     plt.figure(figsize=(10, 10))
        #     ax = plt.subplot(projection='polar')
        #     # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        #     # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet', vmax= max_ly, vmin=min_ly)
        #     pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet')
        #     ax.set_rgrids([1, 3, 5, 7])
        #     ax.set_theta_zero_location('N')
        #     ax.set_theta_direction(-1)
        #     plt.grid(c='black')
        #     plt.colorbar(pc, shrink=.8)
        #     plt.savefig('heatmap2/heatmap_epoch_{}_w8_{}.png'.format(i, index_wd))
        roa_estimate |= lyapunov_nn.safe_set.detach().numpy()
        # lyapunov_nn.safe_set |= roa_estimate
        c_max.append(lyapunov_nn.c_max.detach().numpy())
        c_plot.append(lyapunov_nn.c_max.item())
        # plt.plot(c_plot)
        # plt.xlabel('Epoch')
        # plt.ylabel('c')
        # plt.show()
        # plt.pause(0.1)
        # scatter_heatmap(states, lyapunov_nn.values.numpy(), c_max[-1], initial_safe_set, i)
        # plot_safe_set(lyapunov_nn.lyapunov_function, 200, i, seed, iteration)
        # polar_coordinates(lyapunov_nn.lyapunov_function, index_outer_iters, seed, iteration)
        print('Current safe level (c_k): {}'.format(c_max[-1]))
        print('Safe set size_C : {}'.format((lyapunov_nn.values.detach().numpy() < c_max[-1]).sum()))
        print('Safe set size : {}'.format((lyapunov_nn.safe_set.detach().sum())))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('./ly_loss/loss.jpg')
        # plt.pause(0.1)

        with open('c_max.pkl', 'wb') as f:
            pickle.dump(c_max, f)

        with open('roa_estimate.pkl', 'wb') as f:
            pickle.dump(roa_estimate.sum(), f)

        with open('Safe_set.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set, f)

        with open('Safe_set_size.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set.detach().sum(), f)

        with open('value_collection.pkl', 'wb') as f:
            pickle.dump(value_collection, f)

def lyapunov_train_pre(lyapunov_nn: safe_learning.Lyapunov, seed, cost, term):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    states = lyapunov_nn.states
    dynamics = lyapunov_nn.dynamics
    policy = lyapunov_nn.policy
    print('Lyapunov Training ...')
    value_collection = []
    roa_estimate = np.copy(lyapunov_nn.safe_set)
    c_max = [lyapunov_nn.c_max.detach().numpy(), ]
    # Training hyperparameters
    train_batch_size = 512
    outer_iters = 20
    inner_iters = 10
    horizon = 5
    test_size = int(train_batch_size)
    level_multiplier = 1.2
    batch_size = train_batch_size
    learning_rate_init = 3e-4
    learning_rate_20 = 3e-4
    learning_rate_30 = 3e-4
    learning_rate_40 = 9e-5
    # states_stand = torch.tensor(lyapunov_nn.states, dtype=torch.float64).detach()
    # x_ly_mean = torch.mean(states_stand, dim=0)
    # x_ly_mean = torch.tensor(x_ly_mean, dtype=torch.float64).detach()
    # x_ly_std = torch.std(states_stand, dim=0)
    # x_ly_std = torch.tensor(x_ly_std, dtype=torch.float64).detach()
    # data_min = states_stand.min(axis=0)[0]
    # data_min = torch.tensor(data_min, dtype=torch.float64).detach()
    # data_max = states_stand.max(axis=0)[0]
    # data_max = torch.tensor(data_max, dtype=torch.float64).detach()
    # epsilon = torch.tensor(1e-8, dtype=torch.float64)
    # ranges = data_max - data_min + epsilon
    # ranges = torch.tensor(ranges, dtype=torch.float64).detach()
    # learning_rate = 7e-2
    lagrange_multiplier = 100
    loss = Loss(lagrange_multiplier, OPTIONS.eps)
    # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
    # Train
    print('Current metrics ...')
    # c = lyapunov_nn.c_max.numpy()
    c = lyapunov_nn.c_max.detach().numpy()
    # num_safe = lyapunov_nn.safe_set.sum()
    print('Safe level (c_k): {}'.format(c))
    print('')
    optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate_init)
    time.sleep(0.5)
    safe_fraction = []
    losses = []
    c_plot = []
    for i in range(outer_iters):
        if i == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_20
        if i == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_30

        if i == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_40
        index_outer_iters = i
        print('Iteration (k): {}'.format(len(c_max)))
        print('ROAESTIMATE: {}'.format(roa_estimate.sum()))
        time.sleep(0.5)
        # Identify the "gap" states, i.e., those between V(c_k) and V(a * c_k) for a > 1
        c = lyapunov_nn.c_max

        # idx_decrease = ((lyapunov_nn.lyapunov_function(torch.tensor(dynamics.build_evaluation(lyapunov_nn.states))) -
        #                  lyapunov_nn.lyapunov_function(torch.tensor(lyapunov_nn.states))) < 0).numpy().ravel()
        #
        # # idx_decrease_all = np.zeros(lyapunov_nn.values.shape[0],dtype=bool)
        # # for i in range(lyapunov_nn.values.shape[0]):
        # #     if idx_decrease[i]:
        # #         idx_decrease_all[i] = True
        # decrease_region = lyapunov_nn.values.clone()
        # for i in range(lyapunov_nn.values.shape[0]):
        #     if not idx_decrease[i]:
        #         decrease_region[i] = 1000
        #
        # idx_small = (decrease_region <= c).numpy().ravel()
        # idx_big = (decrease_region <= level_multiplier * c).numpy().ravel()
        # idx_gap = np.logical_and(idx_big, ~idx_small)

        idx_small = (lyapunov_nn.values <= c).numpy().ravel()
        idx_big = (lyapunov_nn.values <= level_multiplier * c).numpy().ravel()
        idx_gap = np.logical_and(idx_big, ~idx_small)
        # # 判断gap区域的点
        gap_states = states[idx_gap]
        if gap_states.any():
            for _ in range(horizon):
                # action_gap = policy(gap_states)
                # gap_states = dynamics(gap_states, action_gap)
                gap_states = dynamics.build_evaluation(gap_states)
            # dist = []
            # for i in range(gap_states.shape[0]):
            #     #TODO 考虑位置？需更改
            #     dist.append(math.sqrt(gap_states[i][0] ** 2 + gap_states[i][1] ** 2))
            with torch.no_grad():
                gap_future_values = lyapunov_nn.lyapunov_function(torch.tensor(gap_states[:, :-1], dtype=torch.float64) ).detach()
            # roa_estimate[idx_gap] |= np.array(dist) <= 7
            # roa_estimate[idx_small] |= idx_small  # 或运算
            roa_estimate = np.logical_or(roa_estimate, idx_small)
            roa_estimate[idx_gap] |= (gap_future_values <= c).numpy().ravel()  # 或运算
        # for index in range(roa_estimate.shape[0]-25):
        #     for k in range(25):
        #         roll, pitch, _ = np.rad2deg(tr.quat_to_euler(lyapunov_nn.states_all[index+k, 36:40]))
        #         if (np.abs(roll) > 30) | (np.abs(pitch) > 30):
        #             roa_estimate[index] = False
        #             break

        target_idx = np.logical_or(idx_big, roa_estimate)
        target_set = states[target_idx]
        print('target_set_size:{}'.format(target_set.shape[0]))
        # # 整个ROA进行分类为+1，0，后续梯度下降会换成+1，-1
        target_labels = roa_estimate[target_idx].astype(OPTIONS.np_dtype).reshape([-1, 1])
        idx_range = target_set.shape[0]
        # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
        # SGD for classification
        for _ in tqdm(range(inner_iters)):
            # Training step
            # idx_batch_eval = tf.compat.v1.random_uniform([batch_size, ], 0, idx_range, dtype=tf.int32,
            #                                              name='batch_sample')
            torch.set_printoptions(precision=8)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                print(name, param.data)
            optimizer.zero_grad()  # 重置梯度
            # for name, param in lyapunov_nn.lyapunov_function.named_parameters():
            #     print(name, param.data)
            idx_batch_eval = torch.randint(low=0, high=idx_range, size=(batch_size,),
                                           dtype=torch.int32,requires_grad=False)
            train_states = target_set[idx_batch_eval]
            train_state_torch = torch.tensor(train_states[:, :-1], dtype=torch.float64)

            # writer.add_graph(lyapunov_nn.lyapunov_function,torch.tensor(train_state_torch, dtype=torch.float64))
            # writer.close()
            # train_states = states
            train_level = c
            train_roa_labels = target_labels[idx_batch_eval]  # 确定他们的标签
            # train_roa_labels = lyapunov_nn.safe_set.astype(OPTIONS.np_dtype).reshape([-1, 1])
            class_weights, class_counts = balanced_class_weights(train_roa_labels.astype(bool), scale_by_total=True)



            # 计算loss，min
            # def loss(value,nn_train_value):
            #     class_labels = 2 * train_roa_labels - 1
            #     decision_distance = train_level - value
            #     class_labels_torch = torch.tensor(class_labels, dtype=torch.float64)
            #     class_weights_torch = torch.tensor(class_weights, dtype=torch.float64)
            #     # 创建一个300行1列的全0矩阵，数据类型为float64
            #     zero_matrix = torch.zeros((300, 1), dtype=torch.float64,requires_grad=False)
            #
            #     classifier_loss = class_weights_torch * torch.maximum(- class_labels_torch * decision_distance,  zero_matrix)
            #     classifier_loss_ = torch.tensor(classifier_loss, dtype=torch.float64, requires_grad=True)
            #     tf_dv_nn_train = nn_train_value - value
            #     stop_gra=value.detach()
            #     train_roa_labels_torch = torch.tensor(train_roa_labels, dtype=torch.float64)
            #     decrease_loss = train_roa_labels_torch * torch.maximum(tf_dv_nn_train, torch.zeros([300,1])) / (stop_gra + OPTIONS.eps)
            #     decrease_loss_ = torch.tensor(decrease_loss, dtype=torch.float64, requires_grad=True)
            #     res = (classifier_loss_ + lagrange_multiplier * decrease_loss_).mean()
            #     return res

            # print(lyapunov_nn.lyapunov_function)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            # batch_states = train_states.clone().detach()
            # train_state_torch = (train_state_torch - x_ly_mean) / x_ly_std  #标准化
            # train_state_torch = (train_state_torch - data_min) / ranges  # 归一化
            values = lyapunov_nn.lyapunov_function(train_state_torch)
            value_collection.append(torch.max(values).detach().numpy())
            print('values:{}'.format(values))
            # print(id(values))
            next_states = dynamics.build_evaluation(train_states)
            next_states_torch = torch.tensor(next_states[:, :-1], dtype=torch.float64)
            # next_states_torch = (next_states_torch - x_ly_mean) / x_ly_std
            # next_states_torch = (next_states_torch - data_min) / ranges    # 归一化
            nn_train_value = lyapunov_nn.lyapunov_function(next_states_torch)
            # print('nn_train_value:{}'.format(nn_train_value))
            # print(id(nn_train_value))
            # values = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64, requires_grad=False)
            # loss = Loss(lagrange_multiplier,OPTIONS.eps)
            objective = loss(values, nn_train_value, train_level, train_roa_labels, class_weights)
            # criterion = nn.MSELoss()
            # # values = torch.tensor(values, dtype=torch.float64,requires_grad=True)
            # # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64,requires_grad=True)
            # objective = criterion(values, train_roa_labels)
            print(' loss: {}'.format(objective))
            # 在训练循环中
            loss_value = objective.item()
            losses.append(loss_value)

             # 绘制图形
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.draw()
            plt.pause(0.1)  # 暂停一会，以便图形更新


            objective.backward()  # 自动计算梯度
            # 检查是否可以进行第二次backward
            # try:
            #     outputs = lyapunov_nn.lyapunov_function.forward(train_states)
            #     lyapunov_nn.lyapunov_function.forward(train_states)
            #     loss = loss(outputs, nn_train_value)
            #     loss.backward()  # 第二次backward
            # except RuntimeError as e:
            #     print("捕获到错误：", e)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            optimizer.step()  # 应用梯度更新



        # Update Lyapunov values and ROA estimate, based on new parameter values
        with torch.no_grad():
            lyapunov_nn.values = lyapunov_nn.lyapunov_function(torch.tensor(states[:, :-1], dtype=torch.float64)).detach()
        lyapunov_nn.update_c()

        # ss = np.linspace(0, 7.5, 10)
        # sd = np.linspace(-np.pi, np.pi, 50)
        # ws = 8
        # wd = np.linspace(0, 2 * np.pi, 20)
        # values = np.zeros((ss.shape[0], sd.shape[0]))
        # for index_ss in range(ss.shape[0]):
        #     for index_sd in range(sd.shape[0]):
        #         input_array = [ss[index_ss] * np.sin(sd[index_sd]), ss[index_ss] * np.cos(sd[index_sd]),
        #                        ws * math.sin(wd[0]), ws * math.cos(wd[0])]
        #         input_ = np.array(input_array).reshape(1, 4)
        #         input_ = torch.tensor(input_, dtype=torch.float64)
        #         values[index_ss, index_sd] = lyapunov_nn.lyapunov_function(input_).detach()
        # r, theta = np.meshgrid(ss, sd)
        # func = interp2d(sd, ss, values, kind='cubic')
        # # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        # # r_new = np.linspace(0, 2, 10)  # y
        #
        # # v_plot = func(t_new, r_new)
        # # t_plot, r_plot = np.meshgrid(t_new, r_new)
        #
        # plt.figure(figsize=(10, 10))
        # ax = plt.subplot(projection='polar')
        # # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        # # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet', vmax= max_ly, vmin=min_ly)
        # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet')
        # ax.set_rgrids([1, 3, 5, 7])
        # ax.set_theta_zero_location('N')
        # ax.set_theta_direction(-1)
        # plt.grid(c='black')
        # plt.colorbar(pc, shrink=.8)
        # # plt.savefig('heatmap2/heatmap_epoch_{}_w8_{}.png'.format(i, index_wd))
        # plt.savefig('heatmap2/heatmap_epoch_{}.png'.format(i))

        # [-0.04213739  1.3599194 - 1.9086523   0.1175271   1.1737214 - 1.6618658
        #  - 0.01790294  1.6895126 - 1.4089415   0.02065039  1.524482 - 1.0981531
        #
        #  - 0.40626395 - 1.1319396 - 0.558339    0.53220224  2.689868    2.5254738
        #  - 0.76223963 - 0.02025877  2.2770653 - 0.5148225   0.60094297  1.2247174
        #
        #  - 0.11555663  1.0826958 - 1.97015     0.17678288  1.5650104 - 0.916298
        #  - 0.11042087  1.6241096 - 1.0401481 - 0.04433194  1.5419348 - 0.9460856
        #
        #  0.90024114  0.06285354 - 0.01576584  0.43054238 - 0.14796436  1.1092571
        #  - 0.6419087   0.7881173 - 0.11366197 - 0.65463597]

        # FR_hip = np.array(-0.04213739).astype(np.float32)
        # FR_thigh = np.linspace(0.15, 1.9, 20)
        # FR_calf = np.array(-1.9086523).astype(np.float32)
        # FL_hip = np.array(0.1175271).astype(np.float32)
        # FL_thigh = np.array(1.1737214).astype(np.float32)
        # FL_calf = np.array(-1.6618658).astype(np.float32)
        # RR_hip = np.array( -0.01790294).astype(np.float32)
        # RR_thigh = np.array(1.6895126).astype(np.float32)
        # RR_calf = np.array(-1.4089415).astype(np.float32)
        # RL_hip = np.array(0.02065039).astype(np.float32)
        # RL_thigh = np.array(1.524482).astype(np.float32)
        # RL_calf = np.array(-1.0981531).astype(np.float32)
        #
        # FR_hip_vel = np.array(-0.40626395).astype(np.float32)
        # FR_thigh_vel = np.linspace(-6.5, 5.6, 20)
        # FR_calf_vel = np.array(-0.558339).astype(np.float32)
        # FL_hip_vel = np.array(0.53220224).astype(np.float32)
        # FL_thigh_vel = np.array(2.689868).astype(np.float32)
        # FL_calf_vel = np.array(2.5254738).astype(np.float32)
        # RR_hip_vel = np.array( -0.76223963).astype(np.float32)
        # RR_thigh_vel = np.array(-0.02025877).astype(np.float32)
        # RR_calf_vel = np.array(2.2770653).astype(np.float32)
        # RL_hip_vel = np.array(-0.5148225).astype(np.float32)
        # RL_thigh_vel = np.array(0.60094297).astype(np.float32)
        # RL_calf_vel = np.array(1.2247174).astype(np.float32)
        # prev_action = np.array([-0.11555663,  1.0826958, -1.97015, 0.17678288,  1.5650104, -0.916298,
        #  -0.11042087,  1.6241096, -1.0401481, -0.04433194,  1.5419348, -0.9460856]).astype(np.float32)
        # values = np.zeros((FR_thigh.shape[0],  FR_thigh_vel.shape[0]))
        # for index_fr in range(FR_thigh.shape[0]):
        #     for index_frv in range(FR_thigh_vel.shape[0]):
        #         input_array = [FR_hip, FR_thigh[index_fr], FR_calf, FL_hip, FL_thigh,  FL_calf,
        #                        RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf,
        #                        FR_hip_vel, FR_thigh_vel[index_frv], FR_calf_vel,
        #                        FL_hip_vel, FL_thigh_vel, FL_calf_vel,
        #                        RR_hip_vel, RR_thigh_vel, RR_calf_vel,
        #                        RL_hip_vel, RL_thigh_vel, RL_calf_vel,
        #                        -0.11555663,  1.0826958, -1.97015, 0.17678288,  1.5650104, -0.916298,
        #                        -0.11042087,  1.6241096, -1.0401481, -0.04433194,  1.5419348, -0.9460856]
        #         input_ = np.array(input_array).reshape(1, 36)
        #         input_ = torch.tensor(input_, dtype=torch.float64).detach()
        #         input_ = (input_ - x_ly_mean) / x_ly_std
        #         # input_ = (input_ - data_min) / ranges
        #         values[index_fr, index_frv] = lyapunov_nn.lyapunov_function(input_).detach()
        # r, theta = np.meshgrid(abs(FR_thigh_vel), FR_thigh)
        # # func = interp2d(FR_thigh, FR_thigh_vel, values, kind='cubic')
        # # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        # # r_new = np.linspace(0, 2, 10)  # y
        #
        # # v_plot = func(t_new, r_new)
        # # t_plot, r_plot = np.meshgrid(t_new, r_new)
        #
        # plt.figure(figsize=(10, 10))
        # ax = plt.subplot(projection='polar')
        # # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        # # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet', vmax= max_ly, vmin=min_ly)
        # pc = ax.pcolor(theta, r, values, shading='auto', cmap='jet')
        # ax.set_rgrids([1.3, 2.6, 3.9, 5.2])
        # ax.set_theta_zero_location('N')
        # ax.set_theta_direction(-1)
        # plt.grid(c='black')
        # plt.colorbar(pc, shrink=.8)
        # # plt.savefig('heatmap2/heatmap_epoch_{}_w8_{}.png'.format(i, index_wd))
        # plt.savefig('heatmap_3w_step/heatmap_epoch_{}.png'.format(i))





        # ss = np.linspace(0, 7.5, 10)
        # sd = np.linspace(-np.pi, np.pi, 50)
        # ws = 8
        # wd = np.linspace(0, 2 * np.pi, 20)
        # values = np.zeros((ss.shape[0], sd.shape[0]))
        # for index_ss in range(ss.shape[0]):
        #     for index_sd in range(sd.shape[0]):
        #         input_array = [ss[index_ss] * np.sin(sd[index_sd]), ss[index_ss] * np.cos(sd[index_sd]),
        #                        ws * math.sin(wd[0]), ws * math.cos(wd[0])]
        #         input_ = np.array(input_array).reshape(1, 4)
        #         input_ = torch.tensor(input_, dtype=torch.float64)
        #         values[index_ss, index_sd] = lyapunov_nn.lyapunov_function(input_).detach()
        # r, theta = np.meshgrid(ss, sd)
        # func = interp2d(sd, ss, values, kind='cubic')
        # # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        # # r_new = np.linspace(0, 2, 10)  # y
        #
        # # v_plot = func(t_new, r_new)
        # # t_plot, r_plot = np.meshgrid(t_new, r_new)
        #
        # plt.figure(figsize=(10, 10))
        # ax = plt.subplot(projection='polar')
        # # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        # # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet', vmax= max_ly, vmin=min_ly)
        # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet')
        # ax.set_rgrids([1, 3, 5, 7])
        # ax.set_theta_zero_location('N')
        # ax.set_theta_direction(-1)
        # plt.grid(c='black')
        # plt.colorbar(pc, shrink=.8)
        # # plt.savefig('heatmap2/heatmap_epoch_{}_w8_{}.png'.format(i, index_wd))
        # plt.savefig('heatmap_2w_step/heatmap_epoch_{}.png'.format(i))

        # ss = np.linspace(0, 7.5, 10)
        # sd = np.linspace(-np.pi, np.pi, 50)
        # ws = 8
        # wd = np.linspace(0, 2 * np.pi, 20)
        # values = np.zeros((ss.shape[0], sd.shape[0]))

        # for index_wd in range(wd.shape[0]):
        #     for index_ss in range(ss.shape[0]):
        #         for index_sd in range(sd.shape[0]):
        #             input_array = [ss[index_ss]*np.sin(sd[index_sd]), ss[index_ss]*np.cos(sd[index_sd]), ws*math.sin(wd[index_wd]), ws*math.cos(wd[index_wd])]
        #             input_ = np.array(input_array)
        #             values[index_ss, index_sd] = lyapunov_nn.lyapunov_function(input_)
        #     r, theta = np.meshgrid(ss, sd)
        #     func = interp2d(sd, ss, values, kind='cubic')
        #     # t_new = np.linspace(-np.pi, np.pi, 50)  # x
        #     # r_new = np.linspace(0, 2, 10)  # y

        #     # v_plot = func(t_new, r_new)
        #     # t_plot, r_plot = np.meshgrid(t_new, r_new)

        #     plt.figure(figsize=(10, 10))
        #     ax = plt.subplot(projection='polar')
        #     # pc = ax.pcolor(t_plot, r_plot, v_plot, shading='auto', cmap='jet')
        #     # pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet', vmax= max_ly, vmin=min_ly)
        #     pc = ax.pcolor(theta, r, values.T, shading='auto', cmap='jet')
        #     ax.set_rgrids([1, 3, 5, 7])
        #     ax.set_theta_zero_location('N')
        #     ax.set_theta_direction(-1)
        #     plt.grid(c='black')
        #     plt.colorbar(pc, shrink=.8)
        #     plt.savefig('heatmap2/heatmap_epoch_{}_w8_{}.png'.format(i, index_wd))
        roa_estimate |= lyapunov_nn.safe_set.detach().numpy()
        # lyapunov_nn.safe_set |= roa_estimate
        c_max.append(lyapunov_nn.c_max.detach().numpy())
        c_plot.append(lyapunov_nn.c_max.item())
        plt.plot(c_plot)
        plt.xlabel('Epoch')
        plt.ylabel('c')
        plt.show()
        plt.pause(0.1)
        # scatter_heatmap(states, lyapunov_nn.values.numpy(), c_max[-1], initial_safe_set, i)
        # plot_safe_set(lyapunov_nn.lyapunov_function, 200, i, seed, iteration)
        # polar_coordinates(lyapunov_nn.lyapunov_function, index_outer_iters, seed, iteration)
        print('Current safe level (c_k): {}'.format(c_max[-1]))
        print('Safe set size_C : {}'.format((lyapunov_nn.values.detach().numpy() < c_max[-1]).sum()))
        print('Safe set size : {}'.format((lyapunov_nn.safe_set.detach().sum())))

        with open('c_max_pre.pkl', 'wb') as f:
            pickle.dump(c_max, f)

        with open('roa_estimate.pkl', 'wb') as f:
            pickle.dump(roa_estimate.sum(), f)

        with open('Safe_set.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set, f)

        with open('Safe_set_size.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set.detach().sum(), f)

        with open('value_collection.pkl', 'wb') as f:
            pickle.dump(value_collection, f)

def lyapunov_train_swimmer(lyapunov_nn: safe_learning.Lyapunov, seed, cost, term):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    states = lyapunov_nn.states
    dynamics = lyapunov_nn.dynamics
    policy = lyapunov_nn.policy
    print('Lyapunov Training ...')
    value_collection = []
    # term = term
    # cost = cost
    roa_estimate = np.copy(lyapunov_nn.safe_set)
    c_max = [lyapunov_nn.c_max.detach().numpy(), ]
    # Training hyperparameters
    train_batch_size = 512
    outer_iters = 20
    inner_iters = 10
    horizon = 5
    test_size = int(train_batch_size)
    level_multiplier = 1.2
    batch_size = train_batch_size
    learning_rate_init = 3e-4
    learning_rate_10 = 1e-4
    learning_rate_30 = 3e-4
    learning_rate_40 = 3e-4
    # states_stand = torch.tensor(lyapunov_nn.states, dtype=torch.float64).detach()
    # x_ly_mean = torch.mean(states_stand, dim=0)
    # x_ly_mean = torch.tensor(x_ly_mean, dtype=torch.float64).detach()
    # x_ly_std = torch.std(states_stand, dim=0)
    # x_ly_std = torch.tensor(x_ly_std, dtype=torch.float64).detach()
    # data_min = states_stand.min(axis=0)[0]
    # data_min = torch.tensor(data_min, dtype=torch.float64).detach()
    # data_max = states_stand.max(axis=0)[0]
    # data_max = torch.tensor(data_max, dtype=torch.float64).detach()
    # epsilon = torch.tensor(1e-8, dtype=torch.float64)
    # ranges = data_max - data_min + epsilon
    # ranges = torch.tensor(ranges, dtype=torch.float64).detach()
    # learning_rate = 7e-2
    lagrange_multiplier = 100
    loss = Loss(lagrange_multiplier, OPTIONS.eps)
    # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
    # Train
    print('Current metrics ...')
    # c = lyapunov_nn.c_max.numpy()
    c = lyapunov_nn.c_max.detach().numpy()
    # num_safe = lyapunov_nn.safe_set.sum()
    print('Safe level (c_k): {}'.format(c))
    print('')
    optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate_init)
    time.sleep(0.5)
    safe_fraction = []
    losses = []
    c_plot = []
    for i in range(outer_iters):
        if i == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_10
        if i == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_30

        if i == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_40
        index_outer_iters = i
        print('Iteration (k): {}'.format(len(c_max)))
        print('ROAESTIMATE: {}'.format(roa_estimate.sum()))
        time.sleep(0.5)
        # Identify the "gap" states, i.e., those between V(c_k) and V(a * c_k) for a > 1
        c = lyapunov_nn.c_max

        # idx_decrease = ((lyapunov_nn.lyapunov_function(torch.tensor(dynamics.build_evaluation(lyapunov_nn.states))) -
        #                  lyapunov_nn.lyapunov_function(torch.tensor(lyapunov_nn.states))) < 0).numpy().ravel()
        #
        # # idx_decrease_all = np.zeros(lyapunov_nn.values.shape[0],dtype=bool)
        # # for i in range(lyapunov_nn.values.shape[0]):
        # #     if idx_decrease[i]:
        # #         idx_decrease_all[i] = True
        # decrease_region = lyapunov_nn.values.clone()
        # for i in range(lyapunov_nn.values.shape[0]):
        #     if not idx_decrease[i]:
        #         decrease_region[i] = 1000
        #
        # idx_small = (decrease_region <= c).numpy().ravel()
        # idx_big = (decrease_region <= level_multiplier * c).numpy().ravel()
        # idx_gap = np.logical_and(idx_big, ~idx_small)

        idx_small = (lyapunov_nn.values <= c).numpy().ravel()
        idx_big = (lyapunov_nn.values <= level_multiplier * c).numpy().ravel()
        idx_gap = np.logical_and(idx_big, ~idx_small)
        # # 判断gap区域的点
        gap_states = states[idx_gap]
        if gap_states.any():
            # for _ in range(horizon):
            #     # action_gap = policy(gap_states)
            #     # gap_states = dynamics(gap_states, action_gap)
            #     gap_states = dynamics.build_evaluation(gap_states)
            index_gap = []
            dim = lyapunov_nn.dynamics.states_after.shape[1]
            gap_states = gap_states.reshape(-1, dim)
            for index_gap_state in range(gap_states.shape[0]):
                for pre in range(horizon+1):
                    if ((gap_states[index_gap_state][-1] + pre) == lyapunov_nn.dynamics.states_after[-1][-1]):
                        index_gap.append(lyapunov_nn.dynamics.states_after[-1][-1] - 1.0)
                        break
                    elif term[0][(gap_states[index_gap_state][-1] + pre).astype(int)] == np.bool_(True) or cost[0][(gap_states[index_gap_state][-1] + pre).astype(int)] != 0.000:
                        index_gap.append(gap_states[index_gap_state][-1] + pre)
                        break
                    else:
                        if(pre == horizon):
                            index_gap.append(gap_states[index_gap_state][-1] + pre)
                            break
                        else:
                            continue
            index_gap = np.array(index_gap).astype(int)
            gap_states = lyapunov_nn.dynamics.states_after[index_gap]

            # for i in range(gap_states.shape[0]):
            #     #TODO 考虑位置？需更改
            #     dist.append(math.sqrt(gap_states[i][0] ** 2 + gap_states[i][1] ** 2))
            with torch.no_grad():
                gap_future_values = lyapunov_nn.lyapunov_function((torch.tensor(gap_states[:, :-1], dtype=torch.float64))).detach()
            # roa_estimate[idx_gap] |= np.array(dist) <= 7
            # roa_estimate[idx_small] |= idx_small  # 或运算
            roa_estimate = np.logical_or(roa_estimate, idx_small)
            roa_estimate[idx_gap] |= (gap_future_values <= c).numpy().ravel()  # 或运算
        # for index in range(roa_estimate.shape[0]-25):
        #     for k in range(25):
        #         roll, pitch, _ = np.rad2deg(tr.quat_to_euler(lyapunov_nn.states_all[index+k, 36:40]))
        #         if (np.abs(roll) > 30) | (np.abs(pitch) > 30):
        #             roa_estimate[index] = False
        #             break

        target_idx = np.logical_or(idx_big, roa_estimate)
        target_set = states[target_idx]
        print('target_set_size:{}'.format(target_set.shape[0]))
        # # 整个ROA进行分类为+1，0，后续梯度下降会换成+1，-1
        target_labels = roa_estimate[target_idx].astype(OPTIONS.np_dtype).reshape([-1, 1])
        idx_range = target_set.shape[0]
        # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
        # SGD for classification
        for _ in tqdm(range(inner_iters)):
            # Training step
            # idx_batch_eval = tf.compat.v1.random_uniform([batch_size, ], 0, idx_range, dtype=tf.int32,
            #                                              name='batch_sample')
            torch.set_printoptions(precision=8)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                print(name, param.data)
            optimizer.zero_grad()  # 重置梯度
            # for name, param in lyapunov_nn.lyapunov_function.named_parameters():
            #     print(name, param.data)
            idx_batch_eval = torch.randint(low=0, high=idx_range, size=(batch_size,),
                                           dtype=torch.int32,requires_grad=False)
            train_states = target_set[idx_batch_eval]
            train_state_torch = torch.tensor(train_states[:, :-1],dtype=torch.float64)

            # writer.add_graph(lyapunov_nn.lyapunov_function,torch.tensor(train_state_torch, dtype=torch.float64))
            # writer.close()
            # train_states = states
            train_level = c
            train_roa_labels = target_labels[idx_batch_eval]  # 确定他们的标签
            # train_roa_labels = lyapunov_nn.safe_set.astype(OPTIONS.np_dtype).reshape([-1, 1])
            class_weights, class_counts = balanced_class_weights(train_roa_labels.astype(bool), scale_by_total=True)


            # print(lyapunov_nn.lyapunov_function)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            # batch_states = train_states.clone().detach()
            # train_state_torch = (train_state_torch - x_ly_mean) / x_ly_std  #标准化
            # train_state_torch = (train_state_torch - data_min) / ranges  # 归一化
            values = lyapunov_nn.lyapunov_function(train_state_torch)
            value_collection.append(torch.max(values).detach().numpy())
            print('values:{}'.format(values))
            # print(id(values))
            next_states = dynamics.build_evaluation(train_states)
            next_states_torch = torch.tensor(next_states[:, :-1],dtype=torch.float64)
            # next_states_torch = (next_states_torch - x_ly_mean) / x_ly_std
            # next_states_torch = (next_states_torch - data_min) / ranges    # 归一化
            nn_train_value = lyapunov_nn.lyapunov_function(next_states_torch)
            # print('nn_train_value:{}'.format(nn_train_value))
            # print(id(nn_train_value))
            # values = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64, requires_grad=False)
            # loss = Loss(lagrange_multiplier,OPTIONS.eps)
            objective = loss(values, nn_train_value, train_level, train_roa_labels, class_weights)
            # criterion = nn.MSELoss()
            # # values = torch.tensor(values, dtype=torch.float64,requires_grad=True)
            # # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64,requires_grad=True)
            # objective = criterion(values, train_roa_labels)
            print(' loss: {}'.format(objective))
            # 在训练循环中
            loss_value = objective.item()
            losses.append(loss_value)

             # 绘制图形
            # plt.plot(losses)
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.draw()
            # plt.pause(0.1)  # 暂停一会，以便图形更新


            objective.backward()  # 自动计算梯度
            # 检查是否可以进行第二次backward
            # try:
            #     outputs = lyapunov_nn.lyapunov_function.forward(train_states)
            #     lyapunov_nn.lyapunov_function.forward(train_states)
            #     loss = loss(outputs, nn_train_value)
            #     loss.backward()  # 第二次backward
            # except RuntimeError as e:
            #     print("捕获到错误：", e)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            optimizer.step()  # 应用梯度更新



        # Update Lyapunov values and ROA estimate, based on new parameter values
        with torch.no_grad():
            lyapunov_nn.values = lyapunov_nn.lyapunov_function(torch.tensor(states[:, :-1], dtype=torch.float64)).detach()
        lyapunov_nn.update_c()

        roa_estimate |= lyapunov_nn.safe_set.detach().numpy()
        # lyapunov_nn.safe_set |= roa_estimate
        c_max.append(lyapunov_nn.c_max.detach().numpy())
        c_plot.append(lyapunov_nn.c_max.item())
        # plt.plot(c_plot)
        # plt.xlabel('Epoch')
        # plt.ylabel('c')
        # plt.show()
        # plt.pause(0.1)
        # scatter_heatmap(states, lyapunov_nn.values.numpy(), c_max[-1], initial_safe_set, i)
        # plot_safe_set(lyapunov_nn.lyapunov_function, 200, i, seed, iteration)
        # polar_coordinates(lyapunov_nn.lyapunov_function, index_outer_iters, seed, iteration)
        print('Current safe level (c_k): {}'.format(c_max[-1]))
        print('Safe set size_C : {}'.format((lyapunov_nn.values.detach().numpy() < c_max[-1]).sum()))
        print('Safe set size : {}'.format((lyapunov_nn.safe_set.detach().sum())))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('./ly_loss/loss.jpg')
        # plt.pause(0.1)

        with open('c_max.pkl', 'wb') as f:
            pickle.dump(c_max, f)

        with open('roa_estimate.pkl', 'wb') as f:
            pickle.dump(roa_estimate.sum(), f)

        with open('Safe_set.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set, f)

        with open('Safe_set_size.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set.detach().sum(), f)

        with open('value_collection.pkl', 'wb') as f:
            pickle.dump(value_collection, f)

def lyapunov_train_walker(lyapunov_nn: safe_learning.Lyapunov, seed, cost, term):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    states = lyapunov_nn.states
    dynamics = lyapunov_nn.dynamics
    policy = lyapunov_nn.policy
    print('Lyapunov Training ...')
    value_collection = []
    roa_estimate = np.copy(lyapunov_nn.safe_set)
    c_max = [lyapunov_nn.c_max.detach().numpy(), ]
    # Training hyperparameters
    train_batch_size = 512
    outer_iters = 20
    inner_iters = 10
    horizon = 5
    test_size = int(train_batch_size)
    level_multiplier = 1.2
    batch_size = train_batch_size
    learning_rate_init = 3e-4
    learning_rate_20 = 3e-4
    learning_rate_30 = 3e-4
    learning_rate_40 = 3e-4
    # states_stand = torch.tensor(lyapunov_nn.states, dtype=torch.float64).detach()
    # x_ly_mean = torch.mean(states_stand, dim=0)
    # x_ly_mean = torch.tensor(x_ly_mean, dtype=torch.float64).detach()
    # x_ly_std = torch.std(states_stand, dim=0)
    # x_ly_std = torch.tensor(x_ly_std, dtype=torch.float64).detach()
    # data_min = states_stand.min(axis=0)[0]
    # data_min = torch.tensor(data_min, dtype=torch.float64).detach()
    # data_max = states_stand.max(axis=0)[0]
    # data_max = torch.tensor(data_max, dtype=torch.float64).detach()
    # epsilon = torch.tensor(1e-8, dtype=torch.float64)
    # ranges = data_max - data_min + epsilon
    # ranges = torch.tensor(ranges, dtype=torch.float64).detach()
    # learning_rate = 7e-2
    lagrange_multiplier = 100
    loss = Loss(lagrange_multiplier, OPTIONS.eps)
    # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
    # Train
    print('Current metrics ...')
    # c = lyapunov_nn.c_max.numpy()
    c = lyapunov_nn.c_max.detach().numpy()
    # num_safe = lyapunov_nn.safe_set.sum()
    print('Safe level (c_k): {}'.format(c))
    print('')
    optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate_init)
    time.sleep(0.5)
    safe_fraction = []
    losses = []
    c_plot = []
    for i in range(outer_iters):
        if i == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_20
        if i == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_30

        if i == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_40
        index_outer_iters = i
        print('Iteration (k): {}'.format(len(c_max)))
        print('ROAESTIMATE: {}'.format(roa_estimate.sum()))
        time.sleep(0.5)
        # Identify the "gap" states, i.e., those between V(c_k) and V(a * c_k) for a > 1
        c = lyapunov_nn.c_max

        # idx_decrease = ((lyapunov_nn.lyapunov_function(torch.tensor(dynamics.build_evaluation(lyapunov_nn.states))) -
        #                  lyapunov_nn.lyapunov_function(torch.tensor(lyapunov_nn.states))) < 0).numpy().ravel()
        #
        # # idx_decrease_all = np.zeros(lyapunov_nn.values.shape[0],dtype=bool)
        # # for i in range(lyapunov_nn.values.shape[0]):
        # #     if idx_decrease[i]:
        # #         idx_decrease_all[i] = True
        # decrease_region = lyapunov_nn.values.clone()
        # for i in range(lyapunov_nn.values.shape[0]):
        #     if not idx_decrease[i]:
        #         decrease_region[i] = 1000
        #
        # idx_small = (decrease_region <= c).numpy().ravel()
        # idx_big = (decrease_region <= level_multiplier * c).numpy().ravel()
        # idx_gap = np.logical_and(idx_big, ~idx_small)

        idx_small = (lyapunov_nn.values <= c).numpy().ravel()
        idx_big = (lyapunov_nn.values <= level_multiplier * c).numpy().ravel()
        idx_gap = np.logical_and(idx_big, ~idx_small)
        # # 判断gap区域的点
        gap_states = states[idx_gap]
        if gap_states.any():
            for _ in range(horizon):
                # action_gap = policy(gap_states)
                # gap_states = dynamics(gap_states, action_gap)
                gap_states = dynamics.build_evaluation(gap_states)
                # index_gap = []
                # dim = lyapunov_nn.dynamics.states_after.shape[1]
                # states_gap = gap_states.reshape(-1, dim)
                # for i in range(gap_states.shape[0]):
                #     if states[i][-1] == 1000.0:
                #         index.append(999.0)
                #     elif (states[i][-1] == self.states_after[-1][-1]):
                #         index.append(states[i][-1] - 1.0)
                #     else:
                #         index.append(states[i][-1])
                # index = np.array(index).astype(int)
                # return self.states_after[index]
            # dist = []
            # for i in range(gap_states.shape[0]):
            #     #TODO 考虑位置？需更改
            #     dist.append(math.sqrt(gap_states[i][0] ** 2 + gap_states[i][1] ** 2))
            with torch.no_grad():
                gap_future_values = lyapunov_nn.lyapunov_function((torch.tensor(gap_states[:, :-1], dtype=torch.float64))).detach()
            # roa_estimate[idx_gap] |= np.array(dist) <= 7
            # roa_estimate[idx_small] |= idx_small  # 或运算
            roa_estimate = np.logical_or(roa_estimate, idx_small)
            roa_estimate[idx_gap] |= (gap_future_values <= c).numpy().ravel()  # 或运算
        # for index in range(roa_estimate.shape[0]-25):
        #     for k in range(25):
        #         roll, pitch, _ = np.rad2deg(tr.quat_to_euler(lyapunov_nn.states_all[index+k, 36:40]))
        #         if (np.abs(roll) > 30) | (np.abs(pitch) > 30):
        #             roa_estimate[index] = False
        #             break

        target_idx = np.logical_or(idx_big, roa_estimate)
        target_set = states[target_idx]
        print('target_set_size:{}'.format(target_set.shape[0]))
        # # 整个ROA进行分类为+1，0，后续梯度下降会换成+1，-1
        target_labels = roa_estimate[target_idx].astype(OPTIONS.np_dtype).reshape([-1, 1])
        idx_range = target_set.shape[0]
        # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
        # SGD for classification
        for _ in tqdm(range(inner_iters)):
            # Training step
            # idx_batch_eval = tf.compat.v1.random_uniform([batch_size, ], 0, idx_range, dtype=tf.int32,
            #                                              name='batch_sample')
            torch.set_printoptions(precision=8)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                print(name, param.data)
            optimizer.zero_grad()  # 重置梯度
            # for name, param in lyapunov_nn.lyapunov_function.named_parameters():
            #     print(name, param.data)
            idx_batch_eval = torch.randint(low=0, high=idx_range, size=(batch_size,),
                                           dtype=torch.int32,requires_grad=False)
            train_states = target_set[idx_batch_eval]
            train_state_torch = torch.tensor(train_states[:, :-1],dtype=torch.float64)

            # writer.add_graph(lyapunov_nn.lyapunov_function,torch.tensor(train_state_torch, dtype=torch.float64))
            # writer.close()
            # train_states = states
            train_level = c
            train_roa_labels = target_labels[idx_batch_eval]  # 确定他们的标签
            # train_roa_labels = lyapunov_nn.safe_set.astype(OPTIONS.np_dtype).reshape([-1, 1])
            class_weights, class_counts = balanced_class_weights(train_roa_labels.astype(bool), scale_by_total=True)

            # print(lyapunov_nn.lyapunov_function)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            # batch_states = train_states.clone().detach()
            # train_state_torch = (train_state_torch - x_ly_mean) / x_ly_std  #标准化
            # train_state_torch = (train_state_torch - data_min) / ranges  # 归一化
            values = lyapunov_nn.lyapunov_function(train_state_torch)
            value_collection.append(torch.max(values).detach().numpy())
            print('values:{}'.format(values))
            # print(id(values))
            next_states = dynamics.build_evaluation(train_states)
            next_states_torch = torch.tensor(next_states[:, :-1],dtype=torch.float64)
            # next_states_torch = (next_states_torch - x_ly_mean) / x_ly_std
            # next_states_torch = (next_states_torch - data_min) / ranges    # 归一化
            nn_train_value = lyapunov_nn.lyapunov_function(next_states_torch)
            # print('nn_train_value:{}'.format(nn_train_value))
            # print(id(nn_train_value))
            # values = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64, requires_grad=False)
            # loss = Loss(lagrange_multiplier,OPTIONS.eps)
            objective = loss(values, nn_train_value, train_level, train_roa_labels, class_weights)
            # criterion = nn.MSELoss()
            # # values = torch.tensor(values, dtype=torch.float64,requires_grad=True)
            # # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64,requires_grad=True)
            # objective = criterion(values, train_roa_labels)
            print(' loss: {}'.format(objective))
            # 在训练循环中
            loss_value = objective.item()
            losses.append(loss_value)

             # 绘制图形
            # plt.plot(losses)
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.draw()
            # plt.pause(0.1)  # 暂停一会，以便图形更新


            objective.backward()  # 自动计算梯度
            # 检查是否可以进行第二次backward
            # try:
            #     outputs = lyapunov_nn.lyapunov_function.forward(train_states)
            #     lyapunov_nn.lyapunov_function.forward(train_states)
            #     loss = loss(outputs, nn_train_value)
            #     loss.backward()  # 第二次backward
            # except RuntimeError as e:
            #     print("捕获到错误：", e)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            optimizer.step()  # 应用梯度更新



        # Update Lyapunov values and ROA estimate, based on new parameter values
        with torch.no_grad():
            lyapunov_nn.values = lyapunov_nn.lyapunov_function(torch.tensor(states[:, :-1], dtype=torch.float64)).detach()
        lyapunov_nn.update_c()

        roa_estimate |= lyapunov_nn.safe_set.detach().numpy()
        # lyapunov_nn.safe_set |= roa_estimate
        c_max.append(lyapunov_nn.c_max.detach().numpy())
        c_plot.append(lyapunov_nn.c_max.item())
        # plt.plot(c_plot)
        # plt.xlabel('Epoch')
        # plt.ylabel('c')
        # plt.show()
        # plt.pause(0.1)
        # scatter_heatmap(states, lyapunov_nn.values.numpy(), c_max[-1], initial_safe_set, i)
        # plot_safe_set(lyapunov_nn.lyapunov_function, 200, i, seed, iteration)
        # polar_coordinates(lyapunov_nn.lyapunov_function, index_outer_iters, seed, iteration)
        print('Current safe level (c_k): {}'.format(c_max[-1]))
        print('Safe set size_C : {}'.format((lyapunov_nn.values.detach().numpy() < c_max[-1]).sum()))
        print('Safe set size : {}'.format((lyapunov_nn.safe_set.detach().sum())))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('./ly_loss/loss.jpg')
        # plt.pause(0.1)

        with open('c_max.pkl', 'wb') as f:
            pickle.dump(c_max, f)

        with open('roa_estimate.pkl', 'wb') as f:
            pickle.dump(roa_estimate.sum(), f)

        with open('Safe_set.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set, f)

        with open('Safe_set_size.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set.detach().sum(), f)

        with open('value_collection.pkl', 'wb') as f:
            pickle.dump(value_collection, f)

def lyapunov_train_HalfCheetah(lyapunov_nn: safe_learning.Lyapunov, seed, cost, term):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    states = lyapunov_nn.states
    dynamics = lyapunov_nn.dynamics
    policy = lyapunov_nn.policy
    print('Lyapunov Training ...')
    value_collection = []
    roa_estimate = np.copy(lyapunov_nn.safe_set)
    c_max = [lyapunov_nn.c_max.detach().numpy(), ]
    # Training hyperparameters
    train_batch_size = 512
    outer_iters = 20
    inner_iters = 10
    horizon = 5
    test_size = int(train_batch_size)
    level_multiplier = 1.2
    batch_size = train_batch_size
    learning_rate_init = 3e-4
    learning_rate_20 = 3e-4
    learning_rate_30 = 3e-4
    learning_rate_40 = 3e-4
    # states_stand = torch.tensor(lyapunov_nn.states, dtype=torch.float64).detach()
    # x_ly_mean = torch.mean(states_stand, dim=0)
    # x_ly_mean = torch.tensor(x_ly_mean, dtype=torch.float64).detach()
    # x_ly_std = torch.std(states_stand, dim=0)
    # x_ly_std = torch.tensor(x_ly_std, dtype=torch.float64).detach()
    # data_min = states_stand.min(axis=0)[0]
    # data_min = torch.tensor(data_min, dtype=torch.float64).detach()
    # data_max = states_stand.max(axis=0)[0]
    # data_max = torch.tensor(data_max, dtype=torch.float64).detach()
    # epsilon = torch.tensor(1e-8, dtype=torch.float64)
    # ranges = data_max - data_min + epsilon
    # ranges = torch.tensor(ranges, dtype=torch.float64).detach()
    # learning_rate = 7e-2
    lagrange_multiplier = 100
    loss = Loss(lagrange_multiplier, OPTIONS.eps)
    # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
    # Train
    print('Current metrics ...')
    # c = lyapunov_nn.c_max.numpy()
    c = lyapunov_nn.c_max.detach().numpy()
    # num_safe = lyapunov_nn.safe_set.sum()
    print('Safe level (c_k): {}'.format(c))
    print('')
    optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate_init)
    time.sleep(0.5)
    safe_fraction = []
    losses = []
    c_plot = []
    for i in range(outer_iters):
        if i == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_20
        if i == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_30

        if i == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_40
        index_outer_iters = i
        print('Iteration (k): {}'.format(len(c_max)))
        print('ROAESTIMATE: {}'.format(roa_estimate.sum()))
        time.sleep(0.5)
        # Identify the "gap" states, i.e., those between V(c_k) and V(a * c_k) for a > 1
        c = lyapunov_nn.c_max

        idx_small = (lyapunov_nn.values <= c).numpy().ravel()
        idx_big = (lyapunov_nn.values <= level_multiplier * c).numpy().ravel()
        idx_gap = np.logical_and(idx_big, ~idx_small)
        # # 判断gap区域的点
        gap_states = states[idx_gap]
        if gap_states.any():
            for _ in range(horizon):
                # action_gap = policy(gap_states)
                # gap_states = dynamics(gap_states, action_gap)
                gap_states = dynamics.build_evaluation(gap_states)
            # dist = []
            # for i in range(gap_states.shape[0]):
            #     #TODO 考虑位置？需更改
            #     dist.append(math.sqrt(gap_states[i][0] ** 2 + gap_states[i][1] ** 2))
            with torch.no_grad():
                gap_future_values = lyapunov_nn.lyapunov_function((torch.tensor(gap_states[:, :-1], dtype=torch.float64))).detach()
            # roa_estimate[idx_gap] |= np.array(dist) <= 7
            # roa_estimate[idx_small] |= idx_small  # 或运算
            roa_estimate = np.logical_or(roa_estimate, idx_small)
            roa_estimate[idx_gap] |= (gap_future_values <= c).numpy().ravel()  # 或运算
        # for index in range(roa_estimate.shape[0]-25):
        #     for k in range(25):
        #         roll, pitch, _ = np.rad2deg(tr.quat_to_euler(lyapunov_nn.states_all[index+k, 36:40]))
        #         if (np.abs(roll) > 30) | (np.abs(pitch) > 30):
        #             roa_estimate[index] = False
        #             break

        target_idx = np.logical_or(idx_big, roa_estimate)
        target_set = states[target_idx]
        print('target_set_size:{}'.format(target_set.shape[0]))
        # # 整个ROA进行分类为+1，0，后续梯度下降会换成+1，-1
        target_labels = roa_estimate[target_idx].astype(OPTIONS.np_dtype).reshape([-1, 1])
        idx_range = target_set.shape[0]
        # optimizer = torch.optim.SGD(lyapunov_nn.lyapunov_function.parameters(), lr=learning_rate)
        # SGD for classification
        for _ in tqdm(range(inner_iters)):
            # Training step
            # idx_batch_eval = tf.compat.v1.random_uniform([batch_size, ], 0, idx_range, dtype=tf.int32,
            #                                              name='batch_sample')
            torch.set_printoptions(precision=8)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                print(name, param.data)
            optimizer.zero_grad()  # 重置梯度
            # for name, param in lyapunov_nn.lyapunov_function.named_parameters():
            #     print(name, param.data)
            idx_batch_eval = torch.randint(low=0, high=idx_range, size=(batch_size,),
                                           dtype=torch.int32,requires_grad=False)
            train_states = target_set[idx_batch_eval]
            train_state_torch = torch.tensor(train_states[:, :-1],dtype=torch.float64)

            # writer.add_graph(lyapunov_nn.lyapunov_function,torch.tensor(train_state_torch, dtype=torch.float64))
            # writer.close()
            # train_states = states
            train_level = c
            train_roa_labels = target_labels[idx_batch_eval]  # 确定他们的标签
            # train_roa_labels = lyapunov_nn.safe_set.astype(OPTIONS.np_dtype).reshape([-1, 1])
            class_weights, class_counts = balanced_class_weights(train_roa_labels.astype(bool), scale_by_total=True)


            # print(lyapunov_nn.lyapunov_function)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            # batch_states = train_states.clone().detach()
            # train_state_torch = (train_state_torch - x_ly_mean) / x_ly_std  #标准化
            # train_state_torch = (train_state_torch - data_min) / ranges  # 归一化
            values = lyapunov_nn.lyapunov_function(train_state_torch)
            value_collection.append(torch.max(values).detach().numpy())
            print('values:{}'.format(values))
            # print(id(values))
            next_states = dynamics.build_evaluation(train_states)
            next_states_torch = torch.tensor(next_states[:, :-1],dtype=torch.float64)
            # next_states_torch = (next_states_torch - x_ly_mean) / x_ly_std
            # next_states_torch = (next_states_torch - data_min) / ranges    # 归一化
            nn_train_value = lyapunov_nn.lyapunov_function(next_states_torch)
            # print('nn_train_value:{}'.format(nn_train_value))
            # print(id(nn_train_value))
            # values = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64, requires_grad=False)
            # loss = Loss(lagrange_multiplier,OPTIONS.eps)
            objective = loss(values, nn_train_value, train_level, train_roa_labels, class_weights)
            # criterion = nn.MSELoss()
            # # values = torch.tensor(values, dtype=torch.float64,requires_grad=True)
            # # train_roa_labels = torch.tensor(train_roa_labels, dtype=torch.float64,requires_grad=True)
            # objective = criterion(values, train_roa_labels)
            print(' loss: {}'.format(objective))
            # 在训练循环中
            loss_value = objective.item()
            losses.append(loss_value)

             # 绘制图形
            # plt.plot(losses)
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.draw()
            # plt.pause(0.1)  # 暂停一会，以便图形更新


            objective.backward()  # 自动计算梯度
            # 检查是否可以进行第二次backward
            # try:
            #     outputs = lyapunov_nn.lyapunov_function.forward(train_states)
            #     lyapunov_nn.lyapunov_function.forward(train_states)
            #     loss = loss(outputs, nn_train_value)
            #     loss.backward()  # 第二次backward
            # except RuntimeError as e:
            #     print("捕获到错误：", e)
            for name, param in lyapunov_nn.lyapunov_function.named_parameters():
                if param.requires_grad:
                    print(name, param.grad)
            optimizer.step()  # 应用梯度更新



        # Update Lyapunov values and ROA estimate, based on new parameter values
        with torch.no_grad():
            lyapunov_nn.values = lyapunov_nn.lyapunov_function(torch.tensor(states[:, :-1], dtype=torch.float64)).detach()
        lyapunov_nn.update_c()

        roa_estimate |= lyapunov_nn.safe_set.detach().numpy()
        # lyapunov_nn.safe_set |= roa_estimate
        c_max.append(lyapunov_nn.c_max.detach().numpy())
        c_plot.append(lyapunov_nn.c_max.item())
        # plt.plot(c_plot)
        # plt.xlabel('Epoch')
        # plt.ylabel('c')
        # plt.show()
        # plt.pause(0.1)
        # scatter_heatmap(states, lyapunov_nn.values.numpy(), c_max[-1], initial_safe_set, i)
        # plot_safe_set(lyapunov_nn.lyapunov_function, 200, i, seed, iteration)
        # polar_coordinates(lyapunov_nn.lyapunov_function, index_outer_iters, seed, iteration)
        print('Current safe level (c_k): {}'.format(c_max[-1]))
        print('Safe set size_C : {}'.format((lyapunov_nn.values.detach().numpy() < c_max[-1]).sum()))
        print('Safe set size : {}'.format((lyapunov_nn.safe_set.detach().sum())))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('./ly_loss/loss.jpg')
        # plt.pause(0.1)

        with open('c_max.pkl', 'wb') as f:
            pickle.dump(c_max, f)

        with open('roa_estimate.pkl', 'wb') as f:
            pickle.dump(roa_estimate.sum(), f)

        with open('Safe_set.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set, f)

        with open('Safe_set_size.pkl', 'wb') as f:
            pickle.dump(lyapunov_nn.safe_set.detach().sum(), f)

        with open('value_collection.pkl', 'wb') as f:
            pickle.dump(value_collection, f)