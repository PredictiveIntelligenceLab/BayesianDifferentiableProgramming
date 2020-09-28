import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import numpy.random as npr
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from scipy.integrate import odeint
from scipy import stats

from plotting import newfig, savefig

np.random.seed(1234)
tf.set_random_seed(1234)


if __name__ == "__main__":

    def Glycolysis(z, t, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, N, A):
        x1, x2, x3, x4, x5, x6, x7 = z
        
        J = ka * (x4 - x7)
        N1 = N - x5
        A2 = A - x6
        
        v1 = k1 * x1 * x6 * (1 + (x6 / KI) ** q) ** (-1)
        v2 = k2 * x2 * N1
        v3 = k3 * x3 * A2
        v4 = k4 * x4 * x5
        v5 = k5 * x6
        v6 = k6 * x2 * x5
        v7 = k * x7
        
        r1 = J0 - v1
        r2 = 2*v1 - v2 - v6
        r3 = v2 - v3
        r4 = v3 - v4 - J
        r5 = v2 - v4 - v6
        r6 = - 2 * v1 + 2 * v3 - v5
        r7 = phi * J - v7
        dzdt = [r1, r2, r3, r4, r5, r6, r7]
        return dzdt


    def plot_spiral(trajectories, width = 1.):
        x1 = trajectories[:,0]
        x2 = trajectories[:,1]
        x3 = trajectories[:,2]
        x4 = trajectories[:,3]
        x5 = trajectories[:,4]
        x6 = trajectories[:,5]
        x7 = trajectories[:,6]
        plt.plot(x1, linewidth = width)
        plt.plot(x2, linewidth = width)
        plt.plot(x3, linewidth = width)
        plt.plot(x4, linewidth = width)
        plt.plot(x5, linewidth = width)
        plt.plot(x6, linewidth = width)
        plt.plot(x7, linewidth = width)

    Order = 3
    def Glycolysis_BNODE(z, t, W, Order):
        x1, x2, x3, x4, x5, x6, x7 = z

        J0 = W[0]
        k1 = W[1]
        k2 = W[2]
        k3 = W[3]
        k4 = W[4]
        k5 = W[5]
        k6 = W[6]
        k = W[7]
        ka = W[8]
        q = W[9]
        KI = W[10]
        phi = W[11]
        N = W[12]
        A = W[13]

        J = ka * (x4 - x7)
        N1 = N - x5
        A2 = A - x6
        
        v1 = k1 * x1 * x6 * (1 + (x6 / KI) ** q) ** (-1)
        v2 = k2 * x2 * N1
        v3 = k3 * x3 * A2
        v4 = k4 * x4 * x5
        v5 = k5 * x6
        v6 = k6 * x2 * x5
        v7 = k * x7
        
        r1 = J0 - v1
        r2 = 2*v1 - v2 - v6
        r3 = v2 - v3
        r4 = v3 - v4 - J
        r5 = v2 - v4 - v6
        r6 = - 2 * v1 + 2 * v3 - v5
        r7 = phi * J - v7
        dzdt = [r1, r2, r3, r4, r5, r6, r7]
        return dzdt



    data_size = 4000
    batch_time = 60
    batch_size = 1000

    data_size_true = 4000


    N_samples = 100
    N_total = 1500


    J0 = 2.5
    k1 = 100.
    k2 = 6.
    k3 = 16.
    k4 = 100.
    k5 = 1.28
    k6 = 12.
    k = 1.8
    ka = 13.
    q = 4.
    KI = 0.52
    phi = 0.1
    N = 1.
    A = 4.

    z0 = [0.5, 1.9, 0.18, 0.15, 0.16, 0.1, 0.064]
    t_grid_train = np.linspace(0, 5, data_size)
    t_grid_true = np.linspace(0, 10, data_size_true)

    y_train = odeint(Glycolysis, z0, t_grid_train, args=(J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, N, A)) 
    idx = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False)
    y_train = y_train[idx] 
    t_grid_train = t_grid_train[idx]

    sigma_normal1 = np.std(y_train[:,0:1])
    sigma_normal2 = np.std(y_train[:,1:2])
    sigma_normal3 = np.std(y_train[:,2:3])
    sigma_normal4 = np.std(y_train[:,3:4])
    sigma_normal5 = np.std(y_train[:,4:5])
    sigma_normal6 = np.std(y_train[:,5:6])
    sigma_normal7 = np.std(y_train[:,6:7])

    noise_level = 0.02

    y_train[:,0:1] = y_train[:,0:1]/ sigma_normal1 + noise_level * np.random.randn(y_train[:,0:1].shape[0], y_train[:,0:1].shape[1])
    y_train[:,1:2] = y_train[:,1:2]/ sigma_normal2 + noise_level * np.random.randn(y_train[:,1:2].shape[0], y_train[:,1:2].shape[1])
    y_train[:,2:3] = y_train[:,2:3]/ sigma_normal3 + noise_level * np.random.randn(y_train[:,2:3].shape[0], y_train[:,2:3].shape[1])
    y_train[:,3:4] = y_train[:,3:4]/ sigma_normal4 + noise_level * np.random.randn(y_train[:,3:4].shape[0], y_train[:,3:4].shape[1])
    y_train[:,4:5] = y_train[:,4:5]/ sigma_normal5 + noise_level * np.random.randn(y_train[:,4:5].shape[0], y_train[:,4:5].shape[1])
    y_train[:,5:6] = y_train[:,5:6]/ sigma_normal6 + noise_level * np.random.randn(y_train[:,5:6].shape[0], y_train[:,5:6].shape[1])
    y_train[:,6:7] = y_train[:,6:7]/ sigma_normal7 + noise_level * np.random.randn(y_train[:,6:7].shape[0], y_train[:,6:7].shape[1])

    y_train[:,0:1] = y_train[:,0:1] * sigma_normal1
    y_train[:,1:2] = y_train[:,1:2] * sigma_normal2
    y_train[:,2:3] = y_train[:,2:3] * sigma_normal3
    y_train[:,3:4] = y_train[:,3:4] * sigma_normal4
    y_train[:,4:5] = y_train[:,4:5] * sigma_normal5
    y_train[:,5:6] = y_train[:,5:6] * sigma_normal6
    y_train[:,6:7] = y_train[:,6:7] * sigma_normal7


    sigma_normal = np.asarray([sigma_normal1, sigma_normal2, sigma_normal3, sigma_normal4, sigma_normal5, sigma_normal6, sigma_normal7])

    y_true = odeint(Glycolysis, z0, t_grid_true, args=(J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, N, A)) 

    num_dim = 7
    parameters = np.load("parameters.npy")
    precision = np.load("loggammalist.npy")
    loglikelihood = np.load("loglikelihood.npy")
    precision = np.exp(precision) * num_dim
    print("precision", precision)
    print(parameters.shape)
    parameters = np.exp(parameters)
    num_samples = parameters.shape[0]
    length_dict = parameters.shape[1]

    loglikelihood = loglikelihood[-N_total:]
    idx_MAP = np.argmin(loglikelihood)
    print(idx_MAP, "index")
    print(loglikelihood)
    MAP = parameters[idx_MAP, :]
    print(MAP)


    y_MAP = odeint(Glycolysis_BNODE, z0, t_grid_true, args=(MAP, Order)) 

    mu_pred = y_MAP

    y_BNODE = np.zeros((t_grid_true.shape[0], num_dim, N_samples))

    for k in range(N_samples):
        print(k)
        idx_1 = np.random.randint(N_total)
        idx_2 = np.random.randint(N_total)
        W_sample = parameters[-idx_1, :]
        precision_here = precision[-idx_2]
        y_BNODE[:,:,k] = odeint(Glycolysis_BNODE, z0, t_grid_true, args=(W_sample, Order)) 
        Sigma_data = np.ones_like(mu_pred) / np.sqrt(precision_here)
        y_BNODE[:,:,k] = y_BNODE[:,:,k] + Sigma_data * sigma_normal * np.random.normal()
        print((Sigma_data * sigma_normal).shape)



    mu_pred = np.mean(y_BNODE, axis = 2)
    Sigma_pred = np.var(y_BNODE, axis = 2)


    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 16,
                         'lines.linewidth': 2,
                         'axes.labelsize': 20,  # fontsize for x and y labels (was 10)
                         'axes.titlesize': 20,
                         'xtick.labelsize': 16,
                         'ytick.labelsize': 16,
                         'legend.fontsize': 20,
                         'axes.linewidth': 2,
                        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                        "text.usetex": True,                # use LaTeX to write all text
                         })



    plt.figure(4, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,0,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $S_1(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,0,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,0], 'ro', label = "Training data of $S_1(t)$")
    plt.plot(t_grid_true, y_true[:,0], 'r-', label = "True Trajectory of $S_1(t)$")
    plt.plot(t_grid_true, y_MAP[:,0], 'b--', label = "MAP Trajectory of $S_1(t)$")
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$S_1(t)$',fontsize=26)
    plt.ylim((-0.1, 2.8)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x1.png', dpi = 300)  


    plt.figure(5, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,1,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $S_2(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,1,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,1], 'ro', label = "Training data of $S_2(t)$")
    plt.plot(t_grid_true, y_true[:,1], 'r-', label = "True Trajectory of $S_2(t)$")
    plt.plot(t_grid_true, y_MAP[:,1], 'b--', label = "MAP Trajectory of $S_2(t)$")  
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$S_2(t)$',fontsize=26)
    plt.ylim((-0.1, 3.5)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x2.png', dpi = 300)  


    plt.figure(6, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,2,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $S_3(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,2,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,2], 'ro', label = "Training data of $S_3(t)$")
    plt.plot(t_grid_true, y_true[:,2], 'r-', label = "True Trajectory of $S_3(t)$")
    plt.plot(t_grid_true, y_MAP[:,2], 'b--', label = "MAP Trajectory of $S_3(t)$")  
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$S_3(t)$',fontsize=26)
    plt.ylim((0., 0.3)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x3.png', dpi = 300)  


    plt.figure(7, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,3,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $S_4(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,3,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,3], 'ro', label = "Training data of $S_4(t)$")
    plt.plot(t_grid_true, y_true[:,3], 'r-', label = "True Trajectory of $S_4(t)$")
    plt.plot(t_grid_true, y_MAP[:,3], 'b--', label = "MAP Trajectory of $S_4(t)$")  
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$S_4(t)$',fontsize=26)
    plt.ylim((0.0, 0.6)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x4.png', dpi = 300)  


    plt.figure(8, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,4,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $N_2(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,4,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,4], 'ro', label = "Training data of $N_2(t)$")
    plt.plot(t_grid_true, y_true[:,4], 'r-', label = "True Trajectory of $N_2(t)$")
    plt.plot(t_grid_true, y_MAP[:,4], 'b--', label = "MAP Trajectory of $N_2(t)$")  
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$N_2(t)$',fontsize=26)
    plt.ylim((0.05, 0.3)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x5.png', dpi = 300)  


    plt.figure(9, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,5,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $A_3(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,5,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,5], 'ro', label = "Training data of $A_3(t)$")
    plt.plot(t_grid_true, y_true[:,5], 'r-', label = "True Trajectory of $A_3(t)$")
    plt.plot(t_grid_true, y_MAP[:,5], 'b--', label = "MAP Trajectory of $A_3(t)$")  
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$A_3(t)$',fontsize=26)
    plt.ylim((-0.1, 4.5)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x6.png', dpi = 300)  


    plt.figure(10, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,6,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $S_4^{ex}(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,6,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,6], 'ro', label = "Training data of $S_4^{ex}(t)$")
    plt.plot(t_grid_true, y_true[:,6], 'r-', label = "True Trajectory of $S_4^{ex}(t)$")
    plt.plot(t_grid_true, y_MAP[:,6], 'b--', label = "MAP Trajectory of $S_4^{ex}(t)$")  
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$S_4^{ex}(t)$',fontsize=26)
    plt.ylim((0.04, 0.14)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x7.png', dpi = 300)  






