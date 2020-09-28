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

    def VP(z, t, alpha, beta, gamma, sigma):
        x, y = z
        dzdt = [alpha * x - beta * x * y, - gamma * y + sigma * x*y]
        return dzdt

    Order = 3
    def VP_BNODE(z, t, W, Order):
        x, y = z
        alpha = W[0]
        beta = W[1]
        gamma = W[2]
        sigma = W[3]
        dzdt = [alpha * x + beta * x * y, gamma * y + sigma * x*y]
        return dzdt


    data_size = 16001
    batch_time = 320
    batch_size = 1000

    data_size_true = 2000


    N_samples = 100
    N_total = 500


    alpha = 1.
    beta = 0.1
    gamma = 1.5
    sigma = 0.75


    z0 = [5., 5.]
    t_grid_train = np.linspace(0, 25, data_size)
    t_grid_true = np.linspace(0, 45, data_size_true)

    y_train = odeint(VP, z0, t_grid_train, args=(alpha, beta, gamma, sigma)) 
    idx = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False)
    y_train = y_train[idx] 


    y_train = y_train + 0.0 * np.random.randn(y_train.shape[0], y_train.shape[1])
    t_grid_train = t_grid_train[idx]


    sigma_normal1 = np.std(y_train[:,0:1])
    sigma_normal2 = np.std(y_train[:,1:2])

    noise_level = 0.03

    y_train[:,0:1] = y_train[:,0:1]/ sigma_normal1 + noise_level * np.random.randn(y_train[:,0:1].shape[0], y_train[:,0:1].shape[1])
    y_train[:,1:2] = y_train[:,1:2]/ sigma_normal2 + noise_level * np.random.randn(y_train[:,1:2].shape[0], y_train[:,1:2].shape[1])

    y_train[:,0:1] = y_train[:,0:1] * sigma_normal1
    y_train[:,1:2] = y_train[:,1:2] * sigma_normal2

    sigma_normal = np.asarray([sigma_normal1, sigma_normal2])


    y_true = odeint(VP, z0, t_grid_true, args=(alpha, beta, gamma, sigma)) 

    num_dim = 2
    parameters = np.load("parameters.npy")
    precision = np.load("loggammalist.npy")
    loglikelihood = np.load("loglikelihood.npy")
    precision = np.exp(precision)
    print("precision", precision)
    print(parameters.shape)
    loglikelihood = loglikelihood[-N_total:]
    num_samples = parameters.shape[0]
    length_dict = parameters.shape[1]

    idx_MAP = np.argmin(loglikelihood)
    MAP = parameters[idx_MAP, :]


    y_MAP = odeint(VP_BNODE, z0, t_grid_true, args=(MAP, Order)) 

    mu_pred = y_MAP

    y_BNODE = np.zeros((t_grid_true.shape[0], num_dim, N_samples))

    for k in range(N_samples):
        print(k)
        idx_1 = np.random.randint(N_total)
        idx_2 = np.random.randint(N_total)
        W_sample = parameters[-idx_1, :]
        precision_here = precision[-idx_2] * num_dim
        y_BNODE[:,:,k] = odeint(VP_BNODE, z0, t_grid_true, args=(W_sample, Order)) 
        Sigma_data = np.ones_like(mu_pred) / np.sqrt(precision_here)
        y_BNODE[:,:,k] = y_BNODE[:,:,k] + Sigma_data * np.random.normal()


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


    plt.figure(1, figsize=(6,6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(y_train[:,0], y_train[:,1], 'ro', label = "Training data")
    plt.plot(y_true[:,0], y_true[:,1], 'b-', label = "Exact Trajectory")
    plt.xlabel('$x_1$',fontsize=18)
    plt.ylabel('$x_2$',fontsize=18)
    plt.legend(loc='upper right', frameon=False, prop={'size': 13})
    plt.savefig('./Training_data.png', dpi = 300)     


    plt.figure(4, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,0,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $x_1(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,0,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,0], 'ro', label = "Training data of $x_1(t)$")
    plt.plot(t_grid_true, y_true[:,0], 'r-', label = "True Trajectory of $x_1(t)$")
    plt.plot(t_grid_true, y_MAP[:,0], 'b--', label = "MAP Trajectory of $x_1(t)$")    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$x_1(t)$',fontsize=26)
    plt.ylim((-1.1, 9.5)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x1.png', dpi = 300)  


    plt.figure(5, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_true, y_BNODE[:,1,0], '-', color = "gray", alpha = 0.8, label = "Sample Trajectory of $x_2(t)$", linewidth = 0.5)
    plt.plot(t_grid_true, y_BNODE[:,1,:], '-', color = "gray", alpha = 0.8, linewidth = 0.5)
    plt.plot(t_grid_train, y_train[:,1], 'ro', label = "Training data of $x_2(t)$")
    plt.plot(t_grid_true, y_true[:,1], 'r-', label = "True Trajectory of $x_2(t)$")
    plt.plot(t_grid_true, y_MAP[:,1], 'b--', label = "MAP Trajectory of $x_2(t)$")  
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$x_2(t)$',fontsize=26)
    plt.ylim((-5.1, 55.5)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x2.png', dpi = 300)  





