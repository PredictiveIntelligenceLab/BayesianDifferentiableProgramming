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

    def Spiral(z, t, alpha, beta, gamma, theta):
        x, y = z
        dzdt = [alpha * x**3 + beta * y**3, gamma * x**3 + theta * y**3]
        return dzdt


    def Spiral_SINDy(z, t, alpha, beta, gamma, theta, addition):
        x, y = z
        dzdt = [alpha * x**3 + beta * y**3, gamma * x**3 + theta * y**3 + addition * x**2*y]
        return dzdt


    Order = 4
    def Spiral_BNODE(z, t, W, Order):
        x, y = z
        x = x[None,None]
        y = y[None,None]
        temp = x ** 0
        for m in range(1, Order):
            for n in range(m + 1):
                temp = np.concatenate([temp, x**n * y**(m - n)], 1)

        candidate = np.matmul(temp, W)
        dzdt = [candidate[0,0], candidate[0,1]]
        return dzdt


    data_size = 1000
    data_size_true = 2000

    N_samples = 100
    N_total = 500

    alpha = -0.1
    beta = 2.0
    gamma = -2.0
    theta = -0.1


    SINDy_alpha = -0.13949865
    SINDy_gamma = -1.974
    SINDy_beta = 2.09897442   
    SINDy_theta = 0
    SINDy_add = -0.18839257


    z0 = [2., 0.]
    t_grid_train = np.linspace(0, 20, data_size)
    t_grid_true = np.linspace(0, 40, data_size_true)

    y_train = odeint(Spiral, z0, t_grid_train, args=(alpha, beta, gamma, theta)) 
    y_true = odeint(Spiral, z0, t_grid_true, args=(alpha, beta, gamma, theta)) 
    y_SINDy = odeint(Spiral_SINDy, z0, t_grid_true, args=(SINDy_alpha, SINDy_beta, SINDy_gamma, SINDy_theta, SINDy_add)) 


    y_train = y_train + 0.02 * np.random.randn(y_train.shape[0], y_train.shape[1])

    parameters = np.load("parameters.npy")
    precision = np.load("loggammalist.npy")
    loglikelihood = np.load("loglikelihood.npy")
    print(parameters.shape)
    loglikelihood = loglikelihood[-N_total:]
    num_samples = parameters.shape[0]
    length_dict = parameters.shape[1]
    num_dim = parameters.shape[2]
    precision = np.exp(precision) * num_dim
    print("precision", precision)



    idx_MAP = np.argmin(loglikelihood)
    MAP = parameters[idx_MAP, :, :]

    y_MAP = odeint(Spiral_BNODE, z0, t_grid_true, args=(MAP, Order)) 

    mu_pred = y_MAP
    y_BNODE = np.zeros((t_grid_true.shape[0], num_dim, N_samples))

    for k in range(N_samples):
        print(k)
        idx_1 = np.random.randint(N_total)
        idx_2 = np.random.randint(N_total)
        W_sample = parameters[-idx_1, :, :]
        precision_here = precision[-idx_2]
        y_BNODE[:,:,k] = odeint(Spiral_BNODE, z0, t_grid_true, args=(W_sample, Order)) 
        Sigma_data = np.ones_like(mu_pred) / np.sqrt(precision_here)
        y_BNODE[:,:,k] = y_BNODE[:,:,k] + Sigma_data * np.random.normal()
        print((Sigma_data).shape)


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
    plt.xlim((-2.3, 2.3)) 
    plt.ylim((-2.3, 2.3)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 14})
    plt.savefig('./Training_data.png', dpi = 300)     



    plt.figure(2, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_train, y_train[:,0], 'ro', label = "Training data of $x_1(t)$")
    plt.plot(t_grid_true, y_true[:,0], 'r-', label = "True Trajectory of $x_1(t)$")
    plt.plot(t_grid_true, y_SINDy[:,0], 'b--', label = "SINDy Trajectory of $x_1(t)$")
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$x_1(t)$',fontsize=26)
    plt.ylim((-2.3, 2.5)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./SINDy_Prediction_x1.png', dpi = 300)  


    plt.figure(3, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_train, y_train[:,1], 'ro', label = "Training data of $x_2(t)$")
    plt.plot(t_grid_true, y_true[:,1], 'r-', label = "True Trajectory of $x_2(t)$")
    plt.plot(t_grid_true, y_SINDy[:,1], 'b--', label = "SINDy Trajectory of $x_2(t)$")  
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$x_2(t)$',fontsize=26)
    plt.ylim((-2.3, 2.5)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./SINDy_Prediction_x2.png', dpi = 300)  


    plt.figure(4, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_train, y_train[:,0], 'ro', label = "Training data of $x_1(t)$")
    plt.plot(t_grid_true, y_true[:,0], 'r-', label = "True Trajectory of $x_1(t)$")
    plt.plot(t_grid_true, y_MAP[:,0], 'g--', label = "MAP Trajectory of $x_1(t)$")
    lower_0 = mu_pred[:,0] - 2.0*np.sqrt(Sigma_pred[:,0])
    upper_0 = mu_pred[:,0] + 2.0*np.sqrt(Sigma_pred[:,0])
    plt.fill_between(t_grid_true.flatten(), lower_0.flatten(), upper_0.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$x_1(t)$',fontsize=26)
    plt.ylim((-2.1, 2.5)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x1.png', dpi = 300)  


    plt.figure(5, figsize=(12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_grid_train, y_train[:,1], 'ro', label = "Training data of $x_2(t)$")
    plt.plot(t_grid_true, y_true[:,1], 'r-', label = "True Trajectory of $x_2(t)$")
    plt.plot(t_grid_true, y_MAP[:,1], 'g--', label = "MAP Trajectory of $x_2(t)$")  
    lower_1 = mu_pred[:,1] - 2.0*np.sqrt(Sigma_pred[:,1])
    upper_1 = mu_pred[:,1] + 2.0*np.sqrt(Sigma_pred[:,1])
    plt.fill_between(t_grid_true.flatten(), lower_1.flatten(), upper_1.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band") 
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$x_2(t)$',fontsize=26)
    plt.ylim((-2.1, 2.5)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.savefig('./BNODE_Prediction_x2.png', dpi = 300)  



