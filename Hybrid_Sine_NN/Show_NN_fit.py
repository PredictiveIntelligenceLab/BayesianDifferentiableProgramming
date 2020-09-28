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

    layers = [1, 20, 20, 1]
    def get_para_num():
        L = len(layers)
        para_num = 0
        for k in range(L-1):
            para_num = para_num + layers[k] * layers[k+1] + layers[k+1]
        return para_num 

    true_y = np.load("true_y.npy")
    batch_y0 = np.load("batch_y0.npy")

    sigma_normal1 = np.std(true_y[:,0:1])
    

    x1 = true_y[:,0]
    x1_data = batch_y0[:,0] * sigma_normal1


    print(x1.shape)

    num_param = 9
    num_param_NN = get_para_num()


    def NN_forward_pass(H, layers, W):
        num_layers = len(layers)
        W_seq = W

        for k in range(0,num_layers-2):
            W = W_seq[0:layers[k] * layers[k+1]]
            W = np.reshape(W, (layers[k], layers[k+1]))
            W_seq = W_seq[layers[k] * layers[k+1]:]
            b = W_seq[0:layers[k+1]]
            b = np.reshape(b, (1, layers[k+1]))
            W_seq = W_seq[layers[k+1]:]
            H = np.tanh(np.add(np.matmul(H, W), b))

        W = W_seq[0:layers[num_layers-2] * layers[num_layers-1]]
        W = np.reshape(W, (layers[num_layers-2], layers[num_layers-1]))
        W_seq = W_seq[layers[num_layers-2] * layers[num_layers-1]:]
        b = W_seq[0:layers[num_layers-1]]
        b = np.reshape(b, (1, layers[num_layers-1]))
        H = np.add(np.matmul(H, W), b)
        return H


    N_samples = 100
    beta = 8.91
    N_total = 500

    X_test = np.linspace(-np.pi, np.pi, 500)[:,None]
    Y_test = - beta * np.sin(X_test)


    parameters = np.load("parameters.npy")
    precision = np.load("loggammalist.npy") 
    loglikelihood = np.load("loglikelihood.npy")
    print(parameters.shape)
    parameters = parameters[:,num_param:]
    print(parameters.shape)
    loglikelihood = loglikelihood[-N_total:]
    num_samples = parameters.shape[0]
    length_dict = parameters.shape[1]
    num_dim = 2


    idx_MAP = np.argmin(loglikelihood)
    MAP = parameters[idx_MAP, :]

    y_MAP = NN_forward_pass(X_test, layers, MAP)


    y_BNODE = np.zeros((X_test.shape[0], N_samples))

    for k in range(N_samples):
        print(k)
        idx_1 = np.random.randint(N_total)
        W_sample = parameters[-idx_1, :]
        y_BNODE[:,k:k+1] = NN_forward_pass(X_test, layers, W_sample) 


    mu_pred = np.mean(y_BNODE, axis = 1)
    Sigma_pred = np.var(y_BNODE, axis = 1)


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


    plt.figure(4, figsize=(9,6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x1_data, - beta * np.sin(x1_data), 'ro', alpha = 0.5, markersize = 4, label = "Training data")
    plt.plot(X_test, Y_test, 'r-', label = r"$-\beta\sin(x_1)$")
    plt.plot(X_test, y_MAP, 'g--', label = r"MAP curve of NN output")
    lower_0 = mu_pred - 2.0*np.sqrt(Sigma_pred)
    upper_0 = mu_pred + 2.0*np.sqrt(Sigma_pred)
    plt.fill_between(X_test.flatten(), lower_0.flatten(), upper_0.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$x_1$',fontsize=22)
    plt.ylabel(r'$-\beta\sin(x_1)$',fontsize=20)
    plt.axvspan(min(x1), max(x1), alpha=0.1, color='blue')
    plt.xlim((-2.2, 2.)) 
    plt.legend(loc='upper right', frameon=False, prop={'size': 18})
    plt.savefig('./BNODE_Sine.png', dpi = 300)  





