import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import numpy.random as npr
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow.contrib.eager as tfe
from tensorflow.keras import layers, initializers
from scipy.integrate import odeint


keras = tf.keras
tf.enable_eager_execution()

from neural_ode import NeuralODE 

np.random.seed(1234)
tf.set_random_seed(1234)


if __name__ == "__main__":

    def Damped_pendulum(z, t, alpha, beta):
        x, y = z
        dzdt = [y, - alpha * y - beta * np.sin(x)]
        return dzdt

    def plot_spiral(trajectories, width = 1.):
        x = trajectories[:,0]
        y = trajectories[:,1]
        plt.plot(x, y, linewidth=width)


    data_size = 8001
    batch_time = 200
    batch_size = 1000


    alpha = 0.2
    beta = 8.91


    z0 = [-1.193, -3.876]
    true_y0 = tf.to_float([[-1.193, -3.876]])
    t_grid = np.linspace(0, 20, data_size)

    true_yy = odeint(Damped_pendulum, z0, t_grid, args=(alpha, beta)) 
    print(true_yy.shape)

    true_y = true_yy + 0.0 * np.random.randn(true_yy.shape[0], true_yy.shape[1])

    np.save('true_y', true_y)


    sigma_normal1 = np.std(true_y[:,0:1])
    sigma_normal2 = np.std(true_y[:,1:2])

    true_y[:,0:1] = true_y[:,0:1]/ sigma_normal1
    true_y[:,1:2] = true_y[:,1:2]/ sigma_normal2

    true_y0 = tf.to_float([[-1.193/sigma_normal1, -3.876/sigma_normal2]])


    def get_batch():
        """Returns initial point and last point over sampled frament of trajectory"""
        starts = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False)
        batch_y0 = true_y[starts] 
        batch_yN = true_y[starts + batch_time]
        return tf.to_float(batch_y0), tf.to_float(batch_yN)



    layers = [1, 20, 20, 1]
    def get_para_num():
        L = len(layers)
        para_num = 0
        for k in range(L-1):
            para_num = para_num + layers[k] * layers[k+1] + layers[k+1]
        return para_num 


    num_param = 9
    num_param_NN = get_para_num()


    para_num = num_param + num_param_NN

    print("total parameters", para_num, "NN parameters", num_param_NN)


    t0 = t_grid[:batch_time][0]
    t1 = t_grid[:batch_time][-1]
    t_in = np.linspace(t0, t1, 20)  


    batch_y0, batch_yN = get_batch()

    np.save('batch_y0', batch_y0)
    np.save('batch_yN', batch_yN)


    #########################################
    ########## precondition start ###########
    #########################################
    niters_pre = 2000

    class ODEModel_pre(tf.keras.Model):
        def __init__(self):
            super(ODEModel_pre, self).__init__()
            self.Weights = tfe.Variable(tf.random_normal([num_param + num_param_NN, 1], dtype=tf.float32)*0.01, dtype=tf.float32)  

        def forward_pass(self, H, layers, W):
            num_layers = len(layers)
            W_seq = W

            for k in range(0,num_layers-2):
                W = W_seq[0:layers[k] * layers[k+1]]
                W = tf.reshape(W, (layers[k], layers[k+1]))
                W_seq = W_seq[layers[k] * layers[k+1]:]
                b = W_seq[0:layers[k+1]]
                b = tf.reshape(b, (1, layers[k+1]))
                W_seq = W_seq[layers[k+1]:]
                H = tf.tanh(tf.add(tf.matmul(H, W), b))

            W = W_seq[0:layers[num_layers-2] * layers[num_layers-1]]
            W = tf.reshape(W, (layers[num_layers-2], layers[num_layers-1]))
            W_seq = W_seq[layers[num_layers-2] * layers[num_layers-1]:]
            b = W_seq[0:layers[num_layers-1]]
            b = tf.reshape(b, (1, layers[num_layers-1]))
            H = tf.add(tf.matmul(H, W), b)
            return H

        def call(self, inputs, **kwargs):
            t, y = inputs
            h = y
            h1 = h[:,0:1]
            h2 = h[:,1:2]

            p11 = self.Weights[0]
            p12 = self.Weights[1]
            p13 = self.Weights[2]
            p14 = self.Weights[3]
            p15 = self.Weights[4]
            p16 = self.Weights[5]
            p23 = self.Weights[6]
            p25 = self.Weights[7]
            p26 = self.Weights[8]

            w_NN = self.Weights[num_param:,:]

            NN_out = self.forward_pass(sigma_normal1 * h1, layers, w_NN)

            h_out1 = p11 + sigma_normal1 * p12 *h1 + sigma_normal2 * p13 *h2 + sigma_normal1**2 * p14*h1**2\
                    + sigma_normal1 * sigma_normal2 * p15*h1*h2 + sigma_normal2**2 * p16*h2**2
            h_out2 = sigma_normal2 * p23 *h2 \
                    + sigma_normal1 * sigma_normal2 * p25*h1*h2 + sigma_normal2**2 * p26*h2**2 + NN_out
            h_out = tf.concat([h_out1/sigma_normal1, h_out2/sigma_normal2], 1)
            return h_out

            
    model_pre = ODEModel_pre()
    neural_ode_pre = NeuralODE(model_pre, t_in)
    optimizer = tf.train.AdamOptimizer(1e-1)


    def compute_gradients_and_update_pre(batch_y0, batch_yN):
        """Takes start positions (x0, y0) and final positions (xN, yN)"""
        pred_y = neural_ode_pre.forward(batch_y0)
        with tf.GradientTape() as g_pre:
            g_pre.watch(pred_y)
            loss = tf.reduce_mean((pred_y - batch_yN)**2) + tf.reduce_sum(tf.abs(model_pre.trainable_weights[0]))
            
        dLoss = g_pre.gradient(loss, pred_y)
        h_start, dfdh0, dWeights = neural_ode_pre.backward(pred_y, dLoss)
        optimizer.apply_gradients(zip(dWeights, model_pre.weights))
        return loss, dWeights

    # Compile EAGER graph to static (this will be much faster)
    compute_gradients_and_update_pre = tfe.defun(compute_gradients_and_update_pre)

    parameters_pre = np.zeros((para_num, 1))

    for step in range(niters_pre):
        print(step)
        loss, dWeights = compute_gradients_and_update_pre(batch_y0, batch_yN)
        parameters_pre = model_pre.trainable_weights[0].numpy()
        
        print(parameters_pre)

    np.save('parameters_pre', parameters_pre)

    #########################################
    ########## precondition end #############
    #########################################

