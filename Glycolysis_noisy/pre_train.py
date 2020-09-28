import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import numpy.random as npr
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
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

    
    data_size = 4000  
    batch_time = 60   
    batch_size = 1000 

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
    
        
    t_grid = np.linspace(0, 5, data_size)  
    true_y0 = tf.to_float([[0.5, 1.9, 0.18, 0.15, 0.16, 0.1, 0.064]])
    z0 = [0.5, 1.9, 0.18, 0.15, 0.16, 0.1, 0.064]
    true_yy = odeint(Glycolysis, z0, t_grid, args=(J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, N, A)) 

    true_y = true_yy + 0. * np.random.randn(true_yy.shape[0], true_yy.shape[1])

    # normalizing data and system
    sigma_normal1 = np.std(true_y[:,0:1])
    sigma_normal2 = np.std(true_y[:,1:2])
    sigma_normal3 = np.std(true_y[:,2:3])
    sigma_normal4 = np.std(true_y[:,3:4])
    sigma_normal5 = np.std(true_y[:,4:5])
    sigma_normal6 = np.std(true_y[:,5:6])
    sigma_normal7 = np.std(true_y[:,6:7])

    np.save('true_y', true_y)

    # scale for the isotropic noise
    noise_level = 0.02

    true_y[:,0:1] = true_y[:,0:1]/ sigma_normal1 + noise_level * np.random.randn(true_y[:,0:1].shape[0], true_y[:,0:1].shape[1])
    true_y[:,1:2] = true_y[:,1:2]/ sigma_normal2 + noise_level * np.random.randn(true_y[:,1:2].shape[0], true_y[:,1:2].shape[1])
    true_y[:,2:3] = true_y[:,2:3]/ sigma_normal3 + noise_level * np.random.randn(true_y[:,2:3].shape[0], true_y[:,2:3].shape[1])
    true_y[:,3:4] = true_y[:,3:4]/ sigma_normal4 + noise_level * np.random.randn(true_y[:,3:4].shape[0], true_y[:,3:4].shape[1])
    true_y[:,4:5] = true_y[:,4:5]/ sigma_normal5 + noise_level * np.random.randn(true_y[:,4:5].shape[0], true_y[:,4:5].shape[1])
    true_y[:,5:6] = true_y[:,5:6]/ sigma_normal6 + noise_level * np.random.randn(true_y[:,5:6].shape[0], true_y[:,5:6].shape[1])
    true_y[:,6:7] = true_y[:,6:7]/ sigma_normal7 + noise_level * np.random.randn(true_y[:,6:7].shape[0], true_y[:,6:7].shape[1])



    plt.figure(1, figsize=(12,8))
    plot_spiral(true_y, 1)
    plot_spiral(true_yy, 1)
    plt.xlabel('$t$',fontsize=13)
    plt.ylabel('$x$',fontsize=13)
    plt.legend(loc='upper left', frameon=False, prop={'size': 13})
    plt.savefig('./True_trajectory.png', dpi = 600)    


    def get_batch():
        """Returns initial point and last point over sampled frament of trajectory"""
        starts = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False)
        batch_y0 = true_y[starts] 
        batch_yN = true_y[starts + batch_time]
        return tf.to_float(batch_y0), tf.to_float(batch_yN)


    num_param = 14
    para_num = num_param

    t0 = t_grid[:batch_time][0]
    t1 = t_grid[:batch_time][-1]
    t_in = np.linspace(t0, t1, 10) 

    batch_y0, batch_yN = get_batch()

    np.save('batch_y0', batch_y0)
    np.save('batch_yN', batch_yN)

    #########################################
    ########## precondition start ###########
    #########################################
    niters_pre = 10000

    class ODEModel_pre(tf.keras.Model):
        def __init__(self):
            super(ODEModel_pre, self).__init__()
            self.Weights = tfe.Variable(tf.random_normal([num_param, 1], dtype=tf.float32)*0.01, dtype=tf.float32) 
    
        def call(self, inputs, **kwargs):
            t, y = inputs
            h = y
            x1 = h[:,0:1] * sigma_normal1
            x2 = h[:,1:2] * sigma_normal2
            x3 = h[:,2:3] * sigma_normal3
            x4 = h[:,3:4] * sigma_normal4
            x5 = h[:,4:5] * sigma_normal5
            x6 = h[:,5:6] * sigma_normal6
            x7 = h[:,6:7] * sigma_normal7

            J0 = self.Weights[0]  
            k1 = self.Weights[1] 
            k2 = self.Weights[2] 
            k3 = self.Weights[3] 
            k4 = self.Weights[4] 
            k5 = self.Weights[5] 
            k6 = self.Weights[6] 
            k = self.Weights[7]  
            ka = self.Weights[8]  
            q = self.Weights[9] 
            KI = self.Weights[10]  
            phi = self.Weights[11] 
            N = self.Weights[12] 
            A = self.Weights[13] 

            ka = tf.exp(ka)
            q = tf.exp(q)
            N = tf.exp(N)
            A = tf.exp(A)
            k1 = tf.exp(k1)
            k2 = tf.exp(k2)
            k3 = tf.exp(k3)
            k4 = tf.exp(k4)
            k5 = tf.exp(k5)
            k6 = tf.exp(k6)
            k = tf.exp(k)
            KI = tf.exp(KI)
            J0 = tf.exp(J0)
            phi = tf.exp(phi)

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

            h_out = tf.concat([r1/sigma_normal1, r2/sigma_normal2, r3/sigma_normal3, r4/sigma_normal4, r5/sigma_normal5, r6/sigma_normal6, r7/sigma_normal7], 1)
            return h_out


    model_pre = ODEModel_pre()
    neural_ode_pre = NeuralODE(model_pre, t_in)
    optimizer = tf.train.AdamOptimizer(5e-2)


    def compute_gradients_and_update_pre(batch_y0, batch_yN):
        """Takes start positions (x0, y0) and final positions (xN, yN)"""
        pred_y = neural_ode_pre.forward(batch_y0)
        with tf.GradientTape() as g_pre:
            g_pre.watch(pred_y)
            loss = tf.reduce_mean((pred_y - batch_yN)**2) + tf.reduce_sum(tf.abs(model_pre.trainable_weights[0]))
            loss_star = tf.reduce_sum((pred_y - batch_yN)**2)

        dLoss = g_pre.gradient(loss, pred_y)
        h_start, dfdh0, dWeights = neural_ode_pre.backward(pred_y, dLoss)
        optimizer.apply_gradients(zip(dWeights, model_pre.weights))
        return loss, dWeights, loss_star

    # Compile EAGER graph to static (this will be much faster)
    compute_gradients_and_update_pre = tfe.defun(compute_gradients_and_update_pre)

    parameters_pre = np.zeros((para_num, 1))

    for step in range(niters_pre):
        print(step)
        loss, dWeights, loss_star = compute_gradients_and_update_pre(batch_y0, batch_yN)
        print("loss is", loss_star)

        parameters_pre = model_pre.trainable_weights[0].numpy()

        print(np.exp(parameters_pre))

    np.save('parameters_pre', parameters_pre)

    #########################################
    ########## precondition end #############
    #########################################

