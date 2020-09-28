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
    niters = 5000 
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
    z0 = [0.5, 1.9, 0.18, 0.15, 0.16, 0.1, 0.064]
    true_yy = odeint(Glycolysis, z0, t_grid, args=(J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, N, A)) 

    true_y = np.load("true_y.npy")
    batch_y0 = np.load("batch_y0.npy")
    batch_yN = np.load("batch_yN.npy")

    batch_y0 = tf.to_float(batch_y0)
    batch_yN = tf.to_float(batch_yN)

    # normalizing data and system
    sigma_normal1 = np.std(true_y[:,0:1])
    sigma_normal2 = np.std(true_y[:,1:2])
    sigma_normal3 = np.std(true_y[:,2:3])
    sigma_normal4 = np.std(true_y[:,3:4])
    sigma_normal5 = np.std(true_y[:,4:5])
    sigma_normal6 = np.std(true_y[:,5:6])
    sigma_normal7 = np.std(true_y[:,6:7])


    num_param = 14
    para_num = num_param

    t0 = t_grid[:batch_time][0]
    t1 = t_grid[:batch_time][-1]
    t_in = np.linspace(t0, t1, 10) 

    parameters_pre = np.load("parameters_pre.npy")
    print(np.exp(parameters_pre))

    #########################################
    ########## precondition end #############
    #########################################

    initial_weight = parameters_pre
    print(initial_weight.shape, "here")

    class ODEModel(tf.keras.Model):
        def __init__(self):
            super(ODEModel, self).__init__()
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


    model = ODEModel()
    neural_ode = NeuralODE(model, t = t_in)

    def compute_gradients_and_update(batch_y0, batch_yN): 
        """Takes start positions (x0, y0) and final positions (xN, yN)"""
        pred_y = neural_ode.forward(batch_y0)
        with tf.GradientTape() as g:
            g.watch(pred_y)
            loss = tf.reduce_sum((pred_y - batch_yN)**2)
            
        dLoss = g.gradient(loss, pred_y)
        h_start, dfdh0, dWeights = neural_ode.backward(pred_y, dLoss)

        return loss, dWeights

    # Compile EAGER graph to static (this will be much faster)
    compute_gradients_and_update = tfe.defun(compute_gradients_and_update)


    # function to compute the kinetic energy
    def kinetic_energy(V, loggamma_v, loglambda_v):
        q = (np.sum(-V**2) - loggamma_v**2 - loglambda_v**2)/2.0
        return q

    def compute_gradient_param(dWeights, loggamma, loglambda, batch_size, para_num):
        WW = model.trainable_weights[0].numpy()
        dWeights = np.exp(loggamma)/2.0 * dWeights + np.exp(loglambda) * np.sign(WW)
        return dWeights

    def compute_gradient_hyper(loss, weights, loggamma, loglambda, batch_size, para_num):
        grad_loggamma = np.exp(loggamma) * (loss/2.0 + 1.0) - (batch_size/2.0 + 1.0)
        grad_loglambda = np.exp(loglambda) * (np.sum(np.abs(weights)) + 1.0) - (para_num + 1.0)

        return grad_loggamma, grad_loglambda

    def compute_Hamiltonian(loss, weights, loggamma, loglambda, batch_size, para_num):
        H = np.exp(loggamma)*(loss/2.0 + 1.0) + np.exp(loglambda)*(np.sum(np.abs(weights)) + 1.0)\
                 - (batch_size/2.0 + 1.0) * loggamma - (para_num + 1.0) * loglambda 
        return H

    def leap_frog(v_in, w_in, loggamma_in, loglambda_in, loggamma_v_in, loglambda_v_in):
        # assign weights from the previous step to the model
        model.trainable_weights[0].assign(w_in)
        v_new = v_in
        loggamma_v_new = loggamma_v_in
        loglambda_v_new = loglambda_v_in

        loggamma_new = loggamma_in
        loglambda_new = loglambda_in
        w_new = w_in

        for m in range(L):
            loss, dWeights = compute_gradients_and_update(batch_y0, batch_yN) # evaluate the gradient

            dWeights = np.asarray(dWeights[0]) # make the gradient to be numpy array
            dWeights = compute_gradient_param(dWeights, loggamma_new, loglambda_new, batch_size, para_num)
            grad_loggamma, grad_loglambda = compute_gradient_hyper(loss, w_new, loggamma_new, loglambda_new, batch_size, para_num)

            loggamma_v_new = loggamma_v_new - epsilon/2*grad_loggamma
            loglambda_v_new = loglambda_v_new - epsilon/2*grad_loglambda
            v_new = v_new - epsilon/2*(dWeights)
            w_new = model.trainable_weights[0].numpy() + epsilon * v_new
            model.trainable_weights[0].assign(w_new)
            loggamma_new = loggamma_new + epsilon * loggamma_v_new
            loglambda_new = loglambda_new + epsilon * loglambda_v_new
            
            # Second half of the leap frog
            loss, dWeights = compute_gradients_and_update(batch_y0, batch_yN)
            dWeights = np.asarray(dWeights[0])
            dWeights = compute_gradient_param(dWeights, loggamma_new, loglambda_new, batch_size, para_num)
            grad_loggamma, grad_loglambda = compute_gradient_hyper(loss, w_new, loggamma_new, loglambda_new, batch_size, para_num)

            v_new = v_new - epsilon/2*(dWeights)
            loggamma_v_new = loggamma_v_new - epsilon/2*grad_loggamma
            loglambda_v_new = loglambda_v_new - epsilon/2*grad_loglambda

        print(np.exp(loggamma_new))
        print(np.exp(loglambda_new))

        return v_new, w_new, loggamma_new, loglambda_new, loggamma_v_new, loglambda_v_new



    neural_ode_test = NeuralODE(model, t=t_grid[0:data_size:20])
    parameters = np.zeros((niters, para_num)) # book keeping the parameters
    loggammalist = np.zeros((niters, 1)) # book keeping the loggamma
    loglambdalist = np.zeros((niters, 1)) # book keeping the loggamma
    loglikelihood = np.zeros((niters, 1)) # book keeping the loggamma
    L = 10 # leap frog step number
    epsilon = 0.001 # leap frog step size
    epsilon_max = 0.0002    # max 0.001
    epsilon_min = 0.0002    # max 0.001
    

    def compute_epsilon(step):
        coefficient = np.log(epsilon_max/epsilon_min)
        return epsilon_max * np.exp( - step * coefficient / niters)


    # initial weight
    w_temp = initial_weight
    print("initial_w", np.exp(w_temp))
    loggamma_temp = 4. + np.random.normal()
    loglambda_temp = np.random.normal()

    model.trainable_weights[0].assign(w_temp)
    loss_original, _ = compute_gradients_and_update(batch_y0, batch_yN) # compute the initial Hamiltonian

    loggamma_temp = np.log(batch_size / loss_original)
    print("This is initial guess", loggamma_temp, "with loss", loss_original)
    if loggamma_temp > 6.:
        loggamma_temp = 6.
        epsilon_max = 0.0001
        epsilon_min = 0.0001


    # training steps
    for step in range(niters):

        epsilon = compute_epsilon(step)

        print(step)
        v_initial = np.random.randn(para_num, 1) # initialize the velocity
        loggamma_v_initial = np.random.normal()
        loglambda_v_initial = np.random.normal()

        loss_initial, _ = compute_gradients_and_update(batch_y0, batch_yN) # compute the initial Hamiltonian
        loss_initial = compute_Hamiltonian(loss_initial, w_temp, loggamma_temp, loglambda_temp, batch_size, para_num)

        v_new, w_new, loggamma_new, loglambda_new, loggamma_v_new, loglambda_v_new = \
                                leap_frog(v_initial, w_temp, loggamma_temp, loglambda_temp, loggamma_v_initial, loglambda_v_initial)

        # compute the final Hamiltonian
        loss_finial, _ = compute_gradients_and_update(batch_y0, batch_yN)
        loss_finial = compute_Hamiltonian(loss_finial, w_new, loggamma_new, loglambda_new, batch_size, para_num)

        # making decisions
        p_temp = np.exp(-loss_finial + loss_initial + \
                        kinetic_energy(v_new, loggamma_v_new, loglambda_v_new) - kinetic_energy(v_initial, loggamma_v_initial, loglambda_v_initial))

        p = min(1, p_temp)
        p_decision = np.random.uniform()
        if p > p_decision:
            parameters[step:step+1, :] = np.transpose(w_new)
            w_temp = w_new
            loggammalist[step, 0] = loggamma_new
            loglambdalist[step, 0] = loglambda_new
            loglikelihood[step, 0] = loss_finial
            loggamma_temp = loggamma_new
            loglambda_temp = loglambda_new
        else:
            parameters[step:step+1, :] = np.transpose(w_temp)
            model.trainable_weights[0].assign(w_temp)
            loggammalist[step, 0] = loggamma_temp
            loglambdalist[step, 0] = loglambda_temp
            loglikelihood[step, 0] = loss_initial

        print('probability', p)
        print(p > p_decision)


    np.save('parameters', parameters)
    np.save('loggammalist', loggammalist)
    np.save('loglikelihood', loglikelihood)
