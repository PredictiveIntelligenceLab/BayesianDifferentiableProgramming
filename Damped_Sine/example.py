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
    niters = 1500
    batch_size = 1000


    alpha = 0.2
    beta = 8.91


    z0 = [-1.193, -3.876]
    true_y0 = tf.to_float([[-1.193, -3.876]])
    t_grid = np.linspace(0, 20, data_size)

    true_yy = odeint(Damped_pendulum, z0, t_grid, args=(alpha, beta)) 
    print(true_yy.shape)

    true_y = true_yy + 0.0 * np.random.randn(true_yy.shape[0], true_yy.shape[1])

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


    Order = 2
    num_param = 1
    for m in range(1, Order):
        for n in range(m + 1):
            num_param += 1
            
    num_param = num_param * 2


    t0 = t_grid[:batch_time][0]
    t1 = t_grid[:batch_time][-1]
    t_in = np.linspace(t0, t1, 20)  


    batch_y0, batch_yN = get_batch()

    #########################################
    ########## precondition start ###########
    #########################################
    niters_pre = 1000

    class ODEModel_pre(tf.keras.Model):
        def __init__(self):
            super(ODEModel_pre, self).__init__()
            self.Weights = tfe.Variable(tf.random_normal([num_param//2, 2], dtype=tf.float32)*0.01, dtype=tf.float32) 

        def call(self, inputs, **kwargs):
            t, y = inputs
            h = y 
            x1 = h[:,0:1] * sigma_normal1
            x2 = h[:,1:2] * sigma_normal2
            temp = x1 ** 0 
            for m in range(1, Order):
                for n in range(m + 1):
                    temp = tf.concat([temp, x1**n * x2**(m - n)], 1)
                    
            h_out = tf.matmul(temp, self.Weights) / np.asarray([sigma_normal1, sigma_normal2])

            return h_out


    model_pre = ODEModel_pre()
    neural_ode_pre = NeuralODE(model_pre, t_in)
    optimizer = tf.train.AdamOptimizer(3e-2)


    def compute_gradients_and_update_pre(batch_y0, batch_yN):
        """Takes start positions (x0, y0) and final positions (xN, yN)"""
        pred_y = neural_ode_pre.forward(batch_y0)
        with tf.GradientTape() as g_pre:
            g_pre.watch(pred_y)
            loss = tf.reduce_mean((pred_y - batch_yN)**2) + tf.reduce_sum(tf.abs(model_pre.Weights))
            
        dLoss = g_pre.gradient(loss, pred_y)
        h_start, dfdh0, dWeights = neural_ode_pre.backward(pred_y, dLoss)
        optimizer.apply_gradients(zip(dWeights, model_pre.weights))
        return loss, dWeights

    # Compile EAGER graph to static (this will be much faster)
    compute_gradients_and_update_pre = tfe.defun(compute_gradients_and_update_pre)

    parameters_pre = np.zeros((niters_pre, num_param))

    for step in range(niters_pre):
        print(step)
        loss, dWeights = compute_gradients_and_update_pre(batch_y0, batch_yN)

        model_parameters_pre = model_pre.trainable_weights[0].numpy().flatten()
        for k in range(num_param):
            parameters_pre[step, k] = model_parameters_pre[k]
        
        print(parameters_pre[step,:])

    #########################################
    ########## precondition end #############
    #########################################

    initial_weight = model_pre.trainable_weights[0].numpy()
    print(initial_weight.shape, "here")

    class ODEModel(tf.keras.Model):
        def __init__(self, initial_weight):
            super(ODEModel, self).__init__()
            self.Weights = tfe.Variable(initial_weight, dtype=tf.float32) 

        def call(self, inputs, **kwargs):
            t, y = inputs
            h = y 
            x1 = h[:,0:1] * sigma_normal1
            x2 = h[:,1:2] * sigma_normal2
            temp = x1 ** 0 
            for m in range(1, Order):
                for n in range(m + 1):
                    temp = tf.concat([temp, x1**n * x2**(m - n)], 1)
                    
            h_out = tf.matmul(temp, self.Weights) / np.asarray([sigma_normal1, sigma_normal2])
            return h_out


    model = ODEModel(initial_weight)
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
        
        print(model.trainable_weights)

        v_new = v_in
        loggamma_v_new = loggamma_v_in
        loglambda_v_new = loglambda_v_in

        loggamma_new = loggamma_in
        loglambda_new = loglambda_in
        w_new = w_in

        for m in range(L):

            loss, dWeights = compute_gradients_and_update(batch_y0, batch_yN) # evaluate the gradient
            print(loss)

            dWeights = np.asarray(dWeights[0]) # make the gradient to be numpy array
            dWeights = compute_gradient_param(dWeights, loggamma_new, loglambda_new, batch_size, num_param)
            grad_loggamma, grad_loglambda = compute_gradient_hyper(loss, w_new, loggamma_new, loglambda_new, batch_size, num_param)

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
            dWeights = compute_gradient_param(dWeights, loggamma_new, loglambda_new, batch_size, num_param)
            grad_loggamma, grad_loglambda = compute_gradient_hyper(loss, w_new, loggamma_new, loglambda_new, batch_size, num_param)

            v_new = v_new - epsilon/2*(dWeights)
            loggamma_v_new = loggamma_v_new - epsilon/2*grad_loggamma
            loglambda_v_new = loglambda_v_new - epsilon/2*grad_loglambda

#        print(dWeights)
        print(dWeights)
        print(np.exp(loggamma_new))
        print(np.exp(loglambda_new))

        return v_new, w_new, loggamma_new, loglambda_new, loggamma_v_new, loglambda_v_new





    neural_ode_test = NeuralODE(model, t=t_grid[0:data_size:20])
    parameters = np.zeros((niters, num_param//2, 2)) # book keeping the parameters
    loggammalist = np.zeros((niters, 1)) # book keeping the loggamma
    loglambdalist = np.zeros((niters, 1)) # book keeping the loggamma
    loglikelihood = np.zeros((niters, 1)) # book keeping the loggamma
    L = 10 # leap frog step number
    epsilon = 0.001 # leap frog step size


    # initial weight
    w_temp = initial_weight
    loggamma_temp = 4. + np.random.normal()
    loglambda_temp = np.random.normal()

    # training steps
    for step in range(niters):
        print(step)
        v_initial = np.random.randn(num_param//2, 2) # initialize the velocity
        loggamma_v_initial = np.random.normal()
        loglambda_v_initial = np.random.normal()

        loss_initial, _ = compute_gradients_and_update(batch_y0, batch_yN) # compute the initial Hamiltonian
        loss_initial = compute_Hamiltonian(loss_initial, w_temp, loggamma_temp, loglambda_temp, batch_size, num_param)

        v_new, w_new, loggamma_new, loglambda_new, loggamma_v_new, loglambda_v_new = \
                                leap_frog(v_initial, w_temp, loggamma_temp, loglambda_temp, loggamma_v_initial, loglambda_v_initial)

        # compute the final Hamiltonian
        loss_finial, _ = compute_gradients_and_update(batch_y0, batch_yN)
        loss_finial = compute_Hamiltonian(loss_finial, w_new, loggamma_new, loglambda_new, batch_size, num_param)

        # making decisions
        p_temp = np.exp(-loss_finial + loss_initial + \
                        kinetic_energy(v_new, loggamma_v_new, loglambda_v_new) - kinetic_energy(v_initial, loggamma_v_initial, loglambda_v_initial))

        p = min(1, p_temp)
        p_decision = np.random.uniform()
        if p > p_decision:
            parameters[step:step+1, :, :] = w_new
            w_temp = w_new
            loggammalist[step, 0] = loggamma_new
            loglambdalist[step, 0] = loglambda_new
            loglikelihood[step, 0] = loss_finial
            loggamma_temp = loggamma_new
            loglambda_temp = loglambda_new
        else:
            parameters[step:step+1, :, :] = w_temp
            model.trainable_weights[0].assign(w_temp)
            loggammalist[step, 0] = loggamma_temp
            loglambdalist[step, 0] = loglambda_temp
            loglikelihood[step, 0] = loss_initial

        print('probability', p)
        print(p > p_decision)



    np.save('parameters', parameters)
    np.save('loggammalist', loggammalist)
    np.save('loglikelihood', loglikelihood)




