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
    niters = 1500


    alpha = 0.2
    beta = 8.91


    z0 = [-1.193, -3.876]
    true_y0 = tf.to_float([[-1.193, -3.876]])
    t_grid = np.linspace(0, 20, data_size)

    true_yy = odeint(Damped_pendulum, z0, t_grid, args=(alpha, beta)) 
    print(true_yy.shape)


    true_y = np.load("true_y.npy")
    batch_y0 = np.load("batch_y0.npy")
    batch_yN = np.load("batch_yN.npy")

    batch_y0 = tf.to_float(batch_y0)
    batch_yN = tf.to_float(batch_yN)

    # normalizing data and system
    sigma_normal1 = np.std(true_y[:,0:1])
    sigma_normal2 = np.std(true_y[:,1:2])


    t0 = t_grid[:batch_time][0]
    t1 = t_grid[:batch_time][-1]
    t_in = np.linspace(t0, t1, 20)  



    ## NN structure 
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

    parameters_pre = np.load("parameters_pre.npy")
    print(parameters_pre)

    #########################################
    ########## precondition end #############
    #########################################

    initial_weight = parameters_pre
    print(initial_weight.shape, "here")

    class ODEModel(tf.keras.Model):
        def __init__(self):
            super(ODEModel, self).__init__()
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
    #            print(dWeights)
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
    epsilon_max = 0.001    # max 0.001
    epsilon_min = 0.001    # max 0.001
    

    def compute_epsilon(step):
        coefficient = np.log(epsilon_max/epsilon_min)
        return epsilon_max * np.exp( - step * coefficient / niters)


    # initial weight
    w_temp = initial_weight
    print("initial_w", w_temp)
    loggamma_temp = 4. + np.random.normal()
    loglambda_temp = np.random.normal()

    model.trainable_weights[0].assign(w_temp)
    loss_original, _ = compute_gradients_and_update(batch_y0, batch_yN) # compute the initial Hamiltonian

    loggamma_temp = np.log(batch_size / loss_original)
    print("This is initial guess", loggamma_temp, "with loss", loss_original)
    if loggamma_temp > 6.:
        loggamma_temp = 6.
        epsilon_max = 0.0005
        epsilon_min = 0.0005


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

