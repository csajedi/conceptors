# -*- coding: utf-8 -*-

import sys
import conceptors.net as net
import numpy as np
import matplotlib.pyplot as pplot

np.random.seed(123456)
dim_pattern = 6
dim_output = 18
num_neuron = 300
num_dataset = 6
num_time = 1000
num_washout = 500

network = net.ConceptorNetwork(dim_pattern, # num_in
                               dim_output, #
                               num_neuron,
                               washout_length=num_washout,
                               learn_length=num_time)

def read_data(filename, num_dataset):
    # get input matrix P and output matrix Q
    # input matrix P = [P1 ... Pn], for dim_pattern x 1 vector p
    #       totally dim_pattern * total_time
    # output matrix Q = [Q1 ... Qn], for dim_output x 1 vector q
    #       totally dim_output * total_time
    #       each q will be reshaped to a 6x3 matrix
    rdata = np.loadtxt(filename)
    # col 0: cluster, col 1: time
    # col 2-7: in_pattern
    # col 8-*: out_C
    patterns = [None] * num_dataset
    results  = [None] * num_dataset
    for i in range(0, num_dataset):
        data_subset = rdata[rdata[:, 0] == i, 1:]
        p_now = data_subset[:, 1:1 + dim_pattern].transpose()
        q_now = data_subset[:, 1 + dim_pattern + 3: 1 + dim_pattern + 3 + dim_output].transpose()
        patterns[i] = p_now
        results[i]  = q_now
    return patterns, results

patterns, results = read_data('input.txt', 6)
P_matrix = np.hstack(patterns)
Q_matrix = np.hstack(results)

network.train(patterns, results)

# print network.W_out
# y = network.W_out.dot(network.all_train_args)
# print y.shape
# print network.all_train_outs.shape
#
# pplot.figure(1);
# pplot.plot(xrange(500), results[1][4,500:1000]);
# pplot.plot(xrange(500), y[4,0:500]);
# pplot.title("Redout")
# pplot.show();

def print_matrix(X):
    print 'MATRIX', X.shape[0], X.shape[1]
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            print '{0:.6f}'.format(X[i,j]),
        print

out = sys.stdout
sys.stdout = open('conceptor.dat', 'w')
for i in xrange(6):
    print_matrix(network.Cs[0][i])
print_matrix(network.W)
print_matrix(network.W_out)
print_matrix(network.W_bias)
sys.stdout = out

parameter_nl = 0.1;
c_test_length = 5;
state_nl = 0.5;
W_noisy = network.W # + parameter_nl * np.abs(network.W).dot(np.random.rand(network.num_neuron, network.num_neuron) - 0.5);
np.set_printoptions(
    precision=3,
    suppress=False,
    linewidth=100
)

x = 0.5 * np.random.rand(network.num_neuron, 1) + 0.5
# np.savetxt('xarray.txt', x, fmt="%.6g")
for p in xrange(network.num_pattern):
    c = network.Cs[0][p];

    for n in xrange(c_test_length):
        # print x[:10].transpose()
        x = np.tanh(W_noisy.dot(x) + network.W_bias) # + state_nl * (np.random.rand(network.num_neuron, 1) - 0.5);
        x = c.dot(x);


    y = network.W_out.dot(x)
    print y.reshape((3,6))
    print np.asarray(np.average(results[p], 1).reshape((3,6)))
    print

print network.W_out.shape
print network.Cs[3][0]
print network.Cs[3][1]

from scipy.spatial import distance
for i in xrange(6):
    for j in xrange(6):
        print distance.euclidean(np.sort(network.Cs[3][i]),
                                 np.sort(network.Cs[3][j])),
    print

    




