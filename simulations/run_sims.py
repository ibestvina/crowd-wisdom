from krongraph import KronGraph
import pandas as pd
import sys
import numpy as np

netid = int(sys.argv[1])

def to_log(s):
    with open('log/log_' + str(netid) + '.txt', 'a+') as f:
        f.write(s + '\n')
        f.flush()

def to_results(s):
    with open('results/res_' + str(netid) + '.txt', 'a+') as f:
        f.write(str(s) + '\n')
        f.flush()

to_results('k,n0,n1,p_in_a,p_in_b,p_out,beta,sample_perc,mu_a,mu_b,p_in_a_approx,p_in_b_approx,q,q_p_app')

def result_to_str(net, kg, beta, sample_perc):
    return ','.join([str(s) for s in [net.k, net.n0, net.n1, net.p_in_a, net.p_in_b, net.p_out, beta,
                                      sample_perc, kg.mu_a, kg.mu_b, kg.p_in_a_approx, kg.p_in_b_approx, kg.q, kg.q_p_app]])

kron = np.array([[0.770117, 0.794312], [0.794312, 0.0965146]])

nets = pd.read_csv('nets_to_sample.csv')
net = nets.iloc[netid]


sample_at = [i/100 for i in range(1,16)]


initial_sample = max(sample_at)

betas = [0.0, 0.3, 0.8]

for beta in betas:
    kg = KronGraph(kron, int(net.k), int(net.n0), int(net.n1), net.p_in_a, net.p_in_b, net.p_out, initial_sample, beta, logger=to_log)

    for sample_perc in sample_at:
        to_log('Subsampling at ' + str(sample_perc))
        kg.subsample_params_calc(sample_perc)
        to_results(result_to_str(net, kg, beta, sample_perc))







