from krongraph import KronGraph
import pandas as pd
import sys
import numpy as np
import time

repeat = 1000

netid = int(sys.argv[1])

def to_log(s):
    datetime = time.strftime('%d.%m.%Y. %H:%M:%S')
    line = datetime + ' - ' + s + '\n'
    with open('log/log_' + str(netid) + '.txt', 'a+') as f:
        f.write(line)
        f.flush()

def to_results(s):
    with open('results/res_' + str(netid) + '.txt', 'a+') as f:
        f.write(str(s) + '\n')
        f.flush()

to_results('k,n0,n1,p_in_a,p_in_b,p_out,beta,sample_perc,mu_a,mu_b,p_in_a_approx,p_in_b_approx,q,q_p_app')

def result_to_str(net, kg, beta, sample_perc):
    return ','.join([str(s) for s in [net.k, net.n0, net.n1, net.p_in_a, net.p_in_b, net.p_out, beta,
                                      sample_perc, kg.mu_a, kg.mu_b, kg.p_in_a_approx, kg.p_in_b_approx, kg.q, kg.q_p_app]])

nets = pd.read_csv('nets_to_sample.csv', index_col='id')
net = nets.loc[netid]

sample_cnt = 20
sample_at = np.linspace(0, net.sample_size, sample_cnt+1)[1:]
initial_sample = max(sample_at)

kron = np.array([[0.770117, 0.794312], [0.794312, 0.0965146]])

for beta in [0.0, 0.3, 0.8]:
    kg = KronGraph(kron, int(net.k), int(net.n0), int(net.n1), net.p_in_a, net.p_in_b, net.p_out, initial_sample, beta, logger=to_log)

    for i in range(repeat):
        to_log('Repetition {}/{}'.format(i,repeat))
        for sample_perc in sample_at:
            #to_log('Subsampling at ' + str(sample_perc))
            kg.subsample_params_calc(sample_perc)
            to_results(result_to_str(net, kg, beta, sample_perc))
