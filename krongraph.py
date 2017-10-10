import numpy as np
from numpy.random import binomial, choice

def bin_arr(size, n_ones):
    arr = np.array([0] * (size-n_ones) + [1] * n_ones, dtype='bool')
    np.random.shuffle(arr)
    return arr

class KronGraph:
    def __init__(self, kron, k, n0, n1, p_in_a, p_in_b, p_out, sample_size, beta):
        self.kron = kron
        self.k = k
        self.n0 = n0
        self.n1 = n1
        self.p_in_a = p_in_a
        self.p_in_b = p_in_b
        self.p_out = p_out
        self.sample_size = sample_size
        self.beta = beta
        
        self.kron_sum0, self.kron_sum1 = kron[0].sum(), kron[1].sum()
        
        self.Kk_size = kron.shape[0]**k
        self.n_nodes = (n0+n1) * self.Kk_size
        self.n_negative = n0 * self.Kk_size
        self.n_positive = self.n_nodes - self.n_negative
        
        self.p0 = (1-self.beta) / (2-self.beta)
        self.n_negative_sampled = binomial(self.sample_size, self.p0, 1)[0]
        self.n_positive_sampled = self.sample_size - self.n_negative_sampled
        
        self.true_p = n1 / (n0 + n1)
        self.sampled_p = self.n_positive_sampled
        
        self.n_edges = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}

        self.create_sample(sample_size, beta)
        
        
    def weight_vector_part(node):
        nodebin = bin(node)[1:]
        xi = np.array([1])
        for i in nodebin:
            xi = np.kron(xi, self.kron[1]) if i is '1' else np.kron(xi, self.kron[0])
        return xi
        
    def norm_params(self, w_vec, n, p):
        w_vec_sum = w_vec.sum()
        mu = w_vec_sum * n * p
        sigma = np.sqrt(w_vec.dot((1/p-w_vec))* n * p**2)
        return mu, sigma
    
    def node_nbs_part(node, n, p, n_total):
        w_vec = weight_vector_part(node)
        mu, sigma = self.norm_params(w_vec, n, p)
        n_nbs = np.random.normal(loc=mu, scale=sigma)
        return np.random.choice(np.arange(n_total), n_nbs, p=np.repeat(w_vec, n)*p, replace=False)
            
    def create_sample(self):    # p0 = (1-beta), p1 = 1
        sample_negative = set(np.random.choice(np.arange(self.n_negative), self.n_negative_sampled, replace=False))
        sample_positive = set(self.n_negative + np.random.choice(np.arange(self.n_positive), self.n_positive_sampled, replace=False))
        self.sample = {n:{} for n in sample_negative}
        self.sample.upadate({n:{} for n in sample_positive})
        
        
        '''ugly due to recent changes'''
        
        for node in sample_negative:
            negative_nbs = set(node_nbs_part(node, n0, p_in_b, self.n_negative))
            self.sample[node]['nb_neg'] = len(negative_nbs)
            self.n_edges[(0,0)] += len(negative_nbs & sample_negative)
            
            positive_nbs = set(node_nbs_part(node, n1, p_out, self.n_positive))
            self.sample[node]['nb_pos'] = len(positive_nbs)
            self.n_edges[(0,1)] += len(positive_nbs & sample_positive)
            
            self.sample[node]['nb1_ratio'] = self.sample[node]['nb_pos'] / self.sample[node]['nb_neg']
        
        for node in sample_positive:
            negative_nbs = set(node_nbs_part(node, n0, p_out, self.n_negative))
            self.sample[node]['nb_neg'] = len(negative_nbs)
            self.n_edges[(1,0)] += len(negative_nbs & sample_negative)
            
            positive_nbs = set(node_nbs_part(node, n1, p_in_a, self.n_positive))
            self.sample[node]['nb_pos'] = len(positive_nbs)
            self.n_edges[(0,1)] += len(positive_nbs & sample_positive)
            
            self.sample[node]['nb1_ratio'] = self.sample[node]['nb_neg'] / self.sample[node]['nb_pos']
        
        cnt0 = self.n_negative
        cnt1 = self.n_positive
        max_00 = (cnt0 * (cnt0 - 1))/2
        max_11 = (cnt1 * (cnt1 - 1))/2
        max_10 = cnt0 * cnt1
        self.p_out_approx = (self.n_edges[(0,1)] + self.n_edges[(1,0)])/(max_10)
        self.p_in_a_approx = self.n_edges[(1,1)] / max_11
        self.p_in_b_approx = self.n_edges[(0,0)] / max_00
            
            
    def vote(self, node):
        return 0 if node < self.n_negative else 1
    
    def nb1_ratio(node):
        return self.sample[node]['nb1_ratio']
        
