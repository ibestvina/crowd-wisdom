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
        
        self.Kk_size = kron.shape[0]**k
        self.n_nodes = (n0+n1) * self.Kk_size
        self.n_negative = n0 * self.Kk_size
        
        self.sample(sample_size, beta)

    
    def sample(self, sample_size, beta):    # p0 = (1-beta), p1 = 1
        p0 = (1-beta) / (2-beta)
        n_sample_negative = binomial(sample_size, p0, 1)[0]
        sample_negative = bin_arr(self.n_negative, n_sample_negative)
        n_sample_positive = sample_size - n_sample_negative
        sample_positive = bin_arr(self.n_nodes - self.n_negative, n_sample_positive)
        
          
    def vote(self, node):
        return 0 if node < self.n_negatives else 1
        