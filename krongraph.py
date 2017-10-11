import numpy as np
import math


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

        self.kron_sum = [kron[0].sum(), kron[1].sum()]

        self.Kk_size = int(kron.shape[0] ** k)
        self.n_nodes = int((n0 + n1) * self.Kk_size)
        self.n_negative = int(n0 * self.Kk_size)
        self.n_positive = int(self.n_nodes - self.n_negative)

        self.p0 = (1 - self.beta) / (2 - self.beta)
        self.n_negative_sampled = np.random.binomial(self.sample_size, self.p0, 1)[0]
        self.n_positive_sampled = self.sample_size - self.n_negative_sampled

        self.true_p = n1 / (n0 + n1)
        self.sampled_p = self.n_positive_sampled

        self.n_edges = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        
        self
        

        self.create_sample()


    def subsample_params(self, subsample_size):
        # TODO
        pass

    def weight_vector_part(self, node):
        i = node % self.Kk_size
        nodebin = bin(i)[2:].zfill(self.k)
        xi = np.array([1])
        vecsum = 1
        dotprod = 1
        kron_sqr = [self.kron[0].dot(self.kron[0]), self.kron[1].dot(self.kron[1])]
        for i in nodebin:
            if i is '1':
                xi = np.concatenate((xi * self.kron[1][0], xi * self.kron[1][1]))
                vecsum *= self.kron_sum[1]
                dotprod *= kron_sqr[1]
            else:
                xi = np.concatenate((xi * self.kron[0][0], xi * self.kron[0][1]))
                vecsum *= self.kron_sum[0]
                dotprod *= kron_sqr[0]
        return xi, vecsum, dotprod

    @profile
    def node_nbs_part(self, node, n, p, n_total):
        w_vec, vecsum, dotprod = self.weight_vector_part(node)
        mu = vecsum * n * p
        sigma = math.sqrt((vecsum - dotprod * p) * n * p)
        n_nbs = int(np.round(np.random.normal(loc=mu, scale=sigma)))
        if n_nbs < 1: n_nbs = 1
        weights = np.repeat(w_vec, n) * p
        weights = weights / weights.sum()
        return np.random.choice(np.arange(n_total), n_nbs, p=weights, replace=False)

    def create_sample(self):  # p0 = (1-beta), p1 = 1
        sample_negative = set(np.random.choice(np.arange(self.n_negative), self.n_negative_sampled, replace=False))
        sample_positive = set(self.n_negative + np.random.choice(np.arange(self.n_positive), self.n_positive_sampled, replace=False))
        self.sample = {n: {} for n in sample_negative}
        self.sample.update({n: {} for n in sample_positive})

        for node in sample_negative:
            negative_nbs = set(self.node_nbs_part(node, self.n0, self.p_in_b, self.n_negative))
            self.sample[node]['nb_neg'] = len(negative_nbs)
            self.n_edges[(0, 0)] += len(negative_nbs & sample_negative)

            positive_nbs = set(self.node_nbs_part(node, self.n1, self.p_out, self.n_positive) + self.n_negative)
            self.sample[node]['nb_pos'] = len(positive_nbs)
            self.n_edges[(0, 1)] += len(positive_nbs & sample_positive)

            if not self.sample[node]['nb_neg']: self.sample[node]['nb_neg'] = 1
            self.sample[node]['nb1_ratio'] = self.sample[node]['nb_pos'] / self.sample[node]['nb_neg']

        for node in sample_positive:
            negative_nbs = set(self.node_nbs_part(node, self.n0, self.p_out, self.n_negative))
            self.sample[node]['nb_neg'] = len(negative_nbs)
            self.n_edges[(1, 0)] += len(negative_nbs & sample_negative)

            positive_nbs = set(self.node_nbs_part(node, self.n1, self.p_in_a, self.n_positive) + self.n_negative)
            self.sample[node]['nb_pos'] = len(positive_nbs)
            self.n_edges[(1, 1)] += len(positive_nbs & sample_positive)

            if not self.sample[node]['nb_pos']: self.sample[node]['nb_pos'] = 1
            self.sample[node]['nb1_ratio'] = self.sample[node]['nb_neg'] / self.sample[node]['nb_pos']

        cnt0 = self.n_negative
        cnt1 = self.n_positive
        max_00 = (cnt0 * (cnt0 - 1)) / 2
        max_11 = (cnt1 * (cnt1 - 1)) / 2
        max_10 = cnt0 * cnt1
        self.p_out_approx = (self.n_edges[(0, 1)] + self.n_edges[(1, 0)]) / (max_10)
        self.p_in_a_approx = self.n_edges[(1, 1)] / max_11
        self.p_in_b_approx = self.n_edges[(0, 0)] / max_00

    def vote(self, node):
        return 0 if node < self.n_negative else 1

    def nb1_ratio(self, node):
        return self.sample[node]['nb1_ratio']



kron = np.array([[0.770117, 0.794312], [0.794312, 0.0965146]])
k = 16
n0 = 1
n1 = 1
p_in_a = 1
p_in_b = 1
p_out = 0.5
sample_size = 10000
beta = 0


kg = KronGraph(kron, k, n0, n1, p_in_a, p_in_b, p_out, sample_size, beta)
