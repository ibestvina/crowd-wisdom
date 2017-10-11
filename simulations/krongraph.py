import numpy as np
import math


class KronGraph:
    def __init__(self, kron, k, n0, n1, p_in_a, p_in_b, p_out, sample_perc, beta, logger=print):
        self.kron = kron
        self.k = k
        self.n0 = n0
        self.n1 = n1
        self.p_in_a = p_in_a
        self.p_in_b = p_in_b
        self.p_out = p_out
        self.beta = beta
        self.logger = logger

        self.kron_sum = [kron[0].sum(), kron[1].sum()]
        self.kron_sqr = [self.kron[0].dot(self.kron[0]), self.kron[1].dot(self.kron[1])]

        self.Kk_size = int(kron.shape[0] ** k)
        self.n_nodes = int((n0 + n1) * self.Kk_size)
        self.n_negative = int(n0 * self.Kk_size)
        self.n_positive = int(self.n_nodes - self.n_negative)
        self.sample_size = int(self.n_nodes * sample_perc)

        self.p0 = (1 - self.beta) / (2 - self.beta)
        self.n_negative_sampled = np.random.binomial(self.sample_size, self.p0, 1)[0]
        self.n_positive_sampled = self.sample_size - self.n_negative_sampled

        self.true_p = n1 / (n0 + n1)
        self.sampled_p = self.n_positive_sampled

        self.create_sample()

    def weight_vector_part(self, node):
        i = node % self.Kk_size
        nodebin = bin(i)[2:].zfill(self.k)
        xi = np.array([1])
        vecsum = 1
        dotprod = 1

        for i in nodebin:
            if i is '1':
                xi = np.concatenate((xi * self.kron[1][0], xi * self.kron[1][1]))
                vecsum *= self.kron_sum[1]
                dotprod *= self.kron_sqr[1]
            else:
                xi = np.concatenate((xi * self.kron[0][0], xi * self.kron[0][1]))
                vecsum *= self.kron_sum[0]
                dotprod *= self.kron_sqr[0]
        return xi, vecsum, dotprod

    def node_nbs_part(self, node, n, p, n_total):
        w_vec, vecsum, dotprod = self.weight_vector_part(node)
        mu = vecsum * n * p
        sigma = math.sqrt((vecsum - dotprod * p) * n * p)
        n_nbs = int(np.round(np.random.normal(loc=mu, scale=sigma)))
        if n_nbs < 1: n_nbs = 1
        w_vec = w_vec / (vecsum * n)
        weights = np.repeat(w_vec, n)
        return np.random.choice(np.arange(n_total), n_nbs, p=weights, replace=False)

    def create_sample(self):  # p0 = (1-beta), p1 = 1
        self.logger('Create sample started for sample size ' + str(self.sample_size))
        sample_negative = set(np.random.choice(np.arange(self.n_negative), self.n_negative_sampled, replace=False))
        sample_positive = set(
            self.n_negative + np.random.choice(np.arange(self.n_positive), self.n_positive_sampled, replace=False))
        self.sample = {n: {} for n in sample_negative}
        self.sample.update({n: {} for n in sample_positive})

        nodes_to_calc = len(self.sample)
        print_each = int(nodes_to_calc * 0.1)
        i = 0
        for node in sample_negative:
            i += 1
            if not i % print_each: self.logger('Calulating node {}/{}'.format(i, nodes_to_calc))
            negative_nbs = set(self.node_nbs_part(node, self.n0, self.p_in_b, self.n_negative))
            self.sample[node]['nb_neg'] = len(negative_nbs)
            self.sample[node]['nb_neg_samp'] = len(negative_nbs & sample_negative)

            positive_nbs = set(self.node_nbs_part(node, self.n1, self.p_out, self.n_positive) + self.n_negative)
            self.sample[node]['nb_pos'] = len(positive_nbs)
            self.sample[node]['nb_pos_samp'] = len(positive_nbs & sample_positive)

            if not self.sample[node]['nb_neg']: self.sample[node]['nb_neg'] = 1
            self.sample[node]['nb1_ratio'] = self.sample[node]['nb_pos'] / self.sample[node]['nb_neg']

        for node in sample_positive:
            i += 1
            if not i % print_each: self.logger('Calulating node {}/{}'.format(i, nodes_to_calc))
            negative_nbs = set(self.node_nbs_part(node, self.n0, self.p_out, self.n_negative))
            self.sample[node]['nb_neg'] = len(negative_nbs)
            self.sample[node]['nb_neg_samp'] = len(negative_nbs & sample_negative)

            positive_nbs = set(self.node_nbs_part(node, self.n1, self.p_in_a, self.n_positive) + self.n_negative)
            self.sample[node]['nb_pos'] = len(positive_nbs)
            self.sample[node]['nb_pos_samp'] = len(positive_nbs & sample_positive)

            if not self.sample[node]['nb_pos']: self.sample[node]['nb_pos'] = 1
            self.sample[node]['nb1_ratio'] = self.sample[node]['nb_neg'] / self.sample[node]['nb_pos']

    def vote(self, node):
        return 0 if node < self.n_negative else 1

    def nb1_ratio(self, node):
        return self.sample[node]['nb1_ratio']

    def subsample_params_calc(self, subsample_perc):
        subsample_size = int(self.n_nodes * subsample_perc)
        if subsample_size > self.sample_size:
            self.logger('Subsample larger than sample!')
            return
        subsample = np.random.choice(list(self.sample.keys()), subsample_size, replace=False)
        self.q_approx(subsample)
        self.p_approx(subsample)

    def calc_q(self, a, b):
        return (-b + math.sqrt(a * b)) / (a - b)

    def q_approx(self, subsample):
        nb1_ratio_sum = [0, 0]
        nb1_ratio_cnt = [0, 0]
        for node in subsample:
            node_vote = self.vote(node)
            nb1_ratio_sum[node_vote] += self.sample[node]['nb1_ratio']
            nb1_ratio_cnt[node_vote] += 1

        self.mu_b = nb1_ratio_sum[0] / nb1_ratio_cnt[0]
        self.mu_a = nb1_ratio_sum[1] / nb1_ratio_cnt[1]
        self.q = self.calc_q(self.mu_a, self.mu_b)

    def p_approx(self, subsample):
        n_edges = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for node in subsample:
            node_vote = self.vote(node)
            n_edges[(node_vote, 0)] += self.sample[node]['nb_neg_samp']
            n_edges[(node_vote, 1)] += self.sample[node]['nb_pos_samp']

        cnt0 = self.n_negative
        cnt1 = self.n_positive
        max_00 = (cnt0 * (cnt0 - 1)) / 2
        max_11 = (cnt1 * (cnt1 - 1)) / 2
        max_10 = cnt0 * cnt1
        self.p_out_approx = (n_edges[(0, 1)] + n_edges[(1, 0)]) / (max_10)
        self.p_in_a_approx = n_edges[(1, 1)] / max_11
        self.p_in_b_approx = n_edges[(0, 0)] / max_00
        Ra = self.p_out / self.p_in_a
        Rb = self.p_out / self.p_in_b
        x1 = Ra / (self.mu_a + Ra)
        x2 = self.mu_b / (self.mu_b + Rb)
        self.q_p_app = (x1 + x2) / 2


if (False):
    kron = np.array([[0.770117, 0.794312], [0.794312, 0.0965146]])
    k = 18
    n0 = 2
    n1 = 1
    p_in_a = 1
    p_in_b = 1
    p_out = 0.5
    sample_perc = 0.01
    beta = 0.8

    kg = KronGraph(kron, k, n0, n1, p_in_a, p_in_b, p_out, sample_perc, beta)
    kg.subsample_params_calc(sample_perc)

    a = kg.mu_a
    b = kg.mu_b
    print('a:', a)
    print('b:', b)

    q = (-b + math.sqrt(a * b)) / (a - b)
    print('q:', q)

    print('p in a:', kg.p_in_a_approx)
    print('p in b:', kg.p_in_b_approx)
    print('p out:', kg.p_out_approx)

    Ra = p_out / p_in_a
    Rb = p_out / p_in_b
    x1 = Ra / (a + Ra)
    x2 = b / (b + Rb)
    q_p_app = (x1 + x2) / 2
    print('q p_app:', q_p_app)