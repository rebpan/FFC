import numpy as np


class Solution:

    def __init__(self, ins, label):
        self.ins = ins
        self.cluster = {j : list(np.where(label == j)[0]) for j in range(ins.k)}
        # cluster size
        self.clu_sz = [len(self.cluster[j]) for j in range(ins.k)]
        self.label = label
        self.local_obj = np.zeros(ins.k) # sum of pairwise squared distances
        self.assign_cost = np.zeros((ins.n, ins.k))
        self.remove_cost = np.zeros(ins.n)

        # demographic
        uq_sens = np.unique(ins.color)
        n_sens = len(uq_sens)
        self.demographic = np.zeros((ins.k, n_sens), dtype='int32')
        for i in range(ins.n):
            if label[i] == -1:
                continue
            self.demographic[label[i], ins.color[i]] += 1

    def init_costs(self):
        self.calc_obj()
        self.init_remove()
        self.init_assign()

    def calc_obj(self):
        self.local_obj = [0] * self.ins.k
        for j, points_in_cluster in self.cluster.items():
            for a in range(len(points_in_cluster) - 1):
                for b in range(a + 1, len(points_in_cluster)):
                    self.local_obj[j] += self.ins.distance[points_in_cluster[a], points_in_cluster[b]]

    def calc_opt(self):
        opt = 0
        for j in range(self.ins.k):
            opt += self.local_obj[j] / self.clu_sz[j]
        return opt

    def init_remove(self):
        for i in range(self.ins.n):
            self.remove_cost[i] = self.calc_remove_cost(i)

    def init_assign(self):
        for i in range(self.ins.n):
            for j in range(self.ins.k):
                self.assign_cost[i, j] = self.calc_assign_cost(i, j)

    def calc_remove_cost(self, i):
        assert self.label[i] != -1 # check if a point is assigned to a cluster
        j = self.label[i]
        points_in_cluster = self.cluster[j]
        tmp = 0
        for r in points_in_cluster:
            tmp -= self.ins.distance[i, r]
        return tmp

    def calc_assign_cost(self, i, j):
        if self.label[i] == j:
            return 0
        points_in_cluster = self.cluster[j]
        tmp = 0
        for r in points_in_cluster:
            tmp += self.ins.distance[i, r]
        return tmp

    def calc_swap_cost(self, a, b):
        ka = self.label[a]
        kb = self.label[b]
        return [self.remove_cost[a] + self.assign_cost[b, ka] - self.ins.distance[b, a], \
                self.remove_cost[b] + self.assign_cost[a, kb] - self.ins.distance[a, b]], \
               [(self.remove_cost[a] + self.assign_cost[b, ka] - self.ins.distance[b, a]) / self.clu_sz[ka], \
                (self.remove_cost[b] + self.assign_cost[a, kb] - self.ins.distance[a, b]) / self.clu_sz[kb]]

    def swap(self, a, b, swap_cost):
        assert a >= 0 and b >= 0, 'invalid point id'
        assert self.label[a] != -1 and self.label[b] != -1, 'not assigned'
        assert self.label[a] != self.label[b], 'both points are in the same cluster'
        ka = self.label[a]
        kb = self.label[b]

        # update cluster ka and the label of point b
        self.cluster[ka].remove(a)
        self.cluster[ka].append(b)
        self.label[b] = ka
        self.local_obj[ka] += swap_cost[0]

        # update cluster kb and the label of point a
        self.cluster[kb].remove(b)
        self.cluster[kb].append(a)
        self.label[a] = kb
        self.local_obj[kb] += swap_cost[1]

        # update the remove cost of points in ka (kb) and the assign cost of points not in ka (kb)
        points_in_cluster = self.cluster[ka]
        for r in points_in_cluster:
            self.remove_cost[r] += self.ins.distance[r, a] - self.ins.distance[r, b] # remove_cost_of_r - dist_to_b + dist_to_a
        points_in_cluster = self.cluster[kb]
        for r in points_in_cluster:
            self.remove_cost[r] += self.ins.distance[r, b] - self.ins.distance[r, a] # remove_cost_of_r - dist_to_a + dist_to_b
        for i in range(self.ins.n):
            if self.label[i] != ka:
                self.assign_cost[i, ka] += self.ins.distance[i, b] - self.ins.distance[i, a] # assign_cost_of_i + dist_to_b - dist_to_a
            if self.label[i] != kb:
                self.assign_cost[i, kb] += self.ins.distance[i, a] - self.ins.distance[i, b] # assign_cost_of_i + dist_to_a - dist_to_b

        # update the costs of two swap points
        self.remove_cost[b] = self.calc_remove_cost(b)
        self.assign_cost[b, kb] = self.calc_assign_cost(b, kb)
        self.assign_cost[b, ka] = 0
        self.remove_cost[a] = self.calc_remove_cost(a)
        self.assign_cost[a, ka] = self.calc_assign_cost(a, ka)
        self.assign_cost[a, kb] = 0

    def calc_reassign_cost(self, i, j):
        ki = self.label[i]
        return [self.remove_cost[i], self.assign_cost[i, j]],\
                [((self.local_obj[ki]+self.remove_cost[i])/(self.clu_sz[ki]-1))-(self.local_obj[ki]/self.clu_sz[ki]),\
                 (self.local_obj[j]+self.assign_cost[i, j])/(self.clu_sz[j]+1)-(self.local_obj[j]/self.clu_sz[j])]

    def reassign(self, i, j):
        ki = self.label[i]
        ci = self.ins.color[i]

        # update cluster ki
        self.cluster[ki].remove(i)
        self.clu_sz[ki] -= 1
        self.local_obj[ki] += self.remove_cost[i]
        self.demographic[ki, ci] -= 1

        # update cluster j and the label of point i
        self.cluster[j].append(i)
        self.clu_sz[j] += 1
        self.local_obj[j] += self.assign_cost[i, j]
        self.label[i] = j
        self.demographic[j, ci] += 1

        # update the remove cost of points in ki (j) and the assign cost of points not in ki (j)
        points_in_cluster = self.cluster[ki]
        for r in points_in_cluster:
            self.remove_cost[r] += self.ins.distance[r, i] # remove_cost_of_r + dist_to_i
        points_in_cluster = self.cluster[j]
        for r in points_in_cluster:
            self.remove_cost[r] -= self.ins.distance[r, i] # remove_cost_of_r - dist_to_i
        for a in range(self.ins.n):
            if self.label[a] != ki:
                self.assign_cost[a, ki] -= self.ins.distance[a, i] # assign_cost_of_a - dist_to_i
            if self.label[a] != j:
                self.assign_cost[a, j] += self.ins.distance[a, i] # assign_cost_of_a + dist_to_i

        # update the costs of the point
        self.remove_cost[i] = self.calc_remove_cost(i)
        self.assign_cost[i, ki] = self.calc_assign_cost(i, ki)
        self.assign_cost[i, j] = 0
