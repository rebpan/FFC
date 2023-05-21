import numpy as np
from util import calc_fairness, update_centroid, data_load, relax
import matplotlib.pyplot as plt
from initialization import init
from instance import Instance
from solution import Solution
from improvement import improve
from sklearn.preprocessing import scale


def output_info(X, label, k, centroid, fairness):
    print(fairness)
    print(fairness / np.sum(fairness, axis=1).reshape((-1, 1)))
    SEE = 0
    for j in range(k):
        pts_in_j = np.where(label == j)[0]
        SEE += sum(np.linalg.norm(X[pts_in_j] - centroid[j], axis=1) ** 2)
    print("Sum of squares = " + str(SEE))


def cal_bal_clu(fai, rep_ref):
    bal_clu = np.zeros(len(fai))
    rep_clu = fai / sum(fai)
    for i in range(len(fai)):
        bal_clu[i] = min(rep_clu[i]/rep_ref[i], rep_ref[i]/rep_clu[i])
    return min(bal_clu)

if __name__ == '__main__':

    filename = 'bank'
    X_raw, Color, K, bal_thr = data_load(filename)
    # standardization and normalization
    X = scale(X_raw, axis = 0)
    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X = X/(feanorm[:,None]**0.5)

    n = X.shape[0]
    d = X.shape[1]

    label0 = init(X, Color, K)
    centroid = update_centroid(X, d, K, label0)
    fairness = calc_fairness(label0, K, Color)
    output_info(X, label0, K ,centroid, fairness)
    # fig = plt.figure(1)
    # plt.scatter(X[:, 0], X[:, 1], c=label0)

    # Local search by swapping the labels of two points
    ins = Instance(X, K, Color)
    soln = Solution(ins, label0)
    soln.init_costs()
    max_iter = n
    
    rep_ref = np.zeros(len(np.unique(Color)))
    for i in range(len(np.unique(Color))):
        rep_ref[i] = sum(Color == i) / n

    n_relax1 = 0
    flag, fairness = relax(soln, fairness, rep_ref, bal_thr)
    while flag:
        flag, fairness = relax(soln, fairness, rep_ref, bal_thr)
        n_relax1 += 1
    print("======== relaxation ========")
    label1 = np.copy(soln.label)
    fairness = calc_fairness(label1, K, Color)
    centroid = update_centroid(X, d, K, label1)
    output_info(X, label1, K, centroid, fairness)
    
    soln_first = improve(soln, max_iter, "first improvement")
    print("======== first improvement ========")
    label2 = np.copy(soln_first.label)
    fairness = calc_fairness(label2, K, Color)
    centroid = update_centroid(X, d, K, label2)
    output_info(X, label2, K, centroid, fairness)
    np.savetxt('Results/' + filename + '_FFC2.txt', label2+1, fmt='%d')