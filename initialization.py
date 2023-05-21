import numpy as np
from sklearn.cluster import KMeans
from util import calc_fairness, faster_permutations, update_centroid
import math


def init(X, color, k):
    d = X.shape[1]
    uq_color = np.unique(color)
    idx = [np.where(color==i)[0] for i in uq_color]
    sz_group = [len(i) for i in idx]
    dec_ind = np.argsort(sz_group)[::-1]

    idx_main = idx[dec_ind[0]]
    X_main = X[idx_main]
    km_main = KMeans(n_clusters=k).fit(X_main)
    lb_main = km_main.labels_
    C_main = km_main.cluster_centers_

    label_final = np.zeros(len(X), dtype='int32') - 1
    label_final[idx_main] = lb_main
    for i in dec_ind[1:]:
        idx_curr = idx[i]
        X_curr = X[idx_curr]
        km_curr = KMeans(n_clusters=k).fit(X_curr)
        lb_curr = km_curr.labels_
        C_curr = km_curr.cluster_centers_

        # combine two groups
        v = faster_permutations(k)
        m_main = v.ravel()
        m_curr = np.tile(np.arange(k), len(m_main)//k)
        dists = np.sum((C_main[m_main] - C_curr[m_curr]) ** 2, axis=1)
        dists2 = np.reshape(dists, v.shape)
        a = np.argmin(np.sum(dists2, axis=1))
        m_main = m_main[a*k:(a+1)*k]
        m_curr = m_curr[a*k:(a+1)*k]
        lb_curr2 = np.empty(len(lb_curr), dtype='int32', order='F')
        C_curr2 = np.empty_like(C_curr)
        for j in range(k):
            lb_curr2[np.where(lb_curr == m_curr[j])] = m_main[j]
            C_curr2[m_main[j]] = C_curr[m_curr[j]]
        label_final[idx_curr] = lb_curr2

        # Adjustment
        unmarked = [j for j in range(k)]
        ratio = len(idx_main) / len(idx_curr)
        strenght = uq_color[dec_ind[0]]
        weakness = uq_color[i]
        fairness = calc_fairness(label_final, k, color)
        while len(unmarked) > 1:
            t_cluster = np.argmin(fairness[unmarked, weakness]/fairness[unmarked, strenght])
            t_cluster = unmarked[t_cluster]
            lack_of_pts = round(fairness[t_cluster, strenght] / ratio) - fairness[t_cluster, weakness]
            # lack_of_pts = math.ceil(fairness[t_cluster, strenght] / ratio) - fairness[t_cluster, weakness]
            dists3 = np.linalg.norm(C_curr2[unmarked] - C_curr2[t_cluster], axis=1)
            f_cluster = np.argsort(dists3)[1]
            f_cluster = unmarked[f_cluster]
            if lack_of_pts >= len(np.where(lb_curr2 == f_cluster)[0]):
                lack_of_pts = len(np.where(lb_curr2 == f_cluster)[0]) - 1
                cand_idx_curr = np.where(lb_curr2 == f_cluster)[0]
                cand = idx_curr[cand_idx_curr]
                X_cand = X[cand]
                dists3 = np.linalg.norm(X_cand - C_curr2[t_cluster], axis=1)
                cand2_idx_curr = cand_idx_curr[np.argsort(dists3)[:lack_of_pts]]
                lb_curr2[cand2_idx_curr] = t_cluster
                C_curr2 = update_centroid(X_curr, d, k, lb_curr2); # update sub centers
                cand2 = cand[np.argsort(dists3)[:lack_of_pts]]
                label_final[cand2] = t_cluster
                fairness[t_cluster, weakness] += lack_of_pts
                fairness[f_cluster, weakness] -= lack_of_pts
                continue
            # find which points to move
            cand_idx_curr = np.where(lb_curr2 == f_cluster)[0]
            cand = idx_curr[cand_idx_curr]
            X_cand = X[cand]
            dists3 = np.linalg.norm(X_cand - C_curr2[t_cluster], axis=1)
            cand2_idx_curr = cand_idx_curr[np.argsort(dists3)[:lack_of_pts]]
            lb_curr2[cand2_idx_curr] = t_cluster
            C_curr2 = update_centroid(X_curr, d, k, lb_curr2); # update sub centers
            cand2 = cand[np.argsort(dists3)[:lack_of_pts]]
            label_final[cand2] = t_cluster
            fairness[t_cluster, weakness] += lack_of_pts
            fairness[f_cluster, weakness] -= lack_of_pts
            unmarked.remove(t_cluster)
        
        C_main = update_centroid(X, d, k, label_final)
    return label_final
