import numpy as np


def data_load(filename):
    if filename == "elliptical":
        X = np.loadtxt(r"./Datasets/elliptical.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/elliptical_Color.txt", dtype=int)
        K = 2
        bal_thr = 0.8878
    elif filename == "DS577":
        X = np.loadtxt(r"./Datasets/DS577.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/DS577_Color.txt", dtype=int)
        K = 3
        bal_thr = 0.7988
    elif filename == "2d-4c-no0":
        X = np.loadtxt(r"./Datasets/2d-4c-no0.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/2d-4c-no0_Color.txt", dtype=int)
        K = 4
        bal_thr = 0.7719
    elif filename == "2d-4c-no1":
        X = np.loadtxt(r"./Datasets/2d-4c-no1.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/2d-4c-no1_Color.txt", dtype=int)
        K = 4
        bal_thr = 0.7780
    elif filename == "2d-4c-no4":
        X = np.loadtxt(r"./Datasets/2d-4c-no4.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/2d-4c-no4_Color.txt", dtype=int)
        K = 4
        bal_thr = 0.7369
    elif filename == "adult":
        X = np.loadtxt(r"./Datasets/subsampled_adult.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/subsampled_adult_Color.txt", dtype=int)
        K = 5
        bal_thr = 0.9405
    elif filename == "bank":
        X = np.loadtxt(r"./Datasets/subsampled_bank.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/subsampled_bank_Color.txt", dtype=int)
        K = 6
        bal_thr = 0.8076
    elif filename == "census1990":
        X = np.loadtxt(r"./Datasets/subsampled_census1990.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/subsampled_census1990_Color.txt", dtype=int)
        K = 5
        bal_thr = 0.9119
    elif filename == "creditcard":
        X = np.loadtxt(r"./Datasets/subsampled_creditcard.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/subsampled_creditcard_Color.txt", dtype=int)
        K = 5
        bal_thr = 0.9259
    elif filename == "diabetic":
        X = np.loadtxt(r"./Datasets/subsampled_diabetic.txt", delimiter=",")
        Color = np.loadtxt(r"./Datasets/subsampled_diabetic_Color.txt", dtype=int)
        K = 10
        bal_thr = 0.8728
    else:
        return
    return X, Color, K, bal_thr


def get_random_fairlet(idx_blue, idx_red, B, R, size1, size2):
    pts_blue = list(idx_blue[B - size1 : B])
    pts_red = list(idx_red[R - size2 : R])
    return (pts_blue, pts_red)


def random_fairlet_decompose(idx_blue, idx_red, b, r):
    B = len(idx_blue)
    R = len(idx_red)
    if float(B / R) < float(b / r):
        raise Exception("the balance of the original set is not big enough!")
    fairlets = []
    idx_blue = np.random.permutation(idx_blue)
    idx_red = np.random.permutation(idx_red)

    while R - B > r - b:
        new_fairlet = get_random_fairlet(idx_blue, idx_red, B, R, b, r)
        B -= b
        R -= r
        fairlets.append(new_fairlet)

    if R - B > 0:
        new_fairlet = get_random_fairlet(idx_blue, idx_red, B, R, b, R - B + b)
        B -= b
        R = B
        fairlets.append(new_fairlet)

    if R != B:
        raise Exception("R and B don't match!")

    for i in range(B):
        new_fairlet = get_random_fairlet(idx_blue, idx_red, B, R, 1, 1)
        B -= 1
        R -= 1
        fairlets.append(new_fairlet)

    return fairlets


def calc_fairness(label, k, sensitive):
    """
    Arguements:
    label: 0...k
    k: the number of clusters
    sensitive: sensitive attributes of data
    """
    n = len(label)
    # label = np.array(label)
    uq_sens = np.unique(sensitive)
    n_sens = len(uq_sens)
    fairness = np.zeros((k, n_sens), dtype='int32')
    for i in range(n):
        if label[i] == -1:
            continue
        fairness[label[i], sensitive[i]] += 1
    return fairness


def faster_permutations(n):
    # empty() is fast because it does not initialize the values of the array
    # order='F' uses Fortran ordering, which makes accessing elements in the same column fast
    perms = np.empty((np.math.factorial(n), n), dtype=np.uint8, order='F')
    perms[0, 0] = 0

    rows_to_copy = 1
    for i in range(1, n):
        perms[:rows_to_copy, i] = i
        for j in range(1, i + 1):
            start_row = rows_to_copy * j
            end_row = rows_to_copy * (j + 1)
            splitter = i - j
            perms[start_row: end_row, splitter] = i
            perms[start_row: end_row, :splitter] = perms[:rows_to_copy, :splitter]  # left side
            perms[start_row: end_row, splitter + 1:i + 1] = perms[:rows_to_copy, splitter:i]  # right side

        rows_to_copy *= i + 1

    return perms


def update_centroid(X, d, k, label):
    centroid = np.empty((k, d))
    for j in range(k):
        label_j = np.where(label == j)[0]
        centroid[j] = np.mean(X[label_j], axis=0)
    return centroid


def cal_bal_clu(fai, rep_ref):
    bal_clu = np.zeros(len(fai))
    rep_clu = fai / sum(fai)
    for i in range(len(fai)):
        bal_clu[i] = min(rep_clu[i]/rep_ref[i], rep_ref[i]/rep_clu[i])
    return min(bal_clu)

def relax(soln, fairness, rep_ref, bal_thr):
    point = -1
    to = -1
    min_cost = []
    min_cost2 = []
    for i in range(soln.ins.n):
        ki = soln.label[i]
        ci = soln.ins.color[i]
        fai_1 = np.copy(fairness[ki])
        fai_1[ci] -= 1
        
        if cal_bal_clu(fai_1, rep_ref) <= bal_thr:
            continue
        for j in range(soln.ins.k):
            if j == ki:
                continue
            fai_2 = np.copy(fairness[j])
            fai_2[ci] += 1
            if cal_bal_clu(fai_2, rep_ref) <= bal_thr:
                continue
            assign_cost, assign_cost2 = soln.calc_reassign_cost(i, j)
            if sum(assign_cost2) < sum(min_cost2):
                point = i
                to = j
                min_cost = assign_cost
                min_cost2 = assign_cost2
    reassign_flag = len(min_cost2) > 0
    if reassign_flag:
        k_point = soln.label[point]
        k_to = to
        c_point = soln.ins.color[point]
        print(point, k_to)
        soln.reassign(point, to)
        
        fairness[k_point, c_point] -= 1
        fairness[k_to, c_point] += 1
    return reassign_flag, fairness