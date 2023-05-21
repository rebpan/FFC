import sys
import numpy as np
import copy


def improve(soln, max_iter, method="first improvement"):
    soln_copy = copy.deepcopy(soln)
    if method == "best improvement":
        iteration = best_local_search
    else:
        iteration = first_local_search
    it = 0
    flag = True
    while it < max_iter and flag:
        print('Improvement rounds: ' + str(it))
        flag = iteration(soln_copy)
        it += 1
    return soln_copy


def first_local_search(soln):
    for a in range(soln.ins.n - 1):
        for b in range(a + 1, soln.ins.n):
            if soln.label[a] == soln.label[b] or soln.ins.color[a] != soln.ins.color[b]:
                continue
            swap_cost, swap_cost2 = soln.calc_swap_cost(a, b)
            if sum(swap_cost2) < 0:
                soln.swap(a, b, swap_cost)
                return True
    return False


def best_local_search(soln):
    x = y = -1
    min_cost = []
    min_cost2 = []
    for a in range(soln.ins.n - 1):
        for b in range(a + 1, soln.ins.n):
            if soln.label[a] == soln.label[b] or soln.ins.color[a] != soln.ins.color[b]:
                continue
            swap_cost, swap_cost2 = soln.calc_swap_cost(a, b)
            if sum(swap_cost2) < sum(min_cost2):
                x = a
                y = b
                min_cost = swap_cost
                min_cost2 = swap_cost2
    swap_flag = len(min_cost2) > 0
    if swap_flag:
        soln.swap(x, y, min_cost)
    return swap_flag