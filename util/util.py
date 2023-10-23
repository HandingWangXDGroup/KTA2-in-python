import os
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


def non_dominated_sort(pop_obj, front_num):
    unique_pop_obj, unique_indices = np.unique(pop_obj, axis=0, return_inverse=True)
    table, _ = np.histogram(unique_indices, bins=(max(unique_indices) + 1))

    unique_pop_num, obj_num = unique_pop_obj.shape
    front_flag = np.full(unique_pop_num, np.inf)
    max_front_flag = 0

    while np.sum(table[front_flag < np.inf]) < min(front_num, len(unique_indices)):
        max_front_flag += 1
        for i in range(unique_pop_num):
            if front_flag[i] == np.inf:
                dominated = False
                for j in range(i - 1, -1, -1):
                    if front_flag[j] == max_front_flag:
                        if np.sum(unique_pop_obj[i, :] >= unique_pop_obj[j, :]) == obj_num:
                            dominated = True
                        if dominated or obj_num == 2:
                            break
                if not dominated:
                    front_flag[i] = max_front_flag

    non_dominated_ind = front_flag[unique_indices]
    return non_dominated_ind


def statsrexact(v, w):
    n = len(v)
    v = np.sort(v)

    max_w = n * (n + 1) / 2
    folded = w > max_w / 2
    if folded:
        w = max_w - w

    doubled = np.any(v != np.floor(v))
    if doubled:
        v = np.round(2 * v)
        w = np.round(2 * w)

    C = np.zeros(w + 1)
    C[0] = 1
    top = 1

    for vj in v[v <= w]:
        new_top = min(top + vj, w + 1)
        hi = np.arange(min(vj, w + 1), new_top)
        lo = np.arange(0, len(hi))

        C[hi] = C[hi] + C[lo]
        top = new_top

    C = C / (2 ** n)
    p_val = np.sum(C)

    all_w = np.arange(0, w + 1)
    if doubled:
        all_w = all_w / 2
    if folded:
        all_w = n * (n + 1) / 2 - all_w

    P = np.column_stack((all_w, C))

    return p_val


def sign_rank(x, y, alpha=0.05, method="auto"):
    diff_xy = x - y
    eps_diff = np.finfo(float).eps * (np.abs(x) + np.abs(y))

    t = np.isnan(diff_xy)
    diff_xy = diff_xy[~t]
    eps_diff = eps_diff[~t]

    t = np.abs(diff_xy) < eps_diff
    diff_xy = diff_xy[~t]

    n = len(diff_xy)
    if n == 0:
        p, h = 1, 0
        return p, h

    if method == "auto":
        if n <= 15:
            method = 'exact'
        else:
            method = 'approximate'

    neg = diff_xy < 0
    tie_rank = stats.rankdata(np.abs(diff_xy), method='average')

    w = np.sum(tie_rank[neg]).astype("int64")
    r1 = w
    r2 = (n * (n + 1) / 2 - w).astype("int64")
    w = min(w, n * (n + 1) / 2 - w).astype("int64")

    if method == 'approximate':
        z = (w - n * (n + 1) / 4) / np.sqrt((n * (n + 1) * (2 * n + 1)) / 24)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:  # method == 'exact'
        p, _ = statsrexact(tie_rank, w)
        p = min(1, 2 * p)

    h = int(p <= alpha)

    return p, h, r1, r2


def cal_convergence(pop1_obj, pop2_obj, min_point):
    if pop1_obj.shape[0] != pop2_obj.shape[0]:
        flag = 0
    else:
        pop_obj = np.vstack((pop1_obj, pop2_obj))
        pop_obj_norm = (pop_obj - min_point) / (pop_obj.max(axis=0) - min_point)

        distance1 = np.sqrt(np.sum(pop_obj_norm[:pop1_obj.shape[0], :], axis=1))
        distance2 = np.sqrt(np.sum(pop_obj_norm[pop1_obj.shape[0]:, :], axis=1))

        _, flag, r1, r2 = sign_rank(distance1, distance2)
        if flag == 1 and (r1 - r2 < 0):
            flag = 0

    return flag


def pure_diversity(pop_obj):
    if pop_obj.size == 0:
        score = np.nan
    else:
        n = pop_obj.shape[0]
        c = np.zeros((n, n), dtype=bool)
        np.fill_diagonal(c, True)

        dist = cdist(pop_obj, pop_obj, metric='minkowski', p=0.1)
        np.fill_diagonal(dist, np.inf)
        score = 0

        for _ in range(n - 1):
            while True:
                d = np.min(dist, axis=1)
                j = np.argmin(dist, axis=1)
                i = np.argmax(d)

                if dist[j[i], i] != -np.inf:
                    dist[j[i], i] = np.inf
                if dist[i, j[i]] != -np.inf:
                    dist[i, j[i]] = np.inf
                p = np.any(c[i, :].reshape(1, -1), axis=0)
                while not p[j[i]]:
                    new_p = np.any(c[p, :], axis=0)
                    if np.array_equal(p, new_p):
                        break
                    else:
                        p = new_p
                if not p[j[i]]:
                    break
            c[i, j[i]] = True
            c[j[i], i] = True
            dist[i, :] = -np.inf
            score += d[i]

    return score


def mating_selection(ca_dec, ca_obj, da_dec, da_obj, parent_size):
    ca_parent1_index = np.random.randint(0, ca_obj.shape[0], size=int(np.ceil(parent_size / 2)))
    ca_parent2_index = np.random.randint(0, ca_obj.shape[0], size=int(np.ceil(parent_size / 2)))

    a1 = 0 + np.any(ca_obj[ca_parent1_index, :] < ca_obj[ca_parent2_index, :], axis=1)
    a2 = 0 + np.any(ca_obj[ca_parent1_index, :] > ca_obj[ca_parent2_index, :], axis=1)
    dominate = a1 - a2

    choose_index = np.concatenate((ca_parent1_index[dominate == 1], ca_parent2_index[dominate != 1]))
    parent_c_obj = ca_obj[choose_index, :]
    parent_c_dec = ca_dec[choose_index, :]

    parent_c_obj = np.vstack((parent_c_obj, da_obj[np.random.randint(0, da_obj.shape[0],
                                                                     size=int(np.ceil(parent_size / 2))), :]))
    parent_c_dec = np.vstack((parent_c_dec, da_dec[np.random.randint(0, da_dec.shape[0],
                                                                     size=int(np.ceil(parent_size / 2))), :]))
    parent_m_obj = ca_obj[np.random.randint(0, ca_obj.shape[0], size=parent_size), :]
    parent_m_dec = ca_dec[np.random.randint(0, ca_dec.shape[0], size=parent_size), :]

    return parent_c_dec, parent_m_dec


def cross_mutation(parent, problem, pc, pm, yita1, yita2):
    parent1 = parent[:len(parent) // 2, :]
    parent2 = parent[len(parent) // 2:, :]
    n, d = parent1.shape

    # Simulated binary crossover
    beta = np.zeros((n, d))
    mu = np.random.rand(n, d)

    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (yita1 + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (yita1 + 1))
    beta = beta * ((-1) ** np.random.randint(0, 2, size=(n, d)))

    beta[np.random.rand(n, d) < 0.5] = 1
    beta[np.random.rand(n, d) > pc] = 1

    off_cross = np.vstack([(parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
                           (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2])

    # Polynomial mutation
    lower = np.tile(problem.xl, (2 * n, 1))
    upper = np.tile(problem.xu, (2 * n, 1))
    site = np.random.rand(2 * n, d) < (pm / d)
    mu = np.random.rand(2 * n, d)

    temp = site & (mu <= 0.5)
    off_mu = np.minimum(np.maximum(off_cross, lower), upper)
    off_mu[temp] = off_mu[temp] + (upper[temp] - lower[temp]) * (
            (2 * mu[temp] + (1 - 2 * mu[temp]) *
             (1 - (off_mu[temp] - lower[temp]) / (upper[temp] - lower[temp])) ** (yita2 + 1)) ** (1 / (yita2 + 1)) - 1)

    temp = site & (mu > 0.5)
    off_mu[temp] = off_mu[temp] + (upper[temp] - lower[temp]) * (
            1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) *
                 (1 - (upper[temp] - off_mu[temp]) / (upper[temp] - lower[temp])) ** (yita2 + 1)) ** (1 / (yita2 + 1)))

    return off_mu


def mk_dir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass
