import numpy as np
from KTA2_in_python.public.individual import Individual


def lhs(m, n):
    sampling_data = np.zeros((m, n))
    for i in range(n):
        sampling_data[:, i] = (np.random.rand(m) + np.random.permutation(m)) / m

    return sampling_data


def initial(pop_size, problem):

    sample_data = lhs(pop_size, problem.n_var)
    lower = np.tile(problem.xl, (pop_size, 1))
    upper = np.tile(problem.xu, (pop_size, 1))
    pop_dec = lower + (upper - lower) * sample_data
    pop = [Individual(x, problem) for x in pop_dec]
    pop_obj = np.array([ind.obj for ind in pop])

    return pop, pop_dec, pop_obj

