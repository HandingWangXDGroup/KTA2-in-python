from scipy.spatial.distance import cdist
from KTA2_in_python.public.initial_pop import initial
from KTA2_in_python.public.individual import Individual
from KTA2_in_python.model_fit.train_model import *
from KTA2_in_python.util.util import mating_selection
from KTA2_in_python.util.util import cross_mutation
from KTA2_in_python.update.update import *
from KTA2_in_python.sampling.adaptive_sampling import adaptive_sampling

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.igd import IGD

from tqdm import tqdm
from copy import deepcopy


def kta2(run_num, problem, max_fes=300, pop_size=100, tau=0.75, phi=0.1, w_max=10, mu=5):
    p = 1 / problem.n_obj

    population, pop_dec, pop_obj = initial(pop_size, problem)
    train_x = pop_dec
    train_y = pop_obj
    ca = update_ca([], new_pop=population, ca_size=pop_size)
    da = deepcopy(population)

    theta_s = 5 * np.ones((problem.n_obj, problem.n_var))
    theta_is = 5 * np.ones((problem.n_obj, 2, problem.n_var))
    fes = pop_size

    pbar = tqdm(desc="{}-th run".format(run_num), total=max_fes)
    pbar.update(fes)

    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=150)
    pf = problem.pareto_front(ref_dirs)
    indicator = IGD(pf)

    while fes < max_fes:
        model_sensitive, theta_s = train_sensitive_model(problem.n_obj, train_x, train_y, theta_s)

        model_insensitive, centers, theta_is = train_insensitive_model(problem.n_obj, train_x,
                                                                       train_y, theta_is, tau)
        cca_obj = np.array([s.obj for s in ca])
        cca_dec = np.array([s.x for s in ca])
        cda_obj = np.array([s.obj for s in da])
        cda_dec = np.array([s.x for s in da])

        for _ in range(w_max):
            parent_c_dec, parent_m_dec = mating_selection(cca_dec, cca_obj, cda_dec, cda_obj, pop_size)
            offspring1_dec = cross_mutation(parent_c_dec, problem, pc=1, pm=0, yita1=20, yita2=0)
            offspring2_dec = cross_mutation(parent_m_dec, problem, pc=0, pm=1, yita1=0, yita2=20)

            sum_pop_dec = np.vstack((cda_dec, cca_dec, offspring1_dec, offspring2_dec))

            pre_pop_obj = np.zeros((sum_pop_dec.shape[0], problem.n_obj))
            var = np.zeros((sum_pop_dec.shape[0], problem.n_obj))

            for i in range(sum_pop_dec.shape[0]):
                for j in range(problem.n_obj):
                    pre_pop_obj[i, j] = model_sensitive[j].predict_values(sum_pop_dec[i, :].reshape(1, -1))
                    if abs(pre_pop_obj[i, j] - centers[j, 0]) <= abs(pre_pop_obj[i, j] - centers[j, 1]):
                        model = model_insensitive[j][0]
                    else:
                        model = model_insensitive[j][1]
                    pre_pop_obj[i, j] = model.predict_values(sum_pop_dec[i, :].reshape(1, -1))
                    var[i, j] = model.predict_variances(sum_pop_dec[i, :].reshape(1, -1))

            cca_dec, cca_obj = update_cca(sum_pop_dec, pre_pop_obj, pop_size)
            cda_dec, cda_obj, cda_var = update_cda(sum_pop_dec, pre_pop_obj, var, pop_size, p)

        new_pop_dec = adaptive_sampling(cca_obj, cca_dec, cda_obj, cda_dec, cda_var, da, mu, p, phi)

        new_pop_dec, _ = np.unique(new_pop_dec, axis=0, return_index=True)
        pop_eff = []
        for i in range(new_pop_dec.shape[0]):
            dist = cdist(np.real(new_pop_dec[i, :].reshape(1, -1)), np.real(train_x), 'euclidean')
            if np.min(dist) > 1e-5:
                pop_eff.append(new_pop_dec[i, :])

        if len(pop_eff) > 0:
            offspring = [Individual(x, problem) for x in pop_eff]

            fes += len(offspring)

            offspring_dec = np.array([s.x for s in offspring])
            offspring_obj = np.array([s.obj for s in offspring])

            train_x = np.vstack((train_x, offspring_dec))
            train_y = np.vstack((train_y, offspring_obj))

            ca = update_ca(ca, offspring, pop_size)
            da = update_da(da, offspring, pop_size, p)
            da_obj = np.array([s.obj for s in da])
            igd_value = indicator(da_obj)

            pbar.set_postfix(DA_IGD="{:.4f}".format(igd_value))
            pbar.update(len(offspring))

    return da
