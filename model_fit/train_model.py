import warnings
import numpy as np
from smt.surrogate_models import KRG


def train_sensitive_model(obj_num, train_x, train_y, theta):
    model_sensitive = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(obj_num):
            model = KRG(poly='constant', corr='squar_exp', theta0=theta[i, :], theta_bounds=[1e-5, 100],
                        print_global=False)
            model.set_training_values(train_x, train_y[:, i])
            model.train()

            theta[i, :] = model.optimal_theta
            model_sensitive.append(model)

    return model_sensitive, theta


def train_insensitive_model(obj_num, train_x, train_y, theta, tau):
    model_insensitive = []
    centers = np.zeros((obj_num, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(obj_num):
            model_insensitive.append([])
            sort_index = np.argsort(train_y[:, i])
            num = int(np.ceil(train_x.shape[0] * tau))
            min_max_index = [sort_index[:num], sort_index[-num:]]

            for j in range(2):
                sub_train_x = train_x[min_max_index[j], :]
                sub_train_y = train_y[min_max_index[j], i]
                centers[i, j] = np.mean(sub_train_y)

                model = KRG(poly='constant', corr='squar_exp', theta0=theta[i, j, :],
                            theta_bounds=[1e-5, 100], print_global=False)
                model.set_training_values(sub_train_x, sub_train_y)
                model.train()

                model_insensitive[i].append(model)
                theta[i, j, :] = model.optimal_theta

    return model_insensitive, centers, theta

