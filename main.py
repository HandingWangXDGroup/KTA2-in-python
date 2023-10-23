import os
import numpy as np
from util.util import mk_dir
from algorithm.algorithm import kta2
from pymoo.problems.many import dtlz


def main(problems, independent_run_num):
    for problem in problems:
        print("{} test problem".format(problem.name()))
        res_path = "./res/{}/".format(problem.name())
        mk_dir(res_path)

        for i in range(independent_run_num):
            res = kta2(i+1, problem, 300, 100, 0.75, 0.1, 10, 5)
            res_obj = np.array([s.obj for s in res])
            np.save(os.path.join(res_path, "{}.npy".format(i+1)), res_obj)


if __name__ == "__main__":
    test_problems = [dtlz.DTLZ1(10, 3), dtlz.DTLZ2(10, 3), dtlz.DTLZ4(10, 3)]
    main(test_problems, 30)
