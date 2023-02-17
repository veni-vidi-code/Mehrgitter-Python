"""
Dient zum Vergleich der verschiedenen Verfahren

Es kann ggf erforderlich sein einen Ordner "benchmark-results" anzulegen
(Im selben Ordner wie das Skript)
"""

import time

import numpy as np

from implementations.Gitter import standard_schrittweitenfolge
from implementations.dirichlect_ndarrays import dirichlect_randwert_a_l
from implementations.gaussseidel import gauss_seidel_matrices, gauss_seidel_steps
from implementations.helpers import N_l
from implementations.jacobi import jacobi_matrices, jacobi_steps
from implementations.merhgitterverfahren import mehgitterverfaren_steps

import os


def f_x(x):
    return np.pi * np.pi / 8. * (9. * np.sin((3. * np.pi * x) / 2.) + 25 * np.sin(5 * np.pi * x / 2.))


def u_x(x):  # Lösung von f_x
    return np.multiply(np.sin((2. * np.pi) * x), np.cos((np.pi / 2.) * x))


def iter_until(generator, f, limit, a):
    e = next(generator)
    eps = np.linalg.norm((a @ e) - f)
    while eps > limit:
        e = next(generator)
        eps = np.linalg.norm((a @ e) - f)


def mgm(u_star, stufenindex_l: int, w: float, u_0, f, matrices, a, limit: float = 1e-2):
    gen_mgm = mehgitterverfaren_steps(stufenindex_l, 1, 1, u_0, f, matrices, w1=w, w2=w, gamma=1)
    iter_until(gen_mgm, f, limit, a)


def jacobi(u_star, stufenindex_l: int, w: float, u_0, f, a, limit: float = 1e-2):
    a = dirichlect_randwert_a_l(stufenindex_l)
    gen_normal = jacobi_steps(a, u_0, f, w)
    iter_until(gen_normal, f, limit, a)


def gauss_seidel(u_star, stufenindex_l: int, w: float, u_0, f, a, limit: float = 1e-2):
    a = dirichlect_randwert_a_l(stufenindex_l)
    gen_normal = gauss_seidel_steps(a, u_0, f, w)
    iter_until(gen_normal, f, limit, a)


def race(stufenindex_l, mode, w=0.25, runs=10, limit=1e-2):
    startfault = np.random.rand(N_l(stufenindex_l))

    x = standard_schrittweitenfolge(stufenindex_l)
    f = f_x(x)
    u_star = u_x(x)
    # u_star = np.zeros(N_l(stufenindex_l))
    u_0 = u_star + startfault
    # f = np.zeros_like(u_0)
    a = dirichlect_randwert_a_l(stufenindex_l)
    matrices = jacobi_matrices if mode == "jacobi" else gauss_seidel_matrices
    w = 2 * w
    normal_func = jacobi if mode == "jacobi" else gauss_seidel
    time_1 = time.time()  # using time is necessary since most profilers like timeit slow
    # down recursive functions way more than normal functions
    for _ in range(runs):
        mgm(u_star, stufenindex_l, w, u_0.copy(), f, matrices, a, limit)
    time_2 = time.time()
    print("mgm done, switching to jacobi")
    time_3 = time.time()
    for _ in range(runs):
        normal_func(u_star, stufenindex_l, w, u_0.copy(), f, a, limit)
    time_4 = time.time()

    return time_2 - time_1, time_4 - time_3


def benchmark():
    results = []
    l_to_runs = [0, 0, 5000, 1000, 500, 100, 10, 1, 1, 1, 1, 1, 1]
    for mode in ["jacobi", "gauss_seidel"]:
        for stufenindex_l in range(2, 10):
            for w in [0.25]:
                print(f"running {mode} with l={stufenindex_l} and w={w}")
                time_mgm, time_normal = race(stufenindex_l, mode, w, l_to_runs[stufenindex_l], 1e-6)
                results.append({"mode": mode, "l": stufenindex_l, "w": w,
                                "time_mgm": time_mgm, "time_normal": time_normal, "runs": l_to_runs[stufenindex_l],
                                "percentage": time_normal / time_mgm * 100})
                print(time_mgm)
                print(time_normal)
    return results


def combine_results():
    results = {}
    for file in os.listdir("benchmark-results"):
        if file.endswith(".json"):
            with open(os.path.join("benchmark-results", file), "r") as f:
                file_results = json.load(f)
                for result in file_results:
                    key = (result["mode"], result["l"], result["w"])
                    if key not in results:
                        oldresult = (0, 0, 0)
                    else:
                        oldresult = results[key]
                    results[key] = (oldresult[0] + result["time_mgm"], oldresult[1] + result["time_normal"],
                                    oldresult[2] + result["runs"])
    return [{"mode": key[0],
             "l": key[1],
             "w": key[2],
             "time_mgm": value[0],
             "time_normal": value[1],
             "runs": value[2],
             "percentage": value[1] / value[0] * 100} for key, value in results.items()]


if __name__ == "__main__":
    result = benchmark()

    import json
    import datetime

    # save results to file with current timestamp
    with open(f"benchmark-results/results-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json", "w") as f:
        json.dump(result, f, indent=4)

    # combine results from all files in results folder
    combined_results = combine_results()
    with open("benchmark-results.json", "w") as f:
        json.dump(combined_results, f, indent=4)
