import random
from os import getcwd, path, makedirs

import numpy as np
import math

import pandas as pd
from mealpy.swarm_based.GWO import BaseGWO
from mealpy.math_based.HC import BaseHC
from mealpy.physics_based.ASO import BaseASO
from mealpy.swarm_based.PSO import BasePSO
from mealpy.swarm_based.EHO import BaseEHO
from mealpy.swarm_based.WOA import BaseWOA
from mealpy.swarm_based.DO import BaseDO
from IPython.display import display
import concurrent.futures as parallel

d = 40


def F27(x):
    res = 0
    res1 = 0
    for i in range(len(x)):
        res += x[i]
        res1 += pow(x[i], 2)
    return 1 - math.cos(2 * math.pi * res) + 0.1 * res1


def F24(x):
    res = 0
    for i in range(len(x)):
        res += abs(x[i] * math.sin(x[i]) + 0.1 * x[i])
    return res


def F15(x):
    res = 0
    for i in range(len(x)):
        res += (pow(x[i], 2) - 10 * math.cos(2 * math.pi * x[i]))
    return 10 * len(x) + res


def F14(x):
    res = 0
    res1 = 0
    for i in range(len(x)):
        res += pow(x[i] - 1, 2)
    for i in range(1, len(x)):
        res1 += x[i] * x[i - 1]
    return res - res1


def F13(x):
    res = 0
    res1 = 1
    for i in range(len(x)):
        res += (pow(x[i], 2) / 4000)
        res1 *= (math.cos(x[i] / math.sqrt(i + 1))) + 1
    return res - res1


def F12(x):
    res = 0
    for i in range(len(x)):
        res += pow(abs(x[i] + 0.5), 2)
    return res


def F11(x):
    res = 0
    res1 = 1
    for i in range(len(x)):
        res += abs(x[i])
        res1 *= abs(x[i])
    return res + res1


def F6(x):
    res = 0
    res1 = 0
    for i in range(len(x)):
        for j in range(i):
            res1 += x[i]
        res += pow(res1, 2)
        res1 = 0
    return res


def F3(x):
    res = 0
    for i in range(len(x)):
        for j in range(i):
            res += pow(x[i], 2)
    return res


def F2(x):
    res = 0
    for i in range(len(x)):
        res += pow(abs(x[i]), i + 2)
    return res


def F1(x):
    return sum(100.0 * (x[:-1] ** 2.0 - x[1:]) ** 2.0 + (1 - x[:-1]) ** 2.0)


def generate_position(lb=None, ub=None):
    return np.random.uniform(lb, ub, d)


def ranfRange(lb, ub):
    return random.randint(lb, ub)


p1 = {
    "name": "F1",
    "fit_func": F1,
    "lb": -100,
    "ub": 100,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p2 = {
    "name": "F2",
    "fit_func": F2,
    "lb": -100,
    "ub": 100,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p3 = {
    "name": "F3",
    "fit_func": F3,
    "lb": -65,
    "ub": 65,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p6 = {
    "name": "F6",
    "fit_func": F6,
    "lb": -100,
    "ub": 100,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p11 = {
    "name": "F11",
    "fit_func": F11,
    "lb": -10,
    "ub": 10,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p12 = {
    "name": "F12",
    "fit_func": F12,
    "lb": -100,
    "ub": 100,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p13 = {
    "name": "F13",
    "fit_func": F13,
    "lb": -600,
    "ub": 600,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p14 = {
    "name": "F14",
    "fit_func": F14,
    "lb": -1 * pow(d, 2),
    "ub": pow(d, 2),
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p15 = {
    "name": "F15",
    "fit_func": F15,
    "lb": -5.12,
    "ub": 5.12,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p24 = {
    "name": "F24",
    "fit_func": F24,
    "lb": -10,
    "ub": 10,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

p27 = {
    "name": "F27",
    "fit_func": F27,
    "lb": -100,
    "ub": 100,
    "minmax": "min",
    "verbose": False,
    "n_dims": d,
    "generating_positions": generate_position,
}

problems = [p1, p2, p3, p6, p11, p12, p13, p14, p15, p24, p27]

model_name = ["GWO", "ASO", "HC", "PSO", "EHO", "WOA", "DO"]
N_TRIALS = 30
func_names = ["F1", "F2", "F3", "F6", "F11", "F12", "F13", "F14", "F15", "F24", "F27"]
SOL_PATH = "history/functions/"

check_dir3 = f"{getcwd()}/{SOL_PATH}"

if not path.exists(check_dir3): makedirs(check_dir3)


def find_minimum(p):
    """
    We can run multiple functions at the same time.
    Each core (CPU) will handle a prolblem, each problem will run N_TRIALS times
    """

    GWO_lst = []
    HC_lst = []
    ASO_lst = []
    PSO_lst = []
    EHO_lst = []
    WOA_lst = []
    DO_lst = []

    func_DF = []
    a = [[0] * len(model_name)] * 2
    func_DF = pd.DataFrame(np.array(a), columns=model_name)
    func_DF.index = ["AVG", "SD"]

    display(func_DF)
    for id_trial in range(1, N_TRIALS + 1):
        GWOmodel = BaseGWO(p, epoch=200, name=model_name[0], neighbour_size=50)
        ASOmodel = BaseASO(p, epoch=200, name=model_name[1], neighbour_size=50)
        HCmodel = BaseHC(p, epoch=200, name=model_name[2], neighbour_size=25)
        PSOmodel = BasePSO(p, epoch=200, name=model_name[3], neighbour_size=50)
        EHOmodel = BaseEHO(p, epoch=200, name=model_name[4], neighbour_size=50)
        WOAmodel = BaseWOA(p, epoch=200, name=model_name[5], neighbour_size=50)
        DOmodel = BaseDO(p, epoch=200, name=model_name[6], neighbour_size=50)

        GWObest_solution, GWObest_fitness = GWOmodel.solve()
        GWO_lst.append(GWObest_fitness)
        ASObest_solution, ASObest_fitness = ASOmodel.solve()
        ASO_lst.append(ASObest_fitness)
        HCbest_solution, HCbest_fitness = HCmodel.solve()
        HC_lst.append(HCbest_fitness)
        PSObest_solution, PSObest_fitness = PSOmodel.solve()
        PSO_lst.append(PSObest_fitness)
        EHObest_solution, EHObest_fitness = EHOmodel.solve()
        EHO_lst.append(EHObest_fitness)
        WOAbest_solution, WOAbest_fitness = WOAmodel.solve()
        WOA_lst.append(WOAbest_fitness)
        DObest_solution, DObest_fitness = DOmodel.solve()
        DO_lst.append(DObest_fitness)

    print("GWO")
    print(np.average((np.array(GWO_lst))))
    print(np.std((np.array(GWO_lst))))
    func_DF.at["AVG", "GWO"] = np.average((np.array(GWO_lst)))
    func_DF.at["SD", "GWO"] = np.std((np.array(GWO_lst)))
    print("ASO:")
    print(np.average((np.array(ASO_lst))))
    print(np.std((np.array(ASO_lst))))
    func_DF.at["AVG", "ASO"] = np.average((np.array(ASO_lst)))
    func_DF.at["SD", "ASO"] = np.std((np.array(ASO_lst)))
    print("HC:")
    print(np.average((np.array(HC_lst))))
    print(np.std((np.array(HC_lst))))
    func_DF.at["AVG", "HC"] = np.average((np.array(HC_lst)))
    func_DF.at["SD", "HC"] = np.std((np.array(HC_lst)))
    print("PSO:")
    print(np.average((np.array(PSO_lst))))
    print(np.std((np.array(PSO_lst))))
    func_DF.at["AVG", "PSO"] = np.average((np.array(PSO_lst)))
    func_DF.at["SD", "PSO"] = np.std((np.array(PSO_lst)))
    print("EHO:")
    print(np.average((np.array(EHO_lst))))
    print(np.std((np.array(EHO_lst))))
    func_DF.at["AVG", "EHO"] = np.average((np.array(EHO_lst)))
    func_DF.at["SD", "EHO"] = np.std((np.array(EHO_lst)))
    print("WOA:")
    print(np.average((np.array(WOA_lst))))
    print(np.std((np.array(WOA_lst))))
    func_DF.at["AVG", "WOA"] = np.average((np.array(WOA_lst)))
    func_DF.at["SD", "WOA"] = np.std((np.array(WOA_lst)))
    print("DO:")
    print(np.average((np.array(DO_lst))))
    print(np.std((np.array(DO_lst))))
    func_DF.at["AVG", "DO"] = np.average((np.array(DO_lst)))
    func_DF.at["SD", "DO"] = np.std((np.array(DO_lst)))

    display(func_DF)
    func_DF.to_csv(f"{SOL_PATH}/{p.get('name')}_40D.csv", header=True, index=False)

    print(f"Finish function: {str(p.get('name'))}")


if __name__ == '__main__':
    with parallel.ProcessPoolExecutor() as executor:
        results = executor.map(find_minimum, problems)

