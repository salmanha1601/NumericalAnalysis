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
from mealpy.utils.visualize import export_convergence_chart
from pandas import DataFrame
from IPython.display import display
import concurrent.futures as parallel
import time


def obj_function(x):
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

    return [F1(x), F2(x), F3(x), F6(x), F11(x), F12(x), F13(x), F14(x), F15(x), F24(x), F27(x)]


model_name = ["GWO", "ASO", "HC", "PSO", "EHO", "WOA", "DO"]
N_TRIALS = 30
func_names = ["F1", "F2", "F3", "F6", "F11", "F12", "F13", "F14", "F15", "F24", "F27"]
d = 20
PATH_ERROR = "history/error/" + model_name[0] + "/"
PATH_BEST_FIT = "history/best_fit/"
SOL_PATH = "history/functions/"

check_dir1 = f"{getcwd()}/{PATH_ERROR}"
check_dir2 = f"{getcwd()}/{PATH_BEST_FIT}"
check_dir3 = f"{getcwd()}/{SOL_PATH}"

if not path.exists(check_dir1): makedirs(check_dir1)
if not path.exists(check_dir2): makedirs(check_dir2)
if not path.exists(check_dir3): makedirs(check_dir3)


def find_minimum(function_name):
    """
    We can run multiple functions at the same time.
    Each core (CPU) will handle a function, each function will run N_TRIALS times
    """
    print(f"Start running: {function_name}")
    error_full = {}
    error_columns = []
    best_fit_list = []
    best_solution_list = []

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
        problem = {
            "fit_func": obj_function,
            "lb": [-100]*d,
            "ub": [100]*d,
            "minmax": "min",
            "verbose": False,
        }

        GWOmodel = BaseGWO(problem, epoch=200, name=model_name[0], fit_name=function_name)
        ASOmodel = BaseASO(problem, epoch=200, name=model_name[1], fit_name=function_name)
        HCmodel = BaseHC(problem, epoch=200, name=model_name[2], fit_name=function_name)
        PSOmodel = BasePSO(problem, epoch=200, name=model_name[3], fit_name=function_name)
        EHOmodel = BaseEHO(problem, epoch=200, name=model_name[4], fit_name=function_name)
        WOAmodel = BaseWOA(problem, epoch=200,  name=model_name[5], fit_name=function_name)
        DOmodel = BaseDO(problem, epoch=200,  name=model_name[6], fit_name=function_name)

        GWObest_solution, GWObest_fitness = GWOmodel.solve()
        print(GWObest_fitness)
        GWO_lst.append(GWObest_fitness)
        ASObest_solution, ASObest_fitness = ASOmodel.solve()
        print(ASObest_fitness)
        ASO_lst.append(ASObest_fitness)
        HCbest_solution, HCbest_fitness = HCmodel.solve()
        print(HCbest_fitness)
        HC_lst.append(HCbest_fitness)
        PSObest_solution, PSObest_fitness = PSOmodel.solve()
        print(PSObest_fitness)
        PSO_lst.append(PSObest_fitness)
        EHObest_solution, EHObest_fitness = EHOmodel.solve()
        print(EHObest_fitness)
        EHO_lst.append(EHObest_fitness)
        WOAbest_solution, WOAbest_fitness = WOAmodel.solve()
        print(WOAbest_fitness)
        WOA_lst.append(WOAbest_fitness)
        DObest_solution, DObest_fitness = DOmodel.solve()
        print(DObest_fitness)
        DO_lst.append(DObest_fitness)

    print(f"{function_name}:")
    func_DF.at["AVG", "GWO"] = np.average((np.array(GWO_lst)))
    func_DF.at["SD", "GWO"] = np.std((np.array(GWO_lst)))
    # print("ASO:")
    func_DF.at["AVG", "ASO"] = np.average((np.array(ASO_lst)))
    func_DF.at["SD", "ASO"] = np.std((np.array(ASO_lst)))
    # print("HC:")
    func_DF.at["AVG", "HC"] = np.average((np.array(HC_lst)))
    func_DF.at["SD", "HC"] = np.std((np.array(HC_lst)))
    # print("PSO:")
    func_DF.at["AVG", "PSO"] = np.average((np.array(PSO_lst)))
    func_DF.at["SD", "PSO"] = np.std((np.array(PSO_lst)))
    # print("EHO:")
    func_DF.at["AVG", "EHO"] = np.average((np.array(EHO_lst)))
    func_DF.at["SD", "EHO"] = np.std((np.array(EHO_lst)))
    # print("WOA:")
    func_DF.at["AVG", "WOA"] = np.average((np.array(WOA_lst)))
    func_DF.at["SD", "WOA"] = np.std((np.array(WOA_lst)))
    # print("DO:")
    func_DF.at["AVG", "DO"] = np.average((np.array(DO_lst)))
    func_DF.at["SD", "DO"] = np.std((np.array(DO_lst)))

    display(func_DF)
    func_DF.to_csv(f"{SOL_PATH}/{function_name}_20D.csv", header=True, index=False)

    print(f"Finish function: {function_name}")

    """"
    return {
        "func_name": function_name,
        "best_fit_list": best_fit_list,
        "best_solution_list": best_solution_list,
        "model_name": model_name[0]
    }
    """


if __name__ == '__main__':
    best_fit_full = {}
    # best_sol_full = {}
    best_fit_columns = []
    with parallel.ProcessPoolExecutor() as executor:
        results = executor.map(find_minimum, func_names)
    """
    for result in results:
        best_fit_full[result["func_name"]] = result["best_fit_list"]
        # best_sol_full[result["func_name"]] = result["best_solution_list"]
        best_fit_columns.append(result["func_name"])

    df = DataFrame(best_fit_full, columns=best_fit_columns)
    df2 = DataFrame(best_fit_full)
    df.to_csv(f"{PATH_BEST_FIT}/{len([-100] * d)}D_{model_name[0]}_best_fit.csv", header=True, index=False)
    # df2.to_csv(f"{PATH_BEST_SOL}/{len([-100] * 5)}D_{model_name[0]}_best_sol.csv", header=True, index=False)
    """
