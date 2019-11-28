import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
from mpi4py import MPI

# Make sure folder for images exist
import os
try:
    os.mkdir('imgs/')
except FileExistsError as err:
    pass


def f(x):
    """ Function to integrate
    Args:
        x: value to count
    """
    return 1 / np.sqrt(1 + x ** 2)

def g(x):
    """ Analytical expression of f(x)
    Args:
        x: func value
    """
    return np.log(x + np.sqrt(1 + x ** 2))

# from functools import reduce

def integrate(f, limits=(5, 7), num_steps=100000):
    """Compute function integral by using trapezoid rule
    Args:
        f: function to integrate
        g: analytical integral, for self check and error estimation
        limits: (a, b)
        steps: # of points to split the limit
    """
    steps = np.linspace(limits[0], limits[1], num=num_steps, endpoint=True)
    step_size = (limits[1] - limits[0]) / num_steps
    func_values = [f(x) for x in steps]
    result = (sum(func_values) * 2 - func_values[0] - func_values[1]) * 0.5 * step_size
    return result


def main(parallel=False):
    if parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        all_nodes = np.linspace(5, 7, num=size + 1)
        S_d = integrate(f, limits=all_nodes[rank:rank+2], num_steps=100000 // size)
        comm.Barrier()

        S_d = comm.gather(S_d, root=0)

        if rank == 0:
#             print(sum(S_d))
            return sum(S_d)
    else:
        S_d = integrate(f, limits=(5, 7), num_steps=100000)
#         print(S_d)
        return S_d
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Task 7")
    parser.add_argument("-parallel",  help="0 - not parallel, 1 - parallel", type=bool)
    args = parser.parse_args()
    main(args.parallel)

