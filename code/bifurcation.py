"""
Show bifurcation maps defined with parameters
Usage:
mpirun -n 3 python bifurcation.py
"""

import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

def bifurcation(x, r):
    return r * x * (1 - x)

def bifurcation_map(R, steps, n, m):
    r_values = np.linspace(0, R, steps)
    res = []
    for r in r_values:
        values = []
        x = random.random()
        for _ in range(n):
            x = bifurcation(x, r)
        for _ in range(m):
            x = bifurcation(x, r)
            values.append(x)
        res.append(values)
    return r_values, res

def plot_bifurcation(r_values, result, name=None, params=None, save_fig=False):
    """
    Plots given 2D array and optionally saves figure to folder
    """
    fig, ax = plt.subplots(figsize = (14, 11))  # create figure & 1 axis
    ax.plot(r_values, result, '.')
    if save_fig:
        ax.text(0.5, 1., f"n={params[0]}, m={params[1]}, max_R={params[2]}")
        name = f"img/task1/{name}.png"
        fig.savefig(name)   # save the figure to file
#     plt.close(fig)    # close the figure window
    
def main(R, n, steps, m, name):
    r_val, bf = bifurcation_map(R, steps, n, m)
    plot_bifurcation(r_val, bf, name, (n, m, R), save_fig=True)
    
    
if __name__ == "__main__":
    ## Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", type=int, default=4,
                        help="max number of R")
    parser.add_argument("-n", type=int, default=1000,
                        help="iterate for n values")
    parser.add_argument("-steps", type=int, default=100,
                        help="divide interval from 0 to R into steps")
    parser.add_argument("-m", type=int, default=25,
                        help="save last m values")
    parser.add_argument("-name", type=str, default="bifurcation",
                        help="File name")
    # parser.add_argument("-proc", type=int, default=2,
    #                     help="Number of processes to use")
    args = parser.parse_args()
    main(**args)