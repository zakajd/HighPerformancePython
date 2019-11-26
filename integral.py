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


# Function for which we compute integral
def f(x):
    return 1 / x

# Analytical integral
def g(x):
    return np.log(x)

def integrate(f, g=None, limits=(1, 2), nodes=100, rank=None, size=None, comm=None):
    """Compute function integral by using trapezoid rule
    Args:
        f: function to integrate
        g: analytical integral, for self check and error estimation
        limits: (a, b)
        nodes: # of points to split the limit
    """
    all_nodes = np.linspace(limits[0], limits[1], num=nodes)
    left, right = rank * nodes // size, (rank + 1) * nodes // size
    individual_nodes = list(all_nodes[left:right]) # convert to Python object
    if rank == size - 1:
        individual_nodes = all_nodes[left:] # leftmost process take the rest
    # print(f'My rank: {rank}, world size: {size}, {individual_nodes}')
    comm.Barrier()
    partial_result = [(x, f(x)) for x in individual_nodes]
    
    final_result = comm.gather(partial_result, root=0)

    if rank: # rank != 0
        assert final_result is None

    return final_result

def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    final_result = integrate(f, g=None, limits=(1, 10), nodes=10, rank=rank, size=size, comm=comm)
    if rank == 0:
        x = [t[0] for t in final_result]
        y = [t[1] for t in final_result]
        plt.plot(y, x)
        plt.savefig('imgs/integral.png')
        # print(final_result)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Task 7")
    parser.add_argument("-p",  help="# of proccesses")
    args = parser.parse_args()
    main(args)

