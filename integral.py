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

def integrate(f, limits=(5, 7), num_steps=100):
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

def integrate_parallel(f, g=None, limits=(5, 7), nodes=100):
    """Compute function integral by using trapezoid rule
    Args:
        f: function to integrate
        g: analytical integral, for self check and error estimation
        limits: (a, b)
        nodes: # of points to split the limit
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    all_nodes = np.linspace(limits[0], limits[1], num=nodes)
    left, right = rank * nodes // size, (rank + 1) * nodes // size
    individual_nodes = list(all_nodes[left:right]) # convert to Python object
    if rank == size - 1:
        individual_nodes = all_nodes[left:] # leftmost process take the rest
    print(f'My rank: {rank}, world size: {size}, {individual_nodes}')
    comm.Barrier()
    partial_result = [(x, f(x)) for x in individual_nodes]
    
    data = comm.gather(data, root=0)
    
    sendbuf = np.empty(2, dtype = 'i')
    sendbuf[0] = 0
    sendbuf[1] = rank
    
    # Recieve data
    if rank == 0:
        recvbuf = np.empty(2, dtype = 'i')
    else:    
        recvbuf = None
        
    if rank == 0:
        printf(f"{nodes} trapezoids, [a, b] = [{limits}], result = {total_integral}")


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

