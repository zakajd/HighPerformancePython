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

def plot_bifurcation(r_values, result, name=None, params=None, save_fig=False, ):
    """
    Plots given 2D array and optionally saves figure to folder
    """
    fig, ax = plt.subplots(figsize = (14, 11))  # create figure & 1 axis
    ax.plot(r_values, result, '.')
    if save_fig:
        ax.text(0.5, 1., f"n={params[0]}, m={params[1]}, max_R={params[2]}")
        name = f"imgs/{name}.png"
        fig.savefig(name)   # save the figure to file
#     plt.close(fig)    # close the figure window
    
def main(R, n, steps, m, name):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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


### Code from Karim
# def compute_r(u, number_of_iterations, last_m_digits):
#     x = np.random.random()*2 - 1
#     X = np.zeros(last_m_digits)
#     Y = np.zeros(last_m_digits)
#     for n in range(number_of_iterations):
#         x=(u*x)*(1-abs(x))
#         # for every r save last “m” values of x after first “n” values
#         if number_of_iterations - n -1 < Y.shape[0]:
#             Y[number_of_iterations - n -1] = x
#             X[number_of_iterations - n -1] = u
#     return X, Y

# num_of_points = 100000
# last_m_digits = 100
# R=np.linspace(0.7,4,num_of_points)

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# nprocs = comm.Get_size()
# # slice_index = np.arange(0, t_window_positions.shape[0], 
# #                         t_window_positions.shape[0]//(nprocs)).astype(int)
# slice_index = np.linspace(0, R.shape[0], nprocs+1, endpoint=False, dtype=int)
# slice_index[-1] = R.shape[0]
# # print(rank, nprocs)
# # print(slice_index)
# X = np.zeros(last_m_digits*R[slice_index[rank]:slice_index[rank+1]].shape[0])
# Y = np.zeros(last_m_digits*R[slice_index[rank]:slice_index[rank+1]].shape[0])

# for i, u in enumerate(R[slice_index[rank]:slice_index[rank+1]]):
#     X[i*last_m_digits:(i+1)*last_m_digits], Y[i*last_m_digits:(i+1)*last_m_digits] = compute_r(u, int(num_of_points*1.5), last_m_digits)

#     # spectrogram = compute_r(t, y, 
# #                                 t_window_positions[slice_index[rank]:slice_index[rank+1]],
# #                                 window_width=20, nwindowsteps = 1000)

# X_res = comm.gather(X, root = 0)
# Y_res = comm.gather(Y, root = 0)

# if rank==0:
#     X_data = np.hstack(X_res)
#     Y_data = np.hstack(Y_res)
#     plt.figure(figsize=(15,9))
#     plt.plot(X_data, Y_data, ls='', marker=',')
#     plt.xlabel('r')
#     plt.ylabel('x')
#     plt.savefig('bifurcation_parallel.png', bbox_inches='tight', dpi=300)
#     print(X_data.shape, Y_data.shape)
    