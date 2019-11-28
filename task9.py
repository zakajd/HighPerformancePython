from scipy.signal import convolve2d
import numpy as np
K = np.ones((3, 3), dtype=int)
K[1, 1] = 0
def step(M):
    num_alive = convolve2d(M, K, mode='same')
    dead2alive = (M == 0) & (num_alive == 3)
    alive2alive = (M == 1) & (1 < num_alive) & (num_alive < 4)
    M_res = np.zeros_like(M)
    M_res[dead2alive] = 1
    M_res[alive2alive] = 1
    return M_res, np.sum(M_res)
