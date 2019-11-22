from scipy.signal import convolve, convolve2d
from scipy.ndimage import filters
import itertools
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import imageio
import os
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D # 3D plotting
from functools import wraps


def init_map_2d(N, C=2):
    """
    N: map size NxN
    C: number of different players (default=2)
    """
    board = np.random.random_sample((N, N))
    result = np.zeros((C, N, N), dtype=np.uint8)
    for c in range(C):
        result[c] = ((1 / C) * c < board) * (board <= (1 / C) * (c + 1))
    return result

def compress_2d(game_map, N, C):
    """
    Expects input to be C x N x N array
    returns N x N array
    """
    result = np.zeros((N, N))
    for c in range(C):
        result += game_map[c] * c
    return result

def decompress_2d(game_map_2d, N, C):
    """
    Expects input to be N x N array
    returns C x N x N array
    """
    result = np.zeros((C, N, N))
    for c in range(C):
        result[c] = (game_map_2d == c)
    return result#.astype(np.uint8)

def game_step_2d(game_map, r):
    """
    SLOW!
    Works for arbitrary number of colors
    params r, C
    """
    C, N, N = game_map.shape
    
    kernel_2d = np.ones((3, 3)) 
    kernel_2d[1, 1] = 0 # 3D cube with 0 in center
    
    compressed_map = compress_2d(game_map, N, C)
    neighbours_3d = np.zeros((C, N, N))
    for c in range(C):
        neighbours_3d[c] = convolve2d(game_map[c], kernel_2d, mode='same', boundary='wrap')
        
    neighbours_3d *= game_map
    neighbours = neighbours_3d.max(axis=0)
    moving = neighbours < int(kernel_2d.sum() * r)
    num_moving = moving.sum()
    
    moving_colors = compressed_map[moving]
    np.random.shuffle(moving_colors)
    compressed_map[moving] = moving_colors
    
    updated_map = decompress_2d(compressed_map, N, C)
    return updated_map, num_moving

def game_2d(N, C, r, game_length, name=None, figsize=(14,11)):
    """
    Launch a game with board of size N x N, C colours and `num_iterations`
    r: % of neighbours of the same colour
    """
    assert r<= 1, "Wrong r value! 0 <= r <= 1"
    game_map = init_map_2d(N,C)
    move_hist = []
    images = []
    for i in tqdm(range(game_length),desc=f'Number of neighbours={int(r*8)}', leave=False):
        game_map, moved = game_step_2d(game_map, r)
        game_map_2d = prepare_2d_plot(game_map)
        fname = f'imgs/{name}.png'
        plot_2d(game_map_2d, C=C, r=r, i=i, save_image=True, name=fname, figsize=figsize)
        images.append(imageio.imread(fname))
        os.remove(fname)
        move_hist.append(moved)
        
    fname = f'imgs/{name}.gif'
    imageio.mimsave(fname, images, fps = 10)
    return move_hist
        
def prepare_2d_plot(game_map):
    """
    Takes CxNxN matrix and return 
    2D array for plotting
    """
    C, N, N = game_map.shape
    game_map_2d = compress_2d(game_map, N, C)
    return game_map_2d

def plot_2d(game_map_2d, C=2, r=None, i=None, save_image=False, name=None, figsize=(14,11)):
    plt.figure(figsize=figsize)
    
    if C > 2:
        plt.imshow(game_map_2d)
    else:
        plt.imshow(game_map_2d, cmap=plt.cm.gray)
    
    if r:
        plt.title(f'Schelling’s: r = {r}, iteration = {i}', fontsize=20)
    plt.axis('off')
    if save_image:
        plt.savefig(name)
        plt.close()

                     