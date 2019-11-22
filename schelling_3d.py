import os
from scipy.signal import convolve
import itertools
import matplotlib.pyplot as plt
import numpy as np
import imageio
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D # 3D plotting


# Same functions for 3D case
def init_map_3d(N, C=2):
    """
    N: map size NxNxN
    C: number of different players (default=2)
    """
    board = np.random.random_sample((N, N, N))
    result = np.zeros((C, N, N, N), dtype=np.uint8)
    for c in range(C):
        result[c] = ((1 / C) * c < board) * (board <= (1 / C) * (c + 1))
    return result

def compress_3d(game_map):
    """
    Convert CxNxNxN matrix into NxNxN
    """
    C, N, N, N = game_map.shape
    result = np.zeros((N, N, N))
    for c in range(C):
        result += game_map[c] * c
    return result.astype(game_map.dtype)

def decompress_3d(game_map_3d, N, C):
    """
    Decompresses 3D array into 4D array
    Expects input to be N x N x N array
    returns C x N x N x N array
    """
    
    result = np.zeros((C, N, N, N))
    for c in range(C):
        result[c] = (game_map_3d == c)
    return result.astype(game_map_3d.dtype)

def game_step_3d(game_map, r):
    """
    Takes 4D matrix as input, 
    Returns changed 4D matrix as output
    """
    
    C, N, N, N = game_map.shape
    
    kernel_3d = np.ones((3, 3, 3)) 
    kernel_3d[1, 1, 1] = 0 # 4D cube with 0 in center 
    
    compressed_map = compress_3d(game_map)
    # Convolve with 3x3x3 kernel of ones having hole in the centre.
    # max is just fancy way to compress 4D array into 3D
    neighbours_4d = np.zeros((C, N, N, N))
    for c in range(C):
        neighbours_4d[c] = convolve(game_map[c], kernel_3d, mode='same', method='direct')
    neighbours_4d *= game_map
    neighbours = neighbours_4d.max(axis=0)

    moving = neighbours < int(kernel_3d.sum() * r)
    num_moving = moving.sum()
    
    compressed_map = compress_3d(game_map)
    moving_colors = compressed_map[moving]
    np.random.shuffle(moving_colors)
    compressed_map[moving] = moving_colors
    
    updated_map = decompress_3d(compressed_map, N, C)
    return updated_map, num_moving

def game_3d(N, C, r, game_length, name=None, proj=False, figsize=(14,11), fps=3, verbouse=False):
    """
    Launch a game with board of size N x N x N, C colours and `game_length`
    r: % of neighbours of the same colour
    proj: % save projections instead of 3D plane
    """
    assert r<= 1, "Wrong r value! 0 <= r <= 1"
    game_map = init_map_3d(N,C)
    if verbouse:
        # Plot inital conditions
        x, y, z, c, proj_x, proj_y, proj_z = prepare_3d_plot(game_map, projections=True)
        if N < 20: # With big N makes no sence
            plot_3d(x, y, z, c, r=r, save_image=False, figsize=figsize)
        plot_projections(proj_x, proj_y, proj_z)
    
    move_hist = []
    images = []
    fname = f'imgs/{name}.png'
    for i in tqdm(range(game_length), desc=f'Number of neighbours={int(r*27)}', leave=False):
        game_map, moved = game_step_3d(game_map, r)
        x, y, z, c, proj_x, proj_y, proj_z = prepare_3d_plot(game_map, projections=True)
        if proj:
            plot_projections(proj_x, proj_y, proj_z, save_image=True, name=fname)
        else:
            plot_3d(x, y, z, c, r=r, save_image=True, name=fname, figsize=figsize)
        images.append(imageio.imread(fname))
        os.remove(fname)
        move_hist.append(moved)

    fname = f'imgs/{name}.gif'
    imageio.mimsave(fname, images, fps = fps)
    
    if verbouse:
        # Plot final conditions
        x, y, z, c, proj_x, proj_y, proj_z = prepare_3d_plot(game_map, projections=True)
        if N < 20: # With big N makes no sence
            plot_3d(x, y, z, c, r=r, save_image=False, figsize=figsize)
        plot_projections(proj_x, proj_y, proj_z)
    
    return game_map, move_hist

def prepare_3d_plot(game_map, projections=False):
    """
    Takes CxNxNxN matrix and return 
    N^3 x 4 matrix (x, y, z and colour channels)
    """
    C, N, N, N = game_map.shape
    game_map_3d = compress_3d(game_map)
    xyz = np.array(list(itertools.product(range(N), range(N), range(N))))
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    c = game_map_3d.reshape(-1)
    if C == 2:
        # delete entries for one of the colours
        mask = (c ==1)
        c = c[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if projections:
            ## Return also 3 projections on X, Y and Z planes
            proj_x, proj_y, proj_z = game_map_3d.sum(axis=0), game_map_3d.sum(axis=1), game_map_3d.sum(axis=2)
            return x, y, z, c, proj_x, proj_y, proj_z
            
    return x, y, z, c

def plot_projections(proj_x, proj_y, proj_z, save_image=False, name=None):
    fig, ax = plt.subplots(ncols=3, figsize=(15,5))
#     fig.suptitle('test title', fontsize=10)
    ax[0].imshow(proj_x)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[1].imshow(proj_y)
    ax[2].imshow(proj_z)
    ax[0].set_title('Projection on X')
    ax[1].set_title('Projection on Y')
    ax[2].set_title('Projection on Z')
    plt.axis('off')
    if save_image:
        plt.savefig(name)
        plt.close()
    
def plot_3d(x, y, z, c, r=None, save_image=False, name=None, figsize=(14, 11)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
#     ax = plt.axes()
    if c.max() <= 1:
#         ax.scatter(x, y, z, c=c, linewidth=0.5, marker='.', cmap=plt.cm.gray)
#         ax.scatter(x, y, z, linewidth=0.5, marker='o', cmap=plt.cm.gray)
        ax.scatter(x, y, z, c='red', linewidth=0.5, marker='.', s=5, depthshade=False)
    else:
        ax.scatter(x, y, z, c=c, linewidth=0.5, marker='.')
    ax.set_axis_off()
    if r:
        ax.set_title(f'3D model with r={r}', fontsize=20)
        
    if save_image:
        plt.savefig(name)
        plt.close()
                     