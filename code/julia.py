"""
Show Julia set defined with parameters
"""
import cmath
import argparse
import itertools

import glob
import os
import PIL
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.colors import Colormap
from tqdm import tqdm


def f(x, c):
    return x ** 2 + c

def normalize(img):
    result = ((img / (img.max() - img.min() + 1)) * 255).astype(np.uint8)
    return result

def mandelbrot_binary(min_value=-2, max_value=2, density=100, 
                      c= -0.8 + 0.156j, max_iter=100, limit=2):
    """
    Returns binary map
    min_value, max_value: define a square on a complex plane
    density: number of points between min and max value
    max_iter: how long to iterate in each point
    limit: defines divergense
    """
    xvalues = np.linspace(max_value, min_value, density)
    yvalues = np.linspace(min_value, max_value, density)
    values_map = list(itertools.product(xvalues, yvalues)) # suboptimal solution
    values_map = np.array(values_map).reshape(density, density, 2)
    C = (values_map[...,1] + 1j * values_map[...,0]) # array of complex values
   
    M = np.full((density, density), True, dtype=bool)
    Z = np.zeros((density, density), dtype=complex)
    for i in range(max_iter):
        Z[M] = Z[M] * Z[M] + C[M]
        M[np.abs(Z) > limit] = False
    return ~M

def mandelbrot(min_value=-2, max_value=2, density=100, 
               c= -0.8 + 0.156j, max_iter=100, limit=2):
    """
    Returns binary map
    min_value, max_value: define a square on a complex plane
    density: number of points between min and max value
    max_iter: how long to iterate in each point
    limit: defines divergense
    """
    xvalues = np.linspace(max_value, min_value, density)
    yvalues = np.linspace(min_value, max_value, density)
    values_map = list(itertools.product(xvalues, yvalues)) # suboptimal solution
    values_map = np.array(values_map).reshape(density, density, 2)
    C = (values_map[...,1] + 1j * values_map[...,0]) # array of complex values
   
    M = np.full((density, density), True, dtype=bool)
    Z = np.zeros((density, density), dtype=complex)
    I = np.zeros((density, density), dtype=int)
    for i in range(max_iter):
        Z[M] = Z[M] * Z[M] + C[M]
        I[M] += 1
        M[np.abs(Z) > limit] = False
    return 255 - I

def julia_binary(min_value=-2, max_value=2, density=100, 
                 c= -0.8 + 0.156j, max_iter=100, limit=2):
    """
    Returns binary map
    min_value, max_value: define a square on a complex plane
    density: number of points between min and max value
    max_iter: how long to iterate in each point
    limit: defines divergense
    """
    xvalues = np.linspace(max_value, min_value, density)
    yvalues = np.linspace(min_value, max_value, density)
    values_map = list(itertools.product(xvalues, yvalues)) # suboptimal solution
    values_map = np.array(values_map).reshape(density, density, 2)
    Z = (values_map[...,1] + 1j * values_map[...,0]) # array of complex values
   
    M = np.full((density, density), True, dtype=bool)
    for i in range(max_iter):
        Z[M] = Z[M] * Z[M] + c
        M[np.abs(Z) > limit] = False
        
    return ~M

    
def julia(min_value=-2, max_value=2, density=100, c= -0.8 + 0.156j , max_iter=100, 
          limit=2, norm=False, revert=True):
    """
    Args:
        min_value, max_value: define a square on a complex plane
        density: number of points between min and max value
        max_iter: how long to iterate in each point
        limit: defines divergense
        mono: colour in black/white or use num of iterations
    """
    xvalues = np.linspace(max_value, min_value, density)
    yvalues = np.linspace(min_value, max_value, density)
    
    values_map = list(itertools.product(xvalues, yvalues)) # suboptimal solution
    values_map = np.array(values_map).reshape(density, density, 2)
    Z = (values_map[...,1] + 1j * values_map[...,0]) # array of complex values
    M = np.full((density, density), True, dtype=bool)
    I = np.zeros((density, density), dtype=int)
    for i in range(max_iter):
        Z[M] = Z[M] * Z[M] + c
        I[M] += 1
        M[np.abs(Z) > limit] = False
    return 255 - I

def plot_bifurcation(values_map, c=None, save_image=False, name=None):
    
#     values_map_3d = np.zeros((3, values_map.shape[0], values_map.shape[1]))
#   
#     values_map_3d = np.expand_dims((values_map == 1), 0) * [255, 255, 255]
#     values_map_3d = np.expand_dims((values_map == 2), 0) * [0, 255, 0]
    import matplotlib
    from matplotlib import cm
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys)
    plt.figure(figsize=(14, 11))
    plt.imshow(values_map, cmap=mapper, extent=(-2, 2, -2, 2))
    plt.xlabel('Re(Z)', fontsize=20)
    plt.ylabel('Im(Z)', fontsize=20)
    if c:
        plt.title(f'C = {c}', fontsize=20)
    if save_image:
        plt.imsave(name, ax)
        
def plot_fractal(values_map, c=None, save_image=False, name=None):
    """
    Args:
        save_image: if True, don't show plot to avoid spawning too many windows in Jupyter Notebook
    
    """
    
    plt.figure(figsize=(14, 11))
    plt.imshow(values_map, cmap=plt.cm.gray, extent=(-2, 2, -2, 2))
#     plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.xlabel('Re(Z)', fontsize=20)
    plt.ylabel('Im(Z)', fontsize=20)
    if c:
        plt.title(f'C = {c}', fontsize=20)
    if save_image:
        fname = f'img/task2/{name}.png'
        plt.savefig(fname)
        plt.close()
    
def create_julia_gif(r, c_density, duration=15, name='julia', binary=False, **kwargs):
    alfa_values = np.linspace(0, 2 * np.pi, c_density)
    c_values = [cmath.rect(r, angle) for angle in alfa_values]
    images = []
    for i, c in tqdm(enumerate(c_values), total=c_density):
        j = julia_binary(c=c, **kwargs) if binary else julia(c=c, **kwargs)
            
        fname = f'img/task2/{name}_{i}.png'
        plot_fractal(j, c=alfa_values[i], save_image=True, name=f"{name}_{i}")
        images.append(imageio.imread(fname))
        os.remove(fname)
        
    fname = f'img/task2/{name}.gif'
    imageio.mimsave(fname, images, fps = c_density / duration)

    
def main():
    print(args)
    print("Finished!")

    
if __name__ == "__main__":
    ## Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=complex, default=-0.8 + 0.156j,
                        help="complex number for a plot")
    parser.add_argument("-r", type=float, default=0.7,
                        help="radius for a gif") # 0.7885 # as in Wikipedia article
    parser.add_argument("-limit", type=int, default=2)
    parser.add_argument("-max_iter", type=int, default=100,
                        help="Number of iterations in each point")
    parser.add_argument("-min", type=int, default=-1,
                        help="Lower limit of a square")
    parser.add_argument("-max", type=int, default=1,
                        help="Upper limit of a square")
    parser.add_argument("-density", type=int, default=300,
                        help="how many points to take between min&max values")
    parser.add_argument("-name", type=str, default="julia",
                        help="File name")
    parser.add_argument("-g", "--gif", action="store_true", default=False,
                        help="create a gif file")
    parser.add_argument("--norm", action="store_true", default=False,
                        help="Normalize image histogram")
    parser.add_argument("--revert", action="store_true", default=False,
                        help="Revert colours")
    args = parser.parse_args()
    main()