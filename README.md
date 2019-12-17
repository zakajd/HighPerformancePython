# HighPerformancePython
Solutions of tasks in High Performance Python Lab course at Skoltech

**Task 1**: Bifurcation map

Plot classical bifurcation map. $x_{n+1} = r * x_n * (1 - x_n)$ where $x_0 = rand(0, 1)$, $r$ - const.

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T1.1| Implement the map, plot the evolution of x | 1 |
|T1.2| Create a linspace of r’s, for every r save the last “m” values of x after the first “n” values, play around with values | 1 |
|T1.3| Get the bifurcation map | 1 |
|T1.4| Parallel computations | 2 |
|    | Plot speedup versus number of processors graph | 2 |


**Task 2**: Julia set


| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T2.1| Generate black and white image  | 1 |
|T2.2| Use different color for bifurcation points | 1 |
|T2.3| Mandelbrot set  | 2 |
|T2.4| Plot some figures for c = exp(i * alpha), alpha in range(0, 2pi) | 2 |
|T2.5| Plot gif of figures for c = exp(i * alpha), alpha = range(0, 2pi) | 2 |


**Task 3** Schelling model


Ganeral: figsize = (14, 11), fontsize = 20.
| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T3.1| 9 gifs for 9 values of R  | 1 |
|T3.2| Plot # of moving cells vs times for 9 values of R | 1 |

**Task 4**:  Spectrogram

**Task 5**: Matmul

Measure time of different matrix multiplication functions

**Task 6**: Advanced Spectrogram

Use Jupyter notebook templates to make something mre interesting

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T6.1| Add fourth wave packets of harmonic signal (frequency=4 and time_shift=7 cycles). See how to adjust frequency and time shift of signal by using, e.g., figures about AFP and STFT. These figures give you information about the amplitudes, frequencies, signal longitudes and signal time. You may see it like "cause & effect", where "cause" is to change some parameters of forming signal and "effect" are consequent changes in figures | 1 |
|T6.2| Explain, why the "hats" of signal and AFP may be sharp | 1 |
|T6.3| Vary kappa on [0.001, 10], kappa=exp(theta), theta=linspace(ln(0.001), ln(10), 100 steps), create GIFs| 1 |
|T6.3| Explain, why specgram is different| 1 |
|T6.4| PVary n_timestamps_given=4090, 4091, ...,5000 and plot results of computational time on the graph. | 2 |
| | Explain why cProfiler gives different results. What is the bottleneck in this program? How can you improve the program? | 2 |
|T6.5| Parallel STFT | 2 |	
| | Plot speedup versus number of processors graph |	2 |

**Task 7**: Study an integral

- Compute an integral using the trapezoid rule
- Split the job between processes and use Reduce collective communication
- Check the accuracy when taking more nodes (i.e. smaller step)
- Check the speedup when taking more processes
![integral](https://github.com/zakajd/HighPerformancePython/blob/master/imgs/integral.png)

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T7.1| Analytical computation of the integral is given | 1 |
|T7.2| Trapezoidal approximation is used | 1 |
|T7.3| We can arbitrarily choose the number of MPI processes that we want to launch and it does not depend on the number of discretization points | 2 |
|T7.4| Speedup versus number of processors graph is plotted |1|
|T7.5| Error value versus discretization step graph is plotted|	1 |

**Task 8**: Columnwise shifted pictures

- Take a picture
- Split the picture columns between processes
- Shift the columns cyclically

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T8.1| We can arbitrarily choose the number of MPI processes that we want to launch and it does not depend on the width of the picture | 2 |
|T8.2|	Speedup versus number of processors graph is plotted. | 2 |
|T8.3| Total memory consumption versus number of processors is plotted. | 2 |


**Task 9**: Conway's Game of Life

Parallelize using “ghost” cells (red)
Init condition -- Gosper’s glider gun:https://tinyurl.com/yx5hy26m

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T9.1| Sequential implementation is provided | 1 |
|T9.2| Parallel implementation is provided | 2 |
|T9.3| Add parameter to choose # of MPI processes | 2 |
|T9.4| "Gosper's gun" initial condition is given | 1 |
|T9.5| At least two more initial conditions are modelled | 2 |
|T9.6| Plot graph living cells / time for 3 different initial conditions | 2 |	

**Task 10**: Saxpy

Use Saxpy

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T10.1| Explain difference between cupy and numpy. CPU -> GPU data transfer using CuPy | 1 |
|T10.2| Function saxpy that runs on GPU using cupy is provided| 1 |
|T10.3| Graph (OX - size of arrays, OY - computation time) is given. Plot numpy and cupy implementations | 2 |

**Task 11**: CuPy-based Bifurcation map

Bifurcation map (again), now with use of CuPy

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T11.1| Bifurcation map is performed using cupy arrays | 2 |
|T11.2| Cupy bifurcation map implementation is profiled | 2 |
|T11.3| Graph (OX - number total iterations, OY - computation time) is given. Plot cpu, mpi and cupy implementations | 2 |

**Task 12**: Histogram


| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T12.1| Function from the jupyter notebook is plotted with suggested number of points	 | 1 |
|T12.2| Histogram is plotted | 2 |
|T12.3| Computation time of cp.sum() and np.sum() are provided (only those two functions, no need to include plot drawing time into it | 2 |

**Task 13** Image Blur


| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T13.1| Before and after pictures are given (any 100x100 image)	 | 1 |
|T13.2| Measure CuPy implementation time | 2 |
|T13.3| Measure NumPy implementation time | 2 |

**Task 14** Jitted Mandelbrot

Again Mandelbrot, now using JIT optimizer

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T14.1| Profile Mandelbrot code | 1 |
|T14.2| Numba is used | 1 |
|T14.3| Acceleration using numba achieved | 1 |
|T14.4|  Measure computation time Time(# iterations), two plots - numba & regular version | 1 |

**Task 15** Jitted N-body problem


| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T15.1| Jitted function "distance" is created (along the lines of "create_n_random_particles" function, which is given in jupyter file on Canvas) | 1 |
|T15.2| jitted function "distance" is created (along the lines of "create_n_random_particles" function, which is given in jupyter file on Canvass) | 1 |
|T15.3| Acceleration using numba achieved | 1 |

**Bonus task**: 3D Schelling's model

Implement Schelling's model in 3D space. Adding one more axis implies that there are more neighbors. The rest of the rules stays the same as for the 2D case.

The difficult part is visualization. The task is to use python (Paraview with python is also acceptable) to visualize animation in 3D. You can use any techniques to visualize 3D data. You can try to come up with a technique yourself, or you can try to google. The main task is to make the visualization understandable. Think of it that way: you need to make it so that anyone can understand what is going on if the visualization was in your presentation and you gave it to a scientific audience, but they know nothing about your research area.