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

**Task 3**

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

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T7.1| Analytical computation of the integral is given | 1 |
|T7.2| Trapezoidal approximation is used | 1 |
|T7.3| We can arbitrarily choose the number of MPI processes that we want to launch and it does not depend on the number of discretization points | 2 |
|T7.4| Speedup versus number of processors graph is plotted |1|
|T7.5| Error value versus discretization step graph is plotted|	1 |

**Task 8**: Columnwise shifted pictures

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T8.1| We can arbitrarily choose the number of MPI processes that we want to launch and it does not depend on the width of the picture | 2 |
|T8.2|	Speedup versus number of processors graph is plotted. | 2 |
|T8.3| Total memory consumption versus number of processors is plotted. | 2 |


**Task 9**: Conway's Game of Life

| | Criteria name  | Max score |
| -- | ------------- | ------------- |
|T9.1| Sequential implementation is provided | 1 |
|T9.2| Parallel implementation is provided | 2 |
|T9.3| Add parameter to choose # of MPI processes | 2 |
|T9.4| "Gosper's gun" initial condition is given | 1 |
|T9.5| At least two more initial conditions are modelled | 2 |
|T9.6| Plot graph living cells / time for 3 different initial conditions | 2 |	


**Bonus task**: 3D Schelling's model

Implement Schelling's model in 3D space. Adding one more axis implies that there are more neighbors. The rest of the rules stays the same as for the 2D case.

The difficult part is visualization. The task is to use python (Paraview with python is also acceptable) to visualize animation in 3D. You can use any techniques to visualize 3D data. You can try to come up with a technique yourself, or you can try to google. The main task is to make the visualization understandable. Think of it that way: you need to make it so that anyone can understand what is going on if the visualization was in your presentation and you gave it to a scientific audience, but they know nothing about your research area.