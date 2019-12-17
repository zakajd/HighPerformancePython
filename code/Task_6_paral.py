import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))

def my_imshow(x, y, z, w,
              title,
              xlabel,
              ylabel,
              grid_active=False, fig_x_size=15, fig_y_size=10, font_param=20):
    plt.figure(figsize=(fig_x_size, fig_y_size))
    im = plt.imshow(z, aspect='auto',
                    origin='lower',
                    extent=[min(x) / 2 / np.pi, max(x) / 2 / np.pi, y[0], 2 * w[int(len(x) / 2) - 1]])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=font_param)
    plt.title(title, fontsize=font_param * 1.3)
    plt.xlabel(xlabel, fontsize=font_param)
    plt.ylabel(ylabel, fontsize=font_param)
    plt.xticks(fontsize=font_param)
    plt.yticks(fontsize=font_param)
    plt.grid(grid_active)
    plt.ylim(0, 10)
    return im


def form_signal(n_timestamps = 4096):
    t=np.linspace(-20*2*np.pi, 20*2*np.pi, n_timestamps)
    y=np.sin(t)*np.exp(-t**2/2/20**2)               #generate first  wave packets of harmonic signal
    y=y+np.sin(3*t)*np.exp(-(t-5*2*np.pi)**2/2/20**2)  #add      second wave packets of harmonic signal
    y=y+np.sin(4*t)*np.exp(-(t-5*2*np.pi)**2/2/20**2)
    y=y+np.sin(5*t)*np.exp(-(t-10*2*np.pi)**2/2/10**2) #add      third  wave packets of harmonic signal
    return t, y


def window_function(t, window_position, window_width):
    return np.exp(- (t - window_position) ** 2 / 2 / window_width ** 2)


def get_specgram(window_width, t, y, nwindowsteps = 1000):
    t_window_positions=np.linspace(-20 * 2 * np.pi, 20 * 2 * np.pi, nwindowsteps)

    specgram = np.empty([len(t), len(t_window_positions)])

    for i,t_window_position in enumerate(t_window_positions):
        y_window = y * window_function(t, t_window_position, window_width)
        #plot(y_window)
        specgram[:,i]=abs(np.fft.fft(y_window))

    return specgram


def plot_specfram(window_width_given, nwindowsteps_given, t, y, w):
    im = my_imshow(t, w, get_specgram(window_width = window_width_given, t=t, y=y,
                                 nwindowsteps = nwindowsteps_given), w,
              title = "Specgram nwindowsteps_given = "+str(nwindowsteps_given),
              xlabel = "t, cycles",
              ylabel = "Frequency, arb. units")
    # clim(0,0.5)
    plt.ylim(0, 10)
    #im = plt.savefig('fig_paral/Specgram.png')
    im = plt.savefig('fig_paral/Specgram nwindowsteps_given = '+str(nwindowsteps_given)+'.png')


def compute_specg(n_timestamps_given):
    kappa = 1
    window_width_given = kappa * 2 * np.pi
    t, y = form_signal(n_timestamps = n_timestamps_given)

    sp=np.fft.fft(y)
    w=np.fft.fftfreq(len(y), d=(t[1]-t[0])/2/np.pi)

    plot_specfram(window_width_given, n_timestamps_given, t, y, w)


comm = MPI.COMM_WORLD
sendbuf = []
root = 0

if comm.rank == 0:
    init_timestamp = 4096
    final_timestamp = 4100
    step_timestamp = (final_timestamp - init_timestamp)//comm.size

    sendbuf = np.zeros((comm.size, step_timestamp+1))

    for i in range(int(comm.size)):
        buf = np.arange((4096+i*step_timestamp), (4096+(i+1)*step_timestamp+1)) #4501)
        print(buf)
        print(sendbuf[i:])
        sendbuf[i:] = buf

n_timestamps_set = comm.scatter(sendbuf, root)

for n_timestamps_given in n_timestamps_set:
    compute_specg(n_timestamps_given)

if comm.rank == 0:
    print("Finish")
    print("--- %s seconds ---" % (time.time() - start_time))


#time_set = []
#n_timestamps_set = np.arange(4096, 5000)  # ,5101)


#for n_timestamps_given in n_timestamps_set:
#    compute_specg(n_timestamps_given)
#    time_set.append(time.time() - start_time)
