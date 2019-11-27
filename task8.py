from mpi4py import MPI
import numpy as np
import argparse
import time
# import tracemalloc

comm = MPI.COMM_WORLD
w_size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()
IS_MASTER = rank == 0

parser = argparse.ArgumentParser('Task8')
parser.add_argument('-s', type=int, default=2**12, help='size of image')
parser.add_argument('-i', type=int, default=50, help='num iters')
FLAGS = parser.parse_args()
FLAGS.s = (FLAGS.s // w_size) * w_size

numDataPerRank = FLAGS.s // w_size  
data = None
received_data = None
if IS_MASTER:
    # data = np.random.randint(0,2,(FLAGS.s, FLAGS.s))
    data = np.eye(FLAGS.s, dtype=int)
    received_data = np.empty_like(data)
    # print(data)
received = np.empty((numDataPerRank, FLAGS.s), dtype=int) # allocate space for recvbuf

def roll(arr):
    # using not efficient function to show speedup
    res = np.empty_like(arr)
    w = arr.shape[1]
    for i in range(w):
        res[:, (i + 5) % w] = arr[:, i]
    return res
# tracemalloc.start()

times = []
for i in range(FLAGS.i):
    start = time.time()
    comm.Scatter(data, received, root=0)
    received_rolled = roll(received)
    comm.Gather(received_rolled, received_data, root=0)
    end = time.time()
    times.append(end - start)

if IS_MASTER:
    # current, peak = tracemalloc.get_traced_memory()
    # print("Curr: {:.3f}. Peak: {:.3f}".format(current / 2**10, peak / 2**10))
    print(np.mean(times), np.std(times))

# if IS_MASTER:
#     print(received_data)