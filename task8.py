from mpi4py import MPI
import numpy as np
import argparse
import time

comm = MPI.COMM_WORLD
w_size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()
IS_MASTER = rank == 0

parser = argparse.ArgumentParser('Task8')
parser.add_argument('-s', type=int, default=2048, help='size of image')
parser.add_argument('-i', type=int, default=50, help='num iters')
FLAGS = parser.parse_args()
FLAGS.s = (FLAGS.s // w_size) * w_size

numDataPerRank = FLAGS.s // w_size  
data = None

if IS_MASTER:
    # data = np.random.randint(0,2,(FLAGS.s, FLAGS.s))
    data = np.eye(FLAGS.s, dtype=int)
recvbuf = np.empty((numDataPerRank, FLAGS.s), dtype=int) # allocate space for recvbuf
comm.Scatter(data, recvbuf, root=0)
received = np.empty_like(recvbuf, dtype=int)

times = []
for i in range(FLAGS.i):
    comm.Barrier()
    start = time.time()
    if rank % 2:
        request = comm.Send([recvbuf, MPI.INT], dest=(rank+1) % w_size, tag=22)
        request = comm.Recv([received, MPI.INT], source=(rank-1) % w_size, tag=22)
    else:
        request = comm.Recv([received, MPI.INT], source=(rank-1) % w_size, tag=22)
        request = comm.Send([recvbuf, MPI.INT], dest=(rank+1) % w_size, tag=22)
    times.append(time.time() - start)

if IS_MASTER:
    print(np.mean(times), np.std(times))
    # print('Time of shift: {:.3f}±{:.3f}'.format(np.mean(times), np.std(times)))
# time.sleep(rank/100)
# print('Rank: ',rank, ', recvbuf received: \n',received)

