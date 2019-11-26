from mpi4py import MPI
import numpy as np
import argparse
import time

NUM_ITERS = 100
comm = MPI.COMM_WORLD
w_size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()
IS_MASTER = rank == 0

parser = argparse.ArgumentParser('Task8')
parser.add_argument('-s', type=int, default=2048, help='size of image')
FLAGS = parser.parse_args()

numDataPerRank = FLAGS.s // w_size  
data = None
if IS_MASTER:
    # data = np.random.randint(0,2,(FLAGS.s, FLAGS.s))
    data = np.eye(FLAGS.s, dtype=int)
recvbuf = np.empty((numDataPerRank, FLAGS.s), dtype=int) # allocate space for recvbuf
comm.Scatter(data, recvbuf, root=0)
received = np.empty_like(recvbuf, dtype=int)

# comm.Barrier()
# times = []
# for i in range(NUM_ITERS):
#     start = time.time()
if rank % 2:
    request = comm.Isend([recvbuf, MPI.INT], dest=(rank+1) % w_size, tag=22)
    request.wait()
    # request = comm.Irecv([received, MPI.INT], source=(rank-1) % w_size, tag=22)
    # request.wait()
else:
    request = comm.Irecv([received, MPI.INT], source=(rank-1) % w_size, tag=22)
    request.wait()
    # request = comm.Isend([recvbuf, MPI.INT], dest=(rank+1) % w_size, tag=22)
#     t = time.time() - start
#     times.append(t)
comm.Barrier()
if rank % 2:
    request = comm.Irecv([received, MPI.INT], source=(rank-1) % w_size, tag=22)
    request.wait()
    # request = comm.Irecv([received, MPI.INT], source=(rank-1) % w_size, tag=22)
    # request.wait()
else:
    request = comm.Isend([recvbuf, MPI.INT], dest=(rank+1) % w_size, tag=22)
    request.wait()


# if IS_MASTER:
#     print('Time of shift: {:.3f}±{:.3f}'.format(np.mean(times), np.std(times)))
time.sleep(rank/100)
print('Rank: ',rank, ', recvbuf received: \n',received)

