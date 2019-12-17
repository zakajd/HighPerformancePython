#try to parralell
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from random import random
import os 
os.environ["OMP_NUM_THREADS"] = "1"

def map_fun(r, x):
    return r*x*(1-x)

x_i = random()  #initial value of x
n = 200  #number of iterations
last = 200  #number of last values that we take into account
number_r = 1000

comm = MPI.COMM_WORLD #create a communicator
rank = comm.Get_rank() #rank of executing process
size = comm.Get_size() #gives number of ranks in comm
#print("my rank is ", rank)

t1 = MPI.Wtime()
r_values = np.linspace(0, 4, number_r)
r_list = np.array_split(r_values,size)[rank]
numDataPerRank = int((r_values.size)/size)


#sendbuf = np.array([])
sendbuf = []
R = []
x_n = []
#for r in r_values[rank*numDataPerRank: (rank+1)*numDataPerRank]:
for r in r_list:
    x_i = random()
    for i in range(n+last):
        x_i = map_fun(r, x_i)
        if i >= n:
            x_n.append(x_i)
            R.append(r)


sendbuf = x_n
  
t2 = MPI.Wtime()        
#comm.Gather(sendbuf, recvbuf, root=0)
recieved_x = comm.gather(sendbuf,root=0) 
recieved_r = comm.gather(R,root=0) 

if rank == 0:
    recv_x = np.hstack(recieved_x) 
    recv_r = np.hstack(recieved_r)

#print('Rank: ',rank, ', recvbuf received: ',recvbuf, ', size', len(recvbuf))
    #fig = plt.figure()
    #plt.plot(recv_r, recv_x, ls='',marker=',')
    #plt.show()
    #fig.savefig('Bifurcation map with n = {} processes'.format(size))