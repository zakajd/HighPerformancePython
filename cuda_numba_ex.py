"""
SAXPY
-----
Compute `a * x + y`
Where
	`a` is a scalar
	`x` and `y` are vectors
Prefix 'S' indicates single-precision float32 operations
"""
from __future__ import print_function
import sys
import numpy
from numba import cuda, vectorize, float32, void
from timeit import default_timer as timer

# GPU code
# ---------

@cuda.jit(void(float32, float32[:], float32[:], float32[:]))
def saxpy(a, x, y, out):
	# Short for cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	i = cuda.grid(1)
	# Map i to array elements
	if i >= out.size:
		# Out of range?
		return
	# Do actual work
	out[i] = a * x[i] + y[i]

NUM_BLOCKS = 65536
NUM_THREADS = 256
NELEM = NUM_BLOCKS * NUM_THREADS

def main():
	a = numpy.float32(2.)				# Force value to be float32
	x = numpy.arange(NELEM, dtype='float32')
	y = numpy.arange(NELEM, dtype='float32')

	s = timer()
	z = a * x + y
	e = timer()
	print("CALCULATION 1")
	print("Computational time on cpu")
	print(e-s)

	s = timer()
	dx = cuda.to_device(x)
	dy = cuda.to_device(y)
	dout = cuda.device_array_like(x)

	griddim = NUM_BLOCKS
	blockdim = NUM_THREADS

	t = timer()
	saxpy[griddim, blockdim](a, dx, dy, dout)
	k = timer()

	out = dout.copy_to_host()
	e = timer()
	print()
	print("CALCULATION 2")
	print("Data transfer time")
	print(t-s+e-k)
	print("Computation time on gpu with cuda.jit")
	print(k-t)

	if numpy.allclose(z, out):
		print("Correct result")
	else:
		print("Incorrect result")


if __name__ == '__main__':
	main()

