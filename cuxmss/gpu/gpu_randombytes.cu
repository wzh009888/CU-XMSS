/*
   This code was taken from the SPHINCS reference implementation and is public domain.
 */

#include <fcntl.h>
#include <unistd.h>

static int fd = -1;

void randombytes(unsigned char *x, unsigned long long xlen)
{
	int i;

	if (fd == -1) {
		for (;;) {
			fd = open("/dev/urandom", O_RDONLY);
			if (fd != -1) {
				break;
			}
			sleep(1);
		}
	}

	while (xlen > 0) {
		if (xlen < 1048576) {
			i = xlen;
		}else {
			i = 1048576;
		}

		i = read(fd, x, i);
		if (i < 1) {
			sleep(1);
			continue;
		}

		x += i;
		xlen -= i;
	}
} // randombytes

#include "curand_kernel.h"

// 这个函数可能影响性能
// 毋庸置疑，相同的seed、sequence总是会生成相同的伪随机数，
// 并且在调用2 67 ⋅ sequence + offset次curand函数后，也会生成相同的伪随机数。
// 另外补充，不同seed生成的伪随机序列通常是不想关的，但是也存在相关的可能。
// 同一个seed不同sequence生成的伪随机序列一定不相关。
// 针对并行计算，CUDA给出了一些建议：每个核函数可以使用同一个seed，但是最好每个线程都是不同的sequence。
// 每个核函数一个seed和sequence，会导致程序效率异常低下。

__device__ void dev_randombytes(unsigned char *x, unsigned long long xlen)
{
	unsigned long long i;
	unsigned long long sequence;
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	sequence = tid;

	curandState state;

	curand_init(clock64(), sequence, 0, &state);

	for (i = 0; i < xlen; i++) {
		x[i] = curand_uniform(&state) * 255;
		sequence++;
	}
} // dev_randombytes
