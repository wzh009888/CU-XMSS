#include "gpu_utils.h"
#include <iostream>
using namespace std;

__device__ void dev_ull_to_bytes(u8 *out, u32 outlen,
				 u64 in)
{
	int i;

	/* Iterate over out in decreasing order, for big-endianness. */
	for (i = outlen - 1; i >= 0; i--) {
		out[i] = in & 0xff;
		in = in >> 8;
	}
} // dev_ull_to_bytes

__global__ void global_ull_to_bytes(u8 *out, u32 outlen,
				    u64 *in)
{
	int i;

	/* Iterate over out in decreasing order, for big-endianness. */
	for (i = outlen - 1; i >= 0; i--) {
		out[i] = in[0] & 0xff;
		in[0] = in[0] >> 8;
	}
} // global_ull_to_bytes

void face_ull_to_bytes(u8 *out, u32 outlen, u64 in)
{
	CHECK(cudaSetDevice(0));
	u8 *dev_out = NULL;
	u64 *dev_in = NULL;

	CHECK(cudaMalloc((void **)&dev_in, 1 * sizeof(u64)));
	CHECK(cudaMalloc((void **)&dev_out, outlen * sizeof(u8)));
	CHECK(cudaMemcpy(dev_in, &in, 1 * sizeof(u64),
			 HOST_2_DEVICE));

	CHECK(cudaDeviceSynchronize());

	global_ull_to_bytes << < 1, 1 >> > (dev_out, outlen, dev_in);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(out, dev_out, outlen * sizeof(u8), cudaMemcpyDeviceToHost));

	cudaFree(dev_in); cudaFree(dev_out);
} // face_ull_to_bytes

__device__ u64 dev_bytes_to_ull(const u8 *in, u32 inlen)
{
	u64 retval = 0;
	u32 i;

	for (i = 0; i < inlen; i++)
		retval |= ((u64)in[i]) << (8 * (inlen - 1 - i));
	return retval;
} // dev_bytes_to_ull
