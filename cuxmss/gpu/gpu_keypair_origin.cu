#ifndef USING_BDS

#include <string.h>
#include <iostream>
using namespace std;

#include "gpu_utils.h"
#include "gpu_hash_address.h"
#include "gpu_randombytes.h"

#include "gpu_xmss_core_origin.h"
#include "gpu_keypair_origin.h"

__device__ void dev_xmssmt_core_seed_keypair_origin(u8 *pk, u8 *sk, u8 *seed)
{
	/* We do not need the auth path in key generation, but it simplifies the
	   code to have just one treehash routine that computes both root and path
	   in one function. */
	u8 auth_path[TREE_HEIGHT * N];
	u32 top_tree_addr[8] = { 0 };

	dev_set_layer_addr(top_tree_addr, D - 1);

	/* Initialize index to 0. */
	memset(sk, 0, INDEX_BYTES);
	sk += INDEX_BYTES;

	/* Initialize SK_SEED and SK_PRF. */
	memcpy(sk, seed, 2 * N);

	/* Initialize PUB_SEED. */
	memcpy(sk + 3 * N, seed + 2 * N,  N);
	memcpy(pk + N, sk + 3 * N, N);

	/* Compute root node of the top-most subtree. */
	dev_treehash(pk, auth_path, sk, pk + N, 0, top_tree_addr);
	memcpy(sk + 2 * N, pk, N);
} // dev_xmssmt_core_seed_keypair_origin

__device__ void dev_xmssmt_core_seed_ip_keypair_origin(u8 *pk, u8 *sk, u8 *seed)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	/* We do not need the auth path in key generation, but it simplifies the
	   code to have just one treehash routine that computes both root and path
	   in one function. */
	u8 auth_path[TREE_HEIGHT * N];
	u32 top_tree_addr[8] = { 0 };

	dev_set_layer_addr(top_tree_addr, D - 1);

	/* Initialize index to 0. */
	if (tid == 0) memset(sk, 0, INDEX_BYTES);
	sk += INDEX_BYTES;

	/* Initialize SK_SEED and SK_PRF. */
	if (tid == 0) memcpy(sk, seed, 2 * N);

	/* Initialize PUB_SEED. */
	if (tid == 0) memcpy(sk + 3 * N, seed + 2 * N, N);
	if (tid == 0) memcpy(pk + N, sk + 3 * N, N);

	/* Compute root node of the top-most subtree. */
	if (TREE_HEIGHT == 5) {
		dev_treehash_parallel_5(pk, auth_path, sk, pk + N, 0, top_tree_addr);
	} else if (TREE_HEIGHT == 10) {
		dev_treehash_parallel_10(pk, auth_path, sk, pk + N, 0, top_tree_addr);
	} else if (TREE_HEIGHT == 16) {
		dev_treehash_parallel_16(pk, auth_path, sk, pk + N, 0, top_tree_addr);
	} else if (TREE_HEIGHT == 20) {
		dev_treehash_parallel_20(pk, auth_path, sk, pk + N, 0, top_tree_addr);
	} else {
		if (tid == 0) dev_treehash(pk, auth_path, sk, pk + N, 0, top_tree_addr);
	}
	if (tid == 0) memcpy(sk + 2 * N, pk, N);
} // dev_xmssmt_core_seed_ip_keypair_origin

__device__ void dev_xmssmt_core_keypair_origin(u8 *pk, u8 *sk)
{
	u8 seed[3 * N];

#ifndef USING_FIXED_SEEDS
	dev_randombytes(seed, 3 * N);
#else // ifndef USING_FIXED_SEEDS
	for (int i = 0; i < 3 * N; i++)
		seed[i] = 0;
#endif // ifndef USING_FIXED_SEEDS
	dev_xmssmt_core_seed_keypair_origin(pk, sk, seed);
} // dev_xmssmt_core_keypair_origin

__global__ void global_xmssmt_core_ip_keypair_origin(u8 *pk, u8 *sk)
{
	u8 seed[3 * N];

#ifndef USING_FIXED_SEEDS
	dev_randombytes(seed, 3 * N);
#else // ifndef USING_FIXED_SEEDS
	for (int i = 0; i < 3 * N; i++)
		seed[i] = 0;
#endif // ifndef USING_FIXED_SEEDS
	dev_xmssmt_core_seed_ip_keypair_origin(pk, sk, seed);
} // global_xmssmt_core_ip_keypair_origin

__global__ void global_xmssmt_core_keypair_origin(u8 *pk, u8 *sk)
{
	dev_xmssmt_core_keypair_origin(pk, sk);
} // global_xmssmt_core_keypair_origin

__global__ void global_xmssmt_core_dp_keypair_origin(u8 *pk, u8 *sk)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	dev_xmssmt_core_keypair_origin(pk + tid * PK_BYTES, sk + tid * SK_BYTES);
} // global_xmssmt_core_keypair_origin

/*
 * Generates a XMSS key pair for a given parameter set.
 * Format sk: [(32bit) index || SK_SEED || SK_PRF || root || PUB_SEED]
 * Format pk: [root || PUB_SEED], omitting algorithm OID.
 */
void face_xmssmt_core_keypair_origin(u8 *pk, u8 *sk)
{
	u8 *dev_pk = NULL, *dev_sk = NULL;
	int device = DEVICE_USED;

	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));

	CHECK(cudaDeviceSynchronize());

	global_xmssmt_core_dp_keypair_origin << < 1, 1 >> > (dev_pk, dev_sk);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(pk, dev_pk, PK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));

	cudaFree(dev_pk); cudaFree(dev_sk);
} // face_xmssmt_core_keypair_origin

void face_xmssmt_core_dp_keypair_origin(u8 *pk, u8 *sk, u32 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_pk = NULL, *dev_sk = NULL;
	int device = DEVICE_USED;
	int threads = 32;
	int blocks = 1;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	blocks = num / 32;

	CHECK(cudaMalloc((void **)&dev_pk, num * PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sk, num * SK_BYTES * sizeof(u8)));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaDeviceSynchronize());
	global_xmssmt_core_dp_keypair_origin << < blocks, threads >> >
		(dev_pk, dev_sk);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(pk, dev_pk, num * PK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sk, dev_sk, num * SK_BYTES * sizeof(u8), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	cudaFree(dev_pk); cudaFree(dev_sk);
} // face_xmssmt_core_dp_keypair_origin

// inner parallel version
void face_xmssmt_core_ip_keypair_origin(u8 *pk, u8 *sk)
{
	struct timespec start, stop;
	double result;
	u8 *dev_pk = NULL, *dev_sk = NULL;
	int device = DEVICE_USED;
	u32 threads = 1, blocks = 1, maxblock = 1;
	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties(&deviceProp, device);
	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));

	void *kernelArgs[] = { &dev_pk, &dev_sk };

	if (TREE_HEIGHT == 5) {
		threads = 32;
		blocks = WOTS_LEN + 3;
	} else if (TREE_HEIGHT == 10) {
		threads = 32;  // 32 > 64
		int numBlocksPerSm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_ip_keypair_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;
		blocks = 1;
		while (blocks <= maxblock / 2) {
			blocks *= 2;
		}
		blocks = maxblock;
	} else if (TREE_HEIGHT == 16) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_ip_keypair_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;
		blocks = 1;
		while (blocks <= maxblock / 2) {
			blocks *= 2;
		}
		blocks = maxblock;
	} else if (TREE_HEIGHT == 20) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_ip_keypair_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;

		blocks = 1;
		while (blocks <= maxblock / 2) {
			blocks *= 2;
		}
		blocks = maxblock;
	}
#ifdef PRINT_TIME
	printf("!! ip kengen %d %d %d\n", maxblock, threads, blocks);
#endif // ifdef PRINT_TIME

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaDeviceSynchronize());
	cudaLaunchCooperativeKernel((void*)global_xmssmt_core_ip_keypair_origin,
				    blocks, threads, kernelArgs);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(pk, dev_pk, PK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	cudaFree(dev_pk); cudaFree(dev_sk);
} // face_xmssmt_core_ip_keypair_origin

int gpu_xmss_keypair_origin(u8 *pk, u8 *sk)
{
	face_xmssmt_core_keypair_origin(pk, sk);

	return 0;
} // gpu_xmss_keypair_origin

int gpu_xmssmt_keypair_origin(u8 *pk, u8 *sk)
{
	face_xmssmt_core_keypair_origin(pk, sk);

	return 0;
} // gpu_xmssmt_keypair_origin

int gpu_xmss_dp_keypair_origin(u8 *pk, u8 *sk, u32 num)
{
	face_xmssmt_core_dp_keypair_origin(pk, sk, num);

	return 0;
} // gpu_xmss_dp_keypair_origin

int gpu_xmssmt_dp_keypair_origin(u8 *pk, u8 *sk, u32 num)
{
	face_xmssmt_core_dp_keypair_origin(pk, sk, num);

	return 0;
} // gpu_xmssmt_dp_keypair_origin

int gpu_xmss_ip_keypair_origin(u8 *pk, u8 *sk)
{
	face_xmssmt_core_ip_keypair_origin(pk, sk);

	return 0;
} // gpu_xmss_ip_keypair_origin

int gpu_xmssmt_ip_keypair_origin(u8 *pk, u8 *sk)
{
	face_xmssmt_core_ip_keypair_origin(pk, sk);

	return 0;
} // gpu_xmssmt_ip_keypair_origin

#endif // ifndef USING_BDS
