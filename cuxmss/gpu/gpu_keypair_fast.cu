#ifdef USING_BDS

#include "gpu_utils.h"
#include "gpu_wots.h"
#include "gpu_xmss_commons.h"
#include "gpu_hash.h"
#include "gpu_hash_address.h"
#include "gpu_randombytes.h"

#include "gpu_xmss_core_fast.h"
#include "gpu_keypair_fast.h"

#include <string.h>
#include <iostream>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
using namespace std;

__device__ u8 one_wots_sigs[WOTS_SIG_BYTES * D];

__device__ void dev_xmss_core_keypair_fast(u8 *pk, u8 *sk)
{
	u32 addr[8] = { 0 };

	// TODO refactor BDS state not to need separate treehash instances
	bds_state state;
	treehash_inst treehash[TREE_HEIGHT - BDS_K];

	state.treehash = treehash;

	dev_xmss_deserialize_state(&state, sk);

	state.stackoffset = 0;
	state.next_leaf = 0;

	// Set idx = 0
	sk[0] = 0;
	sk[1] = 0;
	sk[2] = 0;
	sk[3] = 0;

#ifndef USING_FIXED_SEEDS
	// Init SK_SEED (n byte) and SK_PRF (n byte)
	dev_randombytes(sk + INDEX_BYTES, 2 * N);
	// Init PUB_SEED (n byte)
	dev_randombytes(sk + INDEX_BYTES + 3 * N, N);
#else // ifndef USING_FIXED_SEEDS
	for (u32 iter = 0; iter < 2 * N; iter++)
		sk[INDEX_BYTES + iter] = 0;
	for (u32 iter = 0; iter < N; iter++)
		sk[INDEX_BYTES + 3 * N + iter] = 0;
#endif // ifndef USING_FIXED_SEEDS
	// Copy PUB_SEED to public key
	memcpy(pk + N, sk + INDEX_BYTES + 3 * N, N);

	// Compute root, whole cost is here
	dev_treehash_init(pk, TREE_HEIGHT, 0, &state,
			  sk + INDEX_BYTES, sk + INDEX_BYTES + 3 * N, addr);
	// copy root to sk
	memcpy(sk + INDEX_BYTES + 2 * N, pk, N);

	/* Write the BDS state into sk. */
	dev_xmss_serialize_state(sk, &state);
} // dev_xmss_core_keypair_fast

__global__ void global_xmss_core_dp_keypair_fast(u8 *pk, u8 *sk, u32 dp_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < dp_num)
		dev_xmss_core_keypair_fast(pk + tid * PK_BYTES, sk + tid * SK_BYTES);
} // global_xmss_core_dp_keypair_fast

__global__ void global_xmss_core_keypair_fast(u8 *pk, u8 *sk)
{
	u32 addr[8] = { 0 };

	// TODO refactor BDS state not to need separate treehash instances
	bds_state state;
	treehash_inst treehash[TREE_HEIGHT - BDS_K];

	state.treehash = treehash;

	dev_xmss_deserialize_state(&state, sk);

	state.stackoffset = 0;
	state.next_leaf = 0;

	// Set idx = 0
	sk[0] = 0;
	sk[1] = 0;
	sk[2] = 0;
	sk[3] = 0;
#ifndef USING_FIXED_SEEDS
	// Init SK_SEED (n byte) and SK_PRF (n byte)
	dev_randombytes(sk + INDEX_BYTES, 2 * N);
	// Init PUB_SEED (n byte)
	dev_randombytes(sk + INDEX_BYTES + 3 * N, N);
#else // ifndef USING_FIXED_SEEDS
	for (u32 iter = 0; iter < 2 * N; iter++)
		sk[INDEX_BYTES + iter] = 0;
	for (u32 iter = 0; iter < N; iter++)
		sk[INDEX_BYTES + 3 * N + iter] = 0;
#endif // ifndef USING_FIXED_SEEDS

	// Copy PUB_SEED to public key
	memcpy(pk + N, sk + INDEX_BYTES + 3 * N, N);

	dev_treehash_init(pk, TREE_HEIGHT, 0, &state,
			  sk + INDEX_BYTES, sk + INDEX_BYTES + 3 * N, addr);

	// copy root to sk
	memcpy(sk + INDEX_BYTES + 2 * N, pk, N);

	/* Write the BDS state into sk. */
	dev_xmss_serialize_state(sk, &state);

} // global_xmss_core_keypair_fast

__global__ void global_xmss_core_ip_keypair_fast(u8 *pk, u8 *sk)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	u32 addr[8] = { 0 };

	// TODO refactor BDS state not to need separate treehash instances
	bds_state state;

	if (tid == 0) {
		treehash_inst treehash[TREE_HEIGHT - BDS_K];
		state.treehash = treehash;
		dev_xmss_deserialize_state(&state, sk);

		state.stackoffset = 0;
		state.next_leaf = 0;

		// Set idx = 0
		sk[0] = 0;
		sk[1] = 0;
		sk[2] = 0;
		sk[3] = 0;

#ifndef USING_FIXED_SEEDS
		// Init SK_SEED (n byte) and SK_PRF (n byte)
		dev_randombytes(sk + INDEX_BYTES, 2 * N);
		// Init PUB_SEED (n byte)
		dev_randombytes(sk + INDEX_BYTES + 3 * N, N);
#else // ifndef USING_FIXED_SEEDS
		for (u32 iter = 0; iter < 2 * N; iter++)
			sk[INDEX_BYTES + iter] = 0;
		for (u32 iter = 0; iter < N; iter++)
			sk[INDEX_BYTES + 3 * N + iter] = 0;
#endif // ifndef USING_FIXED_SEEDS
		// Copy PUB_SEED to public key
		memcpy(pk + N, sk + INDEX_BYTES + 3 * N, N);
	}

	if (TREE_HEIGHT == 10) {
		// dev_treehash_init_parallel_10(pk, TREE_HEIGHT, 0, &state,
		dev_treehash_init_parallel_10(pk, TREE_HEIGHT, 0, &state,
					      sk + INDEX_BYTES, sk + INDEX_BYTES + 3 * N, addr);
	} else if (TREE_HEIGHT == 16) {
		dev_treehash_init_parallel_16(pk, TREE_HEIGHT, 0, &state,
					      sk + INDEX_BYTES, sk + INDEX_BYTES + 3 * N, addr);
	} else if (TREE_HEIGHT == 20) {
		dev_treehash_init_parallel_20(pk, TREE_HEIGHT, 0, &state,
					      sk + INDEX_BYTES, sk + INDEX_BYTES + 3 * N, addr);
	}

	if (tid == 0) {
		// copy root to sk
		memcpy(sk + INDEX_BYTES + 2 * N, pk, N);

		/* Write the BDS state into sk. */
		dev_xmss_serialize_state(sk, &state);
	}
} // global_xmss_core_ip_keypair_fast

// seiral version
void face_xmss_core_keypair_fast(u8 *pk, u8 *sk)
{
	u8 *dev_pk = NULL, *dev_sk = NULL;
	int device = DEVICE_USED;

	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));

	CHECK(cudaDeviceSynchronize());

	global_xmss_core_keypair_fast << < 1, 1 >> > (dev_pk, dev_sk);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(pk, dev_pk, PK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));

	cudaFree(dev_pk); cudaFree(dev_sk);
} // face_xmss_core_sign

// inner parallel version
void face_xmss_core_ip_keypair_fast(u8 *pk, u8 *sk)
{
	struct timespec start, stop;
	double result;
	u8 *dev_pk = NULL, *dev_sk = NULL;
	int device = DEVICE_USED;
	int threads = 1, blocks = 1, maxblock = 1;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));

	void *kernelArgs[] = { &dev_pk, &dev_sk };

	if (TREE_HEIGHT == 10) {
		threads = 32;  // 32 > 64
		int numBlocksPerSm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmss_core_ip_keypair_fast, threads, 0);
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
			(&numBlocksPerSm, global_xmss_core_ip_keypair_fast, threads, 0);
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
			(&numBlocksPerSm, global_xmss_core_ip_keypair_fast, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;
		blocks = 1;
		while (blocks < maxblock / 2) {
			blocks *= 2;
		}
		blocks = maxblock;
	}
#ifdef PRINT_TIME
	printf("xmss ip keygen %d %d %d\n", maxblock, threads, blocks);
#endif // ifdef PRINT_TIME

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaDeviceSynchronize());
	cudaLaunchCooperativeKernel((void*)global_xmss_core_ip_keypair_fast,
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
} // face_xmss_core_ip_keypair

__device__ void dev_xmssmt_core_keypair_fast(u8 *pk, u8 *sk)
{
	u32 addr[8] = { 0 };
	unsigned int i;
	u8 *wots_sigs;

	// TODO refactor BDS state not to need separate treehash instances
	bds_state states[2 * D - 1];
	treehash_inst treehash[(2 * D - 1) * (TREE_HEIGHT - BDS_K)];

	for (i = 0; i < 2 * D - 1; i++) {
		states[i].treehash = treehash + i * (TREE_HEIGHT - BDS_K);
	}

	dev_xmssmt_deserialize_state(states, &wots_sigs, sk);

	for (i = 0; i < 2 * D - 1; i++) {
		states[i].stackoffset = 0;
		states[i].next_leaf = 0;
	}

	// Set idx = 0
	for (i = 0; i < INDEX_BYTES; i++) {
		sk[i] = 0;
	}
#ifndef USING_FIXED_SEEDS
	// Init SK_SEED (N byte) and SK_PRF (N byte)
	dev_randombytes(sk + INDEX_BYTES, 2 * N);
	// Init PUB_SEED (N byte)
	dev_randombytes(sk + INDEX_BYTES + 3 * N, N);
#else // ifndef USING_FIXED_SEEDS
	for (u32 iter = 0; iter < 2 * N; iter++)
		sk[INDEX_BYTES + iter] = 0;
	for (u32 iter = 0; iter < N; iter++)
		sk[INDEX_BYTES + 3 * N + iter] = 0;
#endif // ifndef USING_FIXED_SEEDS
	// Copy PUB_SEED to public key
	memcpy(pk + N, sk + INDEX_BYTES + 3 * N, N);

	// Start with the bottom-most layer
	dev_set_layer_addr(addr, 0);
	// Set up state and compute wots signatures for all but topmost tree root

#if D > 1
	for (i = 0; i < D - 1; i++) {
		// Compute seed for OTS key pair
		dev_treehash_init(pk, TREE_HEIGHT, 0, states + i, sk + INDEX_BYTES, pk + N, addr);
		dev_set_layer_addr(addr, (i + 1));
		dev_wots_sign(wots_sigs + i * WOTS_SIG_BYTES, pk, sk + INDEX_BYTES, pk + N, addr);
	}
#endif // if D > 1
	// Address now points to the single tree on layer d-1
	dev_treehash_init(pk, TREE_HEIGHT, 0, states + i, sk + INDEX_BYTES, pk + N, addr);
	memcpy(sk + INDEX_BYTES + 2 * N, pk, N);

	dev_xmssmt_serialize_state(sk, states);
} // dev_xmssmt_core_keypair_fast

__global__ void global_xmssmt_core_keypair_fast(u8 *pk, u8 *sk)
{
	u32 addr[8] = { 0 };
	unsigned int i;
	u8 *wots_sigs;

	// TODO refactor BDS state not to need separate treehash instances
	bds_state states[2 * D - 1];
	treehash_inst treehash[(2 * D - 1) * (TREE_HEIGHT - BDS_K)];

	for (i = 0; i < 2 * D - 1; i++) {
		states[i].treehash = treehash + i * (TREE_HEIGHT - BDS_K);
	}

	dev_xmssmt_deserialize_state(states, &wots_sigs, sk);

	for (i = 0; i < 2 * D - 1; i++) {
		states[i].stackoffset = 0;
		states[i].next_leaf = 0;
	}

	// Set idx = 0
	for (i = 0; i < INDEX_BYTES; i++) {
		sk[i] = 0;
	}
#ifndef USING_FIXED_SEEDS
	// Init SK_SEED (N byte) and SK_PRF (N byte)
	dev_randombytes(sk + INDEX_BYTES, 2 * N);
	// Init PUB_SEED (N byte)
	dev_randombytes(sk + INDEX_BYTES + 3 * N, N);
#else // ifndef USING_FIXED_SEEDS
	for (u32 iter = 0; iter < 2 * N; iter++)
		sk[INDEX_BYTES + iter] = 0;
	for (u32 iter = 0; iter < N; iter++)
		sk[INDEX_BYTES + 3 * N + iter] = 0;
#endif // ifndef USING_FIXED_SEEDS
	// Copy PUB_SEED to public key
	memcpy(pk + N, sk + INDEX_BYTES + 3 * N, N);

	// Start with the bottom-most layer
	dev_set_layer_addr(addr, 0);
	// Set up state and compute wots signatures for all but topmost tree root
#if D > 1
	for (i = 0; i < D - 1; i++) {
		// Compute seed for OTS key pair
		dev_treehash_init(pk, TREE_HEIGHT, 0, states + i, sk + INDEX_BYTES, pk + N, addr);
		dev_set_layer_addr(addr, (i + 1));
		dev_wots_sign(wots_sigs + i * WOTS_SIG_BYTES, pk, sk + INDEX_BYTES, pk + N, addr);
	}
#endif // if D > 1
	// Address now points to the single tree on layer d-1
	dev_treehash_init(pk, TREE_HEIGHT, 0, states + i, sk + INDEX_BYTES, pk + N, addr);
	memcpy(sk + INDEX_BYTES + 2 * N, pk, N);

	dev_xmssmt_serialize_state(sk, states);

	return;
} // global_xmssmt_core_keypair_fast

__global__ void global_xmssmt_core_ip_keypair_fast(u8 *pk, u8 *sk)
{
	cooperative_groups::grid_group g = cooperative_groups::this_grid();
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	u32 addr[8] = { 0 };
	unsigned int i;
	u8 *wots_sigs;

	// TODO refactor BDS state not to need separate treehash instances
	bds_state states[2 * D - 1];

	if (tid == 0) {
		treehash_inst treehash[(2 * D - 1) * (TREE_HEIGHT - BDS_K)];
		for (i = 0; i < 2 * D - 1; i++) {
			states[i].treehash = treehash + i * (TREE_HEIGHT - BDS_K);
		}

		dev_xmssmt_deserialize_state(states, &wots_sigs, sk);

		for (i = 0; i < 2 * D - 1; i++) {
			states[i].stackoffset = 0;
			states[i].next_leaf = 0;
		}

		// Set idx = 0
		for (i = 0; i < INDEX_BYTES; i++) {
			sk[i] = 0;
		}
	#ifndef USING_FIXED_SEEDS
		// Init SK_SEED (N byte) and SK_PRF (N byte)
		dev_randombytes(sk + INDEX_BYTES, 2 * N);
		// Init PUB_SEED (N byte)
		dev_randombytes(sk + INDEX_BYTES + 3 * N, N);
	#else // ifndef USING_FIXED_SEEDS
		for (int j = 0; j < 2 * N; j++)
			sk[INDEX_BYTES + j] = 0;
		for (int j = 0; j < N; j++)
			sk[INDEX_BYTES + 3 * N + j] = 0;
	#endif // ifndef USING_FIXED_SEEDS
		// Copy PUB_SEED to public key
		memcpy(pk + N, sk + INDEX_BYTES + 3 * N, N);

		// Start with the bottom-most layer
		dev_set_layer_addr(addr, 0);
	}
	g.sync();

	// Set up state and compute wots signatures for all but topmost tree root
#if D > 1
	const unsigned int tnum = gridDim.x * blockDim.x;
	for (i = 0; i < D - 1; i++) {
		// Compute seed for OTS key pair
		if (TREE_HEIGHT == 5) {
			dev_treehash_init_parallel_5(pk, TREE_HEIGHT, 0, states + i,
						     sk + INDEX_BYTES, pk + N, addr);
			dev_set_layer_addr(addr, i + 1); // all threads run this line
			g.sync();

			int offset = 32 * WOTS_LEN;
			if (tnum - offset < WOTS_LEN) offset = 0;
			if (tnum < WOTS_LEN) printf("wrong thread size\n");
			// offset = 0;

			dev_wots_sign_parallel(one_wots_sigs + i * WOTS_SIG_BYTES,
					       pk, sk + INDEX_BYTES, pk + N, addr, offset);

		} else if (TREE_HEIGHT == 10) {
			dev_treehash_init_parallel_10(pk, TREE_HEIGHT, 0, states + i,
						      sk + INDEX_BYTES, pk + N, addr);
			dev_set_layer_addr(addr, i + 1); // all threads run this line
			g.sync();
			int offset = 1;
			while (offset <= tnum / 2) offset *= 2;
			if (tnum - offset < WOTS_LEN) offset = 0;
			if (tnum < WOTS_LEN) printf("wrong thread size\n");

			dev_wots_sign_parallel(one_wots_sigs + i * WOTS_SIG_BYTES,
					       pk, sk + INDEX_BYTES, pk + N, addr, offset);

		} else if (TREE_HEIGHT == 20) {
			dev_treehash_init_parallel_20(pk, TREE_HEIGHT, 0, states + i,
						      sk + INDEX_BYTES, pk + N, addr);
			dev_set_layer_addr(addr, i + 1); // all threads run this line
			g.sync();
			if (tnum < WOTS_LEN) printf("wrong thread size\n");
			dev_wots_sign_parallel(one_wots_sigs + i * WOTS_SIG_BYTES,
					       pk, sk + INDEX_BYTES, pk + N, addr, 0);
		}
	}


#endif // if D > 1
	// Address now points to the single tree on layer d-1
	if (TREE_HEIGHT == 5) {
		dev_treehash_init_parallel_5(pk, TREE_HEIGHT, 0, states + i,
					     sk + INDEX_BYTES, pk + N, addr);
	} else if (TREE_HEIGHT == 10) {
		dev_treehash_init_parallel_10(pk, TREE_HEIGHT, 0, states + i,
					      sk + INDEX_BYTES, pk + N, addr);
	} else if (TREE_HEIGHT == 20) {
		dev_treehash_init_parallel_20(pk, TREE_HEIGHT, 0, states + i,
					      sk + INDEX_BYTES, pk + N, addr);
	}
	if (tid == 0) memcpy(sk + INDEX_BYTES + 2 * N, pk, N);

	g.sync();
	if (tid == 0) {
		int offset = INDEX_BYTES + 4 * N + (2 * D - 1) * \
			     (N + 9 + (TREE_HEIGHT >> 1) * N + TREE_HEIGHT * (8 + 3 * N));
		memcpy(sk + offset, one_wots_sigs, D * WOTS_SIG_BYTES);
	}

	if (tid == 0) dev_xmssmt_serialize_state(sk, states);

	return;
} // global_xmssmt_core_ip_keypair_fast

__global__ void global_xmssmt_core_dp_keypair_fast(u8 *pk, u8 *sk, u32 dp_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < dp_num)
		dev_xmssmt_core_keypair_fast(pk + tid * PK_BYTES, sk + tid * SK_BYTES);
} // global_xmssmt_core_dp_keypair

void face_xmssmt_core_keypair_fast(u8 *pk, u8 *sk)
{
	u8 *dev_pk = NULL, *dev_sk = NULL;
	int device = DEVICE_USED;

	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));

	CHECK(cudaDeviceSynchronize());

	global_xmssmt_core_keypair_fast << < 1, 1 >> > (dev_pk, dev_sk);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(pk, dev_pk, PK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));

	cudaFree(dev_pk); cudaFree(dev_sk);
} // face_xmss_core_sign

void face_common_core_dp_keypair_fast(u8 *pk, u8 *sk, u32 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_pk = NULL, *dev_sk = NULL;
	int device = DEVICE_USED;
	int blocks = 1;
	cudaDeviceProp deviceProp;
	u32 threads = 32;
	int numBlocksPerSm;
	int malloc_size;
	int maxblocks, maxallthreads;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
#ifdef XMSSMT
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmssmt_core_dp_keypair_fast, threads, 0);
#else // ifdef XMSSMT
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmss_core_dp_keypair_fast, threads, 0);
#endif // ifdef XMSSMT
	maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
	maxallthreads = maxblocks * threads;
	if (num < maxallthreads) malloc_size = num / threads * threads + threads;
	else malloc_size = maxallthreads;

	CHECK(cudaMalloc((void **)&dev_pk, malloc_size * PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sk, malloc_size * SK_BYTES * sizeof(u8)));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
	u32 left = num;

	for (u32 iter = 0; iter < loop; iter++) {
	#if LARGE_SCHEME == 1
		u32 s;
		if (maxblocks * threads > left) {
			s = left;
			blocks = s / threads + (s % threads ? 1 : 0);
		} else {
			blocks = maxblocks;
			s = maxallthreads;
		}
	#else // if LARGE_SCHEME == 1
		int q = num / loop;
		int r = num % loop;
		int s = q + ((iter < r) ? 1 : 0);
		blocks = s / threads + (s % threads ? 1 : 0);
	#endif // if LARGE_SCHEME == 1
		printf("dp_keypair_fast: %u %u %u %u\n", maxblocks, blocks, threads, s);

		CHECK(cudaDeviceSynchronize());
#ifdef XMSSMT
		global_xmssmt_core_dp_keypair_fast << < blocks, threads >> >
			(dev_pk, dev_sk, s);
#else // ifdef XMSSMT
		global_xmss_core_dp_keypair_fast << < blocks, threads >> >
			(dev_pk, dev_sk, s);
#endif // ifdef XMSSMT
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(pk, dev_pk, s * PK_BYTES * sizeof(u8), DEVICE_2_HOST));
		CHECK(cudaMemcpy(sk, dev_sk, s * SK_BYTES * sizeof(u8), DEVICE_2_HOST));
		pk += s * PK_BYTES;
		sk += s * SK_BYTES;
		left -= s;
	}

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	cudaFree(dev_pk); cudaFree(dev_sk);
} // face_common_core_dp_keypair_fast

void face_xmssmt_core_ip_keypair_fast(u8 *pk, u8 *sk)
{
	struct timespec start, stop;
	double result;
	u8 *dev_pk = NULL, *dev_sk = NULL;
	int device = DEVICE_USED;
	int threads = 1, blocks = 1, maxblock = 1;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	if (TREE_HEIGHT == 5) {
		threads = 32;
		blocks = WOTS_LEN + 3;
	} else if (TREE_HEIGHT == 10) {
		threads = 32;  // 32 > 64
		int numBlocksPerSm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_ip_keypair_fast, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;
		blocks = 1;
		while (blocks <= maxblock / 2) {
			blocks *= 2;
		}
		int final_blocks = blocks + WOTS_LEN / threads + 1;
		if (final_blocks >= maxblock) printf("X wots || tree\n");
		blocks = maxblock;
	} else if (TREE_HEIGHT == 20) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_ip_keypair_fast, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;
		blocks = 1;
		while (blocks < maxblock / 2) {
			blocks *= 2;
		}
		blocks = maxblock;
	}
#ifdef PRINT_TIME
	printf("ip keygen %d %d %d\n", maxblock, threads, blocks);
#endif // ifdef PRINT_TIME

	void *kernelArgs[] = { &dev_pk, &dev_sk };

	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaDeviceSynchronize());
	cudaLaunchCooperativeKernel((void*)global_xmssmt_core_ip_keypair_fast,
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
} // face_xmssmt_core_ip_keypair

int gpu_xmss_keypair_fast(u8 *pk, u8 *sk)
{
	face_xmss_core_keypair_fast(pk, sk);

	return 0;
} // gpu_xmss_keypair_fast

int gpu_xmssmt_keypair_fast(u8 *pk, u8 *sk)
{
	face_xmssmt_core_keypair_fast(pk, sk);

	return 0;
} // gpu_xmssmt_keypair_fast

int gpu_xmss_dp_keypair_fast(u8 *pk, u8 *sk, u32 num)
{
	face_common_core_dp_keypair_fast(pk, sk, num);

	return 0;
} // gpu_xmss_dp_keypair

int gpu_xmssmt_dp_keypair_fast(u8 *pk, u8 *sk, u32 num)
{
	face_common_core_dp_keypair_fast(pk, sk, num);

	return 0;
} // gpu_xmssmt_dp_keypair

int gpu_xmss_ip_keypair_fast(u8 *pk, u8 *sk)
{
	face_xmss_core_ip_keypair_fast(pk, sk);

	return 0;
} // gpu_xmss_ip_keypair_fast

int gpu_xmssmt_ip_keypair_fast(u8 *pk, u8 *sk)
{
	face_xmssmt_core_ip_keypair_fast(pk, sk);

	return 0;
} // gpu_xmssmt_ip_keypair_fast

#endif // ifdef USING_BDS
