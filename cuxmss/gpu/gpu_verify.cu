#include <string.h>
#include <iostream>
using namespace std;

#include "gpu_verify.h"
#include "gpu_hash.h"
#include "gpu_utils.h"
#include "gpu_wots.h"
#include "gpu_xmss_commons.h"
#include "gpu_hash_address.h"

#include <cooperative_groups.h>

__device__ u8 one_wots_pk[WOTS_SIG_BYTES];
__device__ u8 v_root[N];

__device__ u8 all_m[16384 * XMSS_MLEN];

__device__ int dev_xmssmt_core_sign_open(u8 *m, u64 *mlen,
					 const u8 *sm, u64 smlen, const u8 *pk)
{
	const u8 *pub_root = pk;
	const u8 *pub_seed = pk + N;
	u8 wots_pk[WOTS_SIG_BYTES];
	u8 leaf[N];
	u8 root[N];
	u8 *mhash = root;
	u64 idx = 0;
	u32 i;
	u32 idx_leaf;

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	*mlen = smlen - SIG_BYTES;

	/* Convert the index bytes from the signature to an integer. */
	idx = dev_bytes_to_ull(sm, INDEX_BYTES);

	// /* Put the message all the way at the end of the m buffer, so that we can
	//  * prepend the required other inputs for the hash function. */
	// memcpy(m + SIG_BYTES, sm + SIG_BYTES, *mlen);
	//
	// /* Compute the message hash. */
	// dev_hash_message(mhash, sm + INDEX_BYTES, pk, idx,
	// 		 m + SIG_BYTES - PADDING_LEN - 3 * N, *mlen);
	dev_hash_message_modefied(mhash, sm + INDEX_BYTES, pk, idx,
				  sm + SIG_BYTES, *mlen);

	sm += INDEX_BYTES + N;

	/* For each subtree.. */
	for (i = 0; i < D; i++) {
		idx_leaf = (idx & ((1 << TREE_HEIGHT) - 1));
		idx = idx >> TREE_HEIGHT;

		dev_set_layer_addr(ots_addr, i);
		dev_set_layer_addr(ltree_addr, i);
		dev_set_layer_addr(node_addr, i);

		dev_set_tree_addr(ltree_addr, idx);
		dev_set_tree_addr(ots_addr, idx);
		dev_set_tree_addr(node_addr, idx);

		dev_set_ots_addr(ots_addr, idx_leaf);
		/* Initially, root = mhash, but on subsequent iterations it is the root
		 * of the subtree below the currently processed subtree. */
		dev_wots_pk_from_sig(wots_pk, sm, root, pub_seed, ots_addr);
		sm += WOTS_SIG_BYTES;

		/* Compute the leaf node using the WOTS public key. */
		dev_set_ltree_addr(ltree_addr, idx_leaf);
		dev_l_tree(leaf, wots_pk, pub_seed, ltree_addr);

		/* Compute the root node of this subtree. */
		dev_compute_root(root, leaf, &idx_leaf, sm, pub_seed, node_addr);
		sm += TREE_HEIGHT * N;
	}

	/* Check if the root node equals the root node in the public key. */
	int check_root = 1;

	for (i = 0; i < N; i++)
		if (root[i] != pub_root[i])
			check_root = 0;

	if (check_root == 0) { /* if wrong */
		memset(m, 0, *mlen);
		*mlen = 0;
		return -1;
	}

	/* If verification was successful, copy the message from the signature. */
	memcpy(m, sm, *mlen);

	return 0;
} // dev_xmssmt_core_sign_open

__global__ void global_xmssmt_core_sign_open(u8 *m, u64 *mlen,
					     const u8 *sm, u64 smlen, const u8 *pk, int *right)
{
	*right = dev_xmssmt_core_sign_open(m, mlen, sm, smlen, pk);

} // global_xmssmt_core_sign_open

__global__ void global_xmssmt_dp_core_sign_open(u8 *m, u64 *mlen, const u8 *sm,
						u64 smlen, const u8 *pk, int dp_num, int *right)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ret = 0;

	if (tid < dp_num) {
		ret = dev_xmssmt_core_sign_open(m + tid * XMSS_MLEN, mlen,
						sm + tid * SM_BYTES, smlen, pk + tid * PK_BYTES);
	}

	if (tid == 0) *right = ret;
} // global_xmssmt_dp_core_sign_open

__global__ void global_xmssmt_opk_core_sign_open(u8 *m, u64 *mlen,
						 const u8 *sm, u64 smlen, const u8 *pk, int opk_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	// int ret = 0;

	if (tid < opk_num) {
		dev_xmssmt_core_sign_open(m + tid * XMSS_MLEN, mlen,
					  sm + tid * SM_BYTES, smlen, pk);
	}

	// if (tid < opk_num) {
	// 	ret = dev_xmssmt_core_sign_open(m + tid * SM_BYTES, mlen,
	// 					sm + tid * SM_BYTES, smlen, pk);
	// }
	// g.sync();
	//
	// if (tid < opk_num) {
	// 	if (ret == 0)
	// 		memcpy(m + tid * XMSS_MLEN, sm + tid * SM_BYTES + SIG_BYTES, XMSS_MLEN);
	// 	else
	// 		memset(m + tid * XMSS_MLEN, 0, XMSS_MLEN);
	// }

} // global_xmssmt_opk_core_sign_open

__device__ int dev_xmssmt_core_ip_sign_open(u8 *m, u64 *mlen,
					    const u8 *sm, u64 smlen, const u8 *pk)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	const u8 *pub_root = pk;
	const u8 *pub_seed = pk + N;
	u8 leaf[N];
	u8 root[N];
	u8 *mhash = root;
	u64 idx = 0;
	u32 i;
	u32 idx_leaf;

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	if (tid == 0) *mlen = smlen - SIG_BYTES;

	/* Convert the index bytes from the signature to an integer. */
	idx = dev_bytes_to_ull(sm, INDEX_BYTES);

	/* Put the message all the way at the end of the m buffer, so that we can
	 * prepend the required other inputs for the hash function. */
	if (tid == 0) memcpy(m + SIG_BYTES, sm + SIG_BYTES, *mlen);
	g.sync();

	/* Compute the message hash. */
	dev_hash_message(mhash, sm + INDEX_BYTES, pk, idx,
			 m + SIG_BYTES - PADDING_LEN - 3 * N, *mlen);
	sm += INDEX_BYTES + N;

	/* For each subtree.. */
	for (i = 0; i < D; i++) {
		idx_leaf = (idx & ((1 << TREE_HEIGHT) - 1));
		idx = idx >> TREE_HEIGHT;

		dev_set_layer_addr(ots_addr, i);
		dev_set_layer_addr(ltree_addr, i);
		dev_set_layer_addr(node_addr, i);

		dev_set_tree_addr(ltree_addr, idx);
		dev_set_tree_addr(ots_addr, idx);
		dev_set_tree_addr(node_addr, idx);

		dev_set_ots_addr(ots_addr, idx_leaf);
		/* Initially, root = mhash, but on subsequent iterations it is the root
		 * of the subtree below the currently processed subtree. */
		if (tnum >= WOTS_LEN) {
			if (tid == 0) memcpy(v_root, root, N);
			dev_wots_pk_from_sig_parallel(one_wots_pk, sm, v_root,
						      pub_seed, ots_addr);
		} else {
			if (tid == 0) dev_wots_pk_from_sig(one_wots_pk, sm, v_root,
							   pub_seed, ots_addr);
		}

		sm += WOTS_SIG_BYTES;

		/* Compute the leaf node using the WOTS public key. */
		dev_set_ltree_addr(ltree_addr, idx_leaf);
		if (tnum >= 33) {
			dev_l_tree_parallel(leaf, one_wots_pk, pub_seed, ltree_addr);
		} else {
			if (tid == 0) dev_l_tree(leaf, one_wots_pk, pub_seed, ltree_addr);
		}

		/* Compute the root node of this subtree. */
		if (tid == 0) // cannot be parallelized
			dev_compute_root(root, leaf, &idx_leaf, sm, pub_seed, node_addr);
		sm += TREE_HEIGHT * N;
	}

	/* Check if the root node equals the root node in the public key. */
	if (tid == 0) {
		int check_root = 1;

		for (i = 0; i < N; i++)
			if (root[i] != pub_root[i])
				check_root = 0;

		//if wrong
		if (check_root == 0) {
			memset(m, 0, *mlen);
			*mlen = 0;
			return -1;
		}

		memcpy(m, sm, *mlen);
	}
	return 0;
} // global_xmssmt_core_ip_sign_open

__global__ void global_xmssmt_core_ip_sign_open(u8 *m, u64 *mlen,
						const u8 *sm, u64 smlen, const u8 *pk, int *right)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int ret;

	ret = dev_xmssmt_core_ip_sign_open(m, mlen, sm, smlen, pk);
	if (tid == 0) *right = ret;
} // global_xmssmt_core_ip_sign_open

//input: sm, smlen, pk
//output: m, mlen, right
int face_xmssmt_core_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
			       const u8 *pk)
{
	u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
	u64 *dev_mlen = NULL;
	int right, *dev_right = NULL;
	int device = DEVICE_USED;

	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_m, SM_BYTES * sizeof(u8)));

	CHECK(cudaMalloc((void **)&dev_sm, SM_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sm, sm, SM_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_pk, pk, PK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_mlen, 1 * sizeof(u64)));
	CHECK(cudaMalloc((void **)&dev_right, 1 * sizeof(int)));

	CHECK(cudaDeviceSynchronize());

	global_xmssmt_core_sign_open << < 1, 1 >> >
		(dev_m, dev_mlen, dev_sm, smlen, dev_pk, dev_right);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(m, dev_m, SM_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), DEVICE_2_HOST));
	CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), DEVICE_2_HOST));

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_pk); cudaFree(dev_mlen);
	cudaFree(dev_right);

	return right;
} // face_xmssmt_core_sign_open

//input: sm, smlen, num, pk
//output: m, mlen
int face_common_core_dp_sign_open(u8 *m, u64 *mlen,
				  const u8 *sm, u64 smlen, const u8 *pk, u32 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
	u64 *dev_mlen = NULL;
	int right, *dev_right = NULL;
	int device = DEVICE_USED;
	int blocks = 1, threads = 32;
	cudaDeviceProp deviceProp;
	int numBlocksPerSm;
	int malloc_size;
	int maxblocks, maxallthreads;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmssmt_dp_core_sign_open, threads, 0);
	maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
	maxallthreads = maxblocks * threads;
	if (num < maxallthreads) malloc_size = num / threads * threads + threads;
	else malloc_size = maxallthreads;

	CHECK(cudaMalloc((void **)&dev_m, malloc_size * XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sm, malloc_size * SM_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_pk, malloc_size * PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_mlen, 1 * sizeof(u64)));
	CHECK(cudaMalloc((void **)&dev_right, 1 * sizeof(int)));

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
	#ifdef PRINT_ALL
		printf("dp verify: %u %u %u\n", maxblocks, blocks, threads);
	#endif // ifdef PRINT_ALL

		void *Args[] =
		{ &dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &s, &dev_right };

		CHECK(cudaMemcpy(dev_pk, pk, s * PK_BYTES * sizeof(u8), HOST_2_DEVICE));
		CHECK(cudaMemcpy(dev_sm, sm, s * SM_BYTES * sizeof(u8), HOST_2_DEVICE));

		CHECK(cudaDeviceSynchronize());
		cudaLaunchCooperativeKernel( (void*)global_xmssmt_dp_core_sign_open,
					     blocks, threads, Args);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(m, dev_m, s * XMSS_MLEN * sizeof(u8), DEVICE_2_HOST));
		pk += s * PK_BYTES;
		sm += s * SM_BYTES;
		m += s * XMSS_MLEN;
		left -= s;
	}
	CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), DEVICE_2_HOST));
	CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_pk); cudaFree(dev_mlen);

	return right;
} // face_common_core_dp_sign_open

int face_common_core_msdp_sign_open(u8 *m, u64 *mlen,
				    const u8 *sm, u64 smlen, const u8 *pk, u32 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
	u64 *dev_mlen = NULL;
	int right, *dev_right = NULL;
	int device = DEVICE_USED;
	int blocks = 1, threads = 32;
	cudaDeviceProp deviceProp;
	int numBlocksPerSm;
	u32 malloc_size;
	int maxblocks, maxallthreads;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmssmt_dp_core_sign_open, threads, 0);
	maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
	maxallthreads = maxblocks * threads;
	malloc_size = num / threads * threads + threads;

	CHECK(cudaMalloc((void **)&dev_m, malloc_size * XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sm, malloc_size * SM_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_pk, malloc_size * PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_mlen, 1 * sizeof(u64)));
	CHECK(cudaMalloc((void **)&dev_right, 1 * sizeof(int)));

#if USING_STREAM == 1
	maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
	maxallthreads = 10496;
#else // ifdef USING_STREAM_1
	// Remain the same
#endif // ifdef USING_STREAM_1

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
	u32 left = num;

	cudaStream_t stream[loop];

	for (int i = 0; i < loop; ++i) {
		CHECK(cudaStreamCreate(&stream[i]));
	}

	// for free
	u8 *free_pk = dev_pk;
	u8 *free_m = dev_m;
	u8 *free_sm = dev_sm;

	CHECK(cudaDeviceSynchronize());
	for (u32 iter = 0; iter < loop; iter++) {
	#if LARGE_SCHEME == 1
		u32 s;
		if (maxallthreads > left) {
			s = left;
			blocks = s / threads + (s % threads ? 1 : 0);
		} else {
			blocks = maxallthreads / threads;
			s = maxallthreads;
		}
	#else // if LARGE_SCHEME == 1
		int q = num / loop;
		int r = num % loop;
		int s = q + ((iter < r) ? 1 : 0);
		blocks = s / threads + (s % threads ? 1 : 0);
	#endif // if LARGE_SCHEME == 1
	#ifdef PRINT_ALL
		printf("msdp verify: %u %u %u %u\n", maxblocks, blocks, threads, s);
	# endif // ifdef PRINT_ALL

		void * Args[] =
		{ &dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &s, &dev_right };

		CHECK(cudaMemcpyAsync(dev_pk, pk,
				      s * PK_BYTES * sizeof(u8), HOST_2_DEVICE, stream[iter]));
		CHECK(cudaMemcpyAsync(dev_sm, sm,
				      s * SM_BYTES * sizeof(u8), HOST_2_DEVICE, stream[iter]));

		cudaLaunchCooperativeKernel( (void*)global_xmssmt_dp_core_sign_open,
					     blocks, threads, Args, 0, stream[iter]);

		CHECK(cudaMemcpyAsync(m, dev_m,
				      s * XMSS_MLEN * sizeof(u8), DEVICE_2_HOST, stream[iter]));
		pk += s * PK_BYTES;
		sm += s * SM_BYTES;
		m += s * XMSS_MLEN;
		dev_pk += s * PK_BYTES;
		dev_sm += s * SM_BYTES;
		dev_m += s * XMSS_MLEN;
		left -= s;
	}
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), DEVICE_2_HOST));
	CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	cudaFree(free_m); cudaFree(free_sm); cudaFree(free_pk); cudaFree(dev_mlen);

	return right;
} // face_common_core_msdp_sign_open

int face_xmssmt_core_ip_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
				  const u8 *pk)
{
	struct timespec start, stop;
	double result;
	u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
	u64 *dev_mlen = NULL;
	int right, *dev_right = NULL;
	int device = DEVICE_USED;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	CHECK(cudaMalloc((void **)&dev_m, SM_BYTES * sizeof(u8)));

	CHECK(cudaMalloc((void **)&dev_sm, SM_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));

	CHECK(cudaMalloc((void **)&dev_mlen, 1 * sizeof(u64)));
	CHECK(cudaMalloc((void **)&dev_right, 1 * sizeof(int)));

	void *kernelArgs[] =
	{ &dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &dev_right };
	int threads = 16;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaMemcpy(dev_sm, sm, SM_BYTES * sizeof(u8), HOST_2_DEVICE));
	CHECK(cudaMemcpy(dev_pk, pk, PK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaDeviceSynchronize());
	cudaLaunchCooperativeKernel((void*)global_xmssmt_core_ip_sign_open,
				    deviceProp.multiProcessorCount, threads, kernelArgs);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(m, dev_m, XMSS_MLEN * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(mlen, dev_mlen, sizeof(u64), DEVICE_2_HOST));
	CHECK(cudaMemcpy(&right, dev_right, sizeof(int), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_pk); cudaFree(dev_mlen);
	cudaFree(dev_right);

	return right;
} // face_xmssmt_core_ip_sign_open

int face_xmssmt_core_opk_sign_open(u8 *m, u64 *mlen,
				   const u8 *sm, u64 smlen, const u8 *pk, u32 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
	u64 *dev_mlen = NULL;
	int right, *dev_right = NULL;
	int device = DEVICE_USED;
	int blocks = 1, threads = 32;
	cudaDeviceProp deviceProp;
	int numBlocksPerSm;
	int malloc_size;
	int maxblocks, maxallthreads;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmssmt_opk_core_sign_open, threads, 0);
	maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
	maxallthreads = maxblocks * threads;
	if (num < maxallthreads) malloc_size = num / threads * threads + threads;
	else malloc_size = maxallthreads;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
	u32 left = num;

	CHECK(cudaMalloc((void **)&dev_m, malloc_size * XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sm, malloc_size * SM_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_pk, PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_mlen, 1 * sizeof(u64)));
	CHECK(cudaMalloc((void **)&dev_right, 1 * sizeof(int)));

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
	#ifdef PRINT_ALL
		printf("opk verify: %u %u %u\n", maxblocks, blocks, threads);
	#endif // ifdef PRINT_ALL

		void *Args[] =
		{ &dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &s, &dev_right };

		CHECK(cudaMemcpy(dev_pk, pk, PK_BYTES * sizeof(u8), HOST_2_DEVICE));
		CHECK(cudaMemcpy(dev_sm, sm, s * SM_BYTES * sizeof(u8), HOST_2_DEVICE));

		CHECK(cudaDeviceSynchronize());
		cudaLaunchCooperativeKernel( (void*)global_xmssmt_opk_core_sign_open,
					     blocks, threads, Args);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(m, dev_m, s * XMSS_MLEN * sizeof(u8), DEVICE_2_HOST));
		pk += PK_BYTES;
		sm += s * SM_BYTES;
		m += s * XMSS_MLEN;
		left -= s;
	}
	CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), DEVICE_2_HOST));
	CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_pk); cudaFree(dev_mlen);

	return right;
} // face_xmssmt_core_opk_sign_open

int face_xmssmt_core_msopk_sign_open(u8 *m, u64 *mlen,
				     const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
	u64 *dev_mlen = NULL;
	int right, *dev_right = NULL;
	int device = DEVICE_USED;
	int blocks = 1, threads = 32;
	cudaDeviceProp deviceProp;
	int numBlocksPerSm;
	u32 malloc_size;
	int maxblocks, maxallthreads;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmssmt_opk_core_sign_open, threads, 0);
	maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
	maxallthreads = maxblocks * threads;
	malloc_size = num / threads * threads + threads;

#if USING_STREAM == 1
	maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
	maxallthreads = 10496;
#else // ifdef USING_STREAM_1
	// Remain the same
#endif // ifdef USING_STREAM_1

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
	u32 left = num;

	cudaStream_t stream[loop];

	for (int i = 0; i < loop; ++i) {
		CHECK(cudaStreamCreate(&stream[i]));
	}

	CHECK(cudaMalloc((void **)&dev_m, malloc_size * XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sm, malloc_size * SM_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_pk, loop * PK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_mlen, 1 * sizeof(u64)));
	CHECK(cudaMalloc((void **)&dev_right, 1 * sizeof(int)));

	// for free
	u8 *free_pk = dev_pk;
	u8 *free_m = dev_m;
	u8 *free_sm = dev_sm;

	CHECK(cudaDeviceSynchronize());
	for (u32 iter = 0; iter < loop; iter++) {
	#if LARGE_SCHEME == 1
		u32 s;
		if (maxallthreads > left) {
			s = left;
			blocks = s / threads + (s % threads ? 1 : 0);
		} else {
			blocks = maxallthreads / threads;
			s = maxallthreads;
		}
	#else // if LARGE_SCHEME == 1
		int q = num / loop;
		int r = num % loop;
		int s = q + ((iter < r) ? 1 : 0);
		blocks = s / threads + (s % threads ? 1 : 0);
	#endif // if LARGE_SCHEME == 1
	#ifdef PRINT_ALL
		printf("msopk verify: %u %u %u %u\n", maxblocks, blocks, threads, s);
	#endif // ifdef PRINT_ALL

		void *Args[] =
		{ &dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &s, &dev_right };

		CHECK(cudaMemcpyAsync(dev_pk, pk,
				      PK_BYTES * sizeof(u8), HOST_2_DEVICE, stream[iter]));
		CHECK(cudaMemcpyAsync(dev_sm, sm,
				      s * SM_BYTES * sizeof(u8), HOST_2_DEVICE, stream[iter]));

		cudaLaunchCooperativeKernel( (void*)global_xmssmt_opk_core_sign_open,
					     blocks, threads, Args, 0, stream[iter]);

		CHECK(cudaMemcpyAsync(m, dev_m,
				      s * XMSS_MLEN * sizeof(u8), DEVICE_2_HOST, stream[iter]));
		pk += PK_BYTES;
		sm += s * SM_BYTES;
		m += s * XMSS_MLEN;
		dev_pk += PK_BYTES;
		dev_sm += s * SM_BYTES;
		dev_m += s * XMSS_MLEN;
		left -= s;
	}
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), DEVICE_2_HOST));
	CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	cudaFree(free_m); cudaFree(free_sm); cudaFree(free_pk); cudaFree(dev_mlen);

	return right;
} // face_xmssmt_core_msopk_sign_open

int gpu_xmss_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
		       const u8 *pk)
{
	return face_xmssmt_core_sign_open(m, mlen, sm, smlen, pk);
} // gpu_xmss_sign_open

int gpu_xmssmt_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
			 const u8 *pk)
{
	return face_xmssmt_core_sign_open(m, mlen, sm, smlen, pk);
} // gpu_xmssmt_sign_open

int gpu_xmss_dp_sign_open(u8 *m, u64 *mlen,
			  const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	return face_common_core_dp_sign_open(m, mlen, sm, smlen, pk, num);
} // gpu_xmss_dp_sign_open

int gpu_xmssmt_dp_sign_open(u8 *m, u64 *mlen,
			    const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	return face_common_core_dp_sign_open(m, mlen, sm, smlen, pk, num);
} // gpu_xmssmt_dp_sign_open

int gpu_xmss_msdp_sign_open(u8 *m, u64 *mlen,
			    const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	return face_common_core_msdp_sign_open(m, mlen, sm, smlen, pk, num);
} // gpu_xmss_msdp_sign_open

int gpu_xmssmt_msdp_sign_open(u8 *m, u64 *mlen,
			      const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	return face_common_core_msdp_sign_open(m, mlen, sm, smlen, pk, num);
} // gpu_xmssmt_msdp_sign_open

int gpu_xmss_ip_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
			  const u8 *pk)
{
	return face_xmssmt_core_ip_sign_open(m, mlen, sm, smlen, pk);
} // gpu_xmss_ip_sign_open

int gpu_xmssmt_ip_sign_open(u8 *m, u64 *mlen, const u8 *sm, u64 smlen,
			    const u8 *pk)
{
	return face_xmssmt_core_ip_sign_open(m, mlen, sm, smlen, pk);
} // gpu_xmssmt_ip_sign_open

int gpu_xmss_opk_sign_open(u8 *m, u64 *mlen,
			   const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	return face_xmssmt_core_opk_sign_open(m, mlen, sm, smlen, pk, num);
} // gpu_xmss_dp_sign_open

int gpu_xmssmt_opk_sign_open(u8 *m, u64 *mlen,
			     const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	return face_xmssmt_core_opk_sign_open(m, mlen, sm, smlen, pk, num);
} // gpu_xmssmt_dp_sign_open

int gpu_xmss_msopk_sign_open(u8 *m, u64 *mlen,
			     const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	return face_xmssmt_core_msopk_sign_open(m, mlen, sm, smlen, pk, num);
} // gpu_xmss_dp_sign_open

int gpu_xmssmt_msopk_sign_open(u8 *m, u64 *mlen,
			       const u8 *sm, u64 smlen, const u8 *pk, u64 num)
{
	return face_xmssmt_core_msopk_sign_open(m, mlen, sm, smlen, pk, num);
} // gpu_xmssmt_dp_sign_open
