#ifdef USING_BDS

#include <string.h>
#include <iostream>
using namespace std;

#include "gpu_utils.h"
#include "gpu_wots.h"
#include "gpu_xmss_commons.h"
#include "gpu_hash.h"
#include "gpu_hash_address.h"

#include "gpu_xmss_core_fast.h"
#include "gpu_sign_fast.h"

#include <cooperative_groups.h>

__device__ u8 s_sm[WOTS_SIG_BYTES];
__device__ u8 s_root[N];

__device__ int dev_xmss_core_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
				       const u8 *m, u64 mlen)
{
	const u8 *pub_root = sk + INDEX_BYTES + 2 * N;

	uint16_t i = 0;

	// TODO refactor BDS state not to need separate treehash instances
	bds_state state;
	treehash_inst treehash[TREE_HEIGHT - BDS_K];

	state.treehash = treehash;

	/* Load the BDS state from sk. */
	dev_xmss_deserialize_state(&state, sk);

	// Extract SK
	unsigned long idx = ((unsigned long)sk[0] << 24)
			    | ((unsigned long)sk[1] << 16)
			    | ((unsigned long)sk[2] << 8)
			    | sk[3];

	/* Check if we can still sign with this sk.
	 * If not, return -2
	 *
	 * If this is the last possible signature (because the max index value
	 * is reached), production implementations should delete the secret key
	 * to prevent accidental further use.
	 *
	 * For the case of total tree height of 64 we do not use the last signature
	 * to be on the safe side (there is no index value left to indicate that the
	 * key is finished, hence external handling would be necessary)
	 */
	if (idx >= ((1ULL << FULL_HEIGHT) - 1)) {
		// Delete secret key here. We only do this in memory, production code
		// has to make sure that this happens on disk.
		memset(sk, 0xFF, INDEX_BYTES);
		memset(sk + INDEX_BYTES, 0, (SK_BYTES - INDEX_BYTES));
		if (idx > ((1ULL << FULL_HEIGHT) - 1))
			return -2;      // We already used all one-time keys
		if ((FULL_HEIGHT == 64) && (idx == ((1ULL << FULL_HEIGHT) - 1)))
			return -2;      // We already used all one-time keys
	}

	u8 sk_seed[N];

	memcpy(sk_seed, sk + INDEX_BYTES, N);
	u8 sk_prf[N];

	memcpy(sk_prf, sk + INDEX_BYTES + N, N);
	u8 pub_seed[N];

	memcpy(pub_seed, sk + INDEX_BYTES + 3 * N, N);

	// index as 32 bytes string
	u8 idx_bytes_32[32];

	dev_ull_to_bytes(idx_bytes_32, 32, idx);

	// Update SK
	sk[0] = ((idx + 1) >> 24) & 255;
	sk[1] = ((idx + 1) >> 16) & 255;
	sk[2] = ((idx + 1) >> 8) & 255;
	sk[3] = (idx + 1) & 255;
	// Secret key for this non-forward-secure version is now updated.
	// A production implementation should consider using a file handle instead,
	//  and write the updated secret key at this point!

	// Init working params
	u8 R[N];
	u8 msg_h[N];
	uint32_t ots_addr[8] = { 0 };

	// ---------------------------------
	// Message Hashing
	// ---------------------------------

	// Message Hash:
	// First compute pseudorandom value
	dev_prf(R, idx_bytes_32, sk_prf);

	/* Already put the message in the right place, to make it easier to prepend
	 * things when computing the hash over the message. */
	memcpy(sm + SIG_BYTES, m, mlen);

	/* Compute the message hash. */
	dev_hash_message(msg_h, R, pub_root, idx,
			 sm + SIG_BYTES - PADDING_LEN - 3 * N, mlen);

	// Start collecting signature
	*smlen = 0;

	// Copy index to signature
	sm[0] = (idx >> 24) & 255;
	sm[1] = (idx >> 16) & 255;
	sm[2] = (idx >> 8) & 255;
	sm[3] = idx & 255;

	sm += 4;
	*smlen += 4;

	// Copy R to signature
	for (i = 0; i < N; i++) {
		sm[i] = R[i];
	}

	sm += N;
	*smlen += N;

	// ----------------------------------
	// Now we start to "really sign"
	// ----------------------------------

	// Prepare Address
	dev_set_type(ots_addr, 0);
	dev_set_ots_addr(ots_addr, idx);

	// Compute WOTS signature
	dev_wots_sign(sm, msg_h, sk_seed, pub_seed, ots_addr);

	sm += WOTS_SIG_BYTES;
	*smlen += WOTS_SIG_BYTES;

	// the auth path was already computed during the previous round
	memcpy(sm, state.auth, TREE_HEIGHT * N);

	if (idx < (1U << TREE_HEIGHT) - 1) {
		dev_bds_round(&state, idx, sk_seed, pub_seed, ots_addr);
		dev_bds_treehash_update(&state, (TREE_HEIGHT - BDS_K) >> 1, sk_seed, pub_seed, ots_addr);
	}

	sm += TREE_HEIGHT * N;
	*smlen += TREE_HEIGHT * N;

	memcpy(sm, m, mlen);
	*smlen += mlen;

	/* Write the updated BDS state back into sk. */
	dev_xmss_serialize_state(sk, &state);

	return 0;
} // dev_xmss_core_sign_fast

__global__ void global_xmss_core_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
					   const u8 *m, u64 mlen)
{
	dev_xmss_core_sign_fast(sk, sm, smlen, m, mlen);
} // global_xmss_core_sign_fast

__global__ void global_xmss_core_dp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
					      const u8 *m, u64 mlen, u32 dp_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < dp_num) {
		dev_xmss_core_sign_fast(sk + tid * SK_BYTES,
					sm + tid * SM_BYTES, smlen, m + tid * XMSS_MLEN, mlen);
	}
} // global_xmss_dp_core_sign

__device__ bds_state one_state;
__device__ treehash_inst one_treehash[TREE_HEIGHT - BDS_K];

__device__ int dev_xmss_core_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
					  const u8 *m, u64 mlen)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	const u8 *pub_root = sk + INDEX_BYTES + 2 * N;

	// TODO refactor BDS state not to need separate treehash instances

	one_state.treehash = one_treehash;

	/* Load the BDS state from sk. */
	if (tid == 0) dev_xmss_deserialize_state(&one_state, sk);

	// Extract SK
	unsigned long idx = ((unsigned long)sk[0] << 24)
			    | ((unsigned long)sk[1] << 16)
			    | ((unsigned long)sk[2] << 8)
			    | sk[3];

	/* Check if we can still sign with this sk.
	 * If not, return -2
	 *
	 * If this is the last possible signature (because the max index value
	 * is reached), production implementations should delete the secret key
	 * to prevent accidental further use.
	 *
	 * For the case of total tree height of 64 we do not use the last signature
	 * to be on the safe side (there is no index value left to indicate that the
	 * key is finished, hence external handling would be necessary)
	 */
	if (idx >= ((1ULL << FULL_HEIGHT) - 1)) {
		// Delete secret key here. We only do this in memory, production code
		// has to make sure that this happens on disk.
		if (tid == 0) memset(sk, 0xFF, INDEX_BYTES);
		if (tid == 0) memset(sk + INDEX_BYTES, 0, (SK_BYTES - INDEX_BYTES));
		if (idx > ((1ULL << FULL_HEIGHT) - 1))
			return -2;      // We already used all one-time keys
		if ((FULL_HEIGHT == 64) && (idx == ((1ULL << FULL_HEIGHT) - 1)))
			return -2;      // We already used all one-time keys
	}

	u8 sk_seed[N];

	memcpy(sk_seed, sk + INDEX_BYTES, N);
	u8 sk_prf[N];

	memcpy(sk_prf, sk + INDEX_BYTES + N, N);
	u8 pub_seed[N];

	memcpy(pub_seed, sk + INDEX_BYTES + 3 * N, N);

	// index as 32 bytes string
	u8 idx_bytes_32[32];

	dev_ull_to_bytes(idx_bytes_32, 32, idx);

	// Update SK
	if (tid == 0) sk[0] = ((idx + 1) >> 24) & 255;
	if (tid == 0) sk[1] = ((idx + 1) >> 16) & 255;
	if (tid == 0) sk[2] = ((idx + 1) >> 8) & 255;
	if (tid == 0) sk[3] = (idx + 1) & 255;
	// Secret key for this non-forward-secure version is now updated.
	// A production implementation should consider using a file handle instead,
	//  and write the updated secret key at this point!

	// Init working params
	u8 R[N];
	u8 msg_h[N];
	uint32_t ots_addr[8] = { 0 };

	// ---------------------------------
	// Message Hashing
	// ---------------------------------

	// Message Hash:
	// First compute pseudorandom value
	dev_prf(R, idx_bytes_32, sk_prf);

	/* Already put the message in the right place, to make it easier to prepend
	 * things when computing the hash over the message. */
	if (tid == 0) memcpy(sm + SIG_BYTES, m, mlen);
	g.sync();

	/* Compute the message hash. */
	dev_hash_message(msg_h, R, pub_root, idx,
			 sm + SIG_BYTES - PADDING_LEN - 3 * N, mlen);

	// Start collecting signature
	if (tid == 0) *smlen = 0;

	// Copy index to signature
	if (tid == 0) sm[0] = (idx >> 24) & 255;
	if (tid == 0) sm[1] = (idx >> 16) & 255;
	if (tid == 0) sm[2] = (idx >> 8) & 255;
	if (tid == 0) sm[3] = idx & 255;

	sm += 4;
	if (tid == 0) *smlen += 4;

	// Copy R to signature
	if (tid == 0) memcpy(sm, R, N);

	sm += N;
	if (tid == 0) *smlen += N;

	// ----------------------------------
	// Now we start to "really sign"
	// ----------------------------------

	// Prepare Address
	dev_set_type(ots_addr, 0);
	dev_set_ots_addr(ots_addr, idx);

	// Compute WOTS signature
	if (tnum >= 67) {
		dev_wots_sign_parallel(s_sm, msg_h, sk_seed, pub_seed, ots_addr, 0);
		g.sync();
		if (tid == 0)
			memcpy(sm, s_sm, WOTS_SIG_BYTES);
	} else {
		if (tid == 0) dev_wots_sign(sm, msg_h, sk_seed, pub_seed, ots_addr);
	}
	g.sync();

	sm += WOTS_SIG_BYTES;
	if (tid == 0) *smlen += WOTS_SIG_BYTES;

	// the auth path was already computed during the previous round
	if (tid == 0) memcpy(sm, one_state.auth, TREE_HEIGHT * N);

	g.sync();

	if (idx < (1U << TREE_HEIGHT) - 1) {
		dev_bds_round_parallel(&one_state, idx, sk_seed, pub_seed, ots_addr);
		g.sync();
		dev_bds_treehash_update_parallel(&one_state, (TREE_HEIGHT - BDS_K) >> 1,
						 sk_seed, pub_seed, ots_addr);
	}
	if (tid != 0) return 0;

	sm += TREE_HEIGHT * N;
	*smlen += TREE_HEIGHT * N;

	memcpy(sm, m, mlen);
	*smlen += mlen;

	/* Write the updated BDS state back into sk. */
	dev_xmss_serialize_state(sk, &one_state);

	return 0;
} // dev_xmss_core_ip_sign_fast

__global__ void global_xmss_core_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
					      const u8 *m, u64 mlen)
{
	dev_xmss_core_ip_sign_fast(sk, sm, smlen, m, mlen);
} // global_xmss_core_ip_sign_fast

//input: sk, m, mlen
//output: sk, sm, smlen
int face_xmss_core_sign_fast(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	int device = DEVICE_USED;

	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sk, sk, SK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_m, XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMemcpy(dev_m, m, XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_sm, (SIG_BYTES + XMSS_MLEN) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	CHECK(cudaDeviceSynchronize());

	global_xmss_core_sign_fast << < 1, 1 >> >
		(dev_sk, dev_sm, dev_smlen, dev_m, mlen);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm,
			 (SIG_BYTES + XMSS_MLEN) * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), DEVICE_2_HOST));

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_sk); cudaFree(dev_smlen);

	return 0;
} // face_xmss_core_sign_fast



//input: sk, m, mlen
//output: sk, sm, smlen
int face_xmss_core_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	struct timespec start, stop;
	double result;
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	int device = DEVICE_USED;
	int threads = 1;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sk, sk, SK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_m, XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMemcpy(dev_m, m, XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_sm, (SIG_BYTES + XMSS_MLEN) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	threads = 32;
	void *kernelArgs[] = { &dev_sk, &dev_sm, &dev_smlen, &dev_m, &mlen };

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaDeviceSynchronize());
	cudaLaunchCooperativeKernel((void*)global_xmss_core_ip_sign_fast,
				    3, threads, kernelArgs);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;
	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm,
			 (SIG_BYTES + XMSS_MLEN) * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), DEVICE_2_HOST));

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_sk); cudaFree(dev_smlen);

	return 0;
} // face_xmss_core_sign_fast

__device__ int dev_xmssmt_core_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
					 const u8 *m, u64 mlen)
{
	const u8 *pub_root = sk + INDEX_BYTES + 2 * N;

	uint64_t idx_tree;
	uint32_t idx_leaf;
	uint64_t i, j;
	int needswap_upto = -1;
	unsigned int updates;

	u8 sk_seed[N];
	u8 sk_prf[N];
	u8 pub_seed[N];
	// Init working params
	u8 R[N];
	u8 msg_h[N];
	uint32_t addr[8] = { 0 };
	uint32_t ots_addr[8] = { 0 };
	u8 idx_bytes_32[32];

	u8 *wots_sigs;

	// TODO refactor BDS state not to need separate treehash instances
	bds_state states[2 * D - 1];
	treehash_inst treehash[(2 * D - 1) * (TREE_HEIGHT - BDS_K)];

	for (i = 0; i < 2 * D - 1; i++) {
		states[i].treehash = treehash + i * (TREE_HEIGHT - BDS_K);
	}

	dev_xmssmt_deserialize_state(states, &wots_sigs, sk);

	// Extract SK
	u64 idx = 0;

	for (i = 0; i < INDEX_BYTES; i++) {
		idx |= ((u64)sk[i]) << 8 * (INDEX_BYTES - 1 - i);
	}

	/* Check if we can still sign with this sk.
	 * If not, return -2
	 *
	 * If this is the last possible signature (because the max index value
	 * is reached), production implementations should delete the secret key
	 * to prevent accidental further use.
	 *
	 * For the case of total tree height of 64 we do not use the last signature
	 * to be on the safe side (there is no index value left to indicate that the
	 * key is finished, hence external handling would be necessary)
	 */
	if (idx >= ((1ULL << FULL_HEIGHT) - 1)) {
		// Delete secret key here. We only do this in memory, production code
		// has to make sure that this happens on disk.
		memset(sk, 0xFF, INDEX_BYTES);
		memset(sk + INDEX_BYTES, 0, (SK_BYTES - INDEX_BYTES));
		if (idx > ((1ULL << FULL_HEIGHT) - 1))
			return -2;      // We already used all one-time keys
		if ((FULL_HEIGHT == 64) && (idx == ((1ULL << FULL_HEIGHT) - 1)))
			return -2;      // We already used all one-time keys
	}

	memcpy(sk_seed, sk + INDEX_BYTES, N);
	memcpy(sk_prf, sk + INDEX_BYTES + N, N);
	memcpy(pub_seed, sk + INDEX_BYTES + 3 * N, N);

	// Update SK
	for (i = 0; i < INDEX_BYTES; i++) {
		sk[i] = ((idx + 1) >> 8 * (INDEX_BYTES - 1 - i)) & 255;
	}
	// Secret key for this non-forward-secure version is now updated.
	// A production implementation should consider using a file handle instead,
	//  and write the updated secret key at this point!

	// ---------------------------------
	// Message Hashing
	// ---------------------------------

	// Message Hash:
	// First compute pseudorandom value
	dev_ull_to_bytes(idx_bytes_32, 32, idx);
	dev_prf(R, idx_bytes_32, sk_prf);

	/* Already put the message in the right place, to make it easier to prepend
	 * things when computing the hash over the message. */
	memcpy(sm + SIG_BYTES, m, mlen);

	/* Compute the message hash. */
	dev_hash_message(msg_h, R, pub_root, idx,
			 sm + SIG_BYTES - PADDING_LEN - 3 * N,
			 mlen);

	// Start collecting signature
	*smlen = 0;

	// Copy index to signature
	for (i = 0; i < INDEX_BYTES; i++) {
		sm[i] = (idx >> 8 * (INDEX_BYTES - 1 - i)) & 255;
	}

	sm += INDEX_BYTES;
	*smlen += INDEX_BYTES;

	// Copy R to signature
	for (i = 0; i < N; i++) {
		sm[i] = R[i];
	}

	sm += N;
	*smlen += N;

	// ----------------------------------
	// Now we start to "really sign"
	// ----------------------------------

	// Handle lowest layer separately as it is slightly different...

	// Prepare Address
	dev_set_type(ots_addr, 0);
	idx_tree = idx >> TREE_HEIGHT;
	idx_leaf = (idx & ((1 << TREE_HEIGHT) - 1));
	dev_set_layer_addr(ots_addr, 0);
	dev_set_tree_addr(ots_addr, idx_tree);
	dev_set_ots_addr(ots_addr, idx_leaf);

	// Compute WOTS signature
	dev_wots_sign(sm, msg_h, sk_seed, pub_seed, ots_addr);

	sm += WOTS_SIG_BYTES;
	*smlen += WOTS_SIG_BYTES;

	memcpy(sm, states[0].auth, TREE_HEIGHT * N);
	sm += TREE_HEIGHT * N;
	*smlen += TREE_HEIGHT * N;

	// prepare signature of remaining layers
	for (i = 1; i < D; i++) {
		// put WOTS signature in place
		memcpy(sm, wots_sigs + (i - 1) * WOTS_SIG_BYTES, WOTS_SIG_BYTES);

		sm += WOTS_SIG_BYTES;
		*smlen += WOTS_SIG_BYTES;

		// put AUTH nodes in place
		memcpy(sm, states[i].auth, TREE_HEIGHT * N);
		sm += TREE_HEIGHT * N;
		*smlen += TREE_HEIGHT * N;
	}

	updates = (TREE_HEIGHT - BDS_K) >> 1;

	dev_set_tree_addr(addr, (idx_tree + 1));
	// mandatory update for NEXT_0 (does not count towards h-k/2) if NEXT_0 exists
	if ((1 + idx_tree) * (1 << TREE_HEIGHT) + idx_leaf < (1ULL << FULL_HEIGHT)) {
		dev_bds_state_update(&states[D], sk_seed, pub_seed, addr);
	}

	for (i = 0; i < D; i++) {
		// check if we're not at the end of a tree
		if (!(((idx + 1) & ((1ULL << ((i + 1) * TREE_HEIGHT)) - 1)) == 0)) {
			idx_leaf = (idx >> (TREE_HEIGHT * i)) & ((1 << TREE_HEIGHT) - 1);
			idx_tree = (idx >> (TREE_HEIGHT * (i + 1)));
			dev_set_layer_addr(addr, i);
			dev_set_tree_addr(addr, idx_tree);
			if (i == (unsigned int)(needswap_upto + 1)) {
				dev_bds_round(&states[i], idx_leaf, sk_seed, pub_seed, addr);
			}
			updates = dev_bds_treehash_update(&states[i], updates, sk_seed, pub_seed, addr);
			dev_set_tree_addr(addr, (idx_tree + 1));
			// if a NEXT-tree exists for this level;
			if ((1 + idx_tree) * (1 << TREE_HEIGHT) + idx_leaf < (1ULL << (FULL_HEIGHT - TREE_HEIGHT * i))) {
				if (i > 0 && updates > 0 && states[D + i].next_leaf < (1ULL << FULL_HEIGHT)) {
					dev_bds_state_update(&states[D + i], sk_seed, pub_seed, addr);
					updates--;
				}
			}
		} else if (idx < (1ULL << FULL_HEIGHT) - 1) {
			dev_deep_state_swap(states + D + i, states + i);

			dev_set_layer_addr(ots_addr, (i + 1));
			dev_set_tree_addr(ots_addr, ((idx + 1) >> ((i + 2) * TREE_HEIGHT)));
			dev_set_ots_addr(ots_addr, (((idx >> ((i + 1) * TREE_HEIGHT)) + 1) & ((1 << TREE_HEIGHT) - 1)));

			dev_wots_sign(wots_sigs + i * WOTS_SIG_BYTES, states[i].stack, sk_seed, pub_seed, ots_addr);

			states[D + i].stackoffset = 0;
			states[D + i].next_leaf = 0;

			updates--; // WOTS-signing counts as one update
			needswap_upto = i;
			for (j = 0; j < TREE_HEIGHT - BDS_K; j++) {
				states[i].treehash[j].completed = 1;
			}
		}
	}

	memcpy(sm, m, mlen);
	*smlen += mlen;

	dev_xmssmt_serialize_state(sk, states);

	return 0;
} // dev_xmssmt_core_sign_fast

__global__ void global_xmssmt_core_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
					     const u8 *m, u64 mlen)
{
	dev_xmssmt_core_sign_fast(sk, sm, smlen, m, mlen);
} // global_xmssmt_core_sign_fast

//input: sk, m, mlen
//output: sk, sm, smlen
void face_xmssmt_core_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
				const u8 *m, u64 mlen)
{
	int device = DEVICE_USED;

	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;

	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sk, sk, SK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_m, XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMemcpy(dev_m, m, XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_sm, (SIG_BYTES + XMSS_MLEN) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	CHECK(cudaDeviceSynchronize());
	global_xmssmt_core_sign_fast << < 1, 1 >> >
		(dev_sk, dev_sm, dev_smlen, dev_m, mlen);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm, (SIG_BYTES + XMSS_MLEN) * sizeof(u8),
			 DEVICE_2_HOST));
	CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), DEVICE_2_HOST));

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_sk); cudaFree(dev_smlen);
} // face_xmssmt_core_sign_fast

__device__ void dev_xmssmt_core_dp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
					     const u8 *m, u64 mlen)
{
	const u8 *pub_root = sk + INDEX_BYTES + 2 * N;

	uint64_t idx_tree;
	uint32_t idx_leaf;
	uint64_t i, j;
	int needswap_upto = -1;
	unsigned int updates;

	u8 sk_seed[N];
	u8 sk_prf[N];
	u8 pub_seed[N];
	// Init working params
	u8 R[N];
	u8 msg_h[N];
	uint32_t addr[8] = { 0 };
	uint32_t ots_addr[8] = { 0 };
	u8 idx_bytes_32[32];

	u8 *wots_sigs;

	// TODO refactor BDS state not to need separate treehash instances
	bds_state states[2 * D - 1];
	treehash_inst treehash[(2 * D - 1) * (TREE_HEIGHT - BDS_K)];

	for (i = 0; i < 2 * D - 1; i++) {
		states[i].treehash = treehash + i * (TREE_HEIGHT - BDS_K);
	}

	dev_xmssmt_deserialize_state(states, &wots_sigs, sk);

	// Extract SK
	u64 idx = 0;

	for (i = 0; i < INDEX_BYTES; i++) {
		idx |= ((u64)sk[i]) << 8 * (INDEX_BYTES - 1 - i);
	}

	/* Check if we can still sign with this sk.
	 * If not, return -2
	 *
	 * If this is the last possible signature (because the max index value
	 * is reached), production implementations should delete the secret key
	 * to prevent accidental further use.
	 *
	 * For the case of total tree height of 64 we do not use the last signature
	 * to be on the safe side (there is no index value left to indicate that the
	 * key is finished, hence external handling would be necessary)
	 */
	if (idx >= ((1ULL << FULL_HEIGHT) - 1)) {
		// Delete secret key here. We only do this in memory, production code
		// has to make sure that this happens on disk.
		memset(sk, 0xFF, INDEX_BYTES);
		memset(sk + INDEX_BYTES, 0, (SK_BYTES - INDEX_BYTES));
		if (idx > ((1ULL << FULL_HEIGHT) - 1))
			return;      // We already used all one-time keys
		// return -2;      // We already used all one-time keys
		if ((FULL_HEIGHT == 64) && (idx == ((1ULL << FULL_HEIGHT) - 1)))
			return;      // We already used all one-time keys
		// return -2;      // We already used all one-time keys
	}

	memcpy(sk_seed, sk + INDEX_BYTES, N);
	memcpy(sk_prf, sk + INDEX_BYTES + N, N);
	memcpy(pub_seed, sk + INDEX_BYTES + 3 * N, N);

	// Update SK
	for (i = 0; i < INDEX_BYTES; i++) {
		sk[i] = ((idx + 1) >> 8 * (INDEX_BYTES - 1 - i)) & 255;
	}
	// Secret key for this non-forward-secure version is now updated.
	// A production implementation should consider using a file handle instead,
	//  and write the updated secret key at this point!

	// ---------------------------------
	// Message Hashing
	// ---------------------------------

	// Message Hash:
	// First compute pseudorandom value
	dev_ull_to_bytes(idx_bytes_32, 32, idx);
	dev_prf(R, idx_bytes_32, sk_prf);

	/* Already put the message in the right place, to make it easier to prepend
	 * things when computing the hash over the message. */
	memcpy(sm + SIG_BYTES, m, mlen);

	/* Compute the message hash. */
	dev_hash_message(msg_h, R, pub_root, idx,
			 sm + SIG_BYTES - PADDING_LEN - 3 * N,
			 mlen);

	// Start collecting signature
	*smlen = 0;

	// Copy index to signature
	for (i = 0; i < INDEX_BYTES; i++) {
		sm[i] = (idx >> 8 * (INDEX_BYTES - 1 - i)) & 255;
	}

	sm += INDEX_BYTES;
	*smlen += INDEX_BYTES;

	// Copy R to signature
	for (i = 0; i < N; i++) {
		sm[i] = R[i];
	}

	sm += N;
	*smlen += N;

	// ----------------------------------
	// Now we start to "really sign"
	// ----------------------------------

	// Handle lowest layer separately as it is slightly different...

	// Prepare Address
	dev_set_type(ots_addr, 0);
	idx_tree = idx >> TREE_HEIGHT;
	idx_leaf = (idx & ((1 << TREE_HEIGHT) - 1));
	dev_set_layer_addr(ots_addr, 0);
	dev_set_tree_addr(ots_addr, idx_tree);
	dev_set_ots_addr(ots_addr, idx_leaf);

	// Compute WOTS signature
	dev_wots_sign(sm, msg_h, sk_seed, pub_seed, ots_addr);

	sm += WOTS_SIG_BYTES;
	*smlen += WOTS_SIG_BYTES;

	memcpy(sm, states[0].auth, TREE_HEIGHT * N);
	sm += TREE_HEIGHT * N;
	*smlen += TREE_HEIGHT * N;

	// prepare signature of remaining layers
	for (i = 1; i < D; i++) {
		// put WOTS signature in place
		memcpy(sm, wots_sigs + (i - 1) * WOTS_SIG_BYTES, WOTS_SIG_BYTES);

		sm += WOTS_SIG_BYTES;
		*smlen += WOTS_SIG_BYTES;

		// put AUTH nodes in place
		memcpy(sm, states[i].auth, TREE_HEIGHT * N);
		sm += TREE_HEIGHT * N;
		*smlen += TREE_HEIGHT * N;
	}

	updates = (TREE_HEIGHT - BDS_K) >> 1;

	dev_set_tree_addr(addr, (idx_tree + 1));
	// mandatory update for NEXT_0 (does not count towards h-k/2) if NEXT_0 exists
	if ((1 + idx_tree) * (1 << TREE_HEIGHT) + idx_leaf < (1ULL << FULL_HEIGHT)) {
		dev_bds_state_update(&states[D], sk_seed, pub_seed, addr);
	}

	for (i = 0; i < D; i++) {
		// check if we're not at the end of a tree
		if (!(((idx + 1) & ((1ULL << ((i + 1) * TREE_HEIGHT)) - 1)) == 0)) {
			idx_leaf = (idx >> (TREE_HEIGHT * i)) & ((1 << TREE_HEIGHT) - 1);
			idx_tree = (idx >> (TREE_HEIGHT * (i + 1)));
			dev_set_layer_addr(addr, i);
			dev_set_tree_addr(addr, idx_tree);
			if (i == (unsigned int)(needswap_upto + 1)) {
				dev_bds_round(&states[i], idx_leaf, sk_seed, pub_seed, addr);
			}
			updates = dev_bds_treehash_update(&states[i], updates, sk_seed, pub_seed, addr);
			dev_set_tree_addr(addr, (idx_tree + 1));
			// if a NEXT-tree exists for this level;
			if ((1 + idx_tree) * (1 << TREE_HEIGHT) + idx_leaf < (1ULL << (FULL_HEIGHT - TREE_HEIGHT * i))) {
				if (i > 0 && updates > 0 && states[D + i].next_leaf < (1ULL << FULL_HEIGHT)) {
					dev_bds_state_update(&states[D + i], sk_seed, pub_seed, addr);
					updates--;
				}
			}
		}else if (idx < (1ULL << FULL_HEIGHT) - 1) {
			dev_deep_state_swap(states + D + i, states + i);

			dev_set_layer_addr(ots_addr, (i + 1));
			dev_set_tree_addr(ots_addr, ((idx + 1) >> ((i + 2) * TREE_HEIGHT)));
			dev_set_ots_addr(ots_addr, (((idx >> ((i + 1) * TREE_HEIGHT)) + 1) & ((1 << TREE_HEIGHT) - 1)));

			dev_wots_sign(wots_sigs + i * WOTS_SIG_BYTES, states[i].stack, sk_seed, pub_seed, ots_addr);

			states[D + i].stackoffset = 0;
			states[D + i].next_leaf = 0;

			updates--; // WOTS-signing counts as one update
			needswap_upto = i;
			for (j = 0; j < TREE_HEIGHT - BDS_K; j++) {
				states[i].treehash[j].completed = 1;
			}
		}
	}

	memcpy(sm, m, mlen);
	*smlen += mlen;

	dev_xmssmt_serialize_state(sk, states);
} // dev_xmssmt_core_dp_sign

__global__ void global_xmssmt_core_dp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
						const u8 *m, u64 mlen, u32 dp_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < dp_num) {
		dev_xmssmt_core_sign_fast(sk + tid * SK_BYTES,
					  sm + tid * SM_BYTES, smlen, m + tid * XMSS_MLEN, mlen);
	}
} // global_xmssmt_core_dp_sign

void face_common_core_dp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
				   const u8 *m, u64 mlen, u64 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	int device = DEVICE_USED;
	int blocks = 1;
	cudaDeviceProp deviceProp;
	int threads = 32;
	int numBlocksPerSm;
	int malloc_size;
	int maxblocks, maxallthreads;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
#ifdef XMSSMT
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmssmt_core_dp_sign_fast, threads, 0);
#else // ifdef XMSSMT
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmss_core_dp_sign_fast, threads, 0);
#endif // ifdef XMSSMT
	maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
	maxallthreads = maxblocks * threads;
	if (num < maxallthreads) malloc_size = num / threads * threads + threads;
	else malloc_size = maxallthreads;

	CHECK(cudaMalloc((void **)&dev_sk, malloc_size * SK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_m, malloc_size * XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sm, malloc_size * SM_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

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
		printf("dp_sign_fast: %u %u %u %u\n", maxblocks, blocks, threads, s);

		CHECK(cudaMemcpy(dev_sk, sk, s * SK_BYTES * sizeof(u8), HOST_2_DEVICE));
		CHECK(cudaMemcpy(dev_m, m, s * XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

		void *Args[] =
		{ &dev_sk, &dev_sm, &dev_smlen, &dev_m, &mlen, &s };

		CHECK(cudaDeviceSynchronize());
		#ifdef XMSSMT
		cudaLaunchCooperativeKernel( (void *)global_xmssmt_core_dp_sign_fast,
					     blocks, threads, Args);
		#else // ifdef XMSSMT
		cudaLaunchCooperativeKernel( (void *)global_xmss_core_dp_sign_fast,
					     blocks, threads, Args);
		#endif // ifdef XMSSMT
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(sk, dev_sk, s * SK_BYTES * sizeof(u8), DEVICE_2_HOST));
		CHECK(cudaMemcpy(sm, dev_sm, s * SM_BYTES * sizeof(u8), DEVICE_2_HOST));
		sk += s * SK_BYTES;
		m += s * XMSS_MLEN;
		sm += s * SM_BYTES;
		left -= s;
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	*smlen = SM_BYTES;

	cudaFree(dev_sk); cudaFree(dev_sm); cudaFree(dev_smlen); cudaFree(dev_m);
} // face_xmss_core_dp_sign_fast

void face_common_core_msdp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
				     const u8 *m, u64 mlen, u64 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	int device = DEVICE_USED;
	int blocks = 1, threads = 32;
	cudaDeviceProp deviceProp;
	int numBlocksPerSm;
	int malloc_size;
	int maxblocks, maxallthreads;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
#ifdef XMSSMT
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmssmt_core_dp_sign_fast, threads, 0);
#else // ifdef XMSSMT
	cudaOccupancyMaxActiveBlocksPerMultiprocessor
		(&numBlocksPerSm, global_xmss_core_dp_sign_fast, threads, 0);
#endif // ifdef XMSSMT
	maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
	maxallthreads = maxblocks * threads;
	malloc_size = num / threads * threads + threads;

	CHECK(cudaMalloc((void **)&dev_sk, malloc_size * SK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_m, malloc_size * XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_sm, malloc_size * SM_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

#if USING_STREAM == 1
	maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
	// printf("eee\n");
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
	u8 *free_sk = dev_sk;
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
		printf("dp_sign_fast: %u %u %u %u\n", maxblocks, blocks, threads, s);

		CHECK(cudaMemcpyAsync(dev_sk, sk,
				      s * SK_BYTES * sizeof(u8), HOST_2_DEVICE, stream[iter]));
		CHECK(cudaMemcpyAsync(dev_m, m,
				      s * XMSS_MLEN * sizeof(u8), HOST_2_DEVICE, stream[iter]));

		void *Args[] =
		{ &dev_sk, &dev_sm, &dev_smlen, &dev_m, &mlen, &s };

#ifdef XMSSMT
		cudaLaunchCooperativeKernel( (void *)global_xmssmt_core_dp_sign_fast,
					     blocks, threads, Args, 0, stream[iter]);
#else // ifdef XMSSMT
		cudaLaunchCooperativeKernel( (void *)global_xmss_core_dp_sign_fast,
					     blocks, threads, Args, 0, stream[iter]);
#endif // ifdef XMSSMT

		CHECK(cudaMemcpyAsync(sk, dev_sk,
				      s * SK_BYTES * sizeof(u8), DEVICE_2_HOST, stream[iter]));
		CHECK(cudaMemcpyAsync(sm, dev_sm,
				      s * SM_BYTES * sizeof(u8), DEVICE_2_HOST, stream[iter]));

		sk += s * SK_BYTES;
		m += s * XMSS_MLEN;
		sm += s * SM_BYTES;
		dev_sk += s * SK_BYTES;
		dev_m += s * XMSS_MLEN;
		dev_sm += s * SM_BYTES;
		left -= s;
	}
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	*smlen = SM_BYTES;

	cudaFree(free_sk); cudaFree(free_sm); cudaFree(dev_smlen); cudaFree(free_m);
} // face_common_core_msdp_sign_fast

__device__ bds_state states[2 * D - 1];
__device__ treehash_inst treehash[(2 * D - 1) * (TREE_HEIGHT - BDS_K)];
__device__ unsigned int updates;

__device__ int dev_xmssmt_core_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
					    const u8 *m, u64 mlen)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	const u8 *pub_root = sk + INDEX_BYTES + 2 * N;

	uint64_t idx_tree;
	uint32_t idx_leaf;
	uint64_t i, j;
	int needswap_upto = -1;

	u8 sk_seed[N];
	u8 sk_prf[N];
	u8 pub_seed[N];
	// Init working params
	u8 R[N];
	u8 msg_h[N];
	uint32_t addr[8] = { 0 };
	uint32_t ots_addr[8] = { 0 };
	u8 idx_bytes_32[32];

	u8 *wots_sigs;

	// TODO refactor BDS state not to need separate treehash instances
	// bds_state states[2 * D - 1];
	// treehash_inst treehash[(2 * D - 1) * (TREE_HEIGHT - BDS_K)];

	if (tid == 0)
		for (i = 0; i < 2 * D - 1; i++) {
			states[i].treehash = treehash + i * (TREE_HEIGHT - BDS_K);
		}

	if (tid == 0)
		dev_xmssmt_deserialize_state(states, &wots_sigs, sk);

	// Extract SK
	u64 idx = 0;

	for (i = 0; i < INDEX_BYTES; i++) {
		idx |= ((u64)sk[i]) << 8 * (INDEX_BYTES - 1 - i);
	}

	/* Check if we can still sign with this sk.
	 * If not, return -2
	 *
	 * If this is the last possible signature (because the max index value
	 * is reached), production implementations should delete the secret key
	 * to prevent accidental further use.
	 *
	 * For the case of total tree height of 64 we do not use the last signature
	 * to be on the safe side (there is no index value left to indicate that the
	 * key is finished, hence external handling would be necessary)
	 */
	if (idx >= ((1ULL << FULL_HEIGHT) - 1)) {
		// Delete secret key here. We only do this in memory, production code
		// has to make sure that this happens on disk.
		memset(sk, 0xFF, INDEX_BYTES);
		memset(sk + INDEX_BYTES, 0, (SK_BYTES - INDEX_BYTES));
		if (idx > ((1ULL << FULL_HEIGHT) - 1))
			return -2;      // We already used all one-time keys
		if ((FULL_HEIGHT == 64) && (idx == ((1ULL << FULL_HEIGHT) - 1)))
			return -2;      // We already used all one-time keys
	}

	memcpy(sk_seed, sk + INDEX_BYTES, N);
	memcpy(sk_prf, sk + INDEX_BYTES + N, N);
	memcpy(pub_seed, sk + INDEX_BYTES + 3 * N, N);

	// Update SK
	if (tid == 0)
		for (i = 0; i < INDEX_BYTES; i++) {
			sk[i] = ((idx + 1) >> 8 * (INDEX_BYTES - 1 - i)) & 255;
		}
	// Secret key for this non-forward-secure version is now updated.
	// A production implementation should consider using a file handle instead,
	//  and write the updated secret key at this point!

	// ---------------------------------
	// Message Hashing
	// ---------------------------------

	// Message Hash:
	// First compute pseudorandom value
	dev_ull_to_bytes(idx_bytes_32, 32, idx);
	dev_prf(R, idx_bytes_32, sk_prf);

	/* Already put the message in the right place, to make it easier to prepend
	 * things when computing the hash over the message. */
	if (tid == 0) memcpy(sm + SIG_BYTES, m, mlen);
	g.sync();

	/* Compute the message hash. */
	dev_hash_message(msg_h, R, pub_root, idx,
			 sm + SIG_BYTES - PADDING_LEN - 3 * N, mlen);

	// Start collecting signature
	if (tid == 0) *smlen = 0;

	// Copy index to signature
	if (tid == 0)
		for (i = 0; i < INDEX_BYTES; i++) {
			sm[i] = (idx >> 8 * (INDEX_BYTES - 1 - i)) & 255;
		}

	sm += INDEX_BYTES;
	if (tid == 0) *smlen += INDEX_BYTES;

	// Copy R to signature
	if (tid == 0) memcpy(sm, R, N);

	sm += N;
	if (tid == 0) *smlen += N;

	// ----------------------------------
	// Now we start to "really sign"
	// ----------------------------------

	// Handle lowest layer separately as it is slightly different...

	// Prepare Address
	dev_set_type(ots_addr, 0);
	idx_tree = idx >> TREE_HEIGHT;
	idx_leaf = (idx & ((1 << TREE_HEIGHT) - 1));
	dev_set_layer_addr(ots_addr, 0);
	dev_set_tree_addr(ots_addr, idx_tree);
	dev_set_ots_addr(ots_addr, idx_leaf);

	// Compute WOTS signature
	if (tnum >= 67) {
		g.sync();
		dev_wots_sign_parallel(s_sm, msg_h, sk_seed, pub_seed, ots_addr, 0);
		g.sync();
		if (tid == 0) {
			memcpy(sm, s_sm, WOTS_SIG_BYTES);
		}
	} else {
		if (tid == 0) dev_wots_sign(sm, msg_h, sk_seed, pub_seed, ots_addr);
	}
	g.sync();

	sm += WOTS_SIG_BYTES;
	if (tid == 0) *smlen += WOTS_SIG_BYTES;

	if (tid == 0) memcpy(sm, states[0].auth, TREE_HEIGHT * N);
	sm += TREE_HEIGHT * N;
	if (tid == 0) *smlen += TREE_HEIGHT * N;

	// prepare signature of remaining layers
	for (i = 1; i < D; i++) {
		// put WOTS signature in place
		if (tid == 0) memcpy(sm, wots_sigs + (i - 1) * WOTS_SIG_BYTES, WOTS_SIG_BYTES);

		sm += WOTS_SIG_BYTES;
		if (tid == 0) *smlen += WOTS_SIG_BYTES;

		// put AUTH nodes in place
		if (tid == 0) memcpy(sm, states[i].auth, TREE_HEIGHT * N);
		sm += TREE_HEIGHT * N;
		if (tid == 0) *smlen += TREE_HEIGHT * N;
	}

	updates = (TREE_HEIGHT - BDS_K) >> 1;

	dev_set_tree_addr(addr, (idx_tree + 1));
	// mandatory update for NEXT_0 (does not count towards h-k/2) if NEXT_0 exists
	if ((1 + idx_tree) * (1 << TREE_HEIGHT) + idx_leaf < (1ULL << FULL_HEIGHT)) {
		g.sync();
		dev_bds_state_update_parallel(&states[D], sk_seed, pub_seed, addr);
	}
	g.sync();

	for (i = 0; i < D; i++) {
		// check if we're not at the end of a tree
		if (!(((idx + 1) & ((1ULL << ((i + 1) * TREE_HEIGHT)) - 1)) == 0)) {
			idx_leaf = (idx >> (TREE_HEIGHT * i)) & ((1 << TREE_HEIGHT) - 1);
			idx_tree = (idx >> (TREE_HEIGHT * (i + 1)));

			dev_set_layer_addr(addr, i);
			dev_set_tree_addr(addr, idx_tree);
			if (i == (unsigned int)(needswap_upto + 1)) {
				dev_bds_round_parallel(&states[i], idx_leaf, sk_seed, pub_seed, addr);
			}

			g.sync();
			int ret = dev_bds_treehash_update_parallel(&states[i], updates, sk_seed, pub_seed, addr);
			if (tid == 0) updates = ret;
			g.sync();
			dev_set_tree_addr(addr, (idx_tree + 1));

			if ((1 + idx_tree) * (1 << TREE_HEIGHT) + idx_leaf < (1ULL << (FULL_HEIGHT - TREE_HEIGHT * i))) {
				if (i > 0 && updates > 0 && states[D + i].next_leaf < (1ULL << FULL_HEIGHT)) {
					// if (tid == 0) dev_bds_state_update(&states[D + i], sk_seed, pub_seed, addr);
					dev_bds_state_update_parallel(&states[D + i], sk_seed, pub_seed, addr);

					updates--;
				}
			}
		} else if (idx < (1ULL << FULL_HEIGHT) - 1) {
			if (tid == 0) dev_deep_state_swap(states + D + i, states + i);

			dev_set_layer_addr(ots_addr, (i + 1));
			dev_set_tree_addr(ots_addr, ((idx + 1) >> ((i + 2) * TREE_HEIGHT)));
			dev_set_ots_addr(ots_addr, (((idx >> ((i + 1) * TREE_HEIGHT)) + 1) & ((1 << TREE_HEIGHT) - 1)));

			if (tnum >= 67) {
				g.sync();
				dev_wots_sign_parallel(s_sm, states[i].stack, sk_seed, pub_seed, ots_addr, 0);
				g.sync();
				if (tid == 0) {
					memcpy(wots_sigs + i * WOTS_SIG_BYTES, s_sm, WOTS_SIG_BYTES);
				}
			} else {
				if (tid == 0) dev_wots_sign(wots_sigs + i * WOTS_SIG_BYTES,
							    states[i].stack, sk_seed, pub_seed, ots_addr);
			}

			if (tid == 0) states[D + i].stackoffset = 0;
			if (tid == 0) states[D + i].next_leaf = 0;

			updates--;         // WOTS-signing counts as one update
			needswap_upto = i;
			if (tid == 0)
				for (j = 0; j < TREE_HEIGHT - BDS_K; j++) {
					states[i].treehash[j].completed = 1;
				}
		}
	}

	if (tid == 0) {
		memcpy(sm, m, mlen);
		*smlen += mlen;

		dev_xmssmt_serialize_state(sk, states);
	}

	return 0;
} // dev_xmssmt_core_ip_sign_fast

__global__ void global_xmssmt_core_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
						const u8 *m, u64 mlen)
{
	dev_xmssmt_core_ip_sign_fast(sk, sm, smlen, m, mlen);
} // global_xmssmt_core_ip_sign_fast

//input: sk, m, mlen
//output: sk, sm, smlen
void face_xmssmt_core_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
				   const u8 *m, u64 mlen)
{
	struct timespec start, stop;
	double result;
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	int device = DEVICE_USED;
	int threads = 1;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sk, sk, SK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_m, XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMemcpy(dev_m, m, XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_sm, (SIG_BYTES + XMSS_MLEN) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	threads = 1;
	void *kernelArgs[] = { &dev_sk, &dev_sm, &dev_smlen, &dev_m, &mlen };

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaDeviceSynchronize());
	cudaLaunchCooperativeKernel((void*)global_xmssmt_core_ip_sign_fast,
				    deviceProp.multiProcessorCount, threads, kernelArgs);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm,
			 (SIG_BYTES + XMSS_MLEN) * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), DEVICE_2_HOST));

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_sk); cudaFree(dev_smlen);
} // face_xmssmt_core_ip_sign_fast

int gpu_xmss_sign_fast(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	face_xmss_core_sign_fast(sk, sm, smlen, m, mlen);

	return 0;
} // gpu_xmss_sign_fast

int gpu_xmssmt_sign_fast(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	face_xmssmt_core_sign_fast(sk, sm, smlen, m, mlen);

	return 0;

} // gpu_xmssmt_sign_fast

int gpu_xmss_dp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			  const u8 *m, u64 mlen, u64 num)
{
	face_common_core_dp_sign_fast(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmss_dp_sign_fast

int gpu_xmssmt_dp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			    const u8 *m, u64 mlen, u64 num)
{
	face_common_core_dp_sign_fast(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmssmt_dp_sign_fast

int gpu_xmss_msdp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			    const u8 *m, u64 mlen, u64 num)
{
	face_common_core_msdp_sign_fast(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmss_dp_sign_fast

int gpu_xmssmt_msdp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			      const u8 *m, u64 mlen, u64 num)
{
	face_common_core_msdp_sign_fast(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmssmt_dp_sign_fast

int gpu_xmss_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	face_xmss_core_ip_sign_fast(sk, sm, smlen, m, mlen);

	return 0;
} // gpu_xmss_ip_sign_fast

int gpu_xmssmt_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	face_xmssmt_core_ip_sign_fast(sk, sm, smlen, m, mlen);

	return 0;

} // gpu_xmssmt_ip_sign_fast

#endif // ifdef USING_BDS
