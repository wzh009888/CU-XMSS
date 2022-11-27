#ifndef USING_BDS

#include <string.h>
#include <iostream>
using namespace std;

#include "gpu_utils.h"
#include "gpu_wots.h"
#include "gpu_xmss_commons.h"
#include "gpu_hash.h"
#include "gpu_hash_address.h"

#include "gpu_xmss_core_origin.h"
#include "gpu_sign_origin.h"

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <math.h>

__device__ u8 one_sm[WOTS_SIG_BYTES];
__device__ u8 one_root[N * 32 * 65536];

/**
 * Signs a message. Returns an array containing the signature followed by the
 * message and an updated secret key.
 */
__device__ int dev_xmssmt_core_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
					   const u8 *m, u64 mlen)
{
	const u8 *sk_seed = sk + INDEX_BYTES;
	const u8 *sk_prf = sk + INDEX_BYTES + N;
	const u8 *pub_root = sk + INDEX_BYTES + 2 * N;
	const u8 *pub_seed = sk + INDEX_BYTES + 3 * N;

	u8 root[N];
	u8 *mhash = root;
	u64 idx;
	u8 idx_bytes_32[32];
	unsigned int i;
	uint32_t idx_leaf;

	uint32_t ots_addr[8] = { 0 };

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);

	/* Already put the message in the right place, to make it easier to prepend
	 * things when computing the hash over the message. */
	memcpy(sm + SIG_BYTES, m, mlen);
	*smlen = SIG_BYTES + mlen;

	/* Read and use the current index from the secret key. */
	idx = (unsigned long)dev_bytes_to_ull(sk, INDEX_BYTES);

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

	memcpy(sm, sk, INDEX_BYTES);

	/*************************************************************************
	* THIS IS WHERE PRODUCTION IMPLEMENTATIONS WOULD UPDATE THE SECRET KEY. *
	*************************************************************************/
	/* Increment the index in the secret key. */
	dev_ull_to_bytes(sk, INDEX_BYTES, idx + 1);

	/* Compute the digest randomization value. */
	dev_ull_to_bytes(idx_bytes_32, 32, idx);
	dev_prf(sm + INDEX_BYTES, idx_bytes_32, sk_prf);

	// /* Compute the message hash. */
	dev_hash_message(mhash, sm + INDEX_BYTES, pub_root, idx,
			 sm + SIG_BYTES - PADDING_LEN - 3 * N, mlen);
	sm += INDEX_BYTES + N;

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);

	for (i = 0; i < D; i++) {
		idx_leaf = (idx & ((1 << TREE_HEIGHT) - 1));
		idx = idx >> TREE_HEIGHT;

		dev_set_layer_addr(ots_addr, i);
		dev_set_tree_addr(ots_addr, idx);
		dev_set_ots_addr(ots_addr, idx_leaf);

		/* Compute a WOTS signature. */
		/* Initially, root = mhash, but on subsequent iterations it is the root
		   of the subtree below the currently processed subtree. */
		dev_wots_sign(sm, root, sk_seed, pub_seed, ots_addr);
		sm += WOTS_SIG_BYTES;

		/* Compute the authentication path for the used WOTS leaf. */
		dev_treehash(root, sm, sk_seed, pub_seed, idx_leaf, ots_addr);
		sm += TREE_HEIGHT * N;
	}

	return 0;
} // dev_xmssmt_core_sign_origin

__global__ void global_xmssmt_core_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
					       const u8 *m, u64 mlen)
{
	dev_xmssmt_core_sign_origin(sk, sm, smlen, m, mlen);
}// global_xmssmt_core_sign_origin

__global__ void global_xmssmt_core_dp_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
						  const u8 *m, u64 mlen)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	dev_xmssmt_core_sign_origin(sk + SK_BYTES * tid,
				    sm + tid * (SIG_BYTES + XMSS_MLEN),
				    smlen, m + tid * XMSS_MLEN, mlen);
} // global_xmssmt_core_dp_sign_origin

__device__ int dev_xmssmt_core_ip_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
					      const u8 *m, u64 mlen)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	const u8 *sk_seed = sk + INDEX_BYTES;
	const u8 *sk_prf = sk + INDEX_BYTES + N;
	const u8 *pub_root = sk + INDEX_BYTES + 2 * N;
	const u8 *pub_seed = sk + INDEX_BYTES + 3 * N;

	u8 root[N];
	u8 *mhash = root;
	u64 idx;
	u8 idx_bytes_32[32];
	unsigned int i;
	uint32_t idx_leaf;

	uint32_t ots_addr[8] = { 0 };

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);

	/* Already put the message in the right place, to make it easier to prepend
	 * things when computing the hash over the message. */
	if (tid == 0) memcpy(sm + SIG_BYTES, m, mlen);
	if (tid == 0) *smlen = SIG_BYTES + mlen;

	/* Read and use the current index from the secret key. */
	idx = (unsigned long)dev_bytes_to_ull(sk, INDEX_BYTES);

	if (idx >= ((1ULL << FULL_HEIGHT) - 1)) {
		// Delete secret key here. We only do this in memory, production code
		// has to make sure that this happens on disk.
		if (tid == 0) memset(sk, 0xFF, INDEX_BYTES);
		if (tid == 0) memset(sk + INDEX_BYTES, 0, (SK_BYTES - INDEX_BYTES));
		if (idx > ((1ULL << FULL_HEIGHT) - 1))
			return -2;              // We already used all one-time keys
		if ((FULL_HEIGHT == 64) && (idx == ((1ULL << FULL_HEIGHT) - 1)))
			return -2;              // We already used all one-time keys
	}

	if (tid == 0) memcpy(sm, sk, INDEX_BYTES);

	/*************************************************************************
	* THIS IS WHERE PRODUCTION IMPLEMENTATIONS WOULD UPDATE THE SECRET KEY. *
	*************************************************************************/
	/* Increment the index in the secret key. */
	if (tid == 0) dev_ull_to_bytes(sk, INDEX_BYTES, idx + 1);

	/* Compute the digest randomization value. */
	if (tid == 0) dev_ull_to_bytes(idx_bytes_32, 32, idx);
	if (tid == 0) dev_prf(sm + INDEX_BYTES, idx_bytes_32, sk_prf);
	g.sync();

	/* Compute the message hash. */
	dev_hash_message(mhash, sm + INDEX_BYTES, pub_root, idx,
			 sm + SIG_BYTES - PADDING_LEN - 3 * N, mlen);

	sm += INDEX_BYTES + N;

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);

	for (i = 0; i < D; i++) {
		idx_leaf = (idx & ((1 << TREE_HEIGHT) - 1));
		idx = idx >> TREE_HEIGHT;

		dev_set_layer_addr(ots_addr, i);
		dev_set_tree_addr(ots_addr, idx);
		dev_set_ots_addr(ots_addr, idx_leaf);

		/* Compute a WOTS signature. */
		/* Initially, root = mhash, but on subsequent iterations it is the root
		   of the subtree below the currently processed subtree. */
		if (tnum >= WOTS_LEN) {
			int offset = 1;
			if (TREE_HEIGHT > 5) {
				while (offset <= tnum / 2) offset *= 2;
			} else {
				offset = 32 * WOTS_LEN;
			}
			if (tnum - offset < WOTS_LEN) offset = 0;

			if (tid == 0) memcpy(one_root, root, N);
			dev_wots_sign_parallel(sm, one_root, sk_seed, pub_seed, ots_addr, offset);
		} else {
			if (tid == 0) dev_wots_sign(sm, root, sk_seed, pub_seed, ots_addr);
		}

		sm += WOTS_SIG_BYTES;

		/* Compute the authentication path for the used WOTS leaf. */
		if (TREE_HEIGHT == 5) {
			dev_treehash_parallel_5(root, sm, sk_seed, pub_seed, idx_leaf, ots_addr);
		} else if (TREE_HEIGHT == 10) {
			dev_treehash_parallel_10(root, sm, sk_seed, pub_seed, idx_leaf, ots_addr);
		} else if (TREE_HEIGHT == 16) {
			dev_treehash_parallel_16(root, sm, sk_seed, pub_seed, idx_leaf, ots_addr);
		} else if (TREE_HEIGHT == 20) {
			dev_treehash_parallel_20(root, sm, sk_seed, pub_seed, idx_leaf, ots_addr);
		} else {
			if (tid == 0) dev_treehash(root, sm, sk_seed, pub_seed, idx_leaf, ots_addr);
		}

		sm += TREE_HEIGHT * N;
	}

	return 0;
} // dev_xmssmt_core_ip_sign_origin


// only for test
__device__ int dev_xmssmt_core_opk_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
					       const u8 *m, u64 mlen, u32 num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	const u8 *sk_seed = sk + INDEX_BYTES;
	const u8 *sk_prf = sk + INDEX_BYTES + N;
	const u8 *pub_root = sk + INDEX_BYTES + 2 * N;
	const u8 *pub_seed = sk + INDEX_BYTES + 3 * N;

	u8 *origin_sm = sm;
	u64 origin_idx = (unsigned long)dev_bytes_to_ull(sk, INDEX_BYTES);

	u8 root[N];
	u8 *mhash = root;
	u64 idx;
	u8 idx_bytes_32[32];

	uint32_t ots_addr[8] = { 0 };

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);

	u8* this_sm = origin_sm + tid * (SIG_BYTES + XMSS_MLEN);

	if (tid < num) {
		idx = origin_idx + tid;

		/* Already put the message in the right place, to make it easier to
		   prepend things when computing the hash over the message. */
		memcpy(this_sm + SIG_BYTES, m + tid * XMSS_MLEN, mlen);

		/* Read and use the current index from the secret key. */
		dev_ull_to_bytes(this_sm, INDEX_BYTES, idx);

		/* Increment the index in the secret key. */
		if (tid == 0) *smlen = SIG_BYTES + mlen; // only one
		if (tid == 0) dev_ull_to_bytes(sk, INDEX_BYTES, origin_idx + num);

		/* Compute the digest randomization value. */
		dev_ull_to_bytes(idx_bytes_32, 32, idx);
		dev_prf(this_sm + INDEX_BYTES, idx_bytes_32, sk_prf);

		/* Compute the message hash. */
		dev_hash_message(mhash, this_sm + INDEX_BYTES, pub_root, idx,
				 this_sm + SIG_BYTES - PADDING_LEN - 3 * N, mlen);
		memcpy(one_root + tid * N, root, N);
	}
	g.sync();
	// return 0;

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);

	for (int i = 0; i < D; i++) {
		g.sync();

		dev_set_layer_addr(ots_addr, i);

		// u64 idx_t = (origin_idx + tid) >> (i * TREE_HEIGHT);
		// u32 idx_leaf_t = (u32)(idx_t & ((1 << TREE_HEIGHT) - 1));
		// idx_t = idx_t >> TREE_HEIGHT;
		// if (tid < num) {
		// 	dev_set_tree_addr(ots_addr, idx_t);
		// 	dev_set_ots_addr(ots_addr, idx_leaf_t);
		// 	this_sm = origin_sm + INDEX_BYTES + N + tid * SM_BYTES
		// 		  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N);
		// 	dev_wots_sign(this_sm, one_root + tid * N,
		// 		      sk_seed, pub_seed, ots_addr);
		// }

		int lengths[WOTS_LEN];
		u8 buf[N + 32];
		u8 temp[N];

		for (int j = tid; j < WOTS_LEN * num; j += tnum) {
			int jd = j % WOTS_LEN;
			int jjd = j / WOTS_LEN;
			u8* jsm = origin_sm + INDEX_BYTES + N + jjd * SM_BYTES + jd * N
				  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N);
			u64 idx_t = (origin_idx + jjd) >> (i * TREE_HEIGHT);
			u32 idx_leaf_t = (u32)(idx_t & ((1 << TREE_HEIGHT) - 1));

			idx_t = idx_t >> TREE_HEIGHT;
			u32 addr_temp[8];
			memcpy(addr_temp, ots_addr, 8 * sizeof(u32));
			dev_set_tree_addr(addr_temp, idx_t); // Can't be omitted
			dev_set_ots_addr(addr_temp, idx_leaf_t);

			dev_chain_lengths(lengths, one_root + jjd * N);
			dev_set_hash_addr(addr_temp, 0);
			dev_set_key_and_mask(addr_temp, 0);
			memcpy(buf, pub_seed, N);
			dev_set_chain_addr(addr_temp, jd);
			dev_addr_to_bytes(buf + N, addr_temp);
			dev_prf_keygen(temp, buf, sk_seed);
			dev_gen_chain(temp, temp, 0, lengths[jd], pub_seed, addr_temp);
			memcpy(jsm, temp, N);
		}

		/* Compute the authentication path for the used WOTS leaf. */
		if (TREE_HEIGHT == 5) {
			// u32 hh;
			// if (i > 3) // We can't handle 40 and 60 yet
			// 	hh = 1 << 20;
			// else
			// 	hh = (1 << ((i + 1) * TREE_HEIGHT));
			// int count = (origin_idx + num - 1) / hh - origin_idx / hh;
			// int exe, this_off;
			// // if (tid == 0) printf("count = %d\n", count);
			// for (int j = 0; j < count + 1; j++) {
			// 	// this off 是最底层树的上一个截断点
			// 	this_off = (origin_idx + j * hh) / hh * hh - origin_idx;
			// 	this_off = this_off >= 0 ? this_off : 0;
			// 	if (j == 0) {
			// 		exe = (origin_idx + hh) / hh * hh - origin_idx;
			// 		exe = exe >= num ? num : exe;
			// 	} else {
			// 		exe = num - this_off < hh ? num - this_off : hh;
			// 	}
			// 	// if (tid == 0) printf("i = %d %d %d\n", i, count, exe);
			// 	dev_set_tree_addr(ots_addr,
			// 			  (origin_idx + this_off) >> ((i + 1) * TREE_HEIGHT));
			// 	this_sm = origin_sm + this_off * (SIG_BYTES + XMSS_MLEN)
			// 		  + INDEX_BYTES + N + WOTS_SIG_BYTES
			// 		  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N);
			// 	dev_treehash_opk_parallel_5(one_root + this_off * N,
			// 				    this_sm, sk_seed, pub_seed,
			// 				    origin_idx + this_off, i, ots_addr, exe);
			// 	g.sync();// for print
			// 	// for (int kk = 0; kk < TREE_HEIGHT * N; kk += N) {
			// 	// 	if (tid == 0) printf("%02x ", this_sm[kk]);
			// 	// }
			// 	// if (tid == 0) printf("\n");
			// }
			// g.sync();

			dev_treehash_opk_multi_parallel_5(one_root, origin_sm, sk_seed,
							  pub_seed, origin_idx, i, num, ots_addr);
			g.sync();

			// for (int j = 0; j < num; j++) {
			// 	u64 idx_t = (origin_idx + j) >> (i * TREE_HEIGHT);
			// 	u32 idx_leaf_t = (u32)(idx_t & ((1 << TREE_HEIGHT) - 1));
			// 	idx_t = idx_t >> TREE_HEIGHT;
			// 	dev_set_tree_addr(ots_addr, idx_t);
			// 	this_sm = origin_sm + j * SM_BYTES + INDEX_BYTES + N
			// 		  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N) + WOTS_SIG_BYTES;
			// 	dev_treehash_parallel_5(one_root + j * N, this_sm,
			// 				sk_seed, pub_seed, idx_leaf_t, ots_addr);
			// 	g.sync();
			// }

		} else if (TREE_HEIGHT == 10) {
			// u32 hh;
			// if (i > 1) // We can't handle 40 and 60 yet
			// 	hh = 1 << 20;
			// else
			// 	hh = (1 << ((i + 1) * TREE_HEIGHT));
			// int count = (origin_idx + num - 1) / hh - origin_idx / hh;
			// int exe, this_off;
			// // if (tid == 0) printf("count = %d\n", count);
			// for (int j = 0; j < count + 1; j++) {
			// 	// this off 是最底层树的上一个截断点
			// 	this_off = (origin_idx + j * hh) / hh * hh - origin_idx;
			// 	this_off = this_off >= 0 ? this_off : 0;
			// 	if (j == 0) {
			// 		exe = (origin_idx + hh) / hh * hh - origin_idx;
			// 		exe = exe >= num ? num : exe;
			// 	} else {
			// 		exe = num - this_off < hh ? num - this_off : hh;
			// 	}
			// 	// if (tid == 0) printf("this_off = %d %d\n", this_off, exe);
			// 	dev_set_tree_addr(ots_addr,
			// 			  (origin_idx + this_off) >> ((i + 1) * TREE_HEIGHT));
			// 	this_sm = origin_sm + this_off * (SIG_BYTES + XMSS_MLEN)
			// 		  + INDEX_BYTES + N + WOTS_SIG_BYTES
			// 		  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N);
			// 	dev_treehash_opk_parallel_10(one_root + this_off * N,
			// 				     this_sm, sk_seed, pub_seed,
			// 				     origin_idx + this_off, i, ots_addr, exe);
			// }
			// g.sync();

			dev_treehash_opk_multi_parallel_10(one_root, origin_sm, sk_seed,
							  pub_seed, origin_idx, i, num, ots_addr);
			g.sync();

			// for (int j = 0; j < num; j++) {
			// 	u64 idx_t = (origin_idx + j) >> (i * TREE_HEIGHT);
			// 	u32 idx_leaf_t = (u32)(idx_t & ((1 << TREE_HEIGHT) - 1));
			// 	idx_t = idx_t >> TREE_HEIGHT;
			// 	dev_set_tree_addr(ots_addr, idx_t);
			// 	this_sm = origin_sm + j * (SIG_BYTES + XMSS_MLEN) + INDEX_BYTES + N
			// 		  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N) + WOTS_SIG_BYTES;
			// 	dev_treehash_parallel_10(one_root + j * N, this_sm,
			// 				 sk_seed, pub_seed, idx_leaf_t, ots_addr);
			// 	g.sync();
			// }

		} else if (TREE_HEIGHT == 16) {
			dev_set_tree_addr(ots_addr, origin_idx >> ((i + 1) * TREE_HEIGHT));
			this_sm = origin_sm + INDEX_BYTES + N
				  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N) + WOTS_SIG_BYTES;
			dev_treehash_opk_parallel_16(one_root, this_sm, sk_seed, pub_seed,
						     origin_idx, i, ots_addr, num);
			g.sync();

			// for (int j = 0; j < num; j++) {
			// 	u64 idx_t = (origin_idx + tid) >> (i * TREE_HEIGHT);
			// 	u32 idx_leaf_t = (u32)(idx_t & ((1 << TREE_HEIGHT) - 1));
			// 	idx_t = idx_t >> TREE_HEIGHT;
			// 	dev_set_tree_addr(ots_addr, idx_t);
			// 	this_sm = origin_sm + j * (SIG_BYTES + XMSS_MLEN) + INDEX_BYTES + N
			// 		  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N) + WOTS_SIG_BYTES;
			// 	dev_treehash_parallel_16(one_root + j * N, this_sm,
			// 				 sk_seed, pub_seed, idx_leaf_t, ots_addr);
			// 	g.sync();
			// }

		} else { // 20
			u64 hh;
			// if (i >= 1) // We can't handle 40 and 60 yet
			hh = 1 << 20;
			// else
			// 	hh = (1 << ((i + 1) * TREE_HEIGHT));
			int count = (origin_idx + num - 1) / hh - origin_idx / hh;
			int exe, this_off;
			if (tid == 0) printf("count = %d\n", count);
			for (int j = 0; j < count + 1; j++) {
				// this off 是最底层树的上一个截断点
				this_off = (origin_idx + j * hh) / hh * hh - origin_idx;
				this_off = this_off >= 0 ? this_off : 0;
				if (j == 0) {
					exe = (origin_idx + hh) / hh * hh - origin_idx;
					exe = exe >= num ? num : exe;
				} else {
					exe = num - this_off < hh ? num - this_off : hh;
				}
				if (tid == 0) printf("this_off = %d %d\n", this_off, exe);
				dev_set_tree_addr(ots_addr,
						  (origin_idx + this_off) >> ((i + 1) * TREE_HEIGHT));
				this_sm = origin_sm + this_off * (SIG_BYTES + XMSS_MLEN)
					  + INDEX_BYTES + N + WOTS_SIG_BYTES
					  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N);
				dev_treehash_opk_parallel_20(one_root + this_off * N,
							     this_sm, sk_seed, pub_seed,
							     origin_idx + this_off, i, ots_addr, exe);
			}
			g.sync();


			// dev_set_tree_addr(ots_addr, origin_idx >> ((i + 1) * TREE_HEIGHT));
			// this_sm = origin_sm + INDEX_BYTES + N + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N) + WOTS_SIG_BYTES;
			// dev_treehash_opk_parallel_20(one_root, this_sm, sk_seed, pub_seed, origin_idx, i, ots_addr, num);
			// g.sync();

			// for (int j = 0; j < num; j++) {
			// 	u64 idx_t = (origin_idx + tid) >> (i * TREE_HEIGHT);
			// 	u32 idx_leaf_t = (u32)(idx_t & ((1 << TREE_HEIGHT) - 1));
			// 	idx_t = idx_t >> TREE_HEIGHT;
			// 	dev_set_tree_addr(ots_addr, idx_t);
			// 	this_sm = origin_sm + j * (SIG_BYTES + XMSS_MLEN) + INDEX_BYTES + N
			// 		  + i * (WOTS_SIG_BYTES + TREE_HEIGHT * N) + WOTS_SIG_BYTES;
			// 	dev_treehash_parallel_20(one_root + j * N, this_sm,
			// 				 sk_seed, pub_seed, idx_leaf_t, ots_addr);
			// 	g.sync();
			// }

		}

	}

	return 0;

} // dev_xmssmt_core_opk_sign_origin

__global__ void global_xmssmt_core_ip_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
						  const u8 *m, u64 mlen)
{
	dev_xmssmt_core_ip_sign_origin(sk, sm, smlen, m, mlen);
} // global_xmssmt_core_ip_sign_origin

__global__ void global_xmssmt_core_opk_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
						   const u8 *m, u64 mlen, u64 num)
{
	dev_xmssmt_core_opk_sign_origin(sk, sm, smlen, m, mlen, num);
} // global_xmssmt_core_opk_sign_origin

void face_xmssmt_core_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
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

	CHECK(cudaMalloc((void **)&dev_sm, (SIG_BYTES + XMSS_MLEN + 64) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	CHECK(cudaDeviceSynchronize());
	global_xmssmt_core_sign_origin << < 1, 1 >> >
		(dev_sk, dev_sm, dev_smlen, dev_m, mlen);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm,
			 (SIG_BYTES + XMSS_MLEN) * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), DEVICE_2_HOST));

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_sk); cudaFree(dev_smlen);
} // face_xmssmt_core_sign_origin

void face_xmssmt_core_dp_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
				     const u8 *m, u64 mlen, u64 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	int device = DEVICE_USED;
	int blocks = 1;
	int threads = 32;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
	blocks = num / threads;

	CHECK(cudaMalloc((void **)&dev_sk, num * SK_BYTES * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_m, num * XMSS_MLEN * sizeof(u8)));


	CHECK(cudaMalloc((void **)&dev_sm,
			 num * (SIG_BYTES + XMSS_MLEN) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaMemcpy(dev_sk, sk, num * SK_BYTES * sizeof(u8), HOST_2_DEVICE));
	CHECK(cudaMemcpy(dev_m, m, num * XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaDeviceSynchronize());
	global_xmssmt_core_dp_sign_origin << < blocks, threads >> >
		(dev_sk, dev_sm, dev_smlen, dev_m, mlen);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(sk, dev_sk, num * SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm,
			 num * (SIG_BYTES + XMSS_MLEN) * sizeof(u8), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	*smlen = SIG_BYTES + XMSS_MLEN;

	cudaFree(dev_sk); cudaFree(dev_sm); cudaFree(dev_smlen); cudaFree(dev_m);
} // face_xmssmt_core_dp_sign_origin

// not finished
void face_xmssmt_core_msdp_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
				       const u8 *m, u64 mlen, u64 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	int device = DEVICE_USED;
	int block = 1;
	cudaDeviceProp deviceProp;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);
	block = deviceProp.multiProcessorCount;

	CHECK(cudaMalloc((void **)&dev_sk, num * SK_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sk, sk, num * SK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_m, num * XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMemcpy(dev_m, m, num * XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_sm,
			 num * (SIG_BYTES + XMSS_MLEN) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	CHECK(cudaDeviceSynchronize());
	global_xmssmt_core_dp_sign_origin << < block, num / block >> >
		(dev_sk, dev_sm, dev_smlen, dev_m, mlen);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(sk, dev_sk, num * SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm,
			 num * (SIG_BYTES + XMSS_MLEN) * sizeof(u8), DEVICE_2_HOST));
	*smlen = SIG_BYTES + XMSS_MLEN;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;


	cudaFree(dev_sk); cudaFree(dev_sm); cudaFree(dev_smlen); cudaFree(dev_m);
} // face_xmssmt_core_msdp_sign_origin

void face_xmssmt_core_ip_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
				     const u8 *m, u64 mlen)
{
	struct timespec start, stop;
	double result;
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	cudaDeviceProp deviceProp;
	int device = DEVICE_USED;
	int threads = 1, blocks = 1, maxblock = 1;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sk, sk, SK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_m, XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMemcpy(dev_m, m, XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_sm, (SIG_BYTES + XMSS_MLEN) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	void *kernelArgs[] = { &dev_sk, &dev_sm, &dev_smlen, &dev_m, &mlen };

	if (TREE_HEIGHT == 5) {
		threads = 32;
		blocks = WOTS_LEN + 3;
	} else if (TREE_HEIGHT == 10) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_ip_sign_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;

		blocks = 1;
		while (blocks <= maxblock / 2) {
			blocks *= 2;
		}
		int final_blocks = blocks + WOTS_LEN / threads + 1;
		if (final_blocks <= maxblock) blocks = final_blocks;
		blocks = maxblock;
	} else if (TREE_HEIGHT == 16) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_ip_sign_origin, threads, 0);
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
			(&numBlocksPerSm, global_xmssmt_core_ip_sign_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;

		blocks = 1;
		while (blocks <= maxblock / 2) {
			blocks *= 2;
		}
		blocks = maxblock;
	}
	#ifdef PRINT_ALL
	printf("origin xmssmt ip sign %d %d %d\n", maxblock, threads, blocks);
	#endif // ifdef PRINT_ALL

	CHECK(cudaDeviceSynchronize());
	cudaLaunchCooperativeKernel((void*)global_xmssmt_core_ip_sign_origin,
				    blocks, threads, kernelArgs);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm,
			 (SIG_BYTES + XMSS_MLEN) * sizeof(u8), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), DEVICE_2_HOST));

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_sk); cudaFree(dev_smlen);
} // face_xmssmt_core_ip_sign_origin

void face_common_core_opk_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
				      const u8 *m, u64 mlen, u64 num)
{
	struct timespec start, stop;
	double result;
	u8 *dev_sk = NULL, *dev_sm = NULL, *dev_m = NULL;
	u64 *dev_smlen = NULL;
	cudaDeviceProp deviceProp;
	int device = DEVICE_USED;
	int threads = 1, blocks = 1, maxblock = 1;

	CHECK(cudaSetDevice(device));
	cudaGetDeviceProperties(&deviceProp, device);

	CHECK(cudaMalloc((void **)&dev_sk, SK_BYTES * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sk, sk, SK_BYTES * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_m, num * XMSS_MLEN * sizeof(u8)));
	CHECK(cudaMemcpy(dev_m, m, num * XMSS_MLEN * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_sm,
			 num * (SIG_BYTES + XMSS_MLEN) * sizeof(u8)));
	CHECK(cudaMalloc((void **)&dev_smlen, sizeof(u64)));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	void *kernelArgs[] = { &dev_sk, &dev_sm, &dev_smlen, &dev_m, &mlen, &num };

	if (TREE_HEIGHT == 5) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_opk_sign_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;
		blocks = maxblock;
	} else if (TREE_HEIGHT == 10) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_opk_sign_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;
		blocks = maxblock;
	} else if (TREE_HEIGHT == 16) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_opk_sign_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;

		blocks = 1;
		while (blocks <= maxblock / 2) {
			blocks *= 2;
		}
	} else if (TREE_HEIGHT == 20) {
		threads = 32;
		int numBlocksPerSm = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor
			(&numBlocksPerSm, global_xmssmt_core_opk_sign_origin, threads, 0);
		maxblock = numBlocksPerSm * deviceProp.multiProcessorCount;

		blocks = 1;
		while (blocks <= maxblock / 2) {
			blocks *= 2;
		}
	}
	#ifdef PRINT_ALL
	printf("xmssmt opk sign %d %d %d\n", maxblock, threads, blocks);
	#endif // ifdef PRINT_TIME

	CHECK(cudaDeviceSynchronize());
	cudaLaunchCooperativeKernel((void*)global_xmssmt_core_opk_sign_origin,
				    blocks, threads, kernelArgs);

	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(sk, dev_sk, SK_BYTES * sizeof(u8), DEVICE_2_HOST));
	CHECK(cudaMemcpy(sm, dev_sm, num * SM_BYTES * sizeof(u8), DEVICE_2_HOST));

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	result = (stop.tv_sec - start.tv_sec) * 1e6
		 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	g_result += result;

	CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), DEVICE_2_HOST));

	cudaFree(dev_m); cudaFree(dev_sm); cudaFree(dev_sk); cudaFree(dev_smlen);
} // face_common_core_opk_sign_origin

int gpu_xmss_sign_origin(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	face_xmssmt_core_sign_origin(sk, sm, smlen, m, mlen);

	return 0;
} // gpu_xmss_sign_origin

int gpu_xmssmt_sign_origin(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	face_xmssmt_core_sign_origin(sk, sm, smlen, m, mlen);

	return 0;
} // gpu_xmssmt_sign_origin

int gpu_xmss_dp_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
			    const u8 *m, u64 mlen, u64 num)
{
	face_xmssmt_core_dp_sign_origin(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmss_dp_sign_fast

int gpu_xmssmt_dp_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
			      const u8 *m, u64 mlen, u64 num)
{
	face_xmssmt_core_dp_sign_origin(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmssmt_dp_sign_fast

int gpu_xmss_msdp_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
			      const u8 *m, u64 mlen, u64 num)
{
	face_xmssmt_core_msdp_sign_origin(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmss_dp_sign_fast

int gpu_xmssmt_msdp_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
				const u8 *m, u64 mlen, u64 num)
{
	face_xmssmt_core_msdp_sign_origin(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmssmt_dp_sign_fast

int gpu_xmss_ip_sign_origin(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	face_xmssmt_core_ip_sign_origin(sk, sm, smlen, m, mlen);

	return 0;
} // gpu_xmss_ip_sign_origin

int gpu_xmssmt_ip_sign_origin(u8 *sk, u8 *sm, u64 *smlen, const u8 *m, u64 mlen)
{
	face_xmssmt_core_ip_sign_origin(sk, sm, smlen, m, mlen);

	return 0;
} // gpu_xmssmt_ip_sign_origin

int gpu_xmss_opk_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
			     const u8 *m, u64 mlen, u64 num)
{
	face_common_core_opk_sign_origin(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmss_opk_sign_origin

int gpu_xmssmt_opk_sign_origin(u8 *sk, u8 *sm, u64 *smlen,
			       const u8 *m, u64 mlen, u64 num)
{
	face_common_core_opk_sign_origin(sk, sm, smlen, m, mlen, num);

	return 0;
} // gpu_xmssmt_opk_sign_origin

#endif // ifndef USING_BDS
