#ifndef USING_BDS

#include "gpu_utils.h"
#include "gpu_wots.h"
#include "gpu_hash.h"
#include "gpu_hash_address.h"
#include "gpu_xmss_commons.h"

#include "gpu_xmss_core_origin.h"

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <iostream>
using namespace std;

#if TREE_HEIGHT < 20
#define LEAD_NUM (1 << TREE_HEIGHT)
#else // if TREE_HEIGHT < 20
#define LEAD_NUM (1 << 16)
#endif // if TREE_HEIGHT < 20

__device__ extern u8 branch[LEAD_NUM * N];

__device__ extern u8 wots_pk[WOTS_SIG_BYTES * 65536]; // provided that thread size is 2048
__device__ extern u8 c_topnode[N * 32768 * 2];

__device__ extern u8 one_auth_path[20 * N];
__device__ extern u8 one_treehash_node[20 * N];

__device__ u8 opk_auth_path[20 * 65536 * N];    // provided max 65536 sign

/**
 * For a given leaf index, computes the authentication path and the resulting
 * root node using Merkle's TreeHash algorithm.
 * Expects the layer and tree parts of subtree_addr to be set.
 */
__device__ void dev_treehash(u8 *root, u8 *auth_path,
			     const u8 *sk_seed, const u8 *pub_seed,
			     u32 leaf_idx, const u32 subtree_addr[8])
{
	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 heights[TREE_HEIGHT + 1];
	u32 offset = 0;

	/* The subtree has at most 2^20 leafs, so u32 suffices. */
	u32 idx;
	u32 tree_idx;

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	for (idx = 0; idx < (u32)(1 << TREE_HEIGHT); idx++) {
		/* Add the next leaf node to the stack. */
		dev_set_ltree_addr(ltree_addr, idx);
		dev_set_ots_addr(ots_addr, idx);
		dev_gen_leaf_wots(stack + offset * N,
				  sk_seed, pub_seed, ltree_addr, ots_addr);
		offset++;
		heights[offset - 1] = 0;

		/* If this is a node we need for the auth path.. */
		if ((leaf_idx ^ 0x1) == idx) {
			memcpy(auth_path, stack + (offset - 1) * N, N);
		}

		/* While the top-most nodes are of equal height.. */
		while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
			/* Compute index of the new node, in the next layer. */
			tree_idx = (idx >> (heights[offset - 1] + 1));

			/* Hash the top-most nodes from the stack together. */
			/* Note that tree height is the 'lower' layer, even though we use
			   the index of the new node on the 'higher' layer. This follows
			   from the fact that we address the hash function calls. */
			dev_set_tree_height(node_addr, heights[offset - 1]);
			dev_set_tree_index(node_addr, tree_idx);
			dev_thash_h(stack + (offset - 2) * N,
				    stack + (offset - 2) * N, pub_seed, node_addr);
			offset--;
			/* Note that the top-most node is now one layer higher. */
			heights[offset - 1]++;

			/* If this is a node we need for the auth path.. */
			if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
				memcpy(auth_path + heights[offset - 1] * N,
				       stack + (offset - 1) * N, N);
			}
		}
	}
	memcpy(root, stack, N);
} // dev_treehash

#if TREE_HEIGHT == 5
__device__ void dev_treehash_parallel_5(u8 *root, u8 *auth_path,
					const u8 *sk_seed, const u8 *pub_seed,
					u32 leaf_idx, const u32 subtree_addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();
	int max_threads = 32 * WOTS_LEN;

#if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))
	const unsigned int tnum = gridDim.x * blockDim.x;
	int ttid, id;

	if (tnum < max_threads) {
		if (tid == 0) printf("wrong thread size\n");
		return;
	}
#endif // if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))

#ifdef USING_PARALLEL_WOTS_PKGEN
	#ifdef USING_COALESCED_ACCESS
	u8 buf[N + 32];
	max_threads = tnum - 5;
	if (tid < max_threads) {
		for (int i = tid; i < WOTS_LEN * 32; i += max_threads) {
			dev_set_ots_addr(ots_addr, i / WOTS_LEN);
			dev_set_chain_addr(ots_addr, i % WOTS_LEN);
			dev_set_hash_addr(ots_addr, 0);
			dev_set_key_and_mask(ots_addr, 0);
			memcpy(buf, pub_seed, N);
			dev_addr_to_bytes(buf + N, ots_addr);
			#ifdef USING_LOCAL_MEMORY
			u8 temp[N];
			dev_prf_keygen(temp, buf, sk_seed);
			dev_gen_chain(temp, temp, 0, WOTS_W - 1, pub_seed, ots_addr);
			memcpy(wots_pk + i * N, temp, N);
			#else // ifdef USING_LOCAL_MEMORY
			dev_prf_keygen(wots_pk + i * N, buf, sk_seed);
			dev_gen_chain(wots_pk + i * N, wots_pk + i * N,
				      0, WOTS_W - 1, pub_seed, ots_addr);
			#endif // ifdef USING_LOCAL_MEMORY
		}
	}
	#else // ifdef USING_COALESCED_ACCESS
	ttid = tid % 32;
	id = tid / 32;
	if (tid < 32 * WOTS_LEN) {
		u8 *pk = wots_pk + ttid * WOTS_SIG_BYTES;
		unsigned char buf[N + 32];
		dev_set_ots_addr(ots_addr, ttid);
		dev_set_hash_addr(ots_addr, 0);
		dev_set_key_and_mask(ots_addr, 0);
		memcpy(buf, pub_seed, N);

		dev_set_chain_addr(ots_addr, id);
		dev_addr_to_bytes(buf + N, ots_addr);
		dev_prf_keygen(pk + id * N, buf, sk_seed);
		dev_gen_chain(pk + id * N, pk + id * N,
			      0, WOTS_W - 1, pub_seed, ots_addr);
	}
	#endif // ifdef USING_COALESCED_ACCESS
	g.sync();
#else // ifdef USING_PARALLEL_WOTS_PKGEN
	if (tid < 32) {
		dev_set_ots_addr(ots_addr, tid);
		dev_wots_pkgen(wots_pk + tid * WOTS_SIG_BYTES, sk_seed, pub_seed, ots_addr);
	}
#endif // ifdef USING_PARALLEL_WOTS_PKGEN

#ifdef USING_PARALLEL_L_TREE
	ttid = tid % 32;
	id = tid / 32;
	u8 *begin = wots_pk + ttid * WOTS_SIG_BYTES;
	dev_set_ltree_addr(ltree_addr, ttid);
	if (WOTS_LEN == 67) {
		if (id < 33) {
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 0);
			dev_thash_h(begin + id * 2 * N,
				    begin + id * 2 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 17) {
			memcpy(begin + id * 4 * N + N, begin + id * 4 * N + 2 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 1);
			dev_thash_h(begin + id * 4 * N,
				    begin + id * 4 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 8) {
			memcpy(begin + id * 8 * N + N, begin + id * 8 * N + 4 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 2);
			dev_thash_h(begin + id * 8 * N,
				    begin + id * 8 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 4) {
			memcpy(begin + id * 16 * N + N, begin + id * 16 * N + 8 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 3);
			dev_thash_h(begin + id * 16 * N,
				    begin + id * 16 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 2) {
			memcpy(begin + id * 32 * N + N, begin + id * 32 * N + 16 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 4);
			dev_thash_h(begin + id * 32 * N,
				    begin + id * 32 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 5);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(begin + N, begin + 64 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 6);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(branch + tid * N, begin, N);
		}
		g.sync();
	} else if (WOTS_LEN == 51) {
		if (id < 25) {
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 0);
			dev_thash_h(begin + id * 2 * N,
				    begin + id * 2 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 13) {
			memcpy(begin + id * 4 * N + N, begin + id * 4 * N + 2 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 1);
			dev_thash_h(begin + id * 4 * N,
				    begin + id * 4 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 6) {
			memcpy(begin + id * 8 * N + N, begin + id * 8 * N + 4 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 2);
			dev_thash_h(begin + id * 8 * N,
				    begin + id * 8 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 3) {
			memcpy(begin + id * 16 * N + N, begin + id * 16 * N + 8 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 3);
			dev_thash_h(begin + id * 16 * N,
				    begin + id * 16 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 2) {
			memcpy(begin + id * 32 * N + N, begin + id * 32 * N + 16 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 4);
			dev_thash_h(begin + id * 32 * N,
				    begin + id * 32 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 5);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(branch + tid * N, begin, N);
		}
		g.sync();
	}
#else // ifdef USING_PARALLEL_L_TREE
	if (tid < 32) {
		u8 *pk = wots_pk + tid * WOTS_SIG_BYTES;
		dev_set_ltree_addr(ltree_addr, tid);
		dev_l_tree(branch + tid * N, pk, pub_seed, ltree_addr);
	}
	g.sync();
#endif // ifdef USING_PARALLEL_L_TREE

	if (tid == ((leaf_idx >> 0) ^ 0x1))
		memcpy(one_auth_path, branch + tid * N, N);

	max_threads = 32;
	int p_height = 5; // 2^6

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			if (tid == ((leaf_idx >> i) ^ 0x1))
				memcpy(one_auth_path + i * N, branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(auth_path + i * N, one_auth_path + i * N, N);
		memcpy(root, branch, N);
	}

} // dev_treehash_parallel_5

__device__ void dev_treehash_opk_parallel_5(u8 *root, u8 *auth_path,
					    const u8 *sk_seed, const u8 *pub_seed,
					    u64 idx_ex, u32 iter,
					    const u32 subtree_addr[8], int opk_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();
	int max_threads = 32 * WOTS_LEN;         // 32 * 67/51

#if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))
	const unsigned int tnum = gridDim.x * blockDim.x;
	int ttid = tid % 32;
	int id = tid / 32;

	if (tnum < max_threads) {
		printf("wrong thread size\n");
		return;
	}
#endif // if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))

#ifdef USING_PARALLEL_WOTS_PKGEN
	if (tid < 32 * WOTS_LEN) {
		u8 *pk = wots_pk + ttid * WOTS_SIG_BYTES;
		unsigned char buf[N + 32];
		dev_set_ots_addr(ots_addr, ttid);
		dev_set_hash_addr(ots_addr, 0);
		dev_set_key_and_mask(ots_addr, 0);
		memcpy(buf, pub_seed, N);

		dev_set_chain_addr(ots_addr, id);
		dev_addr_to_bytes(buf + N, ots_addr);
		dev_prf_keygen(pk + id * N, buf, sk_seed);
		dev_gen_chain(pk + id * N, pk + id * N,
			      0, WOTS_W - 1, pub_seed, ots_addr);
	}
	g.sync();
#else // ifdef USING_PARALLEL_WOTS_PKGEN
	if (tid < 32) {
		dev_set_ots_addr(ots_addr, tid);
		dev_wots_pkgen(wots_pk + tid * WOTS_SIG_BYTES, sk_seed, pub_seed, ots_addr);
	}
#endif // ifdef USING_PARALLEL_WOTS_PKGEN

#ifdef USING_PARALLEL_L_TREE
	u8 *begin = wots_pk + ttid * WOTS_SIG_BYTES;
	dev_set_ltree_addr(ltree_addr, ttid);
	if (WOTS_LEN == 67) {
		if (id < 33) {
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 0);
			dev_thash_h(begin + id * 2 * N,
				    begin + id * 2 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 17) {
			memcpy(begin + id * 4 * N + N, begin + id * 4 * N + 2 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 1);
			dev_thash_h(begin + id * 4 * N,
				    begin + id * 4 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 8) {
			memcpy(begin + id * 8 * N + N, begin + id * 8 * N + 4 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 2);
			dev_thash_h(begin + id * 8 * N,
				    begin + id * 8 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 4) {
			memcpy(begin + id * 16 * N + N, begin + id * 16 * N + 8 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 3);
			dev_thash_h(begin + id * 16 * N,
				    begin + id * 16 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 2) {
			memcpy(begin + id * 32 * N + N, begin + id * 32 * N + 16 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 4);
			dev_thash_h(begin + id * 32 * N,
				    begin + id * 32 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 5);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(begin + N, begin + 64 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 6);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(branch + tid * N, begin, N);
		}
		g.sync();
	} else if (WOTS_LEN == 51) {
		if (id < 25) {
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 0);
			dev_thash_h(begin + id * 2 * N,
				    begin + id * 2 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 13) {
			memcpy(begin + id * 4 * N + N, begin + id * 4 * N + 2 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 1);
			dev_thash_h(begin + id * 4 * N,
				    begin + id * 4 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 6) {
			memcpy(begin + id * 8 * N + N, begin + id * 8 * N + 4 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 2);
			dev_thash_h(begin + id * 8 * N,
				    begin + id * 8 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 3) {
			memcpy(begin + id * 16 * N + N, begin + id * 16 * N + 8 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 3);
			dev_thash_h(begin + id * 16 * N,
				    begin + id * 16 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 2) {
			memcpy(begin + id * 32 * N + N, begin + id * 32 * N + 16 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 4);
			dev_thash_h(begin + id * 32 * N,
				    begin + id * 32 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 5);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(branch + tid * N, begin, N);
		}
		g.sync();
	}
#else // ifdef USING_PARALLEL_L_TREE
	if (tid < 32) {
		dev_set_ltree_addr(ltree_addr, tid);
		dev_l_tree(branch + tid * N, wots_pk + tid * WOTS_SIG_BYTES,
			   pub_seed, ltree_addr);
	}
	g.sync();
#endif // ifdef USING_PARALLEL_L_TREE

	for (int j = 0; j < opk_num; j++) {
		u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
		u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
		if (tid == ((leaf_idx_t >> 0) ^ 0x1))
			memcpy(opk_auth_path + j * 5 * N, branch + tid * N, N);
	}

	max_threads = 32;
	int p_height = 5; // 2^6

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			for (int j = 0; j < opk_num; j++) {
				u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
				if (tid == ((leaf_idx_t >> i) ^ 0x1))
					memcpy(opk_auth_path + i * N + j * 5 * N, branch + off, N);
			}
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			for (int j = 0; j < opk_num; j++)
				memcpy(auth_path + i * N + j * (SIG_BYTES + XMSS_MLEN),
				       opk_auth_path + i * N + 5 * N * j, N);

		for (int j = 0; j < opk_num; j++)
			memcpy(root + j * N, branch, N);
	}

} // dev_treehash_opk_parallel_5

__device__ void dev_treehash_opk_multi_parallel_5(u8 *one_root, u8 *origin_sm,
						  const u8 *sk_seed, const u8 *pub_seed,
						  u64 origin_idx, u32 iter, u32 num,
						  u32 subtree_addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	// const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	u8* auth_path;
	u8 *root;
	u64 idx_ex;
	u32 opk_num;

	// g.sync();

	u32 hh;

	if (iter > 3) // We can't handle 40 and 60 yet
		hh = 1 << 20;
	else
		hh = (1 << ((iter + 1) * TREE_HEIGHT));
	int count = (origin_idx + num - 1) / hh - origin_idx / hh;
	int exe, this_off;

	// this off 是最底层树的上一个截断点
	int k = tid / 32;

	this_off = (origin_idx + k * hh) / hh * hh - origin_idx;
	this_off = this_off >= 0 ? this_off : 0;

	if (k == 0) {
		exe = (origin_idx + hh) / hh * hh - origin_idx;
		exe = exe >= num ? num : exe;
	} else {
		exe = num - this_off < hh ? num - this_off : hh;
	}
	dev_set_tree_addr(subtree_addr,
			  (origin_idx + this_off) >> ((iter + 1) * TREE_HEIGHT));

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	auth_path = origin_sm + this_off * SM_BYTES
		    + INDEX_BYTES + N + WOTS_SIG_BYTES
		    + iter * (WOTS_SIG_BYTES + TREE_HEIGHT * N);
	root = one_root + this_off * N;
	idx_ex = origin_idx + this_off;
	int total = 1 << ((iter + 1) * TREE_HEIGHT);

	// addr not same, so we do not parallelize below
	if (tid < 32 * count + 32) {
		dev_set_ots_addr(ots_addr, tid % 32);
		dev_set_ltree_addr(ltree_addr, tid % 32);
		dev_gen_leaf_wots(branch + tid * N,
				  sk_seed, pub_seed, ltree_addr, ots_addr);
	}
	g.sync();

	opk_num = exe;

	if (tid < 32 * count + 32) {
		for (int j = 0; j < opk_num; j++) {
			u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
			u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
			if (tid - 32 * k == ((leaf_idx_t >> 0) ^ 0x1)) {
				memcpy(opk_auth_path + j * 5 * N + k * total * 5 * N,
				       branch + tid * N, N);
			}
		}
	}

	for (int i = 1, ii = 1; i <= 5; i++) {
		g.sync();
		int off = 32 * N * k + 2 * (tid - 32 * k) * ii * N;
		if (tid < (32 >> i) + 32 * k && tid >= 32 * k && tid < 32 * count + 32) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid - 32 * k);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			for (int j = 0; j < opk_num; j++) {
				u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
				if (tid - 32 * k == ((leaf_idx_t >> i) ^ 0x1))
					memcpy(opk_auth_path + i * N + j * 5 * N + k * total * 5 * N,
					       branch + off, N);
			}
		}
		ii *= 2;
	}

	if (tid == 32 * k && tid < 32 * count + 32) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			for (int j = 0; j < opk_num; j++)
				memcpy(auth_path + i * N + j * SM_BYTES,
				       opk_auth_path + i * N + 5 * N * j + k * total * 5 * N, N);


		for (int j = 0; j < opk_num; j++)
			memcpy(root + j * N, branch + k * 32 * N, N);
	}
} // dev_treehash_opk_multi_parallel_5
#endif // if TREE_HEIGHT == 5

#if TREE_HEIGHT == 10
__device__ void dev_treehash_parallel_10(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u32 leaf_idx, const u32 subtree_addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

#if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))
	const unsigned int tnum = gridDim.x * blockDim.x;
	int max_threads = 1;
	while (max_threads <= tnum / 2) max_threads *= 2;
	int ttid, id;

	if (tnum < max_threads) {
		if (tid == 0) printf("wrong thread size, tnum = %d\n", tnum);
		return;
	}
#endif // if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))

#ifdef USING_PARALLEL_WOTS_PKGEN
	#ifdef USING_COALESCED_ACCESS
	u8 buf[N + 32];
	max_threads = tnum - 5;
	if (tid < max_threads) {
		for (int i = tid; i < WOTS_LEN * 1024; i += max_threads) {
			dev_set_ots_addr(ots_addr, i / WOTS_LEN);
			dev_set_chain_addr(ots_addr, i % WOTS_LEN);
			dev_set_hash_addr(ots_addr, 0);
			dev_set_key_and_mask(ots_addr, 0);
			memcpy(buf, pub_seed, N);
			dev_addr_to_bytes(buf + N, ots_addr);
	#ifdef USING_LOCAL_MEMORY
			u8 temp[N];
			dev_prf_keygen(temp, buf, sk_seed);
			dev_gen_chain(temp, temp, 0, WOTS_W - 1, pub_seed, ots_addr);
			memcpy(wots_pk + i * N, temp, N);
	#else // ifdef USING_LOCAL_MEMORY
			dev_prf_keygen(wots_pk + i * N, buf, sk_seed);
			dev_gen_chain(wots_pk + i * N, wots_pk + i * N,
				      0, WOTS_W - 1, pub_seed, ots_addr);
	#endif // ifdef USING_LOCAL_MEMORY
		}
	}
	#else // ifdef USING_COALESCED_ACCESS
	if (tid < max_threads) {
		int para = max_threads / 1024;
		ttid = tid % 1024;
		id = tid / 1024;
		int q = WOTS_LEN / para;
		int r = WOTS_LEN % para;
		int local = q + ((id < r) ? 1 : 0);
		int offset = id * q + ((id < r) ? id : r);

		dev_set_ltree_addr(ltree_addr, ttid);
		dev_set_ots_addr(ots_addr, ttid);
		unsigned char buf[N + 32];

		dev_set_hash_addr(ots_addr, 0);
		dev_set_key_and_mask(ots_addr, 0);
		memcpy(buf, pub_seed, N);

		for (int i = offset; i < offset + local; i++) {
			dev_set_chain_addr(ots_addr, i);
			dev_addr_to_bytes(buf + N, ots_addr);
			dev_prf_keygen(wots_pk + ttid * WOTS_SIG_BYTES + i * N, buf, sk_seed);
		}
		for (int i = offset; i < offset + local; i++) {
			dev_set_chain_addr(ots_addr, i);
			dev_gen_chain(wots_pk + ttid * WOTS_SIG_BYTES + i * N,
				      wots_pk + ttid * WOTS_SIG_BYTES + i * N,
				      0, WOTS_W - 1, pub_seed, ots_addr);
		}
	}
	#endif // ifdef USING_COALESCED_ACCESS
	g.sync();
#else // ifdef USING_PARALLEL_WOTS_PKGEN
	if (tid < 1024) {
		dev_set_ots_addr(ots_addr, tid % 1024);
		dev_wots_pkgen(wots_pk + tid * WOTS_SIG_BYTES, sk_seed, pub_seed, ots_addr);
	}
#endif // ifdef USING_PARALLEL_WOTS_PKGEN

#ifdef USING_PARALLEL_L_TREE
	int scheme = 1;// 1: faster
	dev_set_ltree_addr(ltree_addr, tid % 1024);
	ttid = tid % 1024;
	id = tid / 1024;

	if (scheme == 1 && WOTS_LEN == 67) {
		max_threads = 1;
		while (max_threads <= (tnum - 1024) / 2) {
			max_threads *= 2;
		}

		if (tid < max_threads) {
			int job = 65536 / max_threads;
			u32 l = job;
			u32 height = 0;
			int offset_pk = ttid * WOTS_SIG_BYTES + id * job * N;
			while (l > 1) {
				int parent_nodes = l >> 1;
				for (int i = 0; i < parent_nodes; i++) {
					dev_set_tree_index(ltree_addr, i + id * (job >> 1 >> height));
					dev_thash_h(wots_pk + offset_pk + i * N,
						    wots_pk + offset_pk + i * 2 * N, pub_seed, ltree_addr);
				}
				l = l >> 1;
				height++;
				dev_set_tree_height(ltree_addr, height);
			}
		} else if (tid < max_threads + 1024) {
			u32 l = 3;
			u32 height = 0;
			int offset_pk = ttid * WOTS_SIG_BYTES + 64 * N;
			while (l > 1) {
				int parent_nodes = l >> 1;
				for (int i = 0; i < parent_nodes; i++) {
					dev_set_tree_index(ltree_addr, i + (32 >> height));
					dev_thash_h(wots_pk + offset_pk + i * N,
						    wots_pk + offset_pk + i * 2 * N, pub_seed, ltree_addr);
				}
				if (l & 1) {
					memcpy(wots_pk + offset_pk + (l >> 1) * N,
					       wots_pk + offset_pk + (l - 1) * N, N);
					l = (l >> 1) + 1;
				} else {
					l = l >> 1;
				}
				height++;
				dev_set_tree_height(ltree_addr, height);
			}
		}

		for (int i = 0; i < log(max_threads / 1024.0) / log(2.0); i++) {
			g.sync();
			int div = 65536 / max_threads * pow(2, i);
			int offset_pk = ttid * WOTS_SIG_BYTES + id * div * 2 * N;
			if (tid < 65536 / div / 2) {
				memcpy(wots_pk + offset_pk + N, wots_pk + offset_pk + div * N, N);
				dev_set_tree_index(ltree_addr, id);
				dev_set_tree_height(ltree_addr, log(65536.0 / max_threads) / log(2.0) + i);
				dev_thash_h(wots_pk + offset_pk,
					    wots_pk + offset_pk, pub_seed, ltree_addr);
			}
		}
		if (tid < 1024) {
			int offset_pk = ttid * WOTS_SIG_BYTES;
			memcpy(wots_pk + offset_pk + N, wots_pk + offset_pk + 64 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 6);
			dev_thash_h(wots_pk + offset_pk,
				    wots_pk + offset_pk, pub_seed, ltree_addr);
			memcpy(branch + tid * N, wots_pk + offset_pk, N);
		}
		g.sync();
	} else if (scheme == 2 && WOTS_LEN == 67) {
		int para, q, r, local, offset;
		u8 *begin;
		max_threads = 1;
		while (max_threads <= (tnum - 1024) / 2) {
			max_threads *= 2;
		}

		para = 8;
		q = 33 / para;
		r = 33 % para;
		local = q + ((id < r) ? 1 : 0);
		offset = id * q + ((id < r) ? id : r);

		int offset_pk = ttid * WOTS_SIG_BYTES;
		begin = wots_pk + offset_pk;

		if (id < para) {
			for (int i = offset; i < offset + local; i++) {
				dev_set_tree_index(ltree_addr, i);
				dev_set_tree_height(ltree_addr, 0);
				dev_thash_h(begin + i * 2 * N,
					    begin + i * 2 * N, pub_seed, ltree_addr);
			}
		}
		g.sync();

		para = 8;
		q = 17 / para;
		r = 17 % para;
		local = q + ((id < r) ? 1 : 0);
		offset = id * q + ((id < r) ? id : r);

		if (id < para) {
			for (int i = offset; i < offset + local; i++) {
				memcpy(begin + i * 4 * N + N, begin + i * 4 * N + 2 * N, N);
				dev_set_tree_index(ltree_addr, i);
				dev_set_tree_height(ltree_addr, 1);
				dev_thash_h(begin + i * 4 * N,
					    begin + i * 4 * N, pub_seed, ltree_addr);
			}
		}
		g.sync();

		if (id < para) {
			memcpy(begin + id * 8 * N + N, begin + id * 8 * N + 4 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 2);
			dev_thash_h(begin + id * 8 * N,
				    begin + id * 8 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 4) {
			memcpy(begin + id * 16 * N + N, begin + id * 16 * N + 8 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 3);
			dev_thash_h(begin + id * 16 * N,
				    begin + id * 16 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 2) {
			memcpy(begin + id * 32 * N + N, begin + id * 32 * N + 16 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 4);
			dev_thash_h(begin + id * 32 * N,
				    begin + id * 32 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 5);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(begin + N, begin + 64 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 6);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(branch + ttid * N, begin, N);
		}
		g.sync();
	} else if (WOTS_LEN == 51) {
		int para, q, r, local, offset;
		u8 *begin;
		max_threads = 1;
		while (max_threads <= (tnum - 1024) / 2) {
			max_threads *= 2;
		}

		para = max_threads / 1024;
		q = 25 / para;
		r = 25 % para;
		local = q + ((id < r) ? 1 : 0);
		offset = id * q + ((id < r) ? id : r);

		int offset_pk = ttid * WOTS_SIG_BYTES;
		begin = wots_pk + offset_pk;

		if (id < para) {
			for (int i = offset; i < offset + local; i++) {
				dev_set_tree_index(ltree_addr, i);
				dev_set_tree_height(ltree_addr, 0);
				dev_thash_h(begin + i * 2 * N,
					    begin + i * 2 * N, pub_seed, ltree_addr);
			}
		}
		g.sync();

		para = max_threads / 1024;
		q = 13 / para;
		r = 13 % para;
		local = q + ((id < r) ? 1 : 0);
		offset = id * q + ((id < r) ? id : r);

		if (id < para) {
			for (int i = offset; i < offset + local; i++) {
				memcpy(begin + i * 4 * N + N, begin + i * 4 * N + 2 * N, N);
				dev_set_tree_index(ltree_addr, i);
				dev_set_tree_height(ltree_addr, 1);
				dev_thash_h(begin + i * 4 * N,
					    begin + i * 4 * N, pub_seed, ltree_addr);
			}
		}
		g.sync();

		if (id < 6) {
			memcpy(begin + id * 8 * N + N, begin + id * 8 * N + 4 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 2);
			dev_thash_h(begin + id * 8 * N,
				    begin + id * 8 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 3) {
			memcpy(begin + id * 16 * N + N, begin + id * 16 * N + 8 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 3);
			dev_thash_h(begin + id * 16 * N,
				    begin + id * 16 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 2) {
			memcpy(begin + id * 32 * N + N, begin + id * 32 * N + 16 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 4);
			dev_thash_h(begin + id * 32 * N,
				    begin + id * 32 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 5);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(branch + ttid * N, begin, N);
		}
		g.sync();
	}
#else // ifdef USING_PARALLEL_L_TREE
	if (tid < 1024) {
		dev_set_ltree_addr(ltree_addr, tid % 1024);
		dev_l_tree(branch + tid * N, wots_pk + tid * WOTS_SIG_BYTES, pub_seed, ltree_addr);
	}
	g.sync();
#endif // ifdef USING_PARALLEL_L_TREE

	if (tid == ((leaf_idx >> 0) ^ 0x1))
		memcpy(one_auth_path, branch + tid * N, N);

	max_threads = 1024;
	int p_height = 10;

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			if (tid == ((leaf_idx >> i) ^ 0x1))
				memcpy(one_auth_path + i * N, branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(auth_path + i * N, one_auth_path + i * N, N);
		memcpy(root, branch, N);
	}

} // dev_treehash_parallel_10

__device__ void dev_treehash_opk_parallel_10(u8 *root, u8 *auth_path,
					     const u8 *sk_seed, const u8 *pub_seed,
					     u64 idx_ex, u32 iter,
					     const u32 subtree_addr[8], int opk_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

#if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))
	const unsigned int tnum = gridDim.x * blockDim.x;
	int max_threads, ttid, id;
#endif // if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))

#ifdef USING_PARALLEL_WOTS_PKGEN
	u8 buf[N + 32];
	max_threads = tnum - 5;
	if (tid < max_threads) {
		for (int i = tid; i < WOTS_LEN * 1024; i += max_threads) {
			dev_set_ots_addr(ots_addr, i / WOTS_LEN);
			dev_set_chain_addr(ots_addr, i % WOTS_LEN);
			dev_set_hash_addr(ots_addr, 0);
			dev_set_key_and_mask(ots_addr, 0);
			memcpy(buf, pub_seed, N);
			dev_addr_to_bytes(buf + N, ots_addr);
			u8 temp[N];
			dev_prf_keygen(temp, buf, sk_seed);
			dev_gen_chain(temp, temp, 0, WOTS_W - 1, pub_seed, ots_addr);
			memcpy(wots_pk + i * N, temp, N);
		}
	}
	g.sync();
#else // ifdef USING_PARALLEL_WOTS_PKGEN
	if (tid < 1024) {
		dev_set_ots_addr(ots_addr, tid);
		dev_wots_pkgen(wots_pk + tid * WOTS_SIG_BYTES, sk_seed, pub_seed, ots_addr);
	}
#endif // ifdef USING_PARALLEL_WOTS_PKGEN

#ifdef USING_PARALLEL_L_TREE
	ttid = tid % 1024;
	id = tid / 1024;
	dev_set_ltree_addr(ltree_addr, ttid);

	if (WOTS_LEN == 67) {
		max_threads = 8192; //2048, 4096, 8192

		if (tid < max_threads) {
			int job = 65536 / max_threads;
			u32 l = job;
			u32 height = 0;
			int offset_pk = ttid * WOTS_SIG_BYTES + id * job * N;
			while (l > 1) {
				int parent_nodes = l >> 1;
				for (int i = 0; i < parent_nodes; i++) {
					dev_set_tree_index(ltree_addr, i + id * (job >> 1 >> height));
					dev_thash_h(wots_pk + offset_pk + i * N,
						    wots_pk + offset_pk + i * 2 * N, pub_seed, ltree_addr);
				}
				l = l >> 1;
				height++;
				dev_set_tree_height(ltree_addr, height);
			}
		} else if (tid < max_threads + 1024) {
			u32 l = 3;
			u32 height = 0;
			int offset_pk = ttid * WOTS_SIG_BYTES + 64 * N;
			while (l > 1) {
				int parent_nodes = l >> 1;
				for (int i = 0; i < parent_nodes; i++) {
					dev_set_tree_index(ltree_addr, i + (32 >> height));
					dev_thash_h(wots_pk + offset_pk + i * N,
						    wots_pk + offset_pk + i * 2 * N, pub_seed, ltree_addr);
				}
				if (l & 1) {
					memcpy(wots_pk + offset_pk + (l >> 1) * N,
					       wots_pk + offset_pk + (l - 1) * N, N);
					l = (l >> 1) + 1;
				} else {
					l = l >> 1;
				}
				height++;
				dev_set_tree_height(ltree_addr, height);
			}
		}

		for (int i = 0; i < log(max_threads / 1024.0) / log(2.0); i++) {
			g.sync();
			int div = 65536 / max_threads * pow(2, i);
			int offset_pk = ttid * WOTS_SIG_BYTES + id * div * 2 * N;
			if (tid < 65536 / div / 2) {
				memcpy(wots_pk + offset_pk + N, wots_pk + offset_pk + div * N, N);
				dev_set_tree_index(ltree_addr, id);
				dev_set_tree_height(ltree_addr, log(65536.0 / max_threads) / log(2.0) + i);
				dev_thash_h(wots_pk + offset_pk,
					    wots_pk + offset_pk, pub_seed, ltree_addr);
			}
		}
		if (tid < 1024) {
			int offset_pk = ttid * WOTS_SIG_BYTES;
			memcpy(wots_pk + offset_pk + N, wots_pk + offset_pk + 64 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 6);
			dev_thash_h(wots_pk + offset_pk,
				    wots_pk + offset_pk, pub_seed, ltree_addr);
			memcpy(branch + tid * N, wots_pk + offset_pk, N);
		}
		g.sync();
	} else if (WOTS_LEN == 51) {
		int para, q, r, local, offset;
		u8 *begin;
		max_threads = 10240;

		para = max_threads / 1024;
		q = 25 / para;
		r = 25 % para;
		local = q + ((id < r) ? 1 : 0);
		offset = id * q + ((id < r) ? id : r);

		int offset_pk = ttid * WOTS_SIG_BYTES;
		begin = wots_pk + offset_pk;

		if (id < para) {
			for (int i = offset; i < offset + local; i++) {
				dev_set_tree_index(ltree_addr, i);
				dev_set_tree_height(ltree_addr, 0);
				dev_thash_h(begin + i * 2 * N,
					    begin + i * 2 * N, pub_seed, ltree_addr);
			}
		}
		g.sync();

		para = max_threads / 1024;
		q = 13 / para;
		r = 13 % para;
		local = q + ((id < r) ? 1 : 0);
		offset = id * q + ((id < r) ? id : r);

		if (id < para) {
			for (int i = offset; i < offset + local; i++) {
				memcpy(begin + i * 4 * N + N, begin + i * 4 * N + 2 * N, N);
				dev_set_tree_index(ltree_addr, i);
				dev_set_tree_height(ltree_addr, 1);
				dev_thash_h(begin + i * 4 * N,
					    begin + i * 4 * N, pub_seed, ltree_addr);
			}
		}
		g.sync();

		if (id < 6) {
			memcpy(begin + id * 8 * N + N, begin + id * 8 * N + 4 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 2);
			dev_thash_h(begin + id * 8 * N,
				    begin + id * 8 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 3) {
			memcpy(begin + id * 16 * N + N, begin + id * 16 * N + 8 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 3);
			dev_thash_h(begin + id * 16 * N,
				    begin + id * 16 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id < 2) {
			memcpy(begin + id * 32 * N + N, begin + id * 32 * N + 16 * N, N);
			dev_set_tree_index(ltree_addr, id);
			dev_set_tree_height(ltree_addr, 4);
			dev_thash_h(begin + id * 32 * N,
				    begin + id * 32 * N, pub_seed, ltree_addr);
		}
		g.sync();
		if (id == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 5);
			dev_thash_h(begin, begin, pub_seed, ltree_addr);
			memcpy(branch + ttid * N, begin, N);
		}
		g.sync();
	}
#else // ifdef USING_PARALLEL_L_TREE
	if (tid < 1024) {
		dev_set_ltree_addr(ltree_addr, tid);
		dev_l_tree(branch + tid * N, wots_pk + tid * WOTS_SIG_BYTES,
			   pub_seed, ltree_addr);
	}
	g.sync();
#endif // ifdef USING_PARALLEL_L_TREE

	for (int j = 0; j < opk_num; j++) {
		u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
		u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
		if (tid == ((leaf_idx_t >> 0) ^ 0x1))
			memcpy(opk_auth_path + j * 20 * N, branch + tid * N, N);
	}

	max_threads = 1024;
	int p_height = 10;

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			for (int j = 0; j < opk_num; j++) {
				u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
				if (tid == ((leaf_idx_t >> i) ^ 0x1))
					memcpy(opk_auth_path + i * N + j * 20 * N,
					       branch + off, N);
			}
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			for (int j = 0; j < opk_num; j++)
				memcpy(auth_path + i * N + j * (SIG_BYTES + XMSS_MLEN),
				       opk_auth_path + i * N + 20 * N * j, N);

		for (int j = 0; j < opk_num; j++)
			memcpy(root + j * N, branch, N);
	}

} // dev_treehash_opk_parallel_10

__device__ void dev_treehash_opk_multi_parallel_10(u8 *one_root, u8 *origin_sm,
						   const u8 *sk_seed, const u8 *pub_seed,
						   u64 origin_idx, u32 iter, u32 num,
						   u32 subtree_addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	// const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	u8* auth_path;
	u8 *root;
	u64 idx_ex;
	u32 opk_num;

	// g.sync();

	u32 hh;
	int lead_num = 1 << TREE_HEIGHT;

	if (iter > 1) // We can't handle 40 and 60 yet
		hh = 1 << 20;
	else
		hh = (1 << ((iter + 1) * TREE_HEIGHT));
	int count = (origin_idx + num - 1) / hh - origin_idx / hh;
	int exe, this_off;

	// this off 是最底层树的上一个截断点
	int k = tid / lead_num;

	this_off = (origin_idx + k * hh) / hh * hh - origin_idx;
	this_off = this_off >= 0 ? this_off : 0;

	if (k == 0) {
		exe = (origin_idx + hh) / hh * hh - origin_idx;
		exe = exe >= num ? num : exe;
	} else {
		exe = num - this_off < hh ? num - this_off : hh;
	}
	dev_set_tree_addr(subtree_addr,
			  (origin_idx + this_off) >> ((iter + 1) * TREE_HEIGHT));

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	auth_path = origin_sm + this_off * SM_BYTES
		    + INDEX_BYTES + N + WOTS_SIG_BYTES
		    + iter * (WOTS_SIG_BYTES + TREE_HEIGHT * N);
	root = one_root + this_off * N;
	idx_ex = origin_idx + this_off;
	int total = 1 << ((iter + 1) * TREE_HEIGHT);

	// addr not same, so we do not parallelize below
	if (tid < lead_num * count + lead_num) {
		dev_set_ots_addr(ots_addr, tid % lead_num);
		dev_set_ltree_addr(ltree_addr, tid % lead_num);
		dev_gen_leaf_wots(branch + tid * N,
				  sk_seed, pub_seed, ltree_addr, ots_addr);
	}
	g.sync();

	opk_num = exe;

	if (tid < lead_num * count + lead_num) {
		for (int j = 0; j < opk_num; j++) {
			u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
			u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
			if (tid - lead_num * k == ((leaf_idx_t >> 0) ^ 0x1)) {
				memcpy(opk_auth_path + j * 10 * N + k * total * 10 * N,
				       branch + tid * N, N);
			}
		}
	}

	for (int i = 1, ii = 1; i <= TREE_HEIGHT; i++) {
		g.sync();
		int off = lead_num * N * k + 2 * (tid - lead_num * k) * ii * N;
		if (tid < (lead_num >> i) + lead_num * k
		    && tid >= lead_num * k
		    && tid < lead_num * count + lead_num) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid - lead_num * k);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			for (int j = 0; j < opk_num; j++) {
				u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
				if (tid - lead_num * k == ((leaf_idx_t >> i) ^ 0x1))
					memcpy(opk_auth_path + i * N + j * 10 * N + k * total * 10 * N,
					       branch + off, N);
			}
		}
		ii *= 2;
	}

	if (tid == lead_num * k && tid < lead_num * count + lead_num) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			for (int j = 0; j < opk_num; j++)
				memcpy(auth_path + i * N + j * SM_BYTES,
				       opk_auth_path + i * N + 10 * N * j + k * total * 10 * N, N);


		for (int j = 0; j < opk_num; j++)
			memcpy(root + j * N, branch + k * lead_num * N, N);
	}
} // dev_treehash_opk_multi_parallel_5

// level 1 parallelization
__device__ void dev_treehash_parallel_10_1(u8 *root, u8 *auth_path,
					   const u8 *sk_seed, const u8 *pub_seed,
					   u32 leaf_idx, const u32 subtree_addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 heights[TREE_HEIGHT + 1];
	u32 offset = 0;

	/* The subtree has at most 2^20 leafs, so u32 suffices. */
	u32 idx;
	u32 tree_idx;

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

	u32 lastnode;

	lastnode = (1 << TREE_HEIGHT); // 2^h

	int max_threads = 64;
	int p_height = log(max_threads * 1.0) / log(2.0);// 14;

	if (tnum < max_threads) {
		if (tid == 0) printf("error in thread size\n");
		return;
	}

	if (tid < max_threads) {
		for (idx = tid * lastnode / max_threads;
		     idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + offset * N,
					  sk_seed, pub_seed, ltree_addr, ots_addr);
			offset++;
			heights[offset - 1] = 0;

			if ((leaf_idx ^ 0x1) == idx) {
				memcpy(one_auth_path, stack + (offset - 1) * N, N);
			}

			while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
				tree_idx = (idx >> (heights[offset - 1] + 1));

				dev_set_tree_height(node_addr, heights[offset - 1]);
				dev_set_tree_index(node_addr, tree_idx);
				dev_thash_h(stack + (offset - 2) * N,
					    stack + (offset - 2) * N, pub_seed, node_addr);
				offset--;
				/* Note that the top-most node is now one layer higher. */
				heights[offset - 1]++;

				if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
					memcpy(one_auth_path + heights[offset - 1] * N,
					       stack + (offset - 1) * N, N);
				}
			}
		}
		if (tid == ((leaf_idx >> heights[offset - 1]) ^ 0x1))
			memcpy(one_auth_path + (TREE_HEIGHT - p_height) * N, stack, N);
		memcpy(branch + tid * N, stack, N);
	}

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, TREE_HEIGHT - 1 - p_height + i);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			if (tid == ((leaf_idx >> (i + TREE_HEIGHT - p_height)) ^ 0x1))
				memcpy(one_auth_path + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++) {
			memcpy(auth_path + i * N, one_auth_path + i * N, N);
			// for (int j = 0; j < N; j++)
			// 	printf("%02x%s", one_auth_path[i * N + j], ((j % 4) == 3) ? " " : "");
			// printf("\n");
		}
		memcpy(root, branch, N);
	}

} // dev_treehash_parallel_10_1

__device__ void dev_treehash_opk_parallel_10_1(u8 *root, u8 *auth_path,
					       const u8 *sk_seed, const u8 *pub_seed,
					       u64 idx_ex, u32 iter,
					       const u32 subtree_addr[8], int opk_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 heights[TREE_HEIGHT + 1];
	u32 offset = 0;

	/* The subtree has at most 2^20 leafs, so u32 suffices. */
	u32 idx;
	u32 tree_idx;

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

	u32 lastnode;

	lastnode = (1 << TREE_HEIGHT); // 2^h

	int max_threads = 1024;
	int p_height = log(max_threads * 1.0) / log(2.0);

	if (tnum < max_threads || max_threads > 1024) {
		if (tid == 0) printf("error in thread size\n");
		return;
	}

	u32 leaf_idx_a[1024]; // provided that the maximum is 1024

	for (int i = 0; i < opk_num; i++) {
		u32 ll = ((idx_ex + i) >> (iter * TREE_HEIGHT));
		leaf_idx_a[i] = (ll & ((1 << TREE_HEIGHT) - 1));
	}

	if (tid < max_threads) {
		for (idx = tid * lastnode / max_threads;
		     idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + offset * N,
					  sk_seed, pub_seed, ltree_addr, ots_addr);
			offset++;
			heights[offset - 1] = 0;

			for (int j = 0; j < opk_num; j++) {
				if ((leaf_idx_a[j] ^ 0x1) == idx) {
					memcpy(opk_auth_path + j * 20 * N, stack + (offset - 1) * N, N);
				}
			}

			while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
				tree_idx = (idx >> (heights[offset - 1] + 1));

				dev_set_tree_height(node_addr, heights[offset - 1]);
				dev_set_tree_index(node_addr, tree_idx);
				dev_thash_h(stack + (offset - 2) * N,
					    stack + (offset - 2) * N, pub_seed, node_addr);
				offset--;
				/* Note that the top-most node is now one layer higher. */
				heights[offset - 1]++;


				for (int j = 0; j < opk_num; j++) {
					if (((leaf_idx_a[j] >> heights[offset - 1]) ^ 0x1) == tree_idx) {
						memcpy(opk_auth_path + j * 20 * N + heights[offset - 1] * N,
						       stack + (offset - 1) * N, N);
					}
				}
			}
		}
		for (int j = 0; j < opk_num; j++) {
			if (tid == ((leaf_idx_a[j] >> heights[offset - 1]) ^ 0x1))
				memcpy(opk_auth_path + j * 20 * N + (TREE_HEIGHT - p_height) * N, stack, N);
		}

		memcpy(branch + tid * N, stack, N);
	}

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, TREE_HEIGHT - 1 - p_height + i);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			for (int j = 0; j < opk_num; j++) {
				if (tid == ((leaf_idx_a[j] >> (i + TREE_HEIGHT - p_height)) ^ 0x1))
					memcpy(opk_auth_path + j * 20 * N + (TREE_HEIGHT - p_height + i) * N,
					       branch + off, N);
			}
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int j = 0; j < opk_num; j++)
			for (int i = 0; i < TREE_HEIGHT; i++)
				memcpy(auth_path + i * N + j * (SIG_BYTES + XMSS_MLEN),
				       opk_auth_path + i * N + 20 * N * j, N);

		for (int j = 0; j < opk_num; j++)
			memcpy(root + j * N, branch, N);

	}

} // dev_treehash_opk_parallel_10_1

#endif // if TREE_HEIGHT == 10

__device__ void dev_treehash_parallel_general(u8 *root, u8 *auth_path,
					      const u8 *sk_seed, const u8 *pub_seed,
					      u32 leaf_idx, const u32 subtree_addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}

	if (tnum < max_threads) {
		if (tid == 0) printf("error in thread size\n");
		return;
	}
	// if (tid == 0)
	// 	printf("max_threads = %d %d\n", max_threads, p_height);

	int i_size = (1 << (TREE_HEIGHT - p_height));

	for (int iter = 0; iter < i_size; iter++) {
		g.sync();
		if (tid < tnum) {
			u8 buf[N + 32];
			for (int i = tid; i < WOTS_LEN * max_threads; i += tnum) {
				dev_set_ots_addr(ots_addr, i / WOTS_LEN + iter * max_threads);
				dev_set_chain_addr(ots_addr, i % WOTS_LEN);
				dev_set_hash_addr(ots_addr, 0);
				dev_set_key_and_mask(ots_addr, 0);
				memcpy(buf, pub_seed, N);
				dev_addr_to_bytes(buf + N, ots_addr);
			#ifdef USING_LOCAL_MEMORY
				u8 temp[N];
				dev_prf_keygen(temp, buf, sk_seed);
				dev_gen_chain(temp, temp, 0, WOTS_W - 1, pub_seed, ots_addr);
				memcpy(wots_pk + i * N, temp, N);
			#else // ifdef USING_LOCAL_MEMORY
				dev_prf_keygen(wots_pk + i * N, buf, sk_seed);
				dev_gen_chain(wots_pk + i * N, wots_pk + i * N,
					      0, WOTS_W - 1, pub_seed, ots_addr);
			#endif // ifdef USING_LOCAL_MEMORY
			}
		}
		g.sync();
		if (tid < max_threads) {
			dev_set_ltree_addr(ltree_addr, tid + iter * max_threads);
		#ifdef USING_LOCAL_MEMORY
			u8 temp[WOTS_SIG_BYTES];
			memcpy(temp, wots_pk + tid * WOTS_SIG_BYTES, WOTS_SIG_BYTES);
			dev_l_tree(branch + tid * N, temp, pub_seed, ltree_addr);
		#else // ifdef USING_LOCAL_MEMORY
			dev_l_tree(branch + tid * N, wots_pk + tid * WOTS_SIG_BYTES,
				   pub_seed, ltree_addr);
		#endif // ifdef USING_LOCAL_MEMORY
		}

		g.sync();
		if (iter == leaf_idx / max_threads
		    && tid == ((leaf_idx % max_threads) ^ 0x1))
			memcpy(one_auth_path, branch + tid * N, N);

		for (int i = 1, ii = 1; i <= p_height; i++) {
			g.sync();
			int off = 2 * tid * ii * N;
			if (tid < (max_threads >> i)) {
				dev_set_tree_height(node_addr, i - 1);
				dev_set_tree_index(node_addr, tid + iter * (max_threads >> i));
				memcpy(branch + off + N, branch + off + ii * N, N);
				dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
				// which subtree and which leaf of subtree
				if (iter == leaf_idx / max_threads &&
				    tid == (((leaf_idx % max_threads) >> i) ^ 0x1))
					memcpy(one_auth_path + i * N, branch + off, N);
			}
			ii *= 2;
		}
		if (tid == 0) memcpy(&c_topnode[iter * N], branch, N);
		if (tid == 0 && iter == ((leaf_idx >> p_height) ^ 0x1))
			memcpy(one_auth_path + p_height * N, branch, N);
	}

	for (int i = p_height + 1, ii = 1; i <= TREE_HEIGHT; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (i_size >> (i - p_height))) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(c_topnode + off + N, c_topnode + off + ii * N, N);
			dev_thash_h(c_topnode + off, c_topnode + off, pub_seed, node_addr);
			if (tid == ((leaf_idx >> i) ^ 0x1))
				memcpy(one_auth_path + i * N, c_topnode + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++) {
			memcpy(auth_path + i * N, one_auth_path + i * N, N);
		}
		// for (int i = 0; i < TREE_HEIGHT; i++)
		// 	printf("%02x ", auth_path[i * N]);
		// printf("\n");
		memcpy(root, c_topnode, N);
	}

} // dev_treehash_parallel_general

__device__ void dev_treehash_opk_general(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u64 idx_ex, u32 iter,
					 const u32 subtree_addr[8], int opk_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

	int max_threads = 1;
	int p_height = 0;
	u64 ll;
	u32 leaf_idx;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}

	if (tnum < max_threads) {
		if (tid == 0) printf("error in thread size\n");
		return;
	}
	// if (tid == 0)
	// 	printf("max_threads = %d %d\n", max_threads, p_height);

	int i_size = (1 << (TREE_HEIGHT - p_height));

	for (int it = 0; it < i_size; it++) {
		g.sync();
		if (tid < tnum) {
			u8 buf[N + 32];
			for (int i = tid; i < WOTS_LEN * max_threads; i += tnum) {
				dev_set_ots_addr(ots_addr, i / WOTS_LEN + it * max_threads);
				dev_set_chain_addr(ots_addr, i % WOTS_LEN);
				dev_set_hash_addr(ots_addr, 0);
				dev_set_key_and_mask(ots_addr, 0);
				memcpy(buf, pub_seed, N);
				dev_addr_to_bytes(buf + N, ots_addr);
			#ifdef USING_LOCAL_MEMORY
				u8 temp[N];
				dev_prf_keygen(temp, buf, sk_seed);
				dev_gen_chain(temp, temp, 0, WOTS_W - 1, pub_seed, ots_addr);
				memcpy(wots_pk + i * N, temp, N);
			#else // ifdef USING_LOCAL_MEMORY
				dev_prf_keygen(wots_pk + i * N, buf, sk_seed);
				dev_gen_chain(wots_pk + i * N, wots_pk + i * N,
					      0, WOTS_W - 1, pub_seed, ots_addr);
			#endif // ifdef USING_LOCAL_MEMORY
			}
		}
		g.sync();
		if (tid < max_threads) {
			dev_set_ltree_addr(ltree_addr, tid + it * max_threads);
		#ifdef USING_LOCAL_MEMORY
			u8 temp[WOTS_SIG_BYTES];
			memcpy(temp, wots_pk + tid * WOTS_SIG_BYTES, WOTS_SIG_BYTES);
			dev_l_tree(branch + tid * N, temp, pub_seed, ltree_addr);
		#else // ifdef USING_LOCAL_MEMORY
			dev_l_tree(branch + tid * N, wots_pk + tid * WOTS_SIG_BYTES,
				   pub_seed, ltree_addr);
		#endif // ifdef USING_LOCAL_MEMORY
		}

		g.sync();
		for (int j = 0; j < opk_num; j++) {
			ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
			leaf_idx = (ll & ((1 << TREE_HEIGHT) - 1));
			if (it == leaf_idx / max_threads
			    && tid == ((leaf_idx % max_threads) ^ 0x1))
				memcpy(opk_auth_path + j * 20 * N, branch + tid * N, N);
		}

		for (int i = 1, ii = 1; i <= p_height; i++) {
			g.sync();
			int off = 2 * tid * ii * N;
			if (tid < (max_threads >> i)) {
				dev_set_tree_height(node_addr, i - 1);
				dev_set_tree_index(node_addr, tid + it * (max_threads >> i));
				memcpy(branch + off + N, branch + off + ii * N, N);
				dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
				// which subtree and which leaf of subtree
				for (int j = 0; j < opk_num; j++) {
					ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
					leaf_idx = (ll & ((1 << TREE_HEIGHT) - 1));
					if (it == leaf_idx / max_threads &&
					    tid == (((leaf_idx % max_threads) >> i) ^ 0x1))
						memcpy(opk_auth_path + j * 20 * N + i * N, branch + off, N);
				}
			}
			ii *= 2;
		}
		if (tid == 0) memcpy(&c_topnode[it * N], branch, N);
		for (int j = 0; j < opk_num; j++) {
			ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
			leaf_idx = (ll & ((1 << TREE_HEIGHT) - 1));
			if (tid == 0 && it == ((leaf_idx >> p_height) ^ 0x1))
				memcpy(opk_auth_path + j * 20 * N + p_height * N, branch, N);
		}
	}

	for (int i = p_height + 1, ii = 1; i <= TREE_HEIGHT; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (i_size >> (i - p_height))) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(c_topnode + off + N, c_topnode + off + ii * N, N);
			dev_thash_h(c_topnode + off, c_topnode + off, pub_seed, node_addr);
			for (int j = 0; j < opk_num; j++) {
				ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				leaf_idx = (ll & ((1 << TREE_HEIGHT) - 1));
				if (tid == ((leaf_idx >> i) ^ 0x1))
					memcpy(opk_auth_path + j * 20 * N + i * N, c_topnode + off, N);
			}
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			for (int j = 0; j < opk_num; j++)
				memcpy(auth_path + i * N + j * SM_BYTES,
				       opk_auth_path + i * N + 20 * N * j, N);
		// for (int i = 0; i < TREE_HEIGHT; i++)
		// 	printf("%02x ", opk_auth_path[i * N]);
		// printf("\n");
		// for (int i = 0; i < TREE_HEIGHT; i++)
		// 	printf("%02x ", opk_auth_path[i * N + 20 * N]);
		// printf("\n");
		for (size_t j = 0; j < opk_num; j++)
			memcpy(root + j * N, c_topnode, N);
	}
} // dev_treehash_opk_parallel_16

#if TREE_HEIGHT == 16
#ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_parallel_16(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u32 leaf_idx, const u32 subtree_addr[8])
{
	dev_treehash_parallel_general(root, auth_path, sk_seed, pub_seed,
				      leaf_idx, subtree_addr);
}       // dev_treehash_parallel_16
#else   // ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_parallel_16(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u32 leaf_idx, const u32 subtree_addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 heights[TREE_HEIGHT + 1];
	u32 offset = 0;

	/* The subtree has at most 2^20 leafs, so u32 suffices. */
	u32 idx;
	u32 tree_idx;

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

	u32 lastnode = (1 << TREE_HEIGHT); // 2^h

	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}

	if (tnum < max_threads) {
		if (tid == 0) printf("error in thread size\n");
		return;
	}

	if (tid < max_threads) {
		for (idx = tid * lastnode / max_threads;
		     idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + offset * N,
					  sk_seed, pub_seed, ltree_addr, ots_addr);
			offset++;
			heights[offset - 1] = 0;

			if ((leaf_idx ^ 0x1) == idx) {
				memcpy(one_auth_path, stack + (offset - 1) * N, N);
			}

			while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
				tree_idx = (idx >> (heights[offset - 1] + 1));

				dev_set_tree_height(node_addr, heights[offset - 1]);
				dev_set_tree_index(node_addr, tree_idx);
				dev_thash_h(stack + (offset - 2) * N,
					    stack + (offset - 2) * N, pub_seed, node_addr);
				offset--;
				/* Note that the top-most node is now one layer higher. */
				heights[offset - 1]++;

				if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
					memcpy(one_auth_path + heights[offset - 1] * N,
					       stack + (offset - 1) * N, N);
				}
			}
		}
		if (tid == ((leaf_idx >> heights[offset - 1]) ^ 0x1))
			memcpy(one_auth_path + (TREE_HEIGHT - p_height) * N, stack, N);
		memcpy(branch + tid * N, stack, N);
	}

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, TREE_HEIGHT - 1 - p_height + i);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			if (tid == ((leaf_idx >> (i + TREE_HEIGHT - p_height)) ^ 0x1))
				memcpy(one_auth_path + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++) {
			memcpy(auth_path + i * N, one_auth_path + i * N, N);
		}
		// for (int i = 0; i < TREE_HEIGHT; i++)
		// 	printf("%02x ", auth_path[i * N]);
		// printf("\n");
		memcpy(root, branch, N);
	}

} // dev_treehash_parallel_16
#endif // ifdef USING_COALESCED_ACCESS

#ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_opk_parallel_16(u8 *root, u8 *auth_path,
					     const u8 *sk_seed, const u8 *pub_seed,
					     u64 idx_ex, u32 iter,
					     const u32 subtree_addr[8], int opk_num)
{
	dev_treehash_opk_general(root, auth_path, sk_seed, pub_seed,
				 idx_ex, iter, subtree_addr, opk_num);
} // dev_treehash_opk_parallel_16
#else   // ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_opk_parallel_16(u8 *root, u8 *auth_path,
					     const u8 *sk_seed, const u8 *pub_seed,
					     u64 idx_ex, u32 iter,
					     const u32 subtree_addr[8], int opk_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 heights[TREE_HEIGHT + 1];
	u32 offset = 0;

	/* The subtree has at most 2^20 leafs, so u32 suffices. */
	u32 idx;
	u32 tree_idx;

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

	u32 lastnode = (1 << TREE_HEIGHT); // 2^h

	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}

	if (tnum < max_threads) {
		if (tid == 0) printf("error in thread size\n");
		return;
	}

	if (tid < max_threads) {
		for (idx = tid * lastnode / max_threads;
		     idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + offset * N,
					  sk_seed, pub_seed, ltree_addr, ots_addr);
			offset++;
			heights[offset - 1] = 0;

			for (int j = 0; j < opk_num; j++) {
				u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
				if ((leaf_idx_t ^ 0x1) == idx) {
					memcpy(opk_auth_path + 20 * N * j, stack + (offset - 1) * N, N);
				}
			}

			while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
				tree_idx = (idx >> (heights[offset - 1] + 1));

				dev_set_tree_height(node_addr, heights[offset - 1]);
				dev_set_tree_index(node_addr, tree_idx);
				dev_thash_h(stack + (offset - 2) * N,
					    stack + (offset - 2) * N, pub_seed, node_addr);
				offset--;
				/* Note that the top-most node is now one layer higher. */
				heights[offset - 1]++;

				for (int j = 0; j < opk_num; j++) {
					u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
					u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
					if (((leaf_idx_t >> heights[offset - 1]) ^ 0x1) == tree_idx) {
						memcpy(opk_auth_path + heights[offset - 1] * N + 20 * N * j,
						       stack + (offset - 1) * N, N);
					}
				}
			}
		}
		for (int j = 0; j < opk_num; j++) {
			u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
			u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
			if (tid == ((leaf_idx_t >> heights[offset - 1]) ^ 0x1))
				memcpy(opk_auth_path + (TREE_HEIGHT - p_height) * N + 20 * N * j, stack, N);
		}
		memcpy(branch + tid * N, stack, N);
	}

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, TREE_HEIGHT - 1 - p_height + i);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			for (int j = 0; j < opk_num; j++) {
				u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
				if (tid == ((leaf_idx_t >> (i + TREE_HEIGHT - p_height)) ^ 0x1))
					memcpy(opk_auth_path + (TREE_HEIGHT - p_height + i) * N + j * 20 * N,
					       branch + off, N);
			}
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			for (int j = 0; j < opk_num; j++)
				memcpy(auth_path + i * N + j * (SIG_BYTES + XMSS_MLEN),
				       opk_auth_path + i * N + 20 * N * j, N);

		for (size_t j = 0; j < opk_num; j++)
			memcpy(root + j * N, branch, N);
	}

} // dev_treehash_opk_parallel_16
#endif // ifdef USING_COALESCED_ACCESS
#endif // if TREE_HEIGHT == 16

#if TREE_HEIGHT == 20
#ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_parallel_20(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u32 leaf_idx, const u32 subtree_addr[8])
{
	dev_treehash_parallel_general(root, auth_path, sk_seed, pub_seed,
				      leaf_idx, subtree_addr);
}       // dev_treehash_parallel_20
#else   // ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_parallel_20(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u32 leaf_idx, const u32 subtree_addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 heights[TREE_HEIGHT + 1];
	u32 offset = 0;

	/* The subtree has at most 2^20 leafs, so u32 suffices. */
	u64 idx;
	u64 tree_idx;

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

	u64 lastnode = (1 << TREE_HEIGHT); // 2^h

	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}

	if (tnum < max_threads) {
		if (tid == 0) printf("error in thread size\n");
		return;
	}

	if (tid < max_threads) {
		for (idx = tid * lastnode / max_threads;
		     idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + offset * N,
					  sk_seed, pub_seed, ltree_addr, ots_addr);
			offset++;
			heights[offset - 1] = 0;

			if ((leaf_idx ^ 0x1) == idx) {
				memcpy(one_auth_path, stack + (offset - 1) * N, N);
			}

			while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
				tree_idx = (idx >> (heights[offset - 1] + 1));

				dev_set_tree_height(node_addr, heights[offset - 1]);
				dev_set_tree_index(node_addr, tree_idx);
				dev_thash_h(stack + (offset - 2) * N,
					    stack + (offset - 2) * N, pub_seed, node_addr);
				offset--;
				/* Note that the top-most node is now one layer higher. */
				heights[offset - 1]++;

				if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
					memcpy(one_auth_path + heights[offset - 1] * N,
					       stack + (offset - 1) * N, N);
				}
			}
		}
		if (tid == ((leaf_idx >> heights[offset - 1]) ^ 0x1))
			memcpy(one_auth_path + (TREE_HEIGHT - p_height) * N, stack, N);
		memcpy(branch + tid * N, stack, N);
	}

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, TREE_HEIGHT - 1 - p_height + i);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			if (tid == ((leaf_idx >> (i + TREE_HEIGHT - p_height)) ^ 0x1))
				memcpy(one_auth_path + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++) {
			memcpy(auth_path + i * N, one_auth_path + i * N, N);
			// for (int j = 0; j < N; j++)
			// 	printf("%02x%s", one_auth_path[i * N + j], ((j % 4) == 3) ? " " : "");
			// printf("\n");
		}
		memcpy(root, branch, N);
	}

} // dev_treehash_parallel_20
#endif // ifdef USING_COALESCED_ACCESS

#ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_opk_parallel_20(u8 *root, u8 *auth_path,
					     const u8 *sk_seed, const u8 *pub_seed,
					     u64 idx_ex, u32 iter,
					     const u32 subtree_addr[8], int opk_num)
{
	dev_treehash_opk_general(root, auth_path, sk_seed, pub_seed,
				 idx_ex, iter, subtree_addr, opk_num);
}       // dev_treehash_parallel_20
#else   // ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_opk_parallel_20(u8 *root, u8 *auth_path,
					     const u8 *sk_seed, const u8 *pub_seed,
					     u64 idx_ex, u32 iter,
					     const u32 subtree_addr[8], int opk_num)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 heights[TREE_HEIGHT + 1];
	u32 offset = 0;

	/* The subtree has at most 2^20 leafs, so u32 suffices. */
	u64 idx;
	u64 tree_idx;

	/* We need all three types of addresses in parallel. */
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	/* Select the required subtree. */
	dev_copy_subtree_addr(ots_addr, subtree_addr);
	dev_copy_subtree_addr(ltree_addr, subtree_addr);
	dev_copy_subtree_addr(node_addr, subtree_addr);

	dev_set_type(ots_addr, XMSS_ADDR_TYPE_OTS);
	dev_set_type(ltree_addr, XMSS_ADDR_TYPE_LTREE);
	dev_set_type(node_addr, XMSS_ADDR_TYPE_HASHTREE);

	// g.sync();

	u64 lastnode = (1 << TREE_HEIGHT); // 2^h

	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}

	if (tnum < max_threads) {
		if (tid == 0) printf("error in thread size\n");
		return;
	}

	if (tid < max_threads) {
		for (idx = tid * lastnode / max_threads;
		     idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + offset * N,
					  sk_seed, pub_seed, ltree_addr, ots_addr);
			offset++;
			heights[offset - 1] = 0;

			for (size_t j = 0; j < opk_num; j++) {
				u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
				if ((leaf_idx_t ^ 0x1) == idx) {
					memcpy(opk_auth_path + 20 * N * j, stack + (offset - 1) * N, N);
				}
			}

			while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
				tree_idx = (idx >> (heights[offset - 1] + 1));

				dev_set_tree_height(node_addr, heights[offset - 1]);
				dev_set_tree_index(node_addr, tree_idx);
				dev_thash_h(stack + (offset - 2) * N,
					    stack + (offset - 2) * N, pub_seed, node_addr);
				offset--;
				/* Note that the top-most node is now one layer higher. */
				heights[offset - 1]++;

				for (size_t j = 0; j < opk_num; j++) {
					u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
					u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
					if (((leaf_idx_t >> heights[offset - 1]) ^ 0x1) == tree_idx) {
						memcpy(opk_auth_path + heights[offset - 1] * N + 20 * N * j,
						       stack + (offset - 1) * N, N);
					}
				}
			}
		}
		for (size_t j = 0; j < opk_num; j++) {
			u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
			u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
			if (tid == ((leaf_idx_t >> heights[offset - 1]) ^ 0x1))
				memcpy(opk_auth_path + (TREE_HEIGHT - p_height) * N + 20 * N * j, stack, N);
		}
		memcpy(branch + tid * N, stack, N);
	}

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, TREE_HEIGHT - 1 - p_height + i);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			for (size_t j = 0; j < opk_num; j++) {
				u64 ll = ((idx_ex + j) >> (iter * TREE_HEIGHT));
				u32 leaf_idx_t = (ll & ((1 << TREE_HEIGHT) - 1));
				if (tid == ((leaf_idx_t >> (i + TREE_HEIGHT - p_height)) ^ 0x1))
					memcpy(opk_auth_path + (TREE_HEIGHT - p_height + i) * N + 20 * N * j,
					       branch + off, N);
			}
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			for (int j = 0; j < opk_num; j++)
				memcpy(auth_path + i * N + j * (SIG_BYTES + XMSS_MLEN),
				       opk_auth_path + i * N + 20 * N * j, N);

		for (size_t j = 0; j < opk_num; j++)
			memcpy(root + j * N, branch, N);
	}

} // dev_treehash_parallel_20
#endif // ifdef USING_COALESCED_ACCESS
#endif // if TREE_HEIGHT == 20

#endif // ifndef USING_BDS
