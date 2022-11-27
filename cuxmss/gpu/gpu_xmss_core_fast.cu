
#include "gpu_utils.h"
#include "gpu_wots.h"
#include "gpu_hash.h"
#include "gpu_hash_address.h"
#include "gpu_xmss_commons.h"

#include "gpu_xmss_core_fast.h"

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

__device__ u8 branch[LEAD_NUM * N];

__device__ u8 wots_pk[WOTS_SIG_BYTES * 65536];
__device__ u8 c_topnode[N * 32768 * 2];
__device__ u8 wots_pk_z[WOTS_SIG_BYTES * 32768]; // 16, 20

__device__ u8 one_auth_path[20 * N];
__device__ u8 one_treehash_node[20 * N];
__device__ u8 one_pk[WOTS_SIG_BYTES];

#ifdef USING_BDS
// #ifdef USING_COALESCED_ACCESS
// #endif // ifdef USING_COALESCED_ACCESS

__device__ void dev_xmssmt_serialize_state(u8 *sk, bds_state *states)
{
	u32 i, j;

	/* Skip past the 'regular' sk */
	sk += INDEX_BYTES + 4 * N;

	for (i = 0; i < 2 * D - 1; i++) {
		sk += (TREE_HEIGHT + 1) * N; /* stack */

		dev_ull_to_bytes(sk, 4, states[i].stackoffset);
		sk += 4;

		sk += TREE_HEIGHT + 1;          /* stacklevels */
		sk += TREE_HEIGHT * N;          /* auth */
		sk += (TREE_HEIGHT >> 1) * N;   /* keep */

		for (j = 0; j < TREE_HEIGHT - BDS_K; j++) {
			dev_ull_to_bytes(sk, 1, states[i].treehash[j].h);
			sk += 1;

			dev_ull_to_bytes(sk, 4, states[i].treehash[j].next_idx);
			sk += 4;

			dev_ull_to_bytes(sk, 1, states[i].treehash[j].stackusage);
			sk += 1;

			dev_ull_to_bytes(sk, 1, states[i].treehash[j].completed);
			sk += 1;

			sk += N; /* node */
		}

		/* retain */
		sk += ((1 << BDS_K) - BDS_K - 1) * N;

		dev_ull_to_bytes(sk, 4, states[i].next_leaf);
		sk += 4;
	}
} // xmssmt_serialize_state

__device__ void dev_xmssmt_deserialize_state(bds_state *states,
					     u8 **wots_sigs,  u8 *sk)
{
	u32 i, j;

	/* Skip past the 'regular' sk */
	sk += INDEX_BYTES + 4 * N;

	// TODO These data sizes follow from the (former) test xmss_core_fast.c
	// TODO They should be reconsidered / motivated more explicitly

	for (i = 0; i < 2 * D - 1; i++) {
		states[i].stack = sk;
		sk += (TREE_HEIGHT + 1) * N;

		states[i].stackoffset = dev_bytes_to_ull(sk, 4);
		sk += 4;

		states[i].stacklevels = sk;
		sk += TREE_HEIGHT + 1;

		states[i].auth = sk;
		sk += TREE_HEIGHT * N;

		states[i].keep = sk;
		sk += (TREE_HEIGHT >> 1) * N;

		for (j = 0; j < TREE_HEIGHT - BDS_K; j++) {
			states[i].treehash[j].h = dev_bytes_to_ull(sk, 1);
			sk += 1;

			states[i].treehash[j].next_idx = dev_bytes_to_ull(sk, 4);
			sk += 4;

			states[i].treehash[j].stackusage = dev_bytes_to_ull(sk, 1);
			sk += 1;

			states[i].treehash[j].completed = dev_bytes_to_ull(sk, 1);
			sk += 1;

			states[i].treehash[j].node = sk;
			sk += N;
		}

		states[i].retain = sk;
		sk += ((1 << BDS_K) - BDS_K - 1) * N;

		states[i].next_leaf = dev_bytes_to_ull(sk, 4);
		sk += 4;
	}

	if (D > 1) {
		*wots_sigs = sk;
	}
} // xmssmt_deserialize_state

__device__ void dev_xmss_serialize_state(u8 *sk, bds_state *state)
{
	dev_xmssmt_serialize_state(sk, state);
} // xmss_serialize_state

__device__ void dev_xmss_deserialize_state(bds_state *state, u8 *sk)
{
	dev_xmssmt_deserialize_state(state, NULL, sk);
} // dev_xmss_deserialize_state

__device__ void dev_memswap(void *a, void *b, void *t, u64 len)
{
	memcpy(t, a, len);
	memcpy(a, b, len);
	memcpy(b, t, len);
} // memswap

__device__ void dev_deep_state_swap(bds_state *a, bds_state *b)
{
	// TODO this is extremely ugly and should be refactored
	// TODO right now, this ensures that both 'stack' and 'retain' fit
	u8 t[((TREE_HEIGHT + 1) > ((1 << BDS_K) - BDS_K - 1)
	 ? (TREE_HEIGHT + 1) : ((1 << BDS_K) - BDS_K - 1)) * N];
	u32 i;

	dev_memswap(a->stack, b->stack, t, (TREE_HEIGHT + 1) * N);
	dev_memswap(&a->stackoffset, &b->stackoffset, t, sizeof(a->stackoffset));
	dev_memswap(a->stacklevels, b->stacklevels, t, TREE_HEIGHT + 1);
	dev_memswap(a->auth, b->auth, t, TREE_HEIGHT * N);
	dev_memswap(a->keep, b->keep, t, (TREE_HEIGHT >> 1) * N);

	for (i = 0; i < TREE_HEIGHT - BDS_K; i++) {
		dev_memswap(&a->treehash[i].h, &b->treehash[i].h, t, sizeof(a->treehash[i].h));
		dev_memswap(&a->treehash[i].next_idx, &b->treehash[i].next_idx, t, sizeof(a->treehash[i].next_idx));
		dev_memswap(&a->treehash[i].stackusage, &b->treehash[i].stackusage, t, sizeof(a->treehash[i].stackusage));
		dev_memswap(&a->treehash[i].completed, &b->treehash[i].completed, t, sizeof(a->treehash[i].completed));
		dev_memswap(a->treehash[i].node, b->treehash[i].node, t, N);
	}

	dev_memswap(a->retain, b->retain, t, ((1 << BDS_K) - BDS_K - 1) * N);
	dev_memswap(&a->next_leaf, &b->next_leaf, t, sizeof(a->next_leaf));
} // dev_deep_state_swap

//internal
__device__ int dev_treehash_minheight_on_stack(bds_state *state,
					       const treehash_inst *treehash)
{
	u32 r = TREE_HEIGHT, i;

	for (i = 0; i < treehash->stackusage; i++) {
		if (state->stacklevels[state->stackoffset - i - 1] < r) {
			r = state->stacklevels[state->stackoffset - i - 1];
		}
	}
	return r;
} // treehash_minheight_on_stack

// serial version
__device__ void dev_treehash_init(u8 *node, int height, int index,
				  bds_state *state, const u8 *sk_seed,
				  const u8 *pub_seed, const u32 addr[8])
{
	u32 idx = index; // 0
	// use three different addresses because at this point we use all three formats in parallel
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	u32 lastnode, i;
	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 stacklevels[TREE_HEIGHT + 1];
	u32 stackoffset = 0;
	u32 nodeh;

	lastnode = idx + (1 << TREE_HEIGHT); // 2^h

	for (i = 0; i < TREE_HEIGHT - BDS_K; i++) {
		state->treehash[i].h = i;
		state->treehash[i].completed = 1;
		state->treehash[i].stackusage = 0;
	}
	i = 0;
	for (; idx < lastnode; idx++) {
		dev_set_ltree_addr(ltree_addr, idx);
		dev_set_ots_addr(ots_addr, idx);
		#ifdef NOT_TEST_RATIO
		dev_gen_leaf_wots(stack + stackoffset * N, sk_seed, pub_seed, ltree_addr, ots_addr);
		#endif
		stacklevels[stackoffset] = 0;
		stackoffset++;
		if (TREE_HEIGHT - BDS_K > 0 && i == 3) {
			memcpy(state->treehash[0].node, stack + stackoffset * N, N);
		}
		while (stackoffset > 1 && stacklevels[stackoffset - 1] == stacklevels[stackoffset - 2]) {
			nodeh = stacklevels[stackoffset - 1];
			if (i >> nodeh == 1) {
				memcpy(state->auth + nodeh * N, stack + (stackoffset - 1) * N, N);
			} else {
				if (nodeh < TREE_HEIGHT - BDS_K && i >> nodeh == 3) {
					memcpy(state->treehash[nodeh].node, stack + (stackoffset - 1) * N, N);
				}else if (nodeh >= TREE_HEIGHT - BDS_K) {
					memcpy(state->retain + ((1 << (TREE_HEIGHT - 1 - nodeh)) + nodeh - TREE_HEIGHT + (((i >> nodeh) - 3) >> 1)) * N, stack + (stackoffset - 1) * N, N);
				}
			}
			dev_set_tree_height(node_addr, stacklevels[stackoffset - 1]);
			dev_set_tree_index(node_addr, (idx >> (stacklevels[stackoffset - 1] + 1)));
			dev_thash_h(stack + (stackoffset - 2) * N, stack + (stackoffset - 2) * N, pub_seed, node_addr);
			stacklevels[stackoffset - 2]++;
			stackoffset--;
		}
		i++;
	}

	for (i = 0; i < N; i++) {
		node[i] = stack[i];
	}
} // dev_treehash_init

#if TREE_HEIGHT == 5
__device__ void dev_treehash_init_parallel_5(u8 *node, int height, int index,
					     bds_state *state, const u8 *sk_seed,
					     const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	// g.sync();
	int max_threads;

#ifdef NOT_TEST_RATIO

#if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))
	const unsigned int tnum = gridDim.x * blockDim.x;
	int id, ttid;

	if (tnum < 32 * WOTS_LEN) {
		printf("wrong thread size\n");
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
		u8 buf[N + 32];
		dev_set_ltree_addr(ltree_addr, ttid);
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
		dev_set_ltree_addr(ltree_addr, tid);
		dev_set_ots_addr(ots_addr, tid);
		dev_wots_pkgen(wots_pk + tid * WOTS_SIG_BYTES, sk_seed, pub_seed, ots_addr);
	}
	g.sync();
#endif // ifdef USING_PARALLEL_WOTS_PKGEN

#ifdef USING_PARALLEL_L_TREE
	ttid = tid % 32;
	id = tid / 32;
	u8 *begin = wots_pk + ttid * WOTS_SIG_BYTES;
	dev_set_ltree_addr(ltree_addr, ttid);
#if WOTS_LEN == 67
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
#else // if WOTS_LEN == 67
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
#endif // if WOTS_LEN == 67
#else // ifdef USING_PARALLEL_L_TREE
	if (tid < 32) {
		dev_set_ltree_addr(ltree_addr, tid);
		u8 *pk = wots_pk + tid * WOTS_SIG_BYTES;
		dev_l_tree(branch + tid * N, pk, pub_seed, ltree_addr);
	}
	g.sync();
#endif  // ifdef USING_PARALLEL_L_TREE

#endif  // ifdef NOT_TEST_RATIO

	if (tid == 1) memcpy(one_auth_path, branch + tid * N, N);
	if (tid == 3) memcpy(one_treehash_node, branch + tid * N, N);

	max_threads = 32;
	int p_height = 5; // 2^6

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off,
				    branch + off, pub_seed, node_addr);
			if (tid == 1)
				memcpy(one_auth_path + i * N, branch + off, N);
			if (tid == 3)
				memcpy(one_treehash_node + i * N, branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			state->treehash[i].h = i;
			state->treehash[i].completed = 1;
			state->treehash[i].stackusage = 0;
		}
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(state->auth + i * N, one_auth_path + i * N, N);
		for (int i = 0; i < TREE_HEIGHT - 1; i++)
			memcpy(state->treehash[i].node, one_treehash_node + i * N, N);
		memcpy(node, branch, N);
	}
} // dev_treehash_init_parallel_5
#endif // if TREE_HEIGHT == 5

#if TREE_HEIGHT == 10
// 到达1024时会异常，然后退出程序，这部分没有写好
// 前1024轮全部通过验证
__device__ void dev_treehash_init_parallel_10(u8 *node, int height, int index,
					      bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	g.sync();

#ifdef NOT_TEST_RATIO

#if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_L_TREE))
	const unsigned int tnum = gridDim.x * blockDim.x;
	int max_threads = 1;
	while (max_threads <= tnum / 2) max_threads *= 2;
	int ttid, id;
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
	int para = max_threads / 1024;
	ttid = tid % 1024;
	id = tid / 1024;
	if (tid < max_threads) {
		u8 *pk = wots_pk + ttid * WOTS_SIG_BYTES;
		int q = WOTS_LEN / para;
		int r = WOTS_LEN % para;
		int local = q + ((id < r) ? 1 : 0);
		int offset = id * q + ((id < r) ? id : r);
		u8 buf[N + 32];

		dev_set_ots_addr(ots_addr, ttid);
		dev_set_hash_addr(ots_addr, 0);
		dev_set_key_and_mask(ots_addr, 0);
		memcpy(buf, pub_seed, N);

		for (int i = offset; i < offset + local; i++) {
			dev_set_chain_addr(ots_addr, i);
			dev_addr_to_bytes(buf + N, ots_addr);
			dev_prf_keygen(pk + i * N, buf, sk_seed);
		}
		for (int i = offset; i < offset + local; i++) {
			dev_set_chain_addr(ots_addr, i);
			dev_gen_chain(pk + i * N, pk + i * N,
				      0, WOTS_W - 1, pub_seed, ots_addr);
		}
	}
	#endif // ifdef USING_COALESCED_ACCESS
	g.sync();
#else // ifdef USING_PARALLEL_WOTS_PKGEN
	if (tid < 1024) {
		dev_set_ots_addr(ots_addr, tid);
		dev_wots_pkgen(wots_pk + tid * WOTS_SIG_BYTES,
			       sk_seed, pub_seed, ots_addr);
	}
#endif // ifdef USING_PARALLEL_WOTS_PKGEN

#ifdef USING_PARALLEL_L_TREE
	ttid = tid % 1024;
	id = tid / 1024;
	int scheme = 1;// 1: faster
	dev_set_ltree_addr(ltree_addr, ttid);

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
			u8 *off_pk = wots_pk + ttid * WOTS_SIG_BYTES + 64 * N;
			while (l > 1) {
				int parent_nodes = l >> 1;
				for (int i = 0; i < parent_nodes; i++) {
					dev_set_tree_index(ltree_addr, i + (32 >> height));
					dev_thash_h(off_pk + i * N,
						    off_pk + i * 2 * N, pub_seed, ltree_addr);
				}
				if (l & 1) {
					memcpy(off_pk + (l >> 1) * N, off_pk + (l - 1) * N, N);
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
			u8 *off_pk = wots_pk + ttid * WOTS_SIG_BYTES;
			memcpy(off_pk + N, off_pk + 64 * N, N);
			dev_set_tree_index(ltree_addr, 0);
			dev_set_tree_height(ltree_addr, 6);
			dev_thash_h(off_pk, off_pk, pub_seed, ltree_addr);
			memcpy(branch + tid * N, off_pk, N);
		}
	} else if (scheme == 2 && WOTS_LEN == 67) {
		int para, q, r, local, offset;
		u8 *begin;
		max_threads = 1;
		while (max_threads <= tnum / 2) {
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
	} else if (scheme == 1 && WOTS_LEN == 51) {
		int para, q, r, local, offset;
		u8 *begin;
		// max_threads = 8192;
		max_threads = 1;
		while (max_threads <= tnum / 2) {
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
		dev_set_ltree_addr(ltree_addr, tid);
		dev_l_tree(branch + tid * N, wots_pk + tid * WOTS_SIG_BYTES,
			   pub_seed, ltree_addr);
	}
#endif  // ifdef USING_PARALLEL_L_TREE

#endif  // ifdef NOT_TEST_RATIO

	g.sync();
	if (tid == 1) memcpy(one_auth_path, branch + tid * N, N);
	if (tid == 3) memcpy(one_treehash_node, branch + tid * N, N);

	for (int i = 1, ii = 1; i <= TREE_HEIGHT; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (1024 >> i)) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			if (tid == 1) memcpy(one_auth_path + i * N, branch + off, N);
			if (tid == 3) memcpy(one_treehash_node + i * N, branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			state->treehash[i].h = i;
			state->treehash[i].completed = 1;
			state->treehash[i].stackusage = 0;
		}
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(state->auth + i * N, one_auth_path + i * N, N);
		for (int i = 0; i < TREE_HEIGHT - 1; i++)
			memcpy(state->treehash[i].node, one_treehash_node + i * N, N);
		memcpy(node, branch, N);
	}

} // dev_treehash_init_parallel_10

__device__ void dev_treehash_init_parallel_10_1(u8 *node, int height, int index,
						bds_state *state, const u8 *sk_seed,
						const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	g.sync();

	u32 lastnode;
	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 stacklevels[TREE_HEIGHT + 1];
	u32 stackoffset = 0;
	u32 nodeh;

	lastnode = (1 << TREE_HEIGHT); // 2^h

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			state->treehash[i].h = i;
			state->treehash[i].completed = 1;
			state->treehash[i].stackusage = 0;
		}
	}

	int max_threads = 64;
	int p_height = 6; // 2^6

	if (tid < max_threads) {
		int i = tid * lastnode / max_threads;
		int idx = i;
		for (; idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + stackoffset * N, sk_seed, pub_seed, ltree_addr, ots_addr);
			stacklevels[stackoffset] = 0;
			stackoffset++;
			if (TREE_HEIGHT - BDS_K > 0 && i == 3) {
				memcpy(one_treehash_node, stack + stackoffset * N, N);
			}
			while (stackoffset > 1 && stacklevels[stackoffset - 1] == stacklevels[stackoffset - 2]) {
				nodeh = stacklevels[stackoffset - 1];
				if (i >> nodeh == 1) {
					memcpy(one_auth_path + nodeh * N, stack + (stackoffset - 1) * N, N);
				} else if (nodeh < TREE_HEIGHT - BDS_K && i >> nodeh == 3) {
					memcpy(one_treehash_node + nodeh * N, stack + (stackoffset - 1) * N, N);
				}

				dev_set_tree_height(node_addr, stacklevels[stackoffset - 1]);
				dev_set_tree_index(node_addr, (idx >> (stacklevels[stackoffset - 1] + 1)));
				dev_thash_h(stack + (stackoffset - 2) * N, stack + (stackoffset - 2) * N, pub_seed, node_addr);
				stacklevels[stackoffset - 2]++;
				stackoffset--;
			}
			i++;
		}
		if (tid == 1) memcpy(one_auth_path + (TREE_HEIGHT - p_height) * N, stack, N);
		if (tid == 3) memcpy(one_treehash_node + (TREE_HEIGHT - p_height) * N, stack, N);
		memcpy(branch + tid * N, stack, N);
	}

	for (int i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, TREE_HEIGHT - 1 - p_height + i);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off,
				    branch + off, pub_seed, node_addr);
			if (tid == 1)
				memcpy(one_auth_path + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
			if (tid == 3)
				memcpy(one_treehash_node + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(state->auth + i * N, one_auth_path + i * N, N);
		for (int i = 0; i < TREE_HEIGHT - 1; i++)
			memcpy(state->treehash[i].node, one_treehash_node + i * N, N);
		memcpy(node, branch, N);
	}
} // dev_treehash_init_parallel_10_1

// unfinished
__device__ void dev_treehash_init_parallel_10_2(u8 *node, int height, int index,
						bds_state *state, const u8 *sk_seed,
						const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	g.sync();

	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}
	if (max_threads > 1024) {
		printf("wrong in dev_treehash_init_parallel_10_2\n");
		return;
	}

	if (tid == 0)
		printf("max_threads = %d %d\n", max_threads, p_height);

	int i_size = (1 << (TREE_HEIGHT - p_height));

	for (int iter = 0; iter < i_size; iter++) {
		g.sync();
		if (tid < tnum) {
			u8 buf[N + 32];
			u8 temp[N];
			for (int i = tid; i < WOTS_LEN * max_threads; i += tnum) {
				dev_set_ots_addr(ots_addr, i / WOTS_LEN + iter * max_threads);
				dev_set_chain_addr(ots_addr, i % WOTS_LEN);
				dev_set_hash_addr(ots_addr, 0);
				dev_set_key_and_mask(ots_addr, 0);
				memcpy(buf, pub_seed, N);
				dev_addr_to_bytes(buf + N, ots_addr);
				dev_prf_keygen(temp, buf, sk_seed);
				dev_gen_chain(temp, temp, 0, WOTS_W - 1, pub_seed, ots_addr);
				memcpy(wots_pk_z + i * N, temp, N);
			}
		}
		g.sync();
		if (tid < max_threads) {
			dev_set_ltree_addr(ltree_addr, tid + iter * max_threads);
			dev_l_tree(branch + tid * N, wots_pk_z + tid * WOTS_SIG_BYTES,
				   pub_seed, ltree_addr);
		}

		g.sync();
		if (iter == 0 && tid == 1) memcpy(one_auth_path, branch + tid * N, N);
		if (iter == 0 && tid == 3) memcpy(one_treehash_node, branch + tid * N, N);

		for (int i = 1, ii = 1; i <= p_height; i++) {
			g.sync();
			int off = 2 * tid * ii * N;
			if (tid < (max_threads >> i)) {
				dev_set_tree_height(node_addr, i - 1);
				dev_set_tree_index(node_addr, tid + iter * (max_threads >> i));
				memcpy(branch + off + N, branch + off + ii * N, N);
				dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
				if (tid == 1 && iter == 0)
					memcpy(one_auth_path + i * N, branch + off, N);
				if (tid == 3 && iter == 0)
					memcpy(one_treehash_node + i * N, branch + off, N);
				if (i == p_height - 1 && tid == 1 && iter == 1)
					memcpy(one_treehash_node + (p_height - 1) * N, branch + off, N);
			}
			ii *= 2;
		}
		if (tid == 0) memcpy(&c_topnode[iter * N], branch, N);
		if (tid == 0 && iter == 1)
			memcpy(one_auth_path + p_height * N, branch, N);
		if (tid == 0 && iter == i_size - 1)
			memcpy(one_treehash_node + p_height * N, branch, N);
	}

	for (int i = p_height + 1, ii = 1; i <= TREE_HEIGHT; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (i_size >> (i - p_height))) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(c_topnode + off + N, c_topnode + off + ii * N, N);
			dev_thash_h(c_topnode + off, c_topnode + off, pub_seed, node_addr);
			if (tid == 1)
				memcpy(one_auth_path + i * N, c_topnode + off, N);
			if (tid == 3)
				memcpy(one_treehash_node + i * N,  c_topnode + off, N);
		}
		ii *= 2;
	}
} // dev_treehash_init_parallel_10_2
#endif // if TREE_HEIGHT == 10

#if TREE_HEIGHT == 16
// 正确性有待确定，已经验证到1100次签名

#ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_init_parallel_16(u8 *node, int height, int index,
					      bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	g.sync();

	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}
	// if (tid == 0)
	// 	printf("max_threads = %d %d\n", max_threads, p_height);

	int i_size = (1 << (TREE_HEIGHT - p_height));

	for (int iter = 0; iter < i_size; iter++) {
		g.sync();

#ifdef NOT_TEST_RATIO
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
				memcpy(wots_pk_z + i * N, temp, N);
			#else // ifdef USING_LOCAL_MEMORY
				dev_prf_keygen(wots_pk_z + i * N, buf, sk_seed);
				dev_gen_chain(wots_pk_z + i * N, wots_pk_z + i * N,
					      0, WOTS_W - 1, pub_seed, ots_addr);
			#endif // ifdef USING_LOCAL_MEMORY
			}
		}
		g.sync();
		if (tid < max_threads) {
			dev_set_ltree_addr(ltree_addr, tid + iter * max_threads);
		#ifdef USING_LOCAL_MEMORY
			u8 temp[WOTS_SIG_BYTES];
			memcpy(temp, wots_pk_z + tid * WOTS_SIG_BYTES, WOTS_SIG_BYTES);
			dev_l_tree(branch + tid * N, temp, pub_seed, ltree_addr);
		#else // ifdef USING_LOCAL_MEMORY
			dev_l_tree(branch + tid * N, wots_pk_z + tid * WOTS_SIG_BYTES,
				   pub_seed, ltree_addr);
		#endif  // ifdef USING_LOCAL_MEMORY
		}
#endif                  // ifdef NOT_TEST_RATIO

		g.sync();
		if (iter == 0 && tid == 1) memcpy(one_auth_path, branch + tid * N, N);
		if (iter == 0 && tid == 3) memcpy(one_treehash_node, branch + tid * N, N);

		for (int i = 1, ii = 1; i <= p_height; i++) {
			g.sync();
			int off = 2 * tid * ii * N;
			if (tid < (max_threads >> i)) {
				dev_set_tree_height(node_addr, i - 1);
				dev_set_tree_index(node_addr, tid + iter * (max_threads >> i));
				memcpy(branch + off + N, branch + off + ii * N, N);
				dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
				if (tid == 1 && iter == 0)
					memcpy(one_auth_path + i * N, branch + off, N);
				if (tid == 3 && iter == 0)
					memcpy(one_treehash_node + i * N, branch + off, N);
				if (i == p_height - 1 && tid == 1 && iter == 1)
					memcpy(one_treehash_node + (p_height - 1) * N, branch + off, N);
			}
			ii *= 2;
		}
		if (tid == 0) memcpy(&c_topnode[iter * N], branch, N);
		if (tid == 0 && iter == 1)
			memcpy(one_auth_path + p_height * N, branch, N);
		if (tid == 0 && iter == i_size - 1)
			memcpy(one_treehash_node + p_height * N, branch, N);
	}

	for (int i = p_height + 1, ii = 1; i <= TREE_HEIGHT; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (i_size >> (i - p_height))) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(c_topnode + off + N, c_topnode + off + ii * N, N);
			dev_thash_h(c_topnode + off, c_topnode + off, pub_seed, node_addr);
			if (tid == 1)
				memcpy(one_auth_path + i * N, c_topnode + off, N);
			if (tid == 3)
				memcpy(one_treehash_node + i * N,  c_topnode + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			state->treehash[i].h = i;
			state->treehash[i].completed = 1;
			state->treehash[i].stackusage = 0;
		}
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(state->auth + i * N, one_auth_path + i * N, N);
		for (int i = 0; i < TREE_HEIGHT - 1; i++)
			memcpy(state->treehash[i].node, one_treehash_node + i * N, N);
		memcpy(node, c_topnode, N);

		// for (int i = 0; i < TREE_HEIGHT; i++)
		// 	printf("%02x ", state->auth[i * N]);
		// printf("\n");
		// for (int i = 0; i < TREE_HEIGHT - 1; i++)
		// 	printf("%02x ", state->treehash[i].node[0]);
		// printf("\n");
	}

}       // dev_treehash_init_parallel_16
#else // ifdef USING_COALESCED_ACCESS
__device__ void dev_treehash_init_parallel_16(u8 *node, int height, int index,
					      bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	g.sync();

	u32 lastnode;
	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 stacklevels[TREE_HEIGHT + 1];
	u32 stackoffset = 0;
	u32 nodeh;

	lastnode = (1 << TREE_HEIGHT); // 2^h

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			state->treehash[i].h = i;
			state->treehash[i].completed = 1;
			state->treehash[i].stackusage = 0;
		}
	}

	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}

	if (tid < max_threads) {
		int i = tid * lastnode / max_threads;
		int idx = i;
		for (; idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + stackoffset * N, sk_seed, pub_seed, ltree_addr, ots_addr);
			stacklevels[stackoffset] = 0;
			stackoffset++;
			if (TREE_HEIGHT - BDS_K > 0 && i == 3) {
				memcpy(one_treehash_node, stack + stackoffset * N, N);
			}
			while (stackoffset > 1 && stacklevels[stackoffset - 1] == stacklevels[stackoffset - 2]) {
				nodeh = stacklevels[stackoffset - 1];
				if (i >> nodeh == 1) {
					memcpy(one_auth_path + nodeh * N, stack + (stackoffset - 1) * N, N);
				} else if (nodeh < TREE_HEIGHT - BDS_K && i >> nodeh == 3) {
					memcpy(one_treehash_node + nodeh * N, stack + (stackoffset - 1) * N, N);
				}

				dev_set_tree_height(node_addr, stacklevels[stackoffset - 1]);
				dev_set_tree_index(node_addr, (idx >> (stacklevels[stackoffset - 1] + 1)));
				dev_thash_h(stack + (stackoffset - 2) * N, stack + (stackoffset - 2) * N, pub_seed, node_addr);
				stacklevels[stackoffset - 2]++;
				stackoffset--;
			}
			i++;
		}
		if (tid == 1) memcpy(one_auth_path + (TREE_HEIGHT - p_height) * N, stack, N);
		if (tid == 3) memcpy(one_treehash_node + (TREE_HEIGHT - p_height) * N, stack, N);
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
			if (tid == 1)
				memcpy(one_auth_path + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
			if (tid == 3)
				memcpy(one_treehash_node + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(state->auth + i * N, one_auth_path + i * N, N);

		// for (int i = 0; i < TREE_HEIGHT; i++)
		// 	printf("%02x ", state->auth[i * N]);
		// printf("\n");

		for (int i = 0; i < TREE_HEIGHT - 1; i++)
			memcpy(state->treehash[i].node, one_treehash_node + i * N, N);

		// for (int i = 0; i < TREE_HEIGHT - 1; i++)
		// 	printf("%02x ", state->treehash[i].node[0]);
		// printf("\n");

		memcpy(node, branch, N);
	}

} // dev_treehash_init_parallel_16
#endif // ifdef USING_COALESCED_ACCESS
#endif // if TREE_HEIGHT == 16

#if TREE_HEIGHT == 20

#ifdef USING_COALESCED_ACCESS
//right
__device__ void dev_treehash_init_parallel_20(u8 *node, int height, int index,
					      bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	g.sync();

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			state->treehash[i].h = i;
			state->treehash[i].completed = 1;
			state->treehash[i].stackusage = 0;
		}
	}
	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}

	int i_size = (1 << (TREE_HEIGHT - p_height));

	for (int iter = 0; iter < i_size; iter++) {
		g.sync();
#ifdef NOT_TEST_RATIO
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
				memcpy(wots_pk_z + i * N, temp, N);
		#else // ifdef USING_LOCAL_MEMORY
				dev_prf_keygen(wots_pk_z + i * N, buf, sk_seed);
				dev_gen_chain(wots_pk_z + i * N, wots_pk_z + i * N,
					      0, WOTS_W - 1, pub_seed, ots_addr);
		#endif // ifdef USING_LOCAL_MEMORY
			}
		}
		g.sync();
		if (tid < max_threads) {
			dev_set_ltree_addr(ltree_addr, tid + iter * max_threads);
	    #ifdef USING_LOCAL_MEMORY
			u8 temp[WOTS_SIG_BYTES];
			memcpy(temp, wots_pk_z + tid * WOTS_SIG_BYTES, WOTS_SIG_BYTES);
			dev_l_tree(branch + tid * N, temp, pub_seed, ltree_addr);
	    #else // ifdef USING_LOCAL_MEMORY
			dev_l_tree(branch + tid * N, wots_pk_z + tid * WOTS_SIG_BYTES,
				   pub_seed, ltree_addr);
	    #endif // ifdef USING_LOCAL_MEMORY
		}
#endif // ifdef NOT_TEST_RATIO

		g.sync();
		if (iter == 0 && tid == 1) memcpy(one_auth_path, branch + tid * N, N);
		if (iter == 0 && tid == 3) memcpy(one_treehash_node, branch + tid * N, N);

		for (int i = 1, ii = 1; i <= p_height; i++) {
			g.sync();
			int off = 2 * tid * ii * N;
			if (tid < (max_threads >> i)) {
				dev_set_tree_height(node_addr, i - 1);
				dev_set_tree_index(node_addr, tid + iter * (max_threads >> i));
				memcpy(branch + off + N, branch + off + ii * N, N);
				dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
				if (tid == 1 && iter == 0)
					memcpy(one_auth_path + i * N, branch + off, N);
				if (tid == 3 && iter == 0)
					memcpy(one_treehash_node + i * N, branch + off, N);
				if (i == p_height - 1 && tid == 1 && iter == 1)
					memcpy(one_treehash_node + (p_height - 1) * N, branch + off, N);
			}
			ii *= 2;
		}
		if (tid == 0) memcpy(&c_topnode[iter * N], branch, N);
		if (tid == 0 && iter == 1)
			memcpy(one_auth_path + p_height * N, branch, N);
		if (tid == 0 && iter == i_size - 1)
			memcpy(one_treehash_node + p_height * N, branch, N);
	}

	for (int i = p_height + 1, ii = 1; i <= TREE_HEIGHT; i++) {
		g.sync();
		int off = 2 * tid * ii * N;
		if (tid < (i_size >> (i - p_height))) {
			dev_set_tree_height(node_addr, i - 1);
			dev_set_tree_index(node_addr, tid);
			memcpy(c_topnode + off + N, c_topnode + off + ii * N, N);
			dev_thash_h(c_topnode + off, c_topnode + off, pub_seed, node_addr);
			if (tid == 1)
				memcpy(one_auth_path + i * N, c_topnode + off, N);
			if (tid == 3)
				memcpy(one_treehash_node + i * N,  c_topnode + off, N);
		}
		ii *= 2;
	}

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(state->auth + i * N, one_auth_path + i * N, N);

		// for (int i = 0; i < TREE_HEIGHT; i++)
		// 	printf("%02x ", state->auth[i * N]);
		// printf("\n");

		for (int i = 0; i < TREE_HEIGHT - 1; i++)
			memcpy(state->treehash[i].node, one_treehash_node + i * N, N);

		// for (int i = 0; i < TREE_HEIGHT - 1; i++)
		// 	printf("%02x ", state->treehash[i].node[0]);
		// printf("\n");
		memcpy(node, c_topnode, N);
	}

}       // dev_treehash_init_parallel_20
#else   // ifdef USING_COALESCED_ACCESS
// it has different root and the result is not stable
__device__ void dev_treehash_init_parallel_20(u8 *node, int height, int index,
					      bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	g.sync();

	u32 lastnode;
	u8 stack[(TREE_HEIGHT + 1) * N];
	u32 stacklevels[TREE_HEIGHT + 1];
	u32 stackoffset = 0;
	u32 nodeh;

	lastnode = (1 << TREE_HEIGHT);                 // 2^h

	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			state->treehash[i].h = i;
			state->treehash[i].completed = 1;
			state->treehash[i].stackusage = 0;
		}
	}
	int max_threads = 1;
	int p_height = 0;

	while (max_threads <= tnum / 2) {
		max_threads *= 2;
		p_height++;
	}
	// if (tid == 0)
	// 	printf("%d %d\n", max_threads, p_height);

	// Build the max_threads subtrees
	if (tid < max_threads) {
		int i = tid * lastnode / max_threads;
		int idx = i;
		for (; idx < (tid + 1) * lastnode / max_threads; idx++) {
			dev_set_ltree_addr(ltree_addr, idx);
			dev_set_ots_addr(ots_addr, idx);
			dev_gen_leaf_wots(stack + stackoffset * N, sk_seed, pub_seed, ltree_addr, ots_addr);
			stacklevels[stackoffset] = 0;
			stackoffset++;
			if (TREE_HEIGHT - BDS_K > 0 && i == 3) {
				memcpy(one_treehash_node, stack + stackoffset * N, N);
			}
			while (stackoffset > 1 && stacklevels[stackoffset - 1] == stacklevels[stackoffset - 2]) {
				nodeh = stacklevels[stackoffset - 1];
				if (i >> nodeh == 1) {
					memcpy(one_auth_path + nodeh * N, stack + (stackoffset - 1) * N, N);
				} else if (nodeh < TREE_HEIGHT - BDS_K && i >> nodeh == 3) {
					memcpy(one_treehash_node + nodeh * N, stack + (stackoffset - 1) * N, N);
				}

				dev_set_tree_height(node_addr, stacklevels[stackoffset - 1]);
				dev_set_tree_index(node_addr, (idx >> (stacklevels[stackoffset - 1] + 1)));
				dev_thash_h(stack + (stackoffset - 2) * N, stack + (stackoffset - 2) * N, pub_seed, node_addr);
				stacklevels[stackoffset - 2]++;
				stackoffset--;
			}
			i++;
		}
		if (tid == 1) memcpy(one_auth_path + (TREE_HEIGHT - p_height) * N, stack, N);
		if (tid == 3) memcpy(one_treehash_node + (TREE_HEIGHT - p_height) * N, stack, N);
		memcpy(branch + tid * N, stack, N);
	}

	for (u32 i = 1, ii = 1; i <= p_height; i++) {
		g.sync();
		u32 off = 2 * tid * ii * N;
		if (tid < (max_threads >> i)) {
			dev_set_tree_height(node_addr, TREE_HEIGHT - 1 - p_height + i);
			dev_set_tree_index(node_addr, tid);
			memcpy(branch + off + N, branch + off + ii * N, N);
			dev_thash_h(branch + off, branch + off, pub_seed, node_addr);
			if (tid == 1)
				memcpy(one_auth_path + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
			if (tid == 3)
				memcpy(one_treehash_node + (TREE_HEIGHT - p_height + i) * N,
				       branch + off, N);
		}
		ii *= 2;
	}
	g.sync();
	if (tid == 0) {
		for (int i = 0; i < TREE_HEIGHT; i++)
			memcpy(state->auth + i * N, one_auth_path + i * N, N);

		// for (int i = 0; i < TREE_HEIGHT; i++)
		// 	printf("%02x ", state->auth[i * N]);
		// printf("\n");

		for (int i = 0; i < TREE_HEIGHT - 1; i++)
			memcpy(state->treehash[i].node, one_treehash_node + i * N, N);

		// for (int i = 0; i < TREE_HEIGHT - 1; i++)
		// 	printf("%02x ", state->treehash[i].node[0]);
		// printf("\n");
		memcpy(node, branch, N);
	}

}                 // dev_treehash_init_parallel_20
#endif // ifdef USING_COALESCED_ACCESS
#endif // if TREE_HEIGHT == 20

__device__ void dev_treehash_update(treehash_inst *treehash, bds_state *state,
				    const u8 *sk_seed,
				    const u8 *pub_seed,
				    const u32 addr[8])
{
	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	dev_set_ltree_addr(ltree_addr, treehash->next_idx);
	dev_set_ots_addr(ots_addr, treehash->next_idx);

	u8 nodebuffer[2 * N];
	u32 nodeheight = 0;

	dev_gen_leaf_wots(nodebuffer, sk_seed, pub_seed, ltree_addr, ots_addr);
	while (treehash->stackusage > 0 && state->stacklevels[state->stackoffset - 1] == nodeheight) {
		memcpy(nodebuffer + N, nodebuffer, N);
		memcpy(nodebuffer, state->stack + (state->stackoffset - 1) * N, N);
		dev_set_tree_height(node_addr, nodeheight);
		dev_set_tree_index(node_addr, (treehash->next_idx >> (nodeheight + 1)));
		dev_thash_h(nodebuffer, nodebuffer, pub_seed, node_addr);
		nodeheight++;
		treehash->stackusage--;
		state->stackoffset--;
	}
	if (nodeheight == treehash->h) {                 // this also implies stackusage == 0
		memcpy(treehash->node, nodebuffer, N);
		treehash->completed = 1;
	}else {
		memcpy(state->stack + state->stackoffset * N, nodebuffer, N);
		treehash->stackusage++;
		state->stacklevels[state->stackoffset] = nodeheight;
		state->stackoffset++;
		treehash->next_idx++;
	}
}                 // treehash_update

__device__ u8 one_nodebuffer[2 * N];

__device__ void dev_treehash_update_parallel(treehash_inst *treehash,
					     bds_state *state, const u8 *sk_seed,
					     const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	dev_set_ltree_addr(ltree_addr, treehash->next_idx);
	dev_set_ots_addr(ots_addr, treehash->next_idx);

	u8 nodebuffer[2 * N];
	u32 nodeheight = 0;

	dev_wots_pkgen_parallel(one_pk, sk_seed, pub_seed, ots_addr);
	dev_l_tree_parallel(one_nodebuffer, one_pk, pub_seed, ltree_addr);

	if (tid == 0) {
		memcpy(nodebuffer, one_nodebuffer, N);

		while (treehash->stackusage > 0 && state->stacklevels[state->stackoffset - 1] == nodeheight) {
			memcpy(nodebuffer + N, nodebuffer, N);
			memcpy(nodebuffer, state->stack + (state->stackoffset - 1) * N, N);
			dev_set_tree_height(node_addr, nodeheight);
			dev_set_tree_index(node_addr, (treehash->next_idx >> (nodeheight + 1)));
			dev_thash_h(nodebuffer, nodebuffer, pub_seed, node_addr);
			nodeheight++;
			if (tid == 0) treehash->stackusage--;
			if (tid == 0) state->stackoffset--;
		}

		if (nodeheight == treehash->h) {                 // this also implies stackusage == 0
			memcpy(treehash->node, nodebuffer, N);
			treehash->completed = 1;
		} else {
			memcpy(state->stack + state->stackoffset * N, nodebuffer, N);
			treehash->stackusage++;
			state->stacklevels[state->stackoffset] = nodeheight;
			state->stackoffset++;
			treehash->next_idx++;
		}
	}
	g.sync();
}                 // dev_treehash_update_parallel

__device__ char dev_bds_treehash_update(bds_state *state, u32 updates,
					const u8 *sk_seed,
					u8 *pub_seed,
					const u32 addr[8])
{
	u32 i, j;
	u32 level, l_min, low;
	u32 used = 0;

	for (j = 0; j < updates; j++) {
		l_min = TREE_HEIGHT;
		level = TREE_HEIGHT - BDS_K;
		for (i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			if (state->treehash[i].completed) {
				low = TREE_HEIGHT;
			}else if (state->treehash[i].stackusage == 0) {
				low = i;
			}else {
				low = dev_treehash_minheight_on_stack(state, &(state->treehash[i]));
			}
			if (low < l_min) {
				level = i;
				l_min = low;
			}
		}
		if (level == TREE_HEIGHT - BDS_K) {
			break;
		}
		dev_treehash_update(&(state->treehash[level]), state, sk_seed, pub_seed, addr);
		used++;
	}
	return updates - used;
}                 // dev_bds_treehash_update

__device__ char dev_bds_treehash_update_parallel(bds_state *state, u32 updates,
						 const u8 *sk_seed, u8 *pub_seed, const u32 addr[8])
{
	u32 i, j;
	u32 level, l_min, low;
	u32 used = 0;

	for (j = 0; j < updates; j++) {
		l_min = TREE_HEIGHT;
		level = TREE_HEIGHT - BDS_K;
		for (i = 0; i < TREE_HEIGHT - BDS_K; i++) {
			if (state->treehash[i].completed) {
				low = TREE_HEIGHT;
			}else if (state->treehash[i].stackusage == 0) {
				low = i;
			}else {
				low = dev_treehash_minheight_on_stack(state, &(state->treehash[i]));
			}
			if (low < l_min) {
				level = i;
				l_min = low;
			}
		}
		if (level == TREE_HEIGHT - BDS_K) {
			break;
		}
		dev_treehash_update_parallel(&(state->treehash[level]), state, sk_seed, pub_seed, addr);
		used++;
	}
	return updates - used;
}                 // dev_bds_treehash_update_parallel

__device__ char dev_bds_state_update(bds_state *state, const u8 *sk_seed,
				     const u8 *pub_seed,
				     const u32 addr[8])
{
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };
	u32 ots_addr[8] = { 0 };

	u32 nodeh;
	int idx = state->next_leaf;

	if (idx == 1 << TREE_HEIGHT) {
		return -1;
	}

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	dev_set_ots_addr(ots_addr, idx);
	dev_set_ltree_addr(ltree_addr, idx);

	dev_gen_leaf_wots(state->stack + state->stackoffset * N, sk_seed, pub_seed, ltree_addr, ots_addr);

	state->stacklevels[state->stackoffset] = 0;
	state->stackoffset++;
	if (TREE_HEIGHT - BDS_K > 0 && idx == 3) {
		memcpy(state->treehash[0].node, state->stack + state->stackoffset * N, N);
	}
	while (state->stackoffset > 1 && state->stacklevels[state->stackoffset - 1] == state->stacklevels[state->stackoffset - 2]) {
		nodeh = state->stacklevels[state->stackoffset - 1];
		if (idx >> nodeh == 1) {
			memcpy(state->auth + nodeh * N, state->stack + (state->stackoffset - 1) * N, N);
		} else {
			if (nodeh < TREE_HEIGHT - BDS_K && idx >> nodeh == 3) {
				memcpy(state->treehash[nodeh].node, state->stack + (state->stackoffset - 1) * N, N);
			}else if (nodeh >= TREE_HEIGHT - BDS_K) {
				memcpy(state->retain + ((1 << (TREE_HEIGHT - 1 - nodeh)) + nodeh - TREE_HEIGHT + (((idx >> nodeh) - 3) >> 1)) * N, state->stack + (state->stackoffset - 1) * N, N);
			}
		}
		dev_set_tree_height(node_addr, state->stacklevels[state->stackoffset - 1]);
		dev_set_tree_index(node_addr, (idx >> (state->stacklevels[state->stackoffset - 1] + 1)));
		dev_thash_h(state->stack + (state->stackoffset - 2) * N, state->stack + (state->stackoffset - 2) * N, pub_seed, node_addr);

		state->stacklevels[state->stackoffset - 2]++;
		state->stackoffset--;
	}
	state->next_leaf++;
	return 0;
}                 // bds_state_update

__device__ char dev_bds_state_update_parallel(bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();


	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };
	u32 ots_addr[8] = { 0 };

	u32 nodeh;
	int idx = state->next_leaf;

	if (idx == 1 << TREE_HEIGHT) {
		return -1;
	}

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	dev_set_ots_addr(ots_addr, idx);
	dev_set_ltree_addr(ltree_addr, idx);


	// dev_gen_leaf_wots(state->stack + state->stackoffset * N, sk_seed, pub_seed, ltree_addr, ots_addr);

	dev_wots_pkgen_parallel(one_pk, sk_seed, pub_seed, ots_addr);

	dev_l_tree_parallel(one_nodebuffer, one_pk, pub_seed, ltree_addr);

	if (tid == 0) {
		memcpy(state->stack + state->stackoffset * N, one_nodebuffer, N);

		state->stacklevels[state->stackoffset] = 0;
		state->stackoffset++;
		if (TREE_HEIGHT - BDS_K > 0 && idx == 3) {
			memcpy(state->treehash[0].node, state->stack + state->stackoffset * N, N);
		}
		while (state->stackoffset > 1 && state->stacklevels[state->stackoffset - 1] == state->stacklevels[state->stackoffset - 2]) {
			nodeh = state->stacklevels[state->stackoffset - 1];
			if (idx >> nodeh == 1) {
				memcpy(state->auth + nodeh * N, state->stack + (state->stackoffset - 1) * N, N);
			} else {
				if (nodeh < TREE_HEIGHT - BDS_K && idx >> nodeh == 3) {
					memcpy(state->treehash[nodeh].node, state->stack + (state->stackoffset - 1) * N, N);
				}else if (nodeh >= TREE_HEIGHT - BDS_K) {
					memcpy(state->retain + ((1 << (TREE_HEIGHT - 1 - nodeh)) + nodeh - TREE_HEIGHT + (((idx >> nodeh) - 3) >> 1)) * N, state->stack + (state->stackoffset - 1) * N, N);
				}
			}
			dev_set_tree_height(node_addr, state->stacklevels[state->stackoffset - 1]);
			dev_set_tree_index(node_addr, (idx >> (state->stacklevels[state->stackoffset - 1] + 1)));
			dev_thash_h(state->stack + (state->stackoffset - 2) * N, state->stack + (state->stackoffset - 2) * N, pub_seed, node_addr);

			state->stacklevels[state->stackoffset - 2]++;
			state->stackoffset--;
		}
		state->next_leaf++;
	}
	return 0;
}                 // dev_bds_state_update_parallel

__device__ void dev_bds_round(bds_state *state, const u64 leaf_idx,
			      const u8 *sk_seed,
			      const u8 *pub_seed, u32 addr[8])
{
	u32 i;
	u32 tau = TREE_HEIGHT;
	u32 startidx;
	u32 offset, rowidx;
	u8 buf[2 * N];

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	for (i = 0; i < TREE_HEIGHT; i++) {
		if (!((leaf_idx >> i) & 1)) {
			tau = i;
			break;
		}
	}

	if (tau > 0) {
		memcpy(buf, state->auth + (tau - 1) * N, N);
		// we need to do this before refreshing state->keep to prevent overwriting
		memcpy(buf + N, state->keep + ((tau - 1) >> 1) * N, N);
	}
	if (!((leaf_idx >> (tau + 1)) & 1) && (tau < TREE_HEIGHT - 1)) {
		memcpy(state->keep + (tau >> 1) * N, state->auth + tau * N, N);
	}
	if (tau == 0) {
		dev_set_ltree_addr(ltree_addr, leaf_idx);
		dev_set_ots_addr(ots_addr, leaf_idx);
		dev_gen_leaf_wots(state->auth, sk_seed, pub_seed, ltree_addr, ots_addr);
	}else {
		dev_set_tree_height(node_addr, (tau - 1));
		dev_set_tree_index(node_addr, leaf_idx >> tau);
		dev_thash_h(state->auth + tau * N, buf, pub_seed, node_addr);
		for (i = 0; i < tau; i++) {
			if (i < TREE_HEIGHT - BDS_K) {
				memcpy(state->auth + i * N, state->treehash[i].node, N);
			}else {
				offset = (1 << (TREE_HEIGHT - 1 - i)) + i - TREE_HEIGHT;
				rowidx = ((leaf_idx >> i) - 1) >> 1;
				memcpy(state->auth + i * N, state->retain + (offset + rowidx) * N, N);
			}
		}

		for (i = 0; i < ((tau < TREE_HEIGHT - BDS_K) ? tau : (TREE_HEIGHT - BDS_K)); i++) {
			startidx = leaf_idx + 1 + 3 * (1 << i);
			if (startidx < 1U << TREE_HEIGHT) {
				state->treehash[i].h = i;
				state->treehash[i].next_idx = startidx;
				state->treehash[i].completed = 0;
				state->treehash[i].stackusage = 0;
			}
		}
	}
}                 // dev_bds_round

__device__ void dev_bds_round_parallel(bds_state *state, const u64 leaf_idx,
				       const u8 *sk_seed, const u8 *pub_seed, u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	u32 i;
	u32 tau = TREE_HEIGHT;
	u32 startidx;
	u32 offset, rowidx;
	u8 buf[2 * N];

	u32 ots_addr[8] = { 0 };
	u32 ltree_addr[8] = { 0 };
	u32 node_addr[8] = { 0 };

	// only copy layer and tree address parts
	dev_copy_subtree_addr(ots_addr, addr);
	// type = ots
	dev_set_type(ots_addr, 0);
	dev_copy_subtree_addr(ltree_addr, addr);
	dev_set_type(ltree_addr, 1);
	dev_copy_subtree_addr(node_addr, addr);
	dev_set_type(node_addr, 2);

	for (i = 0; i < TREE_HEIGHT; i++) {
		if (!((leaf_idx >> i) & 1)) {
			tau = i;
			break;
		}
	}

	if (tau > 0) {
		memcpy(buf, state->auth + (tau - 1) * N, N);
		// we need to do this before refreshing state->keep to prevent overwriting
		memcpy(buf + N, state->keep + ((tau - 1) >> 1) * N, N);
	}
	if (!((leaf_idx >> (tau + 1)) & 1) && (tau < TREE_HEIGHT - 1)) {
		if (tid == 0) memcpy(state->keep + (tau >> 1) * N, state->auth + tau * N, N);
	}
	if (tau == 0) {
		dev_set_ltree_addr(ltree_addr, leaf_idx);
		dev_set_ots_addr(ots_addr, leaf_idx);
		dev_wots_pkgen_parallel(one_pk, sk_seed, pub_seed, ots_addr);
		dev_l_tree_parallel(one_nodebuffer, one_pk, pub_seed, ltree_addr);
		if (tid == 0) memcpy(state->auth, one_nodebuffer, N);
		// dev_gen_leaf_wots(state->auth, sk_seed, pub_seed, ltree_addr, ots_addr);
	} else {
		dev_set_tree_height(node_addr, (tau - 1));
		dev_set_tree_index(node_addr, leaf_idx >> tau);
		if (tid == 0) dev_thash_h(state->auth + tau * N, buf, pub_seed, node_addr);
		for (i = 0; i < tau; i++) {
			if (i < TREE_HEIGHT - BDS_K) {
				if (tid == 0) memcpy(state->auth + i * N, state->treehash[i].node, N);
			}else {
				offset = (1 << (TREE_HEIGHT - 1 - i)) + i - TREE_HEIGHT;
				rowidx = ((leaf_idx >> i) - 1) >> 1;
				if (tid == 0) memcpy(state->auth + i * N, state->retain + (offset + rowidx) * N, N);
			}
		}

		if (tid == 0)
			for (i = 0; i < ((tau < TREE_HEIGHT - BDS_K) ? tau : (TREE_HEIGHT - BDS_K)); i++) {
				startidx = leaf_idx + 1 + 3 * (1 << i);
				if (startidx < 1U << TREE_HEIGHT) {
					state->treehash[i].h = i;
					state->treehash[i].next_idx = startidx;
					state->treehash[i].completed = 0;
					state->treehash[i].stackusage = 0;
				}
			}
	}
}                 // dev_bds_round_parallel

#endif // ifdef USING_BDS
