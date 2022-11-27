#include <iostream>
using namespace std;

#include "gpu_hash.h"
#include "gpu_wots.h"
#include "gpu_hash_address.h"
#include "gpu_xmss_commons.h"

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

__device__ void dev_l_tree(u8 *leaf, u8 *wots_pk, const u8 *pub_seed, u32 addr[8])
{
	u32 l = WOTS_LEN;
	u32 parent_nodes;
	u32 i;
	u32 height = 0;

	dev_set_tree_height(addr, height);

	while (l > 1) {
		parent_nodes = l >> 1;
		for (i = 0; i < parent_nodes; i++) {
			dev_set_tree_index(addr, i);
			/* Hashes the nodes at (i*2)*N and (i*2)*N + 1 */
			dev_thash_h(wots_pk + i * N,
				    wots_pk + (i * 2) * N, pub_seed, addr);
		}
		/* If the row contained an odd number of nodes, the last node was not
		 * hashed. Instead, we pull it up to the next layer. */
		if (l & 1) {
			memcpy(wots_pk + (l >> 1) * N,
			       wots_pk + (l - 1) * N, N);
			l = (l >> 1) + 1;
		} else {
			l = l >> 1;
		}
		height++;
		dev_set_tree_height(addr, height);
	}
	memcpy(leaf, wots_pk, N);
} // dev_l_tree

__device__ u8 wots_pk_l1[(WOTS_LEN >> 1) + 1];
__device__ u8 wots_pk_l2[(WOTS_LEN >> 2) + 1];

__device__ void dev_l_tree_parallel(u8 *leaf, u8 *wots_pk,
				    const u8 *pub_seed, u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int tnum = gridDim.x * blockDim.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();
	u8 *begin = wots_pk;

#if WOTS_LEN == 67
	int para, q, r, local, offset;
	if (tnum == 8) {
		g.sync();
		para = 8;
		q = 33 / para;
		r = 33 % para;
		local = q + ((tid < r) ? 1 : 0);
		offset = tid * q + ((tid < r) ? tid : r);

		if (tid < para) {
			for (int i = offset; i < offset + local; i++) {
				dev_set_tree_index(addr, i);
				dev_set_tree_height(addr, 0);
				dev_thash_h(begin + i * 2 * N,
					    begin + i * 2 * N, pub_seed, addr);
			}
		}
		g.sync();

		para = 8;
		q = 17 / para;
		r = 17 % para;
		local = q + ((tid < r) ? 1 : 0);
		offset = tid * q + ((tid < r) ? tid : r);
		if (tid < para) {
			for (int i = offset; i < offset + local; i++) {
				memcpy(begin + i * 4 * N + N, begin + i * 4 * N + 2 * N, N);
				dev_set_tree_index(addr, i);
				dev_set_tree_height(addr, 1);
				dev_thash_h(begin + i * 4 * N,
					    begin + i * 4 * N, pub_seed, addr);
			}
		}
		g.sync();

		if (tid < 8) {
			memcpy(begin + tid * 8 * N + N, begin + tid * 8 * N + 4 * N, N);
			dev_set_tree_index(addr, tid);
			dev_set_tree_height(addr, 2);
			dev_thash_h(begin + tid * 8 * N,
				    begin + tid * 8 * N, pub_seed, addr);
		}
		g.sync();
		if (tid < 4) {
			memcpy(begin + tid * 16 * N + N, begin + tid * 16 * N + 8 * N, N);
			dev_set_tree_index(addr, tid);
			dev_set_tree_height(addr, 3);
			dev_thash_h(begin + tid * 16 * N,
				    begin + tid * 16 * N, pub_seed, addr);
		}
		g.sync();
		if (tid < 2) {
			memcpy(begin + tid * 32 * N + N, begin + tid * 32 * N + 16 * N, N);
			dev_set_tree_index(addr, tid);
			dev_set_tree_height(addr, 4);
			dev_thash_h(begin + tid * 32 * N,
				    begin + tid * 32 * N, pub_seed, addr);
		}
		g.sync();
		if (tid == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(addr, 0);
			dev_set_tree_height(addr, 5);
			dev_thash_h(begin, begin, pub_seed, addr);
			memcpy(begin + N, begin + 64 * N, N);
			dev_set_tree_index(addr, 0);
			dev_set_tree_height(addr, 6);
			dev_thash_h(begin, begin, pub_seed, addr);
			memcpy(leaf + tid * N, begin, N);
		}
		// g.sync();
	} else if (tnum >= 33) {
		g.sync();
		if (tid < 33) {
			dev_set_tree_index(addr, tid);
			dev_set_tree_height(addr, 0);
			dev_thash_h(begin + tid * 2 * N,
				    begin + tid * 2 * N, pub_seed, addr);
		}
		g.sync();
		if (tid < 17) {
			memcpy(begin + tid * 4 * N + N, begin + tid * 4 * N + 2 * N, N);
			dev_set_tree_index(addr, tid);
			dev_set_tree_height(addr, 1);
			dev_thash_h(begin + tid * 4 * N,
				    begin + tid * 4 * N, pub_seed, addr);
		}
		g.sync();
		if (tid < 8) {
			memcpy(begin + tid * 8 * N + N, begin + tid * 8 * N + 4 * N, N);
			dev_set_tree_index(addr, tid);
			dev_set_tree_height(addr, 2);
			dev_thash_h(begin + tid * 8 * N,
				    begin + tid * 8 * N, pub_seed, addr);
		}
		g.sync();
		if (tid < 4) {
			memcpy(begin + tid * 16 * N + N, begin + tid * 16 * N + 8 * N, N);
			dev_set_tree_index(addr, tid);
			dev_set_tree_height(addr, 3);
			dev_thash_h(begin + tid * 16 * N,
				    begin + tid * 16 * N, pub_seed, addr);
		}
		g.sync();
		if (tid < 2) {
			memcpy(begin + tid * 32 * N + N, begin + tid * 32 * N + 16 * N, N);
			dev_set_tree_index(addr, tid);
			dev_set_tree_height(addr, 4);
			dev_thash_h(begin + tid * 32 * N,
				    begin + tid * 32 * N, pub_seed, addr);
		}
		g.sync();
		if (tid == 0) {
			memcpy(begin + N, begin + 32 * N, N);
			dev_set_tree_index(addr, 0);
			dev_set_tree_height(addr, 5);
			dev_thash_h(begin, begin, pub_seed, addr);
			memcpy(begin + N, begin + 64 * N, N);
			dev_set_tree_index(addr, 0);
			dev_set_tree_height(addr, 6);
			dev_thash_h(begin, begin, pub_seed, addr);
			memcpy(leaf + tid * N, begin, N);
		}
		// g.sync();
	}
#else // if WOTS_LEN == 67
	if (tnum < 25) return; 
	g.sync();
	if (tid < 25) {
		dev_set_tree_index(addr, tid);
		dev_set_tree_height(addr, 0);
		dev_thash_h(begin + tid * 2 * N,
			    begin + tid * 2 * N, pub_seed, addr);
	}
	g.sync();
	if (tid < 13) {
		memcpy(begin + tid * 4 * N + N, begin + tid * 4 * N + 2 * N, N);
		dev_set_tree_index(addr, tid);
		dev_set_tree_height(addr, 1);
		dev_thash_h(begin + tid * 4 * N,
			    begin + tid * 4 * N, pub_seed, addr);
	}
	g.sync();
	if (tid < 6) {
		memcpy(begin + tid * 8 * N + N, begin + tid * 8 * N + 4 * N, N);
		dev_set_tree_index(addr, tid);
		dev_set_tree_height(addr, 2);
		dev_thash_h(begin + tid * 8 * N,
			    begin + tid * 8 * N, pub_seed, addr);
	}
	g.sync();
	if (tid < 3) {
		memcpy(begin + tid * 16 * N + N, begin + tid * 16 * N + 8 * N, N);
		dev_set_tree_index(addr, tid);
		dev_set_tree_height(addr, 3);
		dev_thash_h(begin + tid * 16 * N,
			    begin + tid * 16 * N, pub_seed, addr);
	}
	g.sync();
	if (tid < 2) {
		memcpy(begin + tid * 32 * N + N, begin + tid * 32 * N + 16 * N, N);
		dev_set_tree_index(addr, tid);
		dev_set_tree_height(addr, 4);
		dev_thash_h(begin + tid * 32 * N,
			    begin + tid * 32 * N, pub_seed, addr);
	}
	g.sync();
	if (tid == 0) {
		memcpy(begin + N, begin + 32 * N, N);
		dev_set_tree_index(addr, 0);
		dev_set_tree_height(addr, 5);
		dev_thash_h(begin, begin, pub_seed, addr);
		memcpy(leaf + tid * N, begin, N);
	}
	// g.sync();
#endif // if WOTS_LEN == 67

} // dev_l_tree_parallel


__device__ void dev_compute_root(u8 *root, const u8 *leaf, u32 *leafidx,
				 const u8 *auth_path, const u8 *pub_seed, u32 addr[8])
{
	u32 i;
	u8 buffer[2 * N];

	/* If leafidx is odd (last bit = 1), current path element is a right child
	 * and auth_path has to go left. Otherwise it is the other way around. */
	if (leafidx[0] & 1) {
		memcpy(buffer + N, leaf, N);
		memcpy(buffer, auth_path, N);
	} else {
		memcpy(buffer, leaf, N);
		memcpy(buffer + N, auth_path, N);
	}
	auth_path += N;

	for (i = 0; i < TREE_HEIGHT - 1; i++) {
		dev_set_tree_height(addr, i);
		leafidx[0] >>= 1;
		dev_set_tree_index(addr, leafidx[0]);

		/* Pick the right or left neighbor, depending on parity of the node. */
		if (leafidx[0] & 1) {
			dev_thash_h(buffer + N, buffer, pub_seed, addr);
			memcpy(buffer, auth_path, N);
		} else {
			dev_thash_h(buffer, buffer, pub_seed, addr);
			memcpy(buffer + N, auth_path, N);
		}
		auth_path += N;
	}

	/* The last iteration is exceptional; we do not copy an auth_path node. */
	dev_set_tree_height(addr, TREE_HEIGHT - 1);
	leafidx[0] >>= 1;
	dev_set_tree_index(addr, leafidx[0]);
	dev_thash_h(root, buffer, pub_seed, addr);
} // dev_compute_root

__device__ void dev_gen_leaf_wots(u8 *leaf,
				  const u8 *sk_seed, const u8 *pub_seed,
				  u32 ltree_addr[8], u32 ots_addr[8])
{
	u8 pk[WOTS_SIG_BYTES];

	dev_wots_pkgen(pk, sk_seed, pub_seed, ots_addr);

	dev_l_tree(leaf, pk, pub_seed, ltree_addr);
} // dev_gen_leaf_wots
