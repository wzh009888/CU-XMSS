#include "gpu_common.h"

__device__ void dev_l_tree(u8 *leaf, u8 *wots_pk,
			   const u8 *pub_seed, u32 addr[8]);

__device__ void dev_l_tree_parallel(u8 *leaf, u8 *wots_pk,
				    const u8 *pub_seed, u32 addr[8]);

__device__ void dev_compute_root(u8 *root, const u8 *leaf, u32 *leafidx,
				 const u8 *auth_path, const u8 *pub_seed, u32 addr[8]);

__device__ void dev_gen_leaf_wots(u8 *leaf, const u8 *sk_seed,
				  const u8 *pub_seed, u32 ltree_addr[8], u32 ots_addr[8]);
